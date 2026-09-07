"""CPU-safe render conventions, records, readers, and cross-pass validation."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.data.camera_math import pose_axes
from gaussweave.data.cameras import CameraCollection, load_camera_collection
from gaussweave.data.render_formats import (
    DecodedArray,
    RenderFormatError,
    read_npy_f32,
    read_png,
)
from gaussweave.data.scene_config import SceneConfiguration
from gaussweave.results.artifacts import hash_file

RENDER_CONVENTION_VERSION = "1.0.0"
RENDER_COLLECTION_VERSION = "1.0.0"
RENDER_POLICY_VERSION = "render-pass-probe-v1"
SEMANTIC_REGISTRY_VERSION = "engineering-probe-v1"
CAMERA_ID_RE = re.compile(r"^cam-(train|val|test)-[0-9]{4,}$")
PassName = Literal[
    "rgb",
    "depth",
    "normals",
    "semantic",
    "terminal",
    "instance",
    "alpha",
    "object_id",
    "material_id",
]
Split = Literal["train", "validation", "test"]


class RenderPassError(ValueError):
    """Render policy, record, or decoded output is invalid."""


class RenderValidationError(RenderPassError):
    """One or more decoded render artifacts violates the declared contract."""

    def __init__(self, issues: tuple[RenderValidationIssue, ...]) -> None:
        self.issues = issues
        summary = "; ".join(
            f"{item.camera_id}/{item.pass_name}/{item.rule}: {item.message}"
            for item in issues[:8]
        )
        if len(issues) > 8:
            summary += f"; ... and {len(issues) - 8} more"
        super().__init__(summary)


@dataclass(frozen=True)
class RGBConvention:
    pixel_origin: str = "top_left"
    row_order: str = "top_to_bottom"
    channel_order: str = "RGB"
    numeric_range: tuple[float, float] = (0.0, 1.0)
    transfer_function: str = "sRGB"
    color_space: str = "sRGB"
    source_render_space: str = "scene_linear"
    alpha_treatment: str = "separate_when_enabled"
    background_composition: str = "composited_over_configured_world"
    bit_depth: int = 8
    file_format: str = "png"
    lossless: bool = True
    exposure_applied: bool = True
    view_transform_applied: bool = True
    tone_mapping: str = "Blender Standard view transform"
    clipping_disclosed: bool = True

    def __post_init__(self) -> None:
        if self.numeric_range != (0.0, 1.0) or self.bit_depth not in {8, 16}:
            raise RenderPassError("RGB range/bit depth is unsupported")
        if self.file_format != "png" or not self.lossless:
            raise RenderPassError("RGB must use lossless PNG")


@dataclass(frozen=True)
class DepthConvention:
    pixel_origin: str = "top_left"
    row_order: str = "top_to_bottom"
    units: str = "meters"
    meaning: str = "euclidean_ray_distance_to_first_visible_surface"
    axis_semantics: str = "distance_along_normalized_camera_ray"
    valid_range: str = "near_m <= depth <= far_m"
    invalid_value: float = 0.0
    background_value: float = 0.0
    clipping_behavior: str = "outside_near_far_is_invalid"
    dtype: str = "float32"
    file_format: str = "npy"
    color_managed: bool = False

    def __post_init__(self) -> None:
        if (
            self.units != "meters"
            or self.meaning != "euclidean_ray_distance_to_first_visible_surface"
        ):
            raise RenderPassError("depth must use accepted metric ray distance")
        if self.invalid_value != 0.0 or self.background_value != 0.0:
            raise RenderPassError("depth invalid/background value must be 0.0")
        if self.dtype != "float32" or self.file_format != "npy":
            raise RenderPassError("depth must use safe float32 NPY")


@dataclass(frozen=True)
class NormalConvention:
    pixel_origin: str = "top_left"
    row_order: str = "top_to_bottom"
    coordinate_space: str = "world"
    channel_order: str = "XYZ"
    numeric_range: tuple[float, float] = (-1.0, 1.0)
    normalization: str = "unit_length_for_valid_pixels"
    invalid_value: tuple[float, float, float] = (0.0, 0.0, 0.0)
    background_value: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dtype: str = "float32"
    file_format: str = "npy"
    color_managed: bool = False

    def __post_init__(self) -> None:
        if (
            self.coordinate_space != "world"
            or self.channel_order != "XYZ"
            or self.numeric_range != (-1.0, 1.0)
        ):
            raise RenderPassError("normal coordinate/range convention is unsupported")
        if self.invalid_value != (0.0, 0.0, 0.0):
            raise RenderPassError("normal invalid value must be the zero vector")
        if self.dtype != "float32" or self.file_format != "npy":
            raise RenderPassError("normals must use safe float32 NPY")


@dataclass(frozen=True)
class IntegerConvention:
    pixel_origin: str = "top_left"
    row_order: str = "top_to_bottom"
    integer_valued: bool = True
    interpolation: str = "none_nearest"
    background_id: int = 0
    valid_id_range: tuple[int, int] = (0, 65535)
    dtype: str = "uint16"
    file_format: str = "png"
    bit_depth: int = 16
    color_managed: bool = False
    domains: tuple[str, ...] = (
        "semantic",
        "terminal",
        "instance",
        "object_id",
        "material_id",
    )

    def __post_init__(self) -> None:
        if not self.integer_valued or self.interpolation != "none_nearest":
            raise RenderPassError("mask samples must be uninterpolated integers")
        if self.background_id != 0 or self.valid_id_range != (0, 65535):
            raise RenderPassError("mask ID range/background is unsupported")
        if self.dtype != "uint16" or self.file_format != "png" or self.bit_depth != 16:
            raise RenderPassError("masks must use uint16 grayscale PNG")


@dataclass(frozen=True)
class AlphaConvention:
    pixel_origin: str = "top_left"
    row_order: str = "top_to_bottom"
    meaning: str = "straight_surface_coverage"
    numeric_range: tuple[float, float] = (0.0, 1.0)
    background_value: int = 0
    dtype: str = "uint16"
    file_format: str = "png"
    bit_depth: int = 16
    color_managed: bool = False

    def __post_init__(self) -> None:
        if self.numeric_range != (0.0, 1.0) or self.background_value != 0:
            raise RenderPassError("alpha range/background is unsupported")
        if self.dtype != "uint16" or self.file_format != "png":
            raise RenderPassError("alpha must use uint16 grayscale PNG")


@dataclass(frozen=True)
class RenderConventionRecord:
    convention_schema_version: str
    rgb: RGBConvention
    depth: DepthConvention
    normals: NormalConvention
    masks_and_ids: IntegerConvention
    alpha: AlphaConvention
    image_orientation: str
    scientific_digest: str

    def projection(self) -> dict[str, Any]:
        return {
            "convention_schema_version": self.convention_schema_version,
            "rgb": _json(asdict(self.rgb)),
            "depth": _json(asdict(self.depth)),
            "normals": _json(asdict(self.normals)),
            "masks_and_ids": _json(asdict(self.masks_and_ids)),
            "alpha": _json(asdict(self.alpha)),
            "image_orientation": self.image_orientation,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.projection()
        value["scientific_digest"] = self.scientific_digest
        return value


@dataclass(frozen=True)
class StableIdRegistry:
    registry_version: str
    background_id: int
    semantic_categories: dict[str, int]
    terminal_ids: dict[str, int]
    instance_ids: dict[str, int]
    instance_to_terminal: dict[int, int]
    object_ids: dict[str, int]
    material_ids: dict[str, int]

    def __post_init__(self) -> None:
        if self.background_id != 0:
            raise RenderPassError("registry background ID must be zero")
        for name, mapping in (
            ("semantic", self.semantic_categories),
            ("terminal", self.terminal_ids),
            ("instance", self.instance_ids),
            ("object", self.object_ids),
            ("material", self.material_ids),
        ):
            _validate_registry(name, mapping)
        valid_terminals = set(self.terminal_ids.values())
        valid_instances = set(self.instance_ids.values())
        if set(self.instance_to_terminal) != valid_instances:
            raise RenderPassError("every instance must have one terminal mapping")
        if not set(self.instance_to_terminal.values()) <= valid_terminals:
            raise RenderPassError("instance references an unknown terminal ID")

    def semantic_id(self, category: str) -> int:
        try:
            return self.semantic_categories[category]
        except KeyError as error:
            raise RenderPassError(
                f"unregistered semantic category: {category}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["instance_to_terminal"] = {
            str(key): item for key, item in sorted(self.instance_to_terminal.items())
        }
        return cast(dict[str, Any], _json(value))


@dataclass(frozen=True)
class RenderPassPolicy:
    policy_version: str
    enabled_passes: tuple[PassName, ...]
    conventions: RenderConventionRecord
    registry: StableIdRegistry
    color_management: dict[str, Any]
    normal_source_space: str
    normal_conversion: str
    depth_source: str
    id_rendering_policy: str

    def __post_init__(self) -> None:
        required = {"rgb", "depth", "normals", "semantic", "terminal", "instance"}
        if not required <= set(self.enabled_passes):
            raise RenderPassError("render-pass probe requires all base passes")
        if len(self.enabled_passes) != len(set(self.enabled_passes)):
            raise RenderPassError("enabled passes must be unique")
        if self.normal_source_space != "world":
            raise RenderPassError("Blender normal pass must be declared world-space")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "enabled_passes": list(self.enabled_passes),
            "conventions": self.conventions.to_dict(),
            "registry": self.registry.to_dict(),
            "color_management": self.color_management,
            "normal_source_space": self.normal_source_space,
            "normal_conversion": self.normal_conversion,
            "depth_source": self.depth_source,
            "id_rendering_policy": self.id_rendering_policy,
        }


@dataclass(frozen=True)
class RenderViewRequest:
    scene_id: str
    camera_id: str
    split: Split
    width: int
    height: int
    camera_reference: str
    requested_passes: tuple[PassName, ...]


@dataclass(frozen=True)
class PassArtifact:
    pass_name: PassName
    path: str
    file_format: str
    dtype: str
    channels: int
    width: int
    height: int
    file_digest: str
    decoded_digest: str
    required: bool


@dataclass(frozen=True)
class PassStatistics:
    camera_id: str
    pass_name: PassName
    dtype: str
    shape: tuple[int, ...]
    finite_count: int
    nonfinite_count: int
    invalid_or_background_count: int
    minimum: float | int | None
    maximum: float | int | None
    unique_ids: tuple[int, ...]
    decoded_digest: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["unique_ids"] = list(self.unique_ids)
        return value


@dataclass(frozen=True)
class RenderValidationIssue:
    camera_id: str
    pass_name: str
    rule: str
    pixel_count: int
    message: str


@dataclass(frozen=True)
class PassValidation:
    valid: bool
    camera_count: int
    artifact_count: int
    statistics: tuple[PassStatistics, ...]
    issues: tuple[RenderValidationIssue, ...]
    analytic_checks: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "camera_count": self.camera_count,
            "artifact_count": self.artifact_count,
            "statistics": [item.to_dict() for item in self.statistics],
            "issues": [asdict(item) for item in self.issues],
            "analytic_checks": list(self.analytic_checks),
        }


@dataclass(frozen=True)
class RenderedView:
    scene_id: str
    camera_id: str
    split: Split
    width: int
    height: int
    camera_reference: str
    render_engine: str
    compute_device: str
    sample_count: int
    pass_artifacts: tuple[PassArtifact, ...]
    pass_conventions_digest: str
    warnings: tuple[str, ...]
    scientific_digest: str
    execution_reference: str

    def scientific_projection(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "camera_id": self.camera_id,
            "split": self.split,
            "width": self.width,
            "height": self.height,
            "camera_reference": self.camera_reference,
            "render_engine": self.render_engine,
            "compute_device": self.compute_device,
            "sample_count": self.sample_count,
            "pass_artifacts": [
                {
                    **asdict(item),
                    "file_digest": None,
                }
                for item in self.pass_artifacts
            ],
            "pass_conventions_digest": self.pass_conventions_digest,
        }


@dataclass(frozen=True)
class RenderCollection:
    render_collection_schema_version: str
    scene_id: str
    configuration_digest: str
    camera_collection_digest: str
    convention_digest: str
    policy_version: str
    enabled_passes: tuple[PassName, ...]
    registry: StableIdRegistry
    views: tuple[RenderedView, ...]
    probe_geometry: dict[str, Any]
    scientific_digest: str
    execution_reference: str
    warnings: tuple[str, ...]

    def scientific_projection(self) -> dict[str, Any]:
        return {
            "render_collection_schema_version": self.render_collection_schema_version,
            "scene_id": self.scene_id,
            "configuration_digest": self.configuration_digest,
            "camera_collection_digest": self.camera_collection_digest,
            "convention_digest": self.convention_digest,
            "policy_version": self.policy_version,
            "enabled_passes": list(self.enabled_passes),
            "registry": self.registry.to_dict(),
            "views": [view.scientific_projection() for view in self.views],
            "probe_geometry": self.probe_geometry,
        }


@dataclass(frozen=True)
class RenderFailure:
    failure_schema_version: str
    scene_id: str
    camera_id: str | None
    pass_name: str | None
    category: str
    message: str
    execution_reference: str | None


def default_conventions() -> RenderConventionRecord:
    base = RenderConventionRecord(
        RENDER_CONVENTION_VERSION,
        RGBConvention(),
        DepthConvention(),
        NormalConvention(),
        IntegerConvention(),
        AlphaConvention(),
        "top-left origin; rows serialized top-to-bottom after Blender conversion",
        "",
    )
    return replace(base, scientific_digest=content_digest(base.projection()))


def engineering_probe_registry() -> StableIdRegistry:
    return StableIdRegistry(
        SEMANTIC_REGISTRY_VERSION,
        0,
        {
            "background": 0,
            "structure": 1,
            "terminal": 2,
            "instance": 3,
            "occluder": 4,
            "floor": 5,
            "lighting_fixture": 6,
            "annotation_excluded": 7,
        },
        {"terminal-probe-canonical": 1},
        {"instance-probe-01": 1, "instance-probe-02": 2},
        {1: 1, 2: 1},
        {
            "probe-structure": 1,
            "terminal-probe-a": 2,
            "instance-probe-01": 3,
            "instance-probe-02": 4,
            "floor-probe": 5,
            "occluder-probe": 6,
        },
        {
            "probe-base": 1,
            "probe-accent": 2,
            "probe-floor": 3,
            "probe-occluder": 4,
        },
    )


def build_render_pass_policy(config: SceneConfiguration) -> RenderPassPolicy:
    enabled: list[PassName] = []
    outputs = config.outputs
    if outputs.rgb_format != "png" or outputs.rgb_bit_depth != 8:
        raise RenderPassError("render-pass probe requires lossless 8-bit RGB PNG")
    if outputs.depth_format != "npy":
        raise RenderPassError("render-pass probe requires float32 NPY depth")
    if outputs.normals and outputs.normals_format != "npy":
        raise RenderPassError("render-pass probe requires float32 NPY normals")
    if outputs.masks and (outputs.mask_format != "png" or outputs.mask_bit_depth != 16):
        raise RenderPassError("render-pass probe requires uint16 PNG masks")
    for name, selected in (
        ("rgb", outputs.rgb),
        ("depth", outputs.depth),
        ("normals", outputs.normals),
        ("semantic", outputs.masks and outputs.semantic_masks),
        ("terminal", outputs.masks and outputs.terminal_masks),
        ("instance", outputs.masks and outputs.instance_masks),
        ("alpha", outputs.alpha),
        ("object_id", outputs.object_ids),
        ("material_id", outputs.material_ids),
    ):
        if selected:
            enabled.append(cast(PassName, name))
    conventions = default_conventions()
    if outputs.alpha:
        updated = replace(
            conventions,
            rgb=replace(
                conventions.rgb,
                background_composition=(
                    "transparent_world_with_separate_straight_alpha"
                ),
            ),
            scientific_digest="",
        )
        conventions = replace(
            updated,
            scientific_digest=content_digest(updated.projection()),
        )
    color_management = {
        "source_color_space": config.render.color_space,
        "display_device": "sRGB",
        "view_transform": config.render.view_transform,
        "look": config.render.look,
        "exposure": config.render.exposure,
        "gamma": config.render.gamma,
        "sequencer_space": "sRGB",
        "stored_rgb_color_space": "sRGB",
        "stored_rgb_transfer_function": "sRGB",
        "stored_rgb_is_display_referred": True,
    }
    return RenderPassPolicy(
        RENDER_POLICY_VERSION,
        tuple(enabled),
        conventions,
        engineering_probe_registry(),
        color_management,
        "world",
        "Blender/Cycles Normal pass is world-space; valid vectors are renormalized",
        (
            "Blender Z pass camera-forward z is divided by the normalized "
            "camera-ray forward cosine to store Euclidean ray distance"
        ),
        (
            "integer identity passes use deterministic Cycles CPU sampling; "
            "continuous RGB/depth/normal passes retain the requested device"
        ),
    )


def write_render_request(
    path: Path,
    *,
    config: SceneConfiguration,
    camera_collection: CameraCollection,
    camera_ids: tuple[str, ...],
) -> RenderPassPolicy:
    policy = build_render_pass_policy(config)
    by_id = {record.camera_id: record for record in camera_collection.records}
    requests: list[dict[str, Any]] = []
    for camera_id in camera_ids:
        try:
            camera = by_id[camera_id]
        except KeyError as error:
            raise RenderPassError(f"unknown selected camera ID: {camera_id}") from error
        requests.append(
            _json(
                asdict(
                    RenderViewRequest(
                        config.scene_id,
                        camera_id,
                        camera.split,
                        camera.width,
                        camera.height,
                        "cameras/cameras.json",
                        policy.enabled_passes,
                    )
                )
            )
        )
    payload = {
        "render_request_schema_version": RENDER_COLLECTION_VERSION,
        "scene_id": config.scene_id,
        "configuration_digest": config.scientific_digest,
        "camera_collection_digest": camera_collection.scientific_digest,
        "policy": policy.to_dict(),
        "views": requests,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(cast(Any, payload)))
    return policy


def load_render_collection(path: Path) -> RenderCollection:
    value = _read_json(path)
    try:
        registry = _registry_from_dict(value["registry"])
        views = tuple(_view_from_dict(item) for item in value["views"])
        collection = RenderCollection(
            str(value["render_collection_schema_version"]),
            str(value["scene_id"]),
            str(value["configuration_digest"]),
            str(value["camera_collection_digest"]),
            str(value["convention_digest"]),
            str(value["policy_version"]),
            tuple(cast(PassName, str(item)) for item in value["enabled_passes"]),
            registry,
            views,
            dict(value["probe_geometry"]),
            str(value["scientific_digest"]),
            str(value["execution_reference"]),
            tuple(str(item) for item in value.get("warnings", [])),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, RenderPassError):
            raise
        raise RenderPassError("invalid render collection") from error
    if collection.render_collection_schema_version != RENDER_COLLECTION_VERSION:
        raise RenderPassError("unsupported render collection version")
    if (
        content_digest(collection.scientific_projection())
        != collection.scientific_digest
    ):
        raise RenderPassError("render collection scientific digest mismatch")
    return collection


def validate_render_root(root: Path) -> PassValidation:
    """Decode and semantically validate every selected render view."""

    output = root.expanduser().resolve()
    collection = load_render_collection(output / "metadata/render_collection.json")
    conventions = _read_json(output / "metadata/render_conventions.json")
    if content_digest(
        {key: value for key, value in conventions.items() if key != "scientific_digest"}
    ) != conventions.get("scientific_digest"):
        raise RenderPassError("render convention digest mismatch")
    if conventions["scientific_digest"] != collection.convention_digest:
        raise RenderPassError("collection convention reference mismatch")
    cameras = load_camera_collection(output / "cameras/cameras.json")
    camera_by_id = {record.camera_id: record for record in cameras.records}
    issues: list[RenderValidationIssue] = []
    statistics: list[PassStatistics] = []
    analytics: list[dict[str, Any]] = []
    for view in collection.views:
        if view.camera_id not in camera_by_id:
            issues.append(
                _issue(view.camera_id, "metadata", "stale_camera", 1, "unknown camera")
            )
            continue
        if view.camera_reference != "cameras/cameras.json":
            issues.append(
                _issue(
                    view.camera_id,
                    "metadata",
                    "stale_camera",
                    1,
                    "camera reference differs from the generated collection",
                )
            )
        decoded: dict[str, DecodedArray] = {}
        artifacts_by_name = {item.pass_name: item for item in view.pass_artifacts}
        expected_names = set(collection.enabled_passes)
        if set(artifacts_by_name) != expected_names:
            missing = sorted(expected_names - set(artifacts_by_name))
            extra = sorted(set(artifacts_by_name) - expected_names)
            issues.append(
                _issue(
                    view.camera_id,
                    "metadata",
                    "artifact_membership",
                    len(missing) + len(extra),
                    f"missing={missing}; extra={extra}",
                )
            )
        for pass_name in sorted(expected_names):
            artifact = artifacts_by_name.get(pass_name)
            if artifact is None:
                continue
            try:
                target = _safe_artifact(output, artifact.path)
                if hash_file(target).digest != artifact.file_digest:
                    raise RenderPassError("file digest differs from metadata")
                image = _decode_pass(pass_name, target)
                decoded[pass_name] = image
                if image.width != view.width or image.height != view.height:
                    issues.append(
                        _issue(
                            view.camera_id,
                            pass_name,
                            "dimensions",
                            image.width * image.height,
                            f"decoded {image.width}x{image.height}; "
                            f"expected {view.width}x{view.height}",
                        )
                    )
                if (
                    image.channels != artifact.channels
                    or image.dtype != artifact.dtype
                    or image.decoded_digest != artifact.decoded_digest
                ):
                    issues.append(
                        _issue(
                            view.camera_id,
                            pass_name,
                            "decoded_identity",
                            image.width * image.height,
                            "decoded dtype/channels/digest differs from metadata",
                        )
                    )
                statistics.append(_statistics(view.camera_id, pass_name, image))
            except (OSError, RenderFormatError, RenderPassError) as error:
                issues.append(
                    _issue(
                        view.camera_id,
                        pass_name,
                        "decode",
                        0,
                        str(error),
                    )
                )
        if {"depth", "normals", "semantic", "terminal", "instance"} <= set(decoded):
            _validate_alignment(
                view.camera_id,
                decoded,
                collection.registry,
                issues,
            )
            analytic = _analytic_probe_check(
                view.camera_id,
                decoded,
                camera_by_id[view.camera_id],
                collection,
                issues,
            )
            if analytic is not None:
                analytics.append(analytic)
    result = PassValidation(
        not issues,
        len(collection.views),
        sum(len(view.pass_artifacts) for view in collection.views),
        tuple(statistics),
        tuple(issues),
        tuple(analytics),
    )
    if issues:
        raise RenderValidationError(tuple(issues))
    return result


def _validate_alignment(
    camera_id: str,
    decoded: dict[str, DecodedArray],
    registry: StableIdRegistry,
    issues: list[RenderValidationIssue],
) -> None:
    depth = decoded["depth"]
    normals = decoded["normals"]
    semantic = decoded["semantic"]
    terminal = decoded["terminal"]
    instance = decoded["instance"]
    pixel_count = semantic.width * semantic.height
    if any(
        item.width != semantic.width or item.height != semantic.height
        for item in decoded.values()
    ):
        return
    semantic_values = set(int(value) for value in semantic.values)
    registered_semantics = set(registry.semantic_categories.values())
    unknown_semantics = semantic_values - registered_semantics
    if unknown_semantics:
        issues.append(
            _issue(
                camera_id,
                "semantic",
                "registered_ids",
                sum(int(value) in unknown_semantics for value in semantic.values),
                f"unregistered semantic IDs: {sorted(unknown_semantics)}",
            )
        )
    allowed_terminal = {0, *registry.terminal_ids.values()}
    allowed_instance = {0, *registry.instance_ids.values()}
    _mask_domain(camera_id, "terminal", terminal, allowed_terminal, issues)
    _mask_domain(camera_id, "instance", instance, allowed_instance, issues)
    if "object_id" in decoded:
        _mask_domain(
            camera_id,
            "object_id",
            decoded["object_id"],
            {0, *registry.object_ids.values()},
            issues,
        )
    if "material_id" in decoded:
        _mask_domain(
            camera_id,
            "material_id",
            decoded["material_id"],
            {0, *registry.material_ids.values()},
            issues,
        )
    visible = 0
    invalid_foreground_depth = 0
    invalid_background_depth = 0
    invalid_foreground_normal = 0
    invalid_background_normal = 0
    invalid_terminal_mapping = 0
    invalid_terminal_semantic = 0
    terminal_semantic = registry.semantic_categories["terminal"]
    for pixel in range(pixel_count):
        semantic_id = int(semantic.values[pixel])
        terminal_id = int(terminal.values[pixel])
        instance_id = int(instance.values[pixel])
        depth_value = float(depth.values[pixel])
        normal_offset = pixel * 3
        normal = tuple(float(normals.values[normal_offset + axis]) for axis in range(3))
        normal_norm = math.sqrt(sum(value * value for value in normal))
        foreground = semantic_id != registry.background_id
        if foreground:
            visible += 1
            if not math.isfinite(depth_value) or depth_value <= 0.0:
                invalid_foreground_depth += 1
            if (
                not all(math.isfinite(value) for value in normal)
                or abs(normal_norm - 1.0) > 1e-3
            ):
                invalid_foreground_normal += 1
        else:
            if depth_value != 0.0:
                invalid_background_depth += 1
            if normal != (0.0, 0.0, 0.0):
                invalid_background_normal += 1
        if instance_id:
            expected_terminal = registry.instance_to_terminal.get(instance_id)
            if expected_terminal is None or terminal_id != expected_terminal:
                invalid_terminal_mapping += 1
            if semantic_id != terminal_semantic:
                invalid_terminal_semantic += 1
        elif terminal_id:
            invalid_terminal_mapping += 1
    for pass_name, rule, count, message in (
        (
            "depth",
            "foreground_valid",
            invalid_foreground_depth,
            "visible pixels require finite positive metric depth",
        ),
        (
            "depth",
            "background_invalid",
            invalid_background_depth,
            "background depth must equal 0.0",
        ),
        (
            "normals",
            "foreground_unit",
            invalid_foreground_normal,
            "visible pixels require finite unit world normals",
        ),
        (
            "normals",
            "background_invalid",
            invalid_background_normal,
            "background normals must equal the zero vector",
        ),
        (
            "instance",
            "terminal_mapping",
            invalid_terminal_mapping,
            "instance pixels require their registered terminal ID",
        ),
        (
            "semantic",
            "terminal_semantic",
            invalid_terminal_semantic,
            "instance pixels require terminal semantic category",
        ),
    ):
        if count:
            issues.append(_issue(camera_id, pass_name, rule, count, message))
    if visible == 0:
        issues.append(
            _issue(camera_id, "semantic", "foreground_present", 0, "no foreground")
        )
    if "alpha" in decoded:
        alpha = decoded["alpha"]
        inconsistent = 0
        for pixel in range(pixel_count):
            semantic_value = int(semantic.values[pixel])
            alpha_value = int(alpha.values[pixel])
            if semantic_value != 0 and alpha_value == 0:
                inconsistent += 1
                continue
            if semantic_value != 0 or alpha_value == 0:
                continue
            if alpha_value == 65535:
                inconsistent += 1
        if inconsistent:
            issues.append(
                _issue(
                    camera_id,
                    "alpha",
                    "foreground_agreement",
                    inconsistent,
                    "alpha coverage disagrees with semantic foreground",
                )
            )


def _analytic_probe_check(
    camera_id: str,
    decoded: dict[str, DecodedArray],
    camera: Any,
    collection: RenderCollection,
    issues: list[RenderValidationIssue],
) -> dict[str, Any] | None:
    geometry = collection.probe_geometry
    expected_normal = tuple(float(value) for value in geometry["wall_front_normal"])
    wall_point = tuple(float(value) for value in geometry["wall_front_point_m"])
    semantic_id = collection.registry.semantic_categories["structure"]
    semantic = decoded["semantic"]
    object_ids = decoded.get("object_id")
    wall_object_id = collection.registry.object_ids["probe-structure"]
    depth = decoded["depth"]
    normals = decoded["normals"]
    right, down, forward, position = pose_axes(camera.world_from_camera)
    for row in range(semantic.height // 4, semantic.height * 3 // 4):
        for column in range(semantic.width // 4, semantic.width * 3 // 4):
            pixel = row * semantic.width + column
            if int(semantic.values[pixel]) != semantic_id:
                continue
            if (
                object_ids is not None
                and int(object_ids.values[pixel]) != wall_object_id
            ):
                continue
            offset = pixel * 3
            measured_normal = tuple(
                float(normals.values[offset + axis]) for axis in range(3)
            )
            normal_error = max(
                abs(left - right_value)
                for left, right_value in zip(
                    measured_normal, expected_normal, strict=True
                )
            )
            if normal_error > 0.02:
                continue
            local_direction = (
                (column + 0.5 - camera.cx) / camera.fx,
                (row + 0.5 - camera.cy) / camera.fy,
                1.0,
            )
            length = math.sqrt(sum(value * value for value in local_direction))
            normalized_local = tuple(value / length for value in local_direction)
            direction = tuple(
                right[axis] * normalized_local[0]
                + down[axis] * normalized_local[1]
                + forward[axis] * normalized_local[2]
                for axis in range(3)
            )
            denominator = sum(
                direction[axis] * expected_normal[axis] for axis in range(3)
            )
            if abs(denominator) < 1e-12:
                continue
            expected_depth = (
                sum(
                    (wall_point[axis] - position[axis]) * expected_normal[axis]
                    for axis in range(3)
                )
                / denominator
            )
            measured_depth = float(depth.values[pixel])
            depth_error = abs(measured_depth - expected_depth)
            if expected_depth <= 0.0:
                continue
            if depth_error > 0.05:
                issues.append(
                    _issue(
                        camera_id,
                        "depth",
                        "analytic_wall_depth",
                        1,
                        f"measured={measured_depth}; expected={expected_depth}; "
                        f"error={depth_error}",
                    )
                )
            return {
                "camera_id": camera_id,
                "pixel": [row, column],
                "expected_world_normal": list(expected_normal),
                "measured_world_normal": list(measured_normal),
                "normal_max_abs_error": normal_error,
                "expected_ray_depth_m": expected_depth,
                "measured_ray_depth_m": measured_depth,
                "depth_abs_error_m": depth_error,
                "normal_tolerance": 0.02,
                "depth_tolerance_m": 0.05,
                "valid": normal_error <= 0.02 and depth_error <= 0.05,
            }
    issues.append(
        _issue(
            camera_id,
            "analytic",
            "wall_sample",
            0,
            "no unoccluded analytic wall-front pixel was found",
        )
    )
    return None


def _decode_pass(pass_name: str, path: Path) -> DecodedArray:
    decoded = (
        read_npy_f32(path) if pass_name in {"depth", "normals"} else read_png(path)
    )
    expected = {
        "rgb": ("uint8", {3, 4}),
        "depth": ("float32", {1}),
        "normals": ("float32", {3}),
        "semantic": ("uint16", {1}),
        "terminal": ("uint16", {1}),
        "instance": ("uint16", {1}),
        "alpha": ("uint16", {1}),
        "object_id": ("uint16", {1}),
        "material_id": ("uint16", {1}),
    }
    dtype, channels = expected[pass_name]
    if decoded.dtype != dtype or decoded.channels not in channels:
        raise RenderPassError(
            f"{pass_name} decoded as {decoded.dtype}/{decoded.channels} channels"
        )
    return decoded


def _statistics(
    camera_id: str, pass_name: str, decoded: DecodedArray
) -> PassStatistics:
    floating = decoded.dtype.startswith("float")
    finite_values = [
        float(value) for value in decoded.values if math.isfinite(float(value))
    ]
    nonfinite = len(decoded.values) - len(finite_values)
    if pass_name == "normals":
        invalid = sum(
            all(float(decoded.values[index + axis]) == 0.0 for axis in range(3))
            for index in range(0, len(decoded.values), 3)
        )
    else:
        invalid = sum(float(value) == 0.0 for value in decoded.values)
    unique_ids = (
        tuple(sorted(set(int(value) for value in decoded.values)))
        if not floating and pass_name != "rgb"
        else ()
    )
    return PassStatistics(
        camera_id,
        cast(PassName, pass_name),
        decoded.dtype,
        decoded.shape,
        len(finite_values),
        nonfinite,
        invalid,
        min(finite_values) if finite_values else None,
        max(finite_values) if finite_values else None,
        unique_ids,
        decoded.decoded_digest,
    )


def _mask_domain(
    camera_id: str,
    pass_name: str,
    decoded: DecodedArray,
    allowed: set[int],
    issues: list[RenderValidationIssue],
) -> None:
    unknown = set(int(value) for value in decoded.values) - allowed
    if unknown:
        issues.append(
            _issue(
                camera_id,
                pass_name,
                "registered_ids",
                sum(int(value) in unknown for value in decoded.values),
                f"unregistered IDs: {sorted(unknown)}",
            )
        )


def _view_from_dict(value: dict[str, Any]) -> RenderedView:
    artifacts = tuple(
        PassArtifact(
            cast(PassName, str(item["pass_name"])),
            str(item["path"]),
            str(item["file_format"]),
            str(item["dtype"]),
            int(item["channels"]),
            int(item["width"]),
            int(item["height"]),
            str(item["file_digest"]),
            str(item["decoded_digest"]),
            bool(item["required"]),
        )
        for item in value["pass_artifacts"]
    )
    view = RenderedView(
        str(value["scene_id"]),
        str(value["camera_id"]),
        cast(Split, str(value["split"])),
        int(value["width"]),
        int(value["height"]),
        str(value["camera_reference"]),
        str(value["render_engine"]),
        str(value["compute_device"]),
        int(value["sample_count"]),
        artifacts,
        str(value["pass_conventions_digest"]),
        tuple(str(item) for item in value.get("warnings", [])),
        str(value["scientific_digest"]),
        str(value["execution_reference"]),
    )
    if not CAMERA_ID_RE.fullmatch(view.camera_id):
        raise RenderPassError("render view camera ID is invalid")
    if content_digest(view.scientific_projection()) != view.scientific_digest:
        raise RenderPassError(f"{view.camera_id}: scientific digest mismatch")
    return view


def _registry_from_dict(value: dict[str, Any]) -> StableIdRegistry:
    return StableIdRegistry(
        str(value["registry_version"]),
        int(value["background_id"]),
        {str(key): int(item) for key, item in value["semantic_categories"].items()},
        {str(key): int(item) for key, item in value["terminal_ids"].items()},
        {str(key): int(item) for key, item in value["instance_ids"].items()},
        {int(key): int(item) for key, item in value["instance_to_terminal"].items()},
        {str(key): int(item) for key, item in value["object_ids"].items()},
        {str(key): int(item) for key, item in value["material_ids"].items()},
    )


def _validate_registry(name: str, mapping: dict[str, int]) -> None:
    if not mapping:
        raise RenderPassError(f"{name} registry may not be empty")
    values = list(mapping.values())
    if len(values) != len(set(values)):
        raise RenderPassError(f"{name} registry IDs must be unique")
    minimum = 0 if name == "semantic" else 1
    if any(value < minimum or value > 65535 for value in values):
        raise RenderPassError(f"{name} registry ID exceeds uint16 capacity")


def _issue(
    camera_id: str, pass_name: str, rule: str, count: int, message: str
) -> RenderValidationIssue:
    return RenderValidationIssue(camera_id, pass_name, rule, count, message)


def _safe_artifact(root: Path, portable: str) -> Path:
    if (
        not portable
        or "\\" in portable
        or portable.startswith("/")
        or ".." in portable.split("/")
    ):
        raise RenderPassError(f"unsafe render artifact path: {portable!r}")
    target = root.joinpath(*portable.split("/"))
    try:
        target.resolve().relative_to(root)
    except ValueError as error:
        raise RenderPassError("render artifact escapes root") from error
    if target.is_symlink() or not target.is_file():
        raise RenderPassError(f"render artifact is missing or unsafe: {portable}")
    return target


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RenderPassError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RenderPassError(f"JSON artifact is not an object: {path}")
    return value


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate decoded render-pass outputs")
    subcommands = parser.add_subparsers(dest="action", required=True)
    validate = subcommands.add_parser("validate")
    validate.add_argument("root")
    validate.add_argument("--json", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        result = validate_render_root(Path(options.root))
    except RenderValidationError as error:
        payload = {
            "valid": False,
            "error": str(error),
            "issues": [asdict(item) for item in error.issues],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 7
    except (RenderPassError, RenderFormatError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 7
    payload = result.to_dict()
    if options.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Render passes valid: cameras={result.camera_count}; "
            f"artifacts={result.artifact_count}; analytic={len(result.analytic_checks)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
