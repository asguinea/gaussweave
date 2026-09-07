"""Typed, deterministic procedural scene configuration contracts.

This module is intentionally CPU-only.  It validates generation intent and
builds a process-boundary handoff, but it does not generate geometry or import
Blender, torch, or gsplat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar, cast

from gaussweave.config.errors import (
    DocumentLoadError,
    SchemaValidationError,
    ValidationIssue,
    ValidationStage,
    display_source,
    summarize_value,
)
from gaussweave.config.resolution import (
    canonical_json_bytes,
    content_digest,
    pretty_json_bytes,
)

SCENE_CONFIG_SCHEMA_VERSION = "1.0.0"
SCIENTIFIC_PROJECTION_VERSION = "scene-v1"
SEED_POLICY_VERSION = "sha256-v1"
SEED_STREAMS = (
    "geometry",
    "materials",
    "lighting",
    "cameras",
    "occluders",
    "edits",
    "annotations",
)
UINT32_MAX = 2**32 - 1
SCENE_ID_RE = re.compile(r"^syn-(facade|corridor|colonnade)-[a-z0-9-]+-s([0-9]+)$")
DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
EDIT_ID_RE = re.compile(r"^edit-[a-z0-9][a-z0-9-]*$")
PROFILE_MAX_EDGE = {"smoke": 256, "quick": 512, "standard": 1024, "extended": 4096}


class SceneFamily(StrEnum):
    """Accepted synthetic generator families."""

    FACADE = "facade"
    CORRIDOR = "corridor"
    COLONNADE = "colonnade"


class RepetitionLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class SpacingRegime(StrEnum):
    COMPACT = "compact"
    STANDARD = "standard"
    WIDE = "wide"


class TerminalComplexity(StrEnum):
    SIMPLE = "simple"
    COMPOUND = "compound"


class StructuralRegularity(StrEnum):
    STRICT = "strict"
    MILDLY_IRREGULAR = "mildly_irregular"
    DISRUPTED = "disrupted"
    WEAK = "weak"


class MissingInstancePolicy(StrEnum):
    NONE = "none"
    EXPLICIT = "explicit"
    PATTERN = "pattern"


class AppearanceRegime(StrEnum):
    IDENTICAL = "identical"
    DETERMINISTIC_LOW_VARIATION = "deterministic_low_variation"
    ALTERNATING = "alternating"
    GROUPED = "grouped"
    INSTANCE_SPECIFIC = "instance_specific"


class LightingRegime(StrEnum):
    DIFFUSE = "diffuse"
    DIRECTIONAL = "directional"
    MIXED_INTERIOR = "mixed_interior"
    SPATIALLY_VARYING = "spatially_varying"


class OcclusionRegime(StrEnum):
    CLEAR = "clear"
    PARTIAL_FOREGROUND = "partial_foreground"
    TRUNCATION = "truncation"
    INTER_INSTANCE = "inter_instance"


class EditOperation(StrEnum):
    CHANGE_REPEAT_COUNT = "change_repeat_count"
    REMOVE_REGION = "remove_region"
    CHANGE_SPACING = "change_spacing"
    SCALE_TERMINAL = "scale_terminal"
    REPLACE_TERMINAL = "replace_terminal"
    REMOVE_INSTANCES = "remove_instances"
    EXTEND_REGION = "extend_region"


class ResourceProfile(StrEnum):
    SMOKE = "smoke"
    QUICK = "quick"
    STANDARD = "standard"
    EXTENDED = "extended"


@dataclass(frozen=True)
class CoordinateSystem:
    handedness: str
    up_axis: str
    unit: str
    camera_axes: Mapping[str, str]
    matrix_layout: str


@dataclass(frozen=True)
class GeneratorSettings:
    name: str
    version: str
    implementation_status: str
    external_assets: tuple[str, ...]


@dataclass(frozen=True)
class RenderSettings:
    engine: str
    compute_device: str
    width: int
    height: int
    resolution_percentage: int
    samples: int
    denoise: bool
    color_space: str
    view_transform: str
    look: str
    exposure: float
    gamma: float
    background_kind: str
    background_rgba: tuple[float, float, float, float]
    background_has_valid_depth: bool


@dataclass(frozen=True)
class CameraPolicy:
    trajectory: str
    train_count: int
    validation_count: int
    test_count: int
    image_width: int
    image_height: int
    focal_mode: str
    focal_length_mm: float | None
    horizontal_fov_degrees: float | None
    principal_point: str
    principal_point_px: tuple[float, float] | None
    near_m: float
    far_m: float
    position_min_m: tuple[float, float, float]
    position_max_m: tuple[float, float, float]
    yaw_degrees: tuple[float, float]
    pitch_degrees: tuple[float, float]
    roll_degrees: tuple[float, float]
    minimum_separation_m: float
    minimum_angular_separation_degrees: float
    minimum_structure_coverage: float
    seed: int
    family_intent: Mapping[str, Any]


@dataclass(frozen=True)
class StructureVariation:
    repetition_level: RepetitionLevel
    spacing_regime: SpacingRegime
    terminal_complexity: TerminalComplexity
    regularity: StructuralRegularity
    missing_instance_policy: MissingInstancePolicy
    negative_control: bool
    negative_control_label: str | None
    expected_repeat_groups: tuple[str, ...]
    expected_canonical_terminal_categories: tuple[str, ...]


@dataclass(frozen=True)
class FacadeParameters:
    family: SceneFamily
    rows: int
    columns: int
    horizontal_spacing_m: float
    vertical_spacing_m: float
    terminal_width_m: float
    terminal_height_m: float
    terminal_depth_m: float
    wall_width_m: float
    wall_height_m: float
    wall_depth_m: float
    margins_m: Mapping[str, float]
    floor_band_height_m: float
    sill_depth_m: float
    recess_depth_m: float
    balcony_mode: str
    panel_mode: str
    missing_pattern: str
    missing_instances: tuple[int, ...]


@dataclass(frozen=True)
class CorridorParameters:
    family: SceneFamily
    length_m: float
    width_m: float
    height_m: float
    bay_count: int
    bay_spacing_m: float
    terminal_width_m: float
    terminal_height_m: float
    terminal_depth_m: float
    sidedness: str
    ceiling_light_mode: str
    end_wall_mode: str
    furniture_occluder_mode: str
    end_margin_m: float
    missing_pattern: str
    missing_instances: tuple[int, ...]


@dataclass(frozen=True)
class ColonnadeParameters:
    family: SceneFamily
    count: int
    spacing_m: float
    terminal_radius_m: float | None
    terminal_width_m: float | None
    terminal_height_m: float
    plinth_height_m: float
    capital_height_m: float
    span_mode: str
    row_count: int
    layout: str
    floor_width_m: float
    floor_length_m: float
    floor_thickness_m: float
    background_mode: str
    missing_pattern: str
    missing_instances: tuple[int, ...]


FamilyParameters = FacadeParameters | CorridorParameters | ColonnadeParameters


@dataclass(frozen=True)
class AppearanceSettings:
    regime: AppearanceRegime
    base_material: str
    base_color: tuple[float, float, float, float]
    roughness: float
    metallic: float | None
    palette: tuple[tuple[float, float, float, float], ...]
    roughness_range: tuple[float, float]
    metallic_range: tuple[float, float]
    variation_amplitude: float
    variation_groups: tuple[str, ...]
    group_size: int | None
    alternate_period: int | None
    texture_mode: str
    seed: int
    exact_canonical_reuse: bool
    residual_policy_label: str


@dataclass(frozen=True)
class LightingSettings:
    regime: LightingRegime
    world_energy: float
    world_color: tuple[float, float, float]
    key_light_type: str
    key_energy: float
    key_color: tuple[float, float, float]
    key_position_m: tuple[float, float, float]
    energy_range: tuple[float, float]
    color_temperature_kelvin: tuple[float, float]
    direction_degrees: tuple[float, float, float]
    deterministic_variation: bool
    source_count: int
    fill_light_count: int
    fill_light_energy: float
    spatial_variation_amplitude: float
    seed: int
    cast_shadows: bool
    exposure_intent: str


@dataclass(frozen=True)
class OcclusionSettings:
    regime: OcclusionRegime
    occluder_categories: tuple[str, ...]
    target_fraction: tuple[float, float]
    qualitative_level: str
    occluder_count: int
    foreground_distance_m: tuple[float, float]
    placement_min_m: tuple[float, float, float]
    placement_max_m: tuple[float, float, float]
    may_overlap_repeated_elements: bool
    allow_camera_truncation: bool
    seed: int
    minimum_visible_fraction: float


@dataclass(frozen=True)
class AnnotationSettings:
    emit_ground_truth_grammar: bool
    grammar_source_kind: str
    emit_instance_masks: bool
    emit_semantic_masks: bool
    emit_terminal_annotations: bool
    emit_repeat_group_annotations: bool
    emit_visibility: bool
    grammar_path: str
    instance_table_path: str
    visibility_path: str


@dataclass(frozen=True)
class OutputSettings:
    output_root: str
    rgb: bool
    rgb_format: str
    rgb_bit_depth: int
    depth: bool
    depth_format: str
    depth_unit: str
    depth_semantics: str
    normals: bool
    normals_format: str
    normals_coordinate_system: str
    masks: bool
    mask_format: str
    mask_bit_depth: int
    semantic_masks: bool
    terminal_masks: bool
    instance_masks: bool
    alpha: bool
    object_ids: bool
    material_ids: bool
    geometry_format: str
    generate_preview: bool
    write_scene_file: bool
    write_checksums: bool


@dataclass(frozen=True)
class EditSpecification:
    edit_id: str
    operation: EditOperation
    target_role: str
    parameters: Mapping[str, Any]
    applicability: str
    camera_reuse: str
    changed_region_path: str
    unchanged_region_path: str
    edit_seed: int


@dataclass(frozen=True)
class SourceProvenance:
    source_config: str
    definitions: tuple[str, ...]
    authoring_tool: str
    created_at: str | None


@dataclass(frozen=True)
class SceneConfiguration:
    config_schema_version: str
    definitions_version: str
    dataset_id: str
    scene_id: str
    source_type: str
    family: SceneFamily
    subsets: tuple[str, ...]
    master_seed: int
    seed_policy_version: str
    seed_overrides: Mapping[str, int]
    derived_seeds: Mapping[str, int]
    coordinate_system: CoordinateSystem
    generator: GeneratorSettings
    render: RenderSettings
    camera: CameraPolicy
    structure: StructureVariation
    family_parameters: FamilyParameters
    appearance: AppearanceSettings
    lighting: LightingSettings
    occlusion: OcclusionSettings
    annotations: AnnotationSettings
    outputs: OutputSettings
    edits: tuple[EditSpecification, ...]
    expected_resource_profile: ResourceProfile
    extended_mode: bool
    notes: str | None
    provenance: SourceProvenance

    def to_dict(self) -> dict[str, Any]:
        """Return the fully explicit JSON representation."""

        value = cast(dict[str, Any], _json_value(asdict(self)))
        value["seed_policy"] = {
            "version": value.pop("seed_policy_version"),
            "overrides": value.pop("seed_overrides"),
            "streams": list(SEED_STREAMS),
        }
        render = cast(dict[str, Any], value["render"])
        render["background"] = {
            "kind": render.pop("background_kind"),
            "rgba": render.pop("background_rgba"),
            "has_valid_depth": render.pop("background_has_valid_depth"),
        }
        camera = cast(dict[str, Any], value["camera"])
        camera["split_counts"] = {
            "train": camera.pop("train_count"),
            "validation": camera.pop("validation_count"),
            "test": camera.pop("test_count"),
        }
        camera["position_bounds_m"] = {
            "min": camera.pop("position_min_m"),
            "max": camera.pop("position_max_m"),
        }
        camera["orientation_bounds_degrees"] = {
            "yaw": camera.pop("yaw_degrees"),
            "pitch": camera.pop("pitch_degrees"),
            "roll": camera.pop("roll_degrees"),
        }
        occlusion = cast(dict[str, Any], value["occlusion"])
        occlusion["placement_bounds_m"] = {
            "min": occlusion.pop("placement_min_m"),
            "max": occlusion.pop("placement_max_m"),
        }
        return value

    @property
    def full_digest(self) -> str:
        """Digest every resolved configuration field."""

        return content_digest(self.to_dict())

    @property
    def scientific_digest(self) -> str:
        """Digest only declared scientific identity fields."""

        return content_digest(scientific_projection(self))


@dataclass(frozen=True)
class SceneBlenderHandoff:
    """Validated metadata crossing the Python-to-Blender process boundary."""

    resolved_config_path: str
    output_root: str
    scene_id: str
    master_seed: int
    backend: str
    render_device: str
    generator_name: str
    generator_version: str
    timeout_seconds: float
    provenance_ref: str
    scientific_digest: str

    def ordered_script_arguments(self) -> tuple[str, ...]:
        """Return a shell-free ordered argument vector for a future generator."""

        return (
            "--config",
            self.resolved_config_path,
            "--output-root",
            self.output_root,
            "--scene-id",
            self.scene_id,
            "--master-seed",
            str(self.master_seed),
            "--backend",
            self.backend,
            "--device",
            self.render_device,
            "--generator-version",
            self.generator_version,
            "--provenance-ref",
            self.provenance_ref,
        )


class _Reader:
    def __init__(self, source: str) -> None:
        self.source = source

    def fail(
        self,
        path: tuple[str | int, ...],
        message: str,
        value: object = None,
        *,
        validator: str = "semantic",
    ) -> None:
        raise SchemaValidationError(
            (
                ValidationIssue(
                    "scene_configuration",
                    ValidationStage.SEMANTIC,
                    message,
                    self.source,
                    path,
                    validator=validator,
                    invalid_value=summarize_value(value),
                ),
            )
        )

    def obj(self, value: object, path: tuple[str | int, ...]) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.fail(path, "must be an object", value, validator="type")
        return cast(dict[str, Any], value)

    def exact(
        self,
        value: object,
        path: tuple[str | int, ...],
        keys: set[str],
    ) -> dict[str, Any]:
        result = self.obj(value, path)
        missing = sorted(keys - result.keys())
        extra = sorted(result.keys() - keys)
        if missing:
            self.fail(path, f"missing required fields: {', '.join(missing)}")
        if extra:
            self.fail(path, f"unknown fields: {', '.join(extra)}")
        return result

    def string(
        self, value: object, path: tuple[str | int, ...], *, nonempty: bool = True
    ) -> str:
        if not isinstance(value, str) or (nonempty and not value):
            self.fail(path, "must be a nonempty string", value, validator="type")
        return cast(str, value)

    def integer(
        self,
        value: object,
        path: tuple[str | int, ...],
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            self.fail(path, "must be an integer", value, validator="type")
        result = cast(int, value)
        if minimum is not None and result < minimum:
            self.fail(path, f"must be at least {minimum}", value, validator="minimum")
        if maximum is not None and result > maximum:
            self.fail(path, f"must be at most {maximum}", value, validator="maximum")
        return result

    def number(
        self,
        value: object,
        path: tuple[str | int, ...],
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
    ) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            self.fail(path, "must be a number", value, validator="type")
        result = float(cast(float, value))
        invalid_min = minimum is not None and (
            result <= minimum if exclusive_minimum else result < minimum
        )
        if invalid_min:
            wording = "greater than" if exclusive_minimum else "at least"
            self.fail(path, f"must be {wording} {minimum}", value, validator="minimum")
        if maximum is not None and result > maximum:
            self.fail(path, f"must be at most {maximum}", value, validator="maximum")
        return result

    def boolean(self, value: object, path: tuple[str | int, ...]) -> bool:
        if not isinstance(value, bool):
            self.fail(path, "must be a boolean", value, validator="type")
        return cast(bool, value)

    def enum(self, value: object, path: tuple[str | int, ...], cls: type[_E]) -> _E:
        raw = self.string(value, path)
        try:
            return cls(raw)
        except ValueError:
            accepted = ", ".join(item.value for item in cls)
            self.fail(path, f"must be one of: {accepted}", value, validator="enum")
        raise AssertionError("unreachable")

    def strings(
        self,
        value: object,
        path: tuple[str | int, ...],
        *,
        nonempty: bool = True,
        unique: bool = False,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            self.fail(path, "must be an array", value, validator="type")
        items = tuple(
            self.string(item, (*path, index))
            for index, item in enumerate(cast(list[object], value))
        )
        if nonempty and not items:
            self.fail(path, "must not be empty", value, validator="minItems")
        if unique and len(items) != len(set(items)):
            self.fail(
                path, "must contain unique values", value, validator="uniqueItems"
            )
        return items

    def integers(
        self,
        value: object,
        path: tuple[str | int, ...],
        *,
        minimum: int = 0,
    ) -> tuple[int, ...]:
        if not isinstance(value, list):
            self.fail(path, "must be an array", value, validator="type")
        items = tuple(
            self.integer(item, (*path, index), minimum=minimum)
            for index, item in enumerate(cast(list[object], value))
        )
        if len(items) != len(set(items)):
            self.fail(
                path, "must contain unique values", value, validator="uniqueItems"
            )
        return items

    def numbers(
        self,
        value: object,
        path: tuple[str | int, ...],
        count: int,
    ) -> tuple[float, ...]:
        if not isinstance(value, list) or len(value) != count:
            self.fail(path, f"must be an array of {count} numbers", value)
        return tuple(
            self.number(item, (*path, index))
            for index, item in enumerate(cast(list[object], value))
        )


_E = TypeVar("_E", bound=StrEnum)


def derive_seed(
    master_seed: int, stream: str, *, version: str = SEED_POLICY_VERSION
) -> int:
    """Derive one stable unsigned seed with a versioned SHA-256 contract."""

    if not 0 <= master_seed <= UINT32_MAX:
        raise ValueError("master_seed must be an unsigned 32-bit integer")
    if stream not in SEED_STREAMS:
        raise ValueError(f"unknown seed stream: {stream}")
    if version != SEED_POLICY_VERSION:
        raise ValueError(f"unsupported seed policy: {version}")
    payload = {"master_seed": master_seed, "policy": version, "stream": stream}
    digest = hashlib.sha256(canonical_json_bytes(cast(Any, payload))).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def derive_seeds(
    master_seed: int,
    overrides: Mapping[str, int] | None = None,
    *,
    version: str = SEED_POLICY_VERSION,
) -> dict[str, int]:
    """Derive all named streams, applying only explicit per-stream overrides."""

    selected = dict(overrides or {})
    unknown = sorted(selected.keys() - set(SEED_STREAMS))
    if unknown:
        raise ValueError(f"unknown seed overrides: {', '.join(unknown)}")
    for stream, seed in selected.items():
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed <= UINT32_MAX
        ):
            raise ValueError(f"invalid seed override for {stream}")
    return {
        stream: selected.get(stream, derive_seed(master_seed, stream, version=version))
        for stream in SEED_STREAMS
    }


def load_scene_configuration(path: Path) -> SceneConfiguration:
    """Load, strictly parse, and semantically validate one JSON configuration."""

    source = display_source(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DocumentLoadError(
            ValidationIssue(
                "scene_configuration",
                ValidationStage.LOADING,
                f"could not load JSON: {error}",
                source,
            )
        ) from error
    reader = _Reader(source)
    return _parse_configuration(reader, raw)


def resolve_scene_configuration(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> SceneConfiguration:
    """Resolve provenance path and derived seeds without changing scientific intent."""

    config = load_scene_configuration(path)
    if repository_root is None:
        return config
    root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("scene configuration must be within the repository") from error
    provenance = replace(config.provenance, source_config=relative)
    return replace(config, provenance=provenance)


def scientific_projection(config: SceneConfiguration) -> dict[str, Any]:
    """Return the versioned scientific identity projection."""

    value = config.to_dict()
    outputs = cast(dict[str, Any], value["outputs"])
    outputs.pop("output_root")
    provenance = cast(dict[str, Any], value["provenance"])
    provenance.pop("created_at")
    provenance.pop("source_config")
    value.pop("notes")
    return {
        "scientific_projection_version": SCIENTIFIC_PROJECTION_VERSION,
        "configuration": value,
    }


def write_resolved_scene_configuration(
    config: SceneConfiguration,
    output: Path,
    *,
    canonical_output: Path | None = None,
    digest_output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write deterministic pretty/canonical/digest resolution artifacts."""

    targets = [output]
    if canonical_output is not None:
        targets.append(canonical_output)
    if digest_output is not None:
        targets.append(digest_output)
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite: {', '.join(existing)}")
    value = config.to_dict()
    _write_bytes(output, pretty_json_bytes(value))
    if canonical_output is not None:
        _write_bytes(canonical_output, canonical_json_bytes(value))
    digest_record = {
        "config_schema_version": config.config_schema_version,
        "scene_id": config.scene_id,
        "full_digest": config.full_digest,
        "scientific_projection_version": SCIENTIFIC_PROJECTION_VERSION,
        "scientific_digest": config.scientific_digest,
    }
    if digest_output is not None:
        _write_bytes(digest_output, pretty_json_bytes(cast(Any, digest_record)))
    return digest_record


def build_blender_handoff(
    config: SceneConfiguration,
    *,
    resolved_config_path: str,
    output_root: str,
    backend: str,
    timeout_seconds: float,
    provenance_ref: str,
    render_device: str | None = None,
    generator_version: str | None = None,
) -> SceneBlenderHandoff:
    """Build validated, serializable metadata for the isolated Blender adapter."""

    if backend not in {"wsl", "windows"}:
        raise ValueError("backend must be wsl or windows")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    selected_device = render_device or config.render.compute_device
    if selected_device not in {"cpu", "gpu"}:
        raise ValueError("render_device must be cpu or gpu")
    _validate_embedded_path(resolved_config_path, allow_absolute=True)
    _validate_embedded_path(output_root, allow_absolute=True)
    _validate_embedded_path(provenance_ref, allow_absolute=True)
    return SceneBlenderHandoff(
        resolved_config_path,
        output_root,
        config.scene_id,
        config.master_seed,
        backend,
        selected_device,
        config.generator.name,
        generator_version or config.generator.version,
        timeout_seconds,
        provenance_ref,
        config.scientific_digest,
    )


def _parse_configuration(reader: _Reader, raw: object) -> SceneConfiguration:
    keys = {
        "config_schema_version",
        "definitions_version",
        "dataset_id",
        "scene_id",
        "source_type",
        "family",
        "subsets",
        "master_seed",
        "seed_policy",
        "derived_seeds",
        "coordinate_system",
        "generator",
        "render",
        "camera",
        "structure",
        "family_parameters",
        "appearance",
        "lighting",
        "occlusion",
        "annotations",
        "outputs",
        "edits",
        "expected_resource_profile",
        "extended_mode",
        "notes",
        "provenance",
    }
    value = reader.exact(raw, (), keys)
    schema_version = reader.string(
        value["config_schema_version"], ("config_schema_version",)
    )
    if schema_version != SCENE_CONFIG_SCHEMA_VERSION:
        reader.fail(
            ("config_schema_version",),
            f"must equal {SCENE_CONFIG_SCHEMA_VERSION}",
            schema_version,
        )
    definitions_version = reader.string(
        value["definitions_version"], ("definitions_version",)
    )
    dataset_id = reader.string(value["dataset_id"], ("dataset_id",))
    if not DATASET_ID_RE.fullmatch(dataset_id):
        reader.fail(
            ("dataset_id",), "must use lowercase identifier characters", dataset_id
        )
    scene_id = reader.string(value["scene_id"], ("scene_id",))
    family = reader.enum(value["family"], ("family",), SceneFamily)
    master_seed = reader.integer(
        value["master_seed"], ("master_seed",), minimum=0, maximum=UINT32_MAX
    )
    match = SCENE_ID_RE.fullmatch(scene_id)
    if (
        match is None
        or match.group(1) != family.value
        or int(match.group(2)) != master_seed
    ):
        reader.fail(
            ("scene_id",),
            "must be synthetic, include the selected family, and end with "
            "the master seed",
            scene_id,
        )
    source_type = reader.string(value["source_type"], ("source_type",))
    if source_type != "synthetic":
        reader.fail(("source_type",), "must equal synthetic", source_type)
    subsets = reader.strings(value["subsets"], ("subsets",), unique=True)
    accepted_subsets = {"smoke", "development", "calibration", "frozen_v1", "reserve"}
    if not set(subsets) <= accepted_subsets:
        reader.fail(("subsets",), "contains an invalid synthetic subset", subsets)
    policy = reader.exact(
        value["seed_policy"],
        ("seed_policy",),
        {"version", "overrides", "streams"},
    )
    policy_version = reader.string(policy["version"], ("seed_policy", "version"))
    if policy_version != SEED_POLICY_VERSION:
        reader.fail(
            ("seed_policy", "version"),
            f"must equal {SEED_POLICY_VERSION}",
            policy_version,
        )
    streams = reader.strings(policy["streams"], ("seed_policy", "streams"), unique=True)
    if streams != SEED_STREAMS:
        reader.fail(
            ("seed_policy", "streams"),
            "must list the canonical streams in order",
            streams,
        )
    override_raw = reader.obj(policy["overrides"], ("seed_policy", "overrides"))
    overrides: dict[str, int] = {}
    for stream, seed in override_raw.items():
        if stream not in SEED_STREAMS:
            reader.fail(
                ("seed_policy", "overrides", stream), "unknown seed stream", stream
            )
        overrides[stream] = reader.integer(
            seed, ("seed_policy", "overrides", stream), minimum=0, maximum=UINT32_MAX
        )
    derived_raw = reader.obj(value["derived_seeds"], ("derived_seeds",))
    if set(derived_raw) != set(SEED_STREAMS):
        reader.fail(
            ("derived_seeds",),
            "must contain exactly all canonical seed streams",
            derived_raw,
        )
    derived = {
        stream: reader.integer(
            derived_raw[stream],
            ("derived_seeds", stream),
            minimum=0,
            maximum=UINT32_MAX,
        )
        for stream in SEED_STREAMS
    }
    expected_seeds = derive_seeds(master_seed, overrides, version=policy_version)
    if derived != expected_seeds:
        reader.fail(
            ("derived_seeds",),
            "does not match the versioned derivation policy",
            derived,
        )
    config = SceneConfiguration(
        schema_version,
        definitions_version,
        dataset_id,
        scene_id,
        source_type,
        family,
        subsets,
        master_seed,
        policy_version,
        overrides,
        derived,
        _parse_coordinate(reader, value["coordinate_system"]),
        _parse_generator(reader, value["generator"]),
        _parse_render(reader, value["render"]),
        _parse_camera(reader, value["camera"], family),
        _parse_structure(reader, value["structure"]),
        _parse_family(reader, value["family_parameters"], family),
        _parse_appearance(reader, value["appearance"]),
        _parse_lighting(reader, value["lighting"]),
        _parse_occlusion(reader, value["occlusion"]),
        _parse_annotations(reader, value["annotations"]),
        _parse_outputs(reader, value["outputs"]),
        _parse_edits(reader, value["edits"]),
        reader.enum(
            value["expected_resource_profile"],
            ("expected_resource_profile",),
            ResourceProfile,
        ),
        reader.boolean(value["extended_mode"], ("extended_mode",)),
        (
            None
            if value["notes"] is None
            else reader.string(value["notes"], ("notes",), nonempty=False)
        ),
        _parse_provenance(reader, value["provenance"]),
    )
    _validate_semantics(reader, config)
    return config


def _parse_coordinate(reader: _Reader, raw: object) -> CoordinateSystem:
    path = ("coordinate_system",)
    value = reader.exact(
        raw, path, {"handedness", "up_axis", "unit", "camera_axes", "matrix_layout"}
    )
    axes = reader.exact(
        value["camera_axes"], (*path, "camera_axes"), {"right", "up", "forward"}
    )
    result = CoordinateSystem(
        reader.string(value["handedness"], (*path, "handedness")),
        reader.string(value["up_axis"], (*path, "up_axis")),
        reader.string(value["unit"], (*path, "unit")),
        {
            name: reader.string(axes[name], (*path, "camera_axes", name))
            for name in ("right", "up", "forward")
        },
        reader.string(value["matrix_layout"], (*path, "matrix_layout")),
    )
    expected = (
        "right",
        "+Z",
        "meters",
        {"right": "+X", "up": "+Y", "forward": "-Z"},
        "row_major",
    )
    if (
        result.handedness,
        result.up_axis,
        result.unit,
        dict(result.camera_axes),
        result.matrix_layout,
    ) != expected:
        reader.fail(
            path, "must match the accepted world and camera coordinate convention", raw
        )
    return result


def _parse_generator(reader: _Reader, raw: object) -> GeneratorSettings:
    path = ("generator",)
    value = reader.exact(
        raw, path, {"name", "version", "implementation_status", "external_assets"}
    )
    status = reader.string(
        value["implementation_status"], (*path, "implementation_status")
    )
    if status != "configuration_only":
        reader.fail(
            (*path, "implementation_status"), "must equal configuration_only", status
        )
    assets = reader.strings(
        value["external_assets"],
        (*path, "external_assets"),
        nonempty=False,
        unique=True,
    )
    for index, asset in enumerate(assets):
        _safe_path(reader, asset, (*path, "external_assets", index))
    return GeneratorSettings(
        reader.string(value["name"], (*path, "name")),
        reader.string(value["version"], (*path, "version")),
        status,
        assets,
    )


def _parse_render(reader: _Reader, raw: object) -> RenderSettings:
    path = ("render",)
    keys = {
        "engine",
        "compute_device",
        "width",
        "height",
        "resolution_percentage",
        "samples",
        "denoise",
        "color_space",
        "view_transform",
        "look",
        "exposure",
        "gamma",
        "background",
    }
    value = reader.exact(raw, path, keys)
    background = reader.exact(
        value["background"], (*path, "background"), {"kind", "rgba", "has_valid_depth"}
    )
    rgba = reader.numbers(background["rgba"], (*path, "background", "rgba"), 4)
    if any(component < 0 or component > 1 for component in rgba):
        reader.fail(
            (*path, "background", "rgba"), "components must be within [0, 1]", rgba
        )
    engine = reader.string(value["engine"], (*path, "engine"))
    if engine not in {"cycles", "eevee_next"}:
        reader.fail((*path, "engine"), "must be cycles or eevee_next", engine)
    device = reader.string(value["compute_device"], (*path, "compute_device"))
    if device not in {"cpu", "gpu", "auto"}:
        reader.fail((*path, "compute_device"), "must be cpu, gpu, or auto", device)
    kind = reader.string(background["kind"], (*path, "background", "kind"))
    if kind not in {"solid", "environment", "transparent"}:
        reader.fail((*path, "background", "kind"), "unsupported background kind", kind)
    return RenderSettings(
        engine,
        device,
        reader.integer(value["width"], (*path, "width"), minimum=1),
        reader.integer(value["height"], (*path, "height"), minimum=1),
        reader.integer(
            value["resolution_percentage"],
            (*path, "resolution_percentage"),
            minimum=1,
            maximum=100,
        ),
        reader.integer(value["samples"], (*path, "samples"), minimum=1),
        reader.boolean(value["denoise"], (*path, "denoise")),
        reader.string(value["color_space"], (*path, "color_space")),
        reader.string(value["view_transform"], (*path, "view_transform")),
        reader.string(value["look"], (*path, "look"), nonempty=False),
        reader.number(value["exposure"], (*path, "exposure")),
        reader.number(
            value["gamma"], (*path, "gamma"), minimum=0, exclusive_minimum=True
        ),
        kind,
        cast(tuple[float, float, float, float], rgba),
        reader.boolean(
            background["has_valid_depth"], (*path, "background", "has_valid_depth")
        ),
    )


def _parse_camera(reader: _Reader, raw: object, family: SceneFamily) -> CameraPolicy:
    path = ("camera",)
    keys = {
        "trajectory",
        "split_counts",
        "image_width",
        "image_height",
        "focal_mode",
        "focal_length_mm",
        "horizontal_fov_degrees",
        "principal_point",
        "principal_point_px",
        "near_m",
        "far_m",
        "position_bounds_m",
        "orientation_bounds_degrees",
        "minimum_separation_m",
        "minimum_angular_separation_degrees",
        "minimum_structure_coverage",
        "seed",
        "family_intent",
    }
    value = reader.exact(raw, path, keys)
    counts = reader.exact(
        value["split_counts"], (*path, "split_counts"), {"train", "validation", "test"}
    )
    positions = reader.exact(
        value["position_bounds_m"], (*path, "position_bounds_m"), {"min", "max"}
    )
    orientations = reader.exact(
        value["orientation_bounds_degrees"],
        (*path, "orientation_bounds_degrees"),
        {"yaw", "pitch", "roll"},
    )
    trajectory = reader.string(value["trajectory"], (*path, "trajectory"))
    expected_trajectory = {
        SceneFamily.FACADE: "bounded_frontal_arc",
        SceneFamily.CORRIDOR: "corridor_path",
        SceneFamily.COLONNADE: "colonnade_arc",
    }[family]
    if trajectory != expected_trajectory:
        reader.fail(
            (*path, "trajectory"),
            f"must equal {expected_trajectory} for {family.value}",
            trajectory,
        )
    focal_mode = reader.string(value["focal_mode"], (*path, "focal_mode"))
    if focal_mode not in {"focal_length_mm", "horizontal_fov_degrees"}:
        reader.fail((*path, "focal_mode"), "unsupported focal mode", focal_mode)
    focal_length = (
        None
        if value["focal_length_mm"] is None
        else reader.number(
            value["focal_length_mm"],
            (*path, "focal_length_mm"),
            minimum=0,
            exclusive_minimum=True,
        )
    )
    fov = (
        None
        if value["horizontal_fov_degrees"] is None
        else reader.number(
            value["horizontal_fov_degrees"],
            (*path, "horizontal_fov_degrees"),
            minimum=1,
            maximum=179,
        )
    )
    if (focal_mode == "focal_length_mm") != (focal_length is not None) or (
        focal_mode == "horizontal_fov_degrees"
    ) != (fov is not None):
        reader.fail(
            path, "focal mode must select exactly one non-null focal value", raw
        )
    principal = reader.string(value["principal_point"], (*path, "principal_point"))
    if principal not in {"centered", "explicit"}:
        reader.fail(
            (*path, "principal_point"), "must be centered or explicit", principal
        )
    principal_px = (
        None
        if value["principal_point_px"] is None
        else cast(
            tuple[float, float],
            reader.numbers(
                value["principal_point_px"], (*path, "principal_point_px"), 2
            ),
        )
    )
    if (principal == "explicit") != (principal_px is not None):
        reader.fail(
            path,
            "explicit principal point requires coordinates and centered requires null",
            raw,
        )
    intent = reader.obj(value["family_intent"], (*path, "family_intent"))
    _validate_camera_intent(reader, intent, family)
    return CameraPolicy(
        trajectory,
        reader.integer(counts["train"], (*path, "split_counts", "train"), minimum=1),
        reader.integer(
            counts["validation"], (*path, "split_counts", "validation"), minimum=1
        ),
        reader.integer(counts["test"], (*path, "split_counts", "test"), minimum=1),
        reader.integer(value["image_width"], (*path, "image_width"), minimum=1),
        reader.integer(value["image_height"], (*path, "image_height"), minimum=1),
        focal_mode,
        focal_length,
        fov,
        principal,
        principal_px,
        reader.number(
            value["near_m"], (*path, "near_m"), minimum=0, exclusive_minimum=True
        ),
        reader.number(
            value["far_m"], (*path, "far_m"), minimum=0, exclusive_minimum=True
        ),
        cast(
            tuple[float, float, float],
            reader.numbers(positions["min"], (*path, "position_bounds_m", "min"), 3),
        ),
        cast(
            tuple[float, float, float],
            reader.numbers(positions["max"], (*path, "position_bounds_m", "max"), 3),
        ),
        cast(
            tuple[float, float],
            reader.numbers(
                orientations["yaw"], (*path, "orientation_bounds_degrees", "yaw"), 2
            ),
        ),
        cast(
            tuple[float, float],
            reader.numbers(
                orientations["pitch"], (*path, "orientation_bounds_degrees", "pitch"), 2
            ),
        ),
        cast(
            tuple[float, float],
            reader.numbers(
                orientations["roll"], (*path, "orientation_bounds_degrees", "roll"), 2
            ),
        ),
        reader.number(
            value["minimum_separation_m"],
            (*path, "minimum_separation_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["minimum_angular_separation_degrees"],
            (*path, "minimum_angular_separation_degrees"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["minimum_structure_coverage"],
            (*path, "minimum_structure_coverage"),
            minimum=0,
            maximum=1,
        ),
        reader.integer(value["seed"], (*path, "seed"), minimum=0, maximum=UINT32_MAX),
        dict(intent),
    )


def _validate_camera_intent(
    reader: _Reader, value: dict[str, Any], family: SceneFamily
) -> None:
    path = ("camera", "family_intent")
    keys_by_family = {
        SceneFamily.FACADE: {"arc_radius_m", "azimuth_degrees", "height_m"},
        SceneFamily.CORRIDOR: {"longitudinal_range_m", "lateral_offset_m", "height_m"},
        SceneFamily.COLONNADE: {"radius_m", "azimuth_degrees", "height_m"},
    }
    reader.exact(value, path, keys_by_family[family])
    for key, item in value.items():
        bounds = reader.numbers(item, (*path, key), 2)
        if bounds[0] > bounds[1]:
            reader.fail((*path, key), "minimum must not exceed maximum", item)
        if key in {"arc_radius_m", "radius_m", "height_m"} and bounds[0] <= 0:
            reader.fail((*path, key), "distance bounds must be positive", item)


def _parse_structure(reader: _Reader, raw: object) -> StructureVariation:
    path = ("structure",)
    keys = {
        "repetition_level",
        "spacing_regime",
        "terminal_complexity",
        "regularity",
        "missing_instance_policy",
        "negative_control",
        "negative_control_label",
        "expected_repeat_groups",
        "expected_canonical_terminal_categories",
    }
    value = reader.exact(raw, path, keys)
    negative = reader.boolean(value["negative_control"], (*path, "negative_control"))
    label = (
        None
        if value["negative_control_label"] is None
        else reader.string(
            value["negative_control_label"], (*path, "negative_control_label")
        )
    )
    if negative != (label is not None):
        reader.fail(
            path,
            "negative controls require a label and positive scenes require null",
            raw,
        )
    return StructureVariation(
        reader.enum(
            value["repetition_level"], (*path, "repetition_level"), RepetitionLevel
        ),
        reader.enum(value["spacing_regime"], (*path, "spacing_regime"), SpacingRegime),
        reader.enum(
            value["terminal_complexity"],
            (*path, "terminal_complexity"),
            TerminalComplexity,
        ),
        reader.enum(value["regularity"], (*path, "regularity"), StructuralRegularity),
        reader.enum(
            value["missing_instance_policy"],
            (*path, "missing_instance_policy"),
            MissingInstancePolicy,
        ),
        negative,
        label,
        reader.strings(
            value["expected_repeat_groups"],
            (*path, "expected_repeat_groups"),
            unique=True,
        ),
        reader.strings(
            value["expected_canonical_terminal_categories"],
            (*path, "expected_canonical_terminal_categories"),
            unique=True,
        ),
    )


def _parse_family(
    reader: _Reader, raw: object, family: SceneFamily
) -> FamilyParameters:
    value = reader.obj(raw, ("family_parameters",))
    payload_family = reader.enum(
        value.get("family"), ("family_parameters", "family"), SceneFamily
    )
    if payload_family is not family:
        reader.fail(
            ("family_parameters", "family"),
            "does not match top-level family",
            payload_family.value,
        )
    if family is SceneFamily.FACADE:
        return _parse_facade(reader, value)
    if family is SceneFamily.CORRIDOR:
        return _parse_corridor(reader, value)
    return _parse_colonnade(reader, value)


def _parse_facade(reader: _Reader, raw: object) -> FacadeParameters:
    path = ("family_parameters",)
    keys = {
        "family",
        "rows",
        "columns",
        "horizontal_spacing_m",
        "vertical_spacing_m",
        "terminal_width_m",
        "terminal_height_m",
        "terminal_depth_m",
        "wall_width_m",
        "wall_height_m",
        "wall_depth_m",
        "margins_m",
        "floor_band_height_m",
        "sill_depth_m",
        "recess_depth_m",
        "balcony_mode",
        "panel_mode",
        "missing_pattern",
        "missing_instances",
    }
    value = reader.exact(raw, path, keys)
    margins_raw = reader.exact(
        value["margins_m"], (*path, "margins_m"), {"left", "right", "bottom", "top"}
    )
    margins = {
        key: reader.number(margins_raw[key], (*path, "margins_m", key), minimum=0)
        for key in ("left", "right", "bottom", "top")
    }
    balcony = reader.string(value["balcony_mode"], (*path, "balcony_mode"))
    panel = reader.string(value["panel_mode"], (*path, "panel_mode"))
    if balcony not in {"none", "periodic", "grouped"}:
        reader.fail((*path, "balcony_mode"), "unsupported balcony mode", balcony)
    if panel not in {"none", "simple", "framed"}:
        reader.fail((*path, "panel_mode"), "unsupported panel mode", panel)
    return FacadeParameters(
        SceneFamily.FACADE,
        reader.integer(value["rows"], (*path, "rows"), minimum=1),
        reader.integer(value["columns"], (*path, "columns"), minimum=1),
        reader.number(
            value["horizontal_spacing_m"],
            (*path, "horizontal_spacing_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["vertical_spacing_m"],
            (*path, "vertical_spacing_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["terminal_width_m"],
            (*path, "terminal_width_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["terminal_height_m"],
            (*path, "terminal_height_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["terminal_depth_m"],
            (*path, "terminal_depth_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["wall_width_m"],
            (*path, "wall_width_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["wall_height_m"],
            (*path, "wall_height_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["wall_depth_m"],
            (*path, "wall_depth_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        margins,
        reader.number(
            value["floor_band_height_m"], (*path, "floor_band_height_m"), minimum=0
        ),
        reader.number(value["sill_depth_m"], (*path, "sill_depth_m"), minimum=0),
        reader.number(value["recess_depth_m"], (*path, "recess_depth_m"), minimum=0),
        balcony,
        panel,
        reader.string(value["missing_pattern"], (*path, "missing_pattern")),
        reader.integers(value["missing_instances"], (*path, "missing_instances")),
    )


def _parse_corridor(reader: _Reader, raw: object) -> CorridorParameters:
    path = ("family_parameters",)
    keys = {
        "family",
        "length_m",
        "width_m",
        "height_m",
        "bay_count",
        "bay_spacing_m",
        "terminal_width_m",
        "terminal_height_m",
        "terminal_depth_m",
        "sidedness",
        "ceiling_light_mode",
        "end_wall_mode",
        "furniture_occluder_mode",
        "end_margin_m",
        "missing_pattern",
        "missing_instances",
    }
    value = reader.exact(raw, path, keys)
    sidedness = reader.string(value["sidedness"], (*path, "sidedness"))
    if sidedness not in {"one_sided", "two_sided"}:
        reader.fail((*path, "sidedness"), "must be one_sided or two_sided", sidedness)
    modes = {
        "ceiling_light_mode": {"none", "periodic", "continuous"},
        "end_wall_mode": {"closed", "open", "mixed"},
        "furniture_occluder_mode": {"none", "sparse", "moderate"},
    }
    parsed_modes: dict[str, str] = {}
    for key, accepted in modes.items():
        parsed_modes[key] = reader.string(value[key], (*path, key))
        if parsed_modes[key] not in accepted:
            reader.fail(
                (*path, key),
                f"must be one of: {', '.join(sorted(accepted))}",
                parsed_modes[key],
            )
    return CorridorParameters(
        SceneFamily.CORRIDOR,
        reader.number(
            value["length_m"], (*path, "length_m"), minimum=0, exclusive_minimum=True
        ),
        reader.number(
            value["width_m"], (*path, "width_m"), minimum=0, exclusive_minimum=True
        ),
        reader.number(
            value["height_m"], (*path, "height_m"), minimum=0, exclusive_minimum=True
        ),
        reader.integer(value["bay_count"], (*path, "bay_count"), minimum=1),
        reader.number(
            value["bay_spacing_m"],
            (*path, "bay_spacing_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["terminal_width_m"],
            (*path, "terminal_width_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["terminal_height_m"],
            (*path, "terminal_height_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["terminal_depth_m"],
            (*path, "terminal_depth_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        sidedness,
        parsed_modes["ceiling_light_mode"],
        parsed_modes["end_wall_mode"],
        parsed_modes["furniture_occluder_mode"],
        reader.number(value["end_margin_m"], (*path, "end_margin_m"), minimum=0),
        reader.string(value["missing_pattern"], (*path, "missing_pattern")),
        reader.integers(value["missing_instances"], (*path, "missing_instances")),
    )


def _parse_colonnade(reader: _Reader, raw: object) -> ColonnadeParameters:
    path = ("family_parameters",)
    keys = {
        "family",
        "count",
        "spacing_m",
        "terminal_radius_m",
        "terminal_width_m",
        "terminal_height_m",
        "plinth_height_m",
        "capital_height_m",
        "span_mode",
        "row_count",
        "layout",
        "floor_width_m",
        "floor_length_m",
        "floor_thickness_m",
        "background_mode",
        "missing_pattern",
        "missing_instances",
    }
    value = reader.exact(raw, path, keys)
    radius = (
        None
        if value["terminal_radius_m"] is None
        else reader.number(
            value["terminal_radius_m"],
            (*path, "terminal_radius_m"),
            minimum=0,
            exclusive_minimum=True,
        )
    )
    width = (
        None
        if value["terminal_width_m"] is None
        else reader.number(
            value["terminal_width_m"],
            (*path, "terminal_width_m"),
            minimum=0,
            exclusive_minimum=True,
        )
    )
    if (radius is None) == (width is None):
        reader.fail(path, "exactly one terminal radius or width must be provided", raw)
    span = reader.string(value["span_mode"], (*path, "span_mode"))
    layout = reader.string(value["layout"], (*path, "layout"))
    background = reader.string(value["background_mode"], (*path, "background_mode"))
    if span not in {"arch", "lintel"}:
        reader.fail((*path, "span_mode"), "must be arch or lintel", span)
    if layout not in {"linear", "double_row", "partial_perimeter"}:
        reader.fail((*path, "layout"), "unsupported colonnade layout", layout)
    if background not in {"open", "wall", "environment"}:
        reader.fail(
            (*path, "background_mode"), "unsupported background mode", background
        )
    return ColonnadeParameters(
        SceneFamily.COLONNADE,
        reader.integer(value["count"], (*path, "count"), minimum=1),
        reader.number(
            value["spacing_m"], (*path, "spacing_m"), minimum=0, exclusive_minimum=True
        ),
        radius,
        width,
        reader.number(
            value["terminal_height_m"],
            (*path, "terminal_height_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(value["plinth_height_m"], (*path, "plinth_height_m"), minimum=0),
        reader.number(
            value["capital_height_m"], (*path, "capital_height_m"), minimum=0
        ),
        span,
        reader.integer(value["row_count"], (*path, "row_count"), minimum=1, maximum=2),
        layout,
        reader.number(
            value["floor_width_m"],
            (*path, "floor_width_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["floor_length_m"],
            (*path, "floor_length_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        reader.number(
            value["floor_thickness_m"],
            (*path, "floor_thickness_m"),
            minimum=0,
            exclusive_minimum=True,
        ),
        background,
        reader.string(value["missing_pattern"], (*path, "missing_pattern")),
        reader.integers(value["missing_instances"], (*path, "missing_instances")),
    )


def _parse_appearance(reader: _Reader, raw: object) -> AppearanceSettings:
    path = ("appearance",)
    keys = {
        "regime",
        "base_material",
        "base_color",
        "roughness",
        "metallic",
        "palette",
        "roughness_range",
        "metallic_range",
        "variation_amplitude",
        "variation_groups",
        "group_size",
        "alternate_period",
        "texture_mode",
        "seed",
        "exact_canonical_reuse",
        "residual_policy_label",
    }
    value = reader.exact(raw, path, keys)
    base_color = cast(
        tuple[float, float, float, float],
        reader.numbers(value["base_color"], (*path, "base_color"), 4),
    )
    if any(component < 0 or component > 1 for component in base_color):
        reader.fail((*path, "base_color"), "components must be within [0, 1]")
    palette_raw = value["palette"]
    if not isinstance(palette_raw, list) or not palette_raw:
        reader.fail((*path, "palette"), "must be a nonempty array", palette_raw)
    palette: list[tuple[float, float, float, float]] = []
    for index, color in enumerate(cast(list[object], palette_raw)):
        components = reader.numbers(color, (*path, "palette", index), 4)
        if any(component < 0 or component > 1 for component in components):
            reader.fail(
                (*path, "palette", index), "components must be within [0, 1]", color
            )
        palette.append(cast(tuple[float, float, float, float], components))
    roughness = cast(
        tuple[float, float],
        reader.numbers(value["roughness_range"], (*path, "roughness_range"), 2),
    )
    metallic = cast(
        tuple[float, float],
        reader.numbers(value["metallic_range"], (*path, "metallic_range"), 2),
    )
    for name, bounds in (("roughness_range", roughness), ("metallic_range", metallic)):
        if bounds[0] < 0 or bounds[1] > 1 or bounds[0] > bounds[1]:
            reader.fail((*path, name), "must be ordered bounds within [0, 1]", bounds)
    texture_mode = reader.string(value["texture_mode"], (*path, "texture_mode"))
    if texture_mode not in {"none", "procedural", "external"}:
        reader.fail((*path, "texture_mode"), "unsupported texture mode", texture_mode)
    return AppearanceSettings(
        reader.enum(value["regime"], (*path, "regime"), AppearanceRegime),
        reader.string(value["base_material"], (*path, "base_material")),
        base_color,
        reader.number(value["roughness"], (*path, "roughness"), minimum=0, maximum=1),
        (
            None
            if value["metallic"] is None
            else reader.number(
                value["metallic"], (*path, "metallic"), minimum=0, maximum=1
            )
        ),
        tuple(palette),
        roughness,
        metallic,
        reader.number(
            value["variation_amplitude"],
            (*path, "variation_amplitude"),
            minimum=0,
            maximum=1,
        ),
        reader.strings(
            value["variation_groups"],
            (*path, "variation_groups"),
            nonempty=False,
            unique=True,
        ),
        None
        if value["group_size"] is None
        else reader.integer(value["group_size"], (*path, "group_size"), minimum=1),
        None
        if value["alternate_period"] is None
        else reader.integer(
            value["alternate_period"], (*path, "alternate_period"), minimum=2
        ),
        texture_mode,
        reader.integer(value["seed"], (*path, "seed"), minimum=0, maximum=UINT32_MAX),
        reader.boolean(
            value["exact_canonical_reuse"], (*path, "exact_canonical_reuse")
        ),
        reader.string(value["residual_policy_label"], (*path, "residual_policy_label")),
    )


def _parse_lighting(reader: _Reader, raw: object) -> LightingSettings:
    path = ("lighting",)
    keys = {
        "regime",
        "world_energy",
        "world_color",
        "key_light_type",
        "key_energy",
        "key_color",
        "key_position_m",
        "energy_range",
        "color_temperature_kelvin",
        "direction_degrees",
        "deterministic_variation",
        "source_count",
        "fill_light_count",
        "fill_light_energy",
        "spatial_variation_amplitude",
        "seed",
        "cast_shadows",
        "exposure_intent",
    }
    value = reader.exact(raw, path, keys)
    energy = cast(
        tuple[float, float],
        reader.numbers(value["energy_range"], (*path, "energy_range"), 2),
    )
    temperature = cast(
        tuple[float, float],
        reader.numbers(
            value["color_temperature_kelvin"], (*path, "color_temperature_kelvin"), 2
        ),
    )
    if energy[0] <= 0 or energy[0] > energy[1]:
        reader.fail((*path, "energy_range"), "must be ordered positive bounds", energy)
    if temperature[0] < 1000 or temperature[0] > temperature[1]:
        reader.fail(
            (*path, "color_temperature_kelvin"),
            "must be ordered and physically plausible",
            temperature,
        )
    return LightingSettings(
        reader.enum(value["regime"], (*path, "regime"), LightingRegime),
        reader.number(value["world_energy"], (*path, "world_energy"), minimum=0),
        cast(
            tuple[float, float, float],
            reader.numbers(value["world_color"], (*path, "world_color"), 3),
        ),
        reader.string(value["key_light_type"], (*path, "key_light_type")),
        reader.number(
            value["key_energy"],
            (*path, "key_energy"),
            minimum=0,
            exclusive_minimum=True,
        ),
        cast(
            tuple[float, float, float],
            reader.numbers(value["key_color"], (*path, "key_color"), 3),
        ),
        cast(
            tuple[float, float, float],
            reader.numbers(value["key_position_m"], (*path, "key_position_m"), 3),
        ),
        energy,
        temperature,
        cast(
            tuple[float, float, float],
            reader.numbers(value["direction_degrees"], (*path, "direction_degrees"), 3),
        ),
        reader.boolean(
            value["deterministic_variation"], (*path, "deterministic_variation")
        ),
        reader.integer(value["source_count"], (*path, "source_count"), minimum=1),
        reader.integer(
            value["fill_light_count"], (*path, "fill_light_count"), minimum=0
        ),
        reader.number(
            value["fill_light_energy"], (*path, "fill_light_energy"), minimum=0
        ),
        reader.number(
            value["spatial_variation_amplitude"],
            (*path, "spatial_variation_amplitude"),
            minimum=0,
            maximum=1,
        ),
        reader.integer(value["seed"], (*path, "seed"), minimum=0, maximum=UINT32_MAX),
        reader.boolean(value["cast_shadows"], (*path, "cast_shadows")),
        reader.string(value["exposure_intent"], (*path, "exposure_intent")),
    )


def _parse_occlusion(reader: _Reader, raw: object) -> OcclusionSettings:
    path = ("occlusion",)
    keys = {
        "regime",
        "occluder_categories",
        "target_fraction",
        "qualitative_level",
        "occluder_count",
        "foreground_distance_m",
        "placement_bounds_m",
        "may_overlap_repeated_elements",
        "allow_camera_truncation",
        "seed",
        "minimum_visible_fraction",
    }
    value = reader.exact(raw, path, keys)
    target = cast(
        tuple[float, float],
        reader.numbers(value["target_fraction"], (*path, "target_fraction"), 2),
    )
    distance = cast(
        tuple[float, float],
        reader.numbers(
            value["foreground_distance_m"], (*path, "foreground_distance_m"), 2
        ),
    )
    placement = reader.exact(
        value["placement_bounds_m"],
        (*path, "placement_bounds_m"),
        {"min", "max"},
    )
    placement_min = cast(
        tuple[float, float, float],
        reader.numbers(placement["min"], (*path, "placement_bounds_m", "min"), 3),
    )
    placement_max = cast(
        tuple[float, float, float],
        reader.numbers(placement["max"], (*path, "placement_bounds_m", "max"), 3),
    )
    if any(low > high for low, high in zip(placement_min, placement_max, strict=True)):
        reader.fail(
            (*path, "placement_bounds_m"),
            "minimums must not exceed maximums",
        )
    if target[0] < 0 or target[1] > 1 or target[0] > target[1]:
        reader.fail(
            (*path, "target_fraction"), "must be ordered bounds within [0, 1]", target
        )
    if distance[0] < 0 or distance[0] > distance[1]:
        reader.fail(
            (*path, "foreground_distance_m"),
            "must be ordered nonnegative bounds",
            distance,
        )
    return OcclusionSettings(
        reader.enum(value["regime"], (*path, "regime"), OcclusionRegime),
        reader.strings(
            value["occluder_categories"],
            (*path, "occluder_categories"),
            nonempty=False,
            unique=True,
        ),
        target,
        reader.string(value["qualitative_level"], (*path, "qualitative_level")),
        reader.integer(value["occluder_count"], (*path, "occluder_count"), minimum=0),
        distance,
        placement_min,
        placement_max,
        reader.boolean(
            value["may_overlap_repeated_elements"],
            (*path, "may_overlap_repeated_elements"),
        ),
        reader.boolean(
            value["allow_camera_truncation"], (*path, "allow_camera_truncation")
        ),
        reader.integer(value["seed"], (*path, "seed"), minimum=0, maximum=UINT32_MAX),
        reader.number(
            value["minimum_visible_fraction"],
            (*path, "minimum_visible_fraction"),
            minimum=0,
            maximum=1,
        ),
    )


def _parse_annotations(reader: _Reader, raw: object) -> AnnotationSettings:
    path = ("annotations",)
    keys = {
        "emit_ground_truth_grammar",
        "grammar_source_kind",
        "emit_instance_masks",
        "emit_semantic_masks",
        "emit_terminal_annotations",
        "emit_repeat_group_annotations",
        "emit_visibility",
        "grammar_path",
        "instance_table_path",
        "visibility_path",
    }
    value = reader.exact(raw, path, keys)
    result = AnnotationSettings(
        reader.boolean(
            value["emit_ground_truth_grammar"], (*path, "emit_ground_truth_grammar")
        ),
        reader.string(value["grammar_source_kind"], (*path, "grammar_source_kind")),
        reader.boolean(value["emit_instance_masks"], (*path, "emit_instance_masks")),
        reader.boolean(value["emit_semantic_masks"], (*path, "emit_semantic_masks")),
        reader.boolean(
            value["emit_terminal_annotations"], (*path, "emit_terminal_annotations")
        ),
        reader.boolean(
            value["emit_repeat_group_annotations"],
            (*path, "emit_repeat_group_annotations"),
        ),
        reader.boolean(value["emit_visibility"], (*path, "emit_visibility")),
        reader.string(value["grammar_path"], (*path, "grammar_path")),
        reader.string(value["instance_table_path"], (*path, "instance_table_path")),
        reader.string(value["visibility_path"], (*path, "visibility_path")),
    )
    if result.grammar_source_kind != "ground_truth":
        reader.fail(
            (*path, "grammar_source_kind"),
            "must equal ground_truth",
            result.grammar_source_kind,
        )
    for name in ("grammar_path", "instance_table_path", "visibility_path"):
        _safe_path(reader, cast(str, getattr(result, name)), (*path, name))
    return result


def _parse_outputs(reader: _Reader, raw: object) -> OutputSettings:
    path = ("outputs",)
    keys = {
        "output_root",
        "rgb",
        "rgb_format",
        "rgb_bit_depth",
        "depth",
        "depth_format",
        "depth_unit",
        "depth_semantics",
        "normals",
        "normals_format",
        "normals_coordinate_system",
        "masks",
        "mask_format",
        "mask_bit_depth",
        "semantic_masks",
        "terminal_masks",
        "instance_masks",
        "alpha",
        "object_ids",
        "material_ids",
        "geometry_format",
        "generate_preview",
        "write_scene_file",
        "write_checksums",
    }
    value = reader.exact(raw, path, keys)
    output = OutputSettings(
        reader.string(value["output_root"], (*path, "output_root")),
        reader.boolean(value["rgb"], (*path, "rgb")),
        reader.string(value["rgb_format"], (*path, "rgb_format")),
        reader.integer(value["rgb_bit_depth"], (*path, "rgb_bit_depth")),
        reader.boolean(value["depth"], (*path, "depth")),
        reader.string(value["depth_format"], (*path, "depth_format")),
        reader.string(value["depth_unit"], (*path, "depth_unit")),
        reader.string(value["depth_semantics"], (*path, "depth_semantics")),
        reader.boolean(value["normals"], (*path, "normals")),
        reader.string(value["normals_format"], (*path, "normals_format")),
        reader.string(
            value["normals_coordinate_system"], (*path, "normals_coordinate_system")
        ),
        reader.boolean(value["masks"], (*path, "masks")),
        reader.string(value["mask_format"], (*path, "mask_format")),
        reader.integer(value["mask_bit_depth"], (*path, "mask_bit_depth")),
        reader.boolean(value["semantic_masks"], (*path, "semantic_masks")),
        reader.boolean(value["terminal_masks"], (*path, "terminal_masks")),
        reader.boolean(value["instance_masks"], (*path, "instance_masks")),
        reader.boolean(value["alpha"], (*path, "alpha")),
        reader.boolean(value["object_ids"], (*path, "object_ids")),
        reader.boolean(value["material_ids"], (*path, "material_ids")),
        reader.string(value["geometry_format"], (*path, "geometry_format")),
        reader.boolean(value["generate_preview"], (*path, "generate_preview")),
        reader.boolean(value["write_scene_file"], (*path, "write_scene_file")),
        reader.boolean(value["write_checksums"], (*path, "write_checksums")),
    )
    _safe_path(reader, output.output_root, (*path, "output_root"))
    if not output.rgb:
        reader.fail((*path, "rgb"), "RGB output is required", output.rgb)
    if (
        output.rgb_format not in {"png", "exr"}
        or (output.rgb_format == "png" and output.rgb_bit_depth not in {8, 16})
        or (output.rgb_format == "exr" and output.rgb_bit_depth not in {16, 32})
    ):
        reader.fail(path, "RGB format and bit depth are incompatible", raw)
    if not output.depth or output.depth_format not in {"exr", "npy", "npz"}:
        reader.fail(path, "metric depth output in exr, npy, or npz is required", raw)
    if output.depth_unit != "meters" or output.depth_semantics != "metric_ray_distance":
        reader.fail(
            path, "depth must use meters and metric_ray_distance semantics", raw
        )
    if output.normals and output.normals_format not in {"exr", "npy", "npz"}:
        reader.fail(path, "normal format is incompatible", raw)
    if output.normals_coordinate_system not in {"world", "camera"}:
        reader.fail(path, "normal coordinate system is incompatible", raw)
    if (
        not output.masks
        or output.mask_format != "png"
        or output.mask_bit_depth not in {16, 32}
    ):
        reader.fail(path, "lossless 16- or 32-bit PNG masks are required", raw)
    if output.geometry_format not in {"none", "glb", "ply"}:
        reader.fail(
            (*path, "geometry_format"),
            "unsupported geometry format",
            output.geometry_format,
        )
    return output


def _parse_edits(reader: _Reader, raw: object) -> tuple[EditSpecification, ...]:
    path = ("edits",)
    if not isinstance(raw, list):
        reader.fail(path, "must be an array", raw, validator="type")
    edits: list[EditSpecification] = []
    keys = {
        "edit_id",
        "operation",
        "target_role",
        "parameters",
        "applicability",
        "camera_reuse",
        "changed_region_path",
        "unchanged_region_path",
        "edit_seed",
    }
    parameter_keys = {
        EditOperation.CHANGE_REPEAT_COUNT: {"new_count"},
        EditOperation.REMOVE_REGION: {"region_role"},
        EditOperation.CHANGE_SPACING: {"spacing_m"},
        EditOperation.SCALE_TERMINAL: {"scale"},
        EditOperation.REPLACE_TERMINAL: {"replacement_category"},
        EditOperation.REMOVE_INSTANCES: {"indices"},
        EditOperation.EXTEND_REGION: {"distance_m"},
    }
    for index, item in enumerate(cast(list[object], raw)):
        item_path = (*path, index)
        value = reader.exact(item, item_path, keys)
        edit_id = reader.string(value["edit_id"], (*item_path, "edit_id"))
        if not EDIT_ID_RE.fullmatch(edit_id):
            reader.fail(
                (*item_path, "edit_id"),
                "must match edit-<lowercase-hyphenated-id>",
                edit_id,
            )
        operation = reader.enum(
            value["operation"], (*item_path, "operation"), EditOperation
        )
        parameters = reader.exact(
            value["parameters"], (*item_path, "parameters"), parameter_keys[operation]
        )
        _validate_edit_parameters(
            reader, operation, parameters, (*item_path, "parameters")
        )
        applicability = reader.string(
            value["applicability"], (*item_path, "applicability")
        )
        if applicability not in {"required", "optional"}:
            reader.fail(
                (*item_path, "applicability"),
                "must be required or optional",
                applicability,
            )
        reuse = reader.string(value["camera_reuse"], (*item_path, "camera_reuse"))
        if reuse not in {"reuse_all", "reuse_validation_test", "declared_exclusions"}:
            reader.fail(
                (*item_path, "camera_reuse"), "unsupported camera reuse policy", reuse
            )
        changed = reader.string(
            value["changed_region_path"], (*item_path, "changed_region_path")
        )
        unchanged = reader.string(
            value["unchanged_region_path"], (*item_path, "unchanged_region_path")
        )
        _safe_path(reader, changed, (*item_path, "changed_region_path"))
        _safe_path(reader, unchanged, (*item_path, "unchanged_region_path"))
        edits.append(
            EditSpecification(
                edit_id,
                operation,
                reader.string(value["target_role"], (*item_path, "target_role")),
                dict(parameters),
                applicability,
                reuse,
                changed,
                unchanged,
                reader.integer(
                    value["edit_seed"],
                    (*item_path, "edit_seed"),
                    minimum=0,
                    maximum=UINT32_MAX,
                ),
            )
        )
    ids = [edit.edit_id for edit in edits]
    if len(ids) != len(set(ids)):
        reader.fail(
            path, "edit identifiers must be unique", ids, validator="uniqueItems"
        )
    return tuple(edits)


def _validate_edit_parameters(
    reader: _Reader,
    operation: EditOperation,
    value: dict[str, Any],
    path: tuple[str | int, ...],
) -> None:
    if operation is EditOperation.CHANGE_REPEAT_COUNT:
        reader.integer(value["new_count"], (*path, "new_count"), minimum=1)
    elif operation is EditOperation.REMOVE_REGION:
        reader.string(value["region_role"], (*path, "region_role"))
    elif operation is EditOperation.CHANGE_SPACING:
        reader.number(
            value["spacing_m"], (*path, "spacing_m"), minimum=0, exclusive_minimum=True
        )
    elif operation is EditOperation.SCALE_TERMINAL:
        scale = reader.numbers(value["scale"], (*path, "scale"), 3)
        if any(component <= 0 for component in scale):
            reader.fail(
                (*path, "scale"), "all scale components must be positive", scale
            )
    elif operation is EditOperation.REPLACE_TERMINAL:
        reader.string(value["replacement_category"], (*path, "replacement_category"))
    elif operation is EditOperation.REMOVE_INSTANCES:
        indices = reader.integers(value["indices"], (*path, "indices"))
        if not indices:
            reader.fail((*path, "indices"), "must not be empty", indices)
    else:
        reader.number(
            value["distance_m"],
            (*path, "distance_m"),
            minimum=0,
            exclusive_minimum=True,
        )


def _parse_provenance(reader: _Reader, raw: object) -> SourceProvenance:
    path = ("provenance",)
    value = reader.exact(
        raw, path, {"source_config", "definitions", "authoring_tool", "created_at"}
    )
    source = reader.string(value["source_config"], (*path, "source_config"))
    _safe_path(reader, source, (*path, "source_config"))
    definitions = reader.strings(
        value["definitions"], (*path, "definitions"), unique=True
    )
    for index, definition in enumerate(definitions):
        _safe_path(reader, definition, (*path, "definitions", index))
        if not definition.startswith("definitions/"):
            reader.fail(
                (*path, "definitions", index),
                "must reference the definitions directory",
                definition,
            )
    return SourceProvenance(
        source,
        definitions,
        reader.string(value["authoring_tool"], (*path, "authoring_tool")),
        None
        if value["created_at"] is None
        else reader.string(value["created_at"], (*path, "created_at")),
    )


def _validate_semantics(reader: _Reader, config: SceneConfiguration) -> None:
    camera = config.camera
    if camera.near_m >= camera.far_m:
        reader.fail(("camera",), "near plane must be smaller than far plane")
    if any(
        low > high
        for low, high in zip(camera.position_min_m, camera.position_max_m, strict=True)
    ):
        reader.fail(
            ("camera", "position_bounds_m"), "each minimum must not exceed its maximum"
        )
    if (camera.image_width, camera.image_height) != (
        config.render.width,
        config.render.height,
    ):
        reader.fail(
            ("camera",),
            "camera image dimensions must match render dimensions",
        )
    expected_stream_seeds = {
        "camera": (camera.seed, config.derived_seeds["cameras"]),
        "appearance": (config.appearance.seed, config.derived_seeds["materials"]),
        "lighting": (config.lighting.seed, config.derived_seeds["lighting"]),
        "occlusion": (config.occlusion.seed, config.derived_seeds["occluders"]),
    }
    for name, (declared, expected) in expected_stream_seeds.items():
        if declared != expected:
            reader.fail(
                (name, "seed"),
                f"must equal the derived {name} seed stream",
                declared,
            )
    for name, bounds in (
        ("yaw", camera.yaw_degrees),
        ("pitch", camera.pitch_degrees),
        ("roll", camera.roll_degrees),
    ):
        if bounds[0] > bounds[1]:
            reader.fail(
                ("camera", "orientation_bounds_degrees", name),
                "minimum must not exceed maximum",
            )
    max_edge = max(
        config.render.width * config.render.resolution_percentage // 100,
        config.render.height * config.render.resolution_percentage // 100,
    )
    cap = PROFILE_MAX_EDGE[config.expected_resource_profile.value]
    if max_edge > cap and not (
        config.extended_mode
        and config.expected_resource_profile is ResourceProfile.EXTENDED
    ):
        reader.fail(
            ("render",),
            f"effective resolution edge {max_edge} exceeds "
            f"{config.expected_resource_profile.value} profile cap {cap}",
            max_edge,
        )
    if (
        config.extended_mode
        and config.expected_resource_profile is not ResourceProfile.EXTENDED
    ):
        reader.fail(("extended_mode",), "requires the extended resource profile", True)
    params = config.family_parameters
    missing = params.missing_instances
    policy = config.structure.missing_instance_policy
    if (policy is MissingInstancePolicy.NONE) != (not missing):
        reader.fail(
            ("family_parameters", "missing_instances"),
            "must be empty only when missing_instance_policy is none",
            missing,
        )
    if params.missing_pattern == "none" and missing:
        reader.fail(
            ("family_parameters", "missing_pattern"),
            "none conflicts with explicit missing instances",
        )
    if isinstance(params, FacadeParameters):
        count = params.rows * params.columns
        used_width = (
            (params.columns - 1) * params.horizontal_spacing_m
            + params.terminal_width_m
            + params.margins_m["left"]
            + params.margins_m["right"]
        )
        used_height = (
            (params.rows - 1) * params.vertical_spacing_m
            + params.terminal_height_m
            + params.margins_m["bottom"]
            + params.margins_m["top"]
            + params.floor_band_height_m
        )
        if (
            params.horizontal_spacing_m < params.terminal_width_m
            or params.vertical_spacing_m < params.terminal_height_m
        ):
            reader.fail(
                ("family_parameters",),
                "facade terminal dimensions overlap the declared spacing",
            )
        if used_width > params.wall_width_m or used_height > params.wall_height_m:
            reader.fail(
                ("family_parameters",),
                "facade grid and margins do not fit within the wall",
            )
        if params.recess_depth_m > params.wall_depth_m:
            reader.fail(
                ("family_parameters", "recess_depth_m"), "recess exceeds wall depth"
            )
    elif isinstance(params, CorridorParameters):
        sides = 1 if params.sidedness == "one_sided" else 2
        count = params.bay_count * sides
        used_length = (
            (params.bay_count - 1) * params.bay_spacing_m
            + params.terminal_width_m
            + 2 * params.end_margin_m
        )
        if params.bay_spacing_m < params.terminal_width_m:
            reader.fail(
                ("family_parameters",), "corridor terminals overlap the bay spacing"
            )
        if used_length > params.length_m or params.terminal_height_m > params.height_m:
            reader.fail(
                ("family_parameters",), "corridor bays do not fit within the enclosure"
            )
        if params.terminal_depth_m * 2 >= params.width_m:
            reader.fail(
                ("family_parameters", "terminal_depth_m"),
                "terminal depths consume the corridor width",
            )
    else:
        count = params.count * params.row_count
        diameter = (
            2 * params.terminal_radius_m
            if params.terminal_radius_m is not None
            else cast(float, params.terminal_width_m)
        )
        used_length = (params.count - 1) * params.spacing_m + diameter
        required_width = diameter + (params.spacing_m if params.row_count == 2 else 0)
        if params.spacing_m <= diameter:
            reader.fail(
                ("family_parameters", "spacing_m"), "colonnade terminals overlap"
            )
        if used_length > params.floor_length_m or required_width > params.floor_width_m:
            reader.fail(
                ("family_parameters",), "colonnade does not fit on the declared floor"
            )
        if params.plinth_height_m + params.capital_height_m >= params.terminal_height_m:
            reader.fail(
                ("family_parameters",), "plinth and capital consume the terminal height"
            )
        expected_rows = 2 if params.layout == "double_row" else 1
        if params.row_count != expected_rows:
            reader.fail(
                ("family_parameters", "row_count"), "does not match the selected layout"
            )
    if any(index >= count for index in missing):
        reader.fail(
            ("family_parameters", "missing_instances"),
            f"indices must be smaller than {count}",
            missing,
        )
    regime = config.appearance.regime
    if (
        regime is AppearanceRegime.ALTERNATING
        and config.appearance.alternate_period is None
    ):
        reader.fail(
            ("appearance", "alternate_period"), "alternating regime requires a period"
        )
    if regime is AppearanceRegime.GROUPED and config.appearance.group_size is None:
        reader.fail(
            ("appearance", "group_size"), "grouped regime requires a group size"
        )
    if regime is AppearanceRegime.IDENTICAL and (
        config.appearance.variation_amplitude != 0
        or len(config.appearance.palette) != 1
    ):
        reader.fail(
            ("appearance",), "identical regime requires one color and zero variation"
        )
    if config.occlusion.regime is OcclusionRegime.CLEAR and (
        config.occlusion.occluder_count != 0
        or config.occlusion.target_fraction != (0.0, 0.0)
    ):
        reader.fail(
            ("occlusion",),
            "clear regime requires zero occluders and zero target fraction",
        )
    if config.occlusion.regime is OcclusionRegime.CLEAR and (
        config.occlusion.occluder_categories
        or config.occlusion.qualitative_level != "none"
    ):
        reader.fail(
            ("occlusion",),
            "clear regime requires no occluder categories and qualitative level none",
        )
    if (
        config.occlusion.regime is OcclusionRegime.TRUNCATION
        and not config.occlusion.allow_camera_truncation
    ):
        reader.fail(("occlusion",), "truncation regime must allow camera truncation")
    if config.outputs.semantic_masks != config.annotations.emit_semantic_masks:
        reader.fail(
            ("outputs", "semantic_masks"),
            "must agree with semantic annotation intent",
        )
    if config.outputs.instance_masks != config.annotations.emit_instance_masks:
        reader.fail(
            ("outputs", "instance_masks"),
            "must agree with instance annotation intent",
        )
    if config.outputs.terminal_masks != config.annotations.emit_terminal_annotations:
        reader.fail(
            ("outputs", "terminal_masks"),
            "must agree with terminal annotation intent",
        )
    roles = {
        config.family.value,
        *config.structure.expected_repeat_groups,
        *config.structure.expected_canonical_terminal_categories,
    }
    for index, edit in enumerate(config.edits):
        if edit.target_role not in roles:
            reader.fail(
                ("edits", index, "target_role"),
                "must name a declared family, repeat group, or terminal category",
                edit.target_role,
            )
        if edit.edit_seed != config.derived_seeds["edits"]:
            reader.fail(
                ("edits", index, "edit_seed"),
                "must equal the derived edits stream seed",
                edit.edit_seed,
            )


def _safe_path(reader: _Reader, value: str, path: tuple[str | int, ...]) -> None:
    try:
        _validate_embedded_path(value, allow_absolute=False)
    except ValueError as error:
        reader.fail(path, str(error), value, validator="path")


def _validate_embedded_path(value: str, *, allow_absolute: bool) -> None:
    if "\\" in value:
        raise ValueError("paths must use POSIX separators")
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("path is empty, ambiguous, or traverses a parent")
    if not allow_absolute and path.is_absolute():
        raise ValueError("path must be repository-relative")
    if re.match(r"^[A-Za-z]:", value):
        raise ValueError("drive-qualified paths are not permitted")


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and resolve scene configuration"
    )
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("path")
    validate.add_argument("--json", action="store_true")
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--config", dest="path", required=True)
    resolve.add_argument("--output", required=True)
    resolve.add_argument("--canonical-output")
    resolve.add_argument("--digest-output")
    resolve.add_argument("--repository-root")
    resolve.add_argument("--overwrite", action="store_true")
    resolve.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the scene validate/resolve command implementation."""

    options = _parser().parse_args(arguments)
    try:
        source = Path(options.path)
        config = resolve_scene_configuration(
            source,
            repository_root=(
                Path(options.repository_root)
                if getattr(options, "repository_root", None)
                else None
            ),
        )
        if options.action == "validate":
            result: dict[str, Any] = {
                "valid": True,
                "scene_id": config.scene_id,
                "family": config.family.value,
                "scientific_digest": config.scientific_digest,
                "derived_seeds": dict(config.derived_seeds),
            }
        else:
            result = {
                "valid": True,
                **write_resolved_scene_configuration(
                    config,
                    Path(options.output),
                    canonical_output=(
                        Path(options.canonical_output)
                        if options.canonical_output
                        else None
                    ),
                    digest_output=(
                        Path(options.digest_output) if options.digest_output else None
                    ),
                    overwrite=options.overwrite,
                ),
                "family": config.family.value,
                "derived_seeds": dict(config.derived_seeds),
            }
    except (
        DocumentLoadError,
        SchemaValidationError,
        ValueError,
        FileExistsError,
    ) as error:
        if isinstance(error, SchemaValidationError):
            payload = error.to_dict()
        elif isinstance(error, DocumentLoadError):
            payload = {"valid": False, "errors": [error.issue.to_dict()]}
        else:
            payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3
    if options.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"valid scene configuration: {config.scene_id}; "
            f"family={config.family.value}; "
            f"scientific_digest={config.scientific_digest}; "
            f"derived_seeds={json.dumps(dict(config.derived_seeds), sort_keys=True)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
