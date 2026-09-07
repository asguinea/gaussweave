"""Deterministic camera trajectory, split, validation, and artifact support."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.data.camera_math import (
    Matrix4,
    Vector3,
    angular_distance_degrees,
    bounding_box,
    euclidean_distance,
    focal_from_horizontal_fov,
    focal_from_lens_mm,
    look_at_world_from_camera,
    pose_axes,
    project_point,
    validate_rigid_pose,
)
from gaussweave.data.scene_config import (
    ColonnadeParameters,
    CorridorParameters,
    FacadeParameters,
    SceneConfiguration,
    SceneFamily,
    load_scene_configuration,
)
from gaussweave.results.artifacts import (
    ArtifactDeclaration,
    VerificationState,
    build_inventory,
    hash_file,
    verify_inventory,
    write_inventory,
)

CAMERA_SCHEMA_VERSION = "1.0.0"
CAMERA_POLICY_VERSION = "trajectory-v1"
CAMERA_FORMAT = "gaussweave_camera_json_v1"
CAMERA_MARKER = ".gaussweave-camera-generation"
CAMERA_INVENTORY = "artifact-inventory.json"
CAMERA_ID_RE = re.compile(r"^cam-(train|val|test)-[0-9]{4,}$")
Split = Literal["train", "validation", "test"]


class CameraValidationError(ValueError):
    """Camera metadata or trajectory constraints are invalid."""


@dataclass(frozen=True)
class CameraRecord:
    camera_id: str
    split: Split
    width: int
    height: int
    model: str
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: Matrix4
    near_m: float
    far_m: float
    trajectory_parameter: float
    trajectory_family: str
    position_m: Vector3
    right: Vector3
    down: Vector3
    forward: Vector3
    target_m: Vector3
    coverage: dict[str, Any]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        expected_token = "val" if self.split == "validation" else self.split
        if not CAMERA_ID_RE.fullmatch(self.camera_id) or not self.camera_id.startswith(
            f"cam-{expected_token}-"
        ):
            raise CameraValidationError("camera ID does not match its split")
        if self.width <= 0 or self.height <= 0 or self.model != "pinhole":
            raise CameraValidationError("camera image model is invalid")
        values = (self.fx, self.fy, self.cx, self.cy, self.near_m, self.far_m)
        if not all(math.isfinite(value) for value in values):
            raise CameraValidationError("camera intrinsics and clipping must be finite")
        if (
            self.fx <= 0
            or self.fy <= 0
            or self.near_m <= 0
            or self.far_m <= self.near_m
        ):
            raise CameraValidationError("camera intrinsics or clipping are invalid")
        validate_rigid_pose(self.world_from_camera)
        axes = pose_axes(self.world_from_camera)
        expected = (self.right, self.down, self.forward, self.position_m)
        if any(
            abs(a - b) > 1e-10
            for derived, declared in zip(axes, expected, strict=True)
            for a, b in zip(derived, declared, strict=True)
        ):
            raise CameraValidationError("declared axes/position differ from pose")
        if not 0.0 <= self.trajectory_parameter <= 1.0:
            raise CameraValidationError("trajectory parameter must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["world_from_camera"] = list(self.world_from_camera)
        for key in ("position_m", "right", "down", "forward", "target_m", "warnings"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CameraRecord:
        try:
            matrix = tuple(float(item) for item in value["world_from_camera"])
            if len(matrix) != 16:
                raise ValueError
            vectors = {
                key: _vector(value[key])
                for key in ("position_m", "right", "down", "forward", "target_m")
            }
            split = str(value["split"])
            if split not in {"train", "validation", "test"}:
                raise ValueError
            return cls(
                camera_id=str(value["camera_id"]),
                split=cast(Split, split),
                width=int(value["width"]),
                height=int(value["height"]),
                model=str(value["model"]),
                fx=float(value["fx"]),
                fy=float(value["fy"]),
                cx=float(value["cx"]),
                cy=float(value["cy"]),
                world_from_camera=matrix,
                near_m=float(value["near_m"]),
                far_m=float(value["far_m"]),
                trajectory_parameter=float(value["trajectory_parameter"]),
                trajectory_family=str(value["trajectory_family"]),
                coverage=dict(value["coverage"]),
                warnings=tuple(str(item) for item in value.get("warnings", [])),
                provenance=dict(value["provenance"]),
                **vectors,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CameraValidationError("invalid camera record") from error


@dataclass(frozen=True)
class CameraCollection:
    camera_schema_version: str
    format: str
    scene_id: str
    family: str
    policy_version: str
    configuration_digest: str
    camera_seed: int
    records: tuple[CameraRecord, ...]
    splits: dict[str, tuple[str, ...]]
    counts: dict[str, int]
    trajectory: dict[str, Any]
    separation: dict[str, Any]
    coverage: dict[str, Any]
    diagnostics: dict[str, Any]
    scientific_digest: str

    def scientific_projection(self) -> dict[str, Any]:
        return {
            "camera_schema_version": self.camera_schema_version,
            "format": self.format,
            "scene_id": self.scene_id,
            "family": self.family,
            "policy_version": self.policy_version,
            "configuration_digest": self.configuration_digest,
            "camera_seed": self.camera_seed,
            "records": [record.to_dict() for record in self.records],
            "splits": {name: list(ids) for name, ids in sorted(self.splits.items())},
            "counts": dict(sorted(self.counts.items())),
            "trajectory": self.trajectory,
            "separation": self.separation,
            "coverage": self.coverage,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.scientific_projection()
        value["diagnostics"] = self.diagnostics
        value["scientific_digest"] = self.scientific_digest
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CameraCollection:
        try:
            records = tuple(CameraRecord.from_dict(item) for item in value["records"])
            splits = {
                name: tuple(str(item) for item in value["splits"][name])
                for name in ("train", "validation", "test")
            }
            collection = cls(
                camera_schema_version=str(value["camera_schema_version"]),
                format=str(value["format"]),
                scene_id=str(value["scene_id"]),
                family=str(value["family"]),
                policy_version=str(value["policy_version"]),
                configuration_digest=str(value["configuration_digest"]),
                camera_seed=int(value["camera_seed"]),
                records=records,
                splits=splits,
                counts={name: int(value["counts"][name]) for name in splits},
                trajectory=dict(value["trajectory"]),
                separation=dict(value["separation"]),
                coverage=dict(value["coverage"]),
                diagnostics=dict(value.get("diagnostics", {})),
                scientific_digest=str(value["scientific_digest"]),
            )
            validate_camera_collection(collection)
            return collection
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CameraValidationError):
                raise
            raise CameraValidationError("invalid camera collection") from error


@dataclass(frozen=True)
class CameraGenerationResult:
    valid: bool
    root: str
    scene_id: str
    family: str
    camera_count: int
    counts: dict[str, int]
    camera_seed: int
    scientific_digest: str
    inventory_state: str
    artifact_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_camera_collection(
    config: SceneConfiguration,
    *,
    camera_seed: int | None = None,
    train_count: int | None = None,
    validation_count: int | None = None,
    test_count: int | None = None,
) -> CameraCollection:
    """Generate one ordered, deterministic, family-aware camera collection."""

    policy = config.camera
    seed = policy.seed if camera_seed is None else camera_seed
    if not 0 <= seed <= 2**32 - 1:
        raise CameraValidationError("camera seed must be an unsigned 32-bit integer")
    counts = {
        "train": policy.train_count if train_count is None else train_count,
        "validation": (
            policy.validation_count if validation_count is None else validation_count
        ),
        "test": policy.test_count if test_count is None else test_count,
    }
    if any(value <= 0 for value in counts.values()):
        raise CameraValidationError("all camera split counts must be positive")
    total = sum(counts.values())
    split_by_slot = _split_slots(total, counts)
    fx, fy, cx, cy = _intrinsics(config)
    records: list[CameraRecord] = []
    split_indices = {"train": 0, "validation": 0, "test": 0}
    proxy_descriptions: set[str] = set()
    for slot in range(total):
        split = split_by_slot[slot]
        split_indices[split] += 1
        token = "val" if split == "validation" else split
        camera_id = f"cam-{token}-{split_indices[split]:04d}"
        t = (slot + 0.5) / total
        rng = random.Random(f"{CAMERA_POLICY_VERSION}:{seed}:{slot}")
        position, target, proxy, metadata = _family_pose(config, t, rng)
        pose = look_at_world_from_camera(position, target)
        right, down, forward, derived_position = pose_axes(pose)
        coverage = _coverage(
            proxy,
            pose,
            width=policy.image_width,
            height=policy.image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            target=target,
            near_m=policy.near_m,
            far_m=policy.far_m,
        )
        proxy_descriptions.add(str(metadata["coverage_proxy"]))
        warnings: tuple[str, ...] = ()
        if float(coverage["structure_coverage"]) < policy.minimum_structure_coverage:
            warnings = ("approximate structural coverage is below policy threshold",)
        records.append(
            CameraRecord(
                camera_id,
                split,
                policy.image_width,
                policy.image_height,
                "pinhole",
                fx,
                fy,
                cx,
                cy,
                pose,
                policy.near_m,
                policy.far_m,
                t,
                policy.trajectory,
                derived_position,
                right,
                down,
                forward,
                target,
                coverage,
                warnings,
                {
                    "policy_version": CAMERA_POLICY_VERSION,
                    "camera_seed": seed,
                    "slot": slot,
                    **metadata,
                },
            )
        )
    separation = _separation_diagnostics(
        tuple(records),
        policy.minimum_separation_m,
        policy.minimum_angular_separation_degrees,
    )
    coverage_values = [
        float(record.coverage["structure_coverage"]) for record in records
    ]
    coverage = {
        "method": "cpu_proxy_projection_v1",
        "exact_occlusion_claim": False,
        "minimum_required": policy.minimum_structure_coverage,
        "minimum_observed": min(coverage_values),
        "mean_observed": sum(coverage_values) / len(coverage_values),
        "all_pass": all(
            value >= policy.minimum_structure_coverage for value in coverage_values
        ),
        "proxy_descriptions": sorted(proxy_descriptions),
    }
    splits = {
        name: tuple(record.camera_id for record in records if record.split == name)
        for name in ("train", "validation", "test")
    }
    base = CameraCollection(
        CAMERA_SCHEMA_VERSION,
        CAMERA_FORMAT,
        config.scene_id,
        config.family.value,
        CAMERA_POLICY_VERSION,
        config.scientific_digest,
        seed,
        tuple(records),
        splits,
        counts,
        {
            "family": config.family.value,
            "type": policy.trajectory,
            "parameterization": "global_midpoint_slots",
            "draw_order": [
                "height",
                "lateral_or_radius",
                "target_offset",
                "family_auxiliary",
            ],
            "total_slots": total,
        },
        separation,
        coverage,
        {
            "warning_count": sum(bool(record.warnings) for record in records),
            "record_order": "global trajectory slot",
            "split_policy": "deterministic interleaved holdout v1",
        },
        "",
    )
    result = replace(
        base, scientific_digest=content_digest(base.scientific_projection())
    )
    validate_camera_collection(result)
    return result


def validate_camera_collection(
    collection: CameraCollection,
    *,
    minimum_separation_m: float | None = None,
    minimum_angular_separation_degrees: float | None = None,
    minimum_structure_coverage: float | None = None,
) -> dict[str, Any]:
    if collection.camera_schema_version != CAMERA_SCHEMA_VERSION:
        raise CameraValidationError("unsupported camera schema version")
    if collection.format != CAMERA_FORMAT or not collection.records:
        raise CameraValidationError("unsupported or empty camera collection")
    ids = [record.camera_id for record in collection.records]
    if len(ids) != len(set(ids)):
        raise CameraValidationError("camera IDs must be unique")
    expected_ids = {
        name: tuple(
            record.camera_id for record in collection.records if record.split == name
        )
        for name in ("train", "validation", "test")
    }
    if collection.splits != expected_ids:
        raise CameraValidationError("split lists differ from ordered records")
    if collection.counts != {
        name: len(values) for name, values in expected_ids.items()
    }:
        raise CameraValidationError("split counts differ from split lists")
    digest = content_digest(collection.scientific_projection())
    if collection.scientific_digest != digest:
        raise CameraValidationError("camera scientific digest mismatch")
    position_limit = (
        float(collection.separation["required_position_m"])
        if minimum_separation_m is None
        else minimum_separation_m
    )
    angular_limit = (
        float(collection.separation["required_angle_degrees"])
        if minimum_angular_separation_degrees is None
        else minimum_angular_separation_degrees
    )
    measured = _separation_diagnostics(
        collection.records, position_limit, angular_limit
    )
    if measured["violations"]:
        first = measured["violations"][0]
        raise CameraValidationError(
            "split leakage: "
            f"{first['left']} vs {first['right']} "
            f"({first['position_m']:.9f} m, {first['angle_degrees']:.9f} deg)"
        )
    coverage_limit = (
        float(collection.coverage["minimum_required"])
        if minimum_structure_coverage is None
        else minimum_structure_coverage
    )
    failed = [
        record.camera_id
        for record in collection.records
        if float(record.coverage["structure_coverage"]) < coverage_limit
    ]
    if failed:
        raise CameraValidationError(
            "camera coverage below threshold: " + ", ".join(failed)
        )
    return {
        "valid": True,
        "scene_id": collection.scene_id,
        "camera_count": len(collection.records),
        "counts": collection.counts,
        "scientific_digest": collection.scientific_digest,
        "minimum_cross_split_position_m": measured["minimum_position_m"],
        "minimum_cross_split_angle_degrees": measured["minimum_angle_degrees"],
        "minimum_structure_coverage": min(
            float(record.coverage["structure_coverage"])
            for record in collection.records
        ),
    }


def write_camera_artifacts(
    collection: CameraCollection, root: Path, *, overwrite: bool = False
) -> CameraGenerationResult:
    """Write canonical camera/split diagnostics and a strict inventory."""

    output = _validate_output_root(root)
    marker = output / CAMERA_MARKER
    managed = (
        output / "cameras",
        output / CAMERA_INVENTORY,
        marker,
    )
    if (
        output.exists()
        and any(output.iterdir())
        and not any(path.exists() for path in managed)
    ):
        raise CameraValidationError("refusing to use an unrelated nonempty root")
    if any(path.exists() for path in managed):
        if not overwrite:
            raise CameraValidationError("camera output already exists; use --overwrite")
        if (
            marker.is_symlink()
            or not marker.is_file()
            or f"scene_id={collection.scene_id}"
            not in marker.read_text(encoding="utf-8")
        ):
            raise CameraValidationError("overwrite requires a matching camera marker")
        camera_root = output / "cameras"
        if camera_root.is_symlink():
            raise CameraValidationError(
                "camera artifact directory may not be a symlink"
            )
        if camera_root.exists():
            shutil.rmtree(camera_root)
        for path in (output / CAMERA_INVENTORY, marker):
            if path.exists():
                if path.is_symlink():
                    raise CameraValidationError("managed output may not be a symlink")
                path.unlink()
    camera_dir = output / "cameras"
    camera_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"GaussWeave camera generation root\nscene_id={collection.scene_id}\n",
        encoding="utf-8",
    )
    _write_json(camera_dir / "cameras.json", collection.to_dict())
    for split, filename in (
        ("train", "train.txt"),
        ("validation", "validation.txt"),
        ("test", "test.txt"),
    ):
        (camera_dir / filename).write_text(
            "".join(f"{camera_id}\n" for camera_id in collection.splits[split]),
            encoding="utf-8",
            newline="\n",
        )
    diagnostics = {
        "diagnostics_schema_version": CAMERA_SCHEMA_VERSION,
        "scene_id": collection.scene_id,
        "scientific_digest": collection.scientific_digest,
        "separation": collection.separation,
        "coverage": collection.coverage,
        "diagnostics": collection.diagnostics,
    }
    _write_json(camera_dir / "diagnostics.json", diagnostics)
    checksum_members = (
        "cameras/cameras.json",
        "cameras/train.txt",
        "cameras/validation.txt",
        "cameras/test.txt",
        "cameras/diagnostics.json",
    )
    checksum_lines = [
        f"{hash_file(output / path).digest.removeprefix('sha256:')}  {path}\n"
        for path in checksum_members
    ]
    (camera_dir / "checksums.sha256").write_text(
        "".join(checksum_lines), encoding="utf-8", newline="\n"
    )
    paths = (*checksum_members, "cameras/checksums.sha256")
    declarations = tuple(
        ArtifactDeclaration(
            artifact_id=f"camera-{index:02d}",
            artifact_type="other",
            path=path,
            description=f"deterministic camera artifact: {path}",
        )
        for index, path in enumerate(paths, start=1)
    )
    inventory = build_inventory(
        output,
        declarations,
        inventory_id=f"inventory-cameras-{collection.scene_id}",
        excluded_paths=(CAMERA_MARKER, CAMERA_INVENTORY),
    )
    write_inventory(inventory, output / CAMERA_INVENTORY, overwrite=True)
    verification = verify_inventory(inventory, output, strict=True)
    if verification.state is not VerificationState.VALID:
        raise CameraValidationError("camera artifact inventory failed verification")
    return CameraGenerationResult(
        True,
        output.as_posix(),
        collection.scene_id,
        collection.family,
        len(collection.records),
        collection.counts,
        collection.camera_seed,
        collection.scientific_digest,
        verification.state.value,
        inventory.artifact_count,
    )


def load_camera_collection(path: Path) -> CameraCollection:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CameraValidationError(
            f"could not read camera collection: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CameraValidationError("camera collection must be a JSON object")
    return CameraCollection.from_dict(value)


def _family_pose(
    config: SceneConfiguration, t: float, rng: random.Random
) -> tuple[Vector3, Vector3, tuple[Vector3, ...], dict[str, Any]]:
    parameters = config.family_parameters
    policy = config.camera
    height_jitter = rng.uniform(-0.04, 0.04)
    lateral_jitter = rng.uniform(-0.04, 0.04)
    target_jitter = rng.uniform(-0.035, 0.035)
    auxiliary_jitter = rng.uniform(-0.02, 0.02)
    if isinstance(parameters, FacadeParameters):
        azimuth = _lerp(*_range(policy.family_intent["azimuth_degrees"]), t)
        radius = _lerp(
            *_range(policy.family_intent["arc_radius_m"]), 0.5 + auxiliary_jitter
        )
        height = _lerp(*_range(policy.family_intent["height_m"]), 0.45 + height_jitter)
        angle = math.radians(azimuth)
        position = (radius * math.sin(angle), -radius * math.cos(angle), height)
        target = (
            target_jitter * parameters.wall_width_m,
            0.0,
            parameters.wall_height_m * 0.48,
        )
        proxy = _box_face_points(
            center=(target[0], 0.0, target[2]),
            width=min(parameters.wall_width_m * 0.55, radius * 0.7),
            height=min(parameters.wall_height_m * 0.55, radius * 0.5),
        )
        label = "facade central repeated-region front-face proxy"
    elif isinstance(parameters, CorridorParameters):
        start, end = _range(policy.family_intent["longitudinal_range_m"])
        x = _lerp(start, end, t)
        lateral = _lerp(
            *_range(policy.family_intent["lateral_offset_m"]), 0.5 + lateral_jitter
        )
        height = _lerp(*_range(policy.family_intent["height_m"]), 0.5 + height_jitter)
        forward_distance = min(4.0, max(1.5, end - x + 0.8))
        direction = 1.0 if x < (start + end) / 2.0 else -1.0
        target_x = min(end, max(start, x + direction * forward_distance))
        wall_y = (parameters.width_m * 0.32) * (1.0 if int(t * 8) % 2 else -1.0)
        position = (x, lateral, height)
        target = (target_x, wall_y + target_jitter, parameters.terminal_height_m * 0.55)
        proxy = _box_face_points(
            center=target,
            width=min(parameters.bay_spacing_m * 0.6, 1.2),
            height=min(parameters.height_m * 0.45, 1.35),
            plane="yz",
        )
        label = "corridor nearby bay/side-wall repeated-region proxy"
    elif isinstance(parameters, ColonnadeParameters):
        radius_range = _range(policy.family_intent["radius_m"])
        azimuth_range = _range(policy.family_intent["azimuth_degrees"])
        radius = _lerp(*radius_range, 0.5 + auxiliary_jitter)
        azimuth = math.radians(_lerp(*azimuth_range, t))
        height = _lerp(*_range(policy.family_intent["height_m"]), 0.5 + height_jitter)
        position = (
            radius * math.sin(azimuth),
            -radius * math.cos(azimuth),
            height,
        )
        target = (
            target_jitter * parameters.floor_length_m,
            lateral_jitter * parameters.floor_width_m,
            parameters.terminal_height_m * 0.48,
        )
        proxy = _box_face_points(
            center=target,
            width=min(parameters.floor_length_m * 0.3, radius * 0.7),
            height=min(parameters.terminal_height_m * 0.75, radius * 0.5),
        )
        label = "colonnade central repeated-column region proxy"
    else:
        raise CameraValidationError(
            f"unsupported family parameters: {type(parameters)}"
        )
    return position, target, proxy, {"coverage_proxy": label}


def _box_face_points(
    *,
    center: Vector3,
    width: float,
    height: float,
    plane: str = "xz",
) -> tuple[Vector3, ...]:
    points: list[Vector3] = []
    for horizontal in (-0.5, 0.0, 0.5):
        for vertical in (-0.5, 0.0, 0.5):
            if plane == "xz":
                points.append(
                    (
                        center[0] + horizontal * width,
                        center[1],
                        center[2] + vertical * height,
                    )
                )
            else:
                points.append(
                    (
                        center[0],
                        center[1] + horizontal * width,
                        center[2] + vertical * height,
                    )
                )
    return tuple(points)


def _coverage(
    proxy: tuple[Vector3, ...],
    pose: Matrix4,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target: Vector3,
    near_m: float,
    far_m: float,
) -> dict[str, Any]:
    projected = [
        project_point(point, pose, fx=fx, fy=fy, cx=cx, cy=cy) for point in proxy
    ]
    in_front = [item for item in projected if near_m <= item[2] <= far_m]
    inside = [
        item for item in in_front if 0.0 <= item[0] < width and 0.0 <= item[1] < height
    ]
    finite_pixels = [(item[0], item[1]) for item in in_front]
    area = 0.0
    truncation = True
    if finite_pixels:
        left, top, right, bottom = bounding_box(finite_pixels)
        clipped_width = max(0.0, min(float(width), right) - max(0.0, left))
        clipped_height = max(0.0, min(float(height), bottom) - max(0.0, top))
        area = clipped_width * clipped_height / (width * height)
        truncation = left < 0 or top < 0 or right >= width or bottom >= height
    center = project_point(target, pose, fx=fx, fy=fy, cx=cx, cy=cy)
    _, _, _, position = pose_axes(pose)
    return {
        "method": "cpu_proxy_projection_v1",
        "exact_occlusion_claim": False,
        "proxy_point_count": len(proxy),
        "in_front_fraction": len(in_front) / len(proxy),
        "in_frame_fraction": len(inside) / len(proxy),
        "structure_coverage": len(inside) / len(proxy),
        "projected_area_fraction": area,
        "target_center_visible": (
            near_m <= center[2] <= far_m
            and 0.0 <= center[0] < width
            and 0.0 <= center[1] < height
        ),
        "proxy_truncated": truncation,
        "target_distance_m": euclidean_distance(position, target),
    }


def _split_slots(total: int, counts: dict[str, int]) -> dict[int, Split]:
    remaining = dict(counts)
    result: dict[int, Split] = {}
    # Balanced deficit scheduling interleaves all splits over the full path.
    order: tuple[Split, ...] = ("train", "validation", "test")
    allocated = {name: 0 for name in order}
    for slot in range(total):
        candidates = [name for name in order if remaining[name] > 0]
        split = max(
            candidates,
            key=lambda name: (
                counts[name] * (slot + 1) / total - allocated[name],
                -order.index(name),
            ),
        )
        result[slot] = split
        allocated[split] += 1
        remaining[split] -= 1
    return result


def _separation_diagnostics(
    records: tuple[CameraRecord, ...],
    position_limit: float,
    angular_limit: float,
) -> dict[str, Any]:
    comparisons: list[tuple[CameraRecord, CameraRecord]] = []
    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    test = [record for record in records if record.split == "test"]
    comparisons.extend((left, right) for left in validation for right in train)
    comparisons.extend(
        (left, right) for left in test for right in (*train, *validation)
    )
    values: list[tuple[float, float, str, str]] = []
    violations: list[dict[str, Any]] = []
    for left, right in comparisons:
        distance = euclidean_distance(left.position_m, right.position_m)
        angle = angular_distance_degrees(left.forward, right.forward)
        values.append((distance, angle, left.camera_id, right.camera_id))
        if distance <= 1e-12 and angle <= 1e-10:
            reason = "exact_duplicate"
        elif distance < position_limit and angle < angular_limit:
            reason = "position_and_orientation_too_close"
        else:
            continue
        violations.append(
            {
                "left": left.camera_id,
                "right": right.camera_id,
                "position_m": distance,
                "angle_degrees": angle,
                "reason": reason,
            }
        )
    return {
        "policy": (
            "exact duplicates fail; cross-split views fail when both position "
            "and orientation are below their independent minima"
        ),
        "required_position_m": position_limit,
        "required_angle_degrees": angular_limit,
        "comparison_count": len(values),
        "minimum_position_m": min((item[0] for item in values), default=None),
        "minimum_angle_degrees": min((item[1] for item in values), default=None),
        "violations": violations,
        "passes": not violations,
    }


def _intrinsics(config: SceneConfiguration) -> tuple[float, float, float, float]:
    policy = config.camera
    if policy.focal_mode == "focal_length_mm":
        if policy.focal_length_mm is None:
            raise CameraValidationError("focal length mode lacks focal_length_mm")
        fx = focal_from_lens_mm(policy.image_width, policy.focal_length_mm)
    elif policy.focal_mode == "horizontal_fov_degrees":
        if policy.horizontal_fov_degrees is None:
            raise CameraValidationError("FOV mode lacks horizontal_fov_degrees")
        fx = focal_from_horizontal_fov(
            policy.image_width, policy.horizontal_fov_degrees
        )
    else:
        raise CameraValidationError(f"unsupported focal mode: {policy.focal_mode}")
    fy = fx
    if policy.principal_point == "centered":
        cx = policy.image_width / 2.0
        cy = policy.image_height / 2.0
    elif policy.principal_point_px is not None:
        cx, cy = policy.principal_point_px
    else:
        raise CameraValidationError("explicit principal point is missing")
    return fx, fy, cx, cy


def _range(value: Any) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise CameraValidationError("family camera range must have two values")
    return float(value[0]), float(value[1])


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * max(0.0, min(1.0, t))


def _vector(value: Any) -> Vector3:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError
    return cast(Vector3, tuple(float(item) for item in value))


def _validate_output_root(value: Path) -> Path:
    if ".." in value.parts:
        raise CameraValidationError("camera output root may not contain traversal")
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise CameraValidationError("camera output root may not be a symlink")
    candidate = expanded if expanded.is_absolute() else Path.cwd() / expanded
    for parent in (candidate, *candidate.parents):
        if parent.exists() and parent.is_symlink():
            raise CameraValidationError("camera output root may not cross a symlink")
    resolved = expanded.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise CameraValidationError("camera output root is too broad")
    return resolved


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate camera artifacts"
    )
    subcommands = parser.add_subparsers(dest="action", required=True)
    generate = subcommands.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--root", required=True)
    generate.add_argument("--family", choices=tuple(item.value for item in SceneFamily))
    generate.add_argument("--camera-seed", type=int)
    generate.add_argument("--train-count", type=int)
    generate.add_argument("--validation-count", type=int)
    generate.add_argument("--test-count", type=int)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--json", action="store_true")
    validate = subcommands.add_parser("validate")
    validate.add_argument("path")
    validate.add_argument("--minimum-separation-m", type=float)
    validate.add_argument("--minimum-angular-separation-degrees", type=float)
    validate.add_argument("--minimum-structure-coverage", type=float)
    validate.add_argument("--json", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        if options.action == "generate":
            config = load_scene_configuration(Path(options.config))
            if options.family is not None and options.family != config.family.value:
                raise CameraValidationError("--family differs from scene configuration")
            collection = generate_camera_collection(
                config,
                camera_seed=options.camera_seed,
                train_count=options.train_count,
                validation_count=options.validation_count,
                test_count=options.test_count,
            )
            result: Any = write_camera_artifacts(
                collection, Path(options.root), overwrite=options.overwrite
            ).to_dict()
        else:
            result = validate_camera_collection(
                load_camera_collection(Path(options.path)),
                minimum_separation_m=options.minimum_separation_m,
                minimum_angular_separation_degrees=(
                    options.minimum_angular_separation_degrees
                ),
                minimum_structure_coverage=options.minimum_structure_coverage,
            )
    except CameraValidationError as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 3
    if options.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"Camera {options.action} valid: scene={result['scene_id']}; "
            f"cameras={result['camera_count']}; digest={result['scientific_digest']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
