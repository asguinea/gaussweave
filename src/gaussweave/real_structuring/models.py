"""Typed CPU-only contracts for real-region structural refitting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import content_digest, pretty_json_bytes

PILOT_VERSION = "gw-truck-structured-refit-v1"
INSTANCE_IDS = ("panel-rear", "panel-middle", "panel-front")
CANONICAL_INSTANCE_ID = "panel-middle"
SOURCE_COUNT = 2_541_226
CORE_COUNTS = {
    "panel-rear": 13_614,
    "panel-middle": 9_533,
    "panel-front": 16_644,
}
RESTRICTED_ARTIFACT_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".pgm",
        ".pfm",
        ".ply",
        ".f32",
        ".u8",
        ".u32",
        ".i8",
        ".pt",
        ".pth",
        ".ckpt",
    }
)


class StructuringError(RuntimeError):
    """A real-region artifact or operation violates the scientific contract."""


class OwnershipStatus(IntEnum):
    """Exactly-one ownership status for each source Gaussian."""

    EXPLICIT_BACKGROUND = 0
    PANEL_REAR_CORE = 1
    PANEL_MIDDLE_CORE = 2
    PANEL_FRONT_CORE = 3
    BOUNDARY_GUARD = 4
    EXCLUDED_INVALID = 5


def detect_restricted_artifact_paths(paths: list[str]) -> list[str]:
    """Return repository paths that contain generated or licensed payloads."""

    restricted_roots = ("datasets/", "checkpoints/", "runs/", "results/")
    return [
        path
        for path in paths
        if path.replace("\\", "/").lower().endswith(tuple(RESTRICTED_ARTIFACT_SUFFIXES))
        or path.replace("\\", "/").lower().startswith(restricted_roots)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Write deterministic JSON without leaving a partial target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(pretty_json_bytes(cast(Any, value)))
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructuringError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise StructuringError(f"JSON artifact must contain an object: {path}")
    return cast(dict[str, Any], value)


def _vector(values: tuple[float, ...], size: int, name: str) -> None:
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise StructuringError(f"{name} must contain {size} finite values")


def _determinant(axes: tuple[tuple[float, float, float], ...]) -> float:
    a, b, c = axes
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
    )


@dataclass(frozen=True)
class PanelFrame:
    """One deterministic, right-handed panel-local coordinate frame."""

    instance_id: str
    origin_world: tuple[float, float, float]
    axes_world: tuple[tuple[float, float, float], ...]
    local_lower: tuple[float, float, float]
    local_upper: tuple[float, float, float]
    support_plane_normal_world: tuple[float, float, float]
    support_plane_offset: float
    source_annotation: str
    confidence: str
    method: str

    def __post_init__(self) -> None:
        if self.instance_id not in INSTANCE_IDS:
            raise StructuringError("unknown panel instance")
        _vector(self.origin_world, 3, "frame origin")
        _vector(self.local_lower, 3, "local lower bound")
        _vector(self.local_upper, 3, "local upper bound")
        _vector(self.support_plane_normal_world, 3, "support-plane normal")
        if any(a >= b for a, b in zip(self.local_lower, self.local_upper, strict=True)):
            raise StructuringError("frame bounds must have positive extent")
        if len(self.axes_world) != 3:
            raise StructuringError("frame must have three axes")
        for axis in self.axes_world:
            _vector(axis, 3, "frame axis")
        for left in range(3):
            for right in range(3):
                dot = sum(
                    self.axes_world[left][i] * self.axes_world[right][i]
                    for i in range(3)
                )
                expected = 1.0 if left == right else 0.0
                if abs(dot - expected) > 1e-5:
                    raise StructuringError("frame axes must be orthonormal")
        if abs(_determinant(self.axes_world) - 1.0) > 1e-5:
            raise StructuringError("frame axes must be right-handed")
        if self.confidence not in {"high", "medium", "low"}:
            raise StructuringError("invalid frame confidence")

    @property
    def width(self) -> float:
        return self.local_upper[0] - self.local_lower[0]

    @property
    def height(self) -> float:
        return self.local_upper[1] - self.local_lower[1]

    @property
    def depth(self) -> float:
        return self.local_upper[2] - self.local_lower[2]

    @property
    def world_from_local(self) -> tuple[float, ...]:
        x, y, z = self.axes_world
        o = self.origin_world
        return (
            x[0],
            y[0],
            z[0],
            o[0],
            x[1],
            y[1],
            z[1],
            o[1],
            x[2],
            y[2],
            z[2],
            o[2],
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @property
    def scientific_digest(self) -> str:
        return content_digest(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "width": self.width,
                "height": self.height,
                "depth": self.depth,
                "world_from_local": list(self.world_from_local),
                "scientific_digest": self.scientific_digest,
            }
        )
        return result


@dataclass(frozen=True)
class SimilarityRegistration:
    """Proper canonical-local to instance-world similarity transform."""

    instance_id: str
    matrix: tuple[float, ...]
    scale: float
    initialization_matrix: tuple[float, ...]
    objective_before: float
    objective_after: float
    correction_translation_m: float
    rotation_angle_degrees: float
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.instance_id not in INSTANCE_IDS:
            raise StructuringError("unknown registration instance")
        validate_similarity_matrix(self.matrix, maximum_scale_change=0.75)
        validate_similarity_matrix(
            self.initialization_matrix, maximum_scale_change=0.75
        )
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise StructuringError("registration scale must be positive")
        if self.objective_after > self.objective_before + 1e-8:
            raise StructuringError("registration refinement worsened its objective")
        if self.correction_translation_m > 0.35:
            raise StructuringError("registration correction exceeds 0.35 m")

    @property
    def scientific_digest(self) -> str:
        return content_digest(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scientific_digest"] = self.scientific_digest
        return result


def validate_similarity_matrix(
    matrix: tuple[float, ...], *, maximum_scale_change: float
) -> float:
    """Reject nonfinite, singular, reflected, sheared, or excessive transforms."""

    _vector(matrix, 16, "similarity matrix")
    if any(
        abs(matrix[index] - value) > 1e-6
        for index, value in zip((12, 13, 14, 15), (0.0, 0.0, 0.0, 1.0), strict=True)
    ):
        raise StructuringError("invalid homogeneous transform")
    columns = tuple(
        tuple(matrix[row * 4 + column] for row in range(3)) for column in range(3)
    )
    lengths = tuple(math.sqrt(sum(value * value for value in col)) for col in columns)
    if min(lengths) <= 1e-8 or max(lengths) - min(lengths) > 1e-5:
        raise StructuringError("transform must be a nonsingular uniform similarity")
    scale = sum(lengths) / 3
    axes = cast(
        tuple[tuple[float, float, float], ...],
        tuple(tuple(value / scale for value in col) for col in columns),
    )
    for left in range(3):
        for right in range(3):
            dot = sum(axes[left][i] * axes[right][i] for i in range(3))
            if abs(dot - (1.0 if left == right else 0.0)) > 1e-5:
                raise StructuringError("transform contains shear")
    if _determinant(axes) <= 0:
        raise StructuringError("registration reflections are forbidden")
    if abs(scale - 1.0) > maximum_scale_change:
        raise StructuringError("registration scale change exceeds policy")
    return scale


@dataclass(frozen=True)
class Q8Residuals:
    """One symmetric signed-int8 SH0/RGB offset per instance."""

    instance_ids: tuple[str, ...]
    values: tuple[tuple[int, int, int], ...]
    scale: float
    saturation_count: int
    rounding: str = "round_half_away_from_zero"
    zero_point: int = 0
    canonical_zero_instance: str = CANONICAL_INSTANCE_ID

    def __post_init__(self) -> None:
        if not self.instance_ids or len(self.instance_ids) != len(
            set(self.instance_ids)
        ):
            raise StructuringError("q8 instance order must be unique and nonempty")
        if len(self.values) != len(self.instance_ids) or any(
            len(row) != 3 for row in self.values
        ):
            raise StructuringError("q8 residuals must have shape [instances,3]")
        if any(value < -128 or value > 127 for row in self.values for value in row):
            raise StructuringError("q8 payload exceeds signed-int8 range")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise StructuringError("q8 scale must be finite and positive")
        if self.canonical_zero_instance not in self.instance_ids:
            raise StructuringError("q8 canonical instance is not declared")
        canonical_index = self.instance_ids.index(self.canonical_zero_instance)
        if self.values[canonical_index] != (0, 0, 0):
            raise StructuringError("canonical residual must be exactly zero")
        if self.rounding != "round_half_away_from_zero" or self.zero_point != 0:
            raise StructuringError("q8 saturation/rounding policy mismatch")

    def decode(self, instance_id: str) -> tuple[float, float, float]:
        try:
            index = self.instance_ids.index(instance_id)
        except ValueError as error:
            raise StructuringError("unbound q8 residual instance") from error
        return cast(
            tuple[float, float, float],
            tuple(value * self.scale for value in self.values[index]),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["decoded"] = {
            instance_id: self.decode(instance_id) for instance_id in self.instance_ids
        }
        result["scientific_digest"] = content_digest(result)
        return result
