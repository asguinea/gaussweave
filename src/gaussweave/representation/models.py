"""Typed, CPU-safe Gaussian representation primitives for the GW evaluation."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from gaussweave.config.resolution import content_digest as _content_digest
from gaussweave.rendering.models import RenderableGaussians

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Matrix4 = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]

CANONICAL_FRAME = "gaussweave_world_z_up_right_handed"
LOCAL_FRAME = "terminal_local_z_up_right_handed"
PARAMETER_CONVENTION = {
    "means": "xyz_in_world_units",
    "quaternions": "normalized_wxyz",
    "scales": "positive_linear_xyz",
    "opacities": "direct_alpha_in_[0,1]",
    "appearance": "direct_rgb_sh_degree_0_in_[0,1]",
    "dtype": "little_endian_float32",
}


def f32(value: float) -> float:
    """Round a Python float to the canonical little-endian float32 value."""

    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


def content_digest(value: Any) -> str:
    """Type-narrowing wrapper around the canonical JSON digest helper."""

    return _content_digest(cast(Any, value))


def _finite(values: Iterable[float], *, name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")


def _normalized_quaternion(quaternion: Quaternion) -> Quaternion:
    _finite(quaternion, name="quaternion")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    return cast(Quaternion, tuple(f32(value / norm) for value in quaternion))


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    """Compose two normalized wxyz quaternions."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _normalized_quaternion(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector by a normalized wxyz quaternion."""

    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        f32(vx + w * tx + (y * tz - z * ty)),
        f32(vy + w * ty + (z * tx - x * tz)),
        f32(vz + w * tz + (x * ty - y * tx)),
    )


def _binary_digest(rows: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    if rows and isinstance(rows[0], (tuple, list)):
        for row in rows:
            for value in cast(Sequence[float], row):
                digest.update(struct.pack("<f", float(value)))
    else:
        for value in rows:
            digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


@dataclass(frozen=True)
class GaussianArrays:
    """Complete canonical Gaussian arrays with deterministic ordering metadata."""

    means: tuple[Vector3, ...]
    quaternions: tuple[Quaternion, ...]
    scales: tuple[Vector3, ...]
    opacities: tuple[float, ...]
    colors: tuple[Vector3, ...]
    stable_ids: tuple[str, ...]
    coordinate_frame: str = CANONICAL_FRAME
    ordering_policy: str = "stable_ids_lexical_by_construction"
    sh_degree: int = 0

    def __post_init__(self) -> None:
        count = len(self.means)
        if count <= 0:
            raise ValueError("Gaussian arrays must be non-empty")
        if count > 100_000:
            raise ValueError("Gaussian arrays exceed the bounded GW size")
        lengths = {
            len(self.quaternions),
            len(self.scales),
            len(self.opacities),
            len(self.colors),
            len(self.stable_ids),
        }
        if lengths != {count}:
            raise ValueError("all Gaussian fields must have the same leading dimension")
        if len(set(self.stable_ids)) != count:
            raise ValueError("stable Gaussian IDs must be unique")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame must be non-empty")
        if self.sh_degree != 0:
            raise ValueError("GW supports direct RGB / SH degree 0 only")
        for index, mean in enumerate(self.means):
            if len(mean) != 3:
                raise ValueError(f"means[{index}] must have shape [3]")
            _finite(mean, name=f"means[{index}]")
        for index, quaternion in enumerate(self.quaternions):
            if len(quaternion) != 4:
                raise ValueError(f"quaternions[{index}] must have shape [4]")
            _finite(quaternion, name=f"quaternions[{index}]")
            norm = math.sqrt(sum(value * value for value in quaternion))
            if not math.isclose(norm, 1.0, abs_tol=2e-5):
                raise ValueError(f"quaternions[{index}] is not normalized wxyz")
        for index, scale in enumerate(self.scales):
            if len(scale) != 3:
                raise ValueError(f"scales[{index}] must have shape [3]")
            _finite(scale, name=f"scales[{index}]")
            if any(value <= 0.0 for value in scale):
                raise ValueError(f"scales[{index}] must be strictly positive")
        for index, opacity in enumerate(self.opacities):
            if not math.isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
                raise ValueError(f"opacities[{index}] must be in [0,1]")
        for index, color in enumerate(self.colors):
            if len(color) != 3:
                raise ValueError(f"colors[{index}] must have shape [3]")
            _finite(color, name=f"colors[{index}]")
            if any(value < 0.0 or value > 1.0 for value in color):
                raise ValueError(f"colors[{index}] must be in [0,1]")

    @property
    def count(self) -> int:
        return len(self.means)

    @property
    def scientific_digest(self) -> str:
        projection = {
            "count": self.count,
            "coordinate_frame": self.coordinate_frame,
            "ordering_policy": self.ordering_policy,
            "sh_degree": self.sh_degree,
            "parameter_convention": PARAMETER_CONVENTION,
            "field_digests": {
                "means": _binary_digest(self.means),
                "quaternions": _binary_digest(self.quaternions),
                "scales": _binary_digest(self.scales),
                "opacities": _binary_digest(self.opacities),
                "colors": _binary_digest(self.colors),
                "stable_ids": content_digest(list(self.stable_ids)),
            },
        }
        return content_digest(projection)

    @property
    def field_shapes(self) -> dict[str, list[int]]:
        return {
            "means": [self.count, 3],
            "quaternions": [self.count, 4],
            "scales": [self.count, 3],
            "opacities": [self.count],
            "colors": [self.count, 3],
        }

    def to_renderable(self) -> RenderableGaussians:
        return RenderableGaussians(
            means=self.means,
            quaternions=self.quaternions,
            scales=self.scales,
            opacities=self.opacities,
            appearance=self.colors,
            appearance_mode="direct_rgb",
            sh_degree=None,
            stable_ids=self.stable_ids,
        )


@dataclass(frozen=True)
class CanonicalTerminal:
    """The shared terminal Gaussian component in its local frame."""

    component_id: str
    gaussians: GaussianArrays
    provenance: Mapping[str, Any]
    role: str = "shared_terminal_panel"
    local_frame: str = LOCAL_FRAME

    def __post_init__(self) -> None:
        if self.component_id != "terminal-panel-v1":
            raise ValueError("the frozen GW terminal ID is terminal-panel-v1")
        if self.gaussians.count != 256:
            raise ValueError("the frozen GW terminal must contain 256 Gaussians")
        if self.gaussians.coordinate_frame != self.local_frame:
            raise ValueError("terminal Gaussians must use the declared local frame")

    @property
    def local_bounds(self) -> tuple[Vector3, Vector3]:
        lower = tuple(
            min(row[axis] for row in self.gaussians.means) for axis in range(3)
        )
        upper = tuple(
            max(row[axis] for row in self.gaussians.means) for axis in range(3)
        )
        return lower, upper  # type: ignore[return-value]

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "component_id": self.component_id,
                "role": self.role,
                "local_frame": self.local_frame,
                "local_bounds": self.local_bounds,
                "gaussians": self.gaussians.scientific_digest,
                "provenance": dict(self.provenance),
            }
        )


@dataclass(frozen=True)
class UniqueComponent:
    """Non-repeated scene Gaussians stored explicitly."""

    component_id: str
    gaussians: GaussianArrays
    provenance: Mapping[str, Any]
    role: str = "unique_backdrop_and_supports"

    def __post_init__(self) -> None:
        if self.gaussians.count != 512:
            raise ValueError(
                "the frozen GW unique component must contain 512 Gaussians"
            )
        if self.gaussians.coordinate_frame != CANONICAL_FRAME:
            raise ValueError("unique component must use the canonical world frame")

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "component_id": self.component_id,
                "role": self.role,
                "gaussians": self.gaussians.scientific_digest,
                "provenance": dict(self.provenance),
            }
        )


@dataclass(frozen=True)
class SimilarityTransform:
    """Proper, non-singular similarity transform (no reflection or shear)."""

    translation: Vector3
    quaternion: Quaternion = (1.0, 0.0, 0.0, 0.0)
    uniform_scale: float = 1.0

    def __post_init__(self) -> None:
        _finite(self.translation, name="translation")
        normalized = _normalized_quaternion(self.quaternion)
        if any(
            abs(left - right) > 2e-5
            for left, right in zip(self.quaternion, normalized, strict=True)
        ):
            raise ValueError("transform quaternion must be normalized wxyz")
        if not math.isfinite(self.uniform_scale) or self.uniform_scale <= 0.0:
            raise ValueError("uniform_scale must be finite and positive")

    @classmethod
    def from_matrix(
        cls, matrix: Sequence[float], *, tolerance: float = 1e-5
    ) -> SimilarityTransform:
        if len(matrix) != 16:
            raise ValueError("affine transform must contain 16 row-major values")
        values = tuple(float(value) for value in matrix)
        _finite(values, name="affine transform")
        if any(
            abs(values[index] - expected) > tolerance
            for index, expected in zip(
                (12, 13, 14, 15),
                (0, 0, 0, 1),
                strict=True,
            )
        ):
            raise ValueError(
                "affine transform must have homogeneous bottom row [0,0,0,1]"
            )
        columns = (
            (values[0], values[4], values[8]),
            (values[1], values[5], values[9]),
            (values[2], values[6], values[10]),
        )
        lengths = tuple(
            math.sqrt(sum(value * value for value in column)) for column in columns
        )
        if min(lengths) <= tolerance:
            raise ValueError("singular transforms are not supported")
        if max(lengths) - min(lengths) > tolerance:
            raise ValueError("non-uniform scaling is not supported")
        scale = sum(lengths) / 3.0
        rotation_columns = tuple(
            tuple(value / scale for value in column) for column in columns
        )
        for left in range(3):
            for right in range(left + 1, 3):
                dot = sum(
                    rotation_columns[left][i] * rotation_columns[right][i]
                    for i in range(3)
                )
                if abs(dot) > tolerance:
                    raise ValueError("shear is not supported")
        determinant = (
            rotation_columns[0][0]
            * (
                rotation_columns[1][1] * rotation_columns[2][2]
                - rotation_columns[1][2] * rotation_columns[2][1]
            )
            - rotation_columns[1][0]
            * (
                rotation_columns[0][1] * rotation_columns[2][2]
                - rotation_columns[0][2] * rotation_columns[2][1]
            )
            + rotation_columns[2][0]
            * (
                rotation_columns[0][1] * rotation_columns[1][2]
                - rotation_columns[0][2] * rotation_columns[1][1]
            )
        )
        if determinant <= 0.0:
            raise ValueError("reflections are not supported")
        r00, r01, r02 = values[0] / scale, values[1] / scale, values[2] / scale
        r10, r11, r12 = values[4] / scale, values[5] / scale, values[6] / scale
        r20, r21, r22 = values[8] / scale, values[9] / scale, values[10] / scale
        trace = r00 + r11 + r22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            quaternion = (0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s)
        elif r00 > r11 and r00 > r22:
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
            quaternion = ((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s)
        elif r11 > r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
            quaternion = ((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s)
        else:
            s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
            quaternion = ((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s)
        normalized = _normalized_quaternion(quaternion)
        if normalized[0] < 0.0:
            normalized = tuple(-value for value in normalized)  # type: ignore[assignment]
        return cls(
            translation=(f32(values[3]), f32(values[7]), f32(values[11])),
            quaternion=normalized,
            uniform_scale=f32(scale),
        )

    def apply_mean(self, mean: Vector3) -> Vector3:
        rotated = rotate_vector(self.quaternion, mean)
        return tuple(
            f32(self.translation[axis] + self.uniform_scale * rotated[axis])
            for axis in range(3)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class GridRepeat:
    """Deterministic row-major procedural grid and active-instance set."""

    rows: int
    columns: int
    origin: Vector3
    row_step: Vector3
    column_step: Vector3
    active_indices: tuple[int, ...] = field(default_factory=tuple)
    ordering_policy: str = "row_major_active_indices"

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0 or self.rows * self.columns > 32:
            raise ValueError("grid dimensions must produce between 1 and 32 instances")
        _finite((*self.origin, *self.row_step, *self.column_step), name="grid vectors")
        if sum(value * value for value in self.row_step) <= 1e-12:
            raise ValueError("row_step must be non-zero")
        if sum(value * value for value in self.column_step) <= 1e-12:
            raise ValueError("column_step must be non-zero")
        cross = (
            self.row_step[1] * self.column_step[2]
            - self.row_step[2] * self.column_step[1],
            self.row_step[2] * self.column_step[0]
            - self.row_step[0] * self.column_step[2],
            self.row_step[0] * self.column_step[1]
            - self.row_step[1] * self.column_step[0],
        )
        if sum(value * value for value in cross) <= 1e-12:
            raise ValueError("row_step and column_step must not be collinear")
        active = self.active_indices or tuple(range(self.rows * self.columns))
        if tuple(sorted(set(active))) != active:
            raise ValueError("active_indices must be unique and sorted")
        if any(index < 0 or index >= self.rows * self.columns for index in active):
            raise ValueError("active_indices contain an out-of-range grid index")
        object.__setattr__(self, "active_indices", active)

    @property
    def instance_count(self) -> int:
        return len(self.active_indices)

    def instance_id(self, flat_index: int) -> str:
        row, column = divmod(flat_index, self.columns)
        return f"instance-r{row:02d}-c{column:02d}"

    def transform(self, flat_index: int) -> SimilarityTransform:
        if flat_index not in self.active_indices:
            raise ValueError(f"grid index {flat_index} is not active")
        row, column = divmod(flat_index, self.columns)
        translation = tuple(
            f32(
                self.origin[axis]
                + row * self.row_step[axis]
                + column * self.column_step[axis]
            )
            for axis in range(3)
        )
        return SimilarityTransform(translation=translation)  # type: ignore[arg-type]

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "rows": self.rows,
                "columns": self.columns,
                "origin": self.origin,
                "row_step": self.row_step,
                "column_step": self.column_step,
                "active_indices": self.active_indices,
                "ordering_policy": self.ordering_policy,
            }
        )


def quantize_q8(value: float, scale: float) -> tuple[int, bool]:
    """Round half away from zero and saturate to the signed int8 range."""

    if not math.isfinite(value):
        raise ValueError("residual values must be finite")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("q8 scale must be finite and positive")
    magnitude = math.floor(abs(value / scale) + 0.5)
    quantized = magnitude if value >= 0.0 else -magnitude
    clipped = min(127, max(-128, quantized))
    return clipped, clipped != quantized


@dataclass(frozen=True)
class AppearanceResiduals:
    """Per-instance RGB appearance residuals with explicit q8 semantics."""

    values: tuple[tuple[int, int, int], ...]
    instance_ids: tuple[str, ...]
    scale: float = f32(0.08 / 127.0)
    zero_point: int = 0
    dtype: str = "int8"
    channels: tuple[str, str, str] = ("red", "green", "blue")
    clipping_policy: str = "clip_reconstructed_rgb_to_[0,1]"
    ordering_policy: str = "grid_instance_order_then_rgb"
    saturation_count: int = 0

    def __post_init__(self) -> None:
        if len(self.values) != len(self.instance_ids):
            raise ValueError("residual values and instance IDs must align")
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("residual instance IDs must be unique")
        if any(len(row) != 3 for row in self.values):
            raise ValueError("residual rows must have exactly three RGB channels")
        if any(value < -128 or value > 127 for row in self.values for value in row):
            raise ValueError("q8 residuals must use the signed int8 range")
        if self.zero_point != 0 or self.dtype != "int8":
            raise ValueError("GW residuals require symmetric int8 with zero point 0")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("residual scale must be finite and positive")
        if self.saturation_count < 0:
            raise ValueError("saturation_count must be non-negative")

    @classmethod
    def encode(
        cls,
        offsets: Sequence[Vector3],
        instance_ids: Sequence[str],
        *,
        scale: float = f32(0.08 / 127.0),
    ) -> AppearanceResiduals:
        encoded: list[tuple[int, int, int]] = []
        saturation_count = 0
        for row in offsets:
            values: list[int] = []
            for value in row:
                quantized, saturated = quantize_q8(value, scale)
                values.append(quantized)
                saturation_count += int(saturated)
            encoded.append((values[0], values[1], values[2]))
        return cls(
            values=tuple(encoded),
            instance_ids=tuple(instance_ids),
            scale=f32(scale),
            saturation_count=saturation_count,
        )

    def decode(self, instance_id: str) -> Vector3:
        try:
            index = self.instance_ids.index(instance_id)
        except ValueError as error:
            raise KeyError(f"no residual is bound to {instance_id}") from error
        return tuple(
            f32((value - self.zero_point) * self.scale) for value in self.values[index]
        )  # type: ignore[return-value]

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "values": self.values,
                "instance_ids": self.instance_ids,
                "scale": self.scale,
                "zero_point": self.zero_point,
                "dtype": self.dtype,
                "channels": self.channels,
                "clipping_policy": self.clipping_policy,
                "ordering_policy": self.ordering_policy,
                "saturation_count": self.saturation_count,
            }
        )


@dataclass(frozen=True)
class ExplicitRepresentation:
    """Fully materialized explicit Gaussian scene."""

    gaussians: GaussianArrays
    fixture_id: str
    appearance_regime: str
    principal_seed: int
    grid: GridRepeat
    method_id: str = "explicit_gs"
    format_version: str = "gaussweave.explicit.v1"
    limitations: tuple[str, ...] = (
        "direct_rgb_sh_degree_0_only",
        "evaluation_fixture_scale_only",
    )

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "method_id": self.method_id,
                "format_version": self.format_version,
                "fixture_id": self.fixture_id,
                "appearance_regime": self.appearance_regime,
                "principal_seed": self.principal_seed,
                "grid": self.grid.scientific_digest,
                "gaussians": self.gaussians.scientific_digest,
                "limitations": self.limitations,
            }
        )


@dataclass(frozen=True)
class PrunedRepresentation:
    """Budget-matched nonstructural subset of an explicit Gaussian scene."""

    gaussians: GaussianArrays
    retained_original_indices: tuple[int, ...]
    source_representation_digest: str
    fixture_id: str
    appearance_regime: str
    principal_seed: int
    grid: GridRepeat
    target_budget_bytes: int
    actual_complete_bytes: int
    source_gaussian_count: int
    score_summary: Mapping[str, Any]
    method_id: str = "prune_budget_matched"
    format_version: str = "gaussweave.pruned-explicit.v1"
    importance_policy_version: str = "opacity-times-linear-volume-v1"
    ordering_policy: str = "rank_prefix_selected_then_original_index_ascending"
    limitations: tuple[str, ...] = (
        "deterministic_nonsemantic_pruning_control",
        "no_image_or_reference_error_access",
        "direct_rgb_sh_degree_0_only",
        "materialized_explicit_rendering",
    )

    def __post_init__(self) -> None:
        count = self.gaussians.count
        if self.method_id != "prune_budget_matched":
            raise ValueError("pruned representation method ID is frozen")
        if self.appearance_regime != "low_variation":
            raise ValueError("budget-matched pruning applies only to low variation")
        if len(self.retained_original_indices) != count:
            raise ValueError("retained indices must align with retained Gaussians")
        if tuple(sorted(set(self.retained_original_indices))) != (
            self.retained_original_indices
        ):
            raise ValueError("retained indices must be unique and ascending")
        if any(
            index < 0 or index >= self.source_gaussian_count
            for index in self.retained_original_indices
        ):
            raise ValueError("retained index is outside the source representation")
        if count > self.source_gaussian_count:
            raise ValueError("retained count cannot exceed source count")
        if self.target_budget_bytes <= 0 or self.actual_complete_bytes < 0:
            raise ValueError("pruning byte budgets must be non-negative and bounded")

    @property
    def stored_gaussian_count(self) -> int:
        return self.gaussians.count

    @property
    def materialized_gaussian_count(self) -> int:
        return self.gaussians.count

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "method_id": self.method_id,
                "format_version": self.format_version,
                "fixture_id": self.fixture_id,
                "appearance_regime": self.appearance_regime,
                "principal_seed": self.principal_seed,
                "grid": self.grid.scientific_digest,
                "gaussians": self.gaussians.scientific_digest,
                "retained_original_indices": self.retained_original_indices,
                "source_representation_digest": self.source_representation_digest,
                "target_budget_bytes": self.target_budget_bytes,
                "actual_complete_bytes": self.actual_complete_bytes,
                "source_gaussian_count": self.source_gaussian_count,
                "score_summary": dict(self.score_summary),
                "importance_policy_version": self.importance_policy_version,
                "ordering_policy": self.ordering_policy,
                "limitations": self.limitations,
            }
        )


@dataclass(frozen=True)
class StructuralRepresentation:
    """Unique + shared-terminal + grid + optional residual representation."""

    unique: UniqueComponent
    terminal: CanonicalTerminal
    grid: GridRepeat
    fixture_id: str
    appearance_regime: str
    principal_seed: int
    residuals: AppearanceResiduals | None = None
    method_id: str = "struct_shared"
    format_version: str = "gaussweave.structural.v1"
    decoder_id: str = "deterministic_similarity_instancer_v1"
    limitations: tuple[str, ...] = (
        "single_shared_terminal_component",
        "similarity_transforms_only",
        "direct_rgb_sh_degree_0_only",
        "materializes_before_rendering",
    )

    def __post_init__(self) -> None:
        expected_method = (
            "struct_residual_q8" if self.residuals is not None else "struct_shared"
        )
        if self.method_id != expected_method:
            raise ValueError(f"method_id must be {expected_method}")
        instance_ids = tuple(
            self.grid.instance_id(index) for index in self.grid.active_indices
        )
        if self.residuals is not None and self.residuals.instance_ids != instance_ids:
            raise ValueError("residual bindings must exactly match grid instance order")

    @property
    def stored_gaussian_count(self) -> int:
        return self.unique.gaussians.count + self.terminal.gaussians.count

    @property
    def materialized_gaussian_count(self) -> int:
        return (
            self.unique.gaussians.count
            + self.grid.instance_count * self.terminal.gaussians.count
        )

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "method_id": self.method_id,
                "format_version": self.format_version,
                "fixture_id": self.fixture_id,
                "appearance_regime": self.appearance_regime,
                "principal_seed": self.principal_seed,
                "unique": self.unique.scientific_digest,
                "terminal": self.terminal.scientific_digest,
                "grid": self.grid.scientific_digest,
                "residuals": None
                if self.residuals is None
                else self.residuals.scientific_digest,
                "decoder_id": self.decoder_id,
                "limitations": self.limitations,
            }
        )


@dataclass(frozen=True)
class MaterializationResult:
    gaussians: GaussianArrays
    source_digest: str
    method_id: str
    instance_count: int
    stored_gaussian_count: int
    materialized_gaussian_count: int
    clipping_count: int

    def __post_init__(self) -> None:
        if self.gaussians.count != self.materialized_gaussian_count:
            raise ValueError(
                "materialization count metadata does not match the output arrays"
            )


def _append_arrays(
    target: dict[str, list[Any]],
    source: GaussianArrays,
    *,
    transform: SimilarityTransform | None,
    id_prefix: str | None,
    residual: Vector3 | None,
) -> int:
    clipping_count = 0
    for index in range(source.count):
        if transform is None:
            mean = source.means[index]
            quaternion = source.quaternions[index]
            scale = source.scales[index]
        else:
            mean = transform.apply_mean(source.means[index])
            quaternion = quaternion_multiply(
                transform.quaternion, source.quaternions[index]
            )
            scale = cast(
                Vector3,
                tuple(
                    f32(value * transform.uniform_scale)
                    for value in source.scales[index]
                ),
            )
        color_values: list[float] = []
        for channel, value in enumerate(source.colors[index]):
            reconstructed = value + (0.0 if residual is None else residual[channel])
            clipped = min(1.0, max(0.0, reconstructed))
            clipping_count += int(clipped != reconstructed)
            color_values.append(f32(clipped))
        target["means"].append(mean)
        target["quaternions"].append(quaternion)
        target["scales"].append(scale)
        target["opacities"].append(source.opacities[index])
        target["colors"].append(tuple(color_values))
        source_id = source.stable_ids[index]
        target["stable_ids"].append(
            source_id if id_prefix is None else f"{id_prefix}/{source_id}"
        )
    return clipping_count


def materialize(representation: StructuralRepresentation) -> MaterializationResult:
    """Deterministically expand a structural representation into explicit arrays."""

    fields: dict[str, list[Any]] = {
        "means": [],
        "quaternions": [],
        "scales": [],
        "opacities": [],
        "colors": [],
        "stable_ids": [],
    }
    clipping_count = _append_arrays(
        fields,
        representation.unique.gaussians,
        transform=None,
        id_prefix=None,
        residual=None,
    )
    for flat_index in representation.grid.active_indices:
        instance_id = representation.grid.instance_id(flat_index)
        residual = (
            None
            if representation.residuals is None
            else representation.residuals.decode(instance_id)
        )
        clipping_count += _append_arrays(
            fields,
            representation.terminal.gaussians,
            transform=representation.grid.transform(flat_index),
            id_prefix=instance_id,
            residual=residual,
        )
    gaussians = GaussianArrays(
        means=tuple(fields["means"]),
        quaternions=tuple(fields["quaternions"]),
        scales=tuple(fields["scales"]),
        opacities=tuple(fields["opacities"]),
        colors=tuple(fields["colors"]),
        stable_ids=tuple(fields["stable_ids"]),
    )
    return MaterializationResult(
        gaussians=gaussians,
        source_digest=representation.scientific_digest,
        method_id=representation.method_id,
        instance_count=representation.grid.instance_count,
        stored_gaussian_count=representation.stored_gaussian_count,
        materialized_gaussian_count=representation.materialized_gaussian_count,
        clipping_count=clipping_count,
    )


def concatenate_explicit(
    unique: UniqueComponent,
    terminal: CanonicalTerminal,
    grid: GridRepeat,
    offsets: Mapping[str, Vector3] | None,
) -> GaussianArrays:
    """Build the independent explicit reference without calling ``materialize``."""

    fields: dict[str, list[Any]] = {
        "means": [],
        "quaternions": [],
        "scales": [],
        "opacities": [],
        "colors": [],
        "stable_ids": [],
    }
    _append_arrays(
        fields, unique.gaussians, transform=None, id_prefix=None, residual=None
    )
    for flat_index in grid.active_indices:
        instance_id = grid.instance_id(flat_index)
        _append_arrays(
            fields,
            terminal.gaussians,
            transform=grid.transform(flat_index),
            id_prefix=instance_id,
            residual=None if offsets is None else offsets[instance_id],
        )
    return GaussianArrays(
        means=tuple(fields["means"]),
        quaternions=tuple(fields["quaternions"]),
        scales=tuple(fields["scales"]),
        opacities=tuple(fields["opacities"]),
        colors=tuple(fields["colors"]),
        stable_ids=tuple(fields["stable_ids"]),
    )
