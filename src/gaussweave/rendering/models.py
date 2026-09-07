"""Project-owned CPU-safe camera, Gaussian, settings, and result models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

MATRIX_TOLERANCE = 1e-5
QUATERNION_TOLERANCE = 1e-6


class RenderModelError(ValueError):
    """A project rendering model is invalid."""


class GaussianTensorSource(Protocol):
    """Validated Gaussian source consumed by the project renderer."""

    @property
    def appearance_mode(self) -> str: ...

    @property
    def sh_degree(self) -> int | None: ...

    @property
    def color_activation(self) -> str: ...

    @property
    def count(self) -> int: ...

    @property
    def coefficient_shape(self) -> tuple[int, ...]: ...

    def tensors(self, device: str = "cuda:0") -> dict[str, Any]: ...


@dataclass(frozen=True)
class Camera:
    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: tuple[tuple[float, ...], ...]
    near: float = 0.01
    far: float = 1000.0
    coordinate_convention: str = "opencv"
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise RenderModelError("camera dimensions must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.fx, self.fy, self.cx, self.cy, self.near, self.far)
        ):
            raise RenderModelError("camera intrinsics and planes must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise RenderModelError("camera focal lengths must be positive")
        if not 0 < self.near < self.far:
            raise RenderModelError("camera planes require 0 < near < far")
        if self.coordinate_convention != "opencv":
            raise RenderModelError(
                f"unsupported camera convention: {self.coordinate_convention}"
            )
        if self.dtype != "float32":
            raise RenderModelError("reference camera conversion requires float32")
        _validate_matrix(self.world_from_camera)

    def gsplat_matrices(self, device: str = "cuda:0") -> tuple[Any, Any]:
        """Return explicit world-to-camera view and pinhole intrinsic tensors."""

        torch = _torch()
        rotation = [row[:3] for row in self.world_from_camera[:3]]
        translation = [row[3] for row in self.world_from_camera[:3]]
        rotation_t = [
            [rotation[column][row] for column in range(3)] for row in range(3)
        ]
        inverse_translation = [
            -sum(rotation_t[row][column] * translation[column] for column in range(3))
            for row in range(3)
        ]
        view = [[*rotation_t[row], inverse_translation[row]] for row in range(3)] + [
            [0.0, 0.0, 0.0, 1.0]
        ]
        intrinsics = [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
        return (
            torch.tensor(view, dtype=torch.float32, device=device),
            torch.tensor(intrinsics, dtype=torch.float32, device=device),
        )


@dataclass(frozen=True)
class RenderableGaussians:
    means: tuple[tuple[float, float, float], ...]
    quaternions: tuple[tuple[float, float, float, float], ...]
    scales: tuple[tuple[float, float, float], ...]
    opacities: tuple[float, ...]
    appearance: tuple[Any, ...]
    appearance_mode: str = "direct_rgb"
    sh_degree: int | None = None
    scales_are_log: bool = False
    stable_ids: tuple[str, ...] | None = None
    coordinate_frame: str = "world"
    dtype: str = "float32"
    device: str = "cpu"
    color_activation: str = "none"

    def __post_init__(self) -> None:
        count = len(self.means)
        if count < 1:
            raise RenderModelError("at least one Gaussian is required")
        fields = (self.quaternions, self.scales, self.opacities, self.appearance)
        if any(len(field) != count for field in fields):
            raise RenderModelError("Gaussian fields have inconsistent counts")
        if self.stable_ids is not None and len(self.stable_ids) != count:
            raise RenderModelError("stable ID count differs from Gaussian count")
        if self.stable_ids is not None and len(set(self.stable_ids)) != count:
            raise RenderModelError("stable Gaussian IDs must be unique")
        for mean in self.means:
            _finite_vector(mean, 3, "mean")
        for quaternion in self.quaternions:
            _finite_vector(quaternion, 4, "quaternion")
            norm = math.sqrt(sum(value * value for value in quaternion))
            if abs(norm - 1.0) > QUATERNION_TOLERANCE:
                raise RenderModelError("quaternion is not normalized scalar-first wxyz")
        for scale in self.scales:
            _finite_vector(scale, 3, "scale")
            try:
                activated = (
                    tuple(math.exp(value) for value in scale)
                    if self.scales_are_log
                    else scale
                )
            except OverflowError as error:
                raise RenderModelError("log-scale activation overflowed") from error
            if any(value <= 0 for value in activated):
                raise RenderModelError("activated scales must be positive")
        if not all(
            math.isfinite(value) and 0 <= value <= 1 for value in self.opacities
        ):
            raise RenderModelError("activated opacities must be finite in [0, 1]")
        if self.appearance_mode == "direct_rgb":
            if self.sh_degree is not None:
                raise RenderModelError("direct RGB must not declare an SH degree")
            for color in self.appearance:
                _finite_vector(color, 3, "RGB color")
                if any(value < 0 or value > 1 for value in color):
                    raise RenderModelError("direct RGB values must be in [0, 1]")
        elif self.appearance_mode == "sh":
            if self.sh_degree != 0:
                raise RenderModelError("only exact SH degree 0 is currently supported")
            for coefficients in self.appearance:
                if len(coefficients) != 1:
                    raise RenderModelError("SH0 appearance must have shape [N, 1, 3]")
                _finite_vector(coefficients[0], 3, "SH0 coefficient")
        else:
            raise RenderModelError(
                f"unsupported appearance mode: {self.appearance_mode}"
            )
        if self.coordinate_frame != "world" or self.dtype != "float32":
            raise RenderModelError(
                "reference rendering requires world-frame float32 Gaussians"
            )

    @property
    def count(self) -> int:
        return len(self.means)

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return (
            (self.count, 3)
            if self.appearance_mode == "direct_rgb"
            else (self.count, 1, 3)
        )

    def tensors(self, device: str = "cuda:0") -> dict[str, Any]:
        """Materialize validated float32 tensors lazily on an explicit device."""

        torch = _torch()
        scales = torch.tensor(self.scales, dtype=torch.float32, device=device)
        if self.scales_are_log:
            scales = scales.exp()
        return {
            "means": torch.tensor(self.means, dtype=torch.float32, device=device),
            "quaternions": torch.tensor(
                self.quaternions, dtype=torch.float32, device=device
            ),
            "scales": scales,
            "opacities": torch.tensor(
                self.opacities, dtype=torch.float32, device=device
            ),
            "appearance": torch.tensor(
                self.appearance, dtype=torch.float32, device=device
            ),
        }


@dataclass(frozen=True)
class RenderSettings:
    output_buffers: tuple[str, ...] = ("rgb", "alpha")
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_mode: str = "expected"
    packed: bool = True
    antialiasing: bool = False
    deterministic_smoke: bool = True
    camera_batch_size: int = 2
    warmup_count: int = 1
    repetition_count: int = 3
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not set(self.output_buffers) <= {"rgb", "alpha", "depth"}:
            raise RenderModelError("unsupported output buffer")
        if "rgb" not in self.output_buffers:
            raise RenderModelError("the current adapter requires RGB output")
        if self.depth_mode not in {"expected", "accumulated"}:
            raise RenderModelError("depth mode must be expected or accumulated")
        if self.camera_batch_size < 1 or self.warmup_count < 0:
            raise RenderModelError("invalid batch or warm-up count")
        if self.repetition_count < 1:
            raise RenderModelError("repetition count must be positive")
        _finite_vector(self.background_rgb, 3, "background")
        if any(value < 0 or value > 1 for value in self.background_rgb):
            raise RenderModelError("background RGB must be in [0, 1]")


@dataclass(frozen=True)
class RenderResult:
    rgb: Any
    alpha: Any
    depth: Any | None
    width: int
    height: int
    camera_ids: tuple[str, ...]
    stored_gaussian_count: int
    submitted_gaussian_count: int
    metadata: dict[str, Any]
    elapsed_seconds: float
    resource_record: Any
    renderer_name: str
    renderer_version: str


def _validate_matrix(matrix: tuple[tuple[float, ...], ...]) -> None:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise RenderModelError("world_from_camera must have shape [4, 4]")
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise RenderModelError("camera transform contains a nonfinite value")
    if any(
        abs(value - expected) > MATRIX_TOLERANCE
        for value, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0), strict=True)
    ):
        raise RenderModelError("camera transform has an invalid homogeneous bottom row")
    rotation = [row[:3] for row in matrix[:3]]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > MATRIX_TOLERANCE:
                raise RenderModelError("camera rotation must be orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > MATRIX_TOLERANCE:
        raise RenderModelError("camera rotation must be proper with determinant +1")


def _finite_vector(values: Any, length: int, name: str) -> None:
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise RenderModelError(f"{name} must have {length} finite values")


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RenderModelError("PyTorch is required for tensor conversion") from error
    return torch
