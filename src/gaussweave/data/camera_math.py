"""CPU-only camera geometry under the project OpenCV convention.

Camera-local axes are ``+X`` right, ``+Y`` down, and ``+Z`` forward.  A
``world_from_camera`` matrix stores those three axes as columns and the camera
position as its final column.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Vector3 = tuple[float, float, float]
Matrix4 = tuple[float, ...]


def add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def scale(value: Vector3, factor: float) -> Vector3:
    return tuple(component * factor for component in value)  # type: ignore[return-value]


def dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(value: Vector3) -> float:
    return math.sqrt(dot(value, value))


def normalize(value: Vector3, *, tolerance: float = 1e-12) -> Vector3:
    length = norm(value)
    if not math.isfinite(length) or length <= tolerance:
        raise ValueError("cannot normalize a zero or nonfinite vector")
    return scale(value, 1.0 / length)


def look_at_world_from_camera(
    position: Vector3,
    target: Vector3,
    *,
    world_up: Vector3 = (0.0, 0.0, 1.0),
) -> Matrix4:
    """Build a right-handed OpenCV camera pose with deterministic up fallback."""

    forward = normalize(subtract(target, position))
    up = normalize(world_up)
    if abs(dot(forward, up)) > 1.0 - 1e-8:
        candidates = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        up = min(
            enumerate(candidates),
            key=lambda item: (abs(dot(forward, item[1])), item[0]),
        )[1]
    right = normalize(cross(forward, up))
    down = normalize(cross(forward, right))
    return (
        right[0],
        down[0],
        forward[0],
        position[0],
        right[1],
        down[1],
        forward[1],
        position[1],
        right[2],
        down[2],
        forward[2],
        position[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def pose_axes(matrix: Sequence[float]) -> tuple[Vector3, Vector3, Vector3, Vector3]:
    if len(matrix) != 16:
        raise ValueError("world_from_camera must contain 16 values")
    right = (float(matrix[0]), float(matrix[4]), float(matrix[8]))
    down = (float(matrix[1]), float(matrix[5]), float(matrix[9]))
    forward = (float(matrix[2]), float(matrix[6]), float(matrix[10]))
    position = (float(matrix[3]), float(matrix[7]), float(matrix[11]))
    return right, down, forward, position


def rotation_determinant(matrix: Sequence[float]) -> float:
    right, down, forward, _ = pose_axes(matrix)
    return dot(right, cross(down, forward))


def validate_rigid_pose(matrix: Sequence[float], *, tolerance: float = 1e-10) -> None:
    if len(matrix) != 16 or not all(math.isfinite(float(v)) for v in matrix):
        raise ValueError("world_from_camera must contain 16 finite values")
    right, down, forward, _ = pose_axes(matrix)
    for name, axis in (("right", right), ("down", down), ("forward", forward)):
        if abs(norm(axis) - 1.0) > tolerance:
            raise ValueError(f"{name} axis is not unit length")
    if any(
        abs(value) > tolerance
        for value in (dot(right, down), dot(right, forward), dot(down, forward))
    ):
        raise ValueError("camera rotation is not orthogonal")
    if abs(rotation_determinant(matrix) - 1.0) > tolerance:
        raise ValueError("camera rotation determinant is not +1")
    if (
        any(abs(float(matrix[index])) > tolerance for index in (12, 13, 14))
        or abs(float(matrix[15]) - 1.0) > tolerance
    ):
        raise ValueError("camera transform has an invalid homogeneous row")


def focal_from_horizontal_fov(width: int, fov_degrees: float) -> float:
    if width <= 0 or not 0.0 < fov_degrees < 180.0:
        raise ValueError("width and horizontal field of view must be valid")
    return width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))


def focal_from_lens_mm(
    width: int, lens_mm: float, *, sensor_width_mm: float = 36.0
) -> float:
    if width <= 0 or lens_mm <= 0.0 or sensor_width_mm <= 0.0:
        raise ValueError("width, lens, and sensor width must be positive")
    return width * lens_mm / sensor_width_mm


def horizontal_fov_degrees(width: int, fx: float) -> float:
    if width <= 0 or fx <= 0.0:
        raise ValueError("width and focal length must be positive")
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def project_point(
    point: Vector3,
    world_from_camera: Sequence[float],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """Project a world point to pixel coordinates and return forward depth."""

    right, down, forward, position = pose_axes(world_from_camera)
    relative = subtract(point, position)
    x = dot(relative, right)
    y = dot(relative, down)
    z = dot(relative, forward)
    if z <= 0.0:
        return math.nan, math.nan, z
    return fx * x / z + cx, fy * y / z + cy, z


def euclidean_distance(left: Vector3, right: Vector3) -> float:
    return norm(subtract(left, right))


def angular_distance_degrees(left: Vector3, right: Vector3) -> float:
    cosine = max(-1.0, min(1.0, dot(normalize(left), normalize(right))))
    return math.degrees(math.acos(cosine))


def bounding_box(
    points: Iterable[tuple[float, float]],
) -> tuple[float, float, float, float]:
    values = tuple(points)
    if not values:
        raise ValueError("cannot bound an empty point set")
    xs = [point[0] for point in values]
    ys = [point[1] for point in values]
    return min(xs), min(ys), max(xs), max(ys)
