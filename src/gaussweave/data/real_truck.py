"""CPU-safe COLMAP ingestion and Truck dataset qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

from gaussweave.config.resolution import content_digest, pretty_json_bytes

QUALIFICATION_VERSION = "gw-truck-explicit-v1"
DATASET_ID = "tnt-truck"
SCENE_ID = "real-tnt-truck"
REGION_ID = "truck-bed-side-panels"
COORDINATE_CONVENTION = (
    "right-handed COLMAP world; row-major OpenCV world_from_camera; "
    "camera +X right, +Y down, +Z forward"
)
SOURCE_ARCHIVE_SHA256 = (
    "816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1"
)
SOURCE_ARCHIVE_BYTES = 682_628_995
ALLOWED_ROOT = PurePosixPath("/home/gaussweave/datasets/gaussweave/gw-real/tnt-truck")


class RealDataError(RuntimeError):
    """Real-dataset validation failed."""


COLMAP_MODELS: Mapping[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class ColmapCamera:
    """One COLMAP intrinsic calibration."""

    camera_id: int
    model: str
    width: int
    height: int
    parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.camera_id <= 0 or self.width <= 0 or self.height <= 0:
            raise RealDataError("invalid COLMAP camera identity or dimensions")
        if self.model not in {value[0] for value in COLMAP_MODELS.values()}:
            raise RealDataError(f"unsupported COLMAP camera model: {self.model}")
        if not all(math.isfinite(value) for value in self.parameters):
            raise RealDataError("camera parameters must be finite")
        fx, fy, cx, cy = self.intrinsics
        if fx <= 0 or fy <= 0:
            raise RealDataError("camera focal lengths must be positive")
        if not (0 <= cx <= self.width and 0 <= cy <= self.height):
            raise RealDataError("camera principal point is outside the image")

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """Return fx, fy, cx, cy in pixels."""

        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            focal, cx, cy = self.parameters[:3]
            return focal, focal, cx, cy
        fx, fy, cx, cy = self.parameters[:4]
        return fx, fy, cx, cy

    @property
    def distortion(self) -> tuple[float, ...]:
        """Return model-specific distortion parameters."""

        if self.model == "SIMPLE_PINHOLE" or self.model == "PINHOLE":
            return ()
        if self.model == "SIMPLE_RADIAL":
            return self.parameters[3:4]
        if self.model == "RADIAL":
            return self.parameters[3:5]
        return self.parameters[4:]


@dataclass(frozen=True)
class ColmapObservation:
    """Observed 2D coordinate and optional sparse point identity."""

    x: float
    y: float
    point3d_id: int


@dataclass(frozen=True)
class ColmapImage:
    """One COLMAP image/extrinsic record."""

    image_id: int
    qvec_wxyz: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    observations: tuple[ColmapObservation, ...]

    def __post_init__(self) -> None:
        if self.image_id <= 0 or self.camera_id <= 0 or not self.name:
            raise RealDataError("invalid COLMAP image identity")
        values = (*self.qvec_wxyz, *self.tvec)
        if not all(math.isfinite(value) for value in values):
            raise RealDataError("image pose must contain finite values")
        norm = math.sqrt(sum(value * value for value in self.qvec_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise RealDataError("COLMAP quaternion must be normalized scalar-first")


@dataclass(frozen=True)
class ColmapPoint:
    """Sparse 3D point and its COLMAP reprojection metadata."""

    point3d_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ImportedCamera:
    """Portable project camera imported from COLMAP."""

    camera_id: str
    source_image_id: int
    source_camera_id: int
    image_name: str
    width: int
    height: int
    calibration_width: int
    calibration_height: int
    image_scale_x: float
    image_scale_y: float
    model: str
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    world_from_camera: tuple[float, ...]
    digest: str


@dataclass(frozen=True)
class ReprojectionRecord:
    """Per-camera sparse reprojection qualification."""

    camera_id: str
    valid_observations: int
    invalid_observations: int
    median_px: float | None
    p95_px: float | None
    maximum_px: float | None


@dataclass(frozen=True)
class DatasetQualification:
    """Complete CPU-safe source and camera qualification result."""

    manifest: Mapping[str, Any]
    cameras: tuple[ImportedCamera, ...]
    reprojection: tuple[ReprojectionRecord, ...]
    scene_median_px: float
    scene_p95_px: float
    scene_maximum_px: float
    invalid_observations: int
    camera_outliers: tuple[str, ...]
    image_correspondence_failures: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            not self.image_correspondence_failures
            and self.invalid_observations == 0
            and self.scene_p95_px <= 4.0
            and self.scene_maximum_px <= 8.0
        )


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_identity(
    path: Path, *, expected_bytes: int, expected_sha256: str
) -> None:
    """Reject a downloaded artifact whose exact byte identity differs."""

    if expected_bytes <= 0 or len(expected_sha256) != 64:
        raise RealDataError("expected artifact identity is invalid")
    if path.stat().st_size != expected_bytes:
        raise RealDataError(f"artifact size mismatch: {path.name}")
    if sha256_file(path) != expected_sha256.lower():
        raise RealDataError(f"artifact SHA-256 mismatch: {path.name}")


def _read_exact(stream: BinaryIO, size: int, context: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise RealDataError(f"truncated COLMAP {context}")
    return payload


def _unpack(stream: BinaryIO, layout: str, context: str) -> tuple[Any, ...]:
    size = struct.calcsize(layout)
    return struct.unpack(layout, _read_exact(stream, size, context))


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    """Read COLMAP cameras.bin."""

    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as stream:
        count = cast(int, _unpack(stream, "<Q", "camera count")[0])
        if count <= 0 or count > 100_000:
            raise RealDataError("invalid COLMAP camera count")
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(
                stream, "<iiQQ", "camera record"
            )
            if model_id not in COLMAP_MODELS:
                raise RealDataError(f"unknown COLMAP model ID: {model_id}")
            model, parameter_count = COLMAP_MODELS[cast(int, model_id)]
            parameters = cast(
                tuple[float, ...],
                _unpack(stream, f"<{parameter_count}d", "camera parameters"),
            )
            camera = ColmapCamera(
                camera_id=cast(int, camera_id),
                model=model,
                width=cast(int, width),
                height=cast(int, height),
                parameters=parameters,
            )
            if camera.camera_id in cameras:
                raise RealDataError("duplicate COLMAP camera ID")
            cameras[camera.camera_id] = camera
        if stream.read(1):
            raise RealDataError("unexpected trailing bytes in cameras.bin")
    return cameras


def read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    """Read COLMAP cameras.txt."""

    cameras: dict[int, ColmapCamera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 5:
            raise RealDataError("invalid cameras.txt row")
        camera = ColmapCamera(
            camera_id=int(fields[0]),
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            parameters=tuple(float(value) for value in fields[4:]),
        )
        if camera.camera_id in cameras:
            raise RealDataError("duplicate COLMAP camera ID")
        cameras[camera.camera_id] = camera
    if not cameras:
        raise RealDataError("cameras.txt contains no cameras")
    return cameras


def _read_c_string(stream: BinaryIO) -> str:
    value = bytearray()
    while True:
        byte = _read_exact(stream, 1, "image name")
        if byte == b"\0":
            break
        value.extend(byte)
        if len(value) > 4096:
            raise RealDataError("COLMAP image name exceeds safety limit")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RealDataError("COLMAP image name is not UTF-8") from error


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    """Read COLMAP images.bin including sparse observations."""

    images: dict[int, ColmapImage] = {}
    names: set[str] = set()
    with path.open("rb") as stream:
        count = cast(int, _unpack(stream, "<Q", "image count")[0])
        if count <= 0 or count > 1_000_000:
            raise RealDataError("invalid COLMAP image count")
        for _ in range(count):
            values = _unpack(stream, "<i7di", "image record")
            image_id = cast(int, values[0])
            qvec = cast(tuple[float, float, float, float], tuple(values[1:5]))
            tvec = cast(tuple[float, float, float], tuple(values[5:8]))
            camera_id = cast(int, values[8])
            name = _read_c_string(stream)
            point_count = cast(int, _unpack(stream, "<Q", "observation count")[0])
            if point_count > 100_000_000:
                raise RealDataError("COLMAP observation count exceeds safety limit")
            observations = tuple(
                ColmapObservation(
                    *cast(
                        tuple[float, float, int], _unpack(stream, "<ddq", "observation")
                    )
                )
                for _ in range(point_count)
            )
            image = ColmapImage(
                image_id,
                qvec,
                tvec,
                camera_id,
                name,
                observations,
            )
            if image_id in images or name in names:
                raise RealDataError("duplicate COLMAP image ID or name")
            images[image_id] = image
            names.add(name)
        if stream.read(1):
            raise RealDataError("unexpected trailing bytes in images.bin")
    return images


def read_images_text(path: Path) -> dict[int, ColmapImage]:
    """Read paired-record COLMAP images.txt."""

    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) % 2:
        raise RealDataError("images.txt must contain paired records")
    images: dict[int, ColmapImage] = {}
    names: set[str] = set()
    for index in range(0, len(rows), 2):
        head = rows[index].split()
        if len(head) < 10:
            raise RealDataError("invalid images.txt image row")
        observations_raw = rows[index + 1].split()
        if len(observations_raw) % 3:
            raise RealDataError("invalid images.txt observation row")
        observations = tuple(
            ColmapObservation(
                float(observations_raw[offset]),
                float(observations_raw[offset + 1]),
                int(observations_raw[offset + 2]),
            )
            for offset in range(0, len(observations_raw), 3)
        )
        image = ColmapImage(
            image_id=int(head[0]),
            qvec_wxyz=cast(
                tuple[float, float, float, float],
                tuple(float(value) for value in head[1:5]),
            ),
            tvec=cast(
                tuple[float, float, float],
                tuple(float(value) for value in head[5:8]),
            ),
            camera_id=int(head[8]),
            name=" ".join(head[9:]),
            observations=observations,
        )
        if image.image_id in images or image.name in names:
            raise RealDataError("duplicate COLMAP image ID or name")
        images[image.image_id] = image
        names.add(image.name)
    if not images:
        raise RealDataError("images.txt contains no images")
    return images


def read_points_binary(path: Path) -> dict[int, ColmapPoint]:
    """Read COLMAP points3D.bin."""

    points: dict[int, ColmapPoint] = {}
    with path.open("rb") as stream:
        count = cast(int, _unpack(stream, "<Q", "point count")[0])
        if count <= 0 or count > 100_000_000:
            raise RealDataError("invalid COLMAP sparse point count")
        for _ in range(count):
            values = _unpack(stream, "<Q3d3Bd", "sparse point")
            point_id = cast(int, values[0])
            xyz = cast(tuple[float, float, float], tuple(values[1:4]))
            rgb = cast(tuple[int, int, int], tuple(values[4:7]))
            error = cast(float, values[7])
            track_length = cast(int, _unpack(stream, "<Q", "track length")[0])
            if track_length > 10_000_000:
                raise RealDataError("COLMAP track exceeds safety limit")
            track = tuple(
                cast(tuple[int, int], _unpack(stream, "<ii", "track element"))
                for _ in range(track_length)
            )
            point = ColmapPoint(point_id, xyz, rgb, error, track)
            if point_id in points:
                raise RealDataError("duplicate COLMAP point ID")
            points[point_id] = point
        if stream.read(1):
            raise RealDataError("unexpected trailing bytes in points3D.bin")
    return points


def read_points_text(path: Path) -> dict[int, ColmapPoint]:
    """Read COLMAP points3D.txt."""

    points: dict[int, ColmapPoint] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            raise RealDataError("invalid points3D.txt row")
        point = ColmapPoint(
            point3d_id=int(fields[0]),
            xyz=(float(fields[1]), float(fields[2]), float(fields[3])),
            rgb=(int(fields[4]), int(fields[5]), int(fields[6])),
            error=float(fields[7]),
            track=tuple(
                (int(fields[offset]), int(fields[offset + 1]))
                for offset in range(8, len(fields), 2)
            ),
        )
        if point.point3d_id in points:
            raise RealDataError("duplicate COLMAP point ID")
        points[point.point3d_id] = point
    if not points:
        raise RealDataError("points3D.txt contains no points")
    return points


def read_colmap_model(
    sparse_root: Path,
) -> tuple[dict[int, ColmapCamera], dict[int, ColmapImage], dict[int, ColmapPoint]]:
    """Read a consistent binary model, falling back to text only as a set."""

    binaries = tuple(
        sparse_root / name
        for name in (
            "cameras.bin",
            "images.bin",
            "points3D.bin",
        )
    )
    texts = tuple(
        sparse_root / name
        for name in (
            "cameras.txt",
            "images.txt",
            "points3D.txt",
        )
    )
    if all(path.is_file() for path in binaries):
        return (
            read_cameras_binary(binaries[0]),
            read_images_binary(binaries[1]),
            read_points_binary(binaries[2]),
        )
    if all(path.is_file() for path in texts):
        return (
            read_cameras_text(texts[0]),
            read_images_text(texts[1]),
            read_points_text(texts[2]),
        )
    raise RealDataError("COLMAP model requires a complete binary or text triplet")


def qvec_to_rotation(qvec: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """Convert normalized scalar-first COLMAP quaternion to world-to-camera R."""

    if len(qvec) != 4:
        raise RealDataError("quaternion must contain four values")
    w, x, y, z = (float(value) for value in qvec)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise RealDataError("quaternion norm must be positive")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ),
        (
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ),
        (
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
    )


def world_from_camera(image: ColmapImage) -> tuple[float, ...]:
    """Invert COLMAP world-to-camera into row-major OpenCV camera-to-world."""

    rotation = qvec_to_rotation(image.qvec_wxyz)
    inverse = tuple(
        tuple(rotation[column][row] for column in range(3)) for row in range(3)
    )
    center = tuple(
        -sum(inverse[row][column] * image.tvec[column] for column in range(3))
        for row in range(3)
    )
    matrix = (
        inverse[0][0],
        inverse[0][1],
        inverse[0][2],
        center[0],
        inverse[1][0],
        inverse[1][1],
        inverse[1][2],
        center[1],
        inverse[2][0],
        inverse[2][1],
        inverse[2][2],
        center[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )
    validate_rigid_world_from_camera(matrix)
    return matrix


def validate_rigid_world_from_camera(matrix: Sequence[float]) -> None:
    """Validate a row-major proper rigid pose."""

    if len(matrix) != 16 or not all(math.isfinite(value) for value in matrix):
        raise RealDataError("world_from_camera must contain 16 finite values")
    if any(
        abs(matrix[index] - expected) > 1e-10
        for index, expected in zip((12, 13, 14, 15), (0, 0, 0, 1), strict=True)
    ):
        raise RealDataError("invalid homogeneous bottom row")
    rotation = tuple(
        tuple(matrix[row * 4 + column] for column in range(3)) for row in range(3)
    )
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > 1e-10:
                raise RealDataError("camera rotation is not orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-10:
        raise RealDataError("camera rotation determinant is not +1")


def _distort(camera: ColmapCamera, x: float, y: float) -> tuple[float, float]:
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return x, y
    radius2 = x * x + y * y
    if camera.model == "SIMPLE_RADIAL":
        radial = 1 + camera.distortion[0] * radius2
        return x * radial, y * radial
    if camera.model == "RADIAL":
        k1, k2 = camera.distortion
        radial = 1 + k1 * radius2 + k2 * radius2 * radius2
        return x * radial, y * radial
    raise RealDataError(
        f"reprojection is not implemented for distortion model {camera.model}"
    )


def project_point(
    camera: ColmapCamera, image: ColmapImage, xyz: Sequence[float]
) -> tuple[float, float] | None:
    """Project one world-space point through a COLMAP camera."""

    rotation = qvec_to_rotation(image.qvec_wxyz)
    camera_xyz = tuple(
        sum(rotation[row][column] * xyz[column] for column in range(3))
        + image.tvec[row]
        for row in range(3)
    )
    if camera_xyz[2] <= 1e-12:
        return None
    normalized = _distort(
        camera, camera_xyz[0] / camera_xyz[2], camera_xyz[1] / camera_xyz[2]
    )
    fx, fy, cx, cy = camera.intrinsics
    return fx * normalized[0] + cx, fy * normalized[1] + cy


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise RealDataError("cannot compute an empty percentile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _portable_camera(
    image: ColmapImage,
    camera: ColmapCamera,
    split: str,
    image_dimensions: tuple[int, int],
) -> ImportedCamera:
    matrix = world_from_camera(image)
    image_width, image_height = image_dimensions
    scale_x = image_width / camera.width
    scale_y = image_height / camera.height
    identity: Any = {
        "source_image_id": image.image_id,
        "source_camera_id": image.camera_id,
        "image_name": image.name,
        "model": camera.model,
        "calibration_dimensions": [camera.width, camera.height],
        "image_dimensions": [image_width, image_height],
        "image_scale": [scale_x, scale_y],
        "parameters": list(camera.parameters),
        "world_from_camera": list(matrix),
        "coordinate_convention": COORDINATE_CONVENTION,
    }
    fx, fy, cx, cy = camera.intrinsics
    stem = Path(image.name).stem
    return ImportedCamera(
        camera_id=f"cam-{split}-{stem}",
        source_image_id=image.image_id,
        source_camera_id=image.camera_id,
        image_name=image.name,
        width=image_width,
        height=image_height,
        calibration_width=camera.width,
        calibration_height=camera.height,
        image_scale_x=scale_x,
        image_scale_y=scale_y,
        model=camera.model,
        fx=fx * scale_x,
        fy=fy * scale_y,
        cx=cx * scale_x,
        cy=cy * scale_y,
        distortion=camera.distortion,
        world_from_camera=matrix,
        digest=content_digest(identity),
    )


def official_split(names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Reproduce GRAPHDECO's sorted LLFF hold-every-eighth evaluation split."""

    ordered = tuple(sorted(names))
    if len(ordered) < 16:
        raise RealDataError("Truck split requires at least 16 cameras")
    evaluation = tuple(name for index, name in enumerate(ordered) if index % 8 == 0)
    training = tuple(name for index, name in enumerate(ordered) if index % 8 != 0)
    if set(evaluation) & set(training) or len(evaluation) < 8:
        raise RealDataError("invalid deterministic evaluation split")
    return {"training": training, "evaluation": evaluation}


def qualify_dataset(source_root: Path) -> DatasetQualification:
    """Validate official Truck images, cameras, and sparse reprojection."""

    images_root = source_root / "images"
    sparse_root = source_root / "sparse" / "0"
    cameras_raw, images_raw, points = read_colmap_model(sparse_root)
    correspondence: list[str] = []
    names = [image.name for image in images_raw.values()]
    split = official_split(names)
    evaluation = set(split["evaluation"])
    imported: list[ImportedCamera] = []
    errors_by_image: dict[int, list[float]] = {}
    invalid_by_image: dict[int, int] = {}
    all_errors: list[float] = []
    actual_dimensions: set[tuple[int, int]] = set()
    for image in sorted(images_raw.values(), key=lambda value: value.name):
        path = images_root / image.name
        if not path.is_file():
            correspondence.append(f"missing image: {image.name}")
            continue
        try:
            from PIL import Image

            with Image.open(path) as source_image:
                image_dimensions = source_image.size
                source_image.verify()
        except (ImportError, OSError, ValueError) as error:
            correspondence.append(f"invalid image {image.name}: {error}")
            continue
        actual_dimensions.add(image_dimensions)
        camera = cameras_raw.get(image.camera_id)
        if camera is None:
            correspondence.append(f"missing camera {image.camera_id}: {image.name}")
            continue
        imported.append(
            _portable_camera(
                image,
                camera,
                "eval" if image.name in evaluation else "train",
                image_dimensions,
            )
        )
        image_errors: list[float] = []
        invalid = 0
        for observation in image.observations:
            if observation.point3d_id < 0:
                continue
            point = points.get(observation.point3d_id)
            if point is None:
                invalid += 1
                continue
            projected = project_point(camera, image, point.xyz)
            if projected is None:
                invalid += 1
                continue
            reprojection_error = math.hypot(
                projected[0] - observation.x, projected[1] - observation.y
            )
            if not math.isfinite(reprojection_error):
                invalid += 1
                continue
            image_errors.append(reprojection_error)
            all_errors.append(reprojection_error)
        errors_by_image[image.image_id] = image_errors
        invalid_by_image[image.image_id] = invalid
    if len(imported) != len(images_raw):
        correspondence.append("not every image produced one imported camera")
    if not all_errors:
        raise RealDataError("COLMAP model contains no valid reprojection observations")
    records: list[ReprojectionRecord] = []
    outliers: list[str] = []
    by_source_id = {camera.source_image_id: camera for camera in imported}
    for image in sorted(images_raw.values(), key=lambda value: value.name):
        values = errors_by_image.get(image.image_id, [])
        imported_camera = by_source_id.get(image.image_id)
        camera_id = (
            imported_camera.camera_id
            if imported_camera is not None
            else f"missing-{image.image_id}"
        )
        record = ReprojectionRecord(
            camera_id=camera_id,
            valid_observations=len(values),
            invalid_observations=invalid_by_image.get(image.image_id, 0),
            median_px=statistics.median(values) if values else None,
            p95_px=_percentile(values, 0.95) if values else None,
            maximum_px=max(values) if values else None,
        )
        if record.p95_px is None or record.p95_px > 4.0:
            outliers.append(camera_id)
        records.append(record)
    image_files = (
        tuple(path for path in images_root.iterdir() if path.is_file())
        if images_root.is_dir()
        else ()
    )
    declared_names = set(names)
    extra = sorted(path.name for path in image_files if path.name not in declared_names)
    correspondence.extend(f"unmatched image file: {name}" for name in extra)
    camera_models = sorted({camera.model for camera in cameras_raw.values()})
    calibration_dimensions = sorted(
        {(camera.width, camera.height) for camera in cameras_raw.values()}
    )
    asset_digests = {
        name: sha256_file(sparse_root / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    image_digest = content_digest(
        [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(image_files)
        ]
    )
    manifest: dict[str, Any] = {
        "qualification_version": QUALIFICATION_VERSION,
        "dataset_id": DATASET_ID,
        "scene_id": SCENE_ID,
        "source_type": "real",
        "source_package": {
            "authority": "GRAPHDECO/Inria",
            "url": (
                "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
                "datasets/input/tandt_db.zip"
            ),
            "archive_bytes": SOURCE_ARCHIVE_BYTES,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "access_date": "2026-07-29",
            "classification": "official_restricted_local_only",
        },
        "portable_paths": {
            "image_directory": "source/images",
            "sparse_model_directory": "source/sparse/0",
        },
        "camera_count": len(imported),
        "image_count": len(image_files),
        "image_dimensions": [list(value) for value in sorted(actual_dimensions)],
        "calibration_dimensions": [list(value) for value in calibration_dimensions],
        "image_intrinsics_policy": (
            "scale fx,cx by image_width/calibration_width and fy,cy by "
            "image_height/calibration_height, matching GRAPHDECO camera loading"
        ),
        "camera_models": camera_models,
        "distortion_models": [
            model
            for model in camera_models
            if model not in {"PINHOLE", "SIMPLE_PINHOLE"}
        ],
        "split": {
            "policy": "graphdeco_eval_sorted_llffhold_8",
            "training_names": list(split["training"]),
            "evaluation_names": list(split["evaluation"]),
        },
        "coordinate_convention": COORDINATE_CONVENTION,
        "scene_normalization": (
            "source COLMAP world coordinates; pretrained Gaussians retain source frame"
        ),
        "local_data_classification": "restricted_local_only_not_for_git",
        "attribution": (
            "Tanks and Temples; Knapitsch et al. 2017; "
            "3D Gaussian Splatting; Kerbl et al. 2023"
        ),
        "license_status": "ambiguous_not_cleared_for_release",
        "sparse_model_digests": asset_digests,
        "image_set_digest": image_digest,
        "camera_digest": content_digest([asdict(camera) for camera in imported]),
        "limitations": [
            "upstream pixel and derived-render reuse remains blocked",
            "real-scene evidence is supplemental development evidence",
        ],
    }
    manifest["scientific_digest"] = content_digest(manifest)
    return DatasetQualification(
        manifest=manifest,
        cameras=tuple(imported),
        reprojection=tuple(records),
        scene_median_px=statistics.median(all_errors),
        scene_p95_px=_percentile(all_errors, 0.95),
        scene_maximum_px=max(all_errors),
        invalid_observations=sum(invalid_by_image.values()),
        camera_outliers=tuple(outliers),
        image_correspondence_failures=tuple(correspondence),
    )


def validate_dataset_root(root: Path) -> Path:
    """Enforce the approved ignored WSL-ext4 Truck root."""

    text = root.as_posix()
    if not root.is_absolute() or not text.startswith("/home/"):
        raise RealDataError("dataset root must be an absolute WSL /home path")
    if text.startswith("/mnt/") or ":\\" in str(root):
        raise RealDataError("dataset root must not use a Windows mount")
    if ".." in PurePosixPath(text).parts:
        raise RealDataError("dataset root traversal is not allowed")
    if root.is_symlink():
        raise RealDataError("dataset root may not be a symlink")
    resolved = root.resolve(strict=True)
    if resolved != Path(ALLOWED_ROOT.as_posix()):
        raise RealDataError(f"dataset root must equal {ALLOWED_ROOT.as_posix()}")
    for name in (
        "source",
        "pretrained",
        "training",
        "converted",
        "renders",
        "annotations",
        "logs",
        "temporary",
    ):
        child = resolved / name
        if child.is_symlink():
            raise RealDataError(f"local subdirectory may not be a symlink: {name}")
    return resolved


def write_dataset_reports(
    qualification: DatasetQualification, output: Path, *, overwrite: bool = False
) -> None:
    """Write portable source and camera qualification records."""

    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    camera_report: Any = {
        "qualification_version": QUALIFICATION_VERSION,
        "valid": qualification.valid,
        "coordinate_convention": COORDINATE_CONVENTION,
        "camera_count": len(qualification.cameras),
        "cameras": [asdict(camera) for camera in qualification.cameras],
        "reprojection": {
            "scene_median_px": qualification.scene_median_px,
            "scene_p95_px": qualification.scene_p95_px,
            "scene_maximum_px": qualification.scene_maximum_px,
            "invalid_observations": qualification.invalid_observations,
            "camera_outliers": list(qualification.camera_outliers),
            "image_correspondence_failures": list(
                qualification.image_correspondence_failures
            ),
            "per_camera": [asdict(record) for record in qualification.reprojection],
        },
    }
    (output / "source-manifest.json").write_bytes(
        pretty_json_bytes(cast(Any, qualification.manifest))
    )
    (output / "camera-qualification.json").write_bytes(pretty_json_bytes(camera_report))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("source")
    validate.add_argument("--output")
    validate.add_argument("--overwrite", action="store_true")
    validate.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point for CPU-only Truck dataset validation."""

    options = _parser().parse_args(arguments)
    try:
        source = Path(options.source)
        root = validate_dataset_root(source.parent)
        if source.resolve() != root / "source":
            raise RealDataError("source must be the dataset root's source directory")
        qualification = qualify_dataset(source)
        if options.output:
            write_dataset_reports(
                qualification, Path(options.output), overwrite=options.overwrite
            )
        if not qualification.valid:
            raise RealDataError("Truck source camera qualification failed")
    except (FileNotFoundError, FileExistsError, OSError, RealDataError) as error:
        payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, sort_keys=True) if options.json else str(error))
        return 1
    summary: Any = {
        "valid": True,
        "dataset_id": DATASET_ID,
        "camera_count": len(qualification.cameras),
        "image_count": qualification.manifest["image_count"],
        "scene_median_reprojection_px": qualification.scene_median_px,
        "scene_p95_reprojection_px": qualification.scene_p95_px,
        "scene_maximum_reprojection_px": qualification.scene_maximum_px,
        "scientific_digest": qualification.manifest["scientific_digest"],
    }
    print(
        json.dumps(summary, indent=2, sort_keys=True)
        if options.json
        else (
            f"Truck source valid: {summary['camera_count']} cameras; "
            f"p95={summary['scene_p95_reprojection_px']:.3g}px"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
