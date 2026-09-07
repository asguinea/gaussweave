"""Official GRAPHDECO PLY inspection, conversion, and gsplat rendering."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.data.real_truck import (
    DATASET_ID,
    QUALIFICATION_VERSION,
    DatasetQualification,
    RealDataError,
    sha256_file,
)
from gaussweave.rendering.gsplat_renderer import GsplatRenderer, write_smoke_outputs
from gaussweave.rendering.models import Camera, RenderSettings

EXPECTED_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
MODEL_SOURCE_URL = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
    "datasets/pretrained/models.zip"
)
MODEL_ITERATION = 30_000
MODEL_SHA256 = "65ecf4058135a030cddd2198326f67172a4101344b0b54a3fa370cf45ea9688c"
MODEL_BYTES = 630_225_580
MODEL_GAUSSIANS = 2_541_226


class ExplicitGaussianError(RuntimeError):
    """An explicit Gaussian artifact is invalid or incompatible."""


def _safetensors_layout(path: Path) -> tuple[int, Mapping[str, Any]]:
    """Read and validate the non-executable safetensors JSON header."""

    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ExplicitGaussianError("safetensors header is truncated")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 0 or header_length > 16 * 1024 * 1024:
            raise ExplicitGaussianError("safetensors header length is unsafe")
        header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ExplicitGaussianError("safetensors JSON header is truncated")
    try:
        header = json.loads(header_bytes.decode("utf-8").rstrip())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExplicitGaussianError("safetensors JSON header is invalid") from error
    if not isinstance(header, dict):
        raise ExplicitGaussianError("safetensors header must be an object")
    return 8 + header_length, cast(Mapping[str, Any], header)


def _safetensors_field(
    header: Mapping[str, Any],
    name: str,
    *,
    expected_shape: tuple[int, ...],
) -> tuple[int, int]:
    raw = header.get(name)
    if not isinstance(raw, dict):
        raise ExplicitGaussianError(f"safetensors field is missing: {name}")
    if raw.get("dtype") != "F32":
        raise ExplicitGaussianError(f"safetensors field must be F32: {name}")
    shape = raw.get("shape")
    offsets = raw.get("data_offsets")
    if shape != list(expected_shape) or not (
        isinstance(offsets, list)
        and len(offsets) == 2
        and all(isinstance(value, int) for value in offsets)
    ):
        raise ExplicitGaussianError(f"safetensors field layout mismatch: {name}")
    begin, end = int(offsets[0]), int(offsets[1])
    if begin < 0 or end - begin != math.prod(expected_shape) * 4:
        raise ExplicitGaussianError(f"safetensors field byte range mismatch: {name}")
    return begin, end


def convert_safetensors(
    source: Path,
    output: Path,
    *,
    representation_id: str,
    scene_id: str,
    qualification_version: str,
    source_authority: str,
    source_url: str,
    iteration: int,
    overwrite: bool = False,
    chunk_size: int = 100_000,
) -> Mapping[str, Any]:
    """Convert official GRay tensors into project semantic float32 arrays.

    GRay's official ``convert/to_3dgs.py`` copies the stored mean, rotation,
    scale, opacity, and SH tensors into a 3DGS PLY.  This implementation applies
    the same field mapping directly into the project's existing semantic raw
    arrays, avoiding an unnecessary intermediate PLY while preserving source
    row order.
    """

    if chunk_size <= 0:
        raise ExplicitGaussianError("conversion chunk size must be positive")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_offset, header = _safetensors_layout(source)
    mean_record = header.get("mean")
    if not isinstance(mean_record, dict):
        raise ExplicitGaussianError("safetensors mean field is missing")
    mean_shape = mean_record.get("shape")
    if not (
        isinstance(mean_shape, list)
        and len(mean_shape) == 2
        and isinstance(mean_shape[0], int)
        and mean_shape[0] > 0
        and mean_shape[0] <= 10_000_000
        and mean_shape[1] == 3
    ):
        raise ExplicitGaussianError("safetensors mean shape is invalid")
    count = int(mean_shape[0])
    source_fields = {
        "mean": ((count, 3), 3),
        "rotation": ((count, 4), 4),
        "scale": ((count, 3), 3),
        "opacity": ((count, 1), 1),
        "sh_coeffs_dc": ((count, 1, 3), 3),
        "sh_coeffs_rest": ((count, 15, 3), 45),
    }
    ranges = {
        name: _safetensors_field(header, name, expected_shape=shape)
        for name, (shape, _width) in source_fields.items()
    }
    current_degree = header.get("current_sh_degree")
    if not isinstance(current_degree, dict) or current_degree.get("shape") != [1]:
        raise ExplicitGaussianError("safetensors SH degree field is invalid")

    fields: dict[str, tuple[str, tuple[int, ...]]] = {
        "means": ("means.f32", (count, 3)),
        "quaternions": ("quaternions.f32", (count, 4)),
        "scales": ("scales.f32", (count, 3)),
        "opacities": ("opacities.f32", (count,)),
        "sh_coefficients": ("sh-coefficients.f32", (count, 16, 3)),
    }
    temporary_paths = {
        name: output / f".{filename}.part"
        for name, (filename, _shape) in fields.items()
    }
    final_paths = {
        name: output / filename for name, (filename, _shape) in fields.items()
    }
    torch = _torch()
    started = time.perf_counter()
    round_trip = {
        "log_scale_max_abs": 0.0,
        "opacity_activation_max_abs": 0.0,
        "quaternion_norm_max_abs": 0.0,
    }

    def read_rows(stream: BinaryIO, name: str, start: int, rows: int) -> Any:
        _shape, width = source_fields[name]
        begin, _end = ranges[name]
        stream.seek(data_offset + begin + start * width * 4)
        payload = stream.read(rows * width * 4)
        if len(payload) != rows * width * 4:
            raise ExplicitGaussianError(f"safetensors field is truncated: {name}")
        return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(
            rows, width
        )

    try:
        streams = {name: path.open("wb") for name, path in temporary_paths.items()}
        with source.open("rb") as source_stream:
            try:
                for start in range(0, count, chunk_size):
                    rows = min(chunk_size, count - start)
                    means = read_rows(source_stream, "mean", start, rows)
                    quaternions = read_rows(source_stream, "rotation", start, rows)
                    norms = torch.linalg.vector_norm(quaternions, dim=1, keepdim=True)
                    if bool((norms <= 1e-12).any().item()):
                        raise ExplicitGaussianError("source contains a zero quaternion")
                    quaternions = quaternions / norms
                    raw_scales = read_rows(source_stream, "scale", start, rows)
                    scales = torch.exp(raw_scales)
                    raw_opacities = read_rows(
                        source_stream, "opacity", start, rows
                    ).reshape(rows)
                    opacities = torch.sigmoid(raw_opacities)
                    dc = read_rows(source_stream, "sh_coeffs_dc", start, rows).reshape(
                        rows, 1, 3
                    )
                    rest = read_rows(
                        source_stream, "sh_coeffs_rest", start, rows
                    ).reshape(rows, 15, 3)
                    sh = torch.cat((dc, rest), dim=1)
                    _write_tensor(streams["means"], means)
                    _write_tensor(streams["quaternions"], quaternions)
                    _write_tensor(streams["scales"], scales)
                    _write_tensor(streams["opacities"], opacities)
                    _write_tensor(streams["sh_coefficients"], sh)
                    round_trip["log_scale_max_abs"] = max(
                        round_trip["log_scale_max_abs"],
                        float((torch.log(scales) - raw_scales).abs().max().item()),
                    )
                    round_trip["opacity_activation_max_abs"] = max(
                        round_trip["opacity_activation_max_abs"],
                        float(
                            (opacities - torch.sigmoid(raw_opacities))
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    round_trip["quaternion_norm_max_abs"] = max(
                        round_trip["quaternion_norm_max_abs"],
                        float(
                            (torch.linalg.vector_norm(quaternions, dim=1) - 1)
                            .abs()
                            .max()
                            .item()
                        ),
                    )
            finally:
                for stream in streams.values():
                    stream.close()
        for name, temporary in temporary_paths.items():
            expected_values = math.prod(fields[name][1])
            if temporary.stat().st_size != expected_values * 4:
                raise ExplicitGaussianError(f"converted field size mismatch: {name}")
            os.replace(temporary, final_paths[name])
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
    file_records = {
        name: {
            "path": path.name,
            "shape": list(fields[name][1]),
            "dtype": "little_endian_float32",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in final_paths.items()
    }
    source_sha256 = sha256_file(source)
    metadata: dict[str, Any] = {
        "format_version": "gaussweave-explicit-gaussians-v1",
        "qualification_version": qualification_version,
        "representation_id": representation_id,
        "scene_id": scene_id,
        "source_checkpoint": {
            "authority": source_authority,
            "url": source_url,
            "iteration": iteration,
            "file_bytes": source.stat().st_size,
            "sha256": source_sha256,
            "gaussian_count": count,
            "sh_degree": 3,
            "format": "safetensors",
            "conversion_reference": (
                "graphdeco-inria/gray convert/to_3dgs.py and "
                "convert/safetensors_ply_conversion.py"
            ),
        },
        "gaussian_count": count,
        "coordinate_frame": "official model world coordinates",
        "ordering_policy": "source_safetensors_row_order_preserved",
        "stable_id_policy": f"g-{source_sha256[:12]}-<zero-padded-source-row>",
        "quaternion_convention": "normalized scalar-first wxyz",
        "scale_convention": "positive activated standard deviations",
        "opacity_convention": "activated alpha in [0,1]",
        "appearance": {
            "mode": "real_spherical_harmonics",
            "degree": 3,
            "shape": [count, 16, 3],
            "basis": "GRAPHDECO real SH",
            "ordering": "coefficient-major then RGB",
            "color_activation": "SH evaluation followed by renderer clamp",
        },
        "files": file_records,
        "conversion": {
            "precision": "float32 semantic activation",
            "means": "bit-preserving float32 copy",
            "quaternions": "source wxyz normalized",
            "scales": "exp(source log scales)",
            "opacities": "sigmoid(source logits)",
            "sh": "source coefficient-major tensors retained as [N,16,3]",
            "official_parameter_conversion": "lossless",
            "renderer_compatibility_limitation": (
                "GRay documents different kernel, sorting, and perspective "
                "semantics; 3DGS rendering can be blurrier"
            ),
            "round_trip_max_abs": round_trip,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "model_digest": content_digest(
            cast(
                Any,
                {
                    "source_sha256": source_sha256,
                    "files": file_records,
                    "semantic_conventions": {
                        "quaternion": "normalized_wxyz",
                        "scale": "positive_linear",
                        "opacity": "direct_alpha",
                        "sh": "degree_3_coefficient_major_rgb",
                    },
                },
            )
        ),
    }
    (output / "metadata.json").write_bytes(pretty_json_bytes(metadata))
    return metadata


@dataclass(frozen=True)
class PlyLayout:
    """Validated binary little-endian GRAPHDECO PLY layout."""

    vertex_count: int
    properties: tuple[str, ...]
    data_offset: int
    row_size: int

    def __post_init__(self) -> None:
        if self.vertex_count <= 0 or self.vertex_count > 10_000_000:
            raise ExplicitGaussianError("PLY vertex count exceeds the safety policy")
        if self.properties != EXPECTED_PROPERTIES:
            raise ExplicitGaussianError("unsupported GRAPHDECO PLY field layout")
        if self.row_size != 4 * len(self.properties):
            raise ExplicitGaussianError("PLY row size is inconsistent")


@dataclass(frozen=True)
class ExplicitModelInspection:
    """Portable inspection record for one source Gaussian PLY."""

    source_implementation: str
    source_url: str
    scene_id: str
    iteration: int
    file_bytes: int
    sha256: str
    gaussian_count: int
    sh_degree: int
    properties: tuple[str, ...]
    means_finite: bool
    stored_scales_finite: bool
    stored_opacity_finite: bool
    sh_finite: bool
    rotation_finite: bool
    quaternion_norm_min: float
    quaternion_norm_max: float
    activated_scale_min: float
    activated_scale_max: float
    activated_opacity_min: float
    activated_opacity_max: float
    coordinate_frame: str
    scale_activation: str
    opacity_activation: str
    quaternion_convention: str
    sh_layout: str

    @property
    def valid(self) -> bool:
        return (
            self.gaussian_count > 0
            and all(
                (
                    self.means_finite,
                    self.stored_scales_finite,
                    self.stored_opacity_finite,
                    self.sh_finite,
                    self.rotation_finite,
                )
            )
            and self.quaternion_norm_min > 1e-12
            and self.activated_scale_min > 0
            and 0 <= self.activated_opacity_min <= self.activated_opacity_max <= 1
        )


def _read_ply_header(stream: BinaryIO) -> tuple[list[str], int]:
    lines: list[str] = []
    consumed = 0
    while consumed < 1024 * 1024:
        raw = stream.readline()
        if not raw:
            raise ExplicitGaussianError("truncated PLY header")
        consumed += len(raw)
        try:
            line = raw.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ExplicitGaussianError("PLY header is not ASCII") from error
        lines.append(line)
        if line == "end_header":
            return lines, consumed
    raise ExplicitGaussianError("PLY header exceeds the 1 MiB safety limit")


def parse_ply_layout(path: Path) -> PlyLayout:
    """Parse and validate the exact official GRAPHDECO PLY schema."""

    with path.open("rb") as stream:
        lines, data_offset = _read_ply_header(stream)
    if not lines or lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
        raise ExplicitGaussianError("PLY must be binary_little_endian 1.0")
    vertex_count: int | None = None
    properties: list[str] = []
    in_vertex = False
    for line in lines:
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            if len(fields) != 3:
                raise ExplicitGaussianError("invalid PLY vertex element")
            vertex_count = int(fields[2])
            in_vertex = True
        elif fields[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and fields[:2] == ["property", "float"]:
            if len(fields) != 3:
                raise ExplicitGaussianError("invalid PLY float property")
            properties.append(fields[2])
        elif in_vertex and fields[:1] == ["property"]:
            raise ExplicitGaussianError("only float32 vertex fields are supported")
    if vertex_count is None:
        raise ExplicitGaussianError("PLY has no vertex element")
    layout = PlyLayout(
        vertex_count, tuple(properties), data_offset, 4 * len(properties)
    )
    expected_size = layout.data_offset + layout.vertex_count * layout.row_size
    if path.stat().st_size != expected_size:
        actual_size = path.stat().st_size
        raise ExplicitGaussianError(
            f"PLY payload size mismatch: expected {expected_size}, got {actual_size}"
        )
    return layout


def _mapped_rows(path: Path, layout: PlyLayout) -> tuple[Any, mmap.mmap, BinaryIO]:
    torch = _torch()
    stream = path.open("rb")
    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    values = torch.frombuffer(
        mapping,
        dtype=torch.float32,
        count=layout.vertex_count * len(layout.properties),
        offset=layout.data_offset,
    )
    return values.reshape(layout.vertex_count, len(layout.properties)), mapping, stream


def inspect_ply(path: Path) -> ExplicitModelInspection:
    """Inspect every Gaussian field with vectorized CPU validation."""

    layout = parse_ply_layout(path)
    rows, mapping, stream = _mapped_rows(path, layout)
    torch = _torch()
    try:
        finite = torch.isfinite(rows)
        quaternions = rows[:, 58:62]
        norms = torch.linalg.vector_norm(quaternions, dim=1)
        scales = torch.exp(rows[:, 55:58])
        opacity = torch.sigmoid(rows[:, 54])
        inspection = ExplicitModelInspection(
            source_implementation="GRAPHDECO official gaussian-splatting release model",
            source_url=MODEL_SOURCE_URL,
            scene_id=DATASET_ID,
            iteration=MODEL_ITERATION,
            file_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            gaussian_count=layout.vertex_count,
            sh_degree=3,
            properties=layout.properties,
            means_finite=bool(finite[:, 0:3].all().item()),
            stored_scales_finite=bool(finite[:, 55:58].all().item()),
            stored_opacity_finite=bool(finite[:, 54].all().item()),
            sh_finite=bool(finite[:, 6:54].all().item()),
            rotation_finite=bool(finite[:, 58:62].all().item()),
            quaternion_norm_min=float(norms.min().item()),
            quaternion_norm_max=float(norms.max().item()),
            activated_scale_min=float(scales.min().item()),
            activated_scale_max=float(scales.max().item()),
            activated_opacity_min=float(opacity.min().item()),
            activated_opacity_max=float(opacity.max().item()),
            coordinate_frame="source COLMAP world coordinates",
            scale_activation="exp(stored_log_scale)",
            opacity_activation="sigmoid(stored_logit)",
            quaternion_convention="stored scalar-first wxyz; normalized before use",
            sh_layout=(
                "degree 3 real SH; source f_rest channel-major; "
                "converted coefficient-major [N,16,3]"
            ),
        )
    finally:
        del rows
        mapping.close()
        stream.close()
    if not inspection.valid:
        raise ExplicitGaussianError("source PLY contains invalid Gaussian fields")
    return inspection


def validate_model_camera_compatibility(
    cameras_path: Path, qualification: DatasetQualification
) -> Mapping[str, Any]:
    """Prove the official pretrained model uses the qualified source cameras."""

    raw = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ExplicitGaussianError("pretrained cameras.json must contain a list")
    model_records: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("img_name"), str):
            raise ExplicitGaussianError("invalid pretrained camera record")
        image_name = f"{item['img_name']}.jpg"
        if image_name in model_records:
            raise ExplicitGaussianError("duplicate pretrained camera image name")
        model_records[image_name] = cast(Mapping[str, Any], item)
    source_records = {camera.image_name: camera for camera in qualification.cameras}
    if set(model_records) != set(source_records):
        raise ExplicitGaussianError("source/model camera image sets differ")
    maximum_pose_error = 0.0
    maximum_intrinsic_error = 0.0
    for name, source in source_records.items():
        model = model_records[name]
        if (
            int(model["width"]) != source.calibration_width
            or int(model["height"]) != source.calibration_height
        ):
            raise ExplicitGaussianError(
                f"source/model camera dimensions differ: {name}"
            )
        rotation = cast(Sequence[Sequence[float]], model["rotation"])
        position = cast(Sequence[float], model["position"])
        if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            raise ExplicitGaussianError("pretrained camera rotation shape is invalid")
        if len(position) != 3:
            raise ExplicitGaussianError("pretrained camera position shape is invalid")
        expected_rotation = tuple(
            tuple(source.world_from_camera[row * 4 + column] for column in range(3))
            for row in range(3)
        )
        expected_position = tuple(
            source.world_from_camera[row * 4 + 3] for row in range(3)
        )
        maximum_pose_error = max(
            maximum_pose_error,
            max(
                abs(float(rotation[row][column]) - expected_rotation[row][column])
                for row in range(3)
                for column in range(3)
            ),
            max(
                abs(float(position[index]) - expected_position[index])
                for index in range(3)
            ),
        )
        maximum_intrinsic_error = max(
            maximum_intrinsic_error,
            abs(float(model["fx"]) - source.fx / source.image_scale_x),
            abs(float(model["fy"]) - source.fy / source.image_scale_y),
        )
    if maximum_pose_error > 1e-9 or maximum_intrinsic_error > 1e-9:
        raise ExplicitGaussianError("source/model cameras fail exact compatibility")
    result: dict[str, Any] = {
        "valid": True,
        "camera_count": len(model_records),
        "image_set_exact_match": True,
        "calibration_dimensions_exact_match": True,
        "maximum_pose_abs_error": maximum_pose_error,
        "maximum_intrinsic_abs_error": maximum_intrinsic_error,
        "compact_image_policy": (
            "qualified images are loaded at their actual dimensions while preserving "
            "the full-calibration field of view, matching GRAPHDECO loadCam"
        ),
        "cameras_json_bytes": cameras_path.stat().st_size,
        "cameras_json_sha256": sha256_file(cameras_path),
    }
    result["scientific_digest"] = content_digest(result)
    return result


def validate_official_pretrained(
    inspection: ExplicitModelInspection,
) -> ExplicitModelInspection:
    """Require the exact official Truck iteration-30000 artifact identity."""

    expected = {
        "scene_id": DATASET_ID,
        "iteration": MODEL_ITERATION,
        "file_bytes": MODEL_BYTES,
        "sha256": MODEL_SHA256,
        "gaussian_count": MODEL_GAUSSIANS,
        "sh_degree": 3,
    }
    mismatches = [
        name for name, value in expected.items() if getattr(inspection, name) != value
    ]
    if mismatches:
        raise ExplicitGaussianError(
            "official pretrained provenance mismatch: " + ", ".join(mismatches)
        )
    return inspection


def _write_tensor(stream: BinaryIO, tensor: Any) -> None:
    array = tensor.detach().contiguous().numpy()
    array.tofile(stream)


def convert_ply(
    source: Path,
    output: Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 100_000,
) -> Mapping[str, Any]:
    """Convert official stored parameters into project semantic float32 arrays."""

    if chunk_size <= 0:
        raise ExplicitGaussianError("conversion chunk size must be positive")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inspection = inspect_ply(source)
    layout = parse_ply_layout(source)
    fields: dict[str, tuple[str, tuple[int, ...]]] = {
        "means": ("means.f32", (layout.vertex_count, 3)),
        "quaternions": ("quaternions.f32", (layout.vertex_count, 4)),
        "scales": ("scales.f32", (layout.vertex_count, 3)),
        "opacities": ("opacities.f32", (layout.vertex_count,)),
        "sh_coefficients": ("sh-coefficients.f32", (layout.vertex_count, 16, 3)),
    }
    temporary_paths = {
        name: output / f".{filename}.part"
        for name, (filename, _shape) in fields.items()
    }
    final_paths = {
        name: output / filename for name, (filename, _shape) in fields.items()
    }
    rows, mapping, source_stream = _mapped_rows(source, layout)
    torch = _torch()
    started = time.perf_counter()
    round_trip = {
        "log_scale_max_abs": 0.0,
        "opacity_activation_max_abs": 0.0,
        "quaternion_norm_max_abs": 0.0,
    }
    try:
        streams = {name: path.open("wb") for name, path in temporary_paths.items()}
        try:
            for start in range(0, layout.vertex_count, chunk_size):
                raw = rows[start : min(start + chunk_size, layout.vertex_count)]
                means = raw[:, 0:3]
                quaternions = raw[:, 58:62]
                norms = torch.linalg.vector_norm(quaternions, dim=1, keepdim=True)
                if bool((norms <= 1e-12).any().item()):
                    raise ExplicitGaussianError("source contains a zero quaternion")
                quaternions = quaternions / norms
                scales = torch.exp(raw[:, 55:58])
                opacities = torch.sigmoid(raw[:, 54])
                dc = raw[:, 6:9].reshape(-1, 1, 3)
                rest = raw[:, 9:54].reshape(-1, 3, 15).transpose(1, 2)
                sh = torch.cat((dc, rest), dim=1)
                _write_tensor(streams["means"], means)
                _write_tensor(streams["quaternions"], quaternions)
                _write_tensor(streams["scales"], scales)
                _write_tensor(streams["opacities"], opacities)
                _write_tensor(streams["sh_coefficients"], sh)
                round_trip["log_scale_max_abs"] = max(
                    round_trip["log_scale_max_abs"],
                    float((torch.log(scales) - raw[:, 55:58]).abs().max().item()),
                )
                round_trip["opacity_activation_max_abs"] = max(
                    round_trip["opacity_activation_max_abs"],
                    float((opacities - torch.sigmoid(raw[:, 54])).abs().max().item()),
                )
                round_trip["quaternion_norm_max_abs"] = max(
                    round_trip["quaternion_norm_max_abs"],
                    float(
                        (torch.linalg.vector_norm(quaternions, dim=1) - 1)
                        .abs()
                        .max()
                        .item()
                    ),
                )
        finally:
            for file in streams.values():
                file.close()
        for name, temporary in temporary_paths.items():
            expected_values = math.prod(fields[name][1])
            if temporary.stat().st_size != expected_values * 4:
                raise ExplicitGaussianError(f"converted field size mismatch: {name}")
            os.replace(temporary, final_paths[name])
    finally:
        del rows
        mapping.close()
        source_stream.close()
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
    file_records = {
        name: {
            "path": path.name,
            "shape": fields[name][1],
            "dtype": "little_endian_float32",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in final_paths.items()
    }
    metadata: dict[str, Any] = {
        "format_version": "gaussweave-explicit-gaussians-v1",
        "qualification_version": QUALIFICATION_VERSION,
        "representation_id": "gw-truck-explicit-graphdeco-30000",
        "scene_id": DATASET_ID,
        "source_checkpoint": asdict(inspection),
        "gaussian_count": layout.vertex_count,
        "coordinate_frame": "source COLMAP world coordinates",
        "ordering_policy": "source_ply_row_order_preserved",
        "stable_id_policy": f"g-{inspection.sha256[:12]}-<zero-padded-source-row>",
        "quaternion_convention": "normalized scalar-first wxyz",
        "scale_convention": "positive activated standard deviations",
        "opacity_convention": "activated alpha in [0,1]",
        "appearance": {
            "mode": "real_spherical_harmonics",
            "degree": 3,
            "shape": [layout.vertex_count, 16, 3],
            "basis": "GRAPHDECO real SH",
            "ordering": "coefficient-major then RGB",
            "color_activation": "SH evaluation followed by renderer clamp",
        },
        "files": file_records,
        "conversion": {
            "precision": "float32 semantic activation",
            "means": "bit-preserving float32 copy",
            "quaternions": "source wxyz normalized",
            "scales": "exp(source log scales)",
            "opacities": "sigmoid(source logits)",
            "sh": "source channel-major rest transposed to [N,16,3]",
            "round_trip_max_abs": round_trip,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "model_digest": content_digest(
            cast(
                Any,
                {
                    "source_sha256": inspection.sha256,
                    "files": file_records,
                    "semantic_conventions": {
                        "quaternion": "normalized_wxyz",
                        "scale": "positive_linear",
                        "opacity": "direct_alpha",
                        "sh": "degree_3_coefficient_major_rgb",
                    },
                },
            )
        ),
    }
    (output / "metadata.json").write_bytes(pretty_json_bytes(metadata))
    return metadata


@dataclass(frozen=True)
class RealGaussianTensorSource:
    """Lazy project tensor source backed by converted raw float32 fields."""

    root: Path
    metadata: Mapping[str, Any]
    appearance_mode: str = "sh"
    sh_degree: int | None = 3
    color_activation: str = "spherical_harmonics"

    @classmethod
    def load(cls, root: Path) -> RealGaussianTensorSource:
        metadata_path = root / "metadata.json"
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ExplicitGaussianError(
                "invalid converted Gaussian metadata"
            ) from error
        if not isinstance(raw, dict):
            raise ExplicitGaussianError("converted Gaussian metadata must be an object")
        if raw.get("format_version") != "gaussweave-explicit-gaussians-v1":
            raise ExplicitGaussianError("unsupported converted Gaussian format")
        source = cls(root.resolve(), cast(Mapping[str, Any], raw))
        source.validate_files()
        return source

    @property
    def count(self) -> int:
        return int(self.metadata["gaussian_count"])

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return (self.count, 16, 3)

    def validate_files(self) -> None:
        files = cast(Mapping[str, Mapping[str, Any]], self.metadata["files"])
        for name in ("means", "quaternions", "scales", "opacities", "sh_coefficients"):
            record = files.get(name)
            if record is None:
                raise ExplicitGaussianError(f"missing converted field: {name}")
            path = self.root / str(record["path"])
            if path.parent != self.root or not path.is_file():
                raise ExplicitGaussianError(
                    f"unsafe or missing converted field: {name}"
                )
            if path.stat().st_size != int(record["bytes"]):
                raise ExplicitGaussianError(f"converted size mismatch: {name}")
            if sha256_file(path) != record["sha256"]:
                raise ExplicitGaussianError(f"converted checksum mismatch: {name}")

    def tensors(self, device: str = "cuda:0") -> dict[str, Any]:
        torch = _torch()
        files = cast(Mapping[str, Mapping[str, Any]], self.metadata["files"])

        def load(name: str, shape: tuple[int, ...]) -> Any:
            path = self.root / str(files[name]["path"])
            values = torch.from_file(
                str(path), shared=False, size=math.prod(shape), dtype=torch.float32
            )
            return values.reshape(shape).to(device)

        return {
            "means": load("means", (self.count, 3)),
            "quaternions": load("quaternions", (self.count, 4)),
            "scales": load("scales", (self.count, 3)),
            "opacities": load("opacities", (self.count,)),
            "appearance": load("sh_coefficients", (self.count, 16, 3)),
        }


def _camera_from_record(record: Mapping[str, Any]) -> Camera:
    matrix = tuple(
        float(value) for value in cast(Sequence[Any], record["world_from_camera"])
    )
    return Camera(
        camera_id=str(record["camera_id"]),
        width=int(record["width"]),
        height=int(record["height"]),
        fx=float(record["fx"]),
        fy=float(record["fy"]),
        cx=float(record["cx"]),
        cy=float(record["cy"]),
        world_from_camera=tuple(
            tuple(matrix[row * 4 + column] for column in range(4)) for row in range(4)
        ),
        near=0.01,
        far=1000.0,
    )


def render_views(
    converted: Path,
    split_path: Path,
    output: Path,
    *,
    source_images: Path | None = None,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Render every selected camera and record local exploratory fidelity."""

    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    split_raw = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split_raw, dict) or not isinstance(
        split_raw.get("cameras"), list
    ):
        raise ExplicitGaussianError("evaluation split has no camera records")
    records = cast(list[Mapping[str, Any]], split_raw["cameras"])
    if len(records) < 8:
        raise ExplicitGaussianError(
            "evaluation rendering requires at least eight cameras"
        )
    scene = RealGaussianTensorSource.load(converted)
    renderer = GsplatRenderer()
    observations: list[dict[str, Any]] = []
    for record in records:
        camera = _camera_from_record(record)
        result = renderer.render(
            scene,
            camera,
            RenderSettings(
                output_buffers=("rgb", "alpha", "depth"),
                camera_batch_size=1,
                warmup_count=0,
                repetition_count=1,
            ),
            profile_name="standard",
        )
        camera_output = output / camera.camera_id
        paths = write_smoke_outputs(result, camera_output, overwrite=overwrite)
        metric: Mapping[str, Any] | None = None
        if source_images is not None:
            metric = _metric_for_source(
                result.rgb[0], source_images / str(record["image_name"])
            )
        observations.append(
            {
                "camera_id": camera.camera_id,
                "image_name": record["image_name"],
                "width": camera.width,
                "height": camera.height,
                "gaussian_count": scene.count,
                "latency_seconds": result.elapsed_seconds,
                "peak_gpu_allocated_bytes": (
                    result.resource_record.gpu_final.peak_allocated_bytes
                ),
                "peak_gpu_reserved_bytes": (
                    result.resource_record.gpu_final.peak_reserved_bytes
                ),
                "resource_compliance": result.resource_record.compliance.state.value,
                "metrics": metric,
                "outputs": paths,
            }
        )
    summary: dict[str, Any] = {
        "qualification_version": QUALIFICATION_VERSION,
        "representation_id": scene.metadata["representation_id"],
        "model_digest": scene.metadata["model_digest"],
        "camera_count": len(observations),
        "observations": observations,
        "complete": len(observations) == len(records),
        "warnings": [
            "exploratory scene-specific metrics; not frozen evaluation evidence",
            "local-only renders; not cleared for release",
        ],
    }
    (output / "render-summary.json").write_bytes(pretty_json_bytes(summary))
    return summary


def _metric_for_source(rendered: Any, source_path: Path) -> Mapping[str, Any]:
    try:
        from PIL import Image
    except ImportError as error:
        raise ExplicitGaussianError("Pillow is required for source metrics") from error
    torch = _torch()
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (rendered.shape[1], rendered.shape[0]):
            raise ExplicitGaussianError(
                f"source/render dimensions differ: {source_path.name}"
            )
        values = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        values = values.reshape(rgb.height, rgb.width, 3).to(torch.float32) / 255.0
    prediction = rendered.detach().to("cpu")
    difference = prediction - values
    mse = float((difference * difference).mean().item())
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    return {
        "mse_rgb": mse,
        "psnr_rgb_db": "positive_infinity" if math.isinf(psnr) else psnr,
        "mean_absolute_rgb": float(difference.abs().mean().item()),
        "data_range": 1.0,
        "status": "exploratory_development",
    }


def _torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise ExplicitGaussianError(
            "PyTorch is required for Gaussian tensors"
        ) from error
    return torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path")
    inspect.add_argument("--json", action="store_true")
    convert = commands.add_parser("convert")
    convert.add_argument("path")
    convert.add_argument("--output", required=True)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--json", action="store_true")
    render = commands.add_parser("render")
    render.add_argument("path")
    render.add_argument("--cameras", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--source-images")
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    options = _parser().parse_args(arguments)
    try:
        if options.command == "inspect":
            result: Any = asdict(inspect_ply(Path(options.path)))
            result["valid"] = True
        elif options.command == "convert":
            result = dict(
                convert_ply(
                    Path(options.path),
                    Path(options.output),
                    overwrite=options.overwrite,
                )
            )
            result["valid"] = True
        else:
            result = dict(
                render_views(
                    Path(options.path),
                    Path(options.cameras),
                    Path(options.output),
                    source_images=(
                        Path(options.source_images) if options.source_images else None
                    ),
                    overwrite=options.overwrite,
                )
            )
            result["valid"] = True
    except (
        ExplicitGaussianError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RealDataError,
        ValueError,
    ) as error:
        payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, sort_keys=True) if options.json else str(error))
        return 1
    print(
        json.dumps(result, indent=2, sort_keys=True)
        if options.json
        else f"{options.command} passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
