from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from gaussweave.data.real_truck import DatasetQualification, ImportedCamera
from gaussweave.gaussians.real_explicit import (
    EXPECTED_PROPERTIES,
    ExplicitGaussianError,
    convert_ply,
    inspect_ply,
    parse_ply_layout,
    validate_model_camera_compatibility,
    validate_official_pretrained,
)

torch = pytest.importorskip("torch")


def _ply(
    path: Path,
    *,
    properties: tuple[str, ...] = EXPECTED_PROPERTIES,
    valid_quaternion: bool = True,
) -> Path:
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 1",
        *(f"property float {name}" for name in properties),
        "end_header",
        "",
    ]
    values = [0.0] * len(properties)
    by_name = {name: index for index, name in enumerate(properties)}
    for name, value in {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "f_dc_0": 0.1,
        "f_dc_1": 0.2,
        "f_dc_2": 0.3,
        "opacity": 0.0,
        "scale_0": -1.0,
        "scale_1": -2.0,
        "scale_2": -3.0,
        "rot_0": 2.0 if valid_quaternion else 0.0,
    }.items():
        if name in by_name:
            values[by_name[name]] = value
    path.write_bytes(
        "\n".join(header).encode("ascii") + struct.pack(f"<{len(values)}f", *values)
    )
    return path


def test_t11_t12_ply_fields_and_conversion_semantics(tmp_path: Path) -> None:
    source = _ply(tmp_path / "model.ply")
    inspection = inspect_ply(source)
    assert inspection.valid
    assert inspection.gaussian_count == 1
    converted = convert_ply(source, tmp_path / "converted")
    assert converted["gaussian_count"] == 1
    assert converted["appearance"]["shape"] == [1, 16, 3]
    quaternions = torch.from_file(
        str(tmp_path / "converted" / "quaternions.f32"),
        size=4,
        dtype=torch.float32,
    )
    assert quaternions.tolist() == [1.0, 0.0, 0.0, 0.0]
    scales = torch.from_file(
        str(tmp_path / "converted" / "scales.f32"),
        size=3,
        dtype=torch.float32,
    )
    assert scales.tolist() == pytest.approx(
        [math_exp(-1.0), math_exp(-2.0), math_exp(-3.0)]
    )


def math_exp(value: float) -> float:
    import math

    return math.exp(value)


def test_t13_conversion_is_digest_deterministic(tmp_path: Path) -> None:
    source = _ply(tmp_path / "model.ply")
    first = convert_ply(source, tmp_path / "first")
    second = convert_ply(source, tmp_path / "second")
    assert first["model_digest"] == second["model_digest"]
    assert first["files"] == second["files"]


def test_t11_incorrect_quaternion_fails(tmp_path: Path) -> None:
    source = _ply(tmp_path / "zero-quaternion.ply", valid_quaternion=False)
    with pytest.raises(ExplicitGaussianError, match="invalid Gaussian fields"):
        inspect_ply(source)


def test_t11_unsupported_sh_layout_fails(tmp_path: Path) -> None:
    properties = tuple(name for name in EXPECTED_PROPERTIES if name != "f_rest_44")
    source = _ply(tmp_path / "unsupported.ply", properties=properties)
    with pytest.raises(ExplicitGaussianError, match="field layout"):
        parse_ply_layout(source)


def test_t10_unmatched_model_provenance_fails(tmp_path: Path) -> None:
    inspection = inspect_ply(_ply(tmp_path / "community-repack.ply"))
    with pytest.raises(ExplicitGaussianError, match="provenance mismatch"):
        validate_official_pretrained(inspection)


def test_t10_source_model_scene_mismatch_fails(tmp_path: Path) -> None:
    camera = ImportedCamera(
        camera_id="cam-eval-000001",
        source_image_id=1,
        source_camera_id=1,
        image_name="000001.jpg",
        width=979,
        height=546,
        calibration_width=1957,
        calibration_height=1091,
        image_scale_x=979 / 1957,
        image_scale_y=546 / 1091,
        model="PINHOLE",
        fx=581.5,
        fy=578.0,
        cx=489.5,
        cy=273.0,
        distortion=(),
        world_from_camera=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        digest="sha256:test",
    )
    qualification = DatasetQualification(
        manifest={},
        cameras=(camera,),
        reprojection=(),
        scene_median_px=0.0,
        scene_p95_px=0.0,
        scene_maximum_px=0.0,
        invalid_observations=0,
        camera_outliers=(),
        image_correspondence_failures=(),
    )
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
        json.dumps(
            [
                {
                    "img_name": "different-scene",
                    "width": 1957,
                    "height": 1091,
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "position": [0, 0, 0],
                    "fx": 1163,
                    "fy": 1156,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExplicitGaussianError, match="image sets differ"):
        validate_model_camera_compatibility(cameras, qualification)
