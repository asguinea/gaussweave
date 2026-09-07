from __future__ import annotations

import math
from pathlib import Path

import pytest

from gaussweave.data.render_formats import (
    RenderFormatError,
    read_npy_f32,
    read_pfm,
    read_png,
    write_npy_f32,
    write_png_u16,
)


def test_float_pass_round_trip_preserves_orientation_and_invalid_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "depth.npy"
    values = [0.0, 1.25, 2.5, 3.75, 4.0, 5.5]
    write_npy_f32(path, width=3, height=2, channels=1, values=values)

    decoded = read_npy_f32(path)

    assert decoded.shape == (2, 3)
    assert decoded.dtype == "float32"
    assert list(decoded.values) == values
    assert decoded.value(0, 0) == 0.0
    assert decoded.value(1, 2) == 5.5


def test_world_normal_round_trip_is_float32_and_unit_length(tmp_path: Path) -> None:
    path = tmp_path / "normals.npy"
    values = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
    write_npy_f32(path, width=2, height=1, channels=3, values=values)

    decoded = read_npy_f32(path)

    assert decoded.shape == (1, 2, 3)
    assert tuple(decoded.values[:3]) == (0.0, 0.0, 0.0)
    assert math.isclose(
        math.sqrt(sum(float(value) ** 2 for value in decoded.values[3:6])),
        1.0,
    )


def test_uint16_mask_round_trip_preserves_exact_ids(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    values = [0, 1, 2, 255, 256, 65535]
    write_png_u16(path, width=3, height=2, values=values)

    decoded = read_png(path)

    assert decoded.shape == (2, 3)
    assert decoded.dtype == "uint16"
    assert list(decoded.values) == values


@pytest.mark.parametrize(
    "payload,error",
    [
        (b"not-pfm", "PFM"),
        (b"Pf\n1 1\n-1.0\n", "payload"),
    ],
)
def test_corrupt_float_pass_is_rejected(
    tmp_path: Path, payload: bytes, error: str
) -> None:
    path = tmp_path / "bad.pfm"
    path.write_bytes(payload)

    with pytest.raises(RenderFormatError, match=error):
        read_pfm(path)


def test_truncated_or_object_npy_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    write_npy_f32(path, width=2, height=1, channels=1, values=[1.0, 2.0])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(RenderFormatError, match="payload"):
        read_npy_f32(path)

    path.write_bytes(b"\x93NUMPY\x01\x00\x00\x00")
    with pytest.raises(RenderFormatError, match="header"):
        read_npy_f32(path)


def test_corrupt_mask_checksum_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    write_png_u16(path, width=1, height=1, values=[7])
    content = bytearray(path.read_bytes())
    content[-5] ^= 1
    path.write_bytes(content)

    with pytest.raises(RenderFormatError, match="checksum"):
        read_png(path)
