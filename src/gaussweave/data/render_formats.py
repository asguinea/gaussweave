"""Small, dependency-free readers and writers for render-pass artifacts."""

from __future__ import annotations

import ast
import hashlib
import math
import struct
import zlib
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_DECODED_BYTES = 512 * 1024 * 1024
DecodedValues = array[int] | array[float]


class RenderFormatError(ValueError):
    """A render artifact is corrupt, unsupported, or exceeds safety bounds."""


@dataclass(frozen=True)
class DecodedArray:
    """Decoded top-left-origin array plus exact canonical pixel identity."""

    width: int
    height: int
    channels: int
    dtype: str
    values: DecodedValues
    decoded_digest: str

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            (self.height, self.width)
            if self.channels == 1
            else (self.height, self.width, self.channels)
        )

    def value(self, row: int, column: int, channel: int = 0) -> int | float:
        if not 0 <= row < self.height or not 0 <= column < self.width:
            raise IndexError("pixel coordinate is outside the decoded array")
        if not 0 <= channel < self.channels:
            raise IndexError("channel is outside the decoded array")
        return self.values[(row * self.width + column) * self.channels + channel]


def write_pfm(
    path: Path,
    *,
    width: int,
    height: int,
    channels: Literal[1, 3],
    values: list[float] | tuple[float, ...] | array[float],
) -> None:
    """Write top-left row-major float values as little-endian PFM."""

    _dimensions(width, height, channels, len(values), 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = b"Pf" if channels == 1 else b"PF"
    output = array("f")
    row_values = width * channels
    for row in range(height - 1, -1, -1):
        start = row * row_values
        output.extend(float(value) for value in values[start : start + row_values])
    if output.itemsize != 4:
        raise RenderFormatError("platform float array is not float32")
    if _native_byte_order() == "big":
        output.byteswap()
    path.write_bytes(
        kind + b"\n" + f"{width} {height}\n-1.0\n".encode("ascii") + output.tobytes()
    )


def write_npy_f32(
    path: Path,
    *,
    width: int,
    height: int,
    channels: Literal[1, 3],
    values: list[float] | tuple[float, ...] | array[float],
) -> None:
    """Write a safe C-order little-endian float32 NPY v1 array."""

    _dimensions(width, height, channels, len(values), 4)
    shape = (height, width) if channels == 1 else (height, width, channels)
    header = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape!r}, }}"
    ).encode("ascii")
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    output = array("f", (float(value) for value in values))
    if output.itemsize != 4:
        raise RenderFormatError("platform float array is not float32")
    if _native_byte_order() == "big":
        output.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(header))
        + header
        + output.tobytes()
    )


def read_npy_f32(path: Path) -> DecodedArray:
    """Read bounded C-order float32 NPY without enabling object/pickle data."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise RenderFormatError(f"could not read NPY: {error}") from error
    if len(content) < 10 or content[:8] != b"\x93NUMPY\x01\x00":
        raise RenderFormatError("unsupported NPY magic or version")
    header_size = struct.unpack("<H", content[8:10])[0]
    header_end = 10 + header_size
    if header_size <= 0 or header_end > len(content) or header_size > 4096:
        raise RenderFormatError("invalid or oversized NPY header")
    try:
        header = ast.literal_eval(content[10:header_end].decode("ascii").strip())
    except (UnicodeError, SyntaxError, ValueError) as error:
        raise RenderFormatError("invalid NPY header") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise RenderFormatError("unsupported NPY header fields")
    if header["descr"] not in {"<f4", "=f4"} or header["fortran_order"] is not False:
        raise RenderFormatError("NPY must be C-order little-endian float32")
    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or len(shape) not in {2, 3}
        or any(not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise RenderFormatError("unsupported NPY shape")
    height, width = shape[:2]
    channels = 1 if len(shape) == 2 else shape[2]
    if channels not in {1, 3}:
        raise RenderFormatError("NPY render pass must have one or three channels")
    expected = _checked_bytes(width, height, channels, 4)
    payload = content[header_end:]
    if len(payload) != expected:
        raise RenderFormatError(
            f"NPY payload has {len(payload)} bytes; expected {expected}"
        )
    values = array("f")
    values.frombytes(payload)
    if _native_byte_order() == "big":
        values.byteswap()
    canonical = _float32_little_bytes(values)
    return DecodedArray(
        width,
        height,
        channels,
        "float32",
        values,
        f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    )


def read_pfm(path: Path) -> DecodedArray:
    """Read one little- or big-endian PFM into top-left row-major float32."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise RenderFormatError(f"could not read PFM: {error}") from error
    position = 0
    magic, position = _pfm_line(content, position)
    channels = 3 if magic == b"PF" else 1 if magic == b"Pf" else 0
    if channels == 0:
        raise RenderFormatError("unsupported PFM magic")
    dimensions, position = _pfm_line(content, position)
    try:
        width_text, height_text = dimensions.split()
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as error:
        raise RenderFormatError("invalid PFM dimensions") from error
    scale_line, position = _pfm_line(content, position)
    try:
        scale = float(scale_line)
    except ValueError as error:
        raise RenderFormatError("invalid PFM scale") from error
    if not math.isfinite(scale) or scale == 0.0:
        raise RenderFormatError("PFM scale must be finite and nonzero")
    expected = _checked_bytes(width, height, channels, 4)
    payload = content[position:]
    if len(payload) != expected:
        raise RenderFormatError(
            f"PFM payload has {len(payload)} bytes; expected {expected}"
        )
    raw = array("f")
    raw.frombytes(payload)
    file_order = "little" if scale < 0.0 else "big"
    if file_order != _native_byte_order():
        raw.byteswap()
    result = array("f")
    row_values = width * channels
    magnitude = abs(scale)
    for row in range(height - 1, -1, -1):
        start = row * row_values
        result.extend(value * magnitude for value in raw[start : start + row_values])
    canonical = _float32_little_bytes(result)
    return DecodedArray(
        width,
        height,
        channels,
        "float32",
        result,
        f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    )


def write_png_u16(
    path: Path,
    *,
    width: int,
    height: int,
    values: list[int] | tuple[int, ...] | array[int],
) -> None:
    """Write one top-left-origin uint16 grayscale PNG with filter type zero."""

    _dimensions(width, height, 1, len(values), 2)
    rows = bytearray()
    for row in range(height):
        rows.append(0)
        start = row * width
        for value in values[start : start + width]:
            integer = int(value)
            if not 0 <= integer <= 65535:
                raise RenderFormatError("uint16 PNG value is out of range")
            rows.extend(struct.pack(">H", integer))
    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def read_png(path: Path) -> DecodedArray:
    """Decode non-interlaced grayscale/RGB/RGBA 8/16-bit PNG pixels."""

    try:
        content = path.read_bytes()
    except OSError as error:
        raise RenderFormatError(f"could not read PNG: {error}") from error
    if not content.startswith(PNG_SIGNATURE):
        raise RenderFormatError("invalid PNG signature")
    position = len(PNG_SIGNATURE)
    header: tuple[int, int, int, int] | None = None
    compressed = bytearray()
    seen_end = False
    while position < len(content):
        if position + 12 > len(content):
            raise RenderFormatError("truncated PNG chunk")
        length = struct.unpack(">I", content[position : position + 4])[0]
        kind = content[position + 4 : position + 8]
        payload_start = position + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(content):
            raise RenderFormatError("truncated PNG payload")
        payload = content[payload_start:payload_end]
        expected_crc = struct.unpack(">I", content[payload_end:crc_end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise RenderFormatError("PNG chunk checksum mismatch")
        if kind == b"IHDR":
            if header is not None or length != 13:
                raise RenderFormatError("invalid PNG header")
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if compression != 0 or filtering != 0 or interlace != 0:
                raise RenderFormatError("unsupported PNG compression/filter/interlace")
            header = (width, height, bit_depth, color_type)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            seen_end = True
            break
        position = crc_end
    if header is None or not seen_end or not compressed:
        raise RenderFormatError("PNG is missing required chunks")
    width, height, bit_depth, color_type = header
    channel_map = {0: 1, 2: 3, 4: 2, 6: 4}
    channels = channel_map.get(color_type)
    if channels is None or bit_depth not in {8, 16}:
        raise RenderFormatError("unsupported PNG color type or bit depth")
    bytes_per_sample = bit_depth // 8
    bytes_per_pixel = channels * bytes_per_sample
    row_bytes = _checked_bytes(width, 1, channels, bytes_per_sample)
    expected = (row_bytes + 1) * height
    try:
        filtered = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise RenderFormatError("invalid PNG compressed payload") from error
    if len(filtered) != expected:
        raise RenderFormatError(
            f"PNG payload has {len(filtered)} bytes; expected {expected}"
        )
    raw = bytearray()
    previous = bytes(row_bytes)
    cursor = 0
    for _ in range(height):
        filter_type = filtered[cursor]
        cursor += 1
        row = bytearray(filtered[cursor : cursor + row_bytes])
        cursor += row_bytes
        _unfilter(row, previous, bytes_per_pixel, filter_type)
        raw.extend(row)
        previous = bytes(row)
    values = array("B" if bit_depth == 8 else "H")
    if bit_depth == 8:
        values.frombytes(raw)
        dtype = "uint8"
        canonical = bytes(raw)
    else:
        for index in range(0, len(raw), 2):
            values.append(struct.unpack(">H", raw[index : index + 2])[0])
        dtype = "uint16"
        canonical = bytes(raw)
    return DecodedArray(
        width,
        height,
        channels,
        dtype,
        values,
        f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    )


def _pfm_line(content: bytes, position: int) -> tuple[bytes, int]:
    end = content.find(b"\n", position)
    if end < 0 or end - position > 128:
        raise RenderFormatError("invalid or oversized PFM header line")
    line = content[position:end].strip()
    if not line or line.startswith(b"#"):
        raise RenderFormatError("empty/comments are unsupported in PFM headers")
    return line, end + 1


def _dimensions(
    width: int,
    height: int,
    channels: int,
    value_count: int,
    bytes_per_sample: int,
) -> None:
    expected_bytes = _checked_bytes(width, height, channels, bytes_per_sample)
    if value_count != width * height * channels:
        raise RenderFormatError("value count differs from declared dimensions")
    if expected_bytes <= 0:
        raise RenderFormatError("empty arrays are unsupported")


def _checked_bytes(
    width: int, height: int, channels: int, bytes_per_sample: int
) -> int:
    if width <= 0 or height <= 0 or channels <= 0 or bytes_per_sample <= 0:
        raise RenderFormatError("dimensions and sample size must be positive")
    total = width * height * channels * bytes_per_sample
    if total > MAX_DECODED_BYTES:
        raise RenderFormatError("decoded array exceeds the safety limit")
    return total


def _native_byte_order() -> Literal["little", "big"]:
    return "little" if struct.pack("=I", 1)[0] == 1 else "big"


def _float32_little_bytes(values: array[float]) -> bytes:
    copied = array("f", values)
    if copied.itemsize != 4:
        raise RenderFormatError("platform float array is not float32")
    if _native_byte_order() == "big":
        copied.byteswap()
    return copied.tobytes()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _unfilter(
    row: bytearray, previous: bytes, bytes_per_pixel: int, filter_type: int
) -> None:
    for index in range(len(row)):
        left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index]
        upper_left = (
            previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        )
        if filter_type == 0:
            continue
        if filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = _paeth(left, up, upper_left)
        else:
            raise RenderFormatError(f"unsupported PNG filter type {filter_type}")
        row[index] = (row[index] + predictor) & 0xFF


def _paeth(left: int, up: int, upper_left: int) -> int:
    prediction = left + up - upper_left
    left_distance = abs(prediction - left)
    up_distance = abs(prediction - up)
    diagonal_distance = abs(prediction - upper_left)
    if left_distance <= up_distance and left_distance <= diagonal_distance:
        return left
    if up_distance <= diagonal_distance:
        return up
    return upper_left
