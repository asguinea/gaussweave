"""Bounded HTTP-range access to selected members of large official ZIP archives."""

from __future__ import annotations

import binascii
import io
import struct
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast


class RemoteZipError(RuntimeError):
    """A remote ZIP is malformed, unsafe, or does not support bounded access."""


@dataclass(frozen=True)
class RemoteZipEntry:
    """Central-directory metadata required for safe member extraction."""

    name: str
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.name)
        if path.is_absolute() or ".." in path.parts or "\\" in self.name:
            raise RemoteZipError(f"unsafe ZIP member: {self.name}")
        if (
            min(
                self.compressed_size,
                self.uncompressed_size,
                self.local_header_offset,
            )
            < 0
        ):
            raise RemoteZipError(f"negative ZIP metadata: {self.name}")
        if self.compression not in {0, 8}:
            raise RemoteZipError(
                f"unsupported ZIP compression {self.compression}: {self.name}"
            )


class HttpRangeSource:
    """Small seek-like HTTP range client with explicit size and identity."""

    def __init__(self, url: str, *, timeout_seconds: float = 120.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "GaussWeave/0.1"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            length = response.headers.get("Content-Length")
            ranges = response.headers.get("Accept-Ranges", "")
            self.etag = response.headers.get("ETag")
            self.last_modified = response.headers.get("Last-Modified")
        if length is None or not length.isdigit() or int(length) <= 0:
            raise RemoteZipError("remote ZIP has no valid Content-Length")
        if "bytes" not in ranges.lower():
            raise RemoteZipError("remote ZIP does not advertise byte ranges")
        self.size = int(length)

    def read_range(self, start: int, end_inclusive: int) -> bytes:
        """Read one exact inclusive byte range."""

        if start < 0 or end_inclusive < start or end_inclusive >= self.size:
            raise RemoteZipError("invalid HTTP byte range")
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end_inclusive}",
                "User-Agent": "GaussWeave/0.1",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            status = getattr(response, "status", None)
            content_range = response.headers.get("Content-Range", "")
            payload = cast(bytes, response.read())
        expected = end_inclusive - start + 1
        if status != 206 or not content_range.startswith(f"bytes {start}-"):
            raise RemoteZipError("server did not honor the requested byte range")
        if len(payload) != expected:
            raise RemoteZipError("truncated HTTP range response")
        return payload

    def open_range(self, start: int, end_inclusive: int) -> BinaryIO:
        """Open a streaming exact byte range."""

        if start < 0 or end_inclusive < start or end_inclusive >= self.size:
            raise RemoteZipError("invalid HTTP byte range")
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end_inclusive}",
                "User-Agent": "GaussWeave/0.1",
            },
        )
        response = urllib.request.urlopen(request, timeout=self.timeout_seconds)
        status = getattr(response, "status", None)
        content_range = response.headers.get("Content-Range", "")
        if status != 206 or not content_range.startswith(f"bytes {start}-"):
            response.close()
            raise RemoteZipError("server did not honor the requested byte range")
        return cast(BinaryIO, response)


def _zip64_value(
    extra: bytes,
    *,
    uncompressed: int,
    compressed: int,
    offset: int,
) -> tuple[int, int, int]:
    cursor = 0
    payload: bytes | None = None
    while cursor + 4 <= len(extra):
        identifier, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        value = extra[cursor : cursor + size]
        cursor += size
        if identifier == 0x0001:
            payload = value
            break
    if payload is None:
        raise RemoteZipError("ZIP64 sentinel has no ZIP64 extra record")
    values = io.BytesIO(payload)

    def take(current: int, sentinel: int) -> int:
        if current != sentinel:
            return current
        raw = values.read(8)
        if len(raw) != 8:
            raise RemoteZipError("truncated ZIP64 extra record")
        return cast(int, struct.unpack("<Q", raw)[0])

    return (
        take(uncompressed, 0xFFFFFFFF),
        take(compressed, 0xFFFFFFFF),
        take(offset, 0xFFFFFFFF),
    )


def _central_directory_location(source: HttpRangeSource) -> tuple[int, int]:
    tail_size = min(source.size, 1024 * 1024)
    tail_start = source.size - tail_size
    tail = source.read_range(tail_start, source.size - 1)
    eocd_index = tail.rfind(b"PK\x05\x06")
    if eocd_index < 0 or eocd_index + 22 > len(tail):
        raise RemoteZipError("ZIP end-of-central-directory record not found")
    (
        _signature,
        _disk,
        _central_disk,
        _disk_entries,
        _entries,
        central_size32,
        central_offset32,
        _comment_length,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_index)
    if central_size32 != 0xFFFFFFFF and central_offset32 != 0xFFFFFFFF:
        return central_offset32, central_size32
    locator_index = tail.rfind(b"PK\x06\x07", 0, eocd_index)
    if locator_index < 0 or locator_index + 20 > len(tail):
        raise RemoteZipError("ZIP64 locator not found")
    _sig, _disk_number, zip64_offset, _disk_count = struct.unpack_from(
        "<4sLQL", tail, locator_index
    )
    record = source.read_range(zip64_offset, min(zip64_offset + 55, source.size - 1))
    if len(record) < 56 or record[:4] != b"PK\x06\x06":
        raise RemoteZipError("invalid ZIP64 end-of-central-directory record")
    central_size = struct.unpack_from("<Q", record, 40)[0]
    central_offset = struct.unpack_from("<Q", record, 48)[0]
    return central_offset, central_size


def remote_zip_entries(source: HttpRangeSource) -> tuple[RemoteZipEntry, ...]:
    """Return validated central-directory entries without downloading the archive."""

    central_offset, central_size = _central_directory_location(source)
    if central_size > 256 * 1024 * 1024:
        raise RemoteZipError("central directory exceeds the 256 MiB safety ceiling")
    central = source.read_range(central_offset, central_offset + central_size - 1)
    cursor = 0
    entries: list[RemoteZipEntry] = []
    while cursor < len(central):
        if cursor + 46 > len(central) or central[cursor : cursor + 4] != b"PK\x01\x02":
            raise RemoteZipError("malformed central-directory entry")
        (
            _made_by,
            _needed,
            flag,
            compression,
            _time,
            _date,
            crc32,
            compressed,
            uncompressed,
            name_length,
            extra_length,
            comment_length,
            _disk,
            _internal,
            _external,
            local_offset,
        ) = struct.unpack_from("<6H3L5H2L", central, cursor + 4)
        if flag & 0x1:
            raise RemoteZipError("encrypted ZIP members are unsupported")
        payload_start = cursor + 46
        payload_end = payload_start + name_length + extra_length + comment_length
        if payload_end > len(central):
            raise RemoteZipError("truncated central-directory payload")
        name_raw = central[payload_start : payload_start + name_length]
        extra = central[
            payload_start + name_length : payload_start + name_length + extra_length
        ]
        encoding = "utf-8" if flag & 0x800 else "cp437"
        name = name_raw.decode(encoding)
        uncompressed, compressed, local_offset = (
            _zip64_value(
                extra,
                uncompressed=uncompressed,
                compressed=compressed,
                offset=local_offset,
            )
            if 0xFFFFFFFF in {uncompressed, compressed, local_offset}
            else (
                uncompressed,
                compressed,
                local_offset,
            )
        )
        entries.append(
            RemoteZipEntry(
                name=name,
                compression=compression,
                crc32=crc32,
                compressed_size=compressed,
                uncompressed_size=uncompressed,
                local_header_offset=local_offset,
            )
        )
        cursor = payload_end
    return tuple(entries)


def extract_remote_entry(
    source: HttpRangeSource,
    entry: RemoteZipEntry,
    destination: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Extract one remote member atomically and verify size plus CRC32."""

    if entry.name.endswith("/"):
        destination.mkdir(parents=True, exist_ok=True)
        return
    local_header = source.read_range(
        entry.local_header_offset, entry.local_header_offset + 29
    )
    if local_header[:4] != b"PK\x03\x04":
        raise RemoteZipError(f"invalid local header: {entry.name}")
    name_length, extra_length = struct.unpack_from("<HH", local_header, 26)
    data_start = entry.local_header_offset + 30 + name_length + extra_length
    data_end = data_start + entry.compressed_size - 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    crc = 0
    written = 0
    decompressor = zlib.decompressobj(-15) if entry.compression == 8 else None
    try:
        with (
            source.open_range(data_start, data_end) as response,
            temporary.open("wb") as output,
        ):
            remaining = entry.compressed_size
            while remaining:
                block = response.read(min(chunk_size, remaining))
                if not block:
                    raise RemoteZipError(f"truncated member: {entry.name}")
                remaining -= len(block)
                decoded = decompressor.decompress(block) if decompressor else block
                output.write(decoded)
                written += len(decoded)
                crc = binascii.crc32(decoded, crc)
            if decompressor:
                decoded = decompressor.flush()
                output.write(decoded)
                written += len(decoded)
                crc = binascii.crc32(decoded, crc)
        if written != entry.uncompressed_size:
            raise RemoteZipError(f"size mismatch after extraction: {entry.name}")
        if crc & 0xFFFFFFFF != entry.crc32:
            raise RemoteZipError(f"CRC32 mismatch after extraction: {entry.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_remote_prefix(
    url: str,
    prefix: str,
    destination: Path,
    *,
    maximum_uncompressed_bytes: int,
) -> tuple[RemoteZipEntry, ...]:
    """Extract a bounded safe prefix from a remote ZIP."""

    normalized = prefix.strip("/") + "/"
    source = HttpRangeSource(url)
    selected = tuple(
        entry
        for entry in remote_zip_entries(source)
        if entry.name.startswith(normalized)
    )
    files = tuple(entry for entry in selected if not entry.name.endswith("/"))
    if not files:
        raise RemoteZipError(f"remote ZIP contains no files under {normalized}")
    total = sum(entry.uncompressed_size for entry in files)
    if total > maximum_uncompressed_bytes:
        raise RemoteZipError(
            f"selected members require {total} bytes, exceeding the safety ceiling"
        )
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for entry in selected:
        relative = PurePosixPath(entry.name).relative_to(PurePosixPath(normalized))
        target = destination.joinpath(*relative.parts)
        resolved = target.resolve(strict=False)
        if root not in resolved.parents and resolved != root:
            raise RemoteZipError(f"member escapes destination: {entry.name}")
        extract_remote_entry(source, entry, target)
    return selected


def extract_local_prefix(
    archive_path: Path,
    prefix: str,
    destination: Path,
    *,
    maximum_uncompressed_bytes: int,
) -> tuple[str, ...]:
    """Safely extract a bounded prefix from a local, fully downloaded ZIP."""

    normalized = prefix.strip("/") + "/"
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as error:
        raise RemoteZipError(f"invalid ZIP archive: {archive_path.name}") from error
    with archive:
        bad = archive.testzip()
        if bad is not None:
            raise RemoteZipError(f"corrupt ZIP member: {bad}")
        selected = tuple(
            info for info in archive.infolist() if info.filename.startswith(normalized)
        )
        files = tuple(info for info in selected if not info.is_dir())
        if not files:
            raise RemoteZipError(f"local ZIP contains no files under {normalized}")
        total = sum(info.file_size for info in files)
        if total > maximum_uncompressed_bytes:
            raise RemoteZipError(
                f"selected members require {total} bytes, exceeding the safety ceiling"
            )
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        extracted: list[str] = []
        for info in selected:
            name = info.filename
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise RemoteZipError(f"unsafe ZIP member: {name}")
            relative = member.relative_to(PurePosixPath(normalized))
            target = destination.joinpath(*relative.parts)
            resolved = target.resolve(strict=False)
            if root not in resolved.parents and resolved != root:
                raise RemoteZipError(f"member escapes destination: {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.part")
            try:
                with archive.open(info) as source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                if temporary.stat().st_size != info.file_size:
                    raise RemoteZipError(f"size mismatch after extraction: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted.append(name)
        return tuple(extracted)
