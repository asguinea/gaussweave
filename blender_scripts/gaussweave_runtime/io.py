"""Portable path, JSON, digest, and atomic output helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any


class RuntimeIOError(ValueError):
    """A runtime path or serialization value is unsafe."""


def portable_path(value: str) -> str:
    """Validate one POSIX relative artifact path."""

    if not value or value == "." or "\\" in value:
        raise RuntimeIOError("portable path must be a nonempty POSIX path")
    if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
        raise RuntimeIOError("portable path must be relative")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise RuntimeIOError("portable path may not traverse or contain empty parts")
    return PurePosixPath(value).as_posix()


def artifact_path(root: Path, value: str) -> Path:
    """Resolve a portable path beneath a non-symlink output root."""

    portable = portable_path(value)
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*PurePosixPath(portable).parts)
    current = resolved_root
    for part in PurePosixPath(portable).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeIOError(f"artifact path crosses a symlink: {portable}")
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeIOError(
            f"artifact path escapes output root: {portable}"
        ) from error
    return candidate


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize deterministic canonical JSON, rejecting nonfinite values."""

    _reject_nonfinite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    """Serialize readable stable JSON with one terminal newline."""

    _reject_nonfinite(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return a prefixed SHA-256 digest of canonical JSON."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    """Return a prefixed SHA-256 digest of complete file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    """Atomically replace one file inside its existing or newly made parent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise


def write_json(path: Path, value: Any) -> None:
    """Atomically write deterministic readable JSON."""

    atomic_write(path, pretty_json_bytes(value))


def write_checksums(root: Path, paths: list[str], output: str) -> None:
    """Write a sorted SHA-256 checksum file for declared portable paths."""

    lines = []
    for portable in sorted(portable_path(item) for item in paths):
        digest = file_digest(artifact_path(root, portable)).removeprefix("sha256:")
        lines.append(f"{digest}  {portable}")
    atomic_write(artifact_path(root, output), ("\n".join(lines) + "\n").encode("utf-8"))


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeIOError("JSON values must be finite")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeIOError("JSON object keys must be strings")
            _reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_nonfinite(item)
