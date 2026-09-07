"""Fixed-background integrity helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gaussweave.config.resolution import content_digest
from gaussweave.real_structuring.models import (
    StructuringError,
    load_object,
    sha256_file,
)


def background_integrity(root: Path) -> dict[str, Any]:
    metadata = load_object(root / "background" / "metadata.json")
    files = {
        name: {
            "bytes": (root / "background" / record["path"]).stat().st_size,
            "sha256": sha256_file(root / "background" / record["path"]),
        }
        for name, record in metadata["files"].items()
    }
    result = {
        "gaussian_count": metadata["gaussian_count"],
        "scientific_digest": metadata["scientific_digest"],
        "files": files,
    }
    result["integrity_digest"] = content_digest(result)
    return result


def require_background_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Reject any mutation of the fixed explicit background."""

    if before != after:
        raise StructuringError("fixed background changed during fitting")
