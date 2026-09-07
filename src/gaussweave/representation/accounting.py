"""Actual-byte accounting for complete serialized Gaussian representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gaussweave.config.resolution import content_digest
from gaussweave.representation.serialization import validate_representation


def account_representation(
    root: Path, *, reference_root: Path | None = None
) -> dict[str, Any]:
    validation = validate_representation(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    category_bytes: dict[str, int] = {}
    npy_header_bytes = 0
    npy_payload_bytes = 0
    file_records: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        category = str(entry["category"])
        size = int(entry["bytes"])
        category_bytes[category] = category_bytes.get(category, 0) + size
        path = root / entry["path"]
        if path.suffix == ".npy":
            payload = path.read_bytes()
            header_bytes = 10 + int.from_bytes(payload[8:10], "little")
            npy_header_bytes += header_bytes
            npy_payload_bytes += len(payload) - header_bytes
        file_records.append(
            {
                "path": entry["path"],
                "bytes": size,
                "sha256": entry["sha256"],
                "category": category,
            }
        )
    manifest_bytes = (root / "manifest.json").stat().st_size
    category_bytes["representation_manifest"] = manifest_bytes
    file_records.append(
        {
            "path": "manifest.json",
            "bytes": manifest_bytes,
            "category": "representation_manifest",
        }
    )
    file_records.sort(key=lambda record: record["path"])
    complete = int(validation["complete_bytes"])
    if sum(category_bytes.values()) != complete:
        raise ValueError(
            "accounting categories do not sum to complete serialized bytes"
        )
    result: dict[str, Any] = {
        "accounting_version": "gw-actual-bytes-v1",
        "method_id": validation["method_id"],
        "representation_scientific_digest": validation["scientific_digest"],
        "complete_serialized_bytes": complete,
        "category_bytes": dict(sorted(category_bytes.items())),
        "npy_logical_payload_bytes": npy_payload_bytes,
        "npy_serialization_header_bytes": npy_header_bytes,
        "file_count": len(file_records),
        "files": file_records,
        "exclusions": [
            "environment_manifests",
            "artifact_inventories",
            "generated_reports",
            "diagnostic_logs",
            "renderer_outputs",
        ],
    }
    if reference_root is not None:
        reference = account_representation(reference_root)
        result["reference_method_id"] = reference["method_id"]
        result["reference_complete_serialized_bytes"] = reference[
            "complete_serialized_bytes"
        ]
        result["compression_ratio_reference_over_method"] = (
            reference["complete_serialized_bytes"] / complete
        )
    result["accounting_digest"] = content_digest(result)
    return result
