"""CPU-safe ownership and portable-region validators."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gaussweave.real_structuring.models import (
    CORE_COUNTS,
    INSTANCE_IDS,
    PILOT_VERSION,
    SOURCE_COUNT,
    OwnershipStatus,
    StructuringError,
    load_object,
    sha256_file,
)

INSTANCE_VIEWS: Mapping[str, Mapping[str, tuple[int, int, int, int]]] = {
    "panel-rear": {
        "000251.jpg": (36, 145, 242, 236),
        "000249.jpg": (116, 175, 298, 231),
        "000180.jpg": (148, 207, 280, 335),
    },
    "panel-middle": {
        "000251.jpg": (245, 143, 365, 236),
        "000249.jpg": (302, 151, 414, 230),
        "000180.jpg": (284, 200, 445, 320),
    },
    "panel-front": {
        "000251.jpg": (370, 113, 501, 237),
        "000249.jpg": (418, 106, 558, 231),
        "000180.jpg": (450, 181, 610, 307),
    },
}


def validate_ownership(root: Path) -> dict[str, Any]:
    """Validate exactly-one ownership without importing tensor libraries."""

    record = load_object(root / "ownership.json")
    pilot_version = str(record.get("pilot_version", ""))
    if pilot_version != PILOT_VERSION and not pilot_version.startswith("gw-"):
        raise StructuringError("ownership pilot version mismatch")
    payload = root / str(record.get("payload", "ownership.u8"))
    if not payload.is_file() or payload.parent != root.resolve():
        raise StructuringError("ownership payload is missing or unsafe")
    values = payload.read_bytes()
    source_count = int(record.get("source_gaussian_count", SOURCE_COUNT))
    if source_count <= 0 or len(values) != source_count:
        raise StructuringError("ownership payload must classify every source Gaussian")
    if sha256_file(payload) != record.get("payload_sha256"):
        raise StructuringError("ownership payload digest mismatch")
    declared_statuses = record.get("status_codes")
    allowed = (
        {int(value) for value in declared_statuses.values()}
        if isinstance(declared_statuses, dict)
        else {int(status) for status in OwnershipStatus}
    )
    if any(value not in allowed for value in values):
        raise StructuringError("ownership payload contains an unknown status")
    counts = Counter(values)
    declared = (
        CORE_COUNTS
        if pilot_version == PILOT_VERSION
        else {
            str(key): int(value) for key, value in record.get("core_counts", {}).items()
        }
    )
    instance_ids = tuple(
        str(value) for value in record.get("instance_ids", INSTANCE_IDS)
    )
    if not instance_ids or set(declared) != set(instance_ids):
        raise StructuringError("ownership core counts are incomplete or out of order")
    declared = {instance_id: declared[instance_id] for instance_id in instance_ids}
    core_statuses = record.get("core_status_codes")
    if isinstance(core_statuses, dict):
        expected_codes = {
            instance_id: int(core_statuses[instance_id]) for instance_id in instance_ids
        }
    else:
        expected_codes = {
            "panel-rear": int(OwnershipStatus.PANEL_REAR_CORE),
            "panel-middle": int(OwnershipStatus.PANEL_MIDDLE_CORE),
            "panel-front": int(OwnershipStatus.PANEL_FRONT_CORE),
        }
    for instance_id, count in declared.items():
        if counts[expected_codes[instance_id]] != count:
            raise StructuringError(f"qualified core count mismatch: {instance_id}")
    if sum(counts.values()) != source_count:
        raise StructuringError("ownership counts do not cover the source model")
    excluded_code = int(
        record.get("excluded_invalid_status_code", OwnershipStatus.EXCLUDED_INVALID)
    )
    if excluded_code in allowed and counts[excluded_code] != 0:
        raise StructuringError("qualified source model must not exclude invalid rows")
    return {
        "valid": True,
        "source_gaussian_count": source_count,
        "status_counts": {
            str(name): counts[int(code)]
            for name, code in (
                declared_statuses.items()
                if isinstance(declared_statuses, dict)
                else ((status.name.lower(), int(status)) for status in OwnershipStatus)
            )
        },
        "core_counts": dict(declared),
        "core_total": sum(declared.values()),
        "background_complement_count": source_count - sum(declared.values()),
        "payload_sha256": record["payload_sha256"],
        "instance_ids": list(instance_ids),
    }


def validate_region_root(root: Path) -> dict[str, Any]:
    ownership = validate_ownership(root)
    manifest = load_object(root / "manifest.json")
    frames = load_object(root / "frames.json")
    registrations = load_object(root / "registration.json")
    canonical = load_object(root / "canonical" / "metadata.json")
    instance_ids = tuple(ownership["instance_ids"])
    if tuple(item["instance_id"] for item in frames.get("frames", [])) != instance_ids:
        raise StructuringError("panel frames are incomplete or out of order")
    if tuple(item["instance_id"] for item in registrations.get("instances", [])) != (
        instance_ids
    ):
        raise StructuringError("registrations are incomplete or out of order")
    if canonical.get("gaussian_count") != manifest["counts"]["stored_canonical_panel"]:
        raise StructuringError("canonical terminal count mismatch")
    if canonical.get("sh_degree") != 3:
        raise StructuringError("canonical terminal must preserve SH degree 3")
    return {
        "valid": True,
        "ownership": ownership,
        "frame_count": len(frames["frames"]),
        "registration_count": len(registrations["instances"]),
        "canonical_count": canonical["gaussian_count"],
        "canonical_digest": canonical["scientific_digest"],
    }
