"""Actual-byte accounting for the local real structured representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gaussweave.config.resolution import content_digest
from gaussweave.real_structuring.fitting import (
    Q8_METHOD,
    SH1_Q8_METHOD,
    SHARED_METHOD,
)
from gaussweave.real_structuring.models import (
    PILOT_VERSION,
    StructuringError,
    atomic_json,
    load_object,
)


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def account(hybrid_root: Path) -> dict[str, Any]:
    """Measure complete repeated-region and whole-hybrid serialized bytes."""

    manifest = load_object(hybrid_root / "hybrid.json")
    pilot_version = str(manifest.get("pilot_version", ""))
    if pilot_version != PILOT_VERSION and not pilot_version.startswith(
        "gw-truck-panel-"
    ):
        raise StructuringError("hybrid accounting identity mismatch")
    background_bytes = _tree_bytes(hybrid_root / "background")
    canonical_bytes = _tree_bytes(hybrid_root / "canonical")
    transform_bytes = (hybrid_root / "instance-transforms.json").stat().st_size
    binding_bytes = (hybrid_root / "binding.json").stat().st_size
    compact_method = (
        SH1_Q8_METHOD
        if (hybrid_root / "fits" / SH1_Q8_METHOD / "fit-summary.json").is_file()
        else Q8_METHOD
    )
    method_metadata_bytes = {
        SHARED_METHOD: _tree_bytes(hybrid_root / "fits" / SHARED_METHOD),
        compact_method: _tree_bytes(hybrid_root / "fits" / compact_method),
    }
    q8_payload_bytes = (
        (hybrid_root / "fits" / compact_method / "residuals.i8").stat().st_size
    )
    explicit_bytes_per_gaussian = 236
    source_panel_bytes = (
        int(manifest["counts"]["source_explicit_panel"]) * explicit_bytes_per_gaussian
    )
    full_explicit_semantic_bytes = (
        int(manifest["counts"]["source"]) * explicit_bytes_per_gaussian
    )
    hybrid_metadata_bytes = (hybrid_root / "hybrid.json").stat().st_size
    methods: dict[str, Any] = {}
    for method in (SHARED_METHOD, compact_method):
        residual_bytes = q8_payload_bytes if method == compact_method else 0
        structured_panel = (
            canonical_bytes
            + transform_bytes
            + binding_bytes
            + method_metadata_bytes[method]
        )
        complete_hybrid = background_bytes + structured_panel + hybrid_metadata_bytes
        methods[method] = {
            "repeated_region": {
                "source_explicit_panel_bytes": source_panel_bytes,
                "canonical_terminal_serialized_bytes": canonical_bytes,
                "instance_transform_bytes": transform_bytes,
                "residual_payload_bytes": residual_bytes,
                "binding_and_method_metadata_bytes": (
                    binding_bytes + method_metadata_bytes[method] - residual_bytes
                ),
                "complete_structured_panel_bytes": structured_panel,
                "compression_ratio_explicit_over_structured": (
                    source_panel_bytes / structured_panel
                ),
            },
            "whole_hybrid": {
                "fixed_explicit_background_serialized_bytes": background_bytes,
                "structured_panel_bytes": structured_panel,
                "hybrid_manifest_bytes": hybrid_metadata_bytes,
                "complete_hybrid_serialized_bytes": complete_hybrid,
                "full_explicit_semantic_bytes": full_explicit_semantic_bytes,
                "whole_scene_ratio_explicit_over_hybrid": (
                    full_explicit_semantic_bytes / complete_hybrid
                ),
            },
        }
    result = {
        "accounting_version": "gw-real-actual-bytes-v1",
        "pilot_version": pilot_version,
        "accounting_level": "serialized_uncompressed",
        "representation_boundary": (
            "fixed background arrays, canonical arrays and source indexes, instance "
            "transforms, binding, method checkpoint/residual metadata, hybrid manifest"
        ),
        "counts": manifest["counts"],
        "bytes_per_explicit_gaussian_semantic": explicit_bytes_per_gaussian,
        "methods": methods,
        "exclusions": [
            "source photographs",
            "ROI masks",
            "background fitting plates",
            "renders and visual reviews",
            "optimizer transient state",
            "source code and environment",
        ],
        "claim_boundary": (
            "single-scene oracle repeated-region storage diagnostic; no general "
            "compression or runtime-memory claim"
        ),
    }
    for method, record in methods.items():
        whole = record["whole_hybrid"]
        if (
            whole["fixed_explicit_background_serialized_bytes"]
            + whole["structured_panel_bytes"]
            + whole["hybrid_manifest_bytes"]
            != whole["complete_hybrid_serialized_bytes"]
        ):
            raise StructuringError(f"incomplete accounting sum: {method}")
    result["scientific_digest"] = content_digest(result)
    atomic_json(hybrid_root / "accounting-summary.json", result)
    return result
