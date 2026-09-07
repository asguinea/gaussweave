from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.wsl]


def _root() -> Path:
    value = os.environ.get("GAUSSWEAVE_TRUCK_EDIT_ROOT")
    if value is None:
        pytest.skip("GAUSSWEAVE_TRUCK_EDIT_ROOT is not configured")
    root = Path(value).resolve()
    if not (root / "edit-summary.json").is_file():
        pytest.fail("configured Truck edit root is incomplete")
    return root


def test_t3_t4_t5_t6_fixed_content_integrity() -> None:
    summary = json.loads((_root() / "edit-summary.json").read_text())
    integrity = summary["fixed_content_integrity"]
    assert integrity["canonical_byte_identical"] is True
    assert integrity["background_byte_identical"] is True
    assert integrity["instance_transforms_byte_identical"] is True
    assert integrity["residual_payload_byte_identical"] is True
    assert integrity["camera_records_unchanged"] is True


def test_t2_t7_t8_t13_t14_live_materialization_and_ownership() -> None:
    root = _root()
    summary = json.loads((root / "edit-summary.json").read_text())
    assert summary["edited_active_instance_set"]["active_instance_ids"] == [
        "panel-rear",
        "panel-front",
    ]
    materialization = summary["materialization"]
    assert materialization["fixed_explicit_complement"] == 2_533_367
    assert materialization["canonical_stored"] == 2_953
    assert materialization["source_hybrid_materialized"] == 2_542_226
    assert materialization["edited_hybrid_materialized"] == 2_539_273
    sanity = summary["geometry_sanity"]
    assert sanity["removed_instance_absent"] is True
    assert sanity["remaining_instance_counts"] == {
        "panel-front": 1,
        "panel-rear": 1,
    }
    assert sanity["middle_source_core_reintroduced_by_background"] is False
    assert sanity["body_and_frame_preserved_by_background_identity"] is True


def test_t10_t11_t12_all_views_masks_and_localization() -> None:
    root = _root()
    renders = json.loads((root / "render-ledger.json").read_text())["views"]
    masks = json.loads((root / "mask-ledger.json").read_text())["views"]
    localization = json.loads((root / "localization-summary.json").read_text())
    assert len(renders) == len(masks) == len(localization["views"]) == 8
    assert all(record["complete"] for record in renders)
    assert all(
        record["conservative_support_pixels"] + record["stable_outside_pixels"]
        == record["width"] * record["height"]
        for record in masks
    )
    assert localization["aggregate"]["all_outside_regions_stable"] is True


def test_t9_actual_serialization_delta_is_nonzero_and_complete() -> None:
    accounting = json.loads((_root() / "edit-accounting.json").read_text())
    assert accounting["source_representation_bytes"] > 0
    assert accounting["edited_representation_bytes"] > 0
    assert accounting["edit_request_bytes"] > 0
    assert accounting["changed_serialized_bytes"] > 0
    assert accounting["changed_files"]
    assert accounting["unchanged_files"]
