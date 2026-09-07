from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.wsl]


def _root() -> Path:
    value = os.environ.get("GAUSSWEAVE_TRUCK_REMEDIATION_ROOT")
    if value is None:
        pytest.skip("GAUSSWEAVE_TRUCK_REMEDIATION_ROOT is not configured")
    root = Path(value).resolve()
    if not (root / "region" / "manifest.json").is_file():
        pytest.fail("configured Truck remediation root is incomplete")
    return root


def test_remediation_candidate_selection_and_region_identity() -> None:
    root = _root()
    audit = json.loads(
        (root / "candidate-subregion-audit.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "region" / "manifest.json").read_text(encoding="utf-8")
    )
    assert audit["selection"]["selected_candidate_id"] == "candidate-a-equal-interiors"
    assert audit["selection"]["held_out_evidence_used"] is False
    assert manifest["region_version"] == "truck-bed-panel-interiors-v2"
    assert manifest["selection_uses_held_out_evidence"] is False


def test_remediation_ownership_background_and_materialization() -> None:
    import torch

    from gaussweave.real_structuring.hybrid import HybridTensors

    root = _root()
    manifest = json.loads(
        (root / "region" / "manifest.json").read_text(encoding="utf-8")
    )
    counts = manifest["counts"]
    ownership = torch.from_file(
        str(root / "region" / "ownership.u8"),
        shared=False,
        size=counts["source"],
        dtype=torch.uint8,
    )
    background = torch.from_file(
        str(root / "region" / "background-indices.u32"),
        shared=False,
        size=counts["fixed_explicit_background"],
        dtype=torch.int32,
    ).to(torch.int64)
    assert torch.unique(background).numel() == background.numel()
    assert not torch.isin(ownership[background], torch.tensor((1, 2, 3))).any()
    assert (
        counts["fixed_explicit_background"] + counts["source_explicit_panel"]
        == counts["source"]
    )
    hybrid = HybridTensors.load(root / "hybrid")
    assert hybrid.panels()["means"].shape[0] == counts["materialized_canonical_panel"]


def test_remediation_live_fitting_evaluation_and_accounting() -> None:
    root = _root()
    hybrid = root / "hybrid"
    shared = json.loads(
        (hybrid / "fits" / "real_struct_shared" / "fit-summary.json").read_text(
            encoding="utf-8"
        )
    )
    q8 = json.loads(
        (hybrid / "fits" / "real_struct_residual_q8" / "fit-summary.json").read_text(
            encoding="utf-8"
        )
    )
    sh1_q8 = json.loads(
        (
            hybrid / "fits" / "real_struct_residual_sh1_q8" / "fit-summary.json"
        ).read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (hybrid / "evaluation" / "evaluation-summary.json").read_text(encoding="utf-8")
    )
    accounting = json.loads(
        (hybrid / "accounting-summary.json").read_text(encoding="utf-8")
    )
    assert shared["loss_decreased"] is True
    assert q8["loss_decreased"] is True
    assert sh1_q8["loss_decreased"] is True
    assert shared["background_integrity_before"] == shared["background_integrity_after"]
    assert q8["background_integrity_before"] == q8["background_integrity_after"]
    assert sh1_q8["background_integrity_before"] == sh1_q8["background_integrity_after"]
    assert len(evaluation["observations"]) == 24
    assert evaluation["fitting_camera_ids"] == ["cam-train-000251", "cam-train-000180"]
    assert evaluation["compact_residual_method_id"] == "real_struct_residual_sh1_q8"
    assert evaluation["compact_residual_roi_psnr_improvement_db_over_shared"] > 0
    payload = sh1_q8["q8_residual"]
    assert payload["codec_version"] == "gaussweave-real-sh1-q8-v1"
    assert payload["fitted_offsets_accepted"] is True
    assert payload["payload"]["bytes"] == 36
    assert payload["saturation_count"] == 0
    assert accounting["counts"]["stored_canonical_panel"] > 0
    assert "real_struct_residual_sh1_q8" in accounting["methods"]
    for method in accounting["methods"].values():
        whole = method["whole_hybrid"]
        assert (
            whole["fixed_explicit_background_serialized_bytes"]
            + whole["structured_panel_bytes"]
            + whole["hybrid_manifest_bytes"]
            == whole["complete_hybrid_serialized_bytes"]
        )
