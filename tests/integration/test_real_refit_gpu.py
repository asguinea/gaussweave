from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.wsl]


def _root() -> Path:
    value = os.environ.get("GAUSSWEAVE_TRUCK_REFIT_ROOT")
    if value is None:
        pytest.skip("GAUSSWEAVE_TRUCK_REFIT_ROOT is not configured")
    root = Path(value).resolve()
    if not (root / "region" / "manifest.json").is_file():
        pytest.fail("configured R02 local root is incomplete")
    return root


def test_t9_t10_t11_t12_live_canonical_round_trip_and_uniqueness() -> None:
    import torch

    from gaussweave.gaussians.real_explicit import RealGaussianTensorSource
    from gaussweave.real_structuring.hybrid import HybridTensors

    root = _root()
    hybrid_root = root / "hybrid"
    region_root = root / "region"
    hybrid = HybridTensors.load(hybrid_root)
    panels = hybrid.panels()
    count = 9_533
    middle = slice(count, 2 * count)
    source_indices = torch.from_file(
        str(region_root / "canonical" / "source-indices.u32"),
        shared=False,
        size=count,
        dtype=torch.int32,
    ).to(device=hybrid.device, dtype=torch.int64)
    dataset_root = Path(
        json.loads((hybrid_root / "hybrid.json").read_text(encoding="utf-8"))[
            "local_dataset_root"
        ]
    )
    source = RealGaussianTensorSource.load(
        dataset_root / "converted" / "graphdeco-30000"
    ).tensors(hybrid.device)
    assert panels["appearance"].shape == (28_599, 16, 3)
    assert torch.allclose(
        panels["means"][middle], source["means"][source_indices], atol=2e-6
    )
    assert torch.allclose(
        panels["scales"][middle], source["scales"][source_indices], atol=2e-6
    )
    quaternion_dot = (
        (panels["quaternions"][middle] * source["quaternions"][source_indices])
        .sum(dim=1)
        .abs()
    )
    assert torch.allclose(quaternion_dot, torch.ones_like(quaternion_dot), atol=2e-5)
    ownership = torch.from_file(
        str(region_root / "ownership.u8"),
        shared=False,
        size=2_541_226,
        dtype=torch.uint8,
    )
    background = torch.from_file(
        str(region_root / "background-indices.u32"),
        shared=False,
        size=2_501_435,
        dtype=torch.int32,
    ).to(torch.int64)
    assert not torch.isin(ownership[background], torch.tensor((1, 2, 3))).any()
    assert torch.unique(background).numel() == background.numel()


def test_t15_t16_t17_t18_live_fits_preserve_background_and_gradient_scope() -> None:
    root = _root()
    shared = json.loads(
        (
            root / "hybrid" / "fits" / "real_struct_shared" / "fit-summary.json"
        ).read_text(encoding="utf-8")
    )
    q8 = json.loads(
        (
            root / "hybrid" / "fits" / "real_struct_residual_q8" / "fit-summary.json"
        ).read_text(encoding="utf-8")
    )
    assert shared["background_integrity_before"] == shared["background_integrity_after"]
    assert q8["background_integrity_before"] == q8["background_integrity_after"]
    assert shared["loss_decreased"] is True
    assert q8["loss_decreased"] is True
    assert shared["gradient_parameter_groups"] == ["canonical_sh_coefficients"]
    assert q8["gradient_parameter_groups"] == ["per_instance_sh0_residual_rear_front"]
    assert q8["q8_residual"]["saturation_count"] == 0
    assert q8["q8_residual"]["acceptance_uses_evaluation_views"] is False


def test_t20_live_held_out_outputs_are_complete() -> None:
    root = _root()
    evaluation = json.loads(
        (root / "hybrid" / "evaluation" / "evaluation-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(evaluation["evaluation_camera_ids"]) == 8
    assert len(evaluation["observations"]) == 24
    assert evaluation["q8_improves_or_preserves_shared"] is True
    for camera_id in evaluation["evaluation_camera_ids"]:
        camera_root = root / "hybrid" / "evaluation" / "renders" / camera_id
        assert (camera_root / "official_explicit_gs.png").is_file()
        assert (camera_root / "real_struct_shared.png").is_file()
        assert (camera_root / "real_struct_residual_q8.png").is_file()
        assert (camera_root / "strict-roi.u8").is_file()
