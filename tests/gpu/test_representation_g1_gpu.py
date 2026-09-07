from __future__ import annotations

from pathlib import Path

import pytest

from gaussweave.representation.g1 import build_g1, check_g1

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e
def test_g1_all_eight_cameras_and_residual_improvement(tmp_path: Path) -> None:
    root = tmp_path / "g1"
    result = build_g1(repo_root=REPO_ROOT, output_root=root)
    assert result["state"] == "passed"
    assert result["checks"] == {
        "exact_array_equality": True,
        "exact_render_equality_all_8": True,
        "q8_psnr_improves_every_camera": True,
        "q8_psnr_improves_mean_and_min": True,
        "all_8_cameras_meaningful": True,
    }
    assert all(len(value) == 8 for value in result["meaningful_camera_ids"].values())
    assert check_g1(root)["valid"] is True
