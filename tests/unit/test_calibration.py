from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.experiments.calibration import (
    CalibrationError,
    Candidate,
    expand_matrix,
    run_calibration,
    summarize_candidates,
)

CONFIG = Path("configs/experiments/hardware-calibration.json")


def test_matrix_determinism_and_sensitivity() -> None:
    first = expand_matrix(CONFIG)
    second = expand_matrix(CONFIG)
    assert first == second
    assert [item["candidate_id"] for item in first] == [
        "cal-conservative",
        "cal-balanced",
        "cal-standard",
        "cal-demanding",
        "cal-artificial-guard",
    ]
    changed = Candidate(
        **{
            **{
                key: value
                for key, value in first[0].items()
                if key != "resolved_configuration_digest"
            },
            "iterations": 9,
        }
    )
    assert changed.digest != first[0]["resolved_configuration_digest"]


def _record(candidate: Candidate, repetition: int, psnr: float) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "status": "completed",
        "compliance": "compliant",
        "gpu_peak_allocated_bytes": 100,
        "gpu_peak_reserved_bytes": 200,
        "cpu_peak_rss_bytes": 300,
        "training_seconds": 1.0,
        "held_out_psnr_db": psnr,
        "workspace_bytes": 400,
        "checkpoint_scientific_digest": "sha256:stable",
        "repetition": repetition,
    }


def test_selection_is_deterministic_and_explains_rejections() -> None:
    low = Candidate("cal-low", 64, 64, 10, 3, 3, 1, "float32", 2)
    high = Candidate("cal-high", 96, 96, 20, 8, 8, 2, "float32", 2)
    upper = Candidate("cal-upper", 128, 128, 40, 16, 16, 2, "float32", 1)
    records = [
        _record(low, 1, 18.0),
        _record(low, 2, 18.0),
        _record(high, 1, 19.0),
        _record(high, 2, 19.0),
        _record(upper, 1, 20.0),
    ]
    summaries, selected = summarize_candidates((low, high, upper), records)
    assert selected == summarize_candidates((low, high, upper), records)[1]
    assert selected["candidate_id"] == "cal-high"
    rejected = {item["candidate_id"]: item for item in summaries}
    assert "not a repeated finalist" in rejected["cal-upper"]["rejection_reasons"]


def test_calibration_cli_help_and_cpu_import_isolation() -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "gaussweave.cli", "experiment", "calibrate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout and "--root" in help_result.stdout
    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gaussweave.experiments.calibration; "
            "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_result.returncode == 0, import_result.stderr


def test_matrix_config_is_plain_json() -> None:
    assert json.loads(CONFIG.read_text(encoding="utf-8"))["seed"] == 17


def test_calibration_refuses_nonempty_root_before_gpu_work(tmp_path: Path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    (root / "sentinel").write_text("preserve", encoding="utf-8")
    with pytest.raises(CalibrationError, match="not empty"):
        run_calibration(CONFIG, root)
    assert (root / "sentinel").read_text(encoding="utf-8") == "preserve"
