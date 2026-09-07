from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.experiments.lifecycle import load_state
from gaussweave.experiments.smoke_pipeline import (
    DEFAULT_CONFIG,
    PipelineOperationError,
    PipelineResourceError,
    run_smoke_pipeline,
)
from gaussweave.metrics import compute_per_view
from gaussweave.results.artifacts import (
    VerificationState,
    load_inventory,
    verify_inventory,
)
from gaussweave.results.records import load_result, read_ppm

pytestmark = pytest.mark.gpu


def test_live_pipeline_reproduction_failures_and_strict_evidence(
    tmp_path: Path,
) -> None:
    first = run_smoke_pipeline(tmp_path / "first", (DEFAULT_CONFIG,))
    first_run = tmp_path / "first" / first.experiment_id / first.run_id
    state = load_state(first_run / "status.json")
    assert [item.to_status for item in state.history] == [
        "created",
        "validated",
        "running",
        "completed",
    ]
    assert math.isfinite(first.initial_training_loss)
    assert first.final_training_loss < first.initial_training_loss
    assert first.final_gaussian_count == 3
    result = load_result(first_run / first.result_path)
    assert result["status"] == "completed"
    assert result["freeze"]["status"] == "unfrozen"
    assert result["accounting"]["status"] == "not_measured"
    assert result["accounting"]["compression_ratio"] is None
    observation = json.loads(
        (first_run / "metadata/metric-observations.json").read_text()
    )
    direct = compute_per_view(
        "syn-tiny",
        "test-up",
        read_ppm(first_run / "artifacts/reference/test-up-rgb.ppm"),
        read_ppm(first_run / "artifacts/held-out/test-up-rgb.ppm"),
        reference_artifact="artifacts/reference/test-up-rgb.ppm",
        prediction_artifact="artifacts/held-out/test-up-rgb.ppm",
    )
    assert observation["views"][0]["psnr"]["value"] == pytest.approx(direct.psnr.value)
    evidence = load_inventory(first_run / first.evidence_inventory_path)
    assert evidence.artifact_count == first.evidence_artifact_count
    assert (
        verify_inventory(evidence, first_run, strict=True).state
        is VerificationState.VALID
    )
    summary = json.loads((first_run / "metadata/pipeline-summary.json").read_text())
    assert summary["run_id"] == first.run_id
    assert summary["checkpoint_digest"] == first.checkpoint_digest
    assert summary["result_digest"] == first.result_digest
    assert summary["inventory_digest"] == first.inventory_digest
    environment = json.loads((first_run / "metadata/environment.json").read_text())
    assert environment["wsl"]["detected"]
    assert environment["locked_gpu_runtime"]["cuda_available"]

    second = run_smoke_pipeline(tmp_path / "second", (DEFAULT_CONFIG,))
    assert second.scientific_digest == first.scientific_digest
    assert second.run_id == first.run_id
    assert second.scene_psnr_db == pytest.approx(first.scene_psnr_db, abs=1e-5)
    assert second.final_training_loss == pytest.approx(
        first.final_training_loss, abs=1e-7
    )
    with pytest.raises(PipelineOperationError, match="existing run"):
        run_smoke_pipeline(tmp_path / "first", (DEFAULT_CONFIG,))

    failure_root = tmp_path / "failure"
    with pytest.raises(PipelineOperationError) as injected:
        run_smoke_pipeline(failure_root, (DEFAULT_CONFIG,), inject_failure=True)
    failed_run = injected.value.run_directory
    assert failed_run is not None
    failed_state = load_state(failed_run / "status.json")
    assert failed_state.status == "failed" and failed_state.failure_ref
    assert not (failed_run / "artifacts/result.json").exists()
    assert (failed_run / "logs/events.jsonl").is_file()
    assert (failed_run / "metadata/environment.json").is_file()
    partial = load_inventory(failed_run / "artifact-inventory.json")
    assert (
        verify_inventory(partial, failed_run, strict=True).state
        is VerificationState.VALID
    )

    guard_root = tmp_path / "guard"
    with pytest.raises(PipelineResourceError) as guarded:
        run_smoke_pipeline(
            guard_root,
            (DEFAULT_CONFIG,),
            artificial_gpu_limit_bytes=1,
        )
    guard_run = guarded.value.run_directory
    assert guard_run is not None
    guard_state = load_state(guard_run / "status.json")
    assert guard_state.status == "failed"
    failure_record = json.loads(
        (guard_run / (guard_state.failure_ref or "")).read_text()
    )
    assert failure_record["stage"] == "resource.preflight"
    assert not (guard_run / "metadata/training-summary.json").exists()

    (first_run / "metadata/pipeline-summary.json").write_text("corrupt")
    assert (
        verify_inventory(evidence, first_run, strict=True).state
        is VerificationState.INVALID
    )


def test_formal_pipeline_json_and_failure_exit_codes(tmp_path: Path) -> None:
    success = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.cli",
            "experiment",
            "smoke",
            "--config",
            str(DEFAULT_CONFIG),
            "--root",
            str(tmp_path / "cli-success"),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert success.returncode == 0, success.stderr
    assert json.loads(success.stdout)["status"] == "completed"
    failure = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.cli",
            "experiment",
            "smoke",
            "--config",
            str(DEFAULT_CONFIG),
            "--root",
            str(tmp_path / "cli-failure"),
            "--inject-failure",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failure.returncode == 6
    assert json.loads(failure.stdout)["status"] == "failed"
    guard = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.cli",
            "experiment",
            "smoke",
            "--config",
            str(DEFAULT_CONFIG),
            "--root",
            str(tmp_path / "cli-guard"),
            "--artificial-gpu-limit-bytes",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert guard.returncode == 5
    assert json.loads(guard.stdout)["exit_code"] == 5
