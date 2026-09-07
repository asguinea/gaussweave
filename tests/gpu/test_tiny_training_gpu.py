from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.experiments.lifecycle import load_state
from gaussweave.experiments.tiny_training import (
    TinyTrainingConfig,
    fixture_cameras,
    initialize_parameters,
    load_gaussian_checkpoint,
    run_tiny_training,
)
from gaussweave.rendering.gsplat_renderer import GsplatRenderer
from gaussweave.rendering.models import RenderSettings
from gaussweave.results.artifacts import (
    VerificationState,
    load_inventory,
    verify_inventory,
)

pytestmark = pytest.mark.gpu


def test_initialization_determinism_and_seed_change() -> None:
    torch = pytest.importorskip("torch")
    first = initialize_parameters(TinyTrainingConfig(seed=17, iterations=1))
    second = initialize_parameters(TinyTrainingConfig(seed=17, iterations=1))
    changed = initialize_parameters(TinyTrainingConfig(seed=18, iterations=1))
    assert torch.equal(first["means"], second["means"])
    assert not torch.equal(first["means"], changed["means"])


def test_short_training_converges_checkpoint_renders_and_inventories(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    config = TinyTrainingConfig(
        iterations=16, validation_interval=4, logging_interval=4
    )
    outcome = run_tiny_training(tmp_path / "runs", config)
    assert outcome.final_iteration == 16
    assert outcome.gaussian_count == config.maximum_gaussian_count
    assert outcome.final_loss < outcome.initial_loss * 0.98
    state = load_state(outcome.run_directory / "status.json")
    assert state.status == "completed"
    assert [item.to_status for item in state.history] == [
        "created",
        "validated",
        "running",
        "completed",
    ]
    scene, metadata = load_gaussian_checkpoint(outcome.checkpoint_path)
    assert scene.count == 3 and metadata["iteration"] == 16
    initial = initialize_parameters(config)
    initial_means = initial["means"].detach().cpu()
    assert not torch.allclose(torch.tensor(scene.means), initial_means)
    assert all(quaternion == (1.0, 0.0, 0.0, 0.0) for quaternion in scene.quaternions)
    rendered = GsplatRenderer().render(
        scene,
        fixture_cameras()["test-up"],
        RenderSettings(
            output_buffers=("rgb", "alpha"),
            warmup_count=0,
            repetition_count=1,
        ),
    )
    assert torch.isfinite(rendered.rgb).all() and torch.isfinite(rendered.alpha).all()
    summary = json.loads(
        (outcome.run_directory / "metadata/training-summary.json").read_text()
    )
    resource = summary["training_resource"]
    assert resource["gpu_final"]["peak_allocated_bytes"] > 0
    assert resource["gpu_final"]["peak_reserved_bytes"] > 0
    assert resource["compliance"]["state"].startswith("compliant")
    assert summary["held_out_camera_ids"] == ["test-up"]
    assert len(summary["validation_history"]) == 4
    inventory = load_inventory(outcome.run_directory / "artifact-inventory.json")
    assert (
        verify_inventory(inventory, outcome.run_directory, strict=True).state
        is VerificationState.VALID
    )


def _invoke(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.experiments.tiny_training",
            "smoke",
            "--root",
            str(root),
            "--profile",
            "smoke",
            "--iterations",
            "4",
            "--json",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_success_overwrite_failure_and_artificial_guard(tmp_path: Path) -> None:
    success_root = tmp_path / "success"
    success = _invoke(success_root)
    assert success.returncode == 0, success.stderr
    assert json.loads(success.stdout)["valid"]
    refused = _invoke(success_root)
    assert refused.returncode == 1
    failure_root = tmp_path / "failure"
    failure = _invoke(failure_root, "--inject-failure")
    assert failure.returncode == 1
    failed_statuses = list(failure_root.rglob("status.json"))
    assert failed_statuses
    failed_state = load_state(failed_statuses[0])
    assert failed_state.status == "failed" and failed_state.failure_ref
    failure_record = json.loads(
        (failed_statuses[0].parent / failed_state.failure_ref).read_text()
    )
    assert failure_record["resource_snapshot"]
    assert failure_record["last_completed_stage"] == "iteration-1"
    assert "metadata/target-generation.json" in failure_record["partial_artifacts"]
    guard_root = tmp_path / "guard"
    guard = _invoke(guard_root, "--artificial-gpu-limit-bytes", "1")
    assert guard.returncode == 1
    guard_statuses = list(guard_root.rglob("status.json"))
    assert guard_statuses and load_state(guard_statuses[0]).status == "failed"
