from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.results.artifacts import (
    VerificationState,
    load_inventory,
    verify_inventory,
)

pytestmark = pytest.mark.gpu


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gaussweave.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_formal_gpu_render_train_result_and_resource_guard(tmp_path: Path) -> None:
    render_root = tmp_path / "render"
    rendered = _cli(
        "render",
        "smoke",
        "--profile",
        "smoke",
        "--output",
        str(render_root),
        "--json",
    )
    assert rendered.returncode == 0, rendered.stderr
    assert json.loads(rendered.stdout)["valid"] is True
    assert (render_root / "render-metadata.json").is_file()

    training_root = tmp_path / "training"
    trained = _cli(
        "train",
        "smoke",
        "--root",
        str(training_root),
        "--profile",
        "smoke",
        "--iterations",
        "4",
        "--json",
    )
    assert trained.returncode == 0, trained.stderr
    run = Path(json.loads(trained.stdout)["run_directory"])
    result_path = tmp_path / "result.json"
    built = _cli(
        "result",
        "build-smoke",
        "--run-root",
        str(training_root),
        "--output",
        str(result_path),
        "--json",
    )
    assert built.returncode == 0, built.stderr
    assert json.loads(built.stdout)["status"] == "completed"
    validated = _cli("result", "validate", str(result_path), "--json")
    assert validated.returncode == 0
    assert json.loads(validated.stdout)["valid"] is True
    inventory = load_inventory(run / "artifact-inventory.json")
    assert (
        verify_inventory(inventory, run, strict=True).state is VerificationState.VALID
    )

    guard = _cli(
        "resource",
        "demo",
        "--profile",
        "smoke",
        "--device",
        "cuda:0",
        "--artificial-gpu-limit-bytes",
        "1",
        "--json",
    )
    assert guard.returncode == 5
    assert isinstance(json.loads(guard.stdout), dict)

    failed_training = _cli(
        "train",
        "smoke",
        "--root",
        str(tmp_path / "failed-training"),
        "--profile",
        "smoke",
        "--iterations",
        "4",
        "--inject-failure",
        "--json",
    )
    assert failed_training.returncode == 6
    assert json.loads(failed_training.stdout)["valid"] is False
