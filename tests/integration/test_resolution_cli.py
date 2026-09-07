"""Temporary deterministic resolution module invocation tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.cli]

ROOT = Path(__file__).parents[2]
BASE = ROOT / "tests" / "fixtures" / "validation" / "valid_experiment.json"
OVERLAY = ROOT / "tests" / "fixtures" / "resolution" / "operational-overlay.json"
INCOMPLETE = ROOT / "tests" / "fixtures" / "resolution" / "incomplete-layer.json"


def run_resolution(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the temporary resolution module."""

    return subprocess.run(
        [sys.executable, "-m", "gaussweave.config.resolution", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_writes_all_outputs_and_human_summary(tmp_path: Path) -> None:
    output = tmp_path / "resolved.json"
    result = run_resolution(
        "--layer",
        str(BASE),
        "--layer",
        str(OVERLAY),
        "--output",
        str(output),
        "--scene-id",
        "syn-fixture",
        "--seed",
        "0",
    )

    assert result.returncode == 0
    assert "scientific digest: sha256:" in result.stdout
    assert "full digest: sha256:" in result.stdout
    assert "run ID: run-" in result.stdout
    assert output.is_file()
    assert output.with_suffix(".canonical.json").is_file()
    assert output.with_suffix(".provenance.json").is_file()
    assert output.with_suffix(".digests.json").is_file()


def test_cli_json_summary_and_ordered_layers(tmp_path: Path) -> None:
    output = tmp_path / "resolved.json"
    result = run_resolution(
        "--json",
        "--layer",
        str(BASE),
        "--layer",
        str(OVERLAY),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["scientific_digest"].startswith("sha256:")
    resolved = json.loads(output.read_text(encoding="utf-8"))
    assert resolved["execution"]["attempt"] == 2
    assert resolved["outputs"]["run_root"] == "runs/alternate"
    provenance = json.loads(
        output.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    assert [layer["order"] for layer in provenance["layers"]] == [1, 2]


def test_cli_invalid_composition_is_nonzero(tmp_path: Path) -> None:
    result = run_resolution(
        "--json",
        "--layer",
        str(INCOMPLETE),
        "--output",
        str(tmp_path / "invalid.json"),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "required property" in payload["error"]
