"""Temporary module-level validation invocation tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.schema, pytest.mark.cli]

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "validation"


def run_validation(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the temporary validation module."""

    return subprocess.run(
        [sys.executable, "-m", "gaussweave.config.validation", *arguments],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_cli_accepts_valid_document() -> None:
    result = run_validation(
        "--kind",
        "experiment",
        str(FIXTURES / "valid_experiment.json"),
    )

    assert result.returncode == 0
    assert "valid experiment document" in result.stdout


def test_cli_rejects_invalid_document() -> None:
    result = run_validation(
        "--kind",
        "experiment",
        str(FIXTURES / "invalid_experiment.json"),
    )

    assert result.returncode == 1
    assert "experiment schema" in result.stdout


def test_cli_emits_valid_json_error() -> None:
    result = run_validation(
        "--json",
        "--kind",
        "experiment",
        str(FIXTURES / "invalid_experiment.json"),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert payload["errors"]
