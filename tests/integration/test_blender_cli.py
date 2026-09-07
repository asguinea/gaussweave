from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.cli import ExitCode
from gaussweave.data.blender import BlenderBackend, discover_blender

pytestmark = [pytest.mark.integration, pytest.mark.cli, pytest.mark.blender]


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gaussweave.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def blender_executable() -> Path:
    try:
        return discover_blender(backend=BlenderBackend.WSL)
    except Exception as error:
        pytest.skip(f"Blender unavailable for CLI qualification: {error}")


def test_blender_help_human_and_json_cpu(
    tmp_path: Path, blender_executable: Path
) -> None:
    help_result = _cli("blender", "qualify", "--help")
    assert help_result.returncode == 0
    assert "--backend" in help_result.stdout and "--device" in help_result.stdout
    human = _cli(
        "blender",
        "qualify",
        "--root",
        str(tmp_path / "human"),
        "--blender-executable",
        str(blender_executable),
    )
    assert human.returncode == 0
    assert "Blender 4.5.12 LTS: qualified" in human.stdout
    structured = _cli(
        "blender",
        "qualify",
        "--root",
        str(tmp_path / "json"),
        "--blender-executable",
        str(blender_executable),
        "--json",
    )
    assert structured.returncode == 0
    payload = json.loads(structured.stdout)
    assert payload["valid"] and payload["inventory_state"] == "valid"


@pytest.mark.gpu
def test_blender_cli_gpu_probe(tmp_path: Path, blender_executable: Path) -> None:
    completed = _cli(
        "blender",
        "qualify",
        "--root",
        str(tmp_path / "gpu"),
        "--device",
        "gpu",
        "--blender-executable",
        str(blender_executable),
        "--json",
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] in {"qualified", "unsupported"}


def test_blender_cli_missing_executable_and_overwrite_protection(
    tmp_path: Path,
) -> None:
    missing = _cli(
        "blender",
        "qualify",
        "--root",
        str(tmp_path / "missing-root"),
        "--blender-executable",
        str(tmp_path / "missing-blender"),
        "--json",
    )
    assert missing.returncode == ExitCode.ENVIRONMENT
    assert json.loads(missing.stdout)["valid"] is False

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated.txt").write_text("preserve")
    refused = _cli(
        "blender",
        "qualify",
        "--root",
        str(occupied),
        "--overwrite",
        "--json",
    )
    assert refused.returncode == ExitCode.OPERATION
    assert json.loads(refused.stdout)["valid"] is False
    assert (occupied / "unrelated.txt").read_text() == "preserve"
