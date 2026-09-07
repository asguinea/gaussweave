from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaussweave.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/scenes/facade-development-s101.json"


@pytest.mark.integration
@pytest.mark.cli
def test_camera_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["camera", "generate", "--help"])
    assert caught.value.code == 0
    assert "--camera-seed" in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.cli
def test_camera_cli_generate_and_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "camera",
                "generate",
                "--config",
                str(CONFIG),
                "--root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["camera_count"] == 40
    assert (
        main(
            [
                "camera",
                "validate",
                str(tmp_path / "cameras/cameras.json"),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid"] is True


@pytest.mark.integration
@pytest.mark.cli
def test_camera_cli_bad_family_and_safe_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "camera",
        "generate",
        "--config",
        str(CONFIG),
        "--root",
        str(tmp_path),
        "--family",
        "corridor",
        "--json",
    ]
    assert main(arguments) == 3
    assert json.loads(capsys.readouterr().out)["valid"] is False
