from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gaussweave.results.artifacts", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_inventory_and_verify_cli(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("fixture")
    output = tmp_path / "inventory.json"
    created = _run(
        "inventory", "--root", str(tmp_path), "--output", str(output), "--json"
    )
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["state"] == "created"
    verified = _run(
        "verify", "--root", str(tmp_path), "--inventory", str(output), "--json"
    )
    assert verified.returncode == 0
    assert json.loads(verified.stdout)["state"] == "valid"
    assert (
        _run("inventory", "--root", str(tmp_path), "--output", str(output)).returncode
        == 2
    )
    (tmp_path / "unexpected").write_text("x")
    failed = _run(
        "verify", "--root", str(tmp_path), "--inventory", str(output), "--strict"
    )
    assert failed.returncode == 1
