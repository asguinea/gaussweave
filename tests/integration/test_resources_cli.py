from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gaussweave.accounting.resources", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cpu_demo_human_json_and_output(tmp_path: Path) -> None:
    output = tmp_path / "resource.json"
    result = _run(
        "demo",
        "--profile",
        "smoke",
        "--device",
        "cpu",
        "--output",
        str(output),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["compliance"]["state"].startswith("compliant")
    assert json.loads(output.read_text())["operation"] == "resource-demo"
    human = _run("demo", "--profile", "smoke", "--device", "cpu")
    assert human.returncode == 0 and "resource-demo:" in human.stdout


def test_artificial_cpu_guard_demo(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    result = _run(
        "demo",
        "--profile",
        "smoke",
        "--device",
        "cpu",
        "--artificial-cpu-limit-bytes",
        "1",
        "--output",
        str(output),
        "--json",
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["state"] == "resource_failure"
    assert output.is_file()
