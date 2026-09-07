from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _demo(root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.experiments.lifecycle",
            "demo",
            "--root",
            str(root),
            "--mode",
            mode,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_success_demo(tmp_path: Path) -> None:
    result = _demo(tmp_path, "success")
    assert result.returncode == 0, result.stderr
    run_directory = Path(result.stdout.strip())
    status = json.loads((run_directory / "status.json").read_text())
    assert status["status"] == "completed"
    assert len(status["history"]) == 4
    assert (run_directory / "logs/events.jsonl").is_file()
    assert (run_directory / "logs/run.log").is_file()


def test_failure_demo(tmp_path: Path) -> None:
    result = _demo(tmp_path, "failure")
    assert result.returncode == 1
    run_directory = Path(result.stdout.strip())
    status = json.loads((run_directory / "status.json").read_text())
    assert status["status"] == "failed"
    assert len(status["history"]) == 4
    failure = run_directory / status["failure_ref"]
    assert failure.is_file() and failure.with_suffix(".md").is_file()
    assert (run_directory / "logs/events.jsonl").is_file()
    assert (run_directory / "logs/run.log").is_file()
