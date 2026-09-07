from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.cli import ExitCode

FIXTURES = Path(__file__).parents[1] / "fixtures" / "validation"


def _module(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gaussweave.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_root_module_and_complete_help_tree() -> None:
    root = _module("--help")
    assert root.returncode == 0
    for group in (
        "env",
        "blender",
        "schema",
        "config",
        "artifact",
        "lifecycle",
        "resource",
        "render",
        "train",
        "result",
    ):
        assert group in root.stdout
    leaves = (
        ("env", "check"),
        ("blender", "qualify"),
        ("schema", "validate"),
        ("config", "validate"),
        ("config", "resolve"),
        ("artifact", "inventory"),
        ("artifact", "verify"),
        ("lifecycle", "demo"),
        ("resource", "demo"),
        ("render", "smoke"),
        ("train", "smoke"),
        ("result", "build-smoke"),
        ("result", "validate"),
    )
    for command in leaves:
        completed = _module(*command, "--help")
        assert completed.returncode == 0, command
        assert "usage:" in completed.stdout


def test_schema_and_config_validation_json_and_exit_codes(tmp_path: Path) -> None:
    valid = FIXTURES / "valid_experiment.json"
    schema = _module("schema", "validate", "--kind", "experiment", str(valid), "--json")
    assert schema.returncode == ExitCode.SUCCESS
    assert json.loads(schema.stdout)["valid"] is True
    config = _module("config", "validate", str(valid), "--json")
    assert config.returncode == ExitCode.SUCCESS
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    failed = _module("config", "validate", str(invalid), "--json")
    assert failed.returncode == ExitCode.VALIDATION
    assert json.loads(failed.stdout)["valid"] is False
    missing = _module("schema", "validate", "--kind", "grammar", "missing.json")
    assert missing.returncode == ExitCode.VALIDATION
    usage = _module("schema", "validate")
    assert usage.returncode == ExitCode.USAGE


def test_config_resolution_digest_provenance_and_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "resolved.json"
    provenance = tmp_path / "provenance.json"
    completed = _module(
        "config",
        "resolve",
        "--layer",
        str(FIXTURES / "valid_experiment.json"),
        "--output",
        str(output),
        "--provenance-output",
        str(provenance),
        "--json",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["scientific_digest"].startswith("sha256:")
    assert payload["outputs"]["resolved_output"]
    assert output.is_file() and provenance.is_file()
    refused = _module(
        "config",
        "resolve",
        "--layer",
        str(FIXTURES / "valid_experiment.json"),
        "--output",
        str(output),
        "--json",
    )
    assert refused.returncode == ExitCode.VALIDATION


def test_artifact_inventory_verify_strict_and_integrity_exit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.txt").write_text("evidence", encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    created = _module(
        "artifact",
        "inventory",
        "--root",
        str(root),
        "--output",
        str(inventory),
        "--json",
    )
    assert created.returncode == 0
    assert json.loads(created.stdout)["artifact_count"] == 1
    refused = _module(
        "artifact",
        "inventory",
        "--root",
        str(root),
        "--output",
        str(inventory),
        "--json",
    )
    assert refused.returncode == ExitCode.ARTIFACT_INTEGRITY
    valid = _module(
        "artifact",
        "verify",
        "--root",
        str(root),
        "--inventory",
        str(inventory),
        "--strict",
        "--json",
    )
    assert valid.returncode == 0 and json.loads(valid.stdout)["valid"]
    (root / "unexpected.txt").write_text("x", encoding="utf-8")
    invalid = _module(
        "artifact",
        "verify",
        "--root",
        str(root),
        "--inventory",
        str(inventory),
        "--strict",
        "--json",
    )
    assert invalid.returncode == ExitCode.ARTIFACT_INTEGRITY
    assert json.loads(invalid.stdout)["valid"] is False


def test_environment_human_json_output_and_quiet(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"
    human = _module("env", "check", "--repo-root", str(Path.cwd()))
    assert human.returncode == 0 and "WSL:" in human.stdout
    structured = _module(
        "env",
        "check",
        "--repo-root",
        str(Path.cwd()),
        "--output",
        str(output),
        "--json",
    )
    assert structured.returncode == 0 and output.is_file()
    assert isinstance(json.loads(structured.stdout), dict)
    quiet = _module(
        "--quiet", "config", "validate", str(FIXTURES / "valid_experiment.json")
    )
    assert quiet.returncode == 0 and quiet.stdout == ""


def test_lifecycle_resource_and_documented_exit_classes(tmp_path: Path) -> None:
    lifecycle = _module(
        "lifecycle",
        "demo",
        "--root",
        str(tmp_path / "lifecycle-success"),
        "--mode",
        "success",
    )
    assert lifecycle.returncode == 0
    controlled = _module(
        "lifecycle",
        "demo",
        "--root",
        str(tmp_path / "lifecycle-failure"),
        "--mode",
        "failure",
    )
    assert controlled.returncode == ExitCode.OPERATION
    resource = _module(
        "resource",
        "demo",
        "--profile",
        "smoke",
        "--device",
        "cpu",
        "--artificial-cpu-limit-bytes",
        "1",
        "--json",
    )
    assert resource.returncode == ExitCode.RESOURCE
    assert isinstance(json.loads(resource.stdout), dict)
    environment = _module(
        "env",
        "check",
        "--output",
        str(tmp_path),
        "--json",
    )
    assert environment.returncode == ExitCode.ENVIRONMENT
    assert json.loads(environment.stdout)["exit_code"] == ExitCode.ENVIRONMENT


def test_cpu_commands_are_fresh_process_gpu_import_isolated() -> None:
    fixture = FIXTURES / "valid_experiment.json"
    program = (
        "import sys; from gaussweave.cli import main; "
        f"code=main(['config','validate',r'{fixture}','--json']); "
        "assert code == 0; "
        "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "command",
    [
        ("schema", "validate", "--kind", "experiment", "missing.json", "--json"),
        ("artifact", "verify", "--root", ".", "--inventory", "missing.json", "--json"),
    ],
)
def test_json_failures_are_one_clean_object(command: tuple[str, ...]) -> None:
    completed = _module(*command)
    assert completed.returncode != 0
    assert isinstance(json.loads(completed.stdout), dict)
    assert "\x1b[" not in completed.stdout
