from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.data.blender import (
    BlenderBackend,
    BlenderEnvironmentError,
    BlenderError,
    BlenderExecutable,
    BlenderInvocation,
    build_blender_arguments,
    discover_blender,
    execute_blender,
)
from gaussweave.data.blender import _translate_path_arguments as translate_paths


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _identity(executable: Path) -> BlenderExecutable:
    return BlenderExecutable(
        executable.as_posix(),
        "4.5.12 LTS",
        "testhash",
        sys.version,
        "wsl",
    )


def _invocation(
    executable: Path,
    script: Path,
    root: Path,
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 2.0,
) -> BlenderInvocation:
    return BlenderInvocation(
        executable,
        script,
        script.parent,
        ("--output-root", str(root), "--label", "spaces & ordinary;chars"),
        root,
        root,
        timeout,
        environment or {},
        BlenderBackend.WSL,
    )


def test_discovery_precedence_auto_explicit_local_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    (repository / "configs").mkdir(parents=True)
    automatic = _executable(tmp_path / "blender", "print('automatic')\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert (
        discover_blender(backend=BlenderBackend.WSL, repository_root=repository)
        == automatic.resolve()
    )

    explicit = _executable(tmp_path / "explicit blender", "print('explicit')\n")
    assert (
        discover_blender(
            backend=BlenderBackend.WSL,
            explicit=explicit,
            repository_root=repository,
        )
        == explicit.resolve()
    )

    monkeypatch.setenv("PATH", "")
    local = _executable(tmp_path / "local blender", "print('local')\n")
    (repository / "configs" / "local.json").write_text(
        json.dumps({"blender_executable": str(local)}), encoding="utf-8"
    )
    assert (
        discover_blender(backend=BlenderBackend.WSL, repository_root=repository)
        == local.resolve()
    )

    (repository / "configs" / "local.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "gaussweave.data.blender.DEFAULT_WSL_EXECUTABLE",
        tmp_path / "missing-default",
    )
    with pytest.raises(BlenderEnvironmentError, match="not a file"):
        discover_blender(backend=BlenderBackend.WSL, repository_root=repository)


def test_invalid_local_configuration_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    (repository / "configs").mkdir(parents=True)
    (repository / "configs" / "local.json").write_text("{", encoding="utf-8")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(BlenderEnvironmentError, match="invalid local configuration"):
        discover_blender(backend=BlenderBackend.WSL, repository_root=repository)


def test_argument_array_preserves_spaces_and_special_characters(tmp_path: Path) -> None:
    root = tmp_path / "output root & evidence"
    root.mkdir()
    executable = _executable(tmp_path / "blender executable", "pass\n")
    script = _executable(root / "script with spaces.py", "pass\n")
    arguments = build_blender_arguments(_invocation(executable, script, root))
    assert arguments[0] == str(executable)
    assert arguments[6] == str(script)
    assert arguments[-1] == "spaces & ordinary;chars"
    assert "--" in arguments


def test_windows_handoff_translates_config_output_and_provenance_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gaussweave.data.blender._translate_windows_path",
        lambda path, working: f"windows:{path.as_posix()}",
    )

    translated = translate_paths(
        (
            "--config",
            "/work/resolved.json",
            "--output-root",
            "/work/output",
            "--provenance-ref",
            "/work/provenance.json",
            "--scene-id",
            "syn-facade-test-s1",
        ),
        tmp_path,
    )

    assert translated == (
        "--config",
        "windows:/work/resolved.json",
        "--output-root",
        "windows:/work/output",
        "--provenance-ref",
        "windows:/work/provenance.json",
        "--scene-id",
        "syn-facade-test-s1",
    )


def test_script_and_output_root_containment(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = _executable(tmp_path / "outside.py", "pass\n")
    executable = _executable(tmp_path / "blender", "pass\n")
    outside_invocation = BlenderInvocation(
        executable,
        outside,
        root,
        (),
        root,
        root,
        1.0,
        {},
        BlenderBackend.WSL,
    )
    with pytest.raises(BlenderError, match="script must remain"):
        build_blender_arguments(outside_invocation)

    script = _executable(root / "inside.py", "pass\n")
    invocation = BlenderInvocation(
        executable,
        script,
        root,
        (),
        root,
        tmp_path / "other-output",
        1.0,
        {},
        BlenderBackend.WSL,
    )
    with pytest.raises(BlenderError, match="output root must remain"):
        build_blender_arguments(invocation)


def test_external_command_success_captures_evidence(tmp_path: Path) -> None:
    root = tmp_path / "success"
    root.mkdir()
    executable = _executable(
        tmp_path / "fake blender",
        "import os,sys\n"
        "print('safe-stdout:'+os.environ['SELECTED'])\n"
        "print('safe-stderr', file=sys.stderr)\n",
    )
    script = _executable(root / "script.py", "pass\n")
    result = execute_blender(
        _invocation(executable, script, root, environment={"SELECTED": "value"}),
        identity=_identity(executable),
    )
    assert result.success and result.exit_code == 0 and not result.timed_out
    assert result.stdout_ref == "stdout.log"
    assert result.stderr_ref == "stderr.log"
    assert "safe-stdout:value" in (root / "stdout.log").read_text(encoding="utf-8")
    assert "safe-stderr" in (root / "stderr.log").read_text(encoding="utf-8")
    assert result.arguments[0] == str(executable)
    assert result.selected_environment == {"SELECTED": "value"}
    assert result.elapsed_seconds >= 0


def test_external_command_failure_has_durable_failure_report(tmp_path: Path) -> None:
    root = tmp_path / "failure"
    root.mkdir()
    executable = _executable(
        tmp_path / "failing blender",
        "import sys\nprint('before-failure')\n"
        "print('failure-detail', file=sys.stderr)\nsys.exit(7)\n",
    )
    script = _executable(root / "script.py", "pass\n")
    result = execute_blender(
        _invocation(executable, script, root), identity=_identity(executable)
    )
    assert not result.success and result.exit_code == 7 and not result.timed_out
    assert result.failure_ref is not None
    failure = json.loads((root / result.failure_ref).read_text(encoding="utf-8"))
    assert failure["exit_code"] == 7
    assert failure["stderr_ref"] == "stderr.log"
    assert list((root / "metadata").glob("failure-*.md"))
    assert not (root / "render.png").exists()


def test_timeout_is_distinct_and_child_is_terminated(tmp_path: Path) -> None:
    root = tmp_path / "timeout"
    root.mkdir()
    executable = _executable(
        tmp_path / "slow blender",
        "import time\nprint('started', flush=True)\ntime.sleep(5)\n",
    )
    script = _executable(root / "script.py", "pass\n")
    result = execute_blender(
        _invocation(executable, script, root, timeout=0.05),
        identity=_identity(executable),
    )
    assert not result.success and result.timed_out and result.exit_code is None
    assert result.elapsed_seconds < 2
    assert result.failure_ref is not None


def test_import_isolation_in_fresh_process() -> None:
    program = (
        "import sys; import gaussweave; import gaussweave.data; "
        "import gaussweave.data.blender; "
        "assert 'bpy' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "bpy" not in sys.modules
    assert os.environ.get("BLENDER_USER_CONFIG") is None
