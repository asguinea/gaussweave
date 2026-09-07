from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gaussweave.experiments.lifecycle import (
    EXCLUSION_REASONS,
    ExclusionRecord,
    LifecycleError,
    PersistenceError,
    RunLogger,
    RunStatus,
    capture_failures,
    command_failure,
    create_state,
    initialize_run_directory,
    load_state,
    run_command,
    transition,
    write_state,
)
from gaussweave.results.artifacts import (
    ArtifactDeclaration,
    VerificationState,
    build_inventory,
    verify_inventory,
)

DIGEST = "sha256:" + "1" * 64


def _state(timestamp: str = "2026-01-01T00:00:00+00:00"):
    return create_state(
        run_id="run-exp-demo-v1-syn-demo-s1-111111111111-a1",
        experiment_id="exp-demo-v1",
        scene_id="syn-demo",
        attempt=1,
        configuration_ref="resolved_config.json",
        configuration_digest=DIGEST,
        timestamp=timestamp,
    )


def _running():
    state = transition(
        _state(),
        RunStatus.VALIDATED,
        reason="validated",
        actor="test",
        timestamp="2026-01-01T00:00:01+00:00",
    )
    return transition(
        state,
        RunStatus.RUNNING,
        reason="running",
        actor="test",
        timestamp="2026-01-01T00:00:02+00:00",
    )


def test_valid_success_failed_excluded_and_frozen_paths() -> None:
    completed = transition(_running(), RunStatus.COMPLETED, reason="done", actor="test")
    assert (
        transition(completed, RunStatus.FROZEN, reason="freeze", actor="test").status
        == "frozen"
    )
    failed = transition(
        _running(),
        RunStatus.FAILED,
        reason="failed",
        actor="test",
        failure_ref="metadata/failure.json",
    )
    assert (
        transition(
            failed,
            RunStatus.FROZEN,
            reason="preserve",
            actor="test",
            preserve_failed=True,
        ).status
        == "frozen"
    )
    excluded = transition(
        _state(),
        RunStatus.EXCLUDED,
        reason="excluded",
        actor="approver",
        exclusion_ref="metadata/exclusion.json",
    )
    assert (
        transition(excluded, RunStatus.FROZEN, reason="freeze", actor="test").status
        == "frozen"
    )


def test_invalid_skips_reopening_noop_and_failed_freeze() -> None:
    with pytest.raises(LifecycleError):
        transition(_state(), RunStatus.RUNNING, reason="skip", actor="test")
    with pytest.raises(LifecycleError):
        transition(_state(), RunStatus.CREATED, reason="noop", actor="test")
    completed = transition(_running(), RunStatus.COMPLETED, reason="done", actor="test")
    with pytest.raises(LifecycleError):
        transition(completed, RunStatus.RUNNING, reason="reopen", actor="test")
    failed = transition(
        _running(),
        RunStatus.FAILED,
        reason="failed",
        actor="test",
        failure_ref="metadata/failure.json",
    )
    with pytest.raises(LifecycleError):
        transition(failed, RunStatus.FROZEN, reason="freeze", actor="test")


def test_status_round_trip_and_corruption(tmp_path: Path) -> None:
    state = _running()
    output = tmp_path / "status.json"
    write_state(state, output)
    assert load_state(output) == state
    output.write_text("{broken", encoding="utf-8")
    with pytest.raises(PersistenceError):
        load_state(output)


def test_atomic_replace_failure_preserves_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "status.json"
    write_state(_state(), output)
    before = output.read_bytes()

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("controlled replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(PersistenceError):
        write_state(_running(), output)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_run_directory_isolation(tmp_path: Path) -> None:
    directory = initialize_run_directory(tmp_path, _state())
    assert {item.name for item in directory.iterdir()} >= {
        "resolved_config.json",
        "metadata",
        "logs",
        "artifacts",
        "status.json",
    }
    with pytest.raises(LifecycleError):
        initialize_run_directory(tmp_path, _state())
    other = replace(_state(), run_id="run-other")
    second = initialize_run_directory(tmp_path, other)
    assert second != directory


def test_structured_logs_and_privacy(tmp_path: Path) -> None:
    logger = RunLogger(
        tmp_path,
        "run-test",
        repository_root=tmp_path,
        secrets=("literal-secret",),
    )
    event = logger.event(
        "info",
        "event",
        "stage",
        f"at {tmp_path} and {Path.home() / 'private'} literal-secret",
        context={
            "api_token": "do-not-show",
            "binary": b"abc",
            "large": "x" * 3000,
        },
    )
    payload = (tmp_path / "logs/events.jsonl").read_text()
    assert event["run_id"] == "run-test"
    assert all(
        secret not in payload
        for secret in (str(tmp_path), str(Path.home()), "literal-secret", "do-not-show")
    )
    assert "~/" in payload and "<redacted>" in payload
    assert "event" in (tmp_path / "logs/run.log").read_text()


def test_exception_capture_writes_failure_and_partial_artifacts(tmp_path: Path) -> None:
    initial = transition(_state(), RunStatus.VALIDATED, reason="valid", actor="test")
    directory = initialize_run_directory(tmp_path, initial)
    logger = RunLogger(directory, initial.run_id)
    with (
        pytest.raises(ValueError, match="controlled"),
        capture_failures(
            initial,
            directory,
            logger,
            stage="unit",
            partial_artifacts=("artifacts/partial.txt",),
            last_completed_stage="setup",
        ),
    ):
        raise ValueError("controlled")
    failed = load_state(directory / "status.json")
    assert failed.status == "failed"
    record_path = directory / (failed.failure_ref or "")
    record = json.loads(record_path.read_text())
    assert record["partial_artifacts"] == ["artifacts/partial.txt"]
    assert "ValueError" in record["traceback"]
    assert record_path.with_suffix(".md").is_file()
    assert (directory / "logs/events.jsonl").is_file()


def test_external_command_success_failure_timeout_and_redirection(
    tmp_path: Path,
) -> None:
    success = run_command(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        working_directory=tmp_path,
        stdout_path=Path("artifacts/stdout.txt"),
        stderr_path=Path("artifacts/stderr.txt"),
    )
    assert success.success and success.exit_code == 0
    assert success.stdout.strip() == "out" and success.stderr.strip() == "err"
    assert success.elapsed_seconds >= 0
    failure = run_command(
        [sys.executable, "-c", "raise SystemExit(7)"], working_directory=tmp_path
    )
    assert not failure.success and failure.exit_code == 7
    assert command_failure(_running(), failure, stage="command").exit_code == 7
    timeout = run_command(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        working_directory=tmp_path,
        timeout_seconds=0.01,
    )
    assert timeout.timed_out and timeout.exit_code is None
    assert command_failure(_running(), timeout, stage="command").timed_out


def test_exclusion_reasons_are_closed() -> None:
    for reason in EXCLUSION_REASONS:
        record = ExclusionRecord(
            "exclusion-1",
            "run-1",
            reason,
            "reviewer",
            "2026-01-01T00:00:00+00:00",
            "metadata/exclusion.md",
            "approved",
        )
        assert record.reason == reason
    for reason in ("poor_performance", "detector_failure", "unfavorable_metrics"):
        with pytest.raises(LifecycleError):
            ExclusionRecord(
                "exclusion-1",
                "run-1",
                reason,
                "reviewer",
                "2026-01-01T00:00:00+00:00",
                "metadata/exclusion.md",
                "not allowed",
            )


def test_failure_artifacts_integrate_with_inventory(tmp_path: Path) -> None:
    initial = transition(_state(), RunStatus.VALIDATED, reason="valid", actor="test")
    directory = initialize_run_directory(tmp_path, initial)
    logger = RunLogger(directory, initial.run_id)
    with (
        pytest.raises(RuntimeError),
        capture_failures(initial, directory, logger, stage="integration"),
    ):
        raise RuntimeError("failure")
    failed = load_state(directory / "status.json")
    paths = [
        "status.json",
        failed.failure_ref or "",
        (Path(failed.failure_ref or "").with_suffix(".md")).as_posix(),
        "logs/events.jsonl",
        "logs/run.log",
    ]
    inventory = build_inventory(
        directory,
        [
            ArtifactDeclaration(f"record-{index}", "failure_report", path)
            for index, path in enumerate(paths)
        ],
        inventory_id="inventory-failure",
    )
    assert verify_inventory(inventory, directory).state is VerificationState.VALID


def test_cpu_import_isolation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import gaussweave.experiments.lifecycle; "
                "assert 'torch' not in sys.modules; "
                "assert 'gsplat' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
