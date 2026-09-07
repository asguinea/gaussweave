"""CPU-only run lifecycle, structured logging, and failure records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import traceback as traceback_module
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from gaussweave.config.resolution import generate_run_id, pretty_json_bytes
from gaussweave.results.artifacts import ArtifactError, ArtifactPath

STATUS_VERSION = "1.0"
FAILURE_VERSION = "1.0"
MAX_LOG_VALUE_LENGTH = 2048
FAILURE_CATEGORIES = frozenset(
    {
        "invalid_configuration",
        "missing_data",
        "environment_incompatibility",
        "gpu_out_of_memory",
        "cpu_out_of_memory",
        "disk_space",
        "gs_numerical_failure",
        "grammar_induction_failure",
        "invalid_grammar",
        "canonicalization_failure",
        "renderer_failure",
        "metric_failure",
        "interrupted",
        "unknown",
    }
)
EXCLUSION_REASONS = frozenset(
    {
        "corrupted_dataset",
        "generator_defect",
        "invalid_camera_calibration",
        "confirmed_implementation_defect",
        "unrecoverable_hardware_interruption",
        "metric_input_corruption",
    }
)
SECRET_KEYS = frozenset(
    {"password", "passwd", "secret", "token", "api_key", "apikey", "credential"}
)
RUN_ID_RE = re.compile(r"^run-[a-z0-9][a-z0-9._-]*$")
EXPERIMENT_ID_RE = re.compile(r"^exp-[a-z0-9][a-z0-9._-]*-v[0-9]+$")
SCENE_ID_RE = re.compile(r"^(?:syn|real)-[a-z0-9][a-z0-9-]*$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class LifecycleError(RuntimeError):
    """Run lifecycle data or operation is invalid."""


class PersistenceError(LifecycleError):
    """A durable lifecycle record could not be loaded or written."""


class RunStatus(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXCLUDED = "excluded"
    FROZEN = "frozen"


TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.VALIDATED, RunStatus.EXCLUDED}),
    RunStatus.VALIDATED: frozenset({RunStatus.RUNNING, RunStatus.EXCLUDED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.EXCLUDED}
    ),
    RunStatus.COMPLETED: frozenset({RunStatus.EXCLUDED, RunStatus.FROZEN}),
    RunStatus.FAILED: frozenset({RunStatus.FROZEN}),
    RunStatus.EXCLUDED: frozenset({RunStatus.FROZEN}),
    RunStatus.FROZEN: frozenset(),
}


@dataclass(frozen=True)
class Transition:
    sequence: int
    from_status: str | None
    to_status: str
    timestamp: str
    reason: str
    actor: str
    related_id: str | None = None


@dataclass(frozen=True)
class RunState:
    status_version: str
    run_id: str
    experiment_id: str
    scene_id: str | None
    attempt: int
    status: str
    previous_status: str | None
    created_at: str
    updated_at: str
    configuration_ref: str
    configuration_digest: str
    environment_ref: str | None
    artifact_inventory_ref: str | None
    status_message: str
    warnings: tuple[str, ...]
    failure_ref: str | None
    exclusion_ref: str | None
    history: tuple[Transition, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["warnings"] = list(self.warnings)
        value["history"] = [asdict(item) for item in self.history]
        return value


@dataclass(frozen=True)
class ExclusionRecord:
    exclusion_id: str
    run_id: str
    reason: str
    approved_by: str
    approved_at: str
    report_ref: str
    note: str

    def __post_init__(self) -> None:
        if self.reason not in EXCLUSION_REASONS:
            raise LifecycleError(f"unsupported exclusion reason: {self.reason}")
        _portable_reference(self.report_ref)
        if not self.approved_by.strip():
            raise LifecycleError("exclusion approval actor is required")


@dataclass(frozen=True)
class FailureRecord:
    failure_version: str
    failure_id: str
    run_id: str
    experiment_id: str
    scene_id: str | None
    method_id: str | None
    attempt: int
    category: str
    stage: str
    severity: str
    message: str
    exception_type: str | None
    traceback: str | None
    command: tuple[str, ...] | None
    exit_code: int | None
    timed_out: bool
    stdout_ref: str | None
    stderr_ref: str | None
    resolved_configuration_ref: str
    environment_ref: str | None
    resource_snapshot: Mapping[str, Any] | None
    last_completed_stage: str | None
    partial_artifacts: tuple[str, ...]
    included_in_denominator: bool
    retry_metadata: Mapping[str, Any]
    remediation: str | None
    created_at: str

    def __post_init__(self) -> None:
        if self.category not in FAILURE_CATEGORIES:
            raise LifecycleError(f"unsupported failure category: {self.category}")
        if self.severity not in {"critical", "high", "medium", "low"}:
            raise LifecycleError(f"unsupported failure severity: {self.severity}")
        _portable_reference(self.resolved_configuration_ref)
        for reference in (
            self.environment_ref,
            self.stdout_ref,
            self.stderr_ref,
            *self.partial_artifacts,
        ):
            if reference is not None:
                _portable_reference(reference)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command) if self.command is not None else None
        value["partial_artifacts"] = list(self.partial_artifacts)
        value["resource_snapshot"] = (
            dict(self.resource_snapshot) if self.resource_snapshot is not None else None
        )
        value["retry_metadata"] = dict(self.retry_metadata)
        return value


@dataclass(frozen=True)
class CommandResult:
    arguments: tuple[str, ...]
    working_directory: str
    selected_environment: Mapping[str, str]
    started_at: str
    ended_at: str
    elapsed_seconds: float
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    success: bool
    stdout_ref: str | None = None
    stderr_ref: str | None = None


class RunLogger:
    """Append matching readable and JSONL events with privacy normalization."""

    def __init__(
        self,
        run_directory: Path,
        run_id: str,
        *,
        repository_root: Path | None = None,
        secrets: Iterable[str] = (),
    ) -> None:
        self.run_id = run_id
        self.log_directory = run_directory / "logs"
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.human_path = self.log_directory / "run.log"
        self.jsonl_path = self.log_directory / "events.jsonl"
        self.repository_root = repository_root.resolve() if repository_root else None
        self.secrets = tuple(value for value in secrets if value)

    def event(
        self,
        severity: str,
        event_name: str,
        stage: str,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
        exception: BaseException | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "timestamp": timestamp or _now(),
            "severity": severity.upper(),
            "event_name": event_name,
            "run_id": self.run_id,
            "stage": stage,
            "message": _sanitize(message, self.repository_root, self.secrets, key=None),
            "context": _sanitize(
                dict(context or {}), self.repository_root, self.secrets, key=None
            ),
            "exception": (
                {
                    "type": type(exception).__name__,
                    "message": _sanitize(
                        str(exception), self.repository_root, self.secrets, key=None
                    ),
                }
                if exception is not None
                else None
            ),
        }
        with self.jsonl_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        human = (
            f"{event['timestamp']} {event['severity']} "
            f"[{stage}] {event_name}: {event['message']}\n"
        )
        with self.human_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(human)
        return event


def create_state(
    *,
    run_id: str,
    experiment_id: str,
    scene_id: str | None,
    attempt: int,
    configuration_ref: str,
    configuration_digest: str,
    environment_ref: str | None = None,
    timestamp: str | None = None,
) -> RunState:
    """Create a typed state with its initial history entry."""

    if attempt < 1:
        raise LifecycleError("attempt must begin at 1")
    if not RUN_ID_RE.fullmatch(run_id):
        raise LifecycleError(f"invalid run ID: {run_id!r}")
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise LifecycleError(f"invalid experiment ID: {experiment_id!r}")
    if scene_id is not None and not SCENE_ID_RE.fullmatch(scene_id):
        raise LifecycleError(f"invalid scene ID: {scene_id!r}")
    if not DIGEST_RE.fullmatch(configuration_digest):
        raise LifecycleError("configuration digest must be sha256:<64 lowercase hex>")
    _portable_reference(configuration_ref)
    if environment_ref is not None:
        _portable_reference(environment_ref)
    now = timestamp or _now()
    initial = Transition(1, None, RunStatus.CREATED.value, now, "run created", "system")
    return RunState(
        STATUS_VERSION,
        run_id,
        experiment_id,
        scene_id,
        attempt,
        RunStatus.CREATED.value,
        None,
        now,
        now,
        configuration_ref,
        configuration_digest,
        environment_ref,
        None,
        "run created",
        (),
        None,
        None,
        (initial,),
    )


def transition(
    state: RunState,
    to_status: RunStatus,
    *,
    reason: str,
    actor: str,
    timestamp: str | None = None,
    related_id: str | None = None,
    failure_ref: str | None = None,
    exclusion_ref: str | None = None,
    artifact_inventory_ref: str | None = None,
    preserve_failed: bool = False,
) -> RunState:
    """Apply one explicit transition; transitions to the current state are rejected."""

    current = RunStatus(state.status)
    if to_status == current:
        raise LifecycleError(f"run is already {current.value}")
    if to_status not in TRANSITIONS[current]:
        raise LifecycleError(
            f"invalid transition: {current.value} -> {to_status.value}"
        )
    if (
        current is RunStatus.FAILED
        and to_status is RunStatus.FROZEN
        and not preserve_failed
    ):
        raise LifecycleError("freezing a failed run requires preserve_failed=True")
    if to_status is RunStatus.FAILED and not failure_ref:
        raise LifecycleError("failed transition requires a failure reference")
    if to_status is RunStatus.EXCLUDED and not exclusion_ref:
        raise LifecycleError("excluded transition requires an exclusion reference")
    for reference in (failure_ref, exclusion_ref, artifact_inventory_ref):
        if reference is not None:
            _portable_reference(reference)
    now = timestamp or _now()
    entry = Transition(
        len(state.history) + 1,
        current.value,
        to_status.value,
        now,
        reason,
        actor,
        related_id,
    )
    return replace(
        state,
        previous_status=current.value,
        status=to_status.value,
        updated_at=now,
        status_message=reason,
        failure_ref=failure_ref or state.failure_ref,
        exclusion_ref=exclusion_ref or state.exclusion_ref,
        artifact_inventory_ref=artifact_inventory_ref or state.artifact_inventory_ref,
        history=(*state.history, entry),
    )


def initialize_run_directory(root: Path, state: RunState) -> Path:
    """Create an isolated bounded run directory and initial status."""

    root.mkdir(parents=True, exist_ok=True)
    experiment = ArtifactPath.resolve(root, state.experiment_id).local
    run_directory = ArtifactPath.resolve(
        root, f"{state.experiment_id}/{state.run_id}"
    ).local
    if run_directory.exists():
        status_path = run_directory / "status.json"
        if not status_path.is_file():
            raise LifecycleError(
                f"refusing to reuse existing run directory: {run_directory}"
            )
        existing = load_state(status_path)
        if (
            existing.run_id != state.run_id
            or existing.experiment_id != state.experiment_id
        ):
            raise LifecycleError("existing run directory belongs to another run")
        raise LifecycleError(f"run already initialized: {state.run_id}")
    experiment.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir()
    for name in ("metadata", "logs", "artifacts"):
        (run_directory / name).mkdir()
    (run_directory / "resolved_config.json").write_bytes(b"{}\n")
    write_state(state, run_directory / "status.json")
    return run_directory


def write_state(state: RunState, path: Path) -> None:
    """Atomically replace a status file."""

    _atomic_write(path, pretty_json_bytes(state.to_dict()))


def load_state(path: Path) -> RunState:
    """Load and validate a durable status record."""

    if path.is_symlink() or not path.is_file():
        raise PersistenceError(f"status is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        history = tuple(Transition(**item) for item in value.pop("history"))
        warnings = tuple(value.pop("warnings", ()))
        state = RunState(**value, history=history, warnings=warnings)
        _validate_state(state)
        return state
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise PersistenceError(f"corrupt status record {path}: {error}") from error


def write_failure(record: FailureRecord, run_directory: Path) -> tuple[Path, Path]:
    """Write machine-readable failure JSON and template-aligned Markdown."""

    metadata = run_directory / "metadata"
    json_path = metadata / f"{record.failure_id}.json"
    report_path = metadata / f"{record.failure_id}.md"
    _atomic_write(json_path, pretty_json_bytes(record.to_dict()))
    _atomic_write(report_path, _failure_markdown(record).encode("utf-8"))
    return json_path, report_path


@contextmanager
def capture_failures(
    state: RunState,
    run_directory: Path,
    logger: RunLogger,
    *,
    stage: str,
    category: str = "unknown",
    severity: str = "high",
    method_id: str | None = None,
    partial_artifacts: Sequence[str] = (),
    last_completed_stage: str | None = None,
    resource_snapshot: Mapping[str, Any] | None = None,
) -> Iterator[RunState]:
    """Run work, durably record ordinary exceptions, transition failed, and re-raise."""

    active = state
    if RunStatus(active.status) is RunStatus.VALIDATED:
        active = transition(
            active, RunStatus.RUNNING, reason="execution started", actor="runner"
        )
        write_state(active, run_directory / "status.json")
    try:
        yield active
    except Exception as error:
        created_at = _now()
        failure_id = _failure_id(active.run_id, stage, created_at)
        safe_message = str(
            _sanitize(str(error), logger.repository_root, logger.secrets, key=None)
        )
        safe_traceback = str(
            _sanitize(
                "".join(traceback_module.format_exception(error)),
                logger.repository_root,
                logger.secrets,
                key=None,
            )
        )
        record = FailureRecord(
            FAILURE_VERSION,
            failure_id,
            active.run_id,
            active.experiment_id,
            active.scene_id,
            method_id,
            active.attempt,
            category,
            stage,
            severity,
            safe_message,
            type(error).__name__,
            safe_traceback,
            None,
            None,
            False,
            None,
            None,
            active.configuration_ref,
            active.environment_ref,
            resource_snapshot,
            last_completed_stage,
            tuple(partial_artifacts),
            True,
            {"automatic_retry": False},
            None,
            created_at,
        )
        json_path, _ = write_failure(record, run_directory)
        failure_ref = json_path.relative_to(run_directory).as_posix()
        failed = transition(
            active,
            RunStatus.FAILED,
            reason=safe_message,
            actor="exception-capture",
            related_id=failure_id,
            failure_ref=failure_ref,
        )
        write_state(failed, run_directory / "status.json")
        logger.event(
            "ERROR",
            "run_failed",
            stage,
            str(error),
            context={"failure_id": failure_id, "partial_artifacts": partial_artifacts},
            exception=error,
        )
        raise


def run_command(
    arguments: Sequence[str],
    *,
    working_directory: Path,
    environment: Mapping[str, str] | None = None,
    selected_environment_keys: Sequence[str] = (),
    timeout_seconds: float | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> CommandResult:
    """Run an argument array without a shell and capture a structured outcome."""

    if not arguments or not all(isinstance(item, str) and item for item in arguments):
        raise LifecycleError("command must be a non-empty argument array")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise LifecycleError("timeout must be positive")
    env = dict(os.environ)
    if environment:
        env.update(environment)
    selected = {
        key: str(_sanitize(env[key], None, (), key=key))
        for key in selected_environment_keys
        if key in env
    }
    started_at = _now()
    started = time.monotonic()
    timed_out = False
    exit_code: int | None
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=working_directory,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )
        stdout, stderr, exit_code = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = _decode_timeout_output(error.stdout)
        stderr = _decode_timeout_output(error.stderr)
        exit_code = None
    elapsed = time.monotonic() - started
    stdout_ref = _write_command_output(stdout_path, stdout, working_directory)
    stderr_ref = _write_command_output(stderr_path, stderr, working_directory)
    return CommandResult(
        tuple(arguments),
        working_directory.resolve().as_posix(),
        selected,
        started_at,
        _now(),
        elapsed,
        stdout,
        stderr,
        exit_code,
        timed_out,
        not timed_out and exit_code == 0,
        stdout_ref,
        stderr_ref,
    )


def command_failure(
    state: RunState,
    result: CommandResult,
    *,
    stage: str,
    category: str = "unknown",
    created_at: str | None = None,
) -> FailureRecord:
    """Adapt a non-successful command outcome into a failure record."""

    if result.success:
        raise LifecycleError("successful command cannot become a failure record")
    timestamp = created_at or _now()
    return FailureRecord(
        FAILURE_VERSION,
        _failure_id(state.run_id, stage, timestamp),
        state.run_id,
        state.experiment_id,
        state.scene_id,
        None,
        state.attempt,
        category,
        stage,
        "high",
        "external command timed out" if result.timed_out else "external command failed",
        "TimeoutExpired" if result.timed_out else None,
        None,
        result.arguments,
        result.exit_code,
        result.timed_out,
        result.stdout_ref,
        result.stderr_ref,
        state.configuration_ref,
        state.environment_ref,
        None,
        None,
        (),
        True,
        {"automatic_retry": False},
        None,
        timestamp,
    )


def _validate_state(state: RunState) -> None:
    RunStatus(state.status)
    if state.status_version != STATUS_VERSION or state.attempt < 1:
        raise LifecycleError("unsupported or invalid status record")
    if not state.history or state.history[-1].to_status != state.status:
        raise LifecycleError("transition history does not match current status")
    if (
        not RUN_ID_RE.fullmatch(state.run_id)
        or not EXPERIMENT_ID_RE.fullmatch(state.experiment_id)
        or (state.scene_id is not None and not SCENE_ID_RE.fullmatch(state.scene_id))
        or not DIGEST_RE.fullmatch(state.configuration_digest)
    ):
        raise LifecycleError("invalid run identity in status record")
    if tuple(item.sequence for item in state.history) != tuple(
        range(1, len(state.history) + 1)
    ):
        raise LifecycleError("transition history sequence is invalid")
    previous: str | None = None
    for item in state.history:
        if item.from_status != previous:
            raise LifecycleError("transition history chain is invalid")
        RunStatus(item.to_status)
        previous = item.to_status
    _portable_reference(state.configuration_ref)
    for reference in (
        state.environment_ref,
        state.artifact_inventory_ref,
        state.failure_ref,
        state.exclusion_ref,
    ):
        if reference is not None:
            _portable_reference(reference)


def _portable_reference(reference: str) -> None:
    try:
        ArtifactPath.resolve(Path.cwd(), reference)
    except ArtifactError as error:
        raise LifecycleError(f"unsafe portable reference: {reference!r}") from error


def _sanitize(
    value: Any,
    repository_root: Path | None,
    secrets: Sequence[str],
    *,
    key: str | None,
) -> Any:
    if key is not None and any(secret in key.lower() for secret in SECRET_KEYS):
        return "<redacted>"
    if isinstance(value, bytes):
        return f"<binary:{len(value)} bytes>"
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else f"<non-finite:{value}>"
    if isinstance(value, str):
        text = value
        home = Path.home().as_posix()
        text = text.replace(str(Path.home()), "~").replace(home, "~")
        if repository_root is not None:
            text = text.replace(str(repository_root), "<repo>").replace(
                repository_root.as_posix(), "<repo>"
            )
        for secret in secrets:
            text = text.replace(secret, "<redacted>")
        return text[:MAX_LOG_VALUE_LENGTH] + (
            "<truncated>" if len(text) > MAX_LOG_VALUE_LENGTH else ""
        )
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(
                item_value, repository_root, secrets, key=str(item_key)
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence):
        return [
            _sanitize(item, repository_root, secrets, key=None) for item in value[:100]
        ]
    return _sanitize(repr(value), repository_root, secrets, key=key)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    except OSError as error:
        raise PersistenceError(f"unable to atomically write {path}: {error}") from error
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _failure_markdown(record: FailureRecord) -> str:
    command = " ".join(record.command or ())
    evidence = record.traceback or command or record.message
    artifacts = "\n".join(
        f"| artifact | `{item}` | partial | |" for item in record.partial_artifacts
    )
    return (
        f"# Failure Report — {record.run_id}\n\n"
        f"**Failure ID:** `{record.failure_id}`\n"
        f"**Run ID:** `{record.run_id}`\n"
        f"**Experiment ID:** `{record.experiment_id}`\n"
        f"**Scene ID:** `{record.scene_id or 'unknown'}`\n"
        f"**Method ID:** `{record.method_id or 'unknown'}`\n"
        f"**Attempt:** `{record.attempt}`\n"
        f"**Status:** open\n"
        f"**Created at:** `{record.created_at}`\n\n"
        f"## 1. Summary\n\n{record.message}\n\n"
        f"## 2. Failure classification\n\n"
        f"**Category:** {record.category}\n"
        f"**Stage:** {record.stage}\n"
        f"**Severity:** {record.severity}\n"
        f"**Included in experiment denominator:** "
        f"{'yes' if record.included_in_denominator else 'no'}\n\n"
        f"## 3. Governing configuration\n\n"
        f"- Resolved configuration: `{record.resolved_configuration_ref}`\n"
        f"- Environment reference: `{record.environment_ref or 'not supplied'}`\n\n"
        f"## 4. What happened\n\n{record.message}\n\n"
        f"## 5. Expected behavior\n\nThe stage should complete successfully.\n\n"
        f"## 6. Error evidence\n\n```text\n{evidence}\n```\n\n"
        f"## 7. Resource state\n\n"
        f"`{json.dumps(record.resource_snapshot, sort_keys=True, default=repr)}`\n\n"
        f"## 8. Last completed stage\n\n"
        f"- Stage: {record.last_completed_stage or 'none'}\n\n"
        f"## 9. Produced artifacts\n\n"
        f"| Artifact | Path | Status | Checksum |\n"
        f"|---|---|---|---|\n{artifacts or '| none | | | |'}\n\n"
        f"## 10. Scientific impact\n\nFailure retained in denominator.\n\n"
        f"## 11. Root-cause assessment\n\nUnknown pending investigation.\n\n"
        f"## 12. Retry or remediation\n\nAutomatic retry: no.\n\n"
        f"## 13. Exclusion decision\n\nExcluded: no.\n\n"
        f"## 14. Resolution\n\nOpen.\n\n"
        f"## 15. Related records\n\n- Status: `status.json`\n"
    )


def _failure_id(run_id: str, stage: str, timestamp: str) -> str:
    suffix = hashlib.sha256(f"{run_id}\0{stage}\0{timestamp}".encode()).hexdigest()[:12]
    return f"failure-{suffix}"


def _write_command_output(path: Path | None, content: str, cwd: Path) -> str | None:
    if path is None:
        return None
    if path.is_absolute():
        try:
            portable = path.resolve().relative_to(cwd.resolve()).as_posix()
        except ValueError as error:
            raise LifecycleError(
                "command output must remain within working directory"
            ) from error
    else:
        portable = path.as_posix()
    resolved = ArtifactPath.resolve(cwd, portable).local
    _atomic_write(resolved, content.encode("utf-8"))
    return portable


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _demo(root: Path, mode: str) -> int:
    digest = "sha256:" + "0" * 64
    experiment_id = "exp-lifecycle-demo-v1"
    scene_id = "syn-demo"
    run_id = generate_run_id(
        experiment_id=experiment_id,
        scene_id=scene_id,
        seed=1,
        scientific_configuration_digest=digest,
        attempt=1,
    )
    state = create_state(
        run_id=run_id,
        experiment_id=experiment_id,
        scene_id=scene_id,
        attempt=1,
        configuration_ref="resolved_config.json",
        configuration_digest=digest,
    )
    run_directory = initialize_run_directory(root, state)
    logger = RunLogger(run_directory, run_id, repository_root=Path.cwd())
    state = transition(
        state, RunStatus.VALIDATED, reason="demo configuration validated", actor="demo"
    )
    write_state(state, run_directory / "status.json")
    logger.event("INFO", "run_validated", "validation", "demo validated")
    try:
        with capture_failures(
            state,
            run_directory,
            logger,
            stage="demo",
            partial_artifacts=("resolved_config.json",),
        ) as active:
            if mode == "failure":
                raise RuntimeError("controlled demo failure")
            completed = transition(
                active, RunStatus.COMPLETED, reason="demo completed", actor="demo"
            )
            write_state(completed, run_directory / "status.json")
            logger.event("INFO", "run_completed", "demo", "controlled success")
    except RuntimeError:
        print(run_directory.as_posix())
        return 1
    print(run_directory.as_posix())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--root", required=True, type=Path)
    demo.add_argument("--mode", choices=("success", "failure"), required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        return _demo(options.root, options.mode)
    except LifecycleError as error:
        print(f"lifecycle error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
