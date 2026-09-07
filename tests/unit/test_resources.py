from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gaussweave.accounting import resources
from gaussweave.accounting.resources import (
    ComplianceState,
    DiskSnapshot,
    ResourceGuardError,
    ResourceProfile,
    classify_compliance,
    cpu_memory_snapshot,
    disk_snapshot,
    guard_with_lifecycle,
    load_profile,
    load_profile_file,
    nvidia_snapshot,
    time_operation,
)
from gaussweave.experiments.lifecycle import (
    RunLogger,
    RunStatus,
    create_state,
    initialize_run_directory,
    load_state,
    transition,
)
from gaussweave.results.artifacts import (
    ArtifactDeclaration,
    VerificationState,
    build_inventory,
    verify_inventory,
)

DIGEST = "sha256:" + "1" * 64


def test_cpu_memory_snapshot_fields() -> None:
    snapshot = cpu_memory_snapshot()
    assert snapshot.process_id > 0
    assert snapshot.source
    assert snapshot.process_rss_bytes is None or snapshot.process_rss_bytes > 0
    assert (
        snapshot.total_system_ram_bytes is None or snapshot.total_system_ram_bytes > 0
    )
    assert (
        snapshot.available_system_ram_bytes is None
        or snapshot.available_system_ram_bytes >= 0
    )


def test_disk_exact_calculation_and_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1000, used=400, free=600),
    )
    passed = disk_snapshot(tmp_path, safety_reserve_bytes=400, warning_fraction=0.1)
    assert (passed.total_bytes, passed.used_bytes, passed.free_bytes) == (
        1000,
        400,
        600,
    )
    assert passed.preflight_state == "pass"
    warning = disk_snapshot(tmp_path, safety_reserve_bytes=580, warning_fraction=0.1)
    assert warning.preflight_state == "warn"
    failed = disk_snapshot(
        tmp_path, safety_reserve_bytes=500, expected_additional_bytes=200
    )
    assert failed.preflight_state == "fail"


def test_profiles_and_overrides() -> None:
    assert {load_profile(name).name for name in resources.PROFILES} == {
        "smoke",
        "quick",
        "standard",
        "extended",
    }
    override = load_profile("smoke", {"smoke": {"cpu_rss_limit_bytes": 123}})
    assert override.cpu_rss_limit_bytes == 123
    with pytest.raises(resources.ResourceError):
        load_profile("unsupported")


def test_committed_g14_standard_profile_loads() -> None:
    profile = load_profile_file(
        Path("configs/hardware/g14-standard.json"), expected_name="standard"
    )
    assert profile.cpu_rss_limit_bytes == 10 * 1024**3
    assert profile.gpu_peak_allocated_limit_bytes == 4 * 1024**3


def _disk(state: str = "pass") -> DiskSnapshot:
    return DiskSnapshot(
        "2026-01-01T00:00:00+00:00", "~", 1000, 100, 900, 100, None, state
    )


def test_all_compliance_states() -> None:
    profile = ResourceProfile("test", 100, 100, 100, 100, None, 0.8, 1.0)
    assert classify_compliance(profile).state is ComplianceState.NOT_MEASURED
    assert (
        classify_compliance(profile, cpu_rss_bytes=50).state
        is ComplianceState.COMPLIANT
    )
    assert (
        classify_compliance(profile, cpu_rss_bytes=85).state
        is ComplianceState.COMPLIANT_WITH_WARNING
    )
    assert (
        classify_compliance(profile, cpu_rss_bytes=101).state
        is ComplianceState.NONCOMPLIANT
    )
    assert (
        classify_compliance(profile, operation_failed=True).state
        is ComplianceState.RESOURCE_FAILURE
    )


def test_cpu_timing_structure() -> None:
    calls: list[int] = []
    result = time_operation(lambda: calls.append(1), warmup_count=2, repetition_count=5)
    assert len(calls) == 7
    assert result.warmup_count == 2 and result.repetition_count == 5
    assert len(result.samples_seconds) == 5
    assert result.minimum_seconds <= result.median_seconds <= result.maximum_seconds
    assert not result.synchronized


def test_nvidia_unavailable_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("missing")

    monkeypatch.setattr(resources.subprocess, "run", unavailable)
    snapshot = nvidia_snapshot()
    assert not snapshot.available
    assert snapshot.total_vram_bytes is None
    assert "FileNotFoundError" in (snapshot.error or "")


def _validated_state():
    state = create_state(
        run_id="run-exp-resource-v1-syn-demo-s1-111111111111-a1",
        experiment_id="exp-resource-v1",
        scene_id="syn-demo",
        attempt=1,
        configuration_ref="resolved_config.json",
        configuration_digest=DIGEST,
    )
    return transition(state, RunStatus.VALIDATED, reason="validated", actor="test")


def test_artificial_cpu_guard_integrates_lifecycle_and_skips_work(
    tmp_path: Path,
) -> None:
    state = _validated_state()
    run_directory = initialize_run_directory(tmp_path / "runs", state)
    logger = RunLogger(run_directory, state.run_id)
    called = False

    def expensive() -> None:
        nonlocal called
        called = True

    profile = load_profile(
        "smoke",
        {"smoke": {"disk_safety_reserve_bytes": 0, "expected_workspace_bytes": 0}},
    )
    with pytest.raises(ResourceGuardError):
        guard_with_lifecycle(
            state,
            run_directory,
            logger,
            profile,
            workspace_root=tmp_path,
            artificial_cpu_limit_bytes=1,
            expensive_operation=expensive,
        )
    assert not called
    failed = load_state(run_directory / "status.json")
    assert failed.status == "failed"
    resource_path = run_directory / "metadata/resource-record.json"
    assert resource_path.is_file()
    assert failed.failure_ref and (run_directory / failed.failure_ref).is_file()
    assert "resource_guard_failed" in (run_directory / "logs/events.jsonl").read_text()
    paths = (
        "metadata/resource-record.json",
        failed.failure_ref,
        "logs/events.jsonl",
        "logs/run.log",
    )
    inventory = build_inventory(
        run_directory,
        [
            ArtifactDeclaration(f"guard-{index}", "resource_record", path)
            for index, path in enumerate(paths)
        ],
        inventory_id="inventory-resource-failure",
    )
    assert verify_inventory(inventory, run_directory).state is VerificationState.VALID


def test_resource_record_artifact_verification(tmp_path: Path) -> None:
    profile = load_profile(
        "smoke",
        {"smoke": {"disk_safety_reserve_bytes": 0, "expected_workspace_bytes": 0}},
    )
    output = tmp_path / "resource.json"
    resources.measure_operation(
        lambda: sum(range(100)),
        operation_name="unit",
        profile=profile,
        workspace_root=tmp_path,
        output=output,
    )
    inventory = build_inventory(
        tmp_path,
        [ArtifactDeclaration("resource", "resource_record", "resource.json")],
        inventory_id="inventory-resource",
    )
    assert verify_inventory(inventory, tmp_path).state is VerificationState.VALID
    assert json.loads(output.read_text())["operation"] == "unit"


def test_accounting_import_is_cpu_only() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gaussweave.accounting; "
            "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
