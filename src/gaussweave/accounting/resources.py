"""CPU-safe resource telemetry, timing, compliance, and laptop guards."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from gaussweave.config.resolution import pretty_json_bytes
from gaussweave.experiments.lifecycle import (
    RunLogger,
    RunState,
    capture_failures,
    load_state,
)
from gaussweave.runtime.environment import inspect_memory, normalize_path

GIB = 1024**3
DEFAULT_DISK_RESERVE_BYTES = 100 * GIB
RESOURCE_RECORD_VERSION = "1.0"


class ResourceError(RuntimeError):
    """Resource telemetry or profile data is invalid."""


class ResourceGuardError(ResourceError):
    """Preflight rejected an operation before expensive work began."""

    def __init__(self, result: ComplianceResult) -> None:
        super().__init__(
            "resource preflight failed: "
            + "; ".join(
                warning.message
                for warning in result.warnings
                if warning.severity == "error"
            )
        )
        self.result = result


class ComplianceState(StrEnum):
    COMPLIANT = "compliant"
    COMPLIANT_WITH_WARNING = "compliant_with_warning"
    NONCOMPLIANT = "noncompliant"
    RESOURCE_FAILURE = "resource_failure"
    NOT_MEASURED = "not_measured"


@dataclass(frozen=True)
class CPUMemorySnapshot:
    timestamp: str
    process_rss_bytes: int | None
    peak_process_rss_bytes: int | None
    total_system_ram_bytes: int | None
    available_system_ram_bytes: int | None
    swap_total_bytes: int | None
    swap_used_bytes: int | None
    process_id: int
    source: str
    error: str | None = None


@dataclass(frozen=True)
class DiskSnapshot:
    timestamp: str
    workspace_path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    safety_reserve_bytes: int
    expected_additional_bytes: int | None
    preflight_state: str


@dataclass(frozen=True)
class GPUMemorySnapshot:
    timestamp: str
    device_index: int
    device_name: str
    total_vram_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    free_device_bytes: int | None
    total_device_bytes: int | None
    synchronized: bool


@dataclass(frozen=True)
class NvidiaSnapshot:
    timestamp: str
    available: bool
    device_index: int | None
    process_gpu_bytes: int | None
    total_vram_bytes: int | None
    free_vram_bytes: int | None
    driver_version: str | None
    utilization_percent: float | None
    temperature_c: float | None
    power_draw_w: float | None
    power_limit_w: float | None
    graphics_clock_mhz: int | None
    throttle_reasons: str | None
    error: str | None


@dataclass(frozen=True)
class TimingMeasurement:
    clock: str
    warmup_count: int
    repetition_count: int
    samples_seconds: tuple[float, ...]
    median_seconds: float | None
    p95_seconds: float | None
    minimum_seconds: float | None
    maximum_seconds: float | None
    synchronized: bool


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    gpu_peak_allocated_limit_bytes: int
    gpu_peak_reserved_limit_bytes: int
    cpu_rss_limit_bytes: int
    disk_safety_reserve_bytes: int
    expected_workspace_bytes: int | None
    warning_fraction: float
    hard_failure_fraction: float


@dataclass(frozen=True)
class ResourceWarning:
    resource: str
    observed_bytes: int | None
    threshold_bytes: int | None
    severity: str
    may_continue: bool
    message: str


@dataclass(frozen=True)
class ComplianceResult:
    state: ComplianceState
    may_start: bool
    may_continue: bool
    warnings: tuple[ResourceWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "may_start": self.may_start,
            "may_continue": self.may_continue,
            "warnings": [asdict(item) for item in self.warnings],
        }


@dataclass(frozen=True)
class OperationResourceRecord:
    record_version: str
    run_id: str | None
    operation: str
    resource_profile: str
    started_at: str
    ended_at: str
    elapsed_seconds: float
    cpu_baseline: CPUMemorySnapshot
    cpu_peak: CPUMemorySnapshot
    disk: DiskSnapshot
    gpu_baseline: GPUMemorySnapshot | None
    gpu_final: GPUMemorySnapshot | None
    nvidia: NvidiaSnapshot
    timing: TimingMeasurement
    compliance: ComplianceResult
    warnings: tuple[ResourceWarning, ...]
    failure_ref: str | None
    environment_ref: str | None
    guard_override: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["compliance"]["state"] = self.compliance.state.value
        return value


PROFILES: Mapping[str, ResourceProfile] = {
    "smoke": ResourceProfile(
        "smoke",
        2 * GIB,
        3 * GIB,
        8 * GIB,
        DEFAULT_DISK_RESERVE_BYTES,
        5 * GIB,
        0.80,
        1.0,
    ),
    "quick": ResourceProfile(
        "quick",
        4 * GIB,
        5 * GIB,
        16 * GIB,
        DEFAULT_DISK_RESERVE_BYTES,
        25 * GIB,
        0.85,
        1.0,
    ),
    "standard": ResourceProfile(
        "standard",
        7 * GIB,
        8 * GIB,
        24 * GIB,
        DEFAULT_DISK_RESERVE_BYTES,
        150 * GIB,
        0.90,
        1.0,
    ),
    "extended": ResourceProfile(
        "extended",
        7 * GIB,
        8 * GIB,
        28 * GIB,
        DEFAULT_DISK_RESERVE_BYTES,
        250 * GIB,
        0.95,
        1.0,
    ),
}


def load_profile(
    name: str, overrides: Mapping[str, Mapping[str, Any]] | None = None
) -> ResourceProfile:
    """Load one validated built-in profile with optional centralized overrides."""

    if name not in PROFILES:
        raise ResourceError(f"unsupported resource profile: {name}")
    values = asdict(PROFILES[name])
    if overrides and name in overrides:
        values.update(overrides[name])
    profile = ResourceProfile(**values)
    byte_fields = (
        profile.gpu_peak_allocated_limit_bytes,
        profile.gpu_peak_reserved_limit_bytes,
        profile.cpu_rss_limit_bytes,
        profile.disk_safety_reserve_bytes,
    )
    if any(isinstance(value, bool) or value < 0 for value in byte_fields):
        raise ResourceError("profile byte limits must be nonnegative integers")
    if not 0 < profile.warning_fraction <= profile.hard_failure_fraction:
        raise ResourceError("profile warning/hard fractions are invalid")
    return profile


def load_profile_file(
    path: Path, *, expected_name: str | None = None
) -> ResourceProfile:
    """Load a validated resource profile from committed JSON configuration."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResourceError(
            f"unable to load resource profile {path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ResourceError("resource profile document must be an object")
    profile_value = document.get("resource_profile")
    if not isinstance(profile_value, dict):
        raise ResourceError("resource profile document has no resource_profile object")
    allowed = set(ResourceProfile.__dataclass_fields__)
    unexpected = sorted(set(profile_value) - allowed)
    if unexpected:
        raise ResourceError(f"unknown resource profile fields: {unexpected}")
    try:
        profile = ResourceProfile(**profile_value)
    except TypeError as error:
        raise ResourceError(f"invalid resource profile fields: {error}") from error
    if expected_name is not None and profile.name != expected_name:
        raise ResourceError(
            f"resource profile name {profile.name!r} does not match {expected_name!r}"
        )
    byte_fields = (
        profile.gpu_peak_allocated_limit_bytes,
        profile.gpu_peak_reserved_limit_bytes,
        profile.cpu_rss_limit_bytes,
        profile.disk_safety_reserve_bytes,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in byte_fields
    ):
        raise ResourceError("profile byte limits must be nonnegative integers")
    if not 0 < profile.warning_fraction <= profile.hard_failure_fraction:
        raise ResourceError("profile warning/hard fractions are invalid")
    return profile


def cpu_memory_snapshot() -> CPUMemorySnapshot:
    """Measure process and Linux/WSL system memory in exact bytes."""

    memory = inspect_memory()
    rss: int | None = None
    peak: int | None = None
    source = "unavailable"
    error: str | None = None
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        values: dict[str, int] = {}
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                values[key.rstrip(":")] = int(value) * 1024
        rss, peak = values.get("VmRSS"), values.get("VmHWM")
        source = "/proc/self/status"
    except OSError as caught:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = int(usage.ru_maxrss) * (1024 if os.name != "nt" else 1)
        source = "getrusage"
        error = str(caught)
    return CPUMemorySnapshot(
        _now(),
        rss,
        peak,
        memory.total_bytes,
        memory.available_bytes,
        memory.swap_total_bytes,
        memory.swap_used_bytes,
        os.getpid(),
        source,
        error or memory.error,
    )


def disk_snapshot(
    workspace_root: Path,
    *,
    safety_reserve_bytes: int = DEFAULT_DISK_RESERVE_BYTES,
    expected_additional_bytes: int | None = None,
    warning_fraction: float = 0.1,
) -> DiskSnapshot:
    """Measure filesystem capacity and deterministic reserve preflight state."""

    usage = shutil.disk_usage(workspace_root.resolve())
    required = safety_reserve_bytes + (expected_additional_bytes or 0)
    if usage.free < required:
        state = "fail"
    elif usage.free < required + max(1, int(safety_reserve_bytes * warning_fraction)):
        state = "warn"
    else:
        state = "pass"
    return DiskSnapshot(
        _now(),
        normalize_path(workspace_root),
        usage.total,
        usage.used,
        usage.free,
        safety_reserve_bytes,
        expected_additional_bytes,
        state,
    )


def gpu_memory_snapshot(
    device: str | int = "cuda:0", *, synchronize: bool = True
) -> GPUMemorySnapshot:
    """Dynamically import PyTorch and capture allocator/device memory."""

    torch = _torch()
    index = torch.device(device).index or 0
    if not torch.cuda.is_available():
        raise ResourceGuardError(_unavailable_cuda())
    if synchronize:
        torch.cuda.synchronize(index)
    properties = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    return GPUMemorySnapshot(
        _now(),
        index,
        properties.name,
        int(properties.total_memory),
        int(torch.cuda.memory_allocated(index)),
        int(torch.cuda.memory_reserved(index)),
        int(torch.cuda.max_memory_allocated(index)),
        int(torch.cuda.max_memory_reserved(index)),
        int(free),
        int(total),
        synchronize,
    )


def reset_gpu_peaks(device: str | int = "cuda:0") -> None:
    torch = _torch()
    index = torch.device(device).index or 0
    torch.cuda.synchronize(index)
    torch.cuda.reset_peak_memory_stats(index)


def nvidia_snapshot(device_index: int = 0) -> NvidiaSnapshot:
    """Capture optional process/device NVIDIA telemetry without raising."""

    fields = (
        "index,memory.total,memory.free,driver_version,utilization.gpu,"
        "temperature.gpu,power.draw,power.limit,clocks.gr,"
        "clocks_throttle_reasons.active"
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            shell=False,
        )
        if result.returncode != 0:
            raise OSError(result.stderr.strip() or "nvidia-smi returned nonzero")
        rows = [row for row in result.stdout.splitlines() if row.strip()]
        row = next(row for row in rows if int(row.split(",", 1)[0]) == device_index)
        values = [item.strip() for item in row.split(",")]
        process_bytes = _query_process_gpu_bytes(device_index)
        return NvidiaSnapshot(
            _now(),
            True,
            device_index,
            process_bytes,
            _mib(values[1]),
            _mib(values[2]),
            values[3] or None,
            _float(values[4]),
            _float(values[5]),
            _float(values[6]),
            _float(values[7]),
            _int(values[8]),
            values[9] or None,
            None,
        )
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError) as error:
        return NvidiaSnapshot(
            _now(),
            False,
            device_index,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            f"{type(error).__name__}: {error}",
        )


def time_operation(
    operation: Callable[[], Any],
    *,
    warmup_count: int = 0,
    repetition_count: int = 1,
    cuda_device: str | None = None,
) -> TimingMeasurement:
    """Separate warmups and return synchronized repeated timing samples."""

    if warmup_count < 0 or repetition_count < 1:
        raise ResourceError("invalid warmup/repetition counts")
    torch = _torch() if cuda_device else None
    for _ in range(warmup_count):
        operation()
    if torch is not None:
        torch.cuda.synchronize(cuda_device)
    samples: list[float] = []
    for _ in range(repetition_count):
        if torch is not None:
            torch.cuda.synchronize(cuda_device)
        started = time.perf_counter()
        operation()
        if torch is not None:
            torch.cuda.synchronize(cuda_device)
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999) - 1))
    return TimingMeasurement(
        "perf_counter",
        warmup_count,
        repetition_count,
        tuple(samples),
        statistics.median(samples),
        ordered[p95_index],
        min(samples),
        max(samples),
        torch is not None,
    )


def classify_compliance(
    profile: ResourceProfile,
    *,
    cpu_rss_bytes: int | None = None,
    gpu_allocated_bytes: int | None = None,
    gpu_reserved_bytes: int | None = None,
    disk: DiskSnapshot | None = None,
    operation_failed: bool = False,
) -> ComplianceResult:
    """Deterministically classify measured values and explain every violation."""

    if operation_failed:
        warning = ResourceWarning(
            "operation", None, None, "error", False, "operation had a resource failure"
        )
        return ComplianceResult(
            ComplianceState.RESOURCE_FAILURE, False, False, (warning,)
        )
    checks = (
        ("cpu_rss", cpu_rss_bytes, profile.cpu_rss_limit_bytes),
        (
            "gpu_peak_allocated",
            gpu_allocated_bytes,
            profile.gpu_peak_allocated_limit_bytes,
        ),
        (
            "gpu_peak_reserved",
            gpu_reserved_bytes,
            profile.gpu_peak_reserved_limit_bytes,
        ),
    )
    warnings: list[ResourceWarning] = []
    measured = False
    hard = False
    caution = False
    for name, observed, threshold in checks:
        if observed is None:
            continue
        measured = True
        hard_threshold = int(threshold * profile.hard_failure_fraction)
        if observed > hard_threshold:
            hard = True
            warnings.append(
                ResourceWarning(
                    name,
                    observed,
                    hard_threshold,
                    "error",
                    False,
                    f"{name} {observed} exceeds hard limit {hard_threshold}",
                )
            )
        elif observed >= threshold * profile.warning_fraction:
            caution = True
            warnings.append(
                ResourceWarning(
                    name,
                    observed,
                    threshold,
                    "warning",
                    True,
                    f"{name} {observed} approaches limit {threshold}",
                )
            )
    if disk is not None:
        measured = True
        if disk.preflight_state == "fail":
            hard = True
            warnings.append(
                ResourceWarning(
                    "disk_free",
                    disk.free_bytes,
                    disk.safety_reserve_bytes + (disk.expected_additional_bytes or 0),
                    "error",
                    False,
                    "free disk is below reserve plus expected workspace",
                )
            )
        elif disk.preflight_state == "warn":
            caution = True
            warnings.append(
                ResourceWarning(
                    "disk_free",
                    disk.free_bytes,
                    disk.safety_reserve_bytes,
                    "warning",
                    True,
                    "free disk is near the configured reserve",
                )
            )
    state = (
        ComplianceState.NONCOMPLIANT
        if hard
        else ComplianceState.COMPLIANT_WITH_WARNING
        if caution
        else ComplianceState.COMPLIANT
        if measured
        else ComplianceState.NOT_MEASURED
    )
    return ComplianceResult(state, not hard, not hard, tuple(warnings))


def preflight(
    profile: ResourceProfile,
    workspace_root: Path,
    *,
    require_cuda: bool = False,
    artificial_cpu_limit_bytes: int | None = None,
    artificial_gpu_limit_bytes: int | None = None,
    expected_workspace_bytes: int | None = None,
    override: bool = False,
) -> tuple[CPUMemorySnapshot, DiskSnapshot, GPUMemorySnapshot | None, ComplianceResult]:
    """Check disk/CPU/GPU guards without performing a large allocation."""

    cpu = cpu_memory_snapshot()
    disk = disk_snapshot(
        workspace_root,
        safety_reserve_bytes=profile.disk_safety_reserve_bytes,
        expected_additional_bytes=(
            expected_workspace_bytes
            if expected_workspace_bytes is not None
            else profile.expected_workspace_bytes
        ),
    )
    gpu: GPUMemorySnapshot | None = None
    extra: list[ResourceWarning] = []
    if (
        artificial_cpu_limit_bytes is not None
        and (cpu.process_rss_bytes or 0) > artificial_cpu_limit_bytes
    ):
        extra.append(
            ResourceWarning(
                "artificial_cpu_limit",
                cpu.process_rss_bytes,
                artificial_cpu_limit_bytes,
                "error",
                False,
                "artificial CPU memory guard triggered",
            )
        )
    if require_cuda or artificial_gpu_limit_bytes is not None:
        try:
            gpu = gpu_memory_snapshot()
        except ResourceError:
            extra.append(
                ResourceWarning(
                    "cuda",
                    None,
                    None,
                    "error",
                    False,
                    "required CUDA device unavailable",
                )
            )
    if artificial_gpu_limit_bytes is not None and gpu is not None:
        observed = max(gpu.allocated_bytes, 1)
        if observed >= artificial_gpu_limit_bytes:
            extra.append(
                ResourceWarning(
                    "artificial_gpu_limit",
                    observed,
                    artificial_gpu_limit_bytes,
                    "error",
                    False,
                    "artificial GPU memory guard triggered",
                )
            )
    compliance = classify_compliance(
        profile, cpu_rss_bytes=cpu.process_rss_bytes, disk=disk
    )
    if extra:
        compliance = ComplianceResult(
            ComplianceState.RESOURCE_FAILURE,
            override,
            override,
            (*compliance.warnings, *extra),
        )
    elif not compliance.may_start:
        compliance = ComplianceResult(
            ComplianceState.RESOURCE_FAILURE,
            override,
            override,
            compliance.warnings,
        )
    if not compliance.may_start and not override:
        raise ResourceGuardError(compliance)
    return cpu, disk, gpu, compliance


def measure_operation(
    operation: Callable[[], Any],
    *,
    operation_name: str,
    profile: ResourceProfile,
    workspace_root: Path,
    device: str = "cpu",
    run_id: str | None = None,
    environment_ref: str | None = None,
    warmup_count: int = 0,
    repetition_count: int = 1,
    output: Path | None = None,
    artificial_cpu_limit_bytes: int | None = None,
    artificial_gpu_limit_bytes: int | None = None,
    override: bool = False,
) -> OperationResourceRecord:
    """Preflight, time, sample peaks, classify, and optionally persist a record."""

    started_at = _now()
    cpu_before, disk, gpu_before, preflight_result = preflight(
        profile,
        workspace_root,
        require_cuda=device.startswith("cuda"),
        artificial_cpu_limit_bytes=artificial_cpu_limit_bytes,
        artificial_gpu_limit_bytes=artificial_gpu_limit_bytes,
        override=override,
    )
    if device.startswith("cuda"):
        reset_gpu_peaks(device)
    timing = time_operation(
        operation,
        warmup_count=warmup_count,
        repetition_count=repetition_count,
        cuda_device=device if device.startswith("cuda") else None,
    )
    cpu_after = cpu_memory_snapshot()
    gpu_after = gpu_memory_snapshot(device) if device.startswith("cuda") else None
    compliance = classify_compliance(
        profile,
        cpu_rss_bytes=cpu_after.peak_process_rss_bytes or cpu_after.process_rss_bytes,
        gpu_allocated_bytes=gpu_after.peak_allocated_bytes if gpu_after else None,
        gpu_reserved_bytes=gpu_after.peak_reserved_bytes if gpu_after else None,
        disk=disk,
    )
    record = OperationResourceRecord(
        RESOURCE_RECORD_VERSION,
        run_id,
        operation_name,
        profile.name,
        started_at,
        _now(),
        sum(timing.samples_seconds),
        cpu_before,
        cpu_after,
        disk,
        gpu_before,
        gpu_after,
        nvidia_snapshot(),
        timing,
        compliance,
        compliance.warnings,
        None,
        environment_ref,
        override,
    )
    if output is not None:
        _write_record(record, output)
    return record


def guard_with_lifecycle(
    state: RunState,
    run_directory: Path,
    logger: RunLogger,
    profile: ResourceProfile,
    *,
    workspace_root: Path,
    artificial_cpu_limit_bytes: int | None = None,
    artificial_gpu_limit_bytes: int | None = None,
    require_cuda: bool = False,
    expensive_operation: Callable[[], Any] | None = None,
) -> None:
    """Turn a preflight rejection into resource records before running work."""

    try:
        preflight(
            profile,
            workspace_root,
            require_cuda=require_cuda,
            artificial_cpu_limit_bytes=artificial_cpu_limit_bytes,
            artificial_gpu_limit_bytes=artificial_gpu_limit_bytes,
        )
    except ResourceGuardError as error:
        cpu = cpu_memory_snapshot()
        disk = disk_snapshot(workspace_root, safety_reserve_bytes=0)
        record = _failure_resource_record(state, profile, cpu, disk, error.result)
        output = run_directory / "metadata/resource-record.json"
        _write_record(record, output)
        logger.event(
            "ERROR",
            "resource_guard_failed",
            "preflight",
            str(error),
            context=error.result.to_dict(),
        )
        try:
            with capture_failures(
                state,
                run_directory,
                logger,
                stage="resource.preflight",
                category="environment_incompatibility",
                resource_snapshot=record.to_dict(),
            ):
                raise
        except ResourceGuardError:
            failed = load_state(run_directory / "status.json")
            _write_record(replace(record, failure_ref=failed.failure_ref), output)
            raise
    if expensive_operation is not None:
        expensive_operation()


def _failure_resource_record(
    state: RunState,
    profile: ResourceProfile,
    cpu: CPUMemorySnapshot,
    disk: DiskSnapshot,
    compliance: ComplianceResult,
) -> OperationResourceRecord:
    timing = TimingMeasurement("perf_counter", 0, 0, (), None, None, None, None, False)
    now = _now()
    return OperationResourceRecord(
        RESOURCE_RECORD_VERSION,
        state.run_id,
        "preflight",
        profile.name,
        now,
        now,
        0.0,
        cpu,
        cpu,
        disk,
        None,
        None,
        nvidia_snapshot(),
        timing,
        compliance,
        compliance.warnings,
        None,
        state.environment_ref,
        False,
    )


def _unavailable_cuda() -> ComplianceResult:
    warning = ResourceWarning(
        "cuda", None, None, "error", False, "required CUDA device unavailable"
    )
    return ComplianceResult(ComplianceState.RESOURCE_FAILURE, False, False, (warning,))


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise ResourceError("PyTorch is required for CUDA telemetry") from error
    return torch


def _query_process_gpu_bytes(device_index: int) -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
        shell=False,
    )
    if result.returncode != 0:
        return None
    total = 0
    found = False
    for row in result.stdout.splitlines():
        parts = [item.strip() for item in row.split(",")]
        if len(parts) == 3 and parts[0] == str(os.getpid()):
            total += _mib(parts[2]) or 0
            found = True
    return total if found else None


def _mib(value: str) -> int | None:
    parsed = _float(value)
    return int(parsed * 1024**2) if parsed is not None else None


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _int(value: str) -> int | None:
    parsed = _float(value)
    return int(parsed) if parsed is not None else None


def _write_record(record: OperationResourceRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(pretty_json_bytes(record.to_dict()))
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    except OSError as error:
        raise ResourceError(f"unable to write resource record: {error}") from error
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _demo(options: argparse.Namespace) -> int:
    profile = load_profile(options.profile)
    torch: Any | None = None
    tensor: Any | None = None
    if options.device.startswith("cuda"):
        torch = _torch()

    def operation() -> None:
        nonlocal tensor
        if torch is not None:
            tensor = torch.ones((256, 256), device=options.device)
            tensor = tensor.square().sum()
        else:
            sum(index * index for index in range(1000))

    try:
        record = measure_operation(
            operation,
            operation_name="resource-demo",
            profile=profile,
            workspace_root=Path.cwd(),
            device=options.device,
            warmup_count=1,
            repetition_count=3,
            output=options.output,
            artificial_cpu_limit_bytes=options.artificial_cpu_limit_bytes,
            artificial_gpu_limit_bytes=options.artificial_gpu_limit_bytes,
        )
    except ResourceGuardError as error:
        payload = {"state": "resource_failure", **error.result.to_dict()}
        if options.output:
            options.output.write_bytes(pretty_json_bytes(payload))
        print(json.dumps(payload, indent=2) if options.json else str(error))
        return 1
    payload = record.to_dict()
    gpu_peak = record.gpu_final.peak_allocated_bytes if record.gpu_final else "n/a"
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if options.json
        else (
            f"{record.operation}: {record.compliance.state.value}; "
            f"elapsed={record.elapsed_seconds:.6f}s; "
            f"cpu_rss={record.cpu_peak.process_rss_bytes}; "
            f"disk_free={record.disk.free_bytes}; "
            f"gpu_peak={gpu_peak}"
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--profile", choices=tuple(PROFILES), required=True)
    demo.add_argument("--device", required=True)
    demo.add_argument("--output", type=Path)
    demo.add_argument("--artificial-cpu-limit-bytes", type=int)
    demo.add_argument("--artificial-gpu-limit-bytes", type=int)
    demo.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    return _demo(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
