"""Dependency-light system and WSL environment inspection.

This module intentionally uses only the Python standard library. It is safe to
import before PyTorch, CUDA Python packages, gsplat, Blender, or PyCOLMAP are
installed.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import locale
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.0"
DISK_SAFETY_RESERVE_BYTES = 100 * 1024**3
DEFAULT_COMMAND_TIMEOUT_SECONDS = 5.0

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one dependency-light external command."""

    available: bool
    executable: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None = None


@dataclass(frozen=True)
class WSLInfo:
    """WSL detection and the evidence supporting it."""

    detected: bool
    evidence: tuple[str, ...]
    generation: str | None
    distribution: str | None
    version_text: str | None


@dataclass(frozen=True)
class PlatformInfo:
    """Execution platform and operating-system identity."""

    system: str
    release: str
    kernel: str
    machine: str
    python_version: str
    python_executable: str
    timezone: str
    locale: str | None
    linux_distribution: str | None
    linux_distribution_version: str | None
    windows_host: str | None


@dataclass(frozen=True)
class RepositoryInfo:
    """Resolved repository location and filesystem classification."""

    path: str
    resolved_by: str
    filesystem_class: str
    warning: str | None


@dataclass(frozen=True)
class CPUInfo:
    """CPU identity."""

    model: str | None
    logical_count: int | None
    architecture: str


@dataclass(frozen=True)
class MemoryInfo:
    """System RAM and swap values in exact bytes."""

    total_bytes: int | None
    available_bytes: int | None
    swap_total_bytes: int | None
    swap_used_bytes: int | None
    error: str | None = None


@dataclass(frozen=True)
class GPUDevice:
    """One GPU reported by nvidia-smi."""

    index: int
    name: str
    driver_version: str
    total_vram_bytes: int
    free_vram_bytes: int
    query_timestamp: str | None


@dataclass(frozen=True)
class GPUInfo:
    """Dependency-light NVIDIA query result."""

    command_available: bool
    executable: str | None
    query_exit_code: int | None
    driver_version: str | None
    devices: tuple[GPUDevice, ...]
    stderr: str
    error: str | None


@dataclass(frozen=True)
class DiskInfo:
    """Repository filesystem capacity in exact bytes."""

    total_bytes: int | None
    used_bytes: int | None
    free_bytes: int | None
    mount_source: str | None
    below_safety_reserve: bool | None
    safety_reserve_bytes: int = DISK_SAFETY_RESERVE_BYTES
    error: str | None = None


@dataclass(frozen=True)
class ToolInfo:
    """Availability and version capture for one development tool."""

    name: str
    available: bool
    executable: str | None
    version_text: str | None
    exit_code: int | None
    error: str | None


@dataclass(frozen=True)
class EnvironmentRecord:
    """Complete environment inspection record."""

    schema_version: str
    timestamp: str
    platform: PlatformInfo
    wsl: WSLInfo
    repository: RepositoryInfo
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: GPUInfo
    disk: DiskInfo
    tools: tuple[ToolInfo, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation."""

        return _as_json_dict(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnvironmentRecord:
        """Reconstruct a record from its JSON-compatible representation."""

        platform_value = _mapping(value["platform"])
        wsl_value = _mapping(value["wsl"])
        repository_value = _mapping(value["repository"])
        cpu_value = _mapping(value["cpu"])
        memory_value = _mapping(value["memory"])
        gpu_value = _mapping(value["gpu"])
        disk_value = _mapping(value["disk"])
        tool_values = value["tools"]
        if not isinstance(tool_values, list):
            raise TypeError("tools must be a list")
        devices_value = gpu_value["devices"]
        if not isinstance(devices_value, list):
            raise TypeError("gpu.devices must be a list")

        return cls(
            schema_version=str(value["schema_version"]),
            timestamp=str(value["timestamp"]),
            platform=PlatformInfo(**platform_value),
            wsl=WSLInfo(**{**wsl_value, "evidence": tuple(wsl_value["evidence"])}),
            repository=RepositoryInfo(**repository_value),
            cpu=CPUInfo(**cpu_value),
            memory=MemoryInfo(**memory_value),
            gpu=GPUInfo(
                command_available=bool(gpu_value["command_available"]),
                executable=_optional_str(gpu_value["executable"]),
                query_exit_code=_optional_int(gpu_value["query_exit_code"]),
                driver_version=_optional_str(gpu_value["driver_version"]),
                devices=tuple(
                    GPUDevice(**_mapping(device)) for device in devices_value
                ),
                stderr=str(gpu_value["stderr"]),
                error=_optional_str(gpu_value["error"]),
            ),
            disk=DiskInfo(**disk_value),
            tools=tuple(ToolInfo(**_mapping(tool)) for tool in tool_values),
            warnings=tuple(str(item) for item in _sequence(value["warnings"])),
            errors=tuple(str(item) for item in _sequence(value["errors"])),
        )


type CommandRunner = Callable[[Sequence[str], float], CommandResult]


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError("expected a sequence")
    return value


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float | str | bytes | bytearray):
        return int(value)
    raise TypeError("expected an integer-compatible value")


def _as_json_dict(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError("record serialization did not produce a dictionary")
    return {str(key): _as_json_value(item) for key, item in value.items()}


def _as_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [_as_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def normalize_path(path: Path) -> str:
    """Remove user-specific home prefixes from a path."""

    resolved = path.resolve()
    candidates = {
        Path.home(),
        Path(os.environ["USERPROFILE"]) if os.environ.get("USERPROFILE") else None,
    }
    for home in candidates:
        if home is None:
            continue
        try:
            relative = resolved.relative_to(home.resolve())
        except ValueError:
            continue
        return PurePosixPath("~", *relative.parts).as_posix()
    return resolved.as_posix()


def run_command(
    arguments: Sequence[str],
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> CommandResult:
    """Run an argv-only command with bounded execution and captured output."""

    if not arguments:
        raise ValueError("command arguments must not be empty")
    executable = shutil.which(arguments[0])
    if executable is None:
        return CommandResult(
            available=False,
            executable=None,
            exit_code=None,
            stdout="",
            stderr="",
            error="executable not found",
        )

    safe_executable = normalize_path(Path(executable))
    try:
        completed = subprocess.run(
            [executable, *arguments[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        return CommandResult(
            available=True,
            executable=safe_executable,
            exit_code=None,
            stdout=_coerce_output(error.stdout),
            stderr=_coerce_output(error.stderr),
            error=f"command timed out after {timeout_seconds:g} seconds",
        )
    except OSError as error:
        return CommandResult(
            available=True,
            executable=safe_executable,
            exit_code=None,
            stdout="",
            stderr="",
            error=f"{type(error).__name__}: {error}",
        )

    return CommandResult(
        available=True,
        executable=safe_executable,
        exit_code=completed.returncode,
        stdout=_coerce_output(completed.stdout),
        stderr=_coerce_output(completed.stderr),
        error=None if completed.returncode == 0 else "command returned nonzero status",
    )


def _coerce_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value.replace("\x00", "").strip()


def detect_wsl(
    *,
    kernel_release: str | None = None,
    proc_version: str | None = None,
    environ: Mapping[str, str] | None = None,
    runner: CommandRunner = run_command,
) -> WSLInfo:
    """Detect WSL from kernel, proc, environment, and optional host evidence."""

    environment = os.environ if environ is None else environ
    release = platform.release() if kernel_release is None else kernel_release
    if proc_version is None:
        proc_version = _read_text(Path("/proc/version"))

    evidence: list[str] = []
    combined = f"{release} {proc_version or ''}".lower()
    if "microsoft" in combined or "wsl" in combined:
        evidence.append("kernel")
    if environment.get("WSL_DISTRO_NAME"):
        evidence.append("WSL_DISTRO_NAME")
    if environment.get("WSL_INTEROP"):
        evidence.append("WSL_INTEROP")

    detected = bool(evidence)
    generation: str | None = None
    if detected:
        generation = (
            "2" if "wsl2" in combined or "microsoft-standard" in combined else None
        )

    version_result = runner(("wsl.exe", "--version"), DEFAULT_COMMAND_TIMEOUT_SECONDS)
    version_text = _combined_version_text(version_result)
    if detected and version_result.available and version_result.exit_code == 0:
        evidence.append("wsl.exe")
        generation = generation or _parse_wsl_generation(version_text)

    return WSLInfo(
        detected=detected,
        evidence=tuple(dict.fromkeys(evidence)),
        generation=generation,
        distribution=environment.get("WSL_DISTRO_NAME"),
        version_text=version_text,
    )


def _parse_wsl_generation(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"WSL version:\s*(\d+)", text, flags=re.IGNORECASE)
    return "2" if match and int(match.group(1)) >= 2 else None


def classify_repository_filesystem(path: Path, *, wsl_detected: bool) -> str:
    """Classify a repository path without probing or mutating it."""

    posix = path.as_posix()
    if re.match(r"^/mnt/[a-zA-Z](?:/|$)", posix):
        return "windows_mounted"
    if wsl_detected and posix.startswith("/"):
        return "wsl_linux"
    return "other_or_unknown"


def resolve_repository_root(
    candidate: Path | None = None,
    *,
    runner: CommandRunner = run_command,
) -> tuple[Path, str]:
    """Resolve the Git top-level when possible, otherwise use a safe fallback."""

    start = (candidate or Path.cwd()).resolve()
    result = runner(
        ("git", "-C", str(start), "rev-parse", "--show-toplevel"),
        DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    if result.exit_code == 0 and result.stdout:
        return Path(result.stdout.splitlines()[0]).resolve(), "git"
    return start, "explicit" if candidate is not None else "current_working_directory"


def inspect_repository(
    path: Path, *, wsl_detected: bool, resolved_by: str
) -> RepositoryInfo:
    """Build repository location metadata and WSL mount warnings."""

    classification = classify_repository_filesystem(path, wsl_detected=wsl_detected)
    warning = None
    if classification == "windows_mounted":
        warning = "repository is on a Windows-mounted filesystem under /mnt/<drive>"
    return RepositoryInfo(
        path=normalize_path(path),
        resolved_by=resolved_by,
        filesystem_class=classification,
        warning=warning,
    )


def inspect_platform(
    wsl: WSLInfo, *, runner: CommandRunner = run_command
) -> PlatformInfo:
    """Collect platform, distribution, Python, timezone, and host details."""

    distro_name, distro_version = _linux_distribution()
    windows_host: str | None
    if platform.system() == "Windows":
        windows_host = f"Windows {platform.release()} ({platform.version()})"
    elif wsl.detected:
        host = runner(("cmd.exe", "/c", "ver"), DEFAULT_COMMAND_TIMEOUT_SECONDS)
        windows_host = _combined_version_text(host)
    else:
        windows_host = None

    local_timezone = datetime.now().astimezone().tzinfo
    return PlatformInfo(
        system=platform.system(),
        release=platform.release(),
        kernel=platform.version(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        python_executable=normalize_path(Path(sys.executable)),
        timezone=str(local_timezone) if local_timezone is not None else "unknown",
        locale=locale.getlocale()[0],
        linux_distribution=distro_name,
        linux_distribution_version=distro_version,
        windows_host=windows_host,
    )


def _linux_distribution() -> tuple[str | None, str | None]:
    values = _parse_key_values(_read_text(Path("/etc/os-release")))
    return values.get("NAME"), values.get("VERSION_ID")


def inspect_cpu() -> CPUInfo:
    """Collect dependency-light CPU identity."""

    model = platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER")
    if not model:
        cpu_values = _parse_colon_values(_read_text(Path("/proc/cpuinfo")))
        model = cpu_values.get("model name") or cpu_values.get("hardware")
    return CPUInfo(
        model=model,
        logical_count=os.cpu_count(),
        architecture=platform.machine(),
    )


def parse_meminfo(text: str) -> MemoryInfo:
    """Parse Linux /proc/meminfo values into exact bytes."""

    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^([^:]+):\s*(\d+)\s*kB$", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    swap_used = None
    if swap_total is not None and swap_free is not None:
        swap_used = max(0, swap_total - swap_free)
    return MemoryInfo(total, available, swap_total, swap_used)


def inspect_memory() -> MemoryInfo:
    """Collect Linux or Windows system memory without third-party packages."""

    meminfo = _read_text(Path("/proc/meminfo"))
    if meminfo:
        parsed = parse_meminfo(meminfo)
        if parsed.total_bytes is not None:
            return parsed
    if platform.system() == "Windows":
        try:
            status = _WindowsMemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx returned false")
            return MemoryInfo(
                total_bytes=int(status.ullTotalPhys),
                available_bytes=int(status.ullAvailPhys),
                swap_total_bytes=int(status.ullTotalPageFile),
                swap_used_bytes=max(
                    0, int(status.ullTotalPageFile - status.ullAvailPageFile)
                ),
            )
        except (AttributeError, OSError) as error:
            return MemoryInfo(
                None, None, None, None, f"{type(error).__name__}: {error}"
            )
    return MemoryInfo(None, None, None, None, "memory telemetry unavailable")


class _WindowsMemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def query_nvidia_gpu(*, runner: CommandRunner = run_command) -> GPUInfo:
    """Query all NVIDIA GPUs through nvidia-smi CSV output."""

    result = runner(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,timestamp",
            "--format=csv,noheader,nounits",
        ),
        DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    if not result.available:
        return GPUInfo(False, None, None, None, (), result.stderr, result.error)
    if result.exit_code != 0:
        return GPUInfo(
            True,
            result.executable,
            result.exit_code,
            None,
            (),
            result.stderr,
            result.error or "nvidia-smi query failed",
        )

    devices: list[GPUDevice] = []
    try:
        for index, line in enumerate(filter(None, result.stdout.splitlines())):
            fields = [part.strip() for part in line.split(",")]
            if len(fields) != 5:
                raise ValueError(f"expected 5 CSV fields, received {len(fields)}")
            devices.append(
                GPUDevice(
                    index=index,
                    name=fields[0],
                    driver_version=fields[1],
                    total_vram_bytes=_mib_to_bytes(fields[2]),
                    free_vram_bytes=_mib_to_bytes(fields[3]),
                    query_timestamp=fields[4] or None,
                )
            )
    except ValueError as error:
        return GPUInfo(
            True,
            result.executable,
            result.exit_code,
            None,
            (),
            result.stderr,
            f"invalid nvidia-smi output: {error}",
        )

    driver = devices[0].driver_version if devices else None
    return GPUInfo(
        True,
        result.executable,
        result.exit_code,
        driver,
        tuple(devices),
        result.stderr,
        None if devices else "nvidia-smi returned no GPU rows",
    )


def _mib_to_bytes(value: str) -> int:
    return int(float(value)) * 1024**2


def inspect_disk(path: Path, *, free_bytes: int | None = None) -> DiskInfo:
    """Inspect repository disk capacity and the initial safety reserve."""

    try:
        usage = shutil.disk_usage(path)
        total = int(usage.total)
        used = int(usage.used)
        free = int(usage.free) if free_bytes is None else free_bytes
        return DiskInfo(
            total_bytes=total,
            used_bytes=used,
            free_bytes=free,
            mount_source=_mount_source(path),
            below_safety_reserve=free < DISK_SAFETY_RESERVE_BYTES,
        )
    except OSError as error:
        return DiskInfo(
            total_bytes=None,
            used_bytes=None,
            free_bytes=None,
            mount_source=None,
            below_safety_reserve=None,
            error=f"{type(error).__name__}: {error}",
        )


def _mount_source(path: Path) -> str | None:
    if platform.system() == "Windows":
        return path.resolve().anchor or None
    mountinfo = _read_text(Path("/proc/self/mountinfo"))
    if not mountinfo:
        return None
    resolved = path.resolve().as_posix()
    candidates: list[tuple[str, str]] = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left = before.split()
        right = after.split()
        if len(left) >= 5 and len(right) >= 2:
            mount_point = left[4].replace("\\040", " ")
            if resolved == mount_point or resolved.startswith(
                f"{mount_point.rstrip('/')}/"
            ):
                candidates.append((mount_point, right[1]))
    if not candidates:
        return None
    return max(candidates, key=lambda item: len(item[0]))[1]


TOOL_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("git", ("git", "--version")),
    ("gcc", ("gcc", "--version")),
    ("g++", ("g++", "--version")),
    ("cmake", ("cmake", "--version")),
    ("ninja", ("ninja", "--version")),
    ("blender", ("blender", "--version")),
    ("colmap", ("colmap", "--version")),
    ("nvidia-smi", ("nvidia-smi", "--version")),
)


def inspect_tool(
    name: str,
    arguments: Sequence[str],
    *,
    runner: CommandRunner = run_command,
) -> ToolInfo:
    """Inspect a tool whose version may be written to stdout or stderr."""

    result = runner(arguments, DEFAULT_COMMAND_TIMEOUT_SECONDS)
    version_text = _combined_version_text(result)
    return ToolInfo(
        name=name,
        available=result.available,
        executable=result.executable,
        version_text=version_text,
        exit_code=result.exit_code,
        error=result.error,
    )


def inspect_tools(*, runner: CommandRunner = run_command) -> tuple[ToolInfo, ...]:
    """Inspect all required development and optional external tools."""

    return tuple(
        inspect_tool(name, arguments, runner=runner)
        for name, arguments in TOOL_COMMANDS
    )


def _combined_version_text(result: CommandResult) -> str | None:
    text = result.stdout or result.stderr
    if not text:
        return None
    return text.strip()


def inspect_environment(
    repo_root: Path | None = None,
    *,
    runner: CommandRunner = run_command,
    timestamp: datetime | None = None,
) -> EnvironmentRecord:
    """Inspect the current system while preserving optional failures."""

    wsl = detect_wsl(runner=runner)
    resolved_root, resolved_by = resolve_repository_root(repo_root, runner=runner)
    repository = inspect_repository(
        resolved_root,
        wsl_detected=wsl.detected,
        resolved_by=resolved_by,
    )
    memory = inspect_memory()
    gpu = query_nvidia_gpu(runner=runner)
    disk = inspect_disk(resolved_root)
    tools = inspect_tools(runner=runner)

    warnings: list[str] = []
    errors: list[str] = []
    if repository.warning:
        warnings.append(repository.warning)
    if disk.below_safety_reserve:
        warnings.append(
            f"repository disk free space is below {DISK_SAFETY_RESERVE_BYTES} bytes"
        )
    if not gpu.command_available:
        warnings.append("nvidia-smi is unavailable")
    elif gpu.error:
        warnings.append(f"NVIDIA GPU telemetry unavailable: {gpu.error}")
    if memory.error:
        errors.append(memory.error)
    if disk.error:
        errors.append(disk.error)

    inspected_at = timestamp or datetime.now(UTC)
    return EnvironmentRecord(
        schema_version=SCHEMA_VERSION,
        timestamp=inspected_at.astimezone(UTC).isoformat(),
        platform=inspect_platform(wsl, runner=runner),
        wsl=wsl,
        repository=repository,
        cpu=inspect_cpu(),
        memory=memory,
        gpu=gpu,
        disk=disk,
        tools=tools,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def format_summary(record: EnvironmentRecord) -> str:
    """Return a concise deterministic human-readable summary."""

    distribution = record.platform.linux_distribution or "unavailable"
    distribution_version = record.platform.linux_distribution_version or ""
    gpu_summary = "unavailable"
    if record.gpu.devices:
        gpu_summary = "; ".join(
            f"{device.name} ({_format_bytes(device.total_vram_bytes)} total, "
            f"{_format_bytes(device.free_vram_bytes)} free)"
            for device in record.gpu.devices
        )
    tools = ", ".join(
        f"{tool.name}={'yes' if tool.available else 'no'}" for tool in record.tools
    )
    warning_text = "; ".join(record.warnings) if record.warnings else "none"
    return "\n".join(
        (
            f"WSL: {'detected' if record.wsl.detected else 'not detected'}"
            + (
                f" (generation {record.wsl.generation})"
                if record.wsl.generation
                else ""
            ),
            f"Distribution: {distribution} {distribution_version}".rstrip(),
            f"Kernel: {record.platform.kernel}",
            f"Repository filesystem: {record.repository.filesystem_class}",
            f"CPU: {record.cpu.model or 'unavailable'} "
            f"({record.cpu.logical_count or 'unknown'} logical CPUs)",
            f"RAM: {_format_bytes(record.memory.total_bytes)} total, "
            f"{_format_bytes(record.memory.available_bytes)} available",
            f"GPU: {gpu_summary}",
            f"NVIDIA driver: {record.gpu.driver_version or 'unavailable'}",
            f"Disk free: {_format_bytes(record.disk.free_bytes)}",
            f"Tools: {tools}",
            f"Warnings: {warning_text}",
        )
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value / 1024**3:.2f} GiB"


def write_record(path: Path, record: EnvironmentRecord) -> None:
    """Atomically write a structured record to an explicit path."""

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(record.to_json())
            temporary.write("\n")
            temporary_name = temporary.name
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _parse_key_values(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value.strip().strip("\"'")
    return values


def _parse_colon_values(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() not in values:
            values[key.strip().lower()] = value.strip()
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete structured JSON record",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically write the structured JSON record to this path",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="repository candidate path; Git resolution is attempted first",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the temporary qualification environment-inspection entry point."""

    parser = _build_parser()
    options = parser.parse_args(arguments)
    try:
        record = inspect_environment(options.repo_root)
        if options.output is not None:
            write_record(options.output, record)
        print(record.to_json() if options.json else format_summary(record))
    except (OSError, TypeError, ValueError) as error:
        print(f"environment inspection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
