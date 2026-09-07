"""CPU-only tests for the dependency-light environment inspector."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

from gaussweave.runtime import environment

pytestmark = [pytest.mark.unit, pytest.mark.resource]


def command_result(
    *,
    available: bool = True,
    executable: str | None = "/usr/bin/tool",
    exit_code: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
) -> environment.CommandResult:
    """Build a concise command fixture."""

    return environment.CommandResult(
        available=available,
        executable=executable,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )


def unavailable_runner(
    arguments: environment.Sequence[str],
    timeout_seconds: float,
) -> environment.CommandResult:
    """Represent every optional external command as unavailable."""

    del arguments, timeout_seconds
    return command_result(
        available=False,
        executable=None,
        exit_code=None,
        error="executable not found",
    )


@pytest.mark.parametrize(
    ("release", "proc_version", "environ", "expected"),
    [
        (
            "5.15.153.1-microsoft-standard-WSL2",
            "Linux version Microsoft",
            {"WSL_DISTRO_NAME": "Ubuntu-24.04", "WSL_INTEROP": "/run/WSL/1"},
            True,
        ),
        ("6.8.0-generic", "Linux version 6.8.0", {}, False),
    ],
)
def test_wsl_detection_uses_controlled_evidence(
    release: str,
    proc_version: str,
    environ: dict[str, str],
    expected: bool,
) -> None:
    detected = environment.detect_wsl(
        kernel_release=release,
        proc_version=proc_version,
        environ=environ,
        runner=unavailable_runner,
    )

    assert detected.detected is expected
    if expected:
        assert detected.generation == "2"
        assert detected.distribution == "Ubuntu-24.04"


def test_wsl_host_command_alone_does_not_classify_process_as_wsl() -> None:
    def runner(
        arguments: environment.Sequence[str],
        timeout_seconds: float,
    ) -> environment.CommandResult:
        del arguments, timeout_seconds
        return command_result(stdout="WSL version: 2.5.0")

    detected = environment.detect_wsl(
        kernel_release="10.0.26200",
        proc_version="",
        environ={},
        runner=runner,
    )

    assert detected.detected is False
    assert detected.evidence == ()
    assert detected.version_text == "WSL version: 2.5.0"


@pytest.mark.parametrize(
    ("path", "wsl", "expected"),
    [
        (Path("/home/research/project"), True, "wsl_linux"),
        (Path("/mnt/c/project"), True, "windows_mounted"),
        (Path("/mnt/d/project"), True, "windows_mounted"),
        (Path("C:/project"), False, "other_or_unknown"),
    ],
)
def test_repository_filesystem_classification(
    path: Path,
    wsl: bool,
    expected: str,
) -> None:
    assert (
        environment.classify_repository_filesystem(path, wsl_detected=wsl) == expected
    )


def test_windows_mount_produces_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, "normalize_path", lambda path: path.as_posix())

    inspected = environment.inspect_repository(
        Path("/mnt/c/project"),
        wsl_detected=True,
        resolved_by="explicit",
    )

    assert inspected.filesystem_class == "windows_mounted"
    assert inspected.warning is not None


def test_path_normalization_removes_home_directory_identity() -> None:
    private_path = Path.home() / "private-workspace" / "artifact.json"
    normalized = environment.normalize_path(private_path)

    assert normalized.startswith("~/")
    assert not normalized.startswith(Path.home().as_posix())


def test_gpu_query_parses_multiple_devices_and_exact_bytes() -> None:
    def runner(
        arguments: environment.Sequence[str],
        timeout_seconds: float,
    ) -> environment.CommandResult:
        del arguments, timeout_seconds
        return command_result(
            executable="/usr/bin/nvidia-smi",
            stdout=(
                "NVIDIA RTX 5070 Laptop GPU, 580.12, 8192, 6144, "
                "2026/07/27 10:00:00.000\n"
                "NVIDIA Test GPU, 580.12, 4096, 2048, "
                "2026/07/27 10:00:00.000"
            ),
        )

    gpu = environment.query_nvidia_gpu(runner=runner)

    assert gpu.error is None
    assert gpu.driver_version == "580.12"
    assert len(gpu.devices) == 2
    assert gpu.devices[0].total_vram_bytes == 8192 * 1024**2
    assert gpu.devices[0].free_vram_bytes == 6144 * 1024**2


@pytest.mark.parametrize(
    "result",
    [
        command_result(
            available=False,
            executable=None,
            exit_code=None,
            error="executable not found",
        ),
        command_result(
            exit_code=9,
            stderr="driver communication failed",
            error="command returned nonzero status",
        ),
        command_result(stdout="malformed"),
    ],
)
def test_gpu_query_failure_remains_structured(
    result: environment.CommandResult,
) -> None:
    def runner(
        arguments: environment.Sequence[str],
        timeout_seconds: float,
    ) -> environment.CommandResult:
        del arguments, timeout_seconds
        return result

    gpu = environment.query_nvidia_gpu(runner=runner)

    assert gpu.devices == ()
    assert gpu.error is not None


def test_meminfo_parser_preserves_exact_bytes() -> None:
    memory = environment.parse_meminfo(
        "\n".join(
            (
                "MemTotal:       32768 kB",
                "MemAvailable:   24576 kB",
                "SwapTotal:       8192 kB",
                "SwapFree:        6144 kB",
            )
        )
    )

    assert memory.total_bytes == 32768 * 1024
    assert memory.available_bytes == 24576 * 1024
    assert memory.swap_total_bytes == 8192 * 1024
    assert memory.swap_used_bytes == 2048 * 1024


def test_disk_inspection_reports_safety_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        environment.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            total=500 * 1024**3,
            used=450 * 1024**3,
            free=50 * 1024**3,
        ),
    )
    monkeypatch.setattr(environment, "_mount_source", lambda path: "test-volume")

    disk = environment.inspect_disk(tmp_path)

    assert disk.total_bytes == 500 * 1024**3
    assert disk.free_bytes == 50 * 1024**3
    assert disk.below_safety_reserve is True
    assert disk.safety_reserve_bytes == 100 * 1024**3


@pytest.mark.parametrize(
    ("result", "available", "version", "error"),
    [
        (command_result(stdout="tool 1.2"), True, "tool 1.2", None),
        (command_result(stderr="tool 2.0"), True, "tool 2.0", None),
        (
            command_result(
                available=False,
                executable=None,
                exit_code=None,
                error="executable not found",
            ),
            False,
            None,
            "executable not found",
        ),
        (
            command_result(
                exit_code=3,
                stderr="failure",
                error="command returned nonzero status",
            ),
            True,
            "failure",
            "command returned nonzero status",
        ),
        (
            command_result(
                exit_code=None,
                error="command timed out after 5 seconds",
            ),
            True,
            None,
            "command timed out after 5 seconds",
        ),
    ],
)
def test_tool_version_handling(
    result: environment.CommandResult,
    available: bool,
    version: str | None,
    error: str | None,
) -> None:
    def runner(
        arguments: environment.Sequence[str],
        timeout_seconds: float,
    ) -> environment.CommandResult:
        del arguments, timeout_seconds
        return result

    tool = environment.inspect_tool("tool", ("tool", "--version"), runner=runner)

    assert tool.available is available
    assert tool.version_text == version
    assert tool.error == error


def test_run_command_captures_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/usr/bin/tool")
    monkeypatch.setattr(environment, "normalize_path", lambda path: path.as_posix())

    def timeout(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd="tool", timeout=1, output="partial")

    monkeypatch.setattr(environment.subprocess, "run", timeout)

    result = environment.run_command(("tool", "--version"), 1)

    assert result.available is True
    assert result.exit_code is None
    assert result.stdout == "partial"
    assert result.error == "command timed out after 1 seconds"


def test_command_output_removes_embedded_nul_characters() -> None:
    assert environment._coerce_output("W\x00S\x00L\x00") == "WSL"


def sample_record() -> environment.EnvironmentRecord:
    """Return a deterministic complete record for serialization tests."""

    return environment.EnvironmentRecord(
        schema_version="1.0",
        timestamp=datetime(2026, 7, 27, tzinfo=UTC).isoformat(),
        platform=environment.PlatformInfo(
            system="Linux",
            release="6.8",
            kernel="Linux test kernel",
            machine="x86_64",
            python_version="3.12.0",
            python_executable="~/venv/bin/python",
            timezone="UTC",
            locale="en_US",
            linux_distribution="Ubuntu",
            linux_distribution_version="24.04",
            windows_host=None,
        ),
        wsl=environment.WSLInfo(True, ("kernel",), "2", "Ubuntu", "WSL version: 2"),
        repository=environment.RepositoryInfo(
            "~/project",
            "git",
            "wsl_linux",
            None,
        ),
        cpu=environment.CPUInfo("Test CPU", 16, "x86_64"),
        memory=environment.MemoryInfo(32, 24, 8, 2),
        gpu=environment.GPUInfo(
            True,
            "/usr/bin/nvidia-smi",
            0,
            "580.12",
            (
                environment.GPUDevice(
                    0,
                    "Test GPU",
                    "580.12",
                    8 * 1024**3,
                    6 * 1024**3,
                    "2026/07/27 10:00:00.000",
                ),
            ),
            "",
            None,
        ),
        disk=environment.DiskInfo(500, 100, 400, "/dev/test", False),
        tools=(environment.ToolInfo("git", True, "/usr/bin/git", "git 2", 0, None),),
    )


@pytest.mark.serialization
def test_json_serialization_is_deterministic_and_round_trips() -> None:
    record = sample_record()

    encoded = record.to_json()
    restored = environment.EnvironmentRecord.from_dict(json.loads(encoded))

    assert encoded == record.to_json()
    assert restored == record


def test_write_record_creates_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "environment.json"

    environment.write_record(output, sample_record())

    restored = environment.EnvironmentRecord.from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )
    assert restored == sample_record()
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.integration
@pytest.mark.parametrize("json_mode", [False, True])
def test_module_entry_point_runs_in_subprocess(json_mode: bool) -> None:
    arguments = [
        sys.executable,
        "-m",
        "gaussweave.runtime.environment",
    ]
    if json_mode:
        arguments.append("--json")

    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    if json_mode:
        value = json.loads(completed.stdout)
        assert value["schema_version"] == "1.0"
    else:
        assert "WSL:" in completed.stdout
        assert "Repository filesystem:" in completed.stdout


def test_environment_module_import_has_no_heavy_side_effects() -> None:
    script = """
import json
import sys
import gaussweave.runtime.environment

prefixes = ("torch", "gsplat", "bpy", "blender", "pycolmap", "gaussweave_native")
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
