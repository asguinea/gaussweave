"""Process-isolated Blender discovery, execution, and qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from gaussweave.accounting.resources import (
    PROFILES,
    classify_compliance,
    cpu_memory_snapshot,
    disk_snapshot,
    nvidia_snapshot,
)
from gaussweave.config.resolution import canonical_json_bytes, pretty_json_bytes
from gaussweave.experiments.lifecycle import (
    CommandResult,
    command_failure,
    create_state,
    run_command,
    write_failure,
)
from gaussweave.results.artifacts import (
    ArtifactDeclaration,
    ArtifactPath,
    VerificationState,
    build_inventory,
    hash_file,
    verify_inventory,
    write_inventory,
)

BLENDER_ENVIRONMENT_VARIABLE = "GAUSSWEAVE_BLENDER_EXECUTABLE"
WINDOWS_BLENDER_ENVIRONMENT_VARIABLE = "GAUSSWEAVE_WINDOWS_BLENDER_EXECUTABLE"
LOCAL_CONFIG_RELATIVE = "configs/local.json"
DEFAULT_WSL_EXECUTABLE = Path("/opt/gaussweave/blender-4.5.12-linux-x64/blender")
QUALIFICATION_MARKER = ".gaussweave-blender-qualification"
QUALIFICATION_SCRIPT = "qualify_blender.py"
VERSION_RE = re.compile(r"^Blender (?P<version>[0-9]+\.[0-9]+\.[0-9]+(?: LTS)?)$", re.M)
PYTHON_INFO_PREFIX = "GAUSSWEAVE_PYTHON="
INVENTORY_PATH = "artifact-inventory.json"
INVENTORY_EXCLUSIONS = (INVENTORY_PATH, QUALIFICATION_MARKER)


class BlenderError(RuntimeError):
    """Blender discovery, validation, or execution failed."""


class BlenderEnvironmentError(BlenderError):
    """The requested Blender environment is unavailable or invalid."""


class BlenderExecutionError(BlenderError):
    """Blender launched but the requested operation failed."""


class BlenderBackend(StrEnum):
    WSL = "wsl"
    WINDOWS = "windows"


class BlenderDevice(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class BlenderExecutable:
    path: str
    version: str
    build_hash: str | None
    embedded_python: str
    backend: str


@dataclass(frozen=True)
class BlenderInvocation:
    executable: Path
    script: Path
    script_root: Path
    ordered_script_arguments: tuple[str, ...]
    working_directory: Path
    output_root: Path
    timeout_seconds: float
    selected_environment: Mapping[str, str]
    backend: BlenderBackend
    stdout_path: Path = Path("stdout.log")
    stderr_path: Path = Path("stderr.log")


@dataclass(frozen=True)
class BlenderExecutionResult:
    executable: BlenderExecutable
    backend: str
    arguments: tuple[str, ...]
    working_directory: str
    selected_environment: Mapping[str, str]
    started_at: str
    ended_at: str
    elapsed_seconds: float
    exit_code: int | None
    timed_out: bool
    stdout_ref: str | None
    stderr_ref: str | None
    generated_artifact_refs: tuple[str, ...]
    warnings: tuple[str, ...]
    failure_ref: str | None
    success: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["arguments"] = list(self.arguments)
        value["selected_environment"] = dict(self.selected_environment)
        value["generated_artifact_refs"] = list(self.generated_artifact_refs)
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class QualificationResult:
    valid: bool
    status: str
    root: str
    backend: str
    device: str
    seed: int
    blender_version: str
    embedded_python: str
    render_engine: str | None
    compute_backend: str | None
    compute_device: str | None
    elapsed_seconds: float
    image_path: str | None
    image_bytes: int | None
    image_digest: str | None
    blend_digest: str | None
    metadata_digest: str | None
    artifact_count: int
    inventory_state: str
    warnings: tuple[str, ...]
    failure_ref: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["warnings"] = list(self.warnings)
        return value


def discover_blender(
    *,
    backend: BlenderBackend,
    explicit: Path | None = None,
    repository_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Discover one executable using explicit, environment, local, then PATH order."""

    env = environment if environment is not None else os.environ
    root = repository_root or _repository_root()
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    variable = (
        BLENDER_ENVIRONMENT_VARIABLE
        if backend is BlenderBackend.WSL
        else WINDOWS_BLENDER_ENVIRONMENT_VARIABLE
    )
    if env.get(variable):
        candidates.append(Path(env[variable]))
    local = _read_local_config(root)
    local_key = (
        "blender_executable"
        if backend is BlenderBackend.WSL
        else "windows_blender_executable"
    )
    if isinstance(local.get(local_key), str):
        candidates.append(Path(str(local[local_key])))
    found = shutil.which("blender" if backend is BlenderBackend.WSL else "blender.exe")
    if found:
        candidates.append(Path(found))
    if backend is BlenderBackend.WSL:
        candidates.append(DEFAULT_WSL_EXECUTABLE)
    if not candidates:
        raise BlenderEnvironmentError(
            f"no {backend.value} Blender executable was configured or discovered"
        )
    selected = candidates[0].expanduser()
    if not selected.is_absolute():
        path_match = shutil.which(str(selected))
        if path_match:
            selected = Path(path_match)
    if not selected.is_file():
        raise BlenderEnvironmentError(f"Blender executable is not a file: {selected}")
    if backend is BlenderBackend.WSL and not os.access(selected, os.X_OK):
        raise BlenderEnvironmentError(
            f"Blender executable is not executable: {selected}"
        )
    return selected.resolve()


def capture_blender_identity(
    executable: Path,
    *,
    backend: BlenderBackend,
    working_directory: Path,
    timeout_seconds: float = 30.0,
) -> BlenderExecutable:
    """Capture version, build hash, and embedded Python through safe commands."""

    version_result = run_command(
        [str(executable), "--version"],
        working_directory=working_directory,
        timeout_seconds=timeout_seconds,
    )
    if not version_result.success:
        raise BlenderEnvironmentError(
            _command_message("Blender version query failed", version_result)
        )
    match = VERSION_RE.search(version_result.stdout)
    if match is None:
        raise BlenderEnvironmentError("Blender version output was not recognized")
    build_match = re.search(r"^\s*build hash:\s*(\S+)", version_result.stdout, re.M)
    expression = (
        "import json,sys;"
        f"print('{PYTHON_INFO_PREFIX}'+json.dumps({{'version':sys.version}}))"
    )
    python_result = run_command(
        [
            str(executable),
            "--background",
            "--factory-startup",
            "--python-expr",
            expression,
        ],
        working_directory=working_directory,
        timeout_seconds=timeout_seconds,
    )
    if not python_result.success:
        raise BlenderEnvironmentError(
            _command_message("Blender embedded Python query failed", python_result)
        )
    info_line = next(
        (
            line.removeprefix(PYTHON_INFO_PREFIX)
            for line in python_result.stdout.splitlines()
            if line.startswith(PYTHON_INFO_PREFIX)
        ),
        None,
    )
    if info_line is None:
        raise BlenderEnvironmentError("embedded Python output was not recognized")
    try:
        embedded_python = str(json.loads(info_line)["version"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise BlenderEnvironmentError("embedded Python output was invalid") from error
    return BlenderExecutable(
        str(executable),
        match.group("version"),
        build_match.group(1) if build_match else None,
        embedded_python,
        backend.value,
    )


def build_blender_arguments(invocation: BlenderInvocation) -> tuple[str, ...]:
    """Validate roots and build a shell-free Blender argument array."""

    if invocation.timeout_seconds <= 0:
        raise BlenderError("timeout must be positive")
    script = _contained_file(invocation.script_root, invocation.script, "script")
    output = invocation.output_root.resolve()
    working = invocation.working_directory.resolve()
    try:
        output.relative_to(working)
    except ValueError as error:
        raise BlenderError(
            "output root must remain within the working directory"
        ) from error
    executable = str(invocation.executable)
    script_value = str(script)
    ordered = invocation.ordered_script_arguments
    if invocation.backend is BlenderBackend.WINDOWS:
        executable = str(invocation.executable)
        script_value = _translate_windows_path(script, working)
        ordered = _translate_path_arguments(ordered, working)
    return (
        executable,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        script_value,
        "--",
        *ordered,
    )


def execute_blender(
    invocation: BlenderInvocation,
    *,
    identity: BlenderExecutable | None = None,
) -> BlenderExecutionResult:
    """Execute Blender via the P1 command runner and persist failures."""

    invocation.output_root.mkdir(parents=True, exist_ok=True)
    (invocation.output_root / "metadata").mkdir(exist_ok=True)
    executable = identity or capture_blender_identity(
        invocation.executable,
        backend=invocation.backend,
        working_directory=invocation.working_directory,
    )
    arguments = build_blender_arguments(invocation)
    result = run_command(
        arguments,
        working_directory=invocation.output_root,
        environment=invocation.selected_environment,
        selected_environment_keys=tuple(invocation.selected_environment),
        timeout_seconds=invocation.timeout_seconds,
        stdout_path=invocation.stdout_path,
        stderr_path=invocation.stderr_path,
    )
    artifacts = tuple(
        path
        for path in (
            "source/qualify_blender.py",
            "scene.blend",
            "render.png",
            "generation-metadata.json",
            "stdout.log",
            "stderr.log",
        )
        if (invocation.output_root / path).is_file()
    )
    failure_ref: str | None = None
    if not result.success:
        state = create_state(
            run_id="run-blender-qualification-s0",
            experiment_id="exp-blender-qualification-v1",
            scene_id="syn-blender-qualification",
            attempt=1,
            configuration_ref="generation-metadata.json",
            configuration_digest=f"sha256:{hashlib.sha256(b'blender').hexdigest()}",
        )
        failure = command_failure(
            state,
            result,
            stage="blender-execution",
            category="environment_incompatibility",
        )
        json_path, _ = write_failure(failure, invocation.output_root)
        failure_ref = json_path.relative_to(invocation.output_root).as_posix()
    warnings = (
        ("Windows fallback execution; outputs are Windows-generated.",)
        if invocation.backend is BlenderBackend.WINDOWS
        else ()
    )
    return BlenderExecutionResult(
        executable,
        invocation.backend.value,
        result.arguments,
        result.working_directory,
        result.selected_environment,
        result.started_at,
        result.ended_at,
        result.elapsed_seconds,
        result.exit_code,
        result.timed_out,
        result.stdout_ref,
        result.stderr_ref,
        artifacts,
        warnings,
        failure_ref,
        result.success,
    )


def qualify_blender(
    root: Path,
    *,
    backend: BlenderBackend = BlenderBackend.WSL,
    device: BlenderDevice = BlenderDevice.CPU,
    seed: int = 17,
    blender_executable: Path | None = None,
    timeout_seconds: float = 180.0,
    overwrite: bool = False,
    repository_root: Path | None = None,
) -> QualificationResult:
    """Run one bounded qualification and strictly verify its evidence."""

    if seed < 0 or seed > 2**32 - 1:
        raise BlenderError("seed must be an unsigned 32-bit integer")
    repository = (repository_root or _repository_root()).resolve()
    output = root.expanduser().resolve()
    _prepare_output_root(output, overwrite=overwrite)
    marker = output / QUALIFICATION_MARKER
    marker.write_text("GaussWeave Blender qualification root\n", encoding="utf-8")
    source = _contained_file(
        repository / "blender_scripts",
        repository / "blender_scripts" / QUALIFICATION_SCRIPT,
        "qualification script",
    )
    copied_script = ArtifactPath.resolve(output, f"source/{QUALIFICATION_SCRIPT}").local
    copied_script.parent.mkdir()
    shutil.copyfile(source, copied_script)
    executable = discover_blender(
        backend=backend,
        explicit=blender_executable,
        repository_root=repository,
    )
    identity = capture_blender_identity(
        executable,
        backend=backend,
        working_directory=output,
        timeout_seconds=min(timeout_seconds, 30.0),
    )
    script_arguments = (
        "--output-root",
        str(output),
        "--seed",
        str(seed),
        "--device",
        device.value,
    )
    invocation = BlenderInvocation(
        executable,
        copied_script,
        output,
        script_arguments,
        output,
        output,
        timeout_seconds,
        {
            "PYTHONHASHSEED": str(seed),
            "BLENDER_USER_CONFIG": str(output / ".blender-user-config"),
            "BLENDER_USER_SCRIPTS": str(output / ".blender-user-scripts"),
        },
        backend,
    )
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    baseline = cpu_memory_snapshot()
    execution = execute_blender(invocation, identity=identity)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_child_bytes = int(max(child_before, child_after)) * (
        1 if sys.platform == "darwin" else 1024
    )
    final_cpu = cpu_memory_snapshot()
    disk = disk_snapshot(output)
    nvidia = nvidia_snapshot()
    compliance = classify_compliance(
        PROFILES["smoke"], cpu_rss_bytes=peak_child_bytes, disk=disk
    )
    resource_record: dict[str, Any] = {
        "record_version": "1.0",
        "operation": "blender-qualification",
        "backend": backend.value,
        "device": device.value,
        "cpu_baseline": asdict(baseline),
        "cpu_final": asdict(final_cpu),
        "peak_child_rss_bytes": peak_child_bytes,
        "disk": asdict(disk),
        "nvidia": asdict(nvidia),
        "compliance": compliance.to_dict(),
        "elapsed_seconds": execution.elapsed_seconds,
    }
    _write_json(output / "resource-record.json", resource_record)
    summary = execution.to_dict()
    summary["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
    }
    summary["seed"] = seed
    summary["device"] = device.value
    _write_json(output / "execution-summary.json", summary)
    if not execution.success:
        raise BlenderExecutionError(
            "Blender timed out"
            if execution.timed_out
            else f"Blender failed with exit code {execution.exit_code}; "
            f"failure: {execution.failure_ref}"
        )
    metadata_path = output / "generation-metadata.json"
    if not metadata_path.is_file():
        raise BlenderExecutionError("Blender did not emit generation metadata")
    metadata = _read_json(metadata_path)
    status = str(metadata.get("status", "failed"))
    gpu_unsupported = device is BlenderDevice.GPU and status == "unsupported"
    required_paths = [
        f"source/{QUALIFICATION_SCRIPT}",
        "scene.blend",
        "generation-metadata.json",
        "stdout.log",
        "stderr.log",
        "resource-record.json",
        "execution-summary.json",
    ]
    if not gpu_unsupported:
        required_paths.append("render.png")
    declarations = tuple(
        ArtifactDeclaration(
            f"blender-{index:02d}",
            _artifact_type(path),
            path,
            description=f"Blender qualification artifact: {path}",
        )
        for index, path in enumerate(required_paths, start=1)
    )
    inventory = build_inventory(
        output,
        declarations,
        inventory_id=f"inventory-blender-{device.value}-s{seed}",
        excluded_paths=INVENTORY_EXCLUSIONS,
    )
    write_inventory(inventory, output / INVENTORY_PATH)
    verification = verify_inventory(inventory, output, strict=True)
    if verification.state is VerificationState.INVALID:
        raise BlenderExecutionError(
            "Blender artifact inventory failed strict verification"
        )
    image = output / "render.png"
    image_identity = hash_file(image) if image.is_file() else None
    if image_identity is not None:
        width, height = _png_dimensions(image)
        expected = metadata.get("render", {}).get("resolution", {})
        if (width, height) != (expected.get("width"), expected.get("height")):
            raise BlenderExecutionError("rendered PNG dimensions do not match metadata")
    blend = output / "scene.blend"
    metadata_identity = hash_file(metadata_path)
    warnings = tuple(execution.warnings) + tuple(
        str(x) for x in metadata.get("warnings", [])
    )
    return QualificationResult(
        True,
        status,
        output.as_posix(),
        backend.value,
        device.value,
        seed,
        identity.version,
        identity.embedded_python,
        _nested_string(metadata, "render", "engine"),
        _nested_string(metadata, "compute", "backend"),
        _nested_string(metadata, "compute", "device_name"),
        execution.elapsed_seconds,
        "render.png" if image.is_file() else None,
        image_identity.bytes if image_identity else None,
        image_identity.digest if image_identity else None,
        hash_file(blend).digest if blend.is_file() else None,
        metadata_identity.digest,
        inventory.artifact_count,
        verification.state.value,
        warnings,
        execution.failure_ref,
    )


def scientific_metadata_digest(path: Path) -> str:
    """Digest scientific metadata after recursively removing volatile fields."""

    metadata = _read_json(path)
    stable = _without_volatile(metadata)
    return f"sha256:{hashlib.sha256(canonical_json_bytes(stable)).hexdigest()}"


def compare_qualification_runs(first: Path, second: Path) -> dict[str, Any]:
    """Compare two clean qualification roots under the declared determinism policy."""

    first_metadata = _read_json(first / "generation-metadata.json")
    second_metadata = _read_json(second / "generation-metadata.json")
    first_stable = _without_volatile(first_metadata)
    second_stable = _without_volatile(second_metadata)
    first_inventory = _read_json(first / INVENTORY_PATH)
    second_inventory = _read_json(second / INVENTORY_PATH)
    first_members = sorted(item["path"] for item in first_inventory["artifacts"])
    second_members = sorted(item["path"] for item in second_inventory["artifacts"])
    first_image = hash_file(first / "render.png")
    second_image = hash_file(second / "render.png")
    exact_file = first_image.digest == second_image.digest
    first_pixels = _png_pixel_payload_digest(first / "render.png")
    second_pixels = _png_pixel_payload_digest(second / "render.png")
    exact_pixels = first_pixels == second_pixels
    exact_metadata = first_stable == second_stable
    membership = first_members == second_members
    return {
        "valid": exact_metadata and exact_pixels and membership,
        "structural_metadata_exact": exact_metadata,
        "rendered_image_file_exact": exact_file,
        "rendered_pixel_payload_exact": exact_pixels,
        "rendered_pixel_payload_digest": first_pixels,
        "render_policy": (
            "decompressed PNG pixel payload must match exactly; whole-file digest is "
            "reported but may differ because Blender writes volatile PNG metadata"
        ),
        "artifact_membership_exact": membership,
        "first_scientific_digest": scientific_metadata_digest(
            first / "generation-metadata.json"
        ),
        "second_scientific_digest": scientific_metadata_digest(
            second / "generation-metadata.json"
        ),
        "blend_digest_equal": (
            hash_file(first / "scene.blend").digest
            == hash_file(second / "scene.blend").digest
        ),
        "blend_policy": (
            "reported but not required because .blend files may contain volatile "
            "Blender-internal state"
        ),
    }


def _prepare_output_root(root: Path, *, overwrite: bool) -> None:
    if root.exists() and not root.is_dir():
        raise BlenderError(f"output root is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise BlenderError(f"refusing to reuse nonempty output root: {root}")
        marker = root / QUALIFICATION_MARKER
        if not marker.is_file():
            raise BlenderError("overwrite requires a marked Blender qualification root")
        for child in root.iterdir():
            try:
                child.resolve().relative_to(root.resolve())
            except ValueError as error:
                raise BlenderError("output child escapes qualification root") from error
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    root.mkdir(parents=True, exist_ok=True)


def _read_local_config(repository_root: Path) -> dict[str, Any]:
    path = repository_root / LOCAL_CONFIG_RELATIVE
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BlenderEnvironmentError(f"invalid local configuration: {path}") from error
    if not isinstance(value, dict):
        raise BlenderEnvironmentError("local configuration must be a JSON object")
    return value


def _contained_file(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise BlenderError(f"{label} must remain within {resolved_root}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise BlenderError(f"{label} is not a regular file: {resolved}")
    return resolved


def _translate_windows_path(path: Path, working_directory: Path) -> str:
    result = run_command(
        ["wslpath", "-w", str(path)],
        working_directory=working_directory,
        timeout_seconds=10.0,
    )
    if not result.success or not result.stdout.strip():
        raise BlenderEnvironmentError(
            _command_message("WSL path translation failed", result)
        )
    return result.stdout.strip()


def _translate_path_arguments(
    arguments: tuple[str, ...], working_directory: Path
) -> tuple[str, ...]:
    translated: list[str] = []
    path_option = False
    for argument in arguments:
        if path_option:
            translated.append(
                _translate_windows_path(Path(argument), working_directory)
            )
            path_option = False
        else:
            translated.append(argument)
            path_option = argument in {
                "--config",
                "--output-root",
                "--provenance",
                "--provenance-ref",
            }
    return tuple(translated)


def _command_message(prefix: str, result: CommandResult) -> str:
    evidence = result.stderr.strip() or result.stdout.strip()
    return f"{prefix}: {evidence or f'exit code {result.exit_code}'}"


def _repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise BlenderEnvironmentError("repository root could not be located")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(pretty_json_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BlenderExecutionError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise BlenderExecutionError(f"JSON artifact is not an object: {path}")
    return value


def _nested_string(value: Mapping[str, Any], first: str, second: str) -> str | None:
    nested = value.get(first)
    if not isinstance(nested, Mapping):
        return None
    item = nested.get(second)
    return item if isinstance(item, str) else None


def _without_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_volatile(item)
            for key, item in value.items()
            if key not in {"created_at", "ended_at", "elapsed_seconds", "started_at"}
        }
    if isinstance(value, list):
        return [_without_volatile(item) for item in value]
    return value


def _artifact_type(path: str) -> str:
    if path == "render.png":
        return "render"
    if path == "resource-record.json":
        return "resource_record"
    if path.startswith("metadata/failure-"):
        return "failure_report"
    return "other"


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise BlenderExecutionError("render output is not a valid PNG")
    return struct.unpack(">II", header[16:24])


def _png_pixel_payload_digest(path: Path) -> str:
    content = path.read_bytes()
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise BlenderExecutionError("render output is not a valid PNG")
    position = 8
    compressed = bytearray()
    while position < len(content):
        if position + 12 > len(content):
            raise BlenderExecutionError("rendered PNG is truncated")
        length = struct.unpack(">I", content[position : position + 4])[0]
        chunk_type = content[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(content):
            raise BlenderExecutionError("rendered PNG has an invalid chunk")
        if chunk_type == b"IDAT":
            compressed.extend(content[position + 8 : position + 8 + length])
        position = end
        if chunk_type == b"IEND":
            break
    try:
        pixels = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise BlenderExecutionError("rendered PNG pixel payload is invalid") from error
    return f"sha256:{hashlib.sha256(pixels).hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qualify headless Blender generation")
    parser.add_argument("--root", required=True)
    parser.add_argument("--backend", choices=("wsl", "windows"), default="wsl")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--blender-executable")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        result = qualify_blender(
            Path(options.root),
            backend=BlenderBackend(options.backend),
            device=BlenderDevice(options.device),
            seed=options.seed,
            blender_executable=(
                Path(options.blender_executable) if options.blender_executable else None
            ),
            timeout_seconds=options.timeout_seconds,
            overwrite=options.overwrite,
        )
    except BlenderEnvironmentError as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 4
    except BlenderError as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 6
    if options.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"Blender {result.blender_version}: {result.status}; "
            f"{result.backend}/{result.device}; artifacts={result.artifact_count}; "
            f"inventory={result.inventory_state}; elapsed={result.elapsed_seconds:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def run_blender_scene_generation(*args: Any, **kwargs: Any) -> Any:
    """Lazily dispatch to the generic scene-generation runner.

    Keeping this wrapper here preserves the accepted Blender adapter API while
    avoiding a circular import between the process adapter and runtime runner.
    """

    from gaussweave.data.blender_runtime import run_blender_scene_generation as run

    return run(*args, **kwargs)
