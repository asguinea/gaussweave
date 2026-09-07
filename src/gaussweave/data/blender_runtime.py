"""Project-side orchestration for the deterministic Blender generator runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gaussweave.accounting.resources import (
    PROFILES,
    classify_compliance,
    cpu_memory_snapshot,
    disk_snapshot,
    nvidia_snapshot,
)
from gaussweave.config.errors import (
    DocumentLoadError,
    SchemaValidationError,
)
from gaussweave.config.resolution import pretty_json_bytes
from gaussweave.data.blender import (
    BlenderBackend,
    BlenderDevice,
    BlenderEnvironmentError,
    BlenderError,
    BlenderExecutionError,
    BlenderInvocation,
    capture_blender_identity,
    discover_blender,
    execute_blender,
)
from gaussweave.data.cameras import (
    generate_camera_collection,
    write_camera_artifacts,
)
from gaussweave.data.render_passes import (
    RenderPassError,
    validate_render_root,
    write_render_request,
)
from gaussweave.data.scene_config import (
    SceneConfiguration,
    build_blender_handoff,
    load_scene_configuration,
)
from gaussweave.results.artifacts import (
    ArtifactDeclaration,
    VerificationState,
    build_inventory,
    hash_file,
    verify_inventory,
    write_inventory,
)

RUNTIME_MARKER = ".gaussweave-runtime-probe"
RUNTIME_INVENTORY = "artifact-inventory.json"
RUNTIME_VERSION = "blender-runtime-v1"
INVENTORY_EXCLUSIONS = (RUNTIME_MARKER, RUNTIME_INVENTORY)


@dataclass(frozen=True)
class BlenderGenerationResult:
    """Typed project-side result for one generic runtime probe."""

    valid: bool
    status: str
    root: str
    scene_id: str
    family: str
    backend: str
    device: str
    master_seed: int
    derived_seeds: dict[str, int]
    configuration_digest: str
    scientific_metadata_digest: str | None
    metadata_file_digest: str | None
    execution_summary_digest: str | None
    blender_version: str
    embedded_python: str
    elapsed_seconds: float
    preview_path: str | None
    preview_digest: str | None
    preview_pixel_digest: str | None
    artifact_count: int
    inventory_state: str
    warnings: tuple[str, ...]
    failure_ref: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["warnings"] = list(self.warnings)
        return value


def run_blender_scene_generation(
    resolved_scene_config: Path | SceneConfiguration,
    output_root: Path,
    *,
    mode: str = "runtime-probe",
    backend: BlenderBackend = BlenderBackend.WSL,
    device: BlenderDevice = BlenderDevice.CPU,
    blender_executable: Path | None = None,
    timeout_seconds: float = 300.0,
    camera_seed: int | None = None,
    camera_ids: tuple[str, ...] | None = None,
    overwrite: bool = False,
    inject_failure: str | None = None,
    repository_root: Path | None = None,
) -> BlenderGenerationResult:
    """Validate, launch, record, and strictly inventory one runtime probe."""

    if mode not in {"runtime-probe", "camera-probe", "render-pass-probe"}:
        raise BlenderError(f"unsupported generation mode: {mode}")
    if timeout_seconds <= 0:
        raise BlenderError("timeout must be positive")
    repository = (repository_root or _repository_root()).resolve()
    if isinstance(resolved_scene_config, SceneConfiguration):
        config = resolved_scene_config
        config_path = repository / config.provenance.source_config
        if not config_path.is_file():
            raise BlenderError(
                "SceneConfiguration provenance does not reference a readable config"
            )
    else:
        config_path = resolved_scene_config.expanduser().resolve()
        config = load_scene_configuration(config_path)
    output = _validate_output_root(output_root)
    _prepare_output_root(output, scene_id=config.scene_id, overwrite=overwrite)
    marker = output / RUNTIME_MARKER
    cameras_path: Path | None = None
    collection = None
    render_pass_mode = mode == "render-pass-probe"
    if mode in {"camera-probe", "render-pass-probe"}:
        collection = generate_camera_collection(config, camera_seed=camera_seed)
        write_camera_artifacts(collection, output)
        cameras_path = output / "cameras/cameras.json"
    marker.write_text(
        f"GaussWeave {mode} root\nscene_id={config.scene_id}\n",
        encoding="utf-8",
    )
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script_root = repository / "blender_scripts"
    script = script_root / "generate_scene.py"
    if script.is_symlink() or not script.is_file():
        raise BlenderEnvironmentError(
            f"generator entry script is unavailable: {script}"
        )
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
    handoff = build_blender_handoff(
        config,
        resolved_config_path=str(config_path),
        output_root=str(output),
        backend=backend.value,
        timeout_seconds=timeout_seconds,
        provenance_ref=config.provenance.source_config,
        render_device=device.value,
        generator_version=RUNTIME_VERSION,
    )
    arguments = list(handoff.ordered_script_arguments())
    arguments[0:0] = ("--mode", mode)
    if cameras_path is not None:
        arguments.extend(("--cameras", str(cameras_path)))
    if render_pass_mode:
        if collection is None:
            raise AssertionError("camera collection was not generated")
        selected = (
            camera_ids
            if camera_ids
            else tuple(
                collection.splits[split][0] for split in ("train", "validation", "test")
            )
        )
        render_request_path = output / "metadata/render_request.json"
        write_render_request(
            render_request_path,
            config=config,
            camera_collection=collection,
            camera_ids=selected,
        )
        arguments.extend(("--render-policy", str(render_request_path)))
    if inject_failure is not None:
        if inject_failure not in {"after-reset", "before-validation", "validation"}:
            raise BlenderError(f"unsupported injected failure: {inject_failure}")
        arguments.extend(("--inject-failure", inject_failure))
    invocation = BlenderInvocation(
        executable,
        script,
        script_root,
        tuple(arguments),
        output.parent,
        output,
        timeout_seconds,
        {
            "PYTHONHASHSEED": str(config.master_seed),
            "BLENDER_USER_CONFIG": str(output / ".blender-user-config"),
            "BLENDER_USER_SCRIPTS": str(output / ".blender-user-scripts"),
        },
        backend,
        Path("logs/stdout.log"),
        Path("logs/stderr.log"),
    )
    baseline = cpu_memory_snapshot()
    execution = execute_blender(invocation, identity=identity)
    final_cpu = cpu_memory_snapshot()
    disk = disk_snapshot(output)
    nvidia = nvidia_snapshot()
    profile = PROFILES[config.expected_resource_profile.value]
    compliance = classify_compliance(
        profile,
        cpu_rss_bytes=final_cpu.peak_process_rss_bytes,
        disk=disk,
    )
    resource_record = {
        "record_version": "1.0",
        "operation": f"blender-{mode}",
        "resource_profile": config.expected_resource_profile.value,
        "backend": backend.value,
        "device": device.value,
        "cpu_baseline": asdict(baseline),
        "cpu_final": asdict(final_cpu),
        "disk": asdict(disk),
        "nvidia": asdict(nvidia),
        "compliance": compliance.to_dict(),
        "elapsed_seconds": execution.elapsed_seconds,
        "failure_ref": execution.failure_ref,
    }
    _write_json(output / "logs/resource-record.json", resource_record)
    command_record = execution.to_dict()
    command_record["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
    }
    _write_json(output / "logs/external-command.json", command_record)
    if not execution.success:
        inventory, verification = _inventory_existing(output, config.scene_id)
        message = (
            "Blender timed out"
            if execution.timed_out
            else f"Blender failed with exit code {execution.exit_code}"
        )
        raise BlenderExecutionError(
            f"{message}; failure: {execution.failure_ref}; "
            f"inventory={verification.state.value}; "
            f"artifacts={inventory.artifact_count}"
        )
    metadata_path = output / "source/generator_metadata.json"
    summary_path = output / "source/execution_summary.json"
    metadata = _read_json(metadata_path)
    summary = _read_json(summary_path)
    _validate_metadata(metadata, config, device)
    render_validation = None
    if render_pass_mode:
        try:
            render_validation = validate_render_root(output)
        except RenderPassError as error:
            raise BlenderExecutionError(
                f"decoded render-pass validation failed: {error}"
            ) from error
    inventory, verification = _inventory_existing(output, config.scene_id)
    if verification.state is VerificationState.INVALID:
        raise BlenderExecutionError(
            "runtime artifact inventory failed strict verification"
        )
    if mode == "runtime-probe":
        preview = output / "preview/runtime_probe.png"
        preview_portable = "preview/runtime_probe.png"
        checksum_paths = {
            "source/resolved_scene_config.json",
            "source/generator_metadata.json",
            "source/runtime_probe.blend",
            "source/execution_summary.json",
            "preview/runtime_probe.png",
        }
    elif mode == "camera-probe":
        if collection is None:
            raise AssertionError("camera collection was not generated")
        preview_portable = f"preview/cameras/{collection.splits['train'][0]}.png"
        preview = output.joinpath(*preview_portable.split("/"))
        checksum_paths = {
            "source/resolved_scene_config.json",
            "source/generator_metadata.json",
            "source/camera_probe.blend",
            "source/camera_installation.json",
            "source/execution_summary.json",
            *(
                f"preview/cameras/{collection.splits[split][0]}.png"
                for split in ("train", "validation", "test")
            ),
        }
    else:
        if collection is None or render_validation is None:
            raise AssertionError("render-pass validation was not completed")
        render_collection = _read_json(output / "metadata/render_collection.json")
        selected_ids = tuple(
            str(view["camera_id"]) for view in render_collection["views"]
        )
        preview_portable = f"renders/rgb/{selected_ids[0]}.png"
        preview = output.joinpath(*preview_portable.split("/"))
        checksum_paths = {
            "source/resolved_scene_config.json",
            "source/generator_metadata.json",
            "source/render_pass_probe.blend",
            "source/camera_installation.json",
            "source/execution_summary.json",
            "metadata/render_request.json",
            "metadata/render_conventions.json",
            "metadata/render_collection.json",
            "metadata/pass_statistics.json",
            *(
                str(artifact["path"])
                for view in render_collection["views"]
                for artifact in view["pass_artifacts"]
            ),
        }
    _validate_png(preview, config.render.width, config.render.height)
    _validate_checksum_file(output, checksum_paths)
    scientific = metadata["scientific"]
    summary_scientific = summary["scientific"]
    warnings = tuple(execution.warnings) + tuple(
        str(value) for value in scientific.get("warnings", [])
    )
    return BlenderGenerationResult(
        True,
        str(summary_scientific["status"]),
        output.as_posix(),
        config.scene_id,
        config.family.value,
        backend.value,
        device.value,
        config.master_seed,
        dict(config.derived_seeds),
        config.full_digest,
        str(metadata["scientific_digest"]),
        hash_file(metadata_path).digest,
        str(summary["execution_summary_digest"]),
        identity.version,
        identity.embedded_python,
        execution.elapsed_seconds,
        preview_portable,
        hash_file(preview).digest,
        _png_pixel_digest(preview),
        inventory.artifact_count,
        verification.state.value,
        warnings,
        execution.failure_ref,
    )


def compare_runtime_probes(first: Path, second: Path) -> dict[str, Any]:
    """Compare deterministic scientific metadata and decoded PNG scanlines."""

    first_metadata = _read_json(first / "source/generator_metadata.json")
    second_metadata = _read_json(second / "source/generator_metadata.json")
    first_preview = first / "preview/runtime_probe.png"
    second_preview = second / "preview/runtime_probe.png"
    return {
        "scientific_metadata_exact": first_metadata["scientific"]
        == second_metadata["scientific"],
        "scientific_digest_equal": first_metadata["scientific_digest"]
        == second_metadata["scientific_digest"],
        "preview_pixels_exact": _png_pixel_digest(first_preview)
        == _png_pixel_digest(second_preview),
        "first_scientific_digest": first_metadata["scientific_digest"],
        "second_scientific_digest": second_metadata["scientific_digest"],
        "first_preview_pixel_digest": _png_pixel_digest(first_preview),
        "second_preview_pixel_digest": _png_pixel_digest(second_preview),
    }


def _validate_output_root(value: Path) -> Path:
    if ".." in value.parts:
        raise BlenderError("output root may not contain traversal")
    expanded = value.expanduser()
    if expanded.is_symlink():
        raise BlenderError("output root may not be a symlink")
    current = expanded if expanded.is_absolute() else Path.cwd() / expanded
    for parent in (current, *current.parents):
        if parent.exists() and parent.is_symlink():
            raise BlenderError("output root may not cross a symlink")
    resolved = expanded.resolve()
    anchor = Path(resolved.anchor)
    if resolved == anchor or resolved == Path.home().resolve():
        raise BlenderError("output root is too broad")
    return resolved


def _prepare_output_root(root: Path, *, scene_id: str, overwrite: bool) -> None:
    if root.exists() and not root.is_dir():
        raise BlenderError(f"output root is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise BlenderError(f"refusing to reuse nonempty output root: {root}")
        marker = root / RUNTIME_MARKER
        if not marker.is_file() or f"scene_id={scene_id}" not in marker.read_text(
            encoding="utf-8"
        ):
            raise BlenderError(
                "overwrite requires a matching marked runtime-probe root"
            )
        resolved_root = root.resolve()
        children = list(root.iterdir())
        for child in children:
            if child.is_symlink():
                raise BlenderError("refusing overwrite with symlink child")
            try:
                child.resolve().relative_to(resolved_root)
            except ValueError as error:
                raise BlenderError("output child escapes runtime-probe root") from error
        for child in children:
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    root.mkdir(parents=True, exist_ok=True)


def _inventory_existing(root: Path, scene_id: str) -> tuple[Any, Any]:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in INVENTORY_EXCLUSIONS
    )
    declarations = tuple(
        ArtifactDeclaration(
            artifact_id=f"runtime-{index:02d}",
            artifact_type=_artifact_type(path),
            path=path,
            description=f"Blender runtime-probe artifact: {path}",
        )
        for index, path in enumerate(paths, start=1)
    )
    inventory = build_inventory(
        root,
        declarations,
        inventory_id=f"inventory-{scene_id}",
        excluded_paths=INVENTORY_EXCLUSIONS,
    )
    write_inventory(inventory, root / RUNTIME_INVENTORY, overwrite=True)
    return inventory, verify_inventory(inventory, root, strict=True)


def _artifact_type(path: str) -> str:
    if path == "source/resolved_scene_config.json":
        return "resolved_config"
    if path.startswith("preview/") or path.startswith("renders/"):
        return "render"
    if path == "logs/resource-record.json":
        return "resource_record"
    if "failure-" in path or path == "source/runtime_failure.json":
        return "failure_report"
    return "other"


def _validate_metadata(
    metadata: dict[str, Any],
    config: SceneConfiguration,
    device: BlenderDevice,
) -> None:
    scientific = metadata.get("scientific")
    if not isinstance(scientific, dict):
        raise BlenderExecutionError("generator metadata lacks scientific content")
    required = {
        "metadata_schema_version",
        "generator",
        "scene_id",
        "family",
        "configuration_digest",
        "seeds",
        "coordinates",
        "collections",
        "names",
        "objects",
        "materials",
        "camera",
        "lights",
        "render",
        "artifacts",
        "environment",
        "warnings",
    }
    if missing := sorted(required - set(scientific)):
        raise BlenderExecutionError(
            "generator metadata is missing: " + ", ".join(missing)
        )
    if scientific["scene_id"] != config.scene_id:
        raise BlenderExecutionError("generator metadata scene ID mismatch")
    if scientific["configuration_digest"] != config.full_digest:
        raise BlenderExecutionError("generator metadata configuration digest mismatch")
    if scientific["seeds"]["derived_seeds"] != config.derived_seeds:
        raise BlenderExecutionError("generator metadata seed mismatch")
    render_device = scientific["render"]["device"]["requested"]
    if render_device != device.value:
        raise BlenderExecutionError("generator metadata device mismatch")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BlenderExecutionError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise BlenderExecutionError(f"JSON artifact is not an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def _validate_png(path: Path, expected_width: int, expected_height: int) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise BlenderExecutionError("runtime preview is missing or empty")
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise BlenderExecutionError("runtime preview is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (expected_width, expected_height):
        raise BlenderExecutionError(
            f"runtime preview dimensions are {width}x{height}, "
            f"expected {expected_width}x{expected_height}"
        )


def _validate_checksum_file(root: Path, expected_paths: set[str]) -> None:
    path = root / "checksums.sha256"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise BlenderExecutionError("runtime checksum file is unreadable") from error
    found: set[str] = set()
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise BlenderExecutionError("runtime checksum line is malformed")
        digest, portable = line[:64], line[66:]
        if (
            len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
            or portable in found
        ):
            raise BlenderExecutionError("runtime checksum record is invalid")
        target = root.joinpath(*portable.split("/"))
        if hash_file(target).digest != f"sha256:{digest}":
            raise BlenderExecutionError(f"runtime checksum mismatch: {portable}")
        found.add(portable)
    if found != expected_paths:
        raise BlenderExecutionError("runtime checksum membership mismatch")


def _png_pixel_digest(path: Path) -> str:
    content = path.read_bytes()
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise BlenderExecutionError("runtime preview is not a PNG")
    position = 8
    compressed = bytearray()
    while position < len(content):
        length = struct.unpack(">I", content[position : position + 4])[0]
        chunk_type = content[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(content):
            raise BlenderExecutionError("runtime preview is truncated")
        if chunk_type == b"IDAT":
            compressed.extend(content[position + 8 : position + 8 + length])
        position = end
        if chunk_type == b"IEND":
            break
    try:
        pixels = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise BlenderExecutionError(
            "runtime preview pixel payload is invalid"
        ) from error
    import hashlib

    return f"sha256:{hashlib.sha256(pixels).hexdigest()}"


def _repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise BlenderEnvironmentError("repository root could not be located")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the generic deterministic Blender runtime probe. "
            "This command does not generate a family scene or dataset."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--mode",
        choices=("runtime-probe", "camera-probe", "render-pass-probe"),
        default="runtime-probe",
    )
    parser.add_argument("--backend", choices=("wsl", "windows"), default="wsl")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--blender-executable")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--camera-seed", type=int)
    parser.add_argument("--camera-id", action="append")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--inject-failure",
        choices=("after-reset", "before-validation", "validation"),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        result = run_blender_scene_generation(
            Path(options.config),
            Path(options.root),
            mode=options.mode,
            backend=BlenderBackend(options.backend),
            device=BlenderDevice(options.device),
            blender_executable=(
                Path(options.blender_executable) if options.blender_executable else None
            ),
            timeout_seconds=options.timeout_seconds,
            camera_seed=options.camera_seed,
            camera_ids=tuple(options.camera_id) if options.camera_id else None,
            overwrite=options.overwrite,
            inject_failure=options.inject_failure,
        )
    except (DocumentLoadError, SchemaValidationError, ValueError) as error:
        if isinstance(error, SchemaValidationError):
            payload = error.to_dict()
        elif isinstance(error, DocumentLoadError):
            payload = {"valid": False, "errors": [error.issue.to_dict()]}
        else:
            payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, sort_keys=True))
        return 3
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
            f"Blender runtime probe {result.scene_id}: {result.status}; "
            f"{result.backend}/{result.device}; seed={result.master_seed}; "
            f"scientific={result.scientific_metadata_digest}; "
            f"artifacts={result.artifact_count}; inventory={result.inventory_state}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
