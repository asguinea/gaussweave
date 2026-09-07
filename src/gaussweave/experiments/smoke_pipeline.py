"""End-to-end reproducible qualification smoke experiment orchestration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from gaussweave.accounting.resources import ResourceGuardError
from gaussweave.config.errors import DocumentLoadError, SchemaValidationError
from gaussweave.config.resolution import (
    ResolutionError,
    ResolvedConfiguration,
    generate_run_id,
    pretty_json_bytes,
    resolve_layers,
    write_resolution_outputs,
)
from gaussweave.experiments.tiny_training import (
    TinyTrainingConfig,
    TrainingError,
    run_tiny_training,
)
from gaussweave.results.artifacts import (
    ArtifactError,
    VerificationState,
    hash_file,
    inventory_tree,
    load_inventory,
    verify_inventory,
    write_inventory,
)
from gaussweave.results.records import (
    ResultRecordError,
    build_smoke_result,
    load_result,
)
from gaussweave.runtime.environment import inspect_environment

DEFAULT_CONFIG = Path(__file__).parents[3] / "configs/experiments/smoke.json"
RESULT_PATH = "artifacts/result.json"
SUMMARY_PATH = "metadata/pipeline-summary.json"
PRIMARY_INVENTORY_PATH = "artifact-inventory.json"
EVIDENCE_INVENTORY_PATH = "evidence-inventory.json"
PIPELINE_VERSION = "1.0.0"


class SmokePipelineError(RuntimeError):
    """Base error for the reproducible smoke pipeline."""

    exit_code = 6

    def __init__(
        self,
        message: str,
        *,
        run_directory: Path | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.run_directory = run_directory
        self.details = dict(details or {})


class PipelineValidationError(SmokePipelineError):
    exit_code = 3


class PipelineResourceError(SmokePipelineError):
    exit_code = 5


class PipelineOperationError(SmokePipelineError):
    exit_code = 6


@dataclass(frozen=True)
class SmokePipelineSummary:
    experiment_id: str
    run_id: str
    result_id: str
    status: str
    scene_id: str
    seed: int
    scientific_digest: str
    complete_digest: str
    final_gaussian_count: int
    initial_training_loss: float
    final_training_loss: float
    scene_psnr_db: float | str
    training_seconds: float | None
    render_median_ms: float | None
    render_p95_ms: float | None
    gpu_peak_allocated_bytes: int | None
    gpu_peak_reserved_bytes: int | None
    resource_compliance: str
    checkpoint_path: str
    checkpoint_bytes: int
    checkpoint_digest: str
    result_path: str
    result_digest: str
    inventory_path: str
    inventory_digest: str
    evidence_inventory_path: str
    evidence_artifact_count: int
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = {
            key: item
            for key, item in self.__dict__.items()
            if key not in {"warnings", "limitations"}
        }
        value["warnings"] = list(self.warnings)
        value["limitations"] = list(self.limitations)
        return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(pretty_json_bytes(dict(value)))
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    except OSError as error:
        raise PipelineOperationError(f"unable to write {path}: {error}") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PipelineValidationError(f"{name} must be an object")
    return value


def _single_integer(values: object, name: str) -> int:
    if (
        not isinstance(values, list)
        or len(values) != 1
        or isinstance(values[0], bool)
        or not isinstance(values[0], int)
    ):
        raise PipelineValidationError(f"{name} must contain exactly one integer")
    return cast(int, values[0])


def _stopping_value(document: Mapping[str, Any], kind: str) -> int:
    stopping = _mapping(document.get("stopping"), "stopping")
    rules = stopping.get("rules")
    if not isinstance(rules, list):
        raise PipelineValidationError("stopping.rules must be an array")
    matches = [
        rule.get("value")
        for rule in rules
        if isinstance(rule, dict) and rule.get("kind") == kind
    ]
    if (
        len(matches) != 1
        or isinstance(matches[0], bool)
        or not isinstance(matches[0], int)
    ):
        raise PipelineValidationError(f"one integer {kind} stopping rule is required")
    return matches[0]


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise PipelineValidationError(f"{name} must be a nonempty string array")
    return tuple(value)


def training_config(resolved: ResolvedConfiguration) -> TinyTrainingConfig:
    """Translate accepted experiment fields into the accepted tiny trainer model."""

    document = resolved.document
    execution = _mapping(document.get("execution"), "execution")
    rendering = _mapping(document.get("rendering"), "rendering")
    method = _mapping(document.get("method"), "method")
    operating = _mapping(method.get("operating_point"), "method.operating_point")
    seed = _single_integer(execution.get("seeds"), "execution.seeds")
    iterations = _stopping_value(document, "iteration_count")
    gaussian_cap = _stopping_value(document, "gaussian_cap")
    initial_count = operating.get("initial_gaussian_count")
    if isinstance(initial_count, bool) or not isinstance(initial_count, int):
        raise PipelineValidationError("initial_gaussian_count must be an integer")
    width = rendering.get("width")
    height = rendering.get("height")
    if width != 64 or height != 64:
        raise PipelineValidationError("tiny qualification rendering must be 64x64")
    if document.get("mode") != "development":
        raise PipelineValidationError("smoke pipeline requires development mode")
    freeze = _mapping(document.get("freeze"), "freeze")
    if freeze.get("status") != "unfrozen":
        raise PipelineValidationError("smoke pipeline result must remain unfrozen")
    return TinyTrainingConfig(
        seed=seed,
        width=width,
        height=height,
        train_camera_ids=_string_tuple(
            operating.get("train_camera_ids"), "train_camera_ids"
        ),
        validation_camera_ids=_string_tuple(
            operating.get("validation_camera_ids"), "validation_camera_ids"
        ),
        test_camera_ids=_string_tuple(
            operating.get("test_camera_ids"), "test_camera_ids"
        ),
        iterations=iterations,
        initial_gaussian_count=initial_count,
        maximum_gaussian_count=gaussian_cap,
        resource_profile=resolved.resource_profile,
    )


def _scene_seed_attempt(
    resolved: ResolvedConfiguration,
) -> tuple[str, int, int]:
    document = resolved.document
    dataset = _mapping(document.get("dataset"), "dataset")
    selection = _mapping(dataset.get("scene_selection"), "dataset.scene_selection")
    scene_ids = _string_tuple(selection.get("scene_ids"), "scene_ids")
    if len(scene_ids) != 1:
        raise PipelineValidationError("smoke pipeline requires exactly one scene")
    execution = _mapping(document.get("execution"), "execution")
    seed = _single_integer(execution.get("seeds"), "execution.seeds")
    attempt = execution.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise PipelineValidationError("execution.attempt must be an integer")
    return scene_ids[0], seed, attempt


def _environment_payload(repository_root: Path) -> dict[str, Any]:
    record = inspect_environment(repository_root).to_dict()
    try:
        import gsplat  # type: ignore[import-untyped]
        import torch

        record["locked_gpu_runtime"] = {
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gsplat_version": getattr(gsplat, "__version__", "unknown"),
        }
    except (ImportError, OSError) as error:
        raise PipelineResourceError(
            f"locked GPU environment unavailable: {error}"
        ) from error
    return record


def _write_resolution(resolved: ResolvedConfiguration, run: Path) -> None:
    write_resolution_outputs(
        resolved,
        run / "resolved_config.json",
        provenance_output=run / "metadata/configuration-provenance.json",
        canonical_output=run / "metadata/resolved-config.canonical.json",
        digest_output=run / "metadata/configuration-digests.json",
        overwrite=True,
    )


def _partial_inventory(run: Path) -> None:
    inventory = inventory_tree(
        run,
        inventory_id="inventory-smoke-partial",
        excluded_paths=(PRIMARY_INVENTORY_PATH,),
    )
    write_inventory(
        inventory,
        run / PRIMARY_INVENTORY_PATH,
        overwrite=(run / PRIMARY_INVENTORY_PATH).exists(),
    )


def run_smoke_pipeline(
    root: Path,
    layer_paths: Sequence[Path],
    *,
    inject_failure: bool = False,
    artificial_gpu_limit_bytes: int | None = None,
    repository_root: Path | None = None,
) -> SmokePipelineSummary:
    """Resolve, execute, evaluate, record, inventory, and verify one smoke run."""

    repository = (repository_root or Path.cwd()).resolve()
    try:
        resolved = resolve_layers(layer_paths, repository_root=repository)
        config = training_config(resolved)
        scene_id, seed, attempt = _scene_seed_attempt(resolved)
        run_id = generate_run_id(
            experiment_id=resolved.experiment_id,
            scene_id=scene_id,
            seed=seed,
            scientific_configuration_digest=resolved.scientific_digest,
            attempt=attempt,
        )
    except (
        DocumentLoadError,
        ResolutionError,
        SchemaValidationError,
        KeyError,
    ) as error:
        raise PipelineValidationError(str(error)) from error
    run = root / resolved.experiment_id / run_id
    if run.exists():
        raise PipelineOperationError(
            f"refusing to reuse existing run directory: {run}", run_directory=run
        )
    environment = _environment_payload(repository)
    try:
        outcome = run_tiny_training(
            root,
            config,
            inject_failure=inject_failure,
            artificial_gpu_limit_bytes=artificial_gpu_limit_bytes,
            run_id=run_id,
            experiment_id=resolved.experiment_id,
            configuration_digest=resolved.scientific_digest,
            resolved_configuration=resolved.document,
            environment_record=environment,
        )
    except Exception as error:
        if run.is_dir():
            _write_resolution(resolved, run)
            _partial_inventory(run)
        error_type = (
            PipelineResourceError
            if isinstance(error, ResourceGuardError)
            else PipelineOperationError
        )
        raise error_type(
            str(error),
            run_directory=run if run.is_dir() else None,
            details={
                "experiment_id": resolved.experiment_id,
                "run_id": run_id,
                "scientific_digest": resolved.scientific_digest,
                "partial_inventory": (
                    (run / PRIMARY_INVENTORY_PATH).as_posix()
                    if (run / PRIMARY_INVENTORY_PATH).is_file()
                    else None
                ),
            },
        ) from error
    run = outcome.run_directory
    try:
        _write_resolution(resolved, run)
        result_outcome = build_smoke_result(
            run,
            run / RESULT_PATH,
            inventory_excluded_paths=(
                RESULT_PATH,
                SUMMARY_PATH,
                EVIDENCE_INVENTORY_PATH,
            ),
        )
        result = load_result(run / RESULT_PATH)
        primary = load_inventory(run / PRIMARY_INVENTORY_PATH)
        metric = result["metrics"][0]
        resources = result["resources"]
        runtime = result["runtime"]
        training_summary = json.loads(
            (run / "metadata/training-summary.json").read_text(encoding="utf-8")
        )
        checkpoint = hash_file(outcome.checkpoint_path)
        summary = SmokePipelineSummary(
            resolved.experiment_id,
            run_id,
            str(result["result_id"]),
            "completed",
            scene_id,
            seed,
            resolved.scientific_digest,
            resolved.full_digest,
            outcome.gaussian_count,
            outcome.initial_loss,
            outcome.final_loss,
            metric["value"],
            training_summary["training_resource"].get("elapsed_seconds"),
            runtime.get("frame_latency_ms_median"),
            runtime.get("frame_latency_ms_p95"),
            resources.get("gpu_peak_allocated_bytes"),
            resources.get("gpu_peak_reserved_bytes"),
            str(resources["compliance"]),
            outcome.checkpoint_path.relative_to(run).as_posix(),
            checkpoint.bytes,
            checkpoint.digest,
            RESULT_PATH,
            str(result_outcome["result_digest"]),
            PRIMARY_INVENTORY_PATH,
            primary.content_digest,
            EVIDENCE_INVENTORY_PATH,
            0,
            tuple(environment.get("warnings", [])),
            tuple(str(item) for item in result["limitations"]),
        )
        _atomic_write(run / SUMMARY_PATH, summary.to_dict())
        evidence = inventory_tree(
            run,
            inventory_id="inventory-smoke-evidence",
            excluded_paths=(EVIDENCE_INVENTORY_PATH,),
        )
        write_inventory(evidence, run / EVIDENCE_INVENTORY_PATH)
        final_summary = SmokePipelineSummary(
            **{
                **summary.__dict__,
                "evidence_artifact_count": evidence.artifact_count,
            }
        )
        _atomic_write(run / SUMMARY_PATH, final_summary.to_dict())
        evidence = inventory_tree(
            run,
            inventory_id="inventory-smoke-evidence",
            excluded_paths=(EVIDENCE_INVENTORY_PATH,),
        )
        write_inventory(evidence, run / EVIDENCE_INVENTORY_PATH, overwrite=True)
        verification = verify_inventory(evidence, run, strict=True)
        if verification.state is not VerificationState.VALID:
            raise PipelineOperationError(
                "final strict artifact verification failed",
                run_directory=run,
                details=verification.to_dict(),
            )
        return final_summary
    except (
        ArtifactError,
        OSError,
        ResultRecordError,
        SchemaValidationError,
        TrainingError,
        ValueError,
    ) as error:
        raise PipelineOperationError(str(error), run_directory=run) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke", help="run the reproducible qualification smoke"
    )
    smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke.add_argument("--overlay", action="append", type=Path, default=[])
    smoke.add_argument("--root", type=Path, required=True)
    smoke.add_argument("--inject-failure", action="store_true")
    smoke.add_argument("--artificial-gpu-limit-bytes", type=int)
    smoke.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        summary = run_smoke_pipeline(
            options.root,
            (options.config, *options.overlay),
            inject_failure=options.inject_failure,
            artificial_gpu_limit_bytes=options.artificial_gpu_limit_bytes,
        )
    except SmokePipelineError as error:
        payload = {
            "valid": False,
            "status": "failed",
            "error": str(error),
            "exit_code": error.exit_code,
            "run_directory": (
                error.run_directory.as_posix()
                if error.run_directory is not None
                else None
            ),
            **error.details,
        }
        print(
            json.dumps(payload, indent=2, sort_keys=True)
            if options.json
            else f"smoke pipeline failed: {error}"
        )
        return error.exit_code
    payload = {"valid": True, **summary.to_dict()}
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if options.json
        else (
            f"{summary.run_id}: loss {summary.initial_training_loss:.6f} -> "
            f"{summary.final_training_loss:.6f}; "
            f"PSNR={summary.scene_psnr_db} dB; "
            f"artifacts={summary.evidence_artifact_count}; "
            f"result={summary.result_path}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
