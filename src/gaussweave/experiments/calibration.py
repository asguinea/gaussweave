"""Bounded deterministic E0 hardware-calibration orchestration."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from gaussweave.accounting.resources import nvidia_snapshot
from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.experiments.lifecycle import load_state
from gaussweave.experiments.tiny_training import (
    TinyTrainingConfig,
    run_tiny_training,
)
from gaussweave.results.records import build_smoke_result
from gaussweave.runtime.environment import inspect_environment

CALIBRATION_VERSION = "1.0"


class CalibrationError(RuntimeError):
    """Calibration configuration or execution is invalid."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    width: int
    height: int
    iterations: int
    initial_gaussian_count: int
    maximum_gaussian_count: int
    camera_batch_size: int
    precision: str
    repetitions: int
    mode: str = "real"
    expected_workspace_bytes: int = 1024**3

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cal-"):
            raise CalibrationError("candidate IDs must start with cal-")
        if min(self.width, self.height, self.iterations, self.repetitions) < 1:
            raise CalibrationError("candidate dimensions/counts must be positive")
        if not 1 <= self.initial_gaussian_count <= self.maximum_gaussian_count:
            raise CalibrationError("candidate Gaussian counts are invalid")
        if self.camera_batch_size not in (1, 2):
            raise CalibrationError("tiny calibration camera batch must be 1 or 2")
        if self.precision != "float32":
            raise CalibrationError(
                "only scientifically comparable float32 is supported"
            )
        if self.mode not in {"real", "artificial_guard"}:
            raise CalibrationError(f"unsupported candidate mode: {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())


def load_calibration(path: Path) -> tuple[dict[str, Any], tuple[Candidate, ...]]:
    """Load and validate one ordered calibration document."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationError(f"unable to load calibration config: {error}") from error
    if not isinstance(document, dict):
        raise CalibrationError("calibration config must be an object")
    if document.get("calibration_version") != CALIBRATION_VERSION:
        raise CalibrationError("unsupported calibration_version")
    if document.get("seed") != 17:
        raise CalibrationError("qualification calibration seed must be 17")
    values = document.get("candidates")
    if not isinstance(values, list) or len(values) < 4:
        raise CalibrationError(
            "calibration matrix must contain at least four candidates"
        )
    try:
        candidates = tuple(Candidate(**value) for value in values)
    except TypeError as error:
        raise CalibrationError(f"invalid candidate fields: {error}") from error
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise CalibrationError("candidate IDs must be unique")
    return document, candidates


def expand_matrix(path: Path) -> tuple[dict[str, Any], ...]:
    """Return deterministic ordered candidate identities for tests and review."""

    _, candidates = load_calibration(path)
    return tuple(
        {
            **candidate.to_dict(),
            "resolved_configuration_digest": candidate.digest,
        }
        for candidate in candidates
    )


def _candidate_experiment(
    base: Mapping[str, Any], candidate: Candidate
) -> dict[str, Any]:
    value = json.loads(json.dumps(base))
    value["experiment_id"] = "exp-hardware-calibration-v1"
    value["description"] = "Unfrozen E0 laptop hardware qualification fixture."
    value["execution"]["resource_profile"] = "standard"
    operating = value["method"]["operating_point"]
    operating["operating_point_id"] = candidate.candidate_id
    operating["initial_gaussian_count"] = candidate.initial_gaussian_count
    operating["maximum_gaussian_count"] = candidate.maximum_gaussian_count
    rendering = value["rendering"]
    rendering["width"] = candidate.width
    rendering["height"] = candidate.height
    rendering["camera_batch_size"] = candidate.camera_batch_size
    for rule in value["stopping"]["rules"]:
        if rule["kind"] == "iteration_count":
            rule["value"] = candidate.iterations
        elif rule["kind"] == "gaussian_cap":
            rule["value"] = candidate.maximum_gaussian_count
        elif rule["kind"] == "memory_guard":
            rule["value"] = "standard"
    value["matrix"] = {
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
    }
    value["notes"] = (
        "Deterministic qualification calibration fixture; not representative of the "
        "representative synthetic benchmark."
    )
    return cast(dict[str, Any], value)


def _workspace_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"expected JSON object: {path}")
    return value


def _record_failure(
    candidate: Candidate,
    repetition: int,
    repeat_root: Path,
    error: Exception,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    statuses = sorted(repeat_root.rglob("status.json"))
    state = load_state(statuses[0]) if len(statuses) == 1 else None
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "repetition": repetition,
        "seed": 17,
        "status": "failed",
        "compliance": "resource_failure",
        "failure_ref": state.failure_ref if state is not None else None,
        "error": str(error),
        "resolution": [candidate.width, candidate.height],
        "iterations": candidate.iterations,
        "initial_gaussian_count": candidate.initial_gaussian_count,
        "maximum_gaussian_count": candidate.maximum_gaussian_count,
        "final_gaussian_count": None,
        "camera_count": 4,
        "camera_batch_size": candidate.camera_batch_size,
        "precision": candidate.precision,
        "initial_loss": None,
        "final_loss": None,
        "held_out_psnr_db": None,
        "training_seconds": None,
        "render_median_ms": None,
        "render_p95_ms": None,
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "process_gpu_peak_bytes": None,
        "cpu_peak_rss_bytes": None,
        "disk_free_bytes": None,
        "workspace_bytes": _workspace_bytes(repeat_root),
        "checkpoint_bytes": None,
        "checkpoint_file_digest": None,
        "checkpoint_scientific_digest": None,
        "warnings": ["controlled artificial resource-guard failure"],
        "telemetry": {"before": dict(before), "after": dict(after)},
    }


def _record_success(
    candidate: Candidate,
    repetition: int,
    run: Path,
    outcome: Any,
    result: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _read(run / "metadata/training-summary.json")
    resource = summary["training_resource"]
    gpu = resource.get("gpu_final") or {}
    cpu = resource.get("cpu_peak") or {}
    nvidia = resource.get("nvidia") or {}
    disk = resource.get("disk") or {}
    runtime = result["result"]["runtime"]
    metric = result["result"]["metrics"][0]
    warnings = [
        warning["message"] for warning in resource["compliance"].get("warnings", [])
    ]
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.digest,
        "repetition": repetition,
        "seed": 17,
        "status": "completed",
        "compliance": resource["compliance"]["state"],
        "failure_ref": None,
        "error": None,
        "resolution": [candidate.width, candidate.height],
        "iterations": candidate.iterations,
        "initial_gaussian_count": candidate.initial_gaussian_count,
        "maximum_gaussian_count": candidate.maximum_gaussian_count,
        "final_gaussian_count": outcome.gaussian_count,
        "camera_count": 4,
        "camera_batch_size": candidate.camera_batch_size,
        "precision": candidate.precision,
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
        "held_out_psnr_db": metric["value"],
        "training_seconds": resource["elapsed_seconds"],
        "render_median_ms": runtime["frame_latency_ms_median"],
        "render_p95_ms": runtime["frame_latency_ms_p95"],
        "gpu_peak_allocated_bytes": gpu.get("peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": gpu.get("peak_reserved_bytes"),
        "process_gpu_peak_bytes": nvidia.get("process_gpu_bytes"),
        "cpu_peak_rss_bytes": cpu.get("peak_process_rss_bytes"),
        "disk_free_bytes": disk.get("free_bytes"),
        "workspace_bytes": _workspace_bytes(run),
        "checkpoint_bytes": summary["checkpoint_bytes"],
        "checkpoint_file_digest": result["result"]["artifacts"][3]["digest"],
        "checkpoint_scientific_digest": outcome.scientific_checkpoint_digest,
        "warnings": warnings,
        "telemetry": {"before": dict(before), "after": dict(after)},
    }


def _variation(values: list[float | int]) -> dict[str, float]:
    floats = [float(value) for value in values]
    return {
        "minimum": min(floats),
        "maximum": max(floats),
        "range": max(floats) - min(floats),
        "mean": statistics.fmean(floats),
        "population_standard_deviation": statistics.pstdev(floats),
    }


def summarize_candidates(
    candidates: Sequence[Candidate], records: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize repetitions and deterministically select a standard candidate."""

    summaries: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        selected = [
            dict(record)
            for record in records
            if record["candidate_id"] == candidate.candidate_id
        ]
        successful = [record for record in selected if record["status"] == "completed"]
        reasons: list[str] = []
        if candidate.mode != "real":
            reasons.append("controlled failure-path candidate")
        if candidate.repetitions < 2:
            reasons.append("not a repeated finalist")
        if len(successful) != candidate.repetitions:
            reasons.append("not all required repetitions completed")
        if any(
            not str(record["compliance"]).startswith("compliant")
            for record in successful
        ):
            reasons.append("resource compliance failure")
        if successful:
            if (
                max(
                    int(record["gpu_peak_allocated_bytes"] or 0)
                    for record in successful
                )
                > 4 * 1024**3
            ):
                reasons.append("GPU allocated target exceeded")
            if (
                max(
                    int(record["gpu_peak_reserved_bytes"] or 0) for record in successful
                )
                > 5 * 1024**3
            ):
                reasons.append("GPU reserved target exceeded")
            if (
                max(int(record["cpu_peak_rss_bytes"] or 0) for record in successful)
                > 10 * 1024**3
            ):
                reasons.append("CPU RSS target exceeded")
            if (
                max(float(record["training_seconds"] or 0) for record in successful)
                > 30
            ):
                reasons.append("qualification runtime target exceeded")
            if (
                max(int(record["workspace_bytes"]) for record in successful)
                > 2 * 1024**3
            ):
                reasons.append("workspace target exceeded")
            psnr = [float(record["held_out_psnr_db"]) for record in successful]
            if len(psnr) > 1 and statistics.pstdev(psnr) > 0.10:
                reasons.append("held-out PSNR variability exceeded 0.10 dB")
            scientific = {
                str(record["checkpoint_scientific_digest"]) for record in successful
            }
            if len(scientific) > 1:
                reasons.append("scientific checkpoint identity was not repeatable")
        summary = {
            "candidate_id": candidate.candidate_id,
            "candidate_digest": candidate.digest,
            "required_repetitions": candidate.repetitions,
            "completed_repetitions": len(successful),
            "eligible": not reasons,
            "rejection_reasons": reasons,
            "variability": {
                field: _variation(
                    [
                        record[field]
                        for record in successful
                        if record[field] is not None
                    ]
                )
                if any(record[field] is not None for record in successful)
                else None
                for field in (
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                    "cpu_peak_rss_bytes",
                    "training_seconds",
                    "held_out_psnr_db",
                    "workspace_bytes",
                )
            },
        }
        summaries.append(summary)
        if summary["eligible"]:
            eligible.append(summary)
    if not eligible:
        raise CalibrationError(
            "no stable standard candidate satisfied the selection rule"
        )
    chosen = max(
        eligible,
        key=lambda summary: (
            summary["variability"]["held_out_psnr_db"]["mean"],
            -summary["variability"]["training_seconds"]["mean"],
            summary["candidate_id"],
        ),
    )
    return summaries, {
        "candidate_id": chosen["candidate_id"],
        "rule_version": "g14-standard-selection-v1",
        "eligibility": {
            "required_repetitions": 2,
            "all_repetitions_completed": True,
            "compliance_prefix": "compliant",
            "gpu_allocated_max_bytes": 4 * 1024**3,
            "gpu_reserved_max_bytes": 5 * 1024**3,
            "cpu_rss_max_bytes": 10 * 1024**3,
            "training_seconds_max": 30,
            "workspace_max_bytes": 2 * 1024**3,
            "psnr_population_sd_max_db": 0.10,
            "scientific_checkpoint_digest_must_match": True,
        },
        "ranking": "highest mean held-out PSNR, then lower mean training time, then ID",
    }


def run_calibration(config_path: Path, root: Path) -> dict[str, Any]:
    """Run every matrix point in an isolated immutable repetition directory."""

    if root.exists() and any(root.iterdir()):
        raise CalibrationError(f"calibration root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    document, candidates = load_calibration(config_path)
    base_path = Path(str(document["base_experiment_config"]))
    base = _read(base_path)
    environment: dict[str, Any] = dict(
        inspect_environment(repo_root=Path.cwd()).to_dict()
    )
    if environment["repository"]["filesystem_class"] != "wsl_linux":
        raise CalibrationError("qualification requires a WSL Linux filesystem checkout")
    records: list[dict[str, Any]] = []
    run_order = 0
    for candidate in candidates:
        experiment = _candidate_experiment(base, candidate)
        for repetition in range(1, candidate.repetitions + 1):
            run_order += 1
            repeat_root = root / candidate.candidate_id / f"repeat-{repetition:02d}"
            before = asdict(nvidia_snapshot())
            config = TinyTrainingConfig(
                seed=int(document["seed"]),
                width=candidate.width,
                height=candidate.height,
                camera_batch_size=candidate.camera_batch_size,
                iterations=candidate.iterations,
                validation_interval=max(1, min(10, candidate.iterations)),
                logging_interval=max(1, min(10, candidate.iterations)),
                initial_gaussian_count=candidate.initial_gaussian_count,
                maximum_gaussian_count=candidate.maximum_gaussian_count,
                resource_profile="standard",
            )
            run_id = (
                "run-exp-calibration-v1-syn-tiny-s17-"
                f"{candidate.digest[7:19]}-a{repetition}"
            )
            try:
                outcome = run_tiny_training(
                    repeat_root,
                    config,
                    artificial_gpu_limit_bytes=(
                        1 if candidate.mode == "artificial_guard" else None
                    ),
                    run_id=run_id,
                    experiment_id="exp-hardware-calibration-v1",
                    configuration_digest=candidate.digest,
                    resolved_configuration=experiment,
                    environment_record=environment,
                )
                result = build_smoke_result(
                    outcome.run_directory,
                    outcome.run_directory / "result.json",
                    inventory_excluded_paths=("result.json",),
                )
                after = asdict(nvidia_snapshot())
                record = _record_success(
                    candidate,
                    repetition,
                    outcome.run_directory,
                    outcome,
                    result,
                    before,
                    after,
                )
            except Exception as error:
                after = asdict(nvidia_snapshot())
                if candidate.mode == "real":
                    record = _record_failure(
                        candidate, repetition, repeat_root, error, before, after
                    )
                else:
                    record = _record_failure(
                        candidate, repetition, repeat_root, error, before, after
                    )
            record["run_order"] = run_order
            records.append(record)
    summaries, selection = summarize_candidates(candidates, records)
    output = {
        "calibration_version": CALIBRATION_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "configuration_digest": content_digest(document),
        "repository": environment["repository"],
        "environment": {
            "platform": environment["platform"],
            "wsl": environment["wsl"],
            "memory": environment["memory"],
            "gpu": environment["gpu"],
            "disk": environment["disk"],
        },
        "workload_policy": document["workload_policy"],
        "candidate_records": records,
        "candidate_summaries": summaries,
        "selection": selection,
    }
    _atomic_write(root / "calibration-summary.json", output)
    return output


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
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
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        summary = run_calibration(options.config, options.root)
    except Exception as error:
        payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, indent=2) if options.json else f"error: {error}")
        return 1
    payload = {
        "valid": True,
        "candidate_records": len(summary["candidate_records"]),
        "selected_candidate": summary["selection"]["candidate_id"],
        "summary": (options.root / "calibration-summary.json").as_posix(),
    }
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if options.json
        else (
            f"calibration: {payload['candidate_records']} records; "
            f"selected={payload['selected_candidate']}; summary={payload['summary']}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
