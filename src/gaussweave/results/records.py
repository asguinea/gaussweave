"""Build and validate traceable qualification smoke result records."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from gaussweave.config.errors import DocumentLoadError, SchemaValidationError
from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.config.validation import validate_result
from gaussweave.metrics import (
    METRIC_VERSION,
    MetricRecord,
    MetricStatus,
    aggregate_scene_psnr,
    compute_per_view,
)
from gaussweave.results.artifacts import hash_file, inventory_tree, write_inventory

RESULT_VERSION = "1.0.0"
DEFINITIONS_VERSION = "definitions-v1.0"


class ResultRecordError(RuntimeError):
    """Raised for invalid run evidence or semantically inconsistent results."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResultRecordError(f"unable to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ResultRecordError(f"expected JSON object: {path}")
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ResultRecordError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(pretty_json_bytes(dict(payload)))
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    except OSError as error:
        raise ResultRecordError(f"unable to write {path}: {error}") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _locate_run(root: Path) -> Path:
    if (root / "status.json").is_file():
        return root
    candidates = sorted(path.parent for path in root.rglob("status.json"))
    if len(candidates) != 1:
        raise ResultRecordError(
            f"run root must contain exactly one run, found {len(candidates)}"
        )
    return candidates[0]


def read_ppm(path: Path) -> list[list[list[float]]]:
    """Read project-owned binary P6 output and normalize its declared range."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ResultRecordError(f"unable to read image {path}: {error}") from error
    tokens: list[bytes] = []
    position = 0
    while len(tokens) < 4:
        while position < len(payload) and payload[position] in b" \t\r\n":
            position += 1
        if position < len(payload) and payload[position] == ord("#"):
            position = payload.find(b"\n", position)
            if position < 0:
                raise ResultRecordError(f"malformed PPM header: {path}")
            continue
        end = position
        while end < len(payload) and payload[end] not in b" \t\r\n":
            end += 1
        tokens.append(payload[position:end])
        position = end
    if tokens[0] != b"P6":
        raise ResultRecordError(f"unsupported image format: {path}")
    try:
        width, height, maximum = (int(item) for item in tokens[1:])
    except ValueError as error:
        raise ResultRecordError(f"invalid PPM header: {path}") from error
    if width < 1 or height < 1 or maximum != 255:
        raise ResultRecordError(f"unsupported PPM dimensions/range: {path}")
    while position < len(payload) and payload[position] in b" \t\r\n":
        position += 1
    pixels = payload[position:]
    if len(pixels) != width * height * 3:
        raise ResultRecordError(f"PPM payload size mismatch: {path}")
    values = [item / 255.0 for item in pixels]
    return [
        [
            values[(row * width + column) * 3 : (row * width + column + 1) * 3]
            for column in range(width)
        ]
        for row in range(height)
    ]


def _artifact(
    run: Path, artifact_id: str, artifact_type: str, relative: str
) -> dict[str, Any]:
    identity = hash_file(run / relative)
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "path": relative,
        "required": True,
        "bytes": identity.bytes,
        "digest": identity.digest,
    }


def _refresh_inventory(run: Path, *, excluded_paths: tuple[str, ...] = ()) -> None:
    inventory = inventory_tree(
        run,
        inventory_id="inventory-tiny-training",
        excluded_paths=("artifact-inventory.json", *excluded_paths),
    )
    write_inventory(inventory, run / "artifact-inventory.json", overwrite=True)


def _metric_observations(
    run: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_id = "syn-tiny"
    observations = []
    test_ids = config.get("test_camera_ids") or summary.get("held_out_camera_ids")
    if not isinstance(test_ids, list):
        raise ResultRecordError("resolved config has no test camera IDs")
    for camera_value in test_ids:
        camera_id = str(camera_value)
        reference_path = f"artifacts/reference/{camera_id}-rgb.ppm"
        prediction_path = f"artifacts/held-out/{camera_id}-rgb.ppm"
        observations.append(
            compute_per_view(
                scene_id,
                camera_id,
                read_ppm(run / reference_path),
                read_ppm(run / prediction_path),
                reference_artifact=reference_path,
                prediction_artifact=prediction_path,
            )
        )
    aggregate = aggregate_scene_psnr(observations, scene_id=scene_id)
    payload = {
        "observation_version": RESULT_VERSION,
        "scene_id": scene_id,
        "views": [item.to_dict() for item in observations],
        "scene_psnr": aggregate.to_dict(),
    }
    return [item.to_dict() for item in observations], payload


def _failed_metric(status: MetricStatus) -> dict[str, Any]:
    return MetricRecord(
        "psnr_rgb_scene_mean",
        METRIC_VERSION,
        None,
        "dB",
        "higher_is_better",
        "scene",
        status,
        0,
        "arithmetic_mean_across_valid_held_out_views",
        "full_image",
        (),
        ("held-out fidelity unavailable because the run did not complete",),
    ).to_dict()


def _resource_sections(
    resource: Mapping[str, Any] | None,
    render: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if resource is None:
        resources = {
            "resource_profile": "smoke",
            "compliance": "not_measured",
            "cpu_peak_rss_bytes": None,
            "gpu_resident_bytes": None,
            "gpu_peak_allocated_bytes": None,
            "gpu_peak_reserved_bytes": None,
            "process_gpu_peak_bytes": None,
            "workspace_peak_bytes": None,
        }
    else:
        cpu = resource.get("cpu_peak") or {}
        gpu = resource.get("gpu_final") or {}
        nvidia = resource.get("nvidia") or {}
        compliance = resource.get("compliance") or {}
        resources = {
            "resource_profile": resource.get("resource_profile", "smoke"),
            "compliance": compliance.get("state", "not_measured"),
            "cpu_peak_rss_bytes": cpu.get("peak_process_rss_bytes"),
            "gpu_resident_bytes": gpu.get("allocated_bytes"),
            "gpu_peak_allocated_bytes": gpu.get("peak_allocated_bytes"),
            "gpu_peak_reserved_bytes": gpu.get("peak_reserved_bytes"),
            "process_gpu_peak_bytes": nvidia.get("process_gpu_bytes"),
            "workspace_peak_bytes": None,
        }
    if render is None:
        runtime = {
            "status": "not_measured",
            "resolution": [1, 1],
            "output_buffers": [],
            "warmup_count": 0,
            "repetition_count": 0,
            "frame_latency_ms_median": None,
            "frame_latency_ms_p95": None,
            "fps_median": None,
            "transform_expansion_ms": None,
            "edit_update_ms": None,
            "edit_to_first_frame_ms": None,
        }
    else:
        render_resource = render.get("_resource") or {}
        timing = render_resource.get("timing") or {}
        median_seconds = timing.get("median_seconds")
        p95_seconds = timing.get("p95_seconds")
        runtime = {
            "status": "valid",
            "resolution": [render["width"], render["height"]],
            "output_buffers": ["rgb", "alpha", "depth"],
            "warmup_count": timing.get("warmup_count", 0),
            "repetition_count": timing.get("repetition_count", 1),
            "frame_latency_ms_median": (
                median_seconds * 1000 if median_seconds is not None else None
            ),
            "frame_latency_ms_p95": (
                p95_seconds * 1000 if p95_seconds is not None else None
            ),
            "fps_median": (
                1.0 / cast(float, median_seconds)
                if median_seconds not in (None, 0)
                else None
            ),
            "transform_expansion_ms": None,
            "edit_update_ms": None,
            "edit_to_first_frame_ms": None,
        }
    return resources, runtime


def validate_result_semantics(record: Mapping[str, Any]) -> None:
    status = record.get("status")
    failure = record.get("failure")
    if status == "failed" and failure is None:
        raise ResultRecordError("failed result requires failure details")
    if status != "failed" and failure is not None:
        raise ResultRecordError("non-failed result cannot carry failure details")
    counts = record.get("gaussian_counts")
    if isinstance(counts, dict):
        stored = counts.get("stored_total")
        source = counts.get("source")
        submitted = counts.get("render_submitted")
        if stored is not None and source is not None and stored != source:
            raise ResultRecordError("stored_total must equal final checkpoint count")
        if stored is not None and submitted is not None and stored != submitted:
            raise ResultRecordError("render_submitted must equal stored_total")
    artifacts = record.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path", "")
            if isinstance(path, str) and (
                path.startswith("/") or "\\" in path or ".." in Path(path).parts
            ):
                raise ResultRecordError("unsafe artifact path")


def build_smoke_result(
    run_root: Path,
    output: Path,
    *,
    overwrite: bool = False,
    inventory_excluded_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build one schema-valid, explicitly unfrozen smoke result."""

    run = _locate_run(run_root)
    state = _read_json(run / "status.json")
    config = _read_json(run / "resolved_config.json")
    run_status = state.get("status")
    completed = run_status == "completed"
    failed = run_status == "failed"
    result_status = "completed" if completed else "failed" if failed else "partial"
    summary = _read_json(run / "metadata/training-summary.json") if completed else {}
    checkpoint = (
        _read_json(run / "artifacts/gaussians.checkpoint.json") if completed else {}
    )
    resource = (
        _read_json(run / "metadata/resource-record.json")
        if (run / "metadata/resource-record.json").is_file()
        else None
    )
    render = (
        _read_json(run / "artifacts/held-out/render-metadata.json")
        if completed
        else None
    )
    if render is not None:
        render["_resource"] = _read_json(
            run / "artifacts/held-out/resource-record.json"
        )
    observations: list[dict[str, Any]] = []
    if completed:
        observations, observation_payload = _metric_observations(run, config, summary)
        _atomic_write(
            run / "metadata/metric-observations.json",
            observation_payload,
            overwrite=True,
        )
        scene_metric = observation_payload["scene_psnr"]["principal"]
    else:
        scene_metric = _failed_metric(
            MetricStatus.RUN_FAILED if failed else MetricStatus.NOT_MEASURED
        )
    _refresh_inventory(run, excluded_paths=inventory_excluded_paths)
    artifacts = [
        _artifact(run, "resolved-config", "resolved_config", "resolved_config.json"),
        _artifact(run, "environment", "environment", "metadata/environment.json"),
        _artifact(run, "artifact-inventory", "other", "artifact-inventory.json"),
    ]
    if completed:
        artifacts.extend(
            [
                _artifact(
                    run,
                    "checkpoint",
                    "checkpoint",
                    "artifacts/gaussians.checkpoint.json",
                ),
                _artifact(
                    run,
                    "held-out-render",
                    "render",
                    "artifacts/held-out/render-metadata.json",
                ),
                _artifact(
                    run,
                    "held-out-reference",
                    "render",
                    "artifacts/reference/render-metadata.json",
                ),
                _artifact(
                    run,
                    "metric-observations",
                    "metric_observations",
                    "metadata/metric-observations.json",
                ),
                _artifact(
                    run,
                    "training-resource",
                    "resource_record",
                    "metadata/resource-record.json",
                ),
            ]
        )
    failure_value = None
    if failed:
        failure_ref = state.get("failure_ref")
        if not isinstance(failure_ref, str):
            raise ResultRecordError("failed run has no failure reference")
        failure_record = _read_json(run / failure_ref)
        artifacts.append(
            _artifact(run, "failure-report", "failure_report", failure_ref)
        )
        allowed_categories = {
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
        category = failure_record.get("category", "unknown")
        failure_value = {
            "category": category if category in allowed_categories else "unknown",
            "stage": failure_record.get("stage", "training"),
            "message": failure_record.get("message", "run failed"),
            "failure_report_ref": failure_ref,
            "included_in_denominator": True,
        }
    resources, runtime = _resource_sections(resource, render)
    final_count = checkpoint.get("gaussian_count") if completed else None
    submitted = render.get("submitted_gaussian_count") if render else None
    created_at = state.get("created_at") or datetime.now(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    training_resource = summary.get("training_resource") or {}
    execution = config.get("execution") or {}
    seeds = execution.get("seeds") if isinstance(execution, dict) else None
    seed = seeds[0] if isinstance(seeds, list) and seeds else config.get("seed")
    if not isinstance(seed, int):
        raise ResultRecordError("resolved configuration has no valid seed")
    record: dict[str, Any] = {
        "schema_version": RESULT_VERSION,
        "definitions_version": DEFINITIONS_VERSION,
        "result_id": f"result-{state['run_id'][4:]}",
        "run_id": state["run_id"],
        "experiment_id": state["experiment_id"],
        "scene_id": "syn-tiny",
        "method": {
            "method_id": "tiny-gs-qualification",
            "method_kind": "full_gs",
            "version": "1",
            "grammar_source": "none",
            "canonical_selection": "none",
            "membership_policy": "none",
            "clipping_policy": "none",
            "appearance_policy": "unchanged",
            "residual_policy": "none",
            "instancing_implementation": "none",
        },
        "operating_point_id": "tiny-smoke",
        "seed": seed,
        "status": result_status,
        "provenance": {
            "code_commit": summary.get(
                "code_commit", state.get("code_commit", "unknown")
            ),
            "working_tree_dirty": bool(state.get("working_tree_dirty", False)),
            "configuration_digest": state.get(
                "configuration_digest", content_digest(config)
            ),
            "environment_ref": "metadata/environment.json",
            "artifact_inventory_ref": "artifact-inventory.json",
            "created_at": created_at,
        },
        "dataset": {
            "dataset_id": "tiny-project-fixture",
            "scene_manifest_ref": "metadata/target-generation.json",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
        },
        "artifacts": artifacts,
        "gaussian_counts": {
            "source": final_count,
            "stored_unique": final_count,
            "stored_canonical": None,
            "stored_residual": None,
            "stored_total": final_count,
            "active_instances": None,
            "instantiated": final_count,
            "render_submitted": submitted,
            "render_visible": None,
            "excluded": None,
        },
        "metrics": [scene_metric],
        "accounting": {
            "status": "not_measured",
            "accounting_record_ref": None,
            "semantic_raw_bytes": None,
            "serialized_uncompressed_bytes": None,
            "serialized_archive_bytes": None,
            "full_gs_reference_bytes": None,
            "compression_ratio": None,
            "components": [],
        },
        "resources": resources,
        "runtime": runtime,
        "qualitative_candidates": [],
        "warnings": [],
        "limitations": [
            "tiny qualification GS run; not the final full GS benchmark",
            "complete deployable representation accounting is deferred",
            (
                "training time is artifact-linked "
                f"({training_resource.get('elapsed_seconds')!r} seconds)"
                if completed
                else "training timing unavailable because the run did not complete"
            ),
        ],
        "failure": failure_value,
        "exclusion": None,
        "supersession": None,
        "freeze": {"status": "unfrozen", "frozen_at": None, "record_digest": None},
    }
    validate_result(record)
    validate_result_semantics(record)
    _atomic_write(output, record, overwrite=overwrite)
    return {
        "result": record,
        "result_digest": content_digest(record),
        "observations": observations,
        "output": output.as_posix(),
        "run_directory": run.as_posix(),
    }


def load_result(path: Path) -> dict[str, Any]:
    record = _read_json(path)
    validate_result(record)
    validate_result_semantics(record)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-smoke", help="build a qualification smoke result")
    build.add_argument("--run-root", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--json", action="store_true")
    validate = subparsers.add_parser("validate", help="validate a result record")
    validate.add_argument("path", type=Path)
    validate.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        if options.command == "build-smoke":
            outcome = build_smoke_result(
                options.run_root, options.output, overwrite=options.overwrite
            )
            payload = {
                "valid": True,
                "status": outcome["result"]["status"],
                "result_id": outcome["result"]["result_id"],
                "result_digest": outcome["result_digest"],
                "output": outcome["output"],
            }
        else:
            record = load_result(options.path)
            payload = {
                "valid": True,
                "status": record["status"],
                "result_id": record["result_id"],
                "source": options.path.as_posix(),
            }
    except (ResultRecordError, DocumentLoadError, SchemaValidationError) as error:
        payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, indent=2) if options.json else f"error: {error}")
        return 1
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if options.json
        else f"{payload['result_id']}: {payload['status']} (schema valid)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
