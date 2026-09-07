from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.config.errors import SchemaValidationError
from gaussweave.config.validation import validate_result
from gaussweave.results.artifacts import load_inventory, verify_inventory
from gaussweave.results.records import (
    ResultRecordError,
    build_smoke_result,
    load_result,
    validate_result_semantics,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _ppm(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P6\n1 1\n255\n" + bytes([value, value, value]))


def _run_fixture(root: Path, *, status: str = "completed") -> Path:
    run = root / "run-fixture"
    run.mkdir(parents=True)
    state: dict[str, object] = {
        "run_id": "run-fixture",
        "experiment_id": "exp-tiny-training-v1",
        "status": status,
        "created_at": "2026-01-01T00:00:00Z",
        "working_tree_dirty": False,
        "code_commit": "fixture",
    }
    config = {
        "seed": 17,
        "test_camera_ids": ["test-up"],
    }
    _write_json(run / "status.json", state)
    _write_json(run / "resolved_config.json", config)
    _write_json(run / "metadata/environment.json", {"environment": "fixture"})
    _write_json(run / "metadata/target-generation.json", {"fixture_id": "tiny"})
    if status == "failed":
        state["failure_ref"] = "failures/failure.json"
        _write_json(run / "status.json", state)
        _write_json(
            run / "failures/failure.json",
            {
                "category": "gs_numerical_failure",
                "stage": "training",
                "message": "controlled",
            },
        )
        return run
    _write_json(
        run / "metadata/training-summary.json",
        {
            "code_commit": "fixture",
            "training_resource": {"elapsed_seconds": 0.5},
        },
    )
    _write_json(
        run / "metadata/resource-record.json",
        {
            "resource_profile": "smoke",
            "compliance": {"state": "compliant"},
            "cpu_peak": {"peak_process_rss_bytes": 100},
            "gpu_final": {
                "allocated_bytes": 10,
                "peak_allocated_bytes": 20,
                "peak_reserved_bytes": 30,
            },
            "nvidia": {"process_gpu_bytes": 40},
        },
    )
    _write_json(run / "artifacts/gaussians.checkpoint.json", {"gaussian_count": 3})
    render = {
        "width": 1,
        "height": 1,
        "submitted_gaussian_count": 3,
    }
    _write_json(run / "artifacts/held-out/render-metadata.json", render)
    _write_json(run / "artifacts/reference/render-metadata.json", render)
    _write_json(
        run / "artifacts/held-out/resource-record.json",
        {
            "timing": {
                "warmup_count": 0,
                "repetition_count": 1,
                "median_seconds": 0.01,
                "p95_seconds": 0.02,
            }
        },
    )
    _ppm(run / "artifacts/reference/test-up-rgb.ppm", 255)
    _ppm(run / "artifacts/held-out/test-up-rgb.ppm", 127)
    return run


def test_completed_result_schema_counts_placeholders_and_links(tmp_path: Path) -> None:
    run = _run_fixture(tmp_path)
    output = tmp_path / "result.json"
    outcome = build_smoke_result(run, output)
    record = outcome["result"]
    validate_result(record)
    assert load_result(output) == record
    assert record["status"] == "completed"
    assert record["gaussian_counts"]["source"] == 3
    assert record["gaussian_counts"]["stored_total"] == 3
    assert record["gaussian_counts"]["render_submitted"] == 3
    assert record["accounting"]["status"] == "not_measured"
    assert record["accounting"]["compression_ratio"] is None
    assert record["accounting"]["serialized_archive_bytes"] is None
    assert record["resources"]["cpu_peak_rss_bytes"] == 100
    assert record["resources"]["gpu_peak_allocated_bytes"] == 20
    assert record["runtime"]["frame_latency_ms_median"] == 10
    assert record["runtime"]["fps_median"] == 100
    assert record["freeze"]["status"] == "unfrozen"
    assert record["metrics"][0]["level"] == "scene"
    inventory = load_inventory(run / "artifact-inventory.json")
    assert verify_inventory(inventory, run, strict=True).state == "valid"


def test_failed_and_partial_result_states(tmp_path: Path) -> None:
    failed = _run_fixture(tmp_path / "failed", status="failed")
    failed_record = build_smoke_result(failed, tmp_path / "failed-result.json")[
        "result"
    ]
    assert failed_record["status"] == "failed"
    assert failed_record["failure"]["category"] == "gs_numerical_failure"
    assert failed_record["metrics"][0]["status"] == "run_failed"
    partial = _run_fixture(tmp_path / "partial", status="running")
    partial_record = build_smoke_result(partial, tmp_path / "partial-result.json")[
        "result"
    ]
    assert partial_record["status"] == "partial"
    assert partial_record["metrics"][0]["status"] == "not_measured"
    assert partial_record["failure"] is None


def test_semantic_and_schema_validation_failures(tmp_path: Path) -> None:
    record = build_smoke_result(_run_fixture(tmp_path), tmp_path / "result.json")[
        "result"
    ]
    missing = dict(record)
    missing.pop("method")
    with pytest.raises(SchemaValidationError):
        validate_result(missing)
    inconsistent = dict(record)
    inconsistent["status"] = "failed"
    with pytest.raises(ResultRecordError):
        validate_result_semantics(inconsistent)
    invalid_unit = json.loads(json.dumps(record))
    invalid_unit["metrics"][0]["unit"] = ""
    with pytest.raises(SchemaValidationError):
        validate_result(invalid_unit)
    unsafe = json.loads(json.dumps(record))
    unsafe["artifacts"][0]["path"] = "../escape"
    with pytest.raises((SchemaValidationError, ResultRecordError)):
        validate_result(unsafe)
        validate_result_semantics(unsafe)


def test_result_cli_build_validate_json_human_and_overwrite(tmp_path: Path) -> None:
    run = _run_fixture(tmp_path)
    output = tmp_path / "result.json"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.results.records",
            "build-smoke",
            "--run-root",
            str(run),
            "--output",
            str(output),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert json.loads(build.stdout)["valid"] is True
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.results.records",
            "build-smoke",
            "--run-root",
            str(run),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 1 and "overwrite" in refused.stdout
    validated = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaussweave.results.records",
            "validate",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validated.returncode == 0
    assert "schema valid" in validated.stdout


def test_metrics_and_results_import_without_gpu_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gaussweave.metrics; import gaussweave.results; "
            "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
