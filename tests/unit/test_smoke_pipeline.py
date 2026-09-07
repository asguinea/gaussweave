from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaussweave.config.resolution import (
    canonical_json_bytes,
    generate_run_id,
    resolve_layers,
)
from gaussweave.config.validation import validate_experiment
from gaussweave.experiments.smoke_pipeline import (
    DEFAULT_CONFIG,
    PipelineValidationError,
    run_smoke_pipeline,
    training_config,
)


def _overlay(tmp_path: Path, name: str, value: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _identity(paths: tuple[Path, ...]) -> tuple[bytes, str, str, str]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    resolved = resolve_layers(paths, timestamp=timestamp)
    document = dict(resolved.document)
    scene = document["dataset"]["scene_selection"]["scene_ids"][0]  # type: ignore[index]
    seed = resolved.seeds[0]
    attempt = document["execution"]["attempt"]  # type: ignore[index]
    run_id = generate_run_id(
        experiment_id=resolved.experiment_id,
        scene_id=scene,  # type: ignore[arg-type]
        seed=seed,
        scientific_configuration_digest=resolved.scientific_digest,
        attempt=attempt,  # type: ignore[arg-type]
    )
    return (
        canonical_json_bytes(document),
        resolved.full_digest,
        resolved.scientific_digest,
        run_id,
    )


def test_committed_config_validates_and_resolves_deterministically() -> None:
    source = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    validate_experiment(source)
    first = _identity((DEFAULT_CONFIG,))
    second = _identity((DEFAULT_CONFIG,))
    assert first == second
    config = training_config(resolve_layers((DEFAULT_CONFIG,)))
    assert config.seed == 17 and config.iterations == 30
    assert config.maximum_gaussian_count == 3
    assert set(config.train_camera_ids).isdisjoint(config.test_camera_ids)


def test_operational_root_is_outside_scientific_identity(tmp_path: Path) -> None:
    baseline = resolve_layers((DEFAULT_CONFIG,))
    output_overlay = _overlay(
        tmp_path,
        "output.json",
        {"outputs": {"run_root": "runs/a-different-operational-root"}},
    )
    changed = resolve_layers((DEFAULT_CONFIG, output_overlay))
    assert changed.scientific_digest == baseline.scientific_digest
    assert changed.full_digest != baseline.full_digest
    assert (
        _identity((DEFAULT_CONFIG,))[-1]
        == _identity((DEFAULT_CONFIG, output_overlay))[-1]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"execution": {"seeds": [18]}},
        {
            "stopping": {
                "rules": [
                    {"kind": "iteration_count", "value": 31},
                    {"kind": "gaussian_cap", "value": 3},
                    {"kind": "memory_guard", "value": "smoke"},
                ]
            }
        },
        {"rendering": {"width": 65}},
        {"method": {"operating_point": {"operating_point_id": "other"}}},
        {"metrics": {"metric_ids": ["psnr_rgb"]}},
    ],
)
def test_scientific_changes_change_identity(tmp_path: Path, change: object) -> None:
    overlay = _overlay(tmp_path, "scientific.json", change)
    baseline = _identity((DEFAULT_CONFIG,))
    changed = _identity((DEFAULT_CONFIG, overlay))
    assert changed[2] != baseline[2]
    assert changed[3] != baseline[3]


def test_invalid_config_fails_before_gpu_or_run_initialization(tmp_path: Path) -> None:
    invalid = _overlay(tmp_path, "invalid.json", {"mode": "frozen"})
    root = tmp_path / "runs"
    with pytest.raises(PipelineValidationError):
        run_smoke_pipeline(root, (DEFAULT_CONFIG, invalid))
    assert not root.exists()


def test_pipeline_help_and_validation_imports_are_cpu_only() -> None:
    program = (
        "import sys; from gaussweave.cli import main; "
        "assert main(['experiment','smoke','--help']) == 0"
    )
    help_result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "torch" not in help_result.stdout
    validate_program = (
        "import sys, json; "
        "from gaussweave.config.validation import validate_experiment; "
        f"validate_experiment(json.load(open(r'{DEFAULT_CONFIG}'))); "
        "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules"
    )
    validated = subprocess.run(
        [sys.executable, "-c", validate_program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr
