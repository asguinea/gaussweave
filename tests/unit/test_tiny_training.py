from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.experiments.tiny_training import (
    TinyTrainingConfig,
    TrainingError,
    load_gaussian_checkpoint,
    save_gaussian_checkpoint,
    scientific_checkpoint_digest,
)
from gaussweave.rendering.fixtures import centered_gaussian


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 0},
        {"train_camera_ids": ()},
        {"test_camera_ids": ()},
        {"test_camera_ids": ("train-center",)},
        {"iterations": 0},
        {"learning_rates": (("means", 0.0),)},
        {"initial_gaussian_count": 4, "maximum_gaussian_count": 3},
        {"dtype": "float16"},
        {"appearance_mode": "sh"},
        {"checkpoint_policy": "periodic"},
    ],
)
def test_training_configuration_validation(changes: dict[str, object]) -> None:
    with pytest.raises(TrainingError):
        TinyTrainingConfig(**changes)  # type: ignore[arg-type]


def test_valid_config_splits_and_digest() -> None:
    config = TinyTrainingConfig()
    sets = [
        set(config.train_camera_ids),
        set(config.validation_camera_ids),
        set(config.test_camera_ids),
    ]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert config.digest.startswith("sha256:")


def test_checkpoint_roundtrip_and_malformed(tmp_path: Path) -> None:
    config = TinyTrainingConfig()
    scene = centered_gaussian()
    path = tmp_path / "checkpoint.json"
    digest = save_gaussian_checkpoint(
        path,
        scene,
        config=config,
        run_id="run-test",
        iteration=3,
        loss_summary={"initial": 1.0, "final": 0.5},
        source_metadata={"policy": "test"},
    )
    loaded, metadata = load_gaussian_checkpoint(path)
    assert loaded.means == scene.means
    assert loaded.quaternions == scene.quaternions
    assert loaded.scales == scene.scales
    assert metadata["artifact_checksum"] == digest
    assert metadata["scientific_checkpoint_digest"].startswith("sha256:")
    original = json.loads(path.read_text())
    cases = []
    missing = dict(original)
    missing.pop("dtype")
    cases.append(missing)
    wrong_version = dict(original)
    wrong_version["format_version"] = "99"
    cases.append(wrong_version)
    wrong_shape = dict(original)
    wrong_shape["means"] = [[0.0, 0.0]]
    cases.append(wrong_shape)
    bad_checksum = dict(original)
    bad_checksum["seed"] = 999
    cases.append(bad_checksum)
    for index, payload in enumerate(cases):
        malformed = tmp_path / f"bad-{index}.json"
        malformed.write_text(json.dumps(payload))
        with pytest.raises(TrainingError):
            load_gaussian_checkpoint(malformed)


def test_scientific_checkpoint_digest_excludes_run_and_validation_timing() -> None:
    base = {
        "format_version": "1.0",
        "training_configuration_digest": "sha256:config",
        "run_id": "run-a",
        "scene_id": "tiny-gs-v1",
        "iteration": 2,
        "gaussian_count": 1,
        "means": [[0.0, 0.0, 3.0]],
        "rotations": [[1.0, 0.0, 0.0, 0.0]],
        "scales": [[0.1, 0.1, 0.1]],
        "opacity_parameters": [0.9],
        "appearance_parameters": [[1.0, 0.0, 0.0]],
        "appearance_mode": "direct_rgb",
        "sh_degree": None,
        "coordinate_frame": "world",
        "dtype": "float32",
        "source_metadata": {"policy": "test"},
        "loss_summary": {
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "validation": [{"loss": 0.4, "elapsed_seconds": 0.1}],
        },
        "seed": 17,
    }
    changed = json.loads(json.dumps(base))
    changed["run_id"] = "run-b"
    changed["loss_summary"]["validation"][0]["elapsed_seconds"] = 9.9
    assert scientific_checkpoint_digest(base) == scientific_checkpoint_digest(changed)


def test_experiments_import_remains_lightweight() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gaussweave.experiments; "
            "import gaussweave.experiments.tiny_training; "
            "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
