from __future__ import annotations

from pathlib import Path

import pytest

from gaussweave.experiments.tiny_training import TinyTrainingConfig, run_tiny_training

pytestmark = pytest.mark.gpu


def test_live_conservative_calibration_workload(tmp_path: Path) -> None:
    outcome = run_tiny_training(
        tmp_path / "calibration",
        TinyTrainingConfig(
            width=48,
            height=48,
            iterations=8,
            validation_interval=4,
            logging_interval=4,
            initial_gaussian_count=3,
            maximum_gaussian_count=3,
            camera_batch_size=1,
            resource_profile="standard",
        ),
    )
    assert outcome.final_loss < outcome.initial_loss
    assert outcome.resource_record["compliance"]["state"].startswith("compliant")
    assert outcome.scientific_checkpoint_digest.startswith("sha256:")
