from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from gaussweave.data.real_truck import qualify_dataset, validate_dataset_root
from gaussweave.gaussians.real_explicit import (
    RealGaussianTensorSource,
    _camera_from_record,
)
from gaussweave.rendering.gsplat_renderer import GsplatRenderer
from gaussweave.rendering.models import RenderSettings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.wsl,
    pytest.mark.slow,
]


def _root() -> Path:
    value = os.environ.get("GAUSSWEAVE_TRUCK_ROOT")
    if not value:
        pytest.skip("GAUSSWEAVE_TRUCK_ROOT is not configured")
    return validate_dataset_root(Path(value))


def test_t7_t8_real_truck_camera_and_reprojection_qualification() -> None:
    qualification = qualify_dataset(_root() / "source")
    assert qualification.valid
    assert len(qualification.cameras) == 251
    assert qualification.scene_p95_px <= 4.0
    assert qualification.invalid_observations == 0


def test_t14_t15_native_model_renders_matching_primary_camera() -> None:
    root = _root()
    split = json.loads(
        (root / "annotations" / "evaluation-split.json").read_text(encoding="utf-8")
    )
    record = next(
        cast(dict[str, Any], item)
        for item in cast(list[object], split["cameras"])
        if cast(dict[str, Any], item)["image_name"] == "000251.jpg"
    )
    scene = RealGaussianTensorSource.load(root / "converted" / "graphdeco-30000")
    result = GsplatRenderer().render(
        scene,
        _camera_from_record(record),
        RenderSettings(
            output_buffers=("rgb", "alpha", "depth"),
            camera_batch_size=1,
            warmup_count=0,
            repetition_count=1,
        ),
        profile_name="standard",
    )
    assert tuple(result.rgb.shape) == (1, 546, 979, 3)
    assert result.stored_gaussian_count == 2_541_226
    assert result.renderer_version == "1.5.3"
    assert result.resource_record.compliance.state.value == "compliant"
