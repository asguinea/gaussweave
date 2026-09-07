from __future__ import annotations

import json
import math

import pytest

from gaussweave.metrics import (
    ImageInputPolicy,
    MetricError,
    MetricRecord,
    MetricStatus,
    PerViewFidelity,
    aggregate_scene_psnr,
    compute_per_view,
    image_diagnostics,
)


def _image(value: float) -> list[list[list[float]]]:
    return [[[value, value, value]]]


def test_known_psnr_mse_and_maximum_error() -> None:
    psnr, mse, maximum, shape = image_diagnostics(_image(0.0), _image(0.5))
    assert mse.value == pytest.approx(0.25)
    assert psnr.value == pytest.approx(10 * math.log10(4))
    assert maximum.value == pytest.approx(0.5)
    assert shape == (1, 1, 3)
    assert mse.orientation == "lower_is_better"
    assert maximum.unit == "normalized_intensity"


def test_perfect_agreement_uses_valid_json_string() -> None:
    psnr, _, _, _ = image_diagnostics(_image(0.25), _image(0.25))
    assert psnr.status is MetricStatus.VALID
    assert psnr.value == "positive_infinity"
    assert "Infinity" not in json.dumps(psnr.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("reference", "prediction", "policy"),
    [
        (_image(0.0), [[[0.0, 0.0]]], ImageInputPolicy()),
        ([], [], ImageInputPolicy()),
        (_image(float("nan")), _image(0.0), ImageInputPolicy()),
        (_image(1.1), _image(0.0), ImageInputPolicy()),
        ([[[1, 1, 1]]], [[[1, 1, 1]]], ImageInputPolicy()),
        ([[[1, 1.0, 1]]], [[[1, 1.0, 1]]], ImageInputPolicy()),
    ],
)
def test_invalid_inputs_are_rejected(
    reference: object, prediction: object, policy: ImageInputPolicy
) -> None:
    with pytest.raises(MetricError):
        image_diagnostics(reference, prediction, policy=policy)  # type: ignore[arg-type]


def test_explicit_integer_conversion_and_clipping_policy() -> None:
    psnr, mse, _, _ = image_diagnostics(
        [[[0, 127, 255]]],
        [[[0, 127, 255]]],
        policy=ImageInputPolicy(integer_bit_depth=8),
    )
    assert psnr.value == "positive_infinity" and mse.value == 0
    clipped, _, _, _ = image_diagnostics(
        _image(2.0),
        _image(1.0),
        policy=ImageInputPolicy(clip_out_of_range=True),
    )
    assert clipped.value == "positive_infinity"


def _view(camera: str, value: float | None) -> PerViewFidelity:
    status = (
        MetricStatus.VALID if value is not None else MetricStatus.COMPUTATION_FAILED
    )
    metric = MetricRecord(
        "psnr_rgb",
        "1.0.0",
        value,
        "dB",
        "higher_is_better",
        "observation",
        status,
        3 if value is not None else 0,
        "all_rgb_scalar_values",
        "full_image",
        (f"ref-{camera}", f"pred-{camera}"),
    )
    diagnostic = MetricRecord(
        "mse_rgb",
        "1.0.0",
        0.0 if value is not None else None,
        "normalized_squared_intensity",
        "lower_is_better",
        "observation",
        status,
        3 if value is not None else 0,
        "all_rgb_scalar_values",
        "full_image",
        (),
    )
    return PerViewFidelity(
        "syn-tiny",
        camera,
        f"ref-{camera}",
        f"pred-{camera}",
        metric,
        diagnostic,
        diagnostic,
        (1, 1, 3),
        "2026-01-01T00:00:00Z",
    )


def test_scene_aggregation_is_order_independent_and_retains_failures() -> None:
    views = [_view("b", 20.0), _view("a", 10.0), _view("c", None)]
    first = aggregate_scene_psnr(views, scene_id="syn-tiny")
    second = aggregate_scene_psnr(list(reversed(views)), scene_id="syn-tiny")
    assert first == second
    assert first.mean == 15
    assert first.median == 15
    assert first.standard_deviation == 5
    assert first.minimum == 10 and first.maximum == 20
    assert first.valid_view_count == 2
    assert first.attempted_view_count == 3
    assert first.failed_or_undefined_view_count == 1
    assert first.principal.level == "scene"


def test_no_valid_scene_view_has_explicit_status() -> None:
    aggregate = aggregate_scene_psnr([_view("a", None)], scene_id="syn-tiny")
    assert aggregate.principal.status is MetricStatus.UNDEFINED
    assert aggregate.principal.value is None
    assert aggregate.valid_view_count == 0


def test_per_view_serialization_has_provenance() -> None:
    record = compute_per_view(
        "syn-tiny",
        "camera-a",
        _image(0.0),
        _image(0.5),
        reference_artifact="reference.ppm",
        prediction_artifact="prediction.ppm",
        computed_at="2026-01-01T00:00:00Z",
    )
    encoded = json.dumps(record.to_dict(), sort_keys=True, allow_nan=False)
    assert json.loads(encoded)["camera_id"] == "camera-a"
    assert record.psnr.source_artifacts == ("reference.ppm", "prediction.ppm")
