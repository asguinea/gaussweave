"""CPU-only image fidelity records and scene aggregation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

METRIC_VERSION = "1.0.0"
Number = int | float
Image = Sequence[Sequence[Sequence[Number]]]


class MetricError(ValueError):
    """Raised when a fidelity input violates the declared image policy."""


class MetricStatus(StrEnum):
    VALID = "valid"
    UNDEFINED = "undefined"
    NOT_APPLICABLE = "not_applicable"
    COMPUTATION_FAILED = "computation_failed"
    RUN_FAILED = "run_failed"
    EXCLUDED = "excluded"
    NOT_MEASURED = "not_measured"


@dataclass(frozen=True)
class ImageInputPolicy:
    """Explicit conversion policy; scientific inputs are never silently clipped."""

    data_range: float = 1.0
    integer_bit_depth: int | None = None
    clip_out_of_range: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.data_range) or self.data_range <= 0:
            raise MetricError("data_range must be finite and positive")
        if self.integer_bit_depth is not None and not 1 <= self.integer_bit_depth <= 32:
            raise MetricError("integer_bit_depth must be in [1, 32]")


DEFAULT_IMAGE_POLICY = ImageInputPolicy()


@dataclass(frozen=True)
class MetricRecord:
    metric_id: str
    metric_version: str
    value: float | int | bool | str | None
    unit: str
    orientation: str
    level: str
    status: MetricStatus
    sample_count: int
    aggregation: str
    mask_or_region: str | None
    source_artifacts: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["source_artifacts"] = list(self.source_artifacts)
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class PerViewFidelity:
    scene_id: str
    camera_id: str
    reference_artifact: str
    prediction_artifact: str
    psnr: MetricRecord
    mse: MetricRecord
    maximum_absolute_error: MetricRecord
    image_shape: tuple[int, int, int]
    computed_at: str
    metric_implementation_version: str = METRIC_VERSION
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "camera_id": self.camera_id,
            "reference_artifact": self.reference_artifact,
            "prediction_artifact": self.prediction_artifact,
            "psnr": self.psnr.to_dict(),
            "mse": self.mse.to_dict(),
            "maximum_absolute_error": self.maximum_absolute_error.to_dict(),
            "image_shape": list(self.image_shape),
            "computed_at": self.computed_at,
            "metric_implementation_version": self.metric_implementation_version,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SceneAggregate:
    principal: MetricRecord
    mean: float | None
    median: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None
    valid_view_count: int
    attempted_view_count: int
    failed_or_undefined_view_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["principal"] = self.principal.to_dict()
        return value


def _normalize(
    image: Image, policy: ImageInputPolicy
) -> tuple[list[float], tuple[int, int, int], tuple[str, ...]]:
    if not image:
        raise MetricError("image must be nonempty")
    height = len(image)
    width = len(image[0])
    if width == 0:
        raise MetricError("image must be nonempty")
    output: list[float] = []
    kinds: set[type[Any]] = set()
    for row in image:
        if len(row) != width:
            raise MetricError("image rows must have identical width")
        for pixel in row:
            if len(pixel) != 3:
                raise MetricError("image must use H x W x 3 RGB layout")
            for item in pixel:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise MetricError("image values must be integer or floating point")
                kinds.add(type(item))
                value = float(item)
                if not math.isfinite(value):
                    raise MetricError("image values must be finite")
                output.append(value)
    integer = all(kind is int for kind in kinds)
    warnings: list[str] = []
    if integer:
        if policy.integer_bit_depth is None:
            raise MetricError("integer images require explicit integer_bit_depth")
        maximum = float((1 << policy.integer_bit_depth) - 1)
        if any(value < 0 or value > maximum for value in output):
            raise MetricError("integer image exceeds declared bit depth")
        output = [value / maximum for value in output]
        warnings.append(
            "integer input converted from explicit "
            f"{policy.integer_bit_depth}-bit range"
        )
    elif any(kind is int for kind in kinds):
        raise MetricError("mixed integer and floating dtypes are unsupported")
    outside = any(value < 0 or value > policy.data_range for value in output)
    if outside and not policy.clip_out_of_range:
        raise MetricError("floating image exceeds declared data range")
    if outside:
        output = [min(policy.data_range, max(0.0, value)) for value in output]
        warnings.append("out-of-range input explicitly clipped")
    warnings.append(f"data_range={policy.data_range:g}; all RGB scalar values included")
    return output, (height, width, 3), tuple(warnings)


def image_diagnostics(
    reference: Image,
    prediction: Image,
    *,
    policy: ImageInputPolicy = DEFAULT_IMAGE_POLICY,
    source_artifacts: tuple[str, ...] = (),
    level: str = "observation",
) -> tuple[MetricRecord, MetricRecord, MetricRecord, tuple[int, int, int]]:
    reference_values, shape, reference_warnings = _normalize(reference, policy)
    prediction_values, prediction_shape, prediction_warnings = _normalize(
        prediction, policy
    )
    if shape != prediction_shape:
        raise MetricError("reference and prediction shapes must match")
    squared = [
        (reference_value - prediction_value) ** 2
        for reference_value, prediction_value in zip(
            reference_values, prediction_values, strict=True
        )
    ]
    absolute = [
        abs(reference_value - prediction_value)
        for reference_value, prediction_value in zip(
            reference_values, prediction_values, strict=True
        )
    ]
    mse_value = sum(squared) / len(squared)
    maximum_error = max(absolute)
    psnr_raw = (
        math.inf
        if mse_value == 0
        else 10.0 * math.log10(policy.data_range**2 / mse_value)
    )
    warnings = tuple(dict.fromkeys((*reference_warnings, *prediction_warnings)))
    psnr_warnings = warnings
    psnr_value: float | str
    if math.isinf(psnr_raw):
        psnr_value = "positive_infinity"
        psnr_warnings = (
            *warnings,
            "perfect agreement serialized as positive_infinity "
            "because JSON forbids infinity",
        )
    else:
        psnr_value = psnr_raw
    return (
        MetricRecord(
            "psnr_rgb",
            METRIC_VERSION,
            psnr_value,
            "dB",
            "higher_is_better",
            level,
            MetricStatus.VALID,
            len(reference_values),
            "all_rgb_scalar_values",
            "full_image",
            source_artifacts,
            psnr_warnings,
        ),
        MetricRecord(
            "mse_rgb",
            METRIC_VERSION,
            mse_value,
            "normalized_squared_intensity",
            "lower_is_better",
            level,
            MetricStatus.VALID,
            len(reference_values),
            "all_rgb_scalar_values",
            "full_image",
            source_artifacts,
            warnings,
        ),
        MetricRecord(
            "maximum_absolute_error_rgb",
            METRIC_VERSION,
            maximum_error,
            "normalized_intensity",
            "lower_is_better",
            level,
            MetricStatus.VALID,
            len(reference_values),
            "all_rgb_scalar_values",
            "full_image",
            source_artifacts,
            warnings,
        ),
        shape,
    )


def compute_per_view(
    scene_id: str,
    camera_id: str,
    reference: Image,
    prediction: Image,
    *,
    reference_artifact: str,
    prediction_artifact: str,
    policy: ImageInputPolicy = DEFAULT_IMAGE_POLICY,
    computed_at: str | None = None,
) -> PerViewFidelity:
    sources = (reference_artifact, prediction_artifact)
    psnr, mse, maximum_error, shape = image_diagnostics(
        reference, prediction, policy=policy, source_artifacts=sources
    )
    return PerViewFidelity(
        scene_id,
        camera_id,
        reference_artifact,
        prediction_artifact,
        psnr,
        mse,
        maximum_error,
        shape,
        computed_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        warnings=psnr.warnings,
    )


def aggregate_scene_psnr(
    views: Sequence[PerViewFidelity], *, scene_id: str
) -> SceneAggregate:
    ordered = sorted(views, key=lambda item: item.camera_id)
    values = [
        float(item.psnr.value)
        for item in ordered
        if item.psnr.status is MetricStatus.VALID
        and isinstance(item.psnr.value, (int, float))
    ]
    attempted = len(ordered)
    failed = attempted - len(values)
    sources = tuple(
        sorted({source for item in ordered for source in item.psnr.source_artifacts})
    )
    if not values:
        status = MetricStatus.UNDEFINED if attempted else MetricStatus.NOT_MEASURED
        principal = MetricRecord(
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
            sources,
            ("no finite valid held-out view PSNR values",),
        )
        return SceneAggregate(
            principal, None, None, None, None, None, 0, attempted, failed
        )
    mean = statistics.fmean(values)
    median = statistics.median(values)
    standard_deviation = statistics.pstdev(values)
    principal = MetricRecord(
        "psnr_rgb_scene_mean",
        METRIC_VERSION,
        mean,
        "dB",
        "higher_is_better",
        "scene",
        MetricStatus.VALID,
        len(values),
        "arithmetic_mean_across_valid_held_out_views",
        "full_image",
        sources,
        (() if not failed else (f"{failed} attempted views were non-valid",)),
    )
    return SceneAggregate(
        principal,
        mean,
        median,
        standard_deviation,
        min(values),
        max(values),
        len(values),
        attempted,
        failed,
    )
