"""CPU-only project-owned fidelity metrics."""

from .fidelity import (
    METRIC_VERSION,
    ImageInputPolicy,
    MetricError,
    MetricRecord,
    MetricStatus,
    PerViewFidelity,
    SceneAggregate,
    aggregate_scene_psnr,
    compute_per_view,
    image_diagnostics,
)

__all__ = [
    "METRIC_VERSION",
    "ImageInputPolicy",
    "MetricError",
    "MetricRecord",
    "MetricStatus",
    "PerViewFidelity",
    "SceneAggregate",
    "aggregate_scene_psnr",
    "compute_per_view",
    "image_diagnostics",
]
