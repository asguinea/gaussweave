"""Deterministic opacity-aware oracle similarity registration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from gaussweave.real_structuring.models import (
    CANONICAL_INSTANCE_ID,
    INSTANCE_IDS,
    PanelFrame,
    SimilarityRegistration,
)


def _matrix(
    frame: PanelFrame, scale: float, correction_local: Sequence[float]
) -> tuple[float, ...]:
    axes = frame.axes_world
    translated = tuple(
        frame.origin_world[row]
        + sum(axes[column][row] * correction_local[column] for column in range(3))
        for row in range(3)
    )
    return (
        scale * axes[0][0],
        scale * axes[1][0],
        scale * axes[2][0],
        translated[0],
        scale * axes[0][1],
        scale * axes[1][1],
        scale * axes[2][1],
        translated[1],
        scale * axes[0][2],
        scale * axes[1][2],
        scale * axes[2][2],
        translated[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _sample(points: Any, weights: Any, maximum: int = 768) -> tuple[Any, Any]:
    torch = _torch()
    if points.shape[0] <= maximum:
        return points, weights
    indexes = (
        torch.linspace(0, points.shape[0] - 1, steps=maximum, dtype=torch.float64)
        .round()
        .to(torch.int64)
    )
    return points[indexes], weights[indexes]


def _robust_objective(source: Any, target: Any, weights: Any) -> tuple[float, Any]:
    torch = _torch()
    distances = torch.cdist(source, target).amin(dim=1)
    cap = torch.quantile(distances, 0.9)
    clipped = torch.minimum(distances, cap)
    objective = (clipped * weights).sum() / weights.sum().clamp_min(1e-12)
    return float(objective.item()), distances


def register_instances(
    frames: tuple[PanelFrame, ...],
    local_points: Mapping[str, Any],
    opacities: Mapping[str, Any],
) -> tuple[SimilarityRegistration, ...]:
    """Fit bounded similarities using only core geometry and annotation frames."""

    torch = _torch()
    by_id = {frame.instance_id: frame for frame in frames}
    canonical_raw = local_points[CANONICAL_INSTANCE_ID].to(torch.float64)
    canonical_weights_raw = opacities[CANONICAL_INSTANCE_ID].to(torch.float64)
    canonical, canonical_weights = _sample(
        canonical_raw, canonical_weights_raw.clamp_min(1e-4)
    )
    results: list[SimilarityRegistration] = []
    for instance_id in INSTANCE_IDS:
        frame = by_id[instance_id]
        target_raw = local_points[instance_id].to(torch.float64)
        target_weights_raw = opacities[instance_id].to(torch.float64)
        target, _target_weights = _sample(
            target_raw, target_weights_raw.clamp_min(1e-4)
        )
        zero = (0.0, 0.0, 0.0)
        correction: tuple[float, ...]
        initialization = _matrix(frame, 1.0, zero)
        before, before_distances = _robust_objective(
            canonical, target, canonical_weights
        )
        if instance_id == CANONICAL_INSTANCE_ID:
            scale = 1.0
            correction = zero
            after = 0.0
            distances = torch.zeros_like(before_distances)
            candidate_diagnostics: dict[str, Any] = {
                "candidate_refinement_accepted": True,
                "candidate_correction_translation_m": 0.0,
                "candidate_objective": 0.0,
                "correction_limit_m": 0.05,
            }
        else:
            canonical_span = torch.quantile(canonical, 0.9, dim=0) - torch.quantile(
                canonical, 0.1, dim=0
            )
            target_span = torch.quantile(target, 0.9, dim=0) - torch.quantile(
                target, 0.1, dim=0
            )
            valid = canonical_span > 1e-4
            estimated = float(
                torch.median(target_span[valid] / canonical_span[valid]).item()
            )
            estimated = min(1.70, max(0.55, estimated))
            # Explicit construction avoids nondeterministic optimizer behavior.
            candidate_scales = sorted(
                {
                    1.0,
                    estimated,
                    min(1.70, max(0.55, estimated * 0.9)),
                    min(1.70, max(0.55, estimated * 0.95)),
                    min(1.70, max(0.55, estimated * 1.05)),
                    min(1.70, max(0.55, estimated * 1.1)),
                }
            )
            target_center = torch.median(target, dim=0).values
            canonical_center = torch.median(canonical, dim=0).values
            best: tuple[float, float, Any, Any] | None = None
            for candidate in candidate_scales:
                correction_tensor = target_center - candidate * canonical_center
                transformed = candidate * canonical + correction_tensor
                objective, candidate_distances = _robust_objective(
                    transformed, target, canonical_weights
                )
                if best is None or objective < best[0]:
                    best = (
                        objective,
                        candidate,
                        correction_tensor,
                        candidate_distances,
                    )
            assert best is not None
            after, scale, correction_tensor, distances = best
            correction = tuple(float(value) for value in correction_tensor.tolist())
            candidate_magnitude = math.sqrt(sum(value * value for value in correction))
            candidate_diagnostics = {
                "candidate_refinement_accepted": candidate_magnitude <= 0.05,
                "candidate_correction_translation_m": candidate_magnitude,
                "candidate_objective": after,
                "correction_limit_m": 0.05,
            }
            if candidate_magnitude > 0.05:
                # A large centroid shift is evidence of incompatible instance shape,
                # not a defensible rigid refinement. Keep the annotation center.
                after = before
                scale = 1.0
                correction = zero
                distances = before_distances
        correction_magnitude = math.sqrt(sum(value * value for value in correction))
        matrix = _matrix(frame, scale, correction)
        quantiles = torch.quantile(
            distances, torch.tensor((0.5, 0.9, 0.95), dtype=torch.float64)
        )
        inside = (
            (scale * canonical + torch.tensor(correction, dtype=torch.float64))
            >= torch.tensor(frame.local_lower, dtype=torch.float64)
        ) & (
            (scale * canonical + torch.tensor(correction, dtype=torch.float64))
            <= torch.tensor(frame.local_upper, dtype=torch.float64)
        )
        result = SimilarityRegistration(
            instance_id=instance_id,
            matrix=matrix,
            scale=scale,
            initialization_matrix=initialization,
            objective_before=max(before, after),
            objective_after=after,
            correction_translation_m=correction_magnitude,
            rotation_angle_degrees=0.0,
            diagnostics={
                "source_core_gaussian_count": int(target_raw.shape[0]),
                "canonical_gaussian_count": int(canonical_raw.shape[0]),
                "nearest_neighbor_median_m": float(quantiles[0].item()),
                "nearest_neighbor_p90_m": float(quantiles[1].item()),
                "nearest_neighbor_p95_m": float(quantiles[2].item()),
                "outlier_fraction_above_p90": float(
                    (distances > quantiles[1]).to(torch.float64).mean().item()
                ),
                "canonical_points_inside_target_bounds_fraction": float(
                    inside.all(dim=1).to(torch.float64).mean().item()
                ),
                "local_scale_compatibility": (
                    "bounded_similarity_required"
                    if abs(scale - 1.0) > 0.05
                    else "rigid_compatible"
                ),
                "visual_plausibility_status": "requires_render_review",
                "objective": "opacity_weighted_trimmed_point_to_nearest_core",
                "refinement_uses_images": False,
                "evaluation_views_used": [],
                **candidate_diagnostics,
            },
        )
        results.append(result)
    return tuple(results)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        from gaussweave.real_structuring.models import StructuringError

        raise StructuringError(
            "PyTorch is required for geometry registration"
        ) from error
    return torch
