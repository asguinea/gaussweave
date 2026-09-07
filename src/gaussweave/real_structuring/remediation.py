"""Bounded Truck terminal remediation.

The module preserves the accepted whole-panel v1 result and provides a
versioned, local-only audit and extraction path for smaller oracle terminals.
Heavy tensor dependencies remain lazy so ordinary package imports stay CPU-safe.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import content_digest
from gaussweave.gaussians.real_explicit import RealGaussianTensorSource
from gaussweave.real_structuring.extraction import (
    _indices,
    _matrix_to_quaternion,
    _quat_multiply,
    _raw,
)
from gaussweave.real_structuring.models import (
    INSTANCE_IDS,
    SOURCE_COUNT,
    OwnershipStatus,
    PanelFrame,
    SimilarityRegistration,
    StructuringError,
    atomic_json,
    load_object,
    sha256_file,
)

REMEDIATION_VERSION = "gw-truck-panel-interiors-v2"
REGION_VERSION = "truck-bed-panel-interiors-v2"
FAILED_REGION_VERSION = "truck-bed-side-panels-v1"
CANONICAL_INSTANCE_ID = "panel-middle"
FITTING_CAMERA_IDS = ("cam-train-000251", "cam-train-000180")
EVALUATION_PREFIX = "cam-eval-"

# Candidate A is deliberately inset from every whole-panel boundary.
EQUAL_INTERIOR_HALF_EXTENTS = (0.24, 0.18, 0.10)
# Candidate C samples the right-side brace in each original frame.
BRACE_HALF_EXTENTS = (0.055, 0.23, 0.14)

_STATUS_BY_ID = {
    "panel-rear": OwnershipStatus.PANEL_REAR_CORE,
    "panel-middle": OwnershipStatus.PANEL_MIDDLE_CORE,
    "panel-front": OwnershipStatus.PANEL_FRONT_CORE,
}


def compare_dimensions(
    dimensions: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Return deterministic per-axis ratios and the rigid 10% gate."""

    if tuple(dimensions) != INSTANCE_IDS:
        raise StructuringError("candidate dimensions must use frozen instance order")
    rows = [tuple(float(value) for value in dimensions[key]) for key in INSTANCE_IDS]
    if any(len(row) != 3 or min(row) <= 0 for row in rows):
        raise StructuringError("candidate dimensions must be positive xyz triples")
    ratios = tuple(
        max(row[axis] for row in rows) / min(row[axis] for row in rows)
        for axis in range(3)
    )
    pairwise = {
        f"{left}__{right}": [
            abs(a - b) / max(a, b)
            for a, b in zip(dimensions[left], dimensions[right], strict=True)
        ]
        for left, right in itertools.combinations(INSTANCE_IDS, 2)
    }
    return {
        "axis_ratios": list(ratios),
        "maximum_pairwise_fractional_difference": max(
            value for values in pairwise.values() for value in values
        ),
        "pairwise_fractional_differences": pairwise,
        "rigid_dimension_gate": max(ratios) <= 1.10 + 1e-9,
    }


def validate_diagonal_scale_transform(
    matrix: Sequence[float],
    *,
    lower: float = 0.80,
    upper: float = 1.25,
) -> tuple[float, float, float]:
    """Validate a proper rotation times bounded diagonal scale, never shear."""

    if len(matrix) != 16 or not all(math.isfinite(float(value)) for value in matrix):
        raise StructuringError("diagonal-scale transform must have 16 finite values")
    if any(
        abs(float(matrix[index]) - expected) > 1e-6
        for index, expected in zip((12, 13, 14, 15), (0, 0, 0, 1), strict=True)
    ):
        raise StructuringError("invalid diagonal-scale homogeneous transform")
    columns = tuple(
        tuple(float(matrix[row * 4 + column]) for row in range(3))
        for column in range(3)
    )
    scales = tuple(
        math.sqrt(sum(value * value for value in column)) for column in columns
    )
    if min(scales) <= 1e-8 or any(value < lower or value > upper for value in scales):
        raise StructuringError("diagonal scale exceeds declared bounds")
    axes = tuple(
        tuple(value / scale for value in column)
        for column, scale in zip(columns, scales, strict=True)
    )
    for left in range(3):
        for right in range(3):
            dot = sum(axes[left][axis] * axes[right][axis] for axis in range(3))
            if abs(dot - (1.0 if left == right else 0.0)) > 1e-5:
                raise StructuringError("diagonal-scale transform contains shear")
    determinant = (
        axes[0][0] * (axes[1][1] * axes[2][2] - axes[1][2] * axes[2][1])
        - axes[1][0] * (axes[0][1] * axes[2][2] - axes[0][2] * axes[2][1])
        + axes[2][0] * (axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1])
    )
    if determinant <= 0:
        raise StructuringError("diagonal-scale transform reflects geometry")
    return cast(tuple[float, float, float], scales)


def transform_anisotropic_gaussians(
    means: Any,
    quaternions: Any,
    scales: Any,
    matrix: Sequence[float],
) -> tuple[Any, Any, Any]:
    """Apply a bounded diagonal-scale affine to Gaussian covariances exactly."""

    torch = _torch()
    diagonal = validate_diagonal_scale_transform(matrix)
    affine = torch.tensor(matrix, dtype=means.dtype, device=means.device).reshape(4, 4)
    linear = affine[:3, :3]
    transformed_means = means @ linear.T + affine[:3, 3]
    rotations = _quaternion_matrices(quaternions)
    covariance = (
        rotations @ torch.diag_embed(scales.square()) @ rotations.transpose(1, 2)
    )
    transformed = linear @ covariance @ linear.T
    eigenvalues, eigenvectors = torch.linalg.eigh(transformed)
    order = torch.argsort(eigenvalues, dim=1, descending=True)
    eigenvalues = torch.gather(eigenvalues, 1, order)
    eigenvectors = torch.gather(eigenvectors, 2, order[:, None, :].expand(-1, 3, -1))
    reflected = torch.linalg.det(eigenvectors) < 0
    eigenvectors[reflected, :, 2] *= -1
    transformed_scales = eigenvalues.clamp_min(0).sqrt()
    transformed_quaternions = _matrices_to_quaternions(eigenvectors)
    if not all(math.isfinite(value) for value in diagonal):
        raise StructuringError("nonfinite diagonal scale")
    return transformed_means, transformed_quaternions, transformed_scales


def select_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select from a strict training-only projection of passing candidates."""

    allowed = {
        "candidate_id",
        "semantic_description",
        "geometry_gate",
        "visibility_fitting_gate",
        "separation_gate",
        "representation_gate",
        "appeal_gate",
        "training_only_score",
        "transform_class",
    }
    projected = [{key: item[key] for key in allowed} for item in candidates]
    passing = [
        item
        for item in projected
        if all(
            bool(item[key])
            for key in (
                "geometry_gate",
                "visibility_fitting_gate",
                "separation_gate",
                "representation_gate",
                "appeal_gate",
            )
        )
    ]
    if not passing:
        return {
            "selected_candidate_id": None,
            "status": "no_candidate_passed",
            "selection_inputs": sorted(allowed),
            "held_out_evidence_used": False,
        }
    winner = max(
        passing,
        key=lambda item: (
            float(item["training_only_score"]),
            str(item["candidate_id"]),
        ),
    )
    return {
        "selected_candidate_id": winner["candidate_id"],
        "status": "selected",
        "transform_class": winner["transform_class"],
        "selection_inputs": sorted(allowed),
        "held_out_evidence_used": False,
        "rationale": (
            "highest training-only score among candidates passing geometry, fitting-"
            "view visibility, separation, representation, and source-view appeal"
        ),
    }


def remediation_stop_decision(
    *,
    selected_candidate: str | None,
    visual_decision: str,
    residual_gate: bool,
    requires_per_gaussian_geometry: bool = False,
) -> dict[str, Any]:
    """Apply the final bounded Truck stop rule."""

    stop_reasons = []
    if selected_candidate is None:
        stop_reasons.append("no_candidate_passed")
    if visual_decision == "not_credible":
        stop_reasons.append("revised_fitting_not_credible")
    if requires_per_gaussian_geometry:
        stop_reasons.append("requires_per_instance_per_gaussian_geometry")
    status = (
        "activate_explicit_fallback" if stop_reasons else "truck_remediation_passed"
    )
    return {
        "status": status,
        "truck_suitable_for_evaluation": not stop_reasons,
        "compact_residual_gate": residual_gate,
        "stop_reasons": stop_reasons,
        "next_step": "select_an_alternative_scene" if stop_reasons else None,
    }


def audit_candidates(
    *,
    v1_region_root: Path,
    dataset_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Measure the three mandated bounded candidates using local geometry only."""

    torch = _torch()
    v1_manifest = load_object(v1_region_root / "manifest.json")
    v1_frames = load_object(v1_region_root / "frames.json")["frames"]
    v1_ownership = torch.frombuffer(
        bytearray((v1_region_root / "ownership.u8").read_bytes()), dtype=torch.uint8
    )
    source = RealGaussianTensorSource.load(
        dataset_root / str(v1_manifest["converted_root_relative"])
    )
    tensors = source.tensors("cpu")
    means = tensors["means"].to(torch.float64)
    split = json.loads(
        (dataset_root / "annotations" / "evaluation-split.json").read_text(
            encoding="utf-8"
        )
    )
    cameras = {
        str(record["camera_id"]): cast(Mapping[str, Any], record)
        for record in split["cameras"]
    }
    frame_by_id = {str(frame["instance_id"]): frame for frame in v1_frames}
    local_by_id: dict[str, Any] = {}
    original_masks: dict[str, Any] = {}
    for instance_id in INSTANCE_IDS:
        frame = frame_by_id[instance_id]
        axes = torch.tensor(frame["axes_world"], dtype=torch.float64).T
        origin = torch.tensor(frame["origin_world"], dtype=torch.float64)
        local_by_id[instance_id] = (means - origin) @ axes
        original_masks[instance_id] = v1_ownership == int(_STATUS_BY_ID[instance_id])
    interior_centers = _bounded_median_centers(
        local_by_id,
        original_masks,
        frame_by_id,
        EQUAL_INTERIOR_HALF_EXTENTS,
        boundary_margin=0.04,
    )
    brace_centers = _bounded_median_centers(
        local_by_id,
        original_masks,
        frame_by_id,
        BRACE_HALF_EXTENTS,
        boundary_margin=0.0,
        pin_x_to_upper=True,
    )

    candidate_specs = (
        {
            "candidate_id": "candidate-a-equal-interiors",
            "semantic_description": "equal-size central sideboard plank interiors",
            "transform_class": "rigid",
            "centers": interior_centers,
            "half_extents": EQUAL_INTERIOR_HALF_EXTENTS,
            "appeal": "promising",
            "intended_qualitative_edit": (
                "duplicate or remove one legible interior plank patch"
            ),
        },
        {
            "candidate_id": "candidate-b-parameterized-panel-family",
            "semantic_description": "whole sideboard parameterized panel family",
            "transform_class": "translation_rotation_diagonal_scale",
            "centers": {instance_id: (0.0, 0.0, 0.0) for instance_id in INSTANCE_IDS},
            "half_extents": None,
            "appeal": "not_promising",
            "intended_qualitative_edit": "resize or duplicate a whole sideboard bay",
        },
        {
            "candidate_id": "candidate-c-vertical-braces",
            "semantic_description": "right-edge vertical sideboard brace segments",
            "transform_class": "rigid",
            "centers": brace_centers,
            "half_extents": BRACE_HALF_EXTENTS,
            "appeal": "promising_with_limitations",
            "intended_qualitative_edit": "duplicate or remove one vertical brace",
        },
    )
    records = [
        _measure_candidate(
            spec=spec,
            local_by_id=local_by_id,
            original_masks=original_masks,
            frames=frame_by_id,
            cameras=cameras,
            opacities=tensors["opacities"],
        )
        for spec in candidate_specs
    ]
    selection = select_candidate(records)
    result = {
        "remediation_version": REMEDIATION_VERSION,
        "failed_region_version": FAILED_REGION_VERSION,
        "source_v1_manifest_digest": v1_manifest["scientific_digest"],
        "candidate_count": len(records),
        "candidates": records,
        "selection": selection,
        "selection_policy": {
            "allowed": [
                "geometry diagnostics",
                "source annotations",
                "registration residuals",
                "fitting-camera geometric visibility",
                "source/fitting-view contact-sheet appeal",
                "storage estimates",
            ],
            "forbidden": [
                "held-out metrics",
                "held-out render appearance",
                "held-out hero ranking",
                "test-view error maps",
            ],
            "held_out_evidence_used": False,
        },
    }
    result["scientific_digest"] = content_digest(cast(Any, result))
    atomic_json(output, result)
    return result


def extract_revised_region(
    *,
    audit_path: Path,
    v1_region_root: Path,
    dataset_root: Path,
    output: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract the selected equal interior terminal without altering v1."""

    torch = _torch()
    audit = load_object(audit_path)
    if audit["selection"]["selected_candidate_id"] != "candidate-a-equal-interiors":
        raise StructuringError("only the selected equal-interior candidate may extract")
    candidate = next(
        item
        for item in audit["candidates"]
        if item["candidate_id"] == "candidate-a-equal-interiors"
    )
    candidate_centers = candidate["center_local_m"]
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite revised region: {output}")
    output.mkdir(parents=True, exist_ok=True)
    v1_manifest = load_object(v1_region_root / "manifest.json")
    old_frames = load_object(v1_region_root / "frames.json")["frames"]
    old_ownership = torch.frombuffer(
        bytearray((v1_region_root / "ownership.u8").read_bytes()), dtype=torch.uint8
    )
    source = RealGaussianTensorSource.load(
        dataset_root / str(v1_manifest["converted_root_relative"])
    )
    tensors = source.tensors("cpu")
    means = tensors["means"].to(torch.float64)
    half = torch.tensor(EQUAL_INTERIOR_HALF_EXTENTS, dtype=torch.float64)
    frames: list[PanelFrame] = []
    local_points: dict[str, Any] = {}
    core_indices: dict[str, Any] = {}
    expanded_masks: dict[str, Any] = {}
    for frame_record in old_frames:
        instance_id = str(frame_record["instance_id"])
        axes = torch.tensor(frame_record["axes_world"], dtype=torch.float64).T
        old_origin = torch.tensor(frame_record["origin_world"], dtype=torch.float64)
        center = torch.tensor(candidate_centers[instance_id], dtype=torch.float64)
        origin = old_origin + axes @ center
        local = (means - old_origin) @ axes - center
        original = old_ownership == int(_STATUS_BY_ID[instance_id])
        selected = original & (local.abs() <= half).all(dim=1)
        if int(selected.sum().item()) < 256:
            raise StructuringError(f"revised interior core too small: {instance_id}")
        expanded = original & (local.abs() <= half + 0.04).all(dim=1) & ~selected
        expanded_masks[instance_id] = expanded
        core_indices[instance_id] = torch.nonzero(selected).flatten()
        local_points[instance_id] = local[selected]
        axes_tuple = tuple(
            tuple(float(value) for value in axis) for axis in frame_record["axes_world"]
        )
        frame = PanelFrame(
            instance_id=instance_id,
            origin_world=cast(
                tuple[float, float, float],
                tuple(float(value) for value in origin.tolist()),
            ),
            axes_world=cast(tuple[tuple[float, float, float], ...], axes_tuple),
            local_lower=cast(
                tuple[float, float, float], tuple(-float(value) for value in half)
            ),
            local_upper=cast(
                tuple[float, float, float], tuple(float(value) for value in half)
            ),
            support_plane_normal_world=cast(
                tuple[float, float, float],
                tuple(
                    float(value) for value in frame_record["support_plane_normal_world"]
                ),
            ),
            support_plane_offset=float(frame_record["support_plane_offset"]),
            source_annotation=(
                f"{REGION_VERSION}/{instance_id}/equal-central-interior"
            ),
            confidence="medium",
            method=(
                "v1 oracle frame with identical fixed 0.48 x 0.36 x 0.20 m "
                "interior bounds; no evaluation evidence"
            ),
        )
        frames.append(frame)
    ownership = torch.full(
        (SOURCE_COUNT,), int(OwnershipStatus.EXPLICIT_BACKGROUND), dtype=torch.uint8
    )
    for instance_id in INSTANCE_IDS:
        ownership[core_indices[instance_id]] = int(_STATUS_BY_ID[instance_id])
    guard = torch.zeros(SOURCE_COUNT, dtype=torch.bool)
    for expanded in expanded_masks.values():
        guard |= expanded
    ownership[guard] = int(OwnershipStatus.BOUNDARY_GUARD)
    ownership_path = output / "ownership.u8"
    ownership.numpy().tofile(ownership_path)
    core_counts = {
        instance_id: int(core_indices[instance_id].numel())
        for instance_id in INSTANCE_IDS
    }
    status_counts = {
        status.name.lower(): int((ownership == int(status)).sum().item())
        for status in OwnershipStatus
    }
    ownership_record = {
        "pilot_version": REMEDIATION_VERSION,
        "region_version": REGION_VERSION,
        "supersedes_region_version": FAILED_REGION_VERSION,
        "source_gaussian_count": SOURCE_COUNT,
        "payload": "ownership.u8",
        "payload_bytes": ownership_path.stat().st_size,
        "payload_sha256": sha256_file(ownership_path),
        "status_codes": {
            status.name.lower(): int(status) for status in OwnershipStatus
        },
        "status_counts": status_counts,
        "core_counts": core_counts,
        "policy": {
            "core": "v1 core points inside identical fixed local interior bounds",
            "guard": "v1 core shell expanded by 0.04 m; retained explicit",
            "excluded_panel_content": "all other v1 panel rows remain explicit",
            "ordering": "official source PLY row order",
        },
    }
    ownership_record["scientific_digest"] = content_digest(cast(Any, ownership_record))
    atomic_json(output / "ownership.json", ownership_record)
    registrations = []
    for frame, instance_id in zip(frames, INSTANCE_IDS, strict=True):
        residual = candidate["registration"][instance_id]
        registrations.append(
            SimilarityRegistration(
                instance_id=instance_id,
                matrix=frame.world_from_local,
                scale=1.0,
                initialization_matrix=frame.world_from_local,
                objective_before=float(residual["initial_residual_m"]),
                objective_after=float(residual["initial_residual_m"]),
                correction_translation_m=0.0,
                rotation_angle_degrees=0.0,
                diagnostics={
                    "source_core_gaussian_count": core_counts[instance_id],
                    "canonical_gaussian_count": core_counts[CANONICAL_INSTANCE_ID],
                    "candidate_refined_residual_m": residual["refined_residual_m"],
                    "candidate_refinement_applied": False,
                    "candidate_refinement_reason": (
                        "fixed equal-core semantics retain annotation-centered "
                        "rigid frame"
                    ),
                    "outlier_fraction": residual["outlier_fraction"],
                    "transform_class": "rigid",
                    "scale_xyz": [1.0, 1.0, 1.0],
                    "shear": False,
                    "evaluation_views_used": [],
                },
            )
        )
    frames_record = {
        "pilot_version": REMEDIATION_VERSION,
        "region_version": REGION_VERSION,
        "canonical_instance_id": CANONICAL_INSTANCE_ID,
        "coordinate_convention": (
            "right-handed local xyz; identical fixed local bounds for all instances"
        ),
        "frames": [frame.to_dict() for frame in frames],
    }
    frames_record["scientific_digest"] = content_digest(cast(Any, frames_record))
    atomic_json(output / "frames.json", frames_record)
    registration_record = {
        "pilot_version": REMEDIATION_VERSION,
        "region_version": REGION_VERSION,
        "method": "annotation-centered rigid equal-core transform",
        "canonical_instance_id": CANONICAL_INSTANCE_ID,
        "evaluation_images_or_losses_used": False,
        "instances": [record.to_dict() for record in registrations],
    }
    registration_record["scientific_digest"] = content_digest(
        cast(Any, registration_record)
    )
    atomic_json(output / "registration.json", registration_record)
    canonical_dir = output / "canonical"
    canonical_dir.mkdir(exist_ok=True)
    middle_indices = core_indices[CANONICAL_INSTANCE_ID]
    middle_frame = frames[1]
    rotation = torch.tensor(middle_frame.axes_world, dtype=torch.float32).T
    frame_quaternion = _matrix_to_quaternion(rotation)
    inverse_frame_quaternion = frame_quaternion.clone()
    inverse_frame_quaternion[1:] *= -1
    local_quaternions = _quat_multiply(
        inverse_frame_quaternion.expand(middle_indices.numel(), -1),
        tensors["quaternions"][middle_indices],
    )
    local_quaternions /= torch.linalg.vector_norm(
        local_quaternions, dim=1, keepdim=True
    )
    canonical_files = {
        "means": _raw(
            canonical_dir / "means.f32",
            local_points[CANONICAL_INSTANCE_ID].to(torch.float32),
        ),
        "quaternions": _raw(canonical_dir / "quaternions.f32", local_quaternions),
        "scales": _raw(canonical_dir / "scales.f32", tensors["scales"][middle_indices]),
        "opacities": _raw(
            canonical_dir / "opacities.f32", tensors["opacities"][middle_indices]
        ),
        "sh_coefficients": _raw(
            canonical_dir / "sh-coefficients.f32",
            tensors["appearance"][middle_indices],
        ),
        "source_indices": _indices(
            canonical_dir / "source-indices.u32", middle_indices
        ),
    }
    canonical_meta = {
        "format_version": "gaussweave-real-canonical-v2",
        "pilot_version": REMEDIATION_VERSION,
        "representation_version": REGION_VERSION,
        "terminal_id": "truck-panel-interior-canonical-v2",
        "source_instance_id": CANONICAL_INSTANCE_ID,
        "gaussian_count": core_counts[CANONICAL_INSTANCE_ID],
        "terminal_count": 3,
        "transform_class": "rigid",
        "sh_degree": 3,
        "coordinate_frame": "panel-middle equal-interior local frame",
        "ordering_policy": "official source PLY row order retained",
        "fields": ["means", "quaternions", "scales", "opacities", "SH degree 3"],
        "local_bounds": [
            list(middle_frame.local_lower),
            list(middle_frame.local_upper),
        ],
        "files": canonical_files,
    }
    canonical_meta["scientific_digest"] = content_digest(cast(Any, canonical_meta))
    atomic_json(canonical_dir / "metadata.json", canonical_meta)
    core_statuses = tuple(int(_STATUS_BY_ID[item]) for item in INSTANCE_IDS)
    background_indices = torch.nonzero(
        ~torch.isin(ownership, torch.tensor(core_statuses, dtype=torch.uint8))
    ).flatten()
    background_record = _indices(output / "background-indices.u32", background_indices)
    split = json.loads(
        (dataset_root / "annotations" / "evaluation-split.json").read_text(
            encoding="utf-8"
        )
    )
    camera_records = {
        str(record["camera_id"]): cast(Mapping[str, Any], record)
        for record in split["cameras"]
    }
    instance_views: dict[str, dict[str, list[int]]] = {
        instance_id: {} for instance_id in INSTANCE_IDS
    }
    for instance_id in INSTANCE_IDS:
        selected_means = means[core_indices[instance_id]]
        for camera_id in FITTING_CAMERA_IDS:
            record = camera_records[camera_id]
            instance_views[instance_id][str(record["image_name"])] = list(
                _project_points_box(selected_means, record)
            )
    counts = {
        "source": SOURCE_COUNT,
        "source_explicit_panel": sum(core_counts.values()),
        "stored_canonical_panel": core_counts[CANONICAL_INSTANCE_ID],
        "materialized_canonical_panel": 3 * core_counts[CANONICAL_INSTANCE_ID],
        "fixed_explicit_background": int(background_indices.numel()),
        "hybrid_materialized": int(background_indices.numel())
        + 3 * core_counts[CANONICAL_INSTANCE_ID],
    }
    manifest = {
        "pilot_version": REMEDIATION_VERSION,
        "region_version": REGION_VERSION,
        "selected_candidate_id": "candidate-a-equal-interiors",
        "local_dataset_root": str(dataset_root.resolve()),
        "source_v1_region_root": str(v1_region_root.resolve()),
        "source_v1_manifest_digest": v1_manifest["scientific_digest"],
        "source_model_digest": source.metadata["model_digest"],
        "converted_root_relative": str(v1_manifest["converted_root_relative"]),
        "ownership": "ownership.json",
        "frames": "frames.json",
        "registration": "registration.json",
        "canonical": "canonical/metadata.json",
        "background_indices": background_record,
        "core_counts": core_counts,
        "counts": counts,
        "instance_views": instance_views,
        "fitting_camera_ids": list(FITTING_CAMERA_IDS),
        "evaluation_camera_ids": [
            str(record["camera_id"])
            for record in split["cameras"]
            if str(record["camera_id"]).startswith(EVALUATION_PREFIX)
        ],
        "selection_uses_held_out_evidence": False,
    }
    manifest["scientific_digest"] = content_digest(cast(Any, manifest))
    atomic_json(output / "manifest.json", manifest)
    return {
        "valid": True,
        "region_version": REGION_VERSION,
        "core_counts": core_counts,
        "status_counts": status_counts,
        "canonical_count": core_counts[CANONICAL_INSTANCE_ID],
        "canonical_digest": canonical_meta["scientific_digest"],
        "background_count": int(background_indices.numel()),
        "manifest_digest": manifest["scientific_digest"],
    }


def _bounded_median_centers(
    local_by_id: Mapping[str, Any],
    original_masks: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    half_extents: Sequence[float],
    *,
    boundary_margin: float,
    pin_x_to_upper: bool = False,
) -> dict[str, tuple[float, float, float]]:
    """Center equal boxes on robust source geometry while keeping them in bounds."""

    torch = _torch()
    half = torch.tensor(half_extents, dtype=torch.float64)
    result = {}
    for instance_id in INSTANCE_IDS:
        points = local_by_id[instance_id][original_masks[instance_id]]
        center = torch.median(points, dim=0).values
        lower = (
            torch.tensor(frames[instance_id]["local_lower"], dtype=torch.float64)
            + half
            + boundary_margin
        )
        upper = (
            torch.tensor(frames[instance_id]["local_upper"], dtype=torch.float64)
            - half
            - boundary_margin
        )
        if bool((lower > upper).any().item()):
            raise StructuringError("candidate box cannot fit original region bounds")
        center = torch.maximum(lower, torch.minimum(upper, center))
        if pin_x_to_upper:
            center[0] = upper[0]
        result[instance_id] = cast(
            tuple[float, float, float],
            tuple(float(value) for value in center.tolist()),
        )
    return result


def _measure_candidate(
    *,
    spec: Mapping[str, Any],
    local_by_id: Mapping[str, Any],
    original_masks: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    cameras: Mapping[str, Mapping[str, Any]],
    opacities: Any,
) -> dict[str, Any]:
    torch = _torch()
    candidate_id = str(spec["candidate_id"])
    masks: dict[str, Any] = {}
    local_selected: dict[str, Any] = {}
    dimensions: dict[str, tuple[float, float, float]] = {}
    guard_counts: dict[str, int] = {}
    boundary_contamination: dict[str, float] = {}
    if spec["half_extents"] is None:
        for instance_id in INSTANCE_IDS:
            masks[instance_id] = original_masks[instance_id]
            local_selected[instance_id] = local_by_id[instance_id][masks[instance_id]]
            frame = frames[instance_id]
            dimensions[instance_id] = cast(
                tuple[float, float, float],
                tuple(
                    float(upper) - float(lower)
                    for lower, upper in zip(
                        frame["local_lower"], frame["local_upper"], strict=True
                    )
                ),
            )
            guard_counts[instance_id] = 0
            boundary_contamination[instance_id] = 0.25
    else:
        half = torch.tensor(spec["half_extents"], dtype=torch.float64)
        for instance_id in INSTANCE_IDS:
            center = torch.tensor(spec["centers"][instance_id], dtype=torch.float64)
            local = local_by_id[instance_id]
            original = original_masks[instance_id]
            selected = original & ((local - center).abs() <= half).all(dim=1)
            expanded = original & ((local - center).abs() <= half + 0.04).all(dim=1)
            masks[instance_id] = selected
            local_selected[instance_id] = local[selected] - center
            dimensions[instance_id] = cast(
                tuple[float, float, float],
                tuple(2 * float(value) for value in half),
            )
            guard_counts[instance_id] = int((expanded & ~selected).sum().item())
            frame = frames[instance_id]
            lower = torch.tensor(frame["local_lower"], dtype=torch.float64)
            upper = torch.tensor(frame["local_upper"], dtype=torch.float64)
            points = local[selected]
            near_original_boundary = (
                ((points - lower) < 0.04) | ((upper - points) < 0.04)
            ).any(dim=1)
            boundary_contamination[instance_id] = float(
                near_original_boundary.to(torch.float64).mean().item()
                if points.numel()
                else 1.0
            )
    comparison = compare_dimensions(dimensions)
    registration = _registration_diagnostics(local_selected, opacities, masks)
    source_counts = {
        instance_id: int(masks[instance_id].sum().item())
        for instance_id in INSTANCE_IDS
    }
    if spec["half_extents"] is None:
        scale_xyz = {
            instance_id: [
                dimensions[instance_id][axis] / dimensions[CANONICAL_INSTANCE_ID][axis]
                for axis in range(3)
            ]
            for instance_id in INSTANCE_IDS
        }
        scale_gate = all(
            0.80 <= value <= 1.25 for row in scale_xyz.values() for value in row
        )
        geometry_gate = (
            scale_gate
            and max(record["refined_residual_m"] for record in registration.values())
            < 0.05
        )
    else:
        scale_xyz = {instance_id: [1.0, 1.0, 1.0] for instance_id in INSTANCE_IDS}
        geometry_gate = (
            bool(comparison["rigid_dimension_gate"])
            and max(record["refined_residual_m"] for record in registration.values())
            < 0.055
        )
    visibility = _candidate_visibility(spec, frames, cameras)
    fitting_visible = all(
        all(area >= 64 for area in visibility[instance_id]["fitting_pixel_areas"])
        for instance_id in INSTANCE_IDS
    )
    separation_gate = (
        min(source_counts.values()) >= 256
        and max(boundary_contamination.values()) <= 0.10
    )
    representation_gate = (
        min(source_counts.values()) >= 256
        and str(spec["transform_class"])
        in {"rigid", "translation_rotation_diagonal_scale"}
        and sum(source_counts.values()) < 100_000
    )
    appeal_gate = spec["appeal"] in {"promising", "promising_with_limitations"}
    canonical_count = source_counts[CANONICAL_INSTANCE_ID]
    explicit_bytes = sum(source_counts.values()) * 236
    expected_structured = canonical_count * 236 + 3 * 16 * 8 + 4096
    mean_refined = (
        sum(float(record["refined_residual_m"]) for record in registration.values()) / 3
    )
    hero_area = sum(
        visibility[instance_id]["fitting_pixel_areas"][0]
        for instance_id in INSTANCE_IDS
    )
    training_score = (
        (1.0 - min(1.0, mean_refined / 0.055)) * 0.45
        + min(1.0, hero_area / 3500) * 0.30
        + min(1.0, expected_structured and explicit_bytes / expected_structured / 4)
        * 0.25
    )
    return {
        "candidate_id": candidate_id,
        "semantic_description": spec["semantic_description"],
        "instance_count": 3,
        "hero_view_pixel_area": hero_area,
        "source_gaussian_counts": source_counts,
        "guard_gaussian_counts": guard_counts,
        "dimensions_m": {
            instance_id: list(dimensions[instance_id]) for instance_id in INSTANCE_IDS
        },
        "center_local_m": {
            instance_id: [float(value) for value in spec["centers"][instance_id]]
            for instance_id in INSTANCE_IDS
        },
        "dimension_comparison": comparison,
        "scale_xyz_from_canonical": scale_xyz,
        "declared_scale_bounds": [0.80, 1.25],
        "transform_class": spec["transform_class"],
        "registration": registration,
        "boundary_contamination_fraction": boundary_contamination,
        "visibility": visibility,
        "expected_terminal_count": 3,
        "expected_repeated_region": {
            "explicit_bytes": explicit_bytes,
            "structured_bytes": expected_structured,
            "explicit_over_structured_ratio": explicit_bytes / expected_structured,
        },
        "intended_qualitative_edit": spec["intended_qualitative_edit"],
        "visual_salience_score": min(1.0, hero_area / 3500),
        "source_view_appeal_classification": spec["appeal"],
        "geometry_gate": geometry_gate,
        "visibility_fitting_gate": fitting_visible,
        "held_out_visibility_count": min(
            visibility[item]["held_out_visible_camera_count"] for item in INSTANCE_IDS
        ),
        "separation_gate": separation_gate,
        "representation_gate": representation_gate,
        "appeal_gate": appeal_gate,
        "training_only_score": training_score,
        "selection_uses_held_out_evidence": False,
    }


def _registration_diagnostics(
    local_selected: Mapping[str, Any],
    opacities: Any,
    masks: Mapping[str, Any],
) -> dict[str, Any]:
    torch = _torch()
    canonical = local_selected[CANONICAL_INSTANCE_ID].to(torch.float64)
    if canonical.shape[0] > 512:
        indexes = torch.linspace(0, canonical.shape[0] - 1, 512).round().to(torch.int64)
        canonical = canonical[indexes]
    result: dict[str, Any] = {}
    for instance_id in INSTANCE_IDS:
        target = local_selected[instance_id].to(torch.float64)
        if target.shape[0] > 512:
            indexes = (
                torch.linspace(0, target.shape[0] - 1, 512).round().to(torch.int64)
            )
            target = target[indexes]
        if canonical.numel() == 0 or target.numel() == 0:
            result[instance_id] = {
                "initial_residual_m": math.inf,
                "refined_residual_m": math.inf,
                "translation_correction_m": math.inf,
                "outlier_fraction": 1.0,
            }
            continue
        distances = torch.cdist(canonical, target).amin(dim=1)
        initial = float(torch.quantile(distances, 0.5).item())
        correction = (
            torch.median(target, dim=0).values - torch.median(canonical, dim=0).values
        )
        refined_distances = torch.cdist(canonical + correction, target).amin(dim=1)
        refined = float(torch.quantile(refined_distances, 0.5).item())
        p90 = torch.quantile(refined_distances, 0.9)
        result[instance_id] = {
            "initial_residual_m": initial,
            "refined_residual_m": refined,
            "translation_correction_m": float(
                torch.linalg.vector_norm(correction).item()
            ),
            "outlier_fraction": float(
                (refined_distances > p90).to(torch.float64).mean().item()
            ),
            "opacity_mean": float(opacities[masks[instance_id]].mean().item()),
        }
    return result


def _candidate_visibility(
    spec: Mapping[str, Any],
    frames: Mapping[str, Mapping[str, Any]],
    cameras: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for instance_id in INSTANCE_IDS:
        frame = frames[instance_id]
        if spec["half_extents"] is None:
            lower = tuple(float(value) for value in frame["local_lower"])
            upper = tuple(float(value) for value in frame["local_upper"])
            center = tuple((a + b) / 2 for a, b in zip(lower, upper, strict=True))
            half = tuple((b - a) / 2 for a, b in zip(lower, upper, strict=True))
        else:
            center = tuple(float(value) for value in spec["centers"][instance_id])
            half = tuple(float(value) for value in spec["half_extents"])
        areas: dict[str, int] = {}
        for camera_id, camera in cameras.items():
            box = _project_local_box(frame, center, half, camera)
            width = max(0, box[2] - box[0])
            height = max(0, box[3] - box[1])
            areas[camera_id] = width * height
        result[instance_id] = {
            "fitting_pixel_areas": [
                areas[camera_id] for camera_id in FITTING_CAMERA_IDS
            ],
            "held_out_visible_camera_count": sum(
                area >= 64
                for camera_id, area in areas.items()
                if camera_id.startswith(EVALUATION_PREFIX)
            ),
            "fitting_visible": all(
                areas[camera_id] >= 64 for camera_id in FITTING_CAMERA_IDS
            ),
        }
    return result


def _project_local_box(
    frame: Mapping[str, Any],
    center: Sequence[float],
    half: Sequence[float],
    camera: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    torch = _torch()
    axes = torch.tensor(frame["axes_world"], dtype=torch.float64).T
    origin = torch.tensor(frame["origin_world"], dtype=torch.float64)
    corners = torch.tensor(
        [
            [center[axis] + sign[axis] * half[axis] for axis in range(3)]
            for sign in itertools.product((-1, 1), repeat=3)
        ],
        dtype=torch.float64,
    )
    world = corners @ axes.T + origin
    return _project_points_box(world, camera)


def _project_points_box(
    world: Any, camera: Mapping[str, Any]
) -> tuple[int, int, int, int]:
    torch = _torch()
    transform = torch.tensor(camera["world_from_camera"], dtype=torch.float64).reshape(
        4, 4
    )
    camera_from_world = torch.linalg.inv(transform)
    homogeneous = torch.cat(
        (world, torch.ones((world.shape[0], 1), dtype=torch.float64)), dim=1
    )
    points = (camera_from_world @ homogeneous.T).T[:, :3]
    valid = points[:, 2] > 1e-5
    if not bool(valid.any().item()):
        return (0, 0, 0, 0)
    points = points[valid]
    x = float(camera["fx"]) * points[:, 0] / points[:, 2] + float(camera["cx"])
    y = float(camera["fy"]) * points[:, 1] / points[:, 2] + float(camera["cy"])
    width, height = int(camera["width"]), int(camera["height"])
    return (
        max(0, int(torch.floor(x.min()).item()) - 2),
        max(0, int(torch.floor(y.min()).item()) - 2),
        min(width, int(torch.ceil(x.max()).item()) + 3),
        min(height, int(torch.ceil(y.max()).item()) + 3),
    )


def _quaternion_matrices(quaternions: Any) -> Any:
    torch = _torch()
    q = quaternions / torch.linalg.vector_norm(quaternions, dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)


def _matrices_to_quaternions(matrices: Any) -> Any:
    torch = _torch()
    results = []
    for matrix in matrices:
        trace = float(torch.trace(matrix).item())
        if trace > 0:
            scale = math.sqrt(trace + 1) * 2
            values = (
                0.25 * scale,
                float((matrix[2, 1] - matrix[1, 2]).item()) / scale,
                float((matrix[0, 2] - matrix[2, 0]).item()) / scale,
                float((matrix[1, 0] - matrix[0, 1]).item()) / scale,
            )
        else:
            diagonal = torch.diagonal(matrix)
            axis = int(torch.argmax(diagonal).item())
            if axis == 0:
                scale = (
                    math.sqrt(1 + float(matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2
                )
                values = (
                    float((matrix[2, 1] - matrix[1, 2]).item()) / scale,
                    0.25 * scale,
                    float((matrix[0, 1] + matrix[1, 0]).item()) / scale,
                    float((matrix[0, 2] + matrix[2, 0]).item()) / scale,
                )
            elif axis == 1:
                scale = (
                    math.sqrt(1 + float(matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2
                )
                values = (
                    float((matrix[0, 2] - matrix[2, 0]).item()) / scale,
                    float((matrix[0, 1] + matrix[1, 0]).item()) / scale,
                    0.25 * scale,
                    float((matrix[1, 2] + matrix[2, 1]).item()) / scale,
                )
            else:
                scale = (
                    math.sqrt(1 + float(matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2
                )
                values = (
                    float((matrix[1, 0] - matrix[0, 1]).item()) / scale,
                    float((matrix[0, 2] + matrix[2, 0]).item()) / scale,
                    float((matrix[1, 2] + matrix[2, 1]).item()) / scale,
                    0.25 * scale,
                )
        quaternion = torch.tensor(values, dtype=matrix.dtype, device=matrix.device)
        quaternion /= torch.linalg.vector_norm(quaternion)
        if quaternion[0] < 0:
            quaternion *= -1
        results.append(quaternion)
    return torch.stack(results)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise StructuringError("PyTorch is required for Truck remediation") from error
    return torch
