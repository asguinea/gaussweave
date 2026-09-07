"""Deterministic panel-frame construction and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from gaussweave.real_structuring.models import (
    INSTANCE_IDS,
    PanelFrame,
    StructuringError,
)


def build_frames(
    annotation: Mapping[str, Any],
    local_bounds: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[PanelFrame, ...]:
    """Build consistently signed annotation-derived frames in frozen instance order."""

    records = {
        str(item["instance_id"]): item
        for item in cast(Sequence[Mapping[str, Any]], annotation["instances"])
    }
    frames: list[PanelFrame] = []
    reference_axes: tuple[tuple[float, float, float], ...] | None = None
    for instance_id in INSTANCE_IDS:
        item = records[instance_id]
        box = item["provisional_3d_box"]
        raw_axes = cast(Sequence[Sequence[float]], box["axes_world"])
        axes = tuple(tuple(float(value) for value in axis) for axis in raw_axes)
        axes = cast(tuple[tuple[float, float, float], ...], axes)
        if reference_axes is None:
            reference_axes = axes
        else:
            for axis, reference in zip(axes, reference_axes, strict=True):
                if sum(a * b for a, b in zip(axis, reference, strict=True)) < 0.999:
                    raise StructuringError(
                        "panel frame semantic axis orientation differs"
                    )
        bounds = local_bounds[instance_id]
        plane = cast(Mapping[str, Any], box["support_plane"])
        frames.append(
            PanelFrame(
                instance_id=instance_id,
                origin_world=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in box["center_world"]),
                ),
                axes_world=axes,
                local_lower=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in bounds[0]),
                ),
                local_upper=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in bounds[1]),
                ),
                support_plane_normal_world=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in plane["normal_world"]),
                ),
                support_plane_offset=float(plane["offset"]),
                source_annotation=(
                    "gw-truck-explicit-v1/truck-bed-side-panels/"
                    f"{instance_id}/provisional_3d_box"
                ),
                confidence=str(item["annotation_confidence"]),
                method=(
                    "camera-aligned oracle frame; core-local min/max bounds; "
                    "signs normalized rear-to-front from cam-train-000251"
                ),
            )
        )
    return tuple(frames)
