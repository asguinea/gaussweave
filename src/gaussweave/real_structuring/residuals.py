"""Deterministic compact per-instance appearance residuals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

from gaussweave.config.resolution import content_digest
from gaussweave.real_structuring.models import (
    CANONICAL_INSTANCE_ID,
    INSTANCE_IDS,
    Q8Residuals,
    StructuringError,
)

SH1_CODEC_VERSION = "gaussweave-real-sh1-q8-v1"


def _round_half_away(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def encode_q8(
    offsets: Sequence[Sequence[float]],
    *,
    scale: float | None = None,
    instance_ids: Sequence[str] = INSTANCE_IDS,
    canonical_zero_instance: str = CANONICAL_INSTANCE_ID,
) -> Q8Residuals:
    """Encode one RGB/SH0 vector per panel using a single declared scale."""

    frozen_ids = tuple(str(value) for value in instance_ids)
    if not frozen_ids or len(frozen_ids) != len(set(frozen_ids)):
        raise StructuringError("float residual instance IDs must be unique")
    if canonical_zero_instance not in frozen_ids:
        raise StructuringError("float residual canonical instance is missing")
    if len(offsets) != len(frozen_ids) or any(len(row) != 3 for row in offsets):
        raise StructuringError("float residual offsets must have shape [instances,3]")
    values = [tuple(float(value) for value in row) for row in offsets]
    values[frozen_ids.index(canonical_zero_instance)] = (0.0, 0.0, 0.0)
    maximum = max(abs(value) for row in values for value in row)
    chosen = maximum / 127 if scale is None and maximum > 0 else (scale or 1.0 / 127)
    if not math.isfinite(chosen) or chosen <= 0:
        raise StructuringError("q8 scale must be finite and positive")
    encoded: list[tuple[int, int, int]] = []
    saturation = 0
    for row in values:
        channels: list[int] = []
        for value in row:
            raw = _round_half_away(value / chosen)
            clipped = min(127, max(-128, raw))
            saturation += int(clipped != raw)
            channels.append(clipped)
        encoded.append((channels[0], channels[1], channels[2]))
    return Q8Residuals(
        instance_ids=frozen_ids,
        values=tuple(encoded),
        scale=chosen,
        saturation_count=saturation,
        canonical_zero_instance=canonical_zero_instance,
    )


def encode_sh1_q8(offsets: Any) -> dict[str, Any]:
    """Encode one signed-int8 degree-1 SH residual per instance."""

    rows = cast(list[list[list[float]]], offsets)
    if (
        len(rows) != len(INSTANCE_IDS)
        or any(len(instance) != 4 for instance in rows)
        or any(len(coefficient) != 3 for instance in rows for coefficient in instance)
    ):
        raise StructuringError("SH1 q8 residuals must have shape [3,4,3]")
    rows[INSTANCE_IDS.index(CANONICAL_INSTANCE_ID)] = [
        [0.0, 0.0, 0.0] for _ in range(4)
    ]
    maximum = max(
        abs(value)
        for instance in rows
        for coefficient in instance
        for value in coefficient
    )
    scale = maximum / 127 if maximum > 0 else 1 / 127
    saturation = 0
    encoded: list[list[list[int]]] = []
    for instance in rows:
        encoded_instance = []
        for coefficient in instance:
            encoded_coefficient = []
            for value in coefficient:
                quantized = _round_half_away(value / scale)
                clipped = max(-127, min(127, quantized))
                saturation += int(clipped != quantized)
                encoded_coefficient.append(clipped)
            encoded_instance.append(encoded_coefficient)
        encoded.append(encoded_instance)
    result: dict[str, Any] = {
        "codec_version": SH1_CODEC_VERSION,
        "instance_ids": list(INSTANCE_IDS),
        "coefficient_count": 4,
        "channel_count": 3,
        "values": encoded,
        "scale": scale,
        "zero_point": 0,
        "rounding": "round_half_away_from_zero",
        "saturation_count": saturation,
        "canonical_zero_instance": CANONICAL_INSTANCE_ID,
        "decoded": [
            [[value * scale for value in coefficient] for coefficient in instance]
            for instance in encoded
        ],
    }
    result["scientific_digest"] = content_digest(cast(Any, result))
    return result
