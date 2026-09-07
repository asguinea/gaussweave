"""Pure validation for generic metric geometry and material specifications."""

from __future__ import annotations

import math
from collections.abc import Sequence


class SpecificationError(ValueError):
    """A generic geometry or material specification is invalid."""


def validate_positive_finite(
    values: Sequence[float], *, label: str = "dimensions"
) -> None:
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in values
    ):
        raise SpecificationError(f"{label} must contain positive finite values")


def validate_material_values(
    base_color: Sequence[float],
    *,
    roughness: float,
    metallic: float,
    alpha: float,
) -> tuple[float, float, float, float]:
    if len(base_color) != 4:
        raise SpecificationError("base color must contain RGBA")
    color = tuple(float(value) for value in base_color)
    numeric = (*color, float(roughness), float(metallic), float(alpha))
    if not all(math.isfinite(value) for value in numeric):
        raise SpecificationError("material values must be finite")
    if any(value < 0.0 or value > 1.0 for value in numeric):
        raise SpecificationError("material values must be in [0, 1]")
    return color


def validate_material_reuse(existing: str | None, requested: str) -> None:
    """Reject reuse when a stored deterministic definition conflicts."""

    if existing != requested:
        raise SpecificationError("conflicting material definition")
