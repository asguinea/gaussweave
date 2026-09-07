"""Generic deterministic Principled BSDF material helpers."""

from __future__ import annotations

from collections.abc import Sequence

import bpy

from .context import GeneratorContext
from .specs import (
    SpecificationError,
    validate_material_reuse,
    validate_material_values,
)


class MaterialError(ValueError):
    """Material parameters or a duplicate definition are invalid."""


def create_principled_material(
    context: GeneratorContext,
    *,
    stable_id: str,
    base_color: Sequence[float],
    roughness: float,
    metallic: float,
    alpha: float = 1.0,
    provenance: str = "runtime-probe",
) -> bpy.types.Material:
    try:
        color = validate_material_values(
            base_color,
            roughness=roughness,
            metallic=metallic,
            alpha=alpha,
        )
    except SpecificationError as error:
        raise MaterialError(str(error)) from error
    name = context.naming.reserve("MAT", stable_id, allow_reuse=True)
    definition = {
        "stable_id": stable_id,
        "name": name,
        "base_color": [round(value, 12) for value in color],
        "roughness": round(float(roughness), 12),
        "metallic": round(float(metallic), 12),
        "alpha": round(float(alpha), 12),
        "provenance": provenance,
    }
    existing = bpy.data.materials.get(name)
    if existing is not None:
        stored = existing.get("gaussweave_definition")
        try:
            validate_material_reuse(stored, repr(definition))
        except SpecificationError as error:
            raise MaterialError(
                f"conflicting material definition: {stable_id}"
            ) from error
        return existing
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color[:3] + (float(alpha),)
    material["gaussweave_id"] = stable_id
    material["gaussweave_provenance"] = provenance
    material["gaussweave_definition"] = repr(definition)
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        raise MaterialError("Principled BSDF node is unavailable")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = float(roughness)
    principled.inputs["Metallic"].default_value = float(metallic)
    principled.inputs["Alpha"].default_value = float(alpha)
    if alpha < 1.0:
        material.surface_render_method = "DITHERED"
    context.materials.append(definition)
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.data is None or not hasattr(obj.data, "materials"):
        raise MaterialError("object does not accept materials")
    obj.data.materials.clear()
    obj.data.materials.append(material)
