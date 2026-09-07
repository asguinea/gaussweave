"""Scientific metadata and volatile execution-summary builders."""

from __future__ import annotations

from typing import Any

from . import GENERATOR_METADATA_VERSION
from .context import GeneratorContext
from .io import content_digest
from .scene import collection_hierarchy


def scientific_metadata(
    context: GeneratorContext, render_settings: dict[str, Any]
) -> dict[str, Any]:
    """Build deterministic metadata with no local paths or timestamps."""

    scientific = {
        "metadata_schema_version": GENERATOR_METADATA_VERSION,
        "purpose": context.purpose,
        "generator": {
            "name": str(context.config["generator"]["name"]),
            "version": context.generator_version,
            "runtime_version": "blender-runtime-v1",
        },
        "scene_id": context.scene_id,
        "family": context.family,
        "configuration_digest": context.configuration_digest,
        "seeds": context.seeds.metadata(),
        "coordinates": {
            "handedness": "right",
            "up_axis": "+Z",
            "unit": "meters",
            "blender_units_per_meter": 1.0,
            "camera_axes": {"right": "+X", "up": "+Y", "forward": "-Z"},
            "matrix_layout": "row_major",
        },
        "collections": collection_hierarchy(context),
        "names": context.naming.records(),
        "objects": sorted(context.objects, key=lambda item: item["stable_id"]),
        "materials": sorted(context.materials, key=lambda item: item["stable_id"]),
        "camera": context.camera,
        "cameras": sorted(context.cameras, key=lambda item: item["camera_id"]),
        "lights": sorted(context.lights, key=lambda item: item["stable_id"]),
        "render": render_settings,
        "artifacts": sorted(context.artifacts, key=lambda item: item["path"]),
        "environment": {
            "blender_version": context.blender_version,
            "embedded_python_version": context.python_version,
            "execution_backend": context.backend,
        },
        "warnings": sorted(context.warnings),
    }
    return {
        "scientific": scientific,
        "scientific_digest": content_digest(scientific),
    }
