"""Validation of the completed runtime-probe state."""

from __future__ import annotations

import math
from typing import Any

import bpy

from . import COLLECTION_ROLES, REQUIRED_SEED_STREAMS
from .context import GeneratorContext
from .io import artifact_path, portable_path


class RuntimeValidationError(RuntimeError):
    """One or more generated-state invariants failed."""


def validate_runtime(
    context: GeneratorContext,
    *,
    render_settings: dict[str, Any],
    required_artifacts: list[str],
) -> None:
    issues: list[str] = []
    if tuple(context.collections) != COLLECTION_ROLES:
        issues.append("required collection hierarchy is incomplete or reordered")
    for role in COLLECTION_ROLES:
        if bpy.data.collections.get(role) is None:
            issues.append(f"missing collection: {role}")
    object_names = [obj.name for obj in bpy.data.objects]
    if len(object_names) != len(set(object_names)):
        issues.append("object names are not unique")
    stable_ids = [
        str(obj["gaussweave_id"]) for obj in bpy.data.objects if "gaussweave_id" in obj
    ]
    if len(stable_ids) != len(set(stable_ids)):
        issues.append("object stable IDs are not unique")
    for obj in bpy.data.objects:
        values = [
            *(float(value) for value in obj.location),
            *(float(value) for value in obj.rotation_euler),
            *(float(value) for value in obj.scale),
        ]
        if not all(math.isfinite(value) for value in values):
            issues.append(f"nonfinite transform: {obj.name}")
        if obj.type == "MESH" and any(
            not math.isfinite(float(value)) or value <= 0 for value in obj.dimensions
        ):
            issues.append(f"invalid dimensions: {obj.name}")
    if bpy.context.scene.camera is None or context.camera is None:
        issues.append("runtime camera is missing")
    if not context.lights:
        issues.append("runtime lights are missing")
    roles = {record["role"] for record in context.objects}
    for role in (
        "structure",
        "terminal-placeholder",
        "instance-placeholder",
        "annotation-helper",
    ):
        if role not in roles:
            issues.append(f"missing expected probe role: {role}")
    if (
        len(
            [
                record
                for record in context.objects
                if record["role"] == "instance-placeholder"
            ]
        )
        < 2
    ):
        issues.append("fewer than two instance placeholders")
    expected = context.config["render"]
    resolution = render_settings["resolution"]
    if (
        resolution["width"] != expected["width"]
        or resolution["height"] != expected["height"]
        or resolution["percentage"] != expected["resolution_percentage"]
    ):
        issues.append("render dimensions do not match resolved configuration")
    if set(context.seeds.seeds) != set(REQUIRED_SEED_STREAMS):
        issues.append("seed streams do not match the accepted registry")
    if context.seeds.seeds != context.config["derived_seeds"]:
        issues.append("seed values differ from the resolved configuration")
    for record in context.objects:
        bounds = record.get("bounds")
        if bounds is None:
            continue
        for prefix in ("local", "world"):
            minimum = bounds[f"{prefix}_min_m"]
            maximum = bounds[f"{prefix}_max_m"]
            if any(a > b for a, b in zip(minimum, maximum, strict=True)):
                issues.append(f"invalid {prefix} bounds: {record['name']}")
    for portable in required_artifacts:
        try:
            portable_path(portable)
            target = artifact_path(context.output_root, portable)
            if not target.is_file() or target.stat().st_size <= 0:
                issues.append(f"missing or empty output: {portable}")
        except (OSError, ValueError) as error:
            issues.append(f"unsafe artifact {portable}: {error}")
    if issues:
        raise RuntimeValidationError("; ".join(issues))
