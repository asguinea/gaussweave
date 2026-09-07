"""Generic runtime-probe fixture assembled only from shared helpers."""

from __future__ import annotations

import math
from typing import Any

import bpy
from mathutils import Vector

from .context import GeneratorContext
from .geometry import (
    apply_transform,
    create_annotation_empty,
    create_box,
    create_cylinder,
    object_record,
)
from .materials import assign_material, create_principled_material


def build_runtime_probe(
    context: GeneratorContext, *, include_camera: bool = True
) -> None:
    """Create a tiny engineering fixture; this is not a family scene."""

    geometry_rng = context.seeds.python("geometry")
    material_rng = context.seeds.python("materials")
    lighting_rng = context.seeds.python("lighting")
    base_color = (
        round(0.25 + material_rng.random() * 0.25, 12),
        0.38,
        0.62,
        1.0,
    )
    accent_color = (
        0.72,
        round(0.25 + material_rng.random() * 0.2, 12),
        0.18,
        1.0,
    )
    base = create_principled_material(
        context,
        stable_id="probe-base",
        base_color=base_color,
        roughness=0.55,
        metallic=0.0,
    )
    reused_base = create_principled_material(
        context,
        stable_id="probe-base",
        base_color=base_color,
        roughness=0.55,
        metallic=0.0,
    )
    if reused_base is not base:
        raise RuntimeError("identical material reuse returned a different data-block")
    accent = create_principled_material(
        context,
        stable_id="probe-accent",
        base_color=accent_color,
        roughness=0.35,
        metallic=0.05,
    )
    root = create_box(
        context,
        stable_id="probe-structure",
        dimensions_m=(4.8, 0.35, 2.6),
        collection_role="STRUCTURE",
    )
    apply_transform(root, location_m=(0.0, 0.45, 1.3))
    assign_material(root, base)
    terminal = create_box(
        context,
        stable_id="terminal-probe-a",
        dimensions_m=(0.75, 0.24, 1.1),
        collection_role="TERMINALS",
    )
    apply_transform(terminal, location_m=(-1.3, 0.18, 1.25))
    assign_material(terminal, accent)
    instances = []
    for index, x in enumerate((-0.35, 0.85), start=1):
        instance = create_cylinder(
            context,
            stable_id=f"instance-probe-{index:02d}",
            radius_m=0.28,
            height_m=1.15,
            vertices=16,
            collection_role="INSTANCES",
        )
        jitter = round((geometry_rng.random() - 0.5) * 0.12, 12)
        apply_transform(
            instance,
            location_m=(x + jitter, -0.05, 0.575),
            rotation_euler_radians=(0.0, 0.0, math.radians(index * 7.0)),
        )
        assign_material(instance, accent if index == 1 else base)
        instances.append(instance)
    annotation = create_annotation_empty(
        context, stable_id="annotation-probe-bounds", location_m=(0.0, 0.0, 1.3)
    )
    camera = (
        _create_camera(context, context.seeds.python("cameras"))
        if include_camera
        else None
    )
    lights = _create_lights(context, lighting_rng)
    context.objects.extend(
        [
            object_record(root, "structure"),
            object_record(terminal, "terminal-placeholder"),
            *(object_record(obj, "instance-placeholder") for obj in instances),
            object_record(annotation, "annotation-helper"),
        ]
    )
    if camera is not None:
        context.camera = _camera_record(camera)
    context.lights.extend(_light_record(light) for light in lights)


def _create_camera(context: GeneratorContext, rng: Any) -> bpy.types.Object:
    stable_id = "camera-runtime-probe"
    name = context.naming.reserve("CAMERA", stable_id)
    data_name = context.naming.reserve("CAMERA_DATA", stable_id + "-data")
    camera_data = bpy.data.cameras.new(data_name)
    camera = bpy.data.objects.new(name, camera_data)
    context.collections["CAMERAS"].objects.link(camera)
    camera["gaussweave_id"] = stable_id
    camera_data["gaussweave_id"] = stable_id + "-data"
    camera.location = (
        round(4.6 + rng.random() * 0.2, 12),
        -6.4,
        3.4,
    )
    camera_data.lens = 50.0
    camera_data.sensor_width = 36.0
    camera_data.clip_start = float(context.config["camera"]["near_m"])
    camera_data.clip_end = float(context.config["camera"]["far_m"])
    _look_at(camera, (0.0, 0.0, 1.2))
    bpy.context.scene.camera = camera
    return camera


def _create_lights(context: GeneratorContext, rng: Any) -> list[bpy.types.Object]:
    records = []
    for stable_id, location, energy in (
        ("light-probe-key", (-3.5, -4.0, 5.5), 700.0),
        ("light-probe-fill", (3.0, -1.5, 3.0), 240.0),
    ):
        name = context.naming.reserve("LIGHT", stable_id)
        data_name = context.naming.reserve("LIGHT_DATA", stable_id + "-data")
        data = bpy.data.lights.new(data_name, type="AREA")
        data.energy = energy * (0.95 + 0.1 * rng.random())
        data.color = (1.0, 0.94, 0.86)
        data.shape = "DISK"
        data.size = 2.0
        obj = bpy.data.objects.new(name, data)
        context.collections["LIGHTING"].objects.link(obj)
        obj.location = location
        obj["gaussweave_id"] = stable_id
        data["gaussweave_id"] = stable_id + "-data"
        _look_at(obj, (0.0, 0.0, 1.0))
        records.append(obj)
    return records


def _look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _camera_record(camera: bpy.types.Object) -> dict[str, Any]:
    return {
        "stable_id": str(camera["gaussweave_id"]),
        "name": camera.name,
        "data_name": camera.data.name,
        "location_m": [round(float(value), 12) for value in camera.location],
        "rotation_euler_radians": [
            round(float(value), 12) for value in camera.rotation_euler
        ],
        "lens_mm": float(camera.data.lens),
        "sensor_width_mm": float(camera.data.sensor_width),
        "near_m": float(camera.data.clip_start),
        "far_m": float(camera.data.clip_end),
        "axes": {"right": "+X", "up": "+Y", "forward": "-Z"},
    }


def _light_record(light: bpy.types.Object) -> dict[str, Any]:
    return {
        "stable_id": str(light["gaussweave_id"]),
        "name": light.name,
        "data_name": light.data.name,
        "type": light.data.type,
        "energy": round(float(light.data.energy), 12),
        "location_m": [round(float(value), 12) for value in light.location],
    }
