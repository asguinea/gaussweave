"""Generic metric geometry helpers for Blender generators."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import bpy
from mathutils import Matrix, Vector

from .context import GeneratorContext
from .specs import SpecificationError, validate_positive_finite


class GeometryError(ValueError):
    """Geometry dimensions or topology are invalid."""


def validate_dimensions(values: Sequence[float], *, label: str = "dimensions") -> None:
    try:
        validate_positive_finite(values, label=label)
    except SpecificationError as error:
        raise GeometryError(str(error)) from error


def create_mesh(
    context: GeneratorContext,
    *,
    stable_id: str,
    vertices: Iterable[Sequence[float]],
    faces: Iterable[Sequence[int]],
    collection_role: str,
) -> bpy.types.Object:
    vertex_list = [tuple(float(value) for value in vertex) for vertex in vertices]
    face_list = [tuple(int(value) for value in face) for face in faces]
    if len(vertex_list) < 3 or not face_list:
        raise GeometryError("mesh must contain vertices and faces")
    if any(
        len(vertex) != 3 or not all(math.isfinite(v) for v in vertex)
        for vertex in vertex_list
    ):
        raise GeometryError("mesh vertices must be finite XYZ triples")
    if any(
        len(face) < 3 or min(face) < 0 or max(face) >= len(vertex_list)
        for face in face_list
    ):
        raise GeometryError("mesh faces must reference valid vertices")
    object_name = context.naming.reserve("OBJ", stable_id)
    mesh_name = context.naming.reserve("MESH", stable_id + "-mesh")
    if bpy.data.objects.get(object_name) or bpy.data.meshes.get(mesh_name):
        raise GeometryError(f"Blender data-block already exists: {stable_id}")
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(vertex_list, [], face_list)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(object_name, mesh)
    context.collections[collection_role].objects.link(obj)
    _attach_identity(obj, stable_id, "geometry")
    return obj


def create_box(
    context: GeneratorContext,
    *,
    stable_id: str,
    dimensions_m: Sequence[float],
    collection_role: str,
) -> bpy.types.Object:
    validate_dimensions(dimensions_m)
    x, y, z = (float(value) / 2.0 for value in dimensions_m)
    vertices = (
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    )
    faces = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (4, 0, 3, 7),
    )
    obj = create_mesh(
        context,
        stable_id=stable_id,
        vertices=vertices,
        faces=faces,
        collection_role=collection_role,
    )
    obj["gaussweave_dimensions_m"] = [float(value) for value in dimensions_m]
    return obj


def create_cylinder(
    context: GeneratorContext,
    *,
    stable_id: str,
    radius_m: float,
    height_m: float,
    vertices: int,
    collection_role: str,
) -> bpy.types.Object:
    validate_dimensions((radius_m, height_m))
    if not isinstance(vertices, int) or isinstance(vertices, bool) or vertices < 3:
        raise GeometryError("cylinder vertices must be an integer >= 3")
    half = float(height_m) / 2.0
    points = []
    for z in (-half, half):
        points.extend(
            (
                float(radius_m) * math.cos(2.0 * math.pi * index / vertices),
                float(radius_m) * math.sin(2.0 * math.pi * index / vertices),
                z,
            )
            for index in range(vertices)
        )
    faces: list[tuple[int, ...]] = []
    faces.append(tuple(range(vertices - 1, -1, -1)))
    faces.append(tuple(range(vertices, vertices * 2)))
    for index in range(vertices):
        next_index = (index + 1) % vertices
        faces.append((index, next_index, vertices + next_index, vertices + index))
    obj = create_mesh(
        context,
        stable_id=stable_id,
        vertices=points,
        faces=faces,
        collection_role=collection_role,
    )
    obj["gaussweave_dimensions_m"] = [
        float(radius_m) * 2.0,
        float(radius_m) * 2.0,
        float(height_m),
    ]
    return obj


def apply_transform(
    obj: bpy.types.Object,
    *,
    location_m: Sequence[float],
    rotation_euler_radians: Sequence[float] = (0.0, 0.0, 0.0),
    scale: Sequence[float] = (1.0, 1.0, 1.0),
) -> None:
    if any(len(values) != 3 for values in (location_m, rotation_euler_radians, scale)):
        raise GeometryError("transform vectors must contain three values")
    values = [
        float(value)
        for group in (location_m, rotation_euler_radians, scale)
        for value in group
    ]
    if not all(math.isfinite(value) for value in values):
        raise GeometryError("transform values must be finite")
    validate_dimensions(tuple(float(value) for value in scale), label="scale")
    obj.location = tuple(float(value) for value in location_m)
    obj.rotation_euler = tuple(float(value) for value in rotation_euler_radians)
    obj.scale = tuple(float(value) for value in scale)


def set_origin_identity(obj: bpy.types.Object) -> None:
    """Make mesh coordinates explicit in object local coordinates."""

    if obj.type == "MESH":
        obj.data.transform(Matrix.Identity(4))


def bounds_record(obj: bpy.types.Object) -> dict[str, list[float]]:
    local_points = [tuple(float(value) for value in corner) for corner in obj.bound_box]
    world_points = [
        tuple(float(value) for value in (obj.matrix_world @ Vector(corner)))
        for corner in obj.bound_box
    ]
    return {
        "local_min_m": [
            round(min(point[index] for point in local_points), 12) for index in range(3)
        ],
        "local_max_m": [
            round(max(point[index] for point in local_points), 12) for index in range(3)
        ],
        "world_min_m": [
            round(min(point[index] for point in world_points), 12) for index in range(3)
        ],
        "world_max_m": [
            round(max(point[index] for point in world_points), 12) for index in range(3)
        ],
    }


def create_annotation_empty(
    context: GeneratorContext,
    *,
    stable_id: str,
    location_m: Sequence[float],
) -> bpy.types.Object:
    name = context.naming.reserve("ANNOTATION", stable_id)
    if bpy.data.objects.get(name):
        raise GeometryError(f"Blender data-block already exists: {name}")
    obj = bpy.data.objects.new(name, None)
    context.collections["ANNOTATIONS"].objects.link(obj)
    obj.empty_display_type = "CUBE"
    obj.empty_display_size = 0.25
    apply_transform(obj, location_m=location_m)
    _attach_identity(obj, stable_id, "annotation")
    return obj


def object_record(obj: bpy.types.Object, role: str) -> dict[str, Any]:
    dimensions = [round(float(value), 12) for value in obj.dimensions]
    return {
        "stable_id": str(obj["gaussweave_id"]),
        "name": obj.name,
        "data_name": obj.data.name if obj.data is not None else None,
        "role": role,
        "type": obj.type,
        "location_m": [round(float(value), 12) for value in obj.location],
        "rotation_euler_radians": [
            round(float(value), 12) for value in obj.rotation_euler
        ],
        "scale": [round(float(value), 12) for value in obj.scale],
        "dimensions_m": dimensions,
        "bounds": bounds_record(obj) if obj.type == "MESH" else None,
        "material_ids": [
            str(material.get("gaussweave_id", material.name))
            for material in (
                obj.data.materials
                if obj.data is not None and hasattr(obj.data, "materials")
                else []
            )
        ],
    }


def _attach_identity(obj: bpy.types.Object, stable_id: str, role: str) -> None:
    obj["gaussweave_id"] = stable_id
    obj["gaussweave_role"] = role
    if obj.data is not None:
        obj.data["gaussweave_id"] = stable_id + "-data"
