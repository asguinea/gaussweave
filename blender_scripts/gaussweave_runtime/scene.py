"""Clean Blender scene initialization and stable collection hierarchy."""

from __future__ import annotations

from typing import Any

import bpy

from . import COLLECTION_ROLES
from .context import GeneratorContext


def reset_scene(context: GeneratorContext) -> bpy.types.Scene:
    """Remove startup/generated state and create the required hierarchy."""

    scene = bpy.context.scene
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in tuple(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for data_collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.worlds,
    ):
        for block in tuple(data_collection):
            data_collection.remove(block)
    scene.name = "GaussWeaveRuntimeScene"
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_set(1)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    scene.render.film_transparent = False
    root = bpy.data.collections.new(COLLECTION_ROLES[0])
    scene.collection.children.link(root)
    context.collections[COLLECTION_ROLES[0]] = root
    for role in COLLECTION_ROLES[1:]:
        collection = bpy.data.collections.new(role)
        root.children.link(collection)
        context.collections[role] = collection
    world = bpy.data.worlds.new("SS_WORLD_RUNTIME")
    world.use_nodes = True
    scene.world = world
    return scene


def collection_hierarchy(context: GeneratorContext) -> list[dict[str, Any]]:
    """Return stable portable collection hierarchy records."""

    root = context.collections[COLLECTION_ROLES[0]]
    return [
        {
            "name": role,
            "parent": None if role == COLLECTION_ROLES[0] else root.name,
        }
        for role in COLLECTION_ROLES
    ]
