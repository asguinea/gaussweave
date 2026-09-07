"""Generate one tiny deterministic scene inside Blender."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _arguments() -> argparse.Namespace:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    return parser.parse_args(sys.argv[separator + 1 :])


def _look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _cycles_device(requested: str) -> tuple[str, str | None, list[str]]:
    warnings: list[str] = []
    bpy.ops.preferences.addon_enable(module="cycles")
    preferences = bpy.context.preferences.addons["cycles"].preferences
    if requested == "cpu":
        return "CPU", "CPU", warnings
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
        except (TypeError, ValueError):
            continue
        devices = [
            device
            for device in preferences.devices
            if device.type == backend and device.name
        ]
        if not devices:
            continue
        for device in preferences.devices:
            device.use = device in devices
        return backend, devices[0].name, warnings
    warnings.append("No compatible Cycles CUDA or OptiX device was detected.")
    return "UNSUPPORTED", None, warnings


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    options = _arguments()
    output = Path(options.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(options.seed)
    seeded_value = round(random.Random(options.seed).uniform(0.2, 0.8), 12)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in tuple(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)
    default_collection = bpy.data.collections.get("Collection")
    if default_collection is not None:
        default_collection.name = "QualificationCollection"
    collection = bpy.data.collections["QualificationCollection"]

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.5))
    cube = bpy.context.object
    cube.name = "QualificationCube"
    cube.data.name = "QualificationCubeMesh"
    cube.scale = (0.75, 0.75, 0.75)
    cube.rotation_euler[2] = math.radians(15.0)
    material = bpy.data.materials.new("QualificationMaterial")
    material.diffuse_color = (seeded_value, 0.32, 0.18, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = material.diffuse_color
        principled.inputs["Roughness"].default_value = 0.42
    cube.data.materials.append(material)

    bpy.ops.object.light_add(type="AREA", location=(2.5, -2.0, 3.5))
    light = bpy.context.object
    light.name = "QualificationKeyLight"
    light.data.name = "QualificationKeyLightData"
    light.data.energy = 650.0
    light.data.shape = "DISK"
    light.data.size = 2.0
    _look_at(light, (0.0, 0.0, 0.5))

    bpy.ops.object.camera_add(location=(3.2, -3.2, 2.6))
    camera = bpy.context.object
    camera.name = "QualificationCamera"
    camera.data.name = "QualificationCameraData"
    camera.data.lens = 50.0
    camera.data.sensor_width = 36.0
    _look_at(camera, (0.0, 0.0, 0.5))

    scene = bpy.context.scene
    scene.name = "QualificationScene"
    scene.camera = camera
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.cycles.seed = options.seed
    scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.filepath = str(output / "render.png")
    backend, device_name, warnings = _cycles_device(options.device)
    status = "unsupported" if backend == "UNSUPPORTED" else "qualified"
    if options.device == "gpu" and status == "qualified":
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"

    metadata = {
        "schema_version": "1.0",
        "status": status,
        "generator": {
            "script": "blender_scripts/qualify_blender.py",
            "purpose": "environment qualification only; not benchmark data",
        },
        "seed": options.seed,
        "seeded_value": seeded_value,
        "blender": {
            "version": bpy.app.version_string,
            "version_tuple": list(bpy.app.version),
            "build_hash": bpy.app.build_hash.decode("ascii"),
            "python": sys.version,
        },
        "coordinates": {
            "handedness": "right-handed",
            "up_axis": "+Z",
            "units": "meters",
            "blender_units_per_meter": 1.0,
            "camera_convention": "+X right, +Y up, -Z forward",
        },
        "collections": [collection.name],
        "objects": {
            "camera": [camera.name],
            "lights": [light.name],
            "geometry": [cube.name],
        },
        "materials": [material.name],
        "camera": {
            "name": camera.name,
            "location": [round(value, 12) for value in camera.location],
            "rotation_euler": [round(value, 12) for value in camera.rotation_euler],
            "lens_mm": camera.data.lens,
            "sensor_width_mm": camera.data.sensor_width,
        },
        "render": {
            "engine": scene.render.engine,
            "resolution": {"width": 64, "height": 64, "percentage": 100},
            "samples": scene.cycles.samples,
            "image_format": "PNG",
            "color_mode": "RGB",
            "color_depth_bits": 8,
            "color_management": {
                "view_transform": scene.view_settings.view_transform,
                "look": scene.view_settings.look,
                "exposure": scene.view_settings.exposure,
                "gamma": scene.view_settings.gamma,
            },
        },
        "compute": {
            "requested": options.device,
            "backend": backend,
            "device_name": device_name,
        },
        "outputs": {
            "blend": "scene.blend",
            "image": "render.png" if status == "qualified" else None,
            "metadata": "generation-metadata.json",
        },
        "warnings": warnings,
    }
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))
    if status == "qualified":
        bpy.ops.render.render(write_still=True)
    _write_json(output / "generation-metadata.json", metadata)


if __name__ == "__main__":
    main()
