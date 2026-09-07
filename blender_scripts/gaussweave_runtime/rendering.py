"""Common deterministic RGB render configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy

from .context import GeneratorContext


class RenderError(ValueError):
    """A requested render engine or device cannot be configured."""


def configure_render(
    context: GeneratorContext,
    scene: bpy.types.Scene,
    output: Path,
    *,
    engine_override: str | None = None,
) -> dict[str, Any]:
    config = context.config["render"]
    requested_engine = str(config["engine"])
    engine = engine_override or requested_engine
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        compute = _cycles_device(context.render_device)
        scene.cycles.device = (
            "GPU" if compute["backend"] not in {"CPU", "UNSUPPORTED"} else "CPU"
        )
        scene.cycles.samples = int(config["samples"])
        scene.cycles.use_denoising = bool(config["denoise"])
        scene.cycles.use_animated_seed = False
        scene.cycles.seed = int(context.seeds.seeds["geometry"] & 0x7FFFFFFF)
        if compute["backend"] == "UNSUPPORTED":
            raise RenderError("requested GPU device is unavailable")
    elif engine == "eevee_next":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        compute = {
            "requested": context.render_device,
            "backend": "EEVEE",
            "device_name": "Blender graphics backend",
        }
        scene.render.image_settings.file_format = "PNG"
    else:
        raise RenderError(f"unsupported render engine: {engine}")
    scene.render.resolution_x = int(config["width"])
    scene.render.resolution_y = int(config["height"])
    scene.render.resolution_percentage = int(config["resolution_percentage"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = str(
        int(context.config["outputs"]["rgb_bit_depth"])
    )
    scene.render.image_settings.compression = 15
    scene.render.filepath = str(output)
    scene.view_settings.view_transform = str(config["view_transform"])
    look = str(config.get("look", ""))
    if look:
        scene.view_settings.look = look
    scene.view_settings.exposure = float(config["exposure"])
    scene.view_settings.gamma = float(config["gamma"])
    background = config["background"]["rgba"]
    world = scene.world
    if world is None or not world.use_nodes:
        raise RenderError("runtime world is unavailable")
    background_node = world.node_tree.nodes.get("Background")
    if background_node is None:
        raise RenderError("world background node is unavailable")
    background_node.inputs["Color"].default_value = tuple(float(v) for v in background)
    background_node.inputs["Strength"].default_value = 1.0
    context.compute = compute
    return {
        "engine": scene.render.engine,
        "requested_engine": requested_engine,
        "engine_override": engine_override,
        "device": compute,
        "resolution": {
            "width": scene.render.resolution_x,
            "height": scene.render.resolution_y,
            "percentage": scene.render.resolution_percentage,
        },
        "samples": int(config["samples"]),
        "denoise": bool(config["denoise"]),
        "image": {
            "format": "PNG",
            "color_mode": "RGB",
            "bit_depth": int(context.config["outputs"]["rgb_bit_depth"]),
        },
        "color_management": {
            "color_space": str(config["color_space"]),
            "view_transform": scene.view_settings.view_transform,
            "look": scene.view_settings.look,
            "exposure": scene.view_settings.exposure,
            "gamma": scene.view_settings.gamma,
        },
        "background": dict(config["background"]),
        "frame": scene.frame_current,
    }


def _cycles_device(requested: str) -> dict[str, str | None]:
    if requested == "cpu":
        return {"requested": requested, "backend": "CPU", "device_name": "CPU"}
    bpy.ops.preferences.addon_enable(module="cycles")
    preferences = bpy.context.preferences.addons["cycles"].preferences
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
        return {
            "requested": requested,
            "backend": backend,
            "device_name": devices[0].name,
        }
    return {"requested": requested, "backend": "UNSUPPORTED", "device_name": None}
