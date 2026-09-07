"""Install project OpenCV cameras into Blender with explicit axis conversion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from .context import GeneratorContext


def load_camera_collection(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load camera collection: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("camera collection is invalid")
    return value


def install_cameras(
    context: GeneratorContext, collection: dict[str, Any]
) -> list[bpy.types.Object]:
    """Create stable Blender cameras from OpenCV ``world_from_camera`` records."""

    installed: list[bpy.types.Object] = []
    metadata: list[dict[str, Any]] = []
    for record in collection["records"]:
        camera_id = str(record["camera_id"])
        matrix = tuple(float(item) for item in record["world_from_camera"])
        if len(matrix) != 16:
            raise ValueError(f"{camera_id}: transform must contain 16 values")
        name = context.naming.reserve("CAMERA", camera_id)
        data_name = context.naming.reserve("CAMERA_DATA", camera_id + "-data")
        data = bpy.data.cameras.new(data_name)
        obj = bpy.data.objects.new(name, data)
        context.collections["CAMERAS"].objects.link(obj)
        obj["gaussweave_id"] = camera_id
        data["gaussweave_id"] = camera_id + "-data"
        obj.matrix_world = opencv_pose_to_blender(matrix)
        width = int(record["width"])
        height = int(record["height"])
        fx = float(record["fx"])
        fy = float(record["fy"])
        cx = float(record["cx"])
        cy = float(record["cy"])
        if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
            raise ValueError(f"{camera_id}: invalid camera intrinsics")
        data.type = "PERSP"
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36.0
        data.lens = fx * data.sensor_width / width
        data.shift_x = 0.5 - cx / width
        data.shift_y = (cy - height / 2.0) / width
        data.clip_start = float(record["near_m"])
        data.clip_end = float(record["far_m"])
        installed.append(obj)
        metadata.append(
            {
                "camera_id": camera_id,
                "name": name,
                "data_name": data_name,
                "split": str(record["split"]),
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "near_m": data.clip_start,
                "far_m": data.clip_end,
                "opencv_to_blender_axes": {
                    "right": "+X to +X",
                    "down": "+Y to -Y",
                    "forward": "+Z to -Z",
                },
            }
        )
    if not installed:
        raise ValueError("camera collection is empty")
    context.cameras.extend(metadata)
    context.camera = {
        "kind": "generated-camera-collection",
        "count": len(metadata),
        "scientific_digest": str(collection["scientific_digest"]),
        "active_camera_id": metadata[0]["camera_id"],
        "records": metadata,
    }
    bpy.context.scene.camera = installed[0]
    return installed


def opencv_pose_to_blender(matrix: tuple[float, ...]) -> Matrix:
    """Convert OpenCV local axes to Blender camera local +X/+Y/-Z axes."""

    right = Vector((matrix[0], matrix[4], matrix[8]))
    down = Vector((matrix[1], matrix[5], matrix[9]))
    forward = Vector((matrix[2], matrix[6], matrix[10]))
    position = Vector((matrix[3], matrix[7], matrix[11]))
    return Matrix(
        (
            (right.x, -down.x, -forward.x, position.x),
            (right.y, -down.y, -forward.y, position.y),
            (right.z, -down.z, -forward.z, position.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def projection_agreement(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    record: dict[str, Any],
    point: tuple[float, float, float],
) -> dict[str, float]:
    """Compare project pinhole projection with Blender's camera projection."""

    matrix = tuple(float(item) for item in record["world_from_camera"])
    position = Vector((matrix[3], matrix[7], matrix[11]))
    right = Vector((matrix[0], matrix[4], matrix[8]))
    down = Vector((matrix[1], matrix[5], matrix[9]))
    forward = Vector((matrix[2], matrix[6], matrix[10]))
    relative = Vector(point) - position
    depth = relative.dot(forward)
    if depth <= 0:
        raise ValueError("projection agreement point is behind camera")
    expected_x = float(record["fx"]) * relative.dot(right) / depth + float(record["cx"])
    expected_y = float(record["fy"]) * relative.dot(down) / depth + float(record["cy"])
    coordinate = world_to_camera_view(scene, camera, Vector(point))
    blender_x = coordinate.x * int(record["width"])
    blender_y = (1.0 - coordinate.y) * int(record["height"])
    error = math.hypot(blender_x - expected_x, blender_y - expected_y)
    return {
        "expected_x_px": expected_x,
        "expected_y_px": expected_y,
        "blender_x_px": blender_x,
        "blender_y_px": blender_y,
        "error_px": error,
    }
