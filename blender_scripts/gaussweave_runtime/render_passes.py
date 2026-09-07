"""Aligned engineering render passes for the headless Blender probe."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import zlib
from array import array
from pathlib import Path
from typing import Any

import bpy

from .context import GeneratorContext
from .geometry import apply_transform, create_box, object_record
from .io import artifact_path, content_digest, file_digest, write_json
from .materials import assign_material, create_principled_material
from .probe import build_runtime_probe
from .rendering import configure_render

PASS_EXTENSIONS = {
    "rgb": "png",
    "depth": "npy",
    "normals": "npy",
    "semantic": "png",
    "terminal": "png",
    "instance": "png",
    "alpha": "png",
    "object_id": "png",
    "material_id": "png",
}


def load_render_request(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load render request: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("policy"), dict):
        raise ValueError("render request is invalid")
    return value


def build_render_pass_probe(
    context: GeneratorContext, request: dict[str, Any]
) -> dict[str, Any]:
    """Build the render-pass fixture and add a probe-only floor and occluder."""

    build_runtime_probe(context, include_camera=False)
    registry = request["policy"]["registry"]
    objects = {
        str(obj["gaussweave_id"]): obj
        for obj in bpy.data.objects
        if "gaussweave_id" in obj
    }
    floor_material = create_principled_material(
        context,
        stable_id="probe-floor",
        base_color=(0.18, 0.22, 0.18, 1.0),
        roughness=0.7,
        metallic=0.0,
        provenance="render-pass-probe",
    )
    occluder_material = create_principled_material(
        context,
        stable_id="probe-occluder",
        base_color=(0.12, 0.6, 0.28, 1.0),
        roughness=0.45,
        metallic=0.0,
        provenance="render-pass-probe",
    )
    floor = create_box(
        context,
        stable_id="floor-probe",
        dimensions_m=(6.0, 6.0, 0.1),
        collection_role="STRUCTURE",
    )
    apply_transform(floor, location_m=(0.0, 0.2, -0.05))
    assign_material(floor, floor_material)
    occluder = create_box(
        context,
        stable_id="occluder-probe",
        dimensions_m=(0.55, 0.35, 1.35),
        collection_role="OCCLUDERS",
    )
    apply_transform(occluder, location_m=(0.1, -1.15, 0.675))
    assign_material(occluder, occluder_material)
    context.objects.extend(
        [
            object_record(floor, "floor"),
            object_record(occluder, "partial-occluder"),
        ]
    )
    objects["floor-probe"] = floor
    objects["occluder-probe"] = occluder
    corridor_oriented = str(context.config["camera"]["trajectory"]) == "corridor_path"
    if corridor_oriented:
        for obj in objects.values():
            x, y, z = (float(value) for value in obj.location)
            obj.location = (5.0 + y, -x, z)
            obj.rotation_euler.z -= math.pi / 2.0
        bpy.context.view_layer.update()
    roles = {
        "probe-structure": ("structure", None, None),
        "terminal-probe-a": ("structure", None, None),
        "instance-probe-01": ("terminal", 1, 1),
        "instance-probe-02": ("terminal", 1, 2),
        "floor-probe": ("floor", None, None),
        "occluder-probe": ("occluder", None, None),
    }
    semantic = registry["semantic_categories"]
    object_ids = registry["object_ids"]
    for stable_id, (category, terminal_id, instance_id) in roles.items():
        obj = objects.get(stable_id)
        if obj is None:
            raise RuntimeError(f"probe object is missing: {stable_id}")
        obj["semantic_category"] = category
        obj["semantic_id"] = int(semantic[category])
        obj["terminal_id"] = int(terminal_id or 0)
        obj["instance_id"] = int(instance_id or 0)
        obj["project_object_id"] = int(object_ids[stable_id])
    material_ids = registry["material_ids"]
    for material in bpy.data.materials:
        stable_id = str(material.get("gaussweave_id", ""))
        if stable_id not in material_ids:
            raise RuntimeError(f"probe material is unregistered: {stable_id}")
        material["project_material_id"] = int(material_ids[stable_id])
        material.pass_index = int(material_ids[stable_id])
    wall_point = [5.275, 0.0, 0.0] if corridor_oriented else [0.0, 0.275, 0.0]
    wall_normal = [-1.0, 0.0, 0.0] if corridor_oriented else [0.0, -1.0, 0.0]
    return {
        "fixture": "generic-render-pass-engineering-probe",
        "benchmark_family": False,
        "camera_trajectory_adaptation": (
            "generic_fixture_rotated_for_longitudinal_camera_path"
            if corridor_oriented
            else "identity"
        ),
        "wall_front_point_m": wall_point,
        "wall_front_normal": wall_normal,
        "wall_bounds_m": (
            {"x": [5.275, 5.625], "y": [-2.4, 2.4], "z": [0.0, 2.6]}
            if corridor_oriented
            else {"x": [-2.4, 2.4], "y": [0.275, 0.625], "z": [0.0, 2.6]}
        ),
        "terminal_mapping": {"instance-probe-01": 1, "instance-probe-02": 1},
        "instance_ids": {"instance-probe-01": 1, "instance-probe-02": 2},
        "occluder": "occluder-probe",
    }


def render_pass_views(
    context: GeneratorContext,
    scene: bpy.types.Scene,
    request: dict[str, Any],
    cameras: list[bpy.types.Object],
) -> dict[str, Any]:
    """Render all selected cameras and serialize aligned pass metadata."""

    policy = request["policy"]
    enabled = tuple(str(item) for item in policy["enabled_passes"])
    records = {
        str(item["camera_id"]): item
        for item in request["_camera_collection"]["records"]
    }
    camera_objects = {str(camera["gaussweave_id"]): camera for camera in cameras}
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_object_index = True
    view_layer.use_pass_material_index = True
    scene.display_settings.display_device = str(
        policy["color_management"]["display_device"]
    )
    scene.sequencer_colorspace_settings.name = str(
        policy["color_management"]["sequencer_space"]
    )
    scene.render.film_transparent = "alpha" in enabled
    first_request = request["views"][0]
    first_rgb = artifact_path(
        context.output_root,
        f"renders/rgb/{first_request['camera_id']}.png",
    )
    requested_engine = str(context.config["render"]["engine"])
    engine_override = "cycles" if requested_engine == "eevee_next" else None
    render_settings = configure_render(
        context,
        scene,
        first_rgb,
        engine_override=engine_override,
    )
    render_settings["display_device"] = scene.display_settings.display_device
    render_settings["sequencer_space"] = scene.sequencer_colorspace_settings.name
    render_settings["film_transparent"] = scene.render.film_transparent
    render_settings["pass_policy_version"] = str(policy["policy_version"])
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"gaussweave-render-passes-{context.scene_id}-")
    )
    views: list[dict[str, Any]] = []
    all_statistics: list[dict[str, Any]] = []
    for view_request in request["views"]:
        camera_id = str(view_request["camera_id"])
        camera = camera_objects.get(camera_id)
        record = records.get(camera_id)
        if camera is None or record is None:
            raise RuntimeError(f"requested camera is unavailable: {camera_id}")
        scene.camera = camera
        if context.camera is not None:
            context.camera["active_camera_id"] = camera_id
        rgb_path = artifact_path(context.output_root, f"renders/rgb/{camera_id}.png")
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(rgb_path)
        base_outputs = _configure_pass_outputs(
            scene,
            staging_root,
            camera_id,
            ("Combined", "Depth", "Normal"),
        )
        bpy.ops.render.render(write_still=True)
        artifacts: list[dict[str, Any]] = []
        if "rgb" in enabled:
            artifacts.append(
                _png_artifact(
                    context,
                    "rgb",
                    f"renders/rgb/{camera_id}.png",
                    int(record["width"]),
                    int(record["height"]),
                    required=True,
                )
            )
        base_passes = _extract_render_passes(
            base_outputs,
            int(record["width"]),
            int(record["height"]),
        )
        depth = _clean_depth(
            base_passes["Depth"],
            float(record["near_m"]),
            float(record["far_m"]),
            int(record["width"]),
            int(record["height"]),
            float(record["fx"]),
            float(record["fy"]),
            float(record["cx"]),
            float(record["cy"]),
        )
        normals = _clean_normals(base_passes["Normal"], depth)
        if "depth" in enabled:
            artifacts.append(
                _write_float_artifact(
                    context,
                    "depth",
                    camera_id,
                    1,
                    int(record["width"]),
                    int(record["height"]),
                    depth,
                )
            )
        if "normals" in enabled:
            artifacts.append(
                _write_float_artifact(
                    context,
                    "normals",
                    camera_id,
                    3,
                    int(record["width"]),
                    int(record["height"]),
                    normals,
                )
            )
        if "alpha" in enabled:
            combined = base_passes["Combined"]
            alpha = [
                int(max(0.0, min(1.0, combined[index * 4 + 3])) * 65535.0 + 0.5)
                for index in range(int(record["width"]) * int(record["height"]))
            ]
            artifacts.append(
                _write_integer_artifact(
                    context,
                    "alpha",
                    camera_id,
                    int(record["width"]),
                    int(record["height"]),
                    alpha,
                )
            )
        requested_cycles_device = (
            str(scene.cycles.device) if scene.render.engine == "CYCLES" else None
        )
        if requested_cycles_device is not None:
            scene.cycles.device = "CPU"
        for domain in (
            "object_id",
            "material_id",
            "semantic",
            "terminal",
            "instance",
        ):
            if domain not in enabled:
                continue
            socket_name = "IndexMA" if domain == "material_id" else "IndexOB"
            if domain != "material_id":
                _assign_object_indices(domain)
            domain_outputs = _configure_pass_outputs(
                scene, staging_root, f"{camera_id}-{domain}", (socket_name,)
            )
            bpy.ops.render.render()
            passes = _extract_render_passes(
                domain_outputs,
                int(record["width"]),
                int(record["height"]),
            )
            artifacts.append(
                _write_integer_artifact(
                    context,
                    domain,
                    camera_id,
                    int(record["width"]),
                    int(record["height"]),
                    _index_values(passes[socket_name]),
                )
            )
        if requested_cycles_device is not None:
            scene.cycles.device = requested_cycles_device
        ordered = sorted(artifacts, key=lambda item: item["pass_name"])
        for artifact in ordered:
            context.register_artifact(
                f"{artifact['pass_name']}-{camera_id}",
                artifact["path"],
                "render",
            )
            all_statistics.append(_artifact_statistics(camera_id, artifact))
        view_scientific = {
            "scene_id": context.scene_id,
            "camera_id": camera_id,
            "split": str(view_request["split"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
            "camera_reference": "cameras/cameras.json",
            "render_engine": str(render_settings["engine"]),
            "compute_device": str(render_settings["device"]["requested"]),
            "sample_count": int(render_settings["samples"]),
            "pass_artifacts": [
                {**artifact, "file_digest": None} for artifact in ordered
            ],
            "pass_conventions_digest": str(policy["conventions"]["scientific_digest"]),
        }
        views.append(
            {
                **view_scientific,
                "pass_artifacts": ordered,
                "warnings": [],
                "scientific_digest": content_digest(view_scientific),
                "execution_reference": "logs/external-command.json",
            }
        )
    write_json(
        artifact_path(context.output_root, "metadata/render_conventions.json"),
        policy["conventions"],
    )
    context.register_artifact(
        "render-conventions", "metadata/render_conventions.json", "other"
    )
    write_json(
        artifact_path(context.output_root, "metadata/pass_statistics.json"),
        {
            "pass_statistics_schema_version": "1.0.0",
            "scene_id": context.scene_id,
            "statistics": all_statistics,
        },
    )
    context.register_artifact(
        "pass-statistics", "metadata/pass_statistics.json", "other"
    )
    shutil.rmtree(staging_root, ignore_errors=True)
    return {
        "render_settings": render_settings,
        "views": views,
        "statistics": all_statistics,
    }


def build_render_collection(
    context: GeneratorContext,
    request: dict[str, Any],
    probe_geometry: dict[str, Any],
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = request["policy"]
    scientific = {
        "render_collection_schema_version": "1.0.0",
        "scene_id": context.scene_id,
        "configuration_digest": str(request["configuration_digest"]),
        "camera_collection_digest": str(request["camera_collection_digest"]),
        "convention_digest": str(policy["conventions"]["scientific_digest"]),
        "policy_version": str(policy["policy_version"]),
        "enabled_passes": list(policy["enabled_passes"]),
        "registry": policy["registry"],
        "views": [
            {
                key: value
                for key, value in view.items()
                if key not in {"warnings", "execution_reference", "scientific_digest"}
            }
            for view in views
        ],
        "probe_geometry": probe_geometry,
    }
    # Match the project-side projection, which replaces file digests with null.
    for view in scientific["views"]:
        view["pass_artifacts"] = [
            {**artifact, "file_digest": None} for artifact in view["pass_artifacts"]
        ]
    return {
        **scientific,
        "views": views,
        "scientific_digest": content_digest(scientific),
        "execution_reference": "logs/external-command.json",
        "warnings": [],
    }


def _assign_object_indices(domain: str) -> None:
    property_name = {
        "semantic": "semantic_id",
        "terminal": "terminal_id",
        "instance": "instance_id",
        "object_id": "project_object_id",
    }[domain]
    for obj in bpy.data.objects:
        obj.pass_index = int(obj.get(property_name, 0))


def _configure_pass_outputs(
    scene: bpy.types.Scene,
    staging_root: Path,
    prefix: str,
    pass_names: tuple[str, ...],
) -> dict[str, Path]:
    scene.use_nodes = True
    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("Blender compositor node tree is unavailable")
    tree.nodes.clear()
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    paths: dict[str, Path] = {}
    for pass_name in pass_names:
        socket_name = "Image" if pass_name == "Combined" else pass_name
        socket = render_layers.outputs.get(socket_name)
        if socket is None:
            raise RuntimeError(f"Blender compositor pass is unavailable: {pass_name}")
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.base_path = str(staging_root)
        output.file_slots[0].path = f"{prefix}-{pass_name.lower()}-"
        output.format.file_format = "OPEN_EXR"
        output.format.color_depth = "32"
        output.format.exr_codec = "ZIP"
        output.format.color_mode = (
            "RGBA"
            if pass_name == "Combined"
            else "RGB"
            if pass_name == "Normal"
            else "BW"
        )
        tree.links.new(socket, output.inputs[0])
        paths[pass_name] = (
            staging_root / f"{prefix}-{pass_name.lower()}-{scene.frame_current:04d}.exr"
        )
    return paths


def _extract_render_passes(
    paths: dict[str, Path], width: int, height: int
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, path in paths.items():
        channels = 3 if name == "Normal" else 4 if name == "Combined" else 1
        result[name] = _load_exr_pixels(path, width, height, channels)
    return result


def _load_exr_pixels(
    path: Path, width: int, height: int, selected_channels: int
) -> list[float]:
    if not path.is_file():
        raise RuntimeError(f"Blender compositor output is missing: {path.name}")
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        source_channels = int(image.channels)
        if tuple(int(value) for value in image.size) != (width, height):
            raise RuntimeError(
                f"{path.name} dimensions are {tuple(image.size)}; "
                f"expected {(width, height)}"
            )
        if source_channels < selected_channels:
            raise RuntimeError(
                f"{path.name} has {source_channels} channels; "
                f"expected at least {selected_channels}"
            )
        raw = list(image.pixels)
        selected = [
            float(raw[pixel * source_channels + channel])
            for pixel in range(width * height)
            for channel in range(selected_channels)
        ]
        return _bottom_to_top(selected, width, height, selected_channels)
    finally:
        bpy.data.images.remove(image)
        path.unlink(missing_ok=True)


def _bottom_to_top(
    values: list[float], width: int, height: int, channels: int
) -> list[float]:
    result: list[float] = []
    row_values = width * channels
    for row in range(height - 1, -1, -1):
        start = row * row_values
        result.extend(values[start : start + row_values])
    return result


def _clean_depth(
    values: list[float],
    near_m: float,
    far_m: float,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> list[float]:
    result = []
    for pixel, value in enumerate(values):
        row, column = divmod(pixel, width)
        ray_scale = math.sqrt(
            ((column + 0.5 - cx) / fx) ** 2 + ((row + 0.5 - cy) / fy) ** 2 + 1.0
        )
        ray_distance = float(value) * ray_scale
        result.append(
            ray_distance
            if math.isfinite(ray_distance)
            and near_m <= ray_distance <= far_m
            and row < height
            else 0.0
        )
    return result


def _clean_normals(values: list[float], depth: list[float]) -> list[float]:
    result: list[float] = []
    for pixel, valid_depth in enumerate(depth):
        vector = values[pixel * 3 : pixel * 3 + 3]
        length = math.sqrt(sum(value * value for value in vector))
        if valid_depth <= 0.0 or not math.isfinite(length) or length <= 1e-12:
            result.extend((0.0, 0.0, 0.0))
        else:
            result.extend(value / length for value in vector)
    return result


def _index_values(values: list[float]) -> list[int]:
    result = []
    for value in values:
        rounded = int(round(value))
        if not math.isfinite(value) or abs(value - rounded) > 1e-4:
            raise RuntimeError(f"interpolated/nonfinite index pass value: {value}")
        if not 0 <= rounded <= 65535:
            raise RuntimeError(f"index pass value exceeds uint16: {rounded}")
        result.append(rounded)
    return result


def _write_float_artifact(
    context: GeneratorContext,
    pass_name: str,
    camera_id: str,
    channels: int,
    width: int,
    height: int,
    values: list[float],
) -> dict[str, Any]:
    portable = f"renders/{pass_name}/{camera_id}.npy"
    path = artifact_path(context.output_root, portable)
    _write_npy_f32(path, width, height, channels, values)
    decoded_digest = _float_digest(values)
    return {
        "pass_name": pass_name,
        "path": portable,
        "file_format": "npy",
        "dtype": "float32",
        "channels": channels,
        "width": width,
        "height": height,
        "file_digest": file_digest(path),
        "decoded_digest": decoded_digest,
        "required": True,
    }


def _write_integer_artifact(
    context: GeneratorContext,
    pass_name: str,
    camera_id: str,
    width: int,
    height: int,
    values: list[int],
) -> dict[str, Any]:
    portable = f"renders/{pass_name}/{camera_id}.png"
    path = artifact_path(context.output_root, portable)
    _write_png_u16(path, width, height, values)
    canonical = b"".join(struct.pack(">H", value) for value in values)
    return {
        "pass_name": pass_name,
        "path": portable,
        "file_format": "png",
        "dtype": "uint16",
        "channels": 1,
        "width": width,
        "height": height,
        "file_digest": file_digest(path),
        "decoded_digest": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        "required": True,
    }


def _png_artifact(
    context: GeneratorContext,
    pass_name: str,
    portable: str,
    width: int,
    height: int,
    *,
    required: bool,
) -> dict[str, Any]:
    path = artifact_path(context.output_root, portable)
    channels, raw = _decode_png_bytes(path)
    return {
        "pass_name": pass_name,
        "path": portable,
        "file_format": "png",
        "dtype": "uint8",
        "channels": channels,
        "width": width,
        "height": height,
        "file_digest": file_digest(path),
        "decoded_digest": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "required": required,
    }


def _artifact_statistics(camera_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "pass_name": artifact["pass_name"],
        "dtype": artifact["dtype"],
        "shape": (
            [artifact["height"], artifact["width"]]
            if artifact["channels"] == 1
            else [artifact["height"], artifact["width"], artifact["channels"]]
        ),
        "decoded_digest": artifact["decoded_digest"],
    }


def _write_npy_f32(
    path: Path,
    width: int,
    height: int,
    channels: int,
    top_left_values: list[float],
) -> None:
    if channels not in {1, 3} or len(top_left_values) != width * height * channels:
        raise RuntimeError("invalid NPY dimensions")
    shape = (height, width) if channels == 1 else (height, width, channels)
    header = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape!r}, }}"
    ).encode("ascii")
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    output = array("f", top_left_values)
    if struct.pack("=I", 1)[0] != 1:
        output.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x93NUMPY"
        + bytes((1, 0))
        + struct.pack("<H", len(header))
        + header
        + output.tobytes()
    )


def _float_digest(values: list[float]) -> str:
    output = array("f", values)
    if struct.pack("=I", 1)[0] != 1:
        output.byteswap()
    return f"sha256:{hashlib.sha256(output.tobytes()).hexdigest()}"


def _write_png_u16(path: Path, width: int, height: int, values: list[int]) -> None:
    if len(values) != width * height:
        raise RuntimeError("invalid uint16 PNG dimensions")
    rows = bytearray()
    for row in range(height):
        rows.append(0)
        start = row * width
        for value in values[start : start + width]:
            rows.extend(struct.pack(">H", value))
    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _decode_png_bytes(path: Path) -> tuple[int, bytes]:
    content = path.read_bytes()
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("RGB output is not a PNG")
    position = 8
    width = height = bit_depth = color_type = 0
    compressed = bytearray()
    while position < len(content):
        length = struct.unpack(">I", content[position : position + 4])[0]
        kind = content[position + 4 : position + 8]
        payload = content[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if bit_depth != 8 or interlace != 0:
                raise RuntimeError("unsupported Blender RGB PNG encoding")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    channels = {2: 3, 6: 4}.get(color_type)
    if channels is None:
        raise RuntimeError("Blender RGB PNG is not RGB/RGBA")
    filtered = zlib.decompress(bytes(compressed))
    row_bytes = width * channels
    if len(filtered) != (row_bytes + 1) * height:
        raise RuntimeError("Blender RGB PNG payload length is invalid")
    raw = bytearray()
    previous = bytes(row_bytes)
    cursor = 0
    for _ in range(height):
        filter_type = filtered[cursor]
        cursor += 1
        row = bytearray(filtered[cursor : cursor + row_bytes])
        cursor += row_bytes
        _unfilter(row, previous, channels, filter_type)
        raw.extend(row)
        previous = bytes(row)
    return channels, bytes(raw)


def _unfilter(
    row: bytearray, previous: bytes, bytes_per_pixel: int, filter_type: int
) -> None:
    for index in range(len(row)):
        left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index]
        upper_left = (
            previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        )
        if filter_type == 0:
            continue
        if filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            prediction = left + up - upper_left
            distances = (
                abs(prediction - left),
                abs(prediction - up),
                abs(prediction - upper_left),
            )
            predictor = (left, up, upper_left)[distances.index(min(distances))]
        else:
            raise RuntimeError(f"unsupported PNG filter {filter_type}")
        row[index] = (row[index] + predictor) & 0xFF
