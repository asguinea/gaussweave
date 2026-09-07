"""Generic GaussWeave Blender entry point for deterministic runtime probes."""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from argparse import Namespace
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bpy

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from gaussweave_runtime.arguments import parse_runtime_arguments
from gaussweave_runtime.cameras import (
    install_cameras,
    load_camera_collection,
    projection_agreement,
)
from gaussweave_runtime.context import GeneratorContext
from gaussweave_runtime.io import (
    artifact_path,
    content_digest,
    file_digest,
    portable_path,
    write_checksums,
    write_json,
)
from gaussweave_runtime.metadata import scientific_metadata
from gaussweave_runtime.probe import build_runtime_probe
from gaussweave_runtime.render_passes import (
    build_render_collection,
    build_render_pass_probe,
    load_render_request,
    render_pass_views,
)
from gaussweave_runtime.rendering import configure_render
from gaussweave_runtime.scene import reset_scene
from gaussweave_runtime.validation import validate_runtime

GENERATOR_VERSION = "blender-runtime-v1"


def parse_arguments(arguments: list[str] | None = None) -> Namespace:
    """Parse only the argument tail following Blender's ``--`` marker."""

    if arguments is None:
        if "--" not in sys.argv:
            raise ValueError("Blender arguments must follow --")
        arguments = sys.argv[sys.argv.index("--") + 1 :]
    return parse_runtime_arguments(arguments)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load resolved config: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("resolved config must be a JSON object")
    required = {
        "config_schema_version",
        "scene_id",
        "family",
        "master_seed",
        "derived_seeds",
        "coordinate_system",
        "generator",
        "render",
        "camera",
        "outputs",
        "provenance",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("resolved config is missing: " + ", ".join(missing))
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or not isinstance(
        provenance.get("source_config"), str
    ):
        raise ValueError("resolved config provenance is invalid")
    portable_path(provenance["source_config"])
    return value


def _validate_handoff(options: Namespace, config: dict[str, Any]) -> None:
    if options.scene_id != config["scene_id"]:
        raise ValueError("handoff scene ID differs from resolved config")
    if options.master_seed != config["master_seed"]:
        raise ValueError("handoff master seed differs from resolved config")
    if options.generator_version != GENERATOR_VERSION:
        raise ValueError(
            f"unsupported generator version {options.generator_version!r}; "
            f"expected {GENERATOR_VERSION!r}"
        )
    coordinates = config["coordinate_system"]
    if (
        coordinates.get("handedness") != "right"
        or coordinates.get("up_axis") != "+Z"
        or coordinates.get("unit") != "meters"
    ):
        raise ValueError("resolved coordinate convention is unsupported")


def run(options: Namespace) -> None:
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    output_root = Path(options.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(Path(options.config))
    _validate_handoff(options, config)
    context = GeneratorContext.create(
        config,
        output_root,
        generator_version=options.generator_version,
        backend=options.backend,
        render_device=options.device,
        blender_version=bpy.app.version_string,
    )
    source_config = artifact_path(output_root, "source/resolved_scene_config.json")
    write_json(source_config, config)
    context.register_artifact(
        "resolved-scene-config", "source/resolved_scene_config.json", "resolved_config"
    )
    scene = reset_scene(context)
    if options.inject_failure == "after-reset":
        raise RuntimeError("injected Blender failure after reset")
    camera_mode = options.mode == "camera-probe"
    render_pass_mode = options.mode == "render-pass-probe"
    installed: list[bpy.types.Object] = []
    camera_collection: dict[str, Any] | None = None
    if render_pass_mode:
        if not options.cameras or not options.render_policy:
            raise ValueError("render-pass-probe requires --cameras and --render-policy")
        request = load_render_request(Path(options.render_policy))
        context.register_artifact(
            "render-request", "metadata/render_request.json", "other"
        )
        camera_collection = load_camera_collection(Path(options.cameras))
        request["_camera_collection"] = camera_collection
        probe_geometry = build_render_pass_probe(context, request)
        context.purpose = (
            "render-pass engineering fixture on generic proxy geometry; "
            "not a benchmark family scene or dataset"
        )
        installed = install_cameras(context, camera_collection)
    else:
        build_runtime_probe(context, include_camera=not camera_mode)
        if camera_mode:
            context.purpose = (
                "camera-probe engineering fixture; generated trajectories installed "
                "on generic proxy geometry, not a benchmark family scene"
            )
    previews: list[str] = []
    projection_checks: list[dict[str, Any]] = []
    if camera_mode or render_pass_mode:
        if not options.cameras:
            raise ValueError(f"{options.mode} requires --cameras")
        if camera_collection is None:
            camera_collection = load_camera_collection(Path(options.cameras))
            installed = install_cameras(context, camera_collection)
        records = {
            str(record["camera_id"]): record for record in camera_collection["records"]
        }
        selected: list[bpy.types.Object] = []
        selected_ids = (
            [str(item["camera_id"]) for item in request["views"]]
            if render_pass_mode
            else [
                str(camera_collection["splits"][split][0])
                for split in ("train", "validation", "test")
            ]
        )
        for camera_id in selected_ids:
            selected.append(
                next(
                    camera
                    for camera in installed
                    if str(camera["gaussweave_id"]) == camera_id
                )
            )
        for camera in selected:
            camera_id = str(camera["gaussweave_id"])
            record = records[camera_id]
            agreement = projection_agreement(
                scene,
                camera,
                record,
                tuple(float(value) for value in record["target_m"]),
            )
            if agreement["error_px"] > 1e-4:
                raise RuntimeError(
                    f"{camera_id}: Blender/project projection mismatch "
                    f"{agreement['error_px']} px"
                )
            projection_checks.append({"camera_id": camera_id, **agreement})
        if render_pass_mode:
            rendered = render_pass_views(context, scene, request, installed)
            render_settings = rendered["render_settings"]
            previews.extend(
                f"renders/rgb/{camera_id}.png"
                for camera_id in selected_ids
                if "rgb" in request["policy"]["enabled_passes"]
            )
            render_collection = build_render_collection(
                context,
                request,
                probe_geometry,
                rendered["views"],
            )
            write_json(
                artifact_path(output_root, "metadata/render_collection.json"),
                render_collection,
            )
            context.register_artifact(
                "render-collection",
                "metadata/render_collection.json",
                "other",
            )
            blend_portable = "source/render_pass_probe.blend"
        else:
            first_id = str(selected[0]["gaussweave_id"])
            first_preview = artifact_path(
                output_root, f"preview/cameras/{first_id}.png"
            )
            render_settings = configure_render(context, scene, first_preview)
            for camera in selected:
                camera_id = str(camera["gaussweave_id"])
                scene.camera = camera
                portable = f"preview/cameras/{camera_id}.png"
                scene.render.filepath = str(artifact_path(output_root, portable))
                bpy.ops.render.render(write_still=True)
                previews.append(portable)
                context.register_artifact(
                    f"camera-preview-{camera_id}", portable, "render"
                )
            blend_portable = "source/camera_probe.blend"
        if context.camera is not None:
            context.camera["active_camera_id"] = str(selected[-1]["gaussweave_id"])
        installation = {
            "camera_installation_schema_version": "1.0.0",
            "coordinate_conversion": (
                "OpenCV +X right/+Y down/+Z forward to "
                "Blender +X right/+Y up/-Z forward"
            ),
            "camera_scientific_digest": camera_collection["scientific_digest"],
            "installed_camera_count": len(installed),
            "rendered_camera_ids": [
                str(camera["gaussweave_id"]) for camera in selected
            ],
            "projection_checks": projection_checks,
        }
        write_json(
            artifact_path(output_root, "source/camera_installation.json"),
            installation,
        )
        context.register_artifact(
            "camera-installation",
            "source/camera_installation.json",
            "other",
        )
    else:
        preview = artifact_path(output_root, "preview/runtime_probe.png")
        render_settings = configure_render(context, scene, preview)
        bpy.ops.render.render(write_still=True)
        previews.append("preview/runtime_probe.png")
        context.register_artifact(
            "runtime-probe-preview", "preview/runtime_probe.png", "render"
        )
        blend_portable = "source/runtime_probe.blend"
    blend = artifact_path(output_root, blend_portable)
    blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    context.register_artifact(f"{options.mode}-blend", blend_portable, "other")
    if options.inject_failure == "before-validation":
        raise RuntimeError("injected Blender failure before validation")
    metadata_path = artifact_path(output_root, "source/generator_metadata.json")
    context.register_artifact(
        "generator-metadata", "source/generator_metadata.json", "other"
    )
    context.register_artifact(
        "execution-summary", "source/execution_summary.json", "other"
    )
    context.register_artifact("checksums", "checksums.sha256", "other")
    metadata = scientific_metadata(context, render_settings)
    write_json(metadata_path, metadata)
    if options.inject_failure == "validation":
        bpy.data.objects.remove(bpy.context.scene.camera, do_unlink=True)
        context.camera = None
    required_artifacts = [
        "source/resolved_scene_config.json",
        "source/generator_metadata.json",
        blend_portable,
        *previews,
    ]
    if camera_mode or render_pass_mode:
        required_artifacts.append("source/camera_installation.json")
    if render_pass_mode:
        required_artifacts.extend(
            [
                "metadata/render_conventions.json",
                "metadata/render_collection.json",
                "metadata/pass_statistics.json",
                *(
                    record["path"]
                    for view in render_collection["views"]
                    for record in view["pass_artifacts"]
                ),
            ]
        )
    validate_runtime(
        context,
        render_settings=render_settings,
        required_artifacts=required_artifacts,
    )
    ended_at = datetime.now(UTC).isoformat()
    metadata_file_digest = file_digest(metadata_path)
    summary_core = {
        "summary_schema_version": "1.0.0",
        "status": "completed",
        "mode": options.mode,
        "scene_id": context.scene_id,
        "family": context.family,
        "configuration_digest": context.configuration_digest,
        "scientific_metadata_digest": metadata["scientific_digest"],
        "generator_metadata_file_digest": metadata_file_digest,
        "backend": options.backend,
        "device": options.device,
        "provenance_ref": str(config["provenance"]["source_config"]),
        "output_refs": {
            "metadata": "source/generator_metadata.json",
            "blend": blend_portable,
            "previews": previews,
            "config": "source/resolved_scene_config.json",
        },
    }
    execution_summary = {
        "scientific": summary_core,
        "execution_summary_digest": content_digest(summary_core),
        "volatile": {
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_seconds": round(time.monotonic() - started, 9),
            "process_id": os.getpid(),
        },
    }
    summary_path = artifact_path(output_root, "source/execution_summary.json")
    write_json(summary_path, execution_summary)
    checksum_paths = [
        "source/resolved_scene_config.json",
        "source/generator_metadata.json",
        blend_portable,
        "source/execution_summary.json",
        *previews,
    ]
    if camera_mode:
        checksum_paths.append("source/camera_installation.json")
    elif render_pass_mode:
        checksum_paths = sorted(
            {
                record["path"]
                for record in context.artifacts
                if record["path"] != "checksums.sha256"
            }
        )
    write_checksums(output_root, checksum_paths, "checksums.sha256")


def main() -> None:
    options: Namespace | None = None
    try:
        options = parse_arguments()
        run(options)
    except BaseException as error:
        output = (
            Path(options.output_root).resolve()
            if options is not None and getattr(options, "output_root", None)
            else None
        )
        if output is not None:
            with suppress(Exception):
                write_json(
                    artifact_path(output, "source/runtime_failure.json"),
                    {
                        "failure_schema_version": "1.0.0",
                        "status": "failed",
                        "category": "generator_runtime_failure",
                        "stage": f"blender-{getattr(options, 'mode', 'runtime-probe')}",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
        raise


if __name__ == "__main__":
    main()
