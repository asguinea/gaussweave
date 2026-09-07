"""Unified qualification command-line interface with lazy subsystem loading."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import sys
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    """Stable public process-exit policy."""

    SUCCESS = 0
    USAGE = 2
    VALIDATION = 3
    ENVIRONMENT = 4
    RESOURCE = 5
    OPERATION = 6
    ARTIFACT_INTEGRITY = 7


def _leaf(
    parents: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    return parents.add_parser(name, help=help_text, description=help_text)


def _structured(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit one JSON object")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaussweave",
        description="GaussWeave reproducible Gaussian tooling.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", action="store_true", help="show dispatch diagnostics"
    )
    verbosity.add_argument(
        "--quiet", action="store_true", help="suppress successful human output"
    )
    groups = parser.add_subparsers(dest="group", required=True)

    blender = _leaf(groups, "blender", "qualify headless Blender generation")
    blender_commands = blender.add_subparsers(dest="action", required=True)
    blender_qualify = _leaf(
        blender_commands, "qualify", "run a deterministic Blender qualification"
    )
    blender_qualify.add_argument("--root", required=True)
    blender_qualify.add_argument("--backend", choices=("wsl", "windows"), default="wsl")
    blender_qualify.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    blender_qualify.add_argument("--seed", type=int, default=17)
    blender_qualify.add_argument("--blender-executable")
    blender_qualify.add_argument("--timeout-seconds", type=float, default=180.0)
    blender_qualify.add_argument("--overwrite", action="store_true")
    _structured(blender_qualify)
    blender_runtime = _leaf(
        blender_commands,
        "runtime-probe",
        "run the generic deterministic Blender generator runtime probe",
    )
    blender_runtime.add_argument("--config", required=True)
    blender_runtime.add_argument("--root", required=True)
    blender_runtime.add_argument("--backend", choices=("wsl", "windows"), default="wsl")
    blender_runtime.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    blender_runtime.add_argument("--blender-executable")
    blender_runtime.add_argument("--timeout-seconds", type=float, default=300.0)
    blender_runtime.add_argument("--overwrite", action="store_true")
    blender_runtime.add_argument(
        "--inject-failure",
        choices=("after-reset", "before-validation", "validation"),
    )
    _structured(blender_runtime)
    blender_camera = _leaf(
        blender_commands,
        "camera-probe",
        "install generated cameras into the generic Blender proxy fixture",
    )
    blender_camera.add_argument("--config", required=True)
    blender_camera.add_argument("--root", required=True)
    blender_camera.add_argument("--backend", choices=("wsl", "windows"), default="wsl")
    blender_camera.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    blender_camera.add_argument("--blender-executable")
    blender_camera.add_argument("--timeout-seconds", type=float, default=300.0)
    blender_camera.add_argument("--camera-seed", type=int)
    blender_camera.add_argument("--overwrite", action="store_true")
    _structured(blender_camera)
    blender_render_pass = _leaf(
        blender_commands,
        "render-pass-probe",
        "render aligned RGB, depth, normal, and identity engineering passes",
    )
    blender_render_pass.add_argument("--config", required=True)
    blender_render_pass.add_argument("--root", required=True)
    blender_render_pass.add_argument(
        "--backend", choices=("wsl", "windows"), default="wsl"
    )
    blender_render_pass.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    blender_render_pass.add_argument("--blender-executable")
    blender_render_pass.add_argument("--timeout-seconds", type=float, default=600.0)
    blender_render_pass.add_argument("--camera-seed", type=int)
    blender_render_pass.add_argument("--camera-id", action="append")
    blender_render_pass.add_argument("--overwrite", action="store_true")
    blender_render_pass.add_argument(
        "--inject-failure",
        choices=("after-reset", "before-validation", "validation"),
    )
    _structured(blender_render_pass)

    camera = _leaf(groups, "camera", "generate and validate deterministic cameras")
    camera_commands = camera.add_subparsers(dest="action", required=True)
    camera_generate = _leaf(
        camera_commands, "generate", "generate camera records and trajectory splits"
    )
    camera_generate.add_argument("--config", required=True)
    camera_generate.add_argument("--root", required=True)
    camera_generate.add_argument(
        "--family", choices=("facade", "corridor", "colonnade")
    )
    camera_generate.add_argument("--camera-seed", type=int)
    camera_generate.add_argument("--train-count", type=int)
    camera_generate.add_argument("--validation-count", type=int)
    camera_generate.add_argument("--test-count", type=int)
    camera_generate.add_argument("--overwrite", action="store_true")
    _structured(camera_generate)
    camera_validate = _leaf(
        camera_commands, "validate", "validate camera metadata and split separation"
    )
    camera_validate.add_argument("path")
    camera_validate.add_argument("--minimum-separation-m", type=float)
    camera_validate.add_argument("--minimum-angular-separation-degrees", type=float)
    camera_validate.add_argument("--minimum-structure-coverage", type=float)
    _structured(camera_validate)

    env = _leaf(groups, "env", "inspect the host, WSL, and GPU environment")
    env_commands = env.add_subparsers(dest="action", required=True)
    env_check = _leaf(env_commands, "check", "inspect environment compatibility")
    env_check.add_argument("--output")
    env_check.add_argument("--repo-root")
    _structured(env_check)

    schema = _leaf(groups, "schema", "validate accepted schema-governed documents")
    schema_commands = schema.add_subparsers(dest="action", required=True)
    schema_validate = _leaf(
        schema_commands, "validate", "validate a document by schema kind"
    )
    schema_validate.add_argument(
        "--kind",
        required=True,
        choices=("scene_manifest", "grammar", "experiment", "result"),
    )
    schema_validate.add_argument("path")
    _structured(schema_validate)

    config = _leaf(groups, "config", "validate and resolve experiment configuration")
    config_commands = config.add_subparsers(dest="action", required=True)
    config_validate = _leaf(
        config_commands, "validate", "validate an experiment configuration"
    )
    config_validate.add_argument("path")
    _structured(config_validate)
    config_resolve = _leaf(
        config_commands, "resolve", "merge ordered configuration layers"
    )
    config_resolve.add_argument("--layer", action="append", required=True)
    config_resolve.add_argument("--output", required=True)
    config_resolve.add_argument("--provenance-output")
    config_resolve.add_argument("--canonical-output")
    config_resolve.add_argument("--digest-output")
    config_resolve.add_argument("--scene-id")
    config_resolve.add_argument("--seed", type=int)
    config_resolve.add_argument("--attempt", type=int, default=1)
    config_resolve.add_argument("--overwrite", action="store_true")
    _structured(config_resolve)

    scene = _leaf(groups, "scene", "validate and resolve procedural scene intent")
    scene_commands = scene.add_subparsers(dest="action", required=True)
    scene_validate = _leaf(
        scene_commands, "validate", "validate a procedural scene configuration"
    )
    scene_validate.add_argument("path")
    _structured(scene_validate)
    scene_resolve = _leaf(
        scene_commands, "resolve", "write a fully resolved scene configuration"
    )
    scene_resolve.add_argument("--config", dest="path", required=True)
    scene_resolve.add_argument("--output", required=True)
    scene_resolve.add_argument("--canonical-output")
    scene_resolve.add_argument("--digest-output")
    scene_resolve.add_argument("--repository-root")
    scene_resolve.add_argument("--overwrite", action="store_true")
    _structured(scene_resolve)

    artifact = _leaf(groups, "artifact", "inventory and verify artifacts")
    artifact_commands = artifact.add_subparsers(dest="action", required=True)
    inventory = _leaf(
        artifact_commands, "inventory", "create a checksummed artifact inventory"
    )
    inventory.add_argument("--root", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--inventory-id", default="inventory-artifacts")
    inventory.add_argument(
        "--artifact-type",
        choices=(
            "accounting_record",
            "binding",
            "checkpoint",
            "environment",
            "failure_report",
            "figure",
            "grammar",
            "metric_observations",
            "other",
            "qualitative_panel",
            "render",
            "resolved_config",
            "resource_record",
            "table",
            "video",
        ),
        default="other",
    )
    inventory.add_argument("--optional", action="append")
    inventory.add_argument("--exclude-hidden", action="store_true")
    inventory.add_argument("--overwrite", action="store_true")
    _structured(inventory)
    verify = _leaf(
        artifact_commands, "verify", "verify inventory checksums and tree state"
    )
    verify.add_argument("--root", required=True)
    verify.add_argument("--inventory", required=True)
    verify.add_argument("--strict", action="store_true")
    _structured(verify)

    lifecycle = _leaf(groups, "lifecycle", "exercise durable run lifecycle records")
    lifecycle_commands = lifecycle.add_subparsers(dest="action", required=True)
    lifecycle_demo = _leaf(
        lifecycle_commands, "demo", "run a lifecycle success or controlled failure"
    )
    lifecycle_demo.add_argument("--root", required=True)
    lifecycle_demo.add_argument("--mode", choices=("success", "failure"), required=True)

    resource = _leaf(groups, "resource", "measure resource telemetry and guards")
    resource_commands = resource.add_subparsers(dest="action", required=True)
    resource_demo = _leaf(resource_commands, "demo", "run a measured resource demo")
    resource_demo.add_argument(
        "--profile",
        choices=("smoke", "quick", "standard", "extended"),
        required=True,
    )
    resource_demo.add_argument("--device", required=True)
    resource_demo.add_argument("--output")
    resource_demo.add_argument("--artificial-cpu-limit-bytes", type=int)
    resource_demo.add_argument("--artificial-gpu-limit-bytes", type=int)
    _structured(resource_demo)

    render = _leaf(groups, "render", "render through the native gsplat adapter")
    render_commands = render.add_subparsers(dest="action", required=True)
    render_smoke = _leaf(render_commands, "smoke", "run the bounded GPU render smoke")
    render_smoke.add_argument("--profile", default="smoke")
    render_smoke.add_argument("--output", required=True)
    render_smoke.add_argument("--no-depth", action="store_true")
    render_smoke.add_argument("--overwrite", action="store_true")
    _structured(render_smoke)
    render_validate = _leaf(
        render_commands,
        "validate",
        "decode and validate aligned synthetic render-pass artifacts",
    )
    render_validate.add_argument("root")
    _structured(render_validate)

    train = _leaf(groups, "train", "run the tiny GS qualification trainer")
    train_commands = train.add_subparsers(dest="action", required=True)
    train_smoke = _leaf(train_commands, "smoke", "run bounded tiny GPU training")
    train_smoke.add_argument("--root", required=True)
    train_smoke.add_argument("--profile", default="smoke")
    train_smoke.add_argument("--iterations", type=int)
    train_smoke.add_argument("--inject-failure", action="store_true")
    train_smoke.add_argument("--artificial-gpu-limit-bytes", type=int)
    _structured(train_smoke)

    experiment = _leaf(
        groups, "experiment", "orchestrate reproducible experiment workflows"
    )
    experiment_commands = experiment.add_subparsers(dest="action", required=True)
    experiment_smoke = _leaf(
        experiment_commands,
        "smoke",
        "run the end-to-end reproducible smoke pipeline",
    )
    experiment_smoke.add_argument("--config", default="configs/experiments/smoke.json")
    experiment_smoke.add_argument("--overlay", action="append")
    experiment_smoke.add_argument("--root", required=True)
    experiment_smoke.add_argument("--inject-failure", action="store_true")
    experiment_smoke.add_argument("--artificial-gpu-limit-bytes", type=int)
    _structured(experiment_smoke)
    experiment_calibrate = _leaf(
        experiment_commands,
        "calibrate",
        "run the bounded E0 hardware qualification matrix",
    )
    experiment_calibrate.add_argument(
        "--config", default="configs/experiments/hardware-calibration.json"
    )
    experiment_calibrate.add_argument("--root", required=True)
    _structured(experiment_calibrate)

    result = _leaf(groups, "result", "build and validate result records")
    result_commands = result.add_subparsers(dest="action", required=True)
    result_build = _leaf(result_commands, "build-smoke", "build a smoke result")
    result_build.add_argument("--run-root", required=True)
    result_build.add_argument("--output", required=True)
    result_build.add_argument("--overwrite", action="store_true")
    _structured(result_build)
    result_validate = _leaf(result_commands, "validate", "validate a result record")
    result_validate.add_argument("path")
    _structured(result_validate)

    real_data = _leaf(groups, "real-data", "validate restricted real-scene data")
    real_data_commands = real_data.add_subparsers(dest="action", required=True)
    real_data_validate = _leaf(
        real_data_commands,
        "validate",
        "validate the official Truck images, COLMAP cameras, and sparse model",
    )
    real_data_validate.add_argument("path")
    real_data_validate.add_argument("--output")
    real_data_validate.add_argument("--overwrite", action="store_true")
    _structured(real_data_validate)

    real_gs = _leaf(groups, "real-gs", "inspect, convert, and render explicit real GS")
    real_gs_commands = real_gs.add_subparsers(dest="action", required=True)
    real_gs_inspect = _leaf(
        real_gs_commands, "inspect", "inspect an official GRAPHDECO Gaussian PLY"
    )
    real_gs_inspect.add_argument("path")
    _structured(real_gs_inspect)
    real_gs_convert = _leaf(
        real_gs_commands,
        "convert",
        "convert a GRAPHDECO PLY into project Gaussian semantics",
    )
    real_gs_convert.add_argument("path")
    real_gs_convert.add_argument("--output", required=True)
    real_gs_convert.add_argument("--overwrite", action="store_true")
    _structured(real_gs_convert)
    real_gs_render = _leaf(
        real_gs_commands,
        "render",
        "render selected cameras with the converted explicit model",
    )
    real_gs_render.add_argument("path")
    real_gs_render.add_argument("--cameras", required=True)
    real_gs_render.add_argument("--output", required=True)
    real_gs_render.add_argument("--source-images")
    real_gs_render.add_argument("--overwrite", action="store_true")
    _structured(real_gs_render)

    real_region = _leaf(
        groups, "real-region", "build and validate structured real-scene regions"
    )
    real_region_commands = real_region.add_subparsers(dest="action", required=True)
    real_region_validate = _leaf(
        real_region_commands,
        "validate",
        "validate a materialized structured-region artifact",
    )
    real_region_validate.add_argument("path")
    _structured(real_region_validate)
    real_region_extract = _leaf(
        real_region_commands,
        "extract",
        "extract deterministic panel ownership and canonical arrays",
    )
    real_region_extract.add_argument("--qualification", required=True)
    real_region_extract.add_argument("--dataset-root", required=True)
    real_region_extract.add_argument("--output", required=True)
    real_region_extract.add_argument("--overwrite", action="store_true")
    _structured(real_region_extract)
    real_region_register = _leaf(
        real_region_commands,
        "register",
        "validate deterministic oracle panel registration",
    )
    real_region_register.add_argument("--region-root", required=True)
    _structured(real_region_register)
    real_region_build = _leaf(
        real_region_commands,
        "build-hybrid",
        "serialize fixed background plus three canonical panel instances",
    )
    real_region_build.add_argument("--region-root", required=True)
    real_region_build.add_argument("--output", required=True)
    real_region_build.add_argument("--overwrite", action="store_true")
    _structured(real_region_build)
    real_region_fit = _leaf(
        real_region_commands,
        "fit",
        "fit shared or q8 panel appearance against real training photographs",
    )
    real_region_fit.add_argument("--hybrid-root", required=True)
    real_region_fit.add_argument(
        "--method",
        required=True,
        choices=("real_struct_shared", "real_struct_residual_q8"),
    )
    real_region_fit.add_argument("--resume", action="store_true")
    real_region_fit.add_argument("--overwrite", action="store_true")
    _structured(real_region_fit)
    real_region_evaluate = _leaf(
        real_region_commands,
        "evaluate",
        "evaluate explicit, shared, and q8 methods on frozen held-out views",
    )
    real_region_evaluate.add_argument("--hybrid-root", required=True)
    real_region_evaluate.add_argument("--overwrite", action="store_true")
    _structured(real_region_evaluate)
    real_region_edit = _leaf(
        real_region_commands,
        "edit",
        "remove one accepted Truck instance through the active-instance set",
    )
    real_region_edit.add_argument("--refit-root", required=True)
    real_region_edit.add_argument(
        "--operation",
        required=True,
        choices=("remove-instance",),
    )
    real_region_edit.add_argument(
        "--instance",
        required=True,
        choices=("panel-middle",),
    )
    real_region_edit.add_argument("--output", required=True)
    real_region_edit.add_argument("--camera", action="append")
    real_region_edit.add_argument("--overwrite", action="store_true")
    _structured(real_region_edit)
    real_region_validate_edit = _leaf(
        real_region_commands,
        "validate-edit",
        "validate a local Truck active-instance edit without heavy imports",
    )
    real_region_validate_edit.add_argument("path")
    real_region_validate_edit.add_argument(
        "--allow-debug-subset",
        action="store_true",
    )
    _structured(real_region_validate_edit)

    return parser


def _append(arguments: list[str], name: str, value: Any) -> None:
    if value is None or value is False:
        return
    option = f"--{name.replace('_', '-')}"
    if value is True:
        arguments.append(option)
    elif isinstance(value, list):
        for item in value:
            arguments.extend((option, str(item)))
    else:
        arguments.extend((option, str(value)))


def _target(options: argparse.Namespace) -> tuple[str, list[str], ExitCode]:
    values = vars(options)
    group = options.group
    action = options.action
    forwarded: list[str] = []
    if group == "blender":
        module = (
            "gaussweave.data.blender_runtime"
            if action in {"runtime-probe", "camera-probe", "render-pass-probe"}
            else "gaussweave.data.blender"
        )
        failure = ExitCode.OPERATION
        if action in {"runtime-probe", "camera-probe", "render-pass-probe"}:
            forwarded.extend(("--mode", action))
    elif group == "camera":
        module = "gaussweave.data.cameras"
        failure = ExitCode.VALIDATION
        forwarded.append(action)
        if action == "validate":
            forwarded.append(options.path)
    elif group == "env":
        module = "gaussweave.runtime.environment"
        failure = ExitCode.ENVIRONMENT
    elif group == "schema":
        module = "gaussweave.config.validation"
        failure = ExitCode.VALIDATION
        _append(forwarded, "kind", options.kind)
        forwarded.append(options.path)
    elif group == "config" and action == "validate":
        module = "gaussweave.config.validation"
        failure = ExitCode.VALIDATION
        forwarded.extend(("--kind", "experiment", options.path))
    elif group == "config":
        module = "gaussweave.config.resolution"
        failure = ExitCode.VALIDATION
    elif group == "scene":
        module = "gaussweave.data.scene_config"
        failure = ExitCode.VALIDATION
        forwarded.append(action)
        if action == "resolve":
            forwarded.extend(("--config", options.path))
        else:
            forwarded.append(options.path)
    elif group == "artifact":
        module = "gaussweave.results.artifacts"
        failure = ExitCode.ARTIFACT_INTEGRITY
        forwarded.append(action)
    elif group == "lifecycle":
        module = "gaussweave.experiments.lifecycle"
        failure = ExitCode.OPERATION
        forwarded.append(action)
    elif group == "resource":
        module = "gaussweave.accounting.resources"
        failure = ExitCode.RESOURCE
        forwarded.append(action)
    elif group == "render":
        module = (
            "gaussweave.data.render_passes"
            if action == "validate"
            else "gaussweave.rendering.gsplat_renderer"
        )
        failure = (
            ExitCode.ARTIFACT_INTEGRITY if action == "validate" else ExitCode.OPERATION
        )
        forwarded.append(action)
        if action == "validate":
            forwarded.append(options.root)
    elif group == "train":
        module = "gaussweave.experiments.tiny_training"
        failure = (
            ExitCode.RESOURCE
            if options.artificial_gpu_limit_bytes is not None
            else ExitCode.OPERATION
        )
        forwarded.append(action)
    elif group == "experiment":
        module = (
            "gaussweave.experiments.calibration"
            if action == "calibrate"
            else "gaussweave.experiments.smoke_pipeline"
        )
        failure = ExitCode.OPERATION
        if action != "calibrate":
            forwarded.append(action)
    elif group == "real-data":
        module = "gaussweave.data.real_truck"
        failure = ExitCode.VALIDATION
        forwarded.extend((action, options.path))
    elif group == "real-gs":
        module = "gaussweave.gaussians.real_explicit"
        failure = ExitCode.OPERATION
        forwarded.extend((action, options.path))
    elif group == "real-region":
        if action in {"edit", "validate-edit"}:
            module = "gaussweave.real_structuring.editing"
            failure = ExitCode.OPERATION if action == "edit" else ExitCode.VALIDATION
            forwarded.append(action)
            if action == "validate-edit":
                forwarded.append(options.path)
        else:
            module = "gaussweave.real_structuring.cli"
            failure = (
                ExitCode.VALIDATION if action == "validate" else ExitCode.OPERATION
            )
            forwarded.append(action)
            if action == "validate":
                forwarded.append(options.path)
    else:
        module = "gaussweave.results.records"
        failure = ExitCode.VALIDATION if action == "validate" else ExitCode.OPERATION
        forwarded.append(action)
        if action == "validate":
            forwarded.append(options.path)
    excluded = {
        "group",
        "action",
        "verbose",
        "quiet",
        "kind",
        "path",
    }
    if group == "real-region" and action not in {"validate", "validate-edit"}:
        excluded.discard("path")
    if group == "render" and action == "validate":
        excluded.add("root")
    for name, value in values.items():
        if name in excluded:
            continue
        _append(forwarded, name, value)
    return module, forwarded, failure


def _json_error(stdout: str, stderr: str, code: ExitCode) -> str:
    stripped = stdout.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                parsed["exit_code"] = int(code)
                return json.dumps(parsed, indent=2, sort_keys=True)
        except json.JSONDecodeError:
            pass
    message = stderr.strip() or stripped or "command failed"
    return json.dumps(
        {"valid": False, "exit_code": int(code), "error": message},
        indent=2,
        sort_keys=True,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Parse, lazily dispatch, normalize output, and return a stable exit code."""

    options = _parser().parse_args(arguments)
    module_name, forwarded, failure_code = _target(options)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        module = importlib.import_module(module_name)
        handler = module.main
        with redirect_stdout(stdout), redirect_stderr(stderr):
            raw_code = int(handler(forwarded))
    except (ImportError, OSError) as error:
        raw_code = 1
        stderr.write(f"{type(error).__name__}: {error}")
        failure_code = ExitCode.ENVIRONMENT
    code = (
        ExitCode.SUCCESS
        if raw_code == 0
        else ExitCode(raw_code)
        if raw_code in {3, 4, 5, 6, 7}
        else failure_code
    )
    json_mode = bool(getattr(options, "json", False))
    if json_mode:
        if code is ExitCode.SUCCESS:
            output = stdout.getvalue().strip()
            try:
                value = json.loads(output)
                if not isinstance(value, dict):
                    raise json.JSONDecodeError("not an object", output, 0)
            except json.JSONDecodeError:
                value = {"valid": True, "output": output}
            print(json.dumps(value, indent=2, sort_keys=True))
        else:
            print(_json_error(stdout.getvalue(), stderr.getvalue(), code))
    elif code is not ExitCode.SUCCESS:
        message = stderr.getvalue().strip() or stdout.getvalue().strip()
        print(message or "command failed", file=sys.stderr)
    elif not options.quiet:
        print(stdout.getvalue().rstrip())
        if options.verbose:
            print(
                f"dispatch: {module_name}; exit_code={int(code)}",
                file=sys.stderr,
            )
    return int(code)
