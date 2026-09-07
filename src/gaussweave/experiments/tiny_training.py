"""Tiny bounded differentiable gsplat qualification trainer."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gaussweave.accounting.resources import (
    cpu_memory_snapshot,
    disk_snapshot,
    gpu_memory_snapshot,
    guard_with_lifecycle,
    load_profile,
    measure_operation,
)
from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.experiments.lifecycle import (
    RunLogger,
    RunStatus,
    capture_failures,
    create_state,
    initialize_run_directory,
    transition,
    write_state,
)
from gaussweave.rendering.fixtures import IDENTITY, separated_gaussians
from gaussweave.rendering.gsplat_renderer import GsplatRenderer, write_smoke_outputs
from gaussweave.rendering.models import Camera, RenderableGaussians, RenderSettings
from gaussweave.results.artifacts import inventory_tree, write_inventory

CHECKPOINT_VERSION = "1.0"
FIXTURE_ID = "tiny-gs-v1"


class TrainingError(RuntimeError):
    """Tiny training configuration, checkpoint, or execution failed."""


@dataclass(frozen=True)
class TinyTrainingConfig:
    training_id: str = "train-tiny-gs-v1"
    seed: int = 17
    device: str = "cuda:0"
    dtype: str = "float32"
    width: int = 64
    height: int = 64
    train_camera_ids: tuple[str, ...] = ("train-center", "train-left")
    validation_camera_ids: tuple[str, ...] = ("validation-right",)
    test_camera_ids: tuple[str, ...] = ("test-up",)
    camera_batch_size: int = 1
    iterations: int = 30
    learning_rates: tuple[tuple[str, float], ...] = (
        ("means", 0.01),
        ("log_scales", 0.01),
        ("opacity_logits", 0.03),
        ("colors", 0.03),
    )
    loss_terms: tuple[tuple[str, float], ...] = (("l1", 1.0), ("l2", 0.25))
    initialization_policy: str = "seeded_bounded_cloud"
    initial_gaussian_count: int = 3
    maximum_gaussian_count: int = 3
    checkpoint_policy: str = "final"
    validation_interval: int = 10
    logging_interval: int = 5
    resource_profile: str = "smoke"
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    appearance_mode: str = "direct_rgb"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise TrainingError("training image dimensions must be positive")
        if not self.train_camera_ids or not self.test_camera_ids:
            raise TrainingError("train and held-out test splits must be nonempty")
        if not 1 <= self.camera_batch_size <= len(self.train_camera_ids):
            raise TrainingError("camera batch size must fit the training split")
        splits = (
            set(self.train_camera_ids),
            set(self.validation_camera_ids),
            set(self.test_camera_ids),
        )
        if any(
            splits[left] & splits[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise TrainingError("camera splits must be disjoint")
        if (
            self.iterations < 1
            or self.validation_interval < 1
            or self.logging_interval < 1
        ):
            raise TrainingError("iterations and intervals must be positive")
        if not self.learning_rates or any(
            value <= 0 for _, value in self.learning_rates
        ):
            raise TrainingError("learning rates must be positive")
        if not self.loss_terms or any(value < 0 for _, value in self.loss_terms):
            raise TrainingError("loss weights must be nonnegative")
        if not 1 <= self.initial_gaussian_count <= self.maximum_gaussian_count:
            raise TrainingError("initial Gaussian count must be within the cap")
        if self.dtype != "float32" or self.appearance_mode != "direct_rgb":
            raise TrainingError("tiny trainer supports float32 direct RGB only")
        if self.checkpoint_policy != "final":
            raise TrainingError("tiny trainer supports final checkpoint policy only")
        if self.initialization_policy != "seeded_bounded_cloud":
            raise TrainingError("unsupported initialization policy")
        load_profile(self.resource_profile)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["learning_rates"] = dict(self.learning_rates)
        value["loss_terms"] = dict(self.loss_terms)
        return value

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class TrainingOutcome:
    run_directory: Path
    initial_loss: float
    final_loss: float
    final_iteration: int
    gaussian_count: int
    checkpoint_path: Path
    checkpoint_digest: str
    scientific_checkpoint_digest: str
    held_out_camera_ids: tuple[str, ...]
    resource_record: Mapping[str, Any]


def fixture_cameras(width: int = 64, height: int = 64) -> dict[str, Camera]:
    """Return deterministic disjoint train, validation, and held-out cameras."""

    transforms = {
        "train-center": IDENTITY,
        "train-left": _translated(-0.25, 0.0, 0.0),
        "validation-right": _translated(0.25, 0.0, 0.0),
        "test-up": _translated(0.0, -0.25, 0.0),
    }
    focal_scale = min(width, height) / 64.0
    return {
        camera_id: Camera(
            camera_id,
            width,
            height,
            80.0 * focal_scale,
            80.0 * focal_scale,
            (width - 1) / 2.0,
            (height - 1) / 2.0,
            transform,
        )
        for camera_id, transform in transforms.items()
    }


def initialization_metadata(config: TinyTrainingConfig) -> dict[str, Any]:
    return {
        "policy": config.initialization_policy,
        "seed": config.seed,
        "count": config.initial_gaussian_count,
        "mean_bounds_m": [-0.6, 0.6],
        "depth_bounds_m": [2.4, 3.6],
        "log_scale": math.log(0.14),
        "rotation": "fixed_identity_wxyz",
        "optimized": ["means", "log_scales", "opacity_logits", "colors"],
        "fixed": ["rotations"],
        "densification": "none",
    }


def initialize_parameters(config: TinyTrainingConfig) -> dict[str, Any]:
    """Create deterministic trainable tensors without using target/test pixels."""

    torch, _ = _dependencies()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    count = config.initial_gaussian_count
    means = torch.rand((count, 3), generator=generator)
    means[:, :2] = means[:, :2] * 1.2 - 0.6
    means[:, 2] = means[:, 2] * 1.2 + 2.4
    colors = torch.rand((count, 3), generator=generator) * 0.6 + 0.2
    parameters = {
        "means": torch.nn.Parameter(means.to(config.device)),
        "log_scales": torch.nn.Parameter(
            torch.full((count, 3), math.log(0.14), device=config.device)
        ),
        "opacity_logits": torch.nn.Parameter(
            torch.full((count,), 1.0, device=config.device)
        ),
        "colors": torch.nn.Parameter(colors.to(config.device)),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count,
            dtype=torch.float32,
            device=config.device,
        ),
    }
    return parameters


def save_gaussian_checkpoint(
    path: Path,
    scene: RenderableGaussians,
    *,
    config: TinyTrainingConfig,
    run_id: str,
    iteration: int,
    loss_summary: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> str:
    """Atomically save a safe JSON deployable checkpoint with self checksum."""

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_VERSION,
        "training_configuration_digest": config.digest,
        "run_id": run_id,
        "scene_id": FIXTURE_ID,
        "iteration": iteration,
        "gaussian_count": scene.count,
        "means": [list(item) for item in scene.means],
        "rotations": [list(item) for item in scene.quaternions],
        "scales": [list(item) for item in scene.scales],
        "opacity_parameters": list(scene.opacities),
        "appearance_parameters": [list(item) for item in scene.appearance],
        "appearance_mode": scene.appearance_mode,
        "sh_degree": scene.sh_degree,
        "coordinate_frame": scene.coordinate_frame,
        "dtype": scene.dtype,
        "source_metadata": dict(source_metadata),
        "loss_summary": dict(loss_summary),
        "seed": config.seed,
    }
    payload["scientific_checkpoint_digest"] = scientific_checkpoint_digest(payload)
    digest = content_digest(payload)
    payload["artifact_checksum"] = digest
    _atomic_write(path, pretty_json_bytes(payload))
    return digest


def scientific_checkpoint_digest(payload: Mapping[str, Any]) -> str:
    """Digest model state at declared 1e-4 checkpoint comparison precision."""
    scientific = {
        key: payload[key]
        for key in (
            "format_version",
            "training_configuration_digest",
            "scene_id",
            "iteration",
            "gaussian_count",
            "means",
            "rotations",
            "scales",
            "opacity_parameters",
            "appearance_parameters",
            "appearance_mode",
            "sh_degree",
            "coordinate_frame",
            "dtype",
            "source_metadata",
            "seed",
        )
        if key in payload
    }
    return content_digest(_normalize_scientific_numbers(scientific))


def _normalize_scientific_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_normalize_scientific_numbers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_scientific_numbers(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_scientific_numbers(item) for key, item in value.items()}
    return value


def load_gaussian_checkpoint(path: Path) -> tuple[RenderableGaussians, dict[str, Any]]:
    """Load and validate a non-executable JSON checkpoint."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"unable to load checkpoint: {error}") from error
    validate_gaussian_checkpoint(payload)
    scene = RenderableGaussians(
        means=tuple(
            (float(item[0]), float(item[1]), float(item[2]))
            for item in payload["means"]
        ),
        quaternions=tuple(
            (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
            for item in payload["rotations"]
        ),
        scales=tuple(
            (float(item[0]), float(item[1]), float(item[2]))
            for item in payload["scales"]
        ),
        opacities=tuple(float(value) for value in payload["opacity_parameters"]),
        appearance=tuple(
            tuple(float(value) for value in item)
            for item in payload["appearance_parameters"]
        ),
        appearance_mode=str(payload["appearance_mode"]),
        sh_degree=payload["sh_degree"],
        coordinate_frame=str(payload["coordinate_frame"]),
        dtype=str(payload["dtype"]),
    )
    if scene.count != payload["gaussian_count"]:
        raise TrainingError("checkpoint Gaussian count mismatch")
    return scene, payload


def validate_gaussian_checkpoint(payload: Mapping[str, Any]) -> None:
    required = {
        "format_version",
        "training_configuration_digest",
        "run_id",
        "scene_id",
        "iteration",
        "gaussian_count",
        "means",
        "rotations",
        "scales",
        "opacity_parameters",
        "appearance_parameters",
        "appearance_mode",
        "sh_degree",
        "coordinate_frame",
        "dtype",
        "source_metadata",
        "loss_summary",
        "seed",
        "artifact_checksum",
    }
    if not required <= set(payload):
        raise TrainingError("checkpoint metadata is incomplete")
    if payload["format_version"] != CHECKPOINT_VERSION:
        raise TrainingError("unsupported checkpoint version")
    unsigned = dict(payload)
    checksum = unsigned.pop("artifact_checksum")
    if checksum != content_digest(unsigned):
        raise TrainingError("checkpoint checksum mismatch")
    count = payload["gaussian_count"]
    if not isinstance(count, int) or count < 1:
        raise TrainingError("invalid checkpoint Gaussian count")
    expected = {
        "means": 3,
        "rotations": 4,
        "scales": 3,
        "appearance_parameters": 3,
    }
    for key, width in expected.items():
        value = payload[key]
        if (
            not isinstance(value, list)
            or len(value) != count
            or any(not isinstance(row, list) or len(row) != width for row in value)
        ):
            raise TrainingError(f"malformed checkpoint tensor: {key}")
    if len(payload["opacity_parameters"]) != count:
        raise TrainingError("malformed checkpoint opacity tensor")


def run_tiny_training(
    root: Path,
    config: TinyTrainingConfig,
    *,
    inject_failure: bool = False,
    artificial_gpu_limit_bytes: int | None = None,
    run_id: str | None = None,
    experiment_id: str = "exp-tiny-training-v1",
    configuration_digest: str | None = None,
    resolved_configuration: Mapping[str, Any] | None = None,
    environment_record: Mapping[str, Any] | None = None,
) -> TrainingOutcome:
    """Execute one lifecycle-managed bounded qualification run."""

    run_id = run_id or (
        f"run-exp-tiny-training-v1-syn-tiny-s{config.seed}-{config.digest[7:19]}-a1"
    )
    state = create_state(
        run_id=run_id,
        experiment_id=experiment_id,
        scene_id="syn-tiny",
        attempt=1,
        configuration_ref="resolved_config.json",
        configuration_digest=configuration_digest or config.digest,
        environment_ref="metadata/environment.json",
    )
    run_directory = initialize_run_directory(root, state)
    _atomic_write(
        run_directory / "resolved_config.json",
        pretty_json_bytes(
            dict(resolved_configuration)
            if resolved_configuration is not None
            else config.to_dict()
        ),
    )
    _atomic_write(
        run_directory / "metadata/environment.json",
        pretty_json_bytes(
            dict(environment_record)
            if environment_record is not None
            else {"environment": "locked-wsl", "device": config.device}
        ),
    )
    logger = RunLogger(run_directory, run_id, repository_root=Path.cwd())
    state = transition(
        state,
        RunStatus.VALIDATED,
        reason="training configuration validated",
        actor="tiny-training",
    )
    write_state(state, run_directory / "status.json")
    guard_with_lifecycle(
        state,
        run_directory,
        logger,
        load_profile(config.resource_profile),
        workspace_root=root,
        artificial_gpu_limit_bytes=artificial_gpu_limit_bytes,
        require_cuda=True,
    )
    torch, gsplat = _dependencies()
    cameras = fixture_cameras(config.width, config.height)
    target_scene = separated_gaussians()
    target_settings = RenderSettings(
        output_buffers=("rgb", "alpha"),
        warmup_count=0,
        repetition_count=1,
        camera_batch_size=4,
    )
    all_ids = (
        *config.train_camera_ids,
        *config.validation_camera_ids,
        *config.test_camera_ids,
    )
    target_result = GsplatRenderer().render(
        target_scene, tuple(cameras[item] for item in all_ids), target_settings
    )
    write_smoke_outputs(
        target_result, run_directory / "artifacts/reference", overwrite=False
    )
    targets = {
        camera_id: target_result.rgb[index].detach()
        for index, camera_id in enumerate(all_ids)
    }
    target_metadata: Any = {
        "fixture_id": FIXTURE_ID,
        "seed": config.seed,
        "generator": "fixed_project_gaussians_v1",
        "camera_ids": list(all_ids),
        "renderer": f"gsplat-{target_result.renderer_version}",
        "test_targets_used_for_gradients": False,
    }
    _atomic_write(
        run_directory / "metadata/target-generation.json",
        pretty_json_bytes(target_metadata),
    )
    parameters = initialize_parameters(config)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [parameters[name]],
                "lr": dict(config.learning_rates)[name],
            }
            for name in ("means", "log_scales", "opacity_logits", "colors")
        ]
    )
    views = {
        camera_id: cameras[camera_id].gsplat_matrices(config.device)
        for camera_id in all_ids
    }
    loss_history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    last_iteration = 0
    initial_loss = float("nan")

    def predict(camera_id: str) -> Any:
        from gsplat import rasterization  # type: ignore[import-untyped]

        view, intrinsics = views[camera_id]
        image, _, _ = rasterization(
            parameters["means"],
            parameters["quaternions"],
            parameters["log_scales"].exp(),
            parameters["opacity_logits"].sigmoid(),
            parameters["colors"].sigmoid(),
            view.unsqueeze(0),
            intrinsics.unsqueeze(0),
            width=config.width,
            height=config.height,
            packed=True,
        )
        return image[0]

    def image_loss(prediction: Any, target: Any) -> Any:
        weights = dict(config.loss_terms)
        difference = prediction - target
        return (
            weights.get("l1", 0.0) * difference.abs().mean()
            + weights.get("l2", 0.0) * difference.square().mean()
        )

    # Compile native forward/backward without applying an optimizer update.
    optimizer.zero_grad(set_to_none=True)
    image_loss(
        predict(config.train_camera_ids[0]), targets[config.train_camera_ids[0]]
    ).backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(config.device)
    with torch.no_grad():
        initial_losses = [
            float(image_loss(predict(camera_id), targets[camera_id]))
            for camera_id in config.train_camera_ids
        ]
        initial_loss = sum(initial_losses) / len(initial_losses)

    def training_operation() -> None:
        nonlocal last_iteration, initial_loss
        for iteration in range(1, config.iterations + 1):
            if parameters["means"].shape[0] > config.maximum_gaussian_count:
                raise TrainingError("Gaussian count cap exceeded")
            start = (iteration - 1) % len(config.train_camera_ids)
            camera_ids = tuple(
                config.train_camera_ids[(start + offset) % len(config.train_camera_ids)]
                for offset in range(config.camera_batch_size)
            )
            optimizer.zero_grad(set_to_none=True)
            batch_losses = [
                image_loss(predict(camera_id), targets[camera_id])
                for camera_id in camera_ids
            ]
            loss = sum(batch_losses) / len(batch_losses)
            if not torch.isfinite(loss):
                raise TrainingError("nonfinite training loss")
            loss.backward()
            for name in ("means", "log_scales", "opacity_logits", "colors"):
                gradient = parameters[name].grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise TrainingError(f"nonfinite or missing gradient: {name}")
            torch.nn.utils.clip_grad_norm_(
                [
                    parameters[name]
                    for name in ("means", "log_scales", "opacity_logits", "colors")
                ],
                10.0,
            )
            optimizer.step()
            with torch.no_grad():
                parameters["means"][:, :2].clamp_(-1.0, 1.0)
                parameters["means"][:, 2].clamp_(1.0, 5.0)
                parameters["log_scales"].clamp_(math.log(0.02), math.log(0.5))
                parameters["opacity_logits"].clamp_(-8.0, 8.0)
                parameters["colors"].clamp_(-8.0, 8.0)
            last_iteration = iteration
            loss_history.append(
                {
                    "iteration": iteration,
                    "camera_ids": list(camera_ids),
                    "loss": float(loss.detach()),
                }
            )
            if iteration % config.validation_interval == 0:
                with torch.no_grad():
                    for validation_id in config.validation_camera_ids:
                        validation_started = time.perf_counter()
                        validation_loss = float(
                            image_loss(predict(validation_id), targets[validation_id])
                        )
                        torch.cuda.synchronize(config.device)
                        validation_history.append(
                            {
                                "iteration": iteration,
                                "camera_id": validation_id,
                                "loss": validation_loss,
                                "elapsed_seconds": (
                                    time.perf_counter() - validation_started
                                ),
                            }
                        )
            if iteration % config.logging_interval == 0:
                logger.event(
                    "INFO",
                    "training_progress",
                    "training",
                    f"iteration {iteration}/{config.iterations}",
                    context={
                        "loss": float(loss.detach()),
                        "gaussian_count": parameters["means"].shape[0],
                    },
                )
            if inject_failure and iteration == min(2, config.iterations):
                raise TrainingError("controlled injected trainer failure")

    try:
        failure_resource_snapshot = {
            "cpu": asdict(cpu_memory_snapshot()),
            "disk": asdict(disk_snapshot(root, safety_reserve_bytes=0)),
            "gpu": asdict(gpu_memory_snapshot(config.device)),
            "profile": config.resource_profile,
            "artificial_failure": inject_failure,
        }
        with capture_failures(
            state,
            run_directory,
            logger,
            stage="training",
            category="gs_numerical_failure",
            partial_artifacts=(
                "resolved_config.json",
                "metadata/target-generation.json",
                "artifacts/reference/render-metadata.json",
            ),
            last_completed_stage="iteration-1" if inject_failure else "iteration-0",
            resource_snapshot=failure_resource_snapshot,
        ) as running:
            resource_path = run_directory / "metadata/resource-record.json"
            resource = measure_operation(
                training_operation,
                operation_name="tiny-gs-training",
                profile=load_profile(config.resource_profile),
                workspace_root=root,
                device=config.device,
                repetition_count=1,
                output=resource_path,
                run_id=run_id,
                environment_ref=state.environment_ref,
            )
            scene = _scene_from_parameters(parameters)
            if scene.count > config.maximum_gaussian_count:
                raise TrainingError("checkpoint would exceed Gaussian cap")
            with torch.no_grad():
                final_losses = [
                    float(image_loss(predict(camera_id), targets[camera_id]))
                    for camera_id in config.train_camera_ids
                ]
            final_loss = sum(final_losses) / len(final_losses)
            checkpoint_path = run_directory / "artifacts/gaussians.checkpoint.json"
            checkpoint_started = time.perf_counter()
            checkpoint_digest = save_gaussian_checkpoint(
                checkpoint_path,
                scene,
                config=config,
                run_id=run_id,
                iteration=last_iteration,
                loss_summary={
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "history": loss_history,
                    "validation": validation_history,
                },
                source_metadata=initialization_metadata(config),
            )
            checkpoint_write_seconds = time.perf_counter() - checkpoint_started
            loaded_scene, checkpoint_metadata = load_gaussian_checkpoint(
                checkpoint_path
            )
            held_out = GsplatRenderer().render(
                loaded_scene,
                tuple(cameras[item] for item in config.test_camera_ids),
                RenderSettings(
                    output_buffers=("rgb", "alpha", "depth"),
                    warmup_count=0,
                    repetition_count=1,
                    camera_batch_size=config.camera_batch_size,
                ),
            )
            write_smoke_outputs(
                held_out, run_directory / "artifacts/held-out", overwrite=False
            )
            validation = GsplatRenderer().render(
                loaded_scene,
                tuple(cameras[item] for item in config.validation_camera_ids),
                RenderSettings(
                    output_buffers=("rgb", "alpha"),
                    warmup_count=0,
                    repetition_count=1,
                ),
            )
            write_smoke_outputs(
                validation, run_directory / "artifacts/validation", overwrite=False
            )
            summary = {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "final_iteration": last_iteration,
                "gaussian_count": scene.count,
                "maximum_gaussian_count": config.maximum_gaussian_count,
                "checkpoint_digest": checkpoint_digest,
                "scientific_checkpoint_digest": checkpoint_metadata[
                    "scientific_checkpoint_digest"
                ],
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_write_seconds": checkpoint_write_seconds,
                "training_resource": resource.to_dict(),
                "held_out_camera_ids": list(config.test_camera_ids),
                "held_out_render_seconds": held_out.elapsed_seconds,
                "validation_render_seconds": validation.elapsed_seconds,
                "validation_history": validation_history,
                "torch_version": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "gsplat_version": gsplat.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "code_commit": _git_commit(),
                "artifact_inventory_ref": "artifact-inventory.json",
            }
            _atomic_write(
                run_directory / "metadata/training-summary.json",
                pretty_json_bytes(summary),
            )
            logger.event(
                "INFO",
                "training_completed",
                "training",
                f"loss {initial_loss:.6f} -> {final_loss:.6f}",
            )
            inventory = inventory_tree(
                run_directory,
                inventory_id="inventory-tiny-training",
                excluded_paths=("artifact-inventory.json",),
            )
            write_inventory(inventory, run_directory / "artifact-inventory.json")
            completed = transition(
                running,
                RunStatus.COMPLETED,
                reason="tiny training completed",
                actor="tiny-training",
                artifact_inventory_ref="artifact-inventory.json",
            )
            write_state(completed, run_directory / "status.json")
            # Status changes after inventory creation; refresh final inventory.
            inventory = inventory_tree(
                run_directory,
                inventory_id="inventory-tiny-training",
                excluded_paths=("artifact-inventory.json",),
            )
            write_inventory(
                inventory, run_directory / "artifact-inventory.json", overwrite=True
            )
            return TrainingOutcome(
                run_directory,
                initial_loss,
                final_loss,
                last_iteration,
                scene.count,
                checkpoint_path,
                checkpoint_digest,
                checkpoint_metadata["scientific_checkpoint_digest"],
                config.test_camera_ids,
                resource.to_dict(),
            )
    except Exception:
        raise


def _scene_from_parameters(parameters: Mapping[str, Any]) -> RenderableGaussians:
    means = parameters["means"].detach().cpu().tolist()
    scales = parameters["log_scales"].detach().exp().cpu().tolist()
    opacity = parameters["opacity_logits"].detach().sigmoid().cpu().tolist()
    colors = parameters["colors"].detach().sigmoid().cpu().tolist()
    quaternions = parameters["quaternions"].detach().cpu().tolist()
    return RenderableGaussians(
        tuple(tuple(item) for item in means),
        tuple(tuple(item) for item in quaternions),
        tuple(tuple(item) for item in scales),
        tuple(opacity),
        tuple(tuple(item) for item in colors),
        stable_ids=tuple(f"trained-{index:04d}" for index in range(len(means))),
    )


def _dependencies() -> tuple[Any, Any]:
    try:
        import gsplat
        import torch
    except (ImportError, OSError) as error:
        raise TrainingError(
            "locked PyTorch/CUDA/gsplat environment unavailable"
        ) from error
    if not torch.cuda.is_available():
        raise TrainingError("CUDA is unavailable")
    return torch, gsplat


def _translated(x: float, y: float, z: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    except OSError as error:
        raise TrainingError(f"unable to write {path}: {error}") from error
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--root", required=True, type=Path)
    smoke.add_argument("--profile", default="smoke")
    smoke.add_argument("--iterations", type=int)
    smoke.add_argument("--inject-failure", action="store_true")
    smoke.add_argument("--artificial-gpu-limit-bytes", type=int)
    smoke.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    config = TinyTrainingConfig(
        resource_profile=options.profile,
        iterations=options.iterations or TinyTrainingConfig.iterations,
    )
    try:
        outcome = run_tiny_training(
            options.root,
            config,
            inject_failure=options.inject_failure,
            artificial_gpu_limit_bytes=options.artificial_gpu_limit_bytes,
        )
    except Exception as error:
        payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, indent=2) if options.json else str(error))
        return 1
    payload = {
        "valid": True,
        "run_directory": outcome.run_directory.as_posix(),
        "initial_loss": outcome.initial_loss,
        "final_loss": outcome.final_loss,
        "iterations": outcome.final_iteration,
        "gaussian_count": outcome.gaussian_count,
        "checkpoint": outcome.checkpoint_path.as_posix(),
        "checkpoint_digest": outcome.checkpoint_digest,
        "held_out_camera_ids": list(outcome.held_out_camera_ids),
        "compliance": outcome.resource_record["compliance"]["state"],
    }
    print(
        json.dumps(payload, indent=2, sort_keys=True)
        if options.json
        else (
            f"tiny training: loss {outcome.initial_loss:.6f} -> "
            f"{outcome.final_loss:.6f}; run={outcome.run_directory}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
