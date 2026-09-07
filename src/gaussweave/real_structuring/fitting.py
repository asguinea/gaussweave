"""Bounded real-photograph fitting for shared SH and compact q8 residuals."""

from __future__ import annotations

import json
import math
import resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import content_digest
from gaussweave.gaussians.real_explicit import _camera_from_record
from gaussweave.real_structuring.background import (
    background_integrity,
    require_background_unchanged,
)
from gaussweave.real_structuring.compositing import (
    Plate,
    composite_panel_over_background,
    crop_camera,
    render_fields,
)
from gaussweave.real_structuring.hybrid import HybridTensors, validate_hybrid
from gaussweave.real_structuring.models import (
    INSTANCE_IDS,
    PILOT_VERSION,
    StructuringError,
    atomic_json,
    load_object,
    sha256_file,
)
from gaussweave.real_structuring.regions import INSTANCE_VIEWS
from gaussweave.real_structuring.residuals import encode_q8, encode_sh1_q8

SHARED_METHOD = "real_struct_shared"
Q8_METHOD = "real_struct_residual_q8"
SH1_Q8_METHOD = "real_struct_residual_sh1_q8"
DRJOHNSON_SHARED_METHOD = "real_drjohnson_struct_shared"
DRJOHNSON_Q8_METHOD = "real_drjohnson_struct_residual_q8"
TRUCK_BAY_SHARED_METHOD = "real_truck_bay_struct_shared"
TRUCK_BAY_Q8_METHOD = "real_truck_bay_struct_residual_q8"
_METHOD_KINDS = {
    SHARED_METHOD: "shared",
    Q8_METHOD: "q8",
    SH1_Q8_METHOD: "sh1_q8",
    DRJOHNSON_SHARED_METHOD: "shared",
    DRJOHNSON_Q8_METHOD: "q8",
    TRUCK_BAY_SHARED_METHOD: "shared",
    TRUCK_BAY_Q8_METHOD: "q8",
}
_SHARED_METHODS = {
    Q8_METHOD: SHARED_METHOD,
    SH1_Q8_METHOD: SHARED_METHOD,
    DRJOHNSON_Q8_METHOD: DRJOHNSON_SHARED_METHOD,
    TRUCK_BAY_Q8_METHOD: TRUCK_BAY_SHARED_METHOD,
}
FITTING_CAMERA_IDS = ("cam-train-000251", "cam-train-000180")
SHARED_ITERATIONS = 24
Q8_ITERATIONS = 18
SH1_Q8_ITERATIONS = 24
Q8_MINIMUM_TRAINING_LOSS_GAIN = 5e-5
SH1_Q8_MINIMUM_TRAINING_LOSS_GAIN = 1e-5


def _fit_box(
    image_name: str,
    instance_views: Mapping[str, Mapping[str, Sequence[int]]] = INSTANCE_VIEWS,
    instance_ids: Sequence[str] = INSTANCE_IDS,
) -> tuple[int, int, int, int]:
    boxes = [instance_views[instance_id][image_name] for instance_id in instance_ids]
    margin = 8
    return (
        max(0, min(box[0] for box in boxes) - margin),
        max(0, min(box[1] for box in boxes) - margin),
        max(box[2] for box in boxes) + margin,
        max(box[3] for box in boxes) + margin,
    )


def _target_and_masks(
    path: Path,
    image_name: str,
    crop: tuple[int, int, int, int],
    device: str,
    instance_views: Mapping[str, Mapping[str, Sequence[int]]] = INSTANCE_VIEWS,
    instance_ids: Sequence[str] = INSTANCE_IDS,
) -> tuple[Any, Any, Any]:
    try:
        from PIL import Image
    except ImportError as error:
        raise StructuringError("Pillow is required for real fitting targets") from error
    torch = _torch()
    with Image.open(path) as image:
        rgb = image.convert("RGB").crop(crop)
        target = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        target = (
            target.reshape(rgb.height, rgb.width, 3).to(device, torch.float32) / 255
        )
    union = torch.zeros((rgb.height, rgb.width), dtype=torch.bool, device=device)
    instance_masks = torch.zeros(
        (len(instance_ids), rgb.height, rgb.width),
        dtype=torch.bool,
        device=device,
    )
    x_offset, y_offset = crop[0], crop[1]
    for index, instance_id in enumerate(instance_ids):
        x0, y0, x1, y1 = instance_views[instance_id][image_name]
        # Four-pixel strict interior excludes ambiguous antialiased boundaries.
        left, top = x0 - x_offset + 4, y0 - y_offset + 4
        right, bottom = x1 - x_offset - 4, y1 - y_offset - 4
        instance_masks[index, top:bottom, left:right] = True
        union |= instance_masks[index]
    if int(union.sum().item()) <= 0:
        raise StructuringError("fitting ROI mask is empty")
    return target, union, instance_masks


def _write_f32(path: Path, tensor: Any) -> dict[str, Any]:
    tensor.detach().to("cpu").contiguous().numpy().tofile(path)
    return {
        "path": path.name,
        "shape": list(tensor.shape),
        "dtype": "little_endian_float32",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_shared(root: Path, device: str, method: str = SHARED_METHOD) -> Any:
    torch = _torch()
    record = load_object(root / "fits" / method / "fit-summary.json")
    field = record["checkpoint"]
    shape = tuple(int(value) for value in field["shape"])
    return (
        torch.from_file(
            str(root / "fits" / method / field["path"]),
            shared=False,
            size=math.prod(shape),
            dtype=torch.float32,
        )
        .reshape(shape)
        .to(device)
    )


def _records(
    hybrid_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], Path, tuple[str, ...]]:
    manifest = load_object(hybrid_root / "hybrid.json")
    dataset_root = Path(str(manifest["local_dataset_root"]))
    split = json.loads(
        (dataset_root / "annotations" / "evaluation-split.json").read_text(
            encoding="utf-8"
        )
    )
    records = {
        str(record["camera_id"]): cast(Mapping[str, Any], record)
        for record in split["cameras"]
    }
    fitting_camera_ids = tuple(
        str(value) for value in manifest.get("fitting_camera_ids", FITTING_CAMERA_IDS)
    )
    if not fitting_camera_ids:
        raise StructuringError("frozen fitting camera set is empty")
    if any(camera_id not in records for camera_id in fitting_camera_ids):
        raise StructuringError("frozen fitting cameras are unavailable")
    if any(camera_id.startswith("cam-eval-") for camera_id in fitting_camera_ids):
        raise StructuringError("evaluation camera entered fitting set")
    return records, dataset_root, fitting_camera_ids


def fitting_config(
    pilot_version: str = PILOT_VERSION,
    *,
    fitting_camera_ids: Sequence[str] = FITTING_CAMERA_IDS,
    instance_ids: Sequence[str] = INSTANCE_IDS,
    canonical_instance_id: str = "panel-middle",
) -> dict[str, Any]:
    result = {
        "pilot_version": pilot_version,
        "seed": 17,
        "fitting_camera_ids": list(fitting_camera_ids),
        "instance_ids": list(instance_ids),
        "canonical_instance_id": canonical_instance_id,
        "evaluation_policy": "all cam-eval-* records withheld from every loss",
        "camera_batch_size": 1,
        "resolution": "native qualified image dimensions; tight ROI crop",
        "background_strategy": (
            "fixed explicit complement pre-rendered once per crop; panel-only "
            "differentiable gsplat; expected-depth visibility composite"
        ),
        "shared": {
            "iterations": SHARED_ITERATIONS,
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "parameters": "all canonical degree-3 SH coefficients only",
            "loss": "ROI mean RGB L1 + 1e-4 mean-square source-SH regularization",
            "sampling": "deterministic round-robin two frozen training cameras",
            "stopping": "fixed iteration count",
            "checkpoint_interval": 8,
        },
        "q8": {
            "iterations": Q8_ITERATIONS,
            "optimizer": "Adam",
            "learning_rate": 0.005,
            "parameters": "one float SH0/RGB vector for rear and front; middle zero",
            "loss": "ROI mean RGB L1 + 5e-3 mean-square residual regularization",
            "quantization": (
                "single symmetric scale; signed int8; half-away-from-zero; "
                "canonical residual exactly zero"
            ),
            "acceptance": (
                "decode fitted offsets only when paired two-camera training-loss "
                f"gain is at least {Q8_MINIMUM_TRAINING_LOSS_GAIN}; otherwise "
                "serialize a zero q8 payload without consulting evaluation views"
            ),
            "stopping": "fixed iteration count",
            "checkpoint_interval": 6,
        },
        "sh1_q8": {
            "iterations": SH1_Q8_ITERATIONS,
            "optimizer": "Adam",
            "learning_rate": 0.003,
            "parameters": (
                "four degree-1 SH vectors for rear and front; middle exactly zero"
            ),
            "loss": "ROI mean RGB L1 + 5e-3 mean-square residual regularization",
            "quantization": (
                "single symmetric scale; signed int8; half-away-from-zero; "
                "canonical residual exactly zero; 36-byte payload"
            ),
            "acceptance": (
                "training-only paired-camera loss gain at least "
                f"{SH1_Q8_MINIMUM_TRAINING_LOSS_GAIN}"
            ),
            "stopping": "fixed iteration count",
            "activation_condition": (
                "only after SH0 q8 has no accepted fitting benefit and shared "
                "geometry passes local visual review"
            ),
        },
        "geometry": "fixed",
        "opacity": "fixed",
        "densification": "disabled",
    }
    result["scientific_digest"] = content_digest(cast(Any, result))
    return result


def fit(
    *,
    hybrid_root: Path,
    method: str,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit one declared method condition against frozen real training photos."""

    if method not in _METHOD_KINDS:
        raise StructuringError("unsupported real structured fitting method")
    method_kind = _METHOD_KINDS[method]
    validate_hybrid(hybrid_root)
    hybrid_manifest = load_object(hybrid_root / "hybrid.json")
    pilot_version = str(hybrid_manifest["pilot_version"])
    raw_views = hybrid_manifest.get("instance_views") or INSTANCE_VIEWS
    instance_views = cast(Mapping[str, Mapping[str, Sequence[int]]], raw_views)
    transforms = load_object(hybrid_root / "instance-transforms.json")["instances"]
    instance_ids = tuple(str(item["instance_id"]) for item in transforms)
    canonical_meta = load_object(hybrid_root / "canonical" / "metadata.json")
    canonical_instance_id = str(canonical_meta["source_instance_id"])
    if canonical_instance_id not in instance_ids:
        raise StructuringError("canonical instance is not declared by the hybrid")
    canonical_index = instance_ids.index(canonical_instance_id)
    method_root = hybrid_root / "fits" / method
    summary_path = method_root / "fit-summary.json"
    if resume and summary_path.is_file():
        summary = load_object(summary_path)
        if (
            summary.get("hybrid_digest")
            != load_object(hybrid_root / "hybrid.json")["scientific_digest"]
        ):
            raise StructuringError("resume identity does not match hybrid input")
        return summary
    if method_root.exists() and any(method_root.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite fitting output: {method_root}")
    method_root.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats()
    records, dataset_root, fitting_camera_ids = _records(hybrid_root)
    background_before = background_integrity(hybrid_root)
    hybrid = HybridTensors.load(hybrid_root)
    data: list[dict[str, Any]] = []
    plate_dir = hybrid_root / "background-plates"
    plate_dir.mkdir(exist_ok=True)
    for camera_id in fitting_camera_ids:
        record = records[camera_id]
        image_name = str(record["image_name"])
        full_camera = _camera_from_record(record)
        box = _fit_box(image_name, instance_views, instance_ids)
        box = (
            max(0, box[0]),
            max(0, box[1]),
            min(full_camera.width, box[2]),
            min(full_camera.height, box[3]),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise StructuringError("fitting camera crop is empty")
        camera = crop_camera(full_camera, box)
        target, union_mask, instance_masks = _target_and_masks(
            dataset_root / "source" / "images" / image_name,
            image_name,
            box,
            hybrid.device,
            instance_views,
            instance_ids,
        )
        with torch.no_grad():
            background = render_fields(hybrid.background, camera)
        plate_path = plate_dir / f"{camera_id}.f32"
        packed_plate = torch.cat(
            (
                background.rgb,
                background.alpha.unsqueeze(-1),
                background.depth.unsqueeze(-1),
            ),
            dim=-1,
        )
        plate_record = _write_f32(plate_path, packed_plate)
        data.append(
            {
                "camera_id": camera_id,
                "image_name": image_name,
                "camera": camera,
                "target": target,
                "union_mask": union_mask,
                "instance_masks": instance_masks,
                "background": Plate(
                    background.rgb.detach(),
                    background.alpha.detach(),
                    background.depth.detach(),
                ),
                "plate": plate_record,
                "target_sha256": sha256_file(
                    dataset_root / "source" / "images" / image_name
                ),
                "roi_pixel_count": int(union_mask.sum().item()),
            }
        )
    hybrid.background.clear()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    losses: list[dict[str, Any]] = []
    if method_kind == "shared":
        source_sh = hybrid.canonical["appearance"].detach().clone()
        parameter = source_sh.clone().requires_grad_(True)
        optimizer = torch.optim.Adam((parameter,), lr=0.01)
        iterations = SHARED_ITERATIONS
        for step in range(iterations):
            item = data[step % len(data)]
            optimizer.zero_grad(set_to_none=True)
            panel = render_fields(hybrid.panels(appearance=parameter), item["camera"])
            prediction = composite_panel_over_background(
                panel, item["background"]
            ).clamp(0, 1)
            mask = item["union_mask"]
            roi_l1 = (prediction[mask] - item["target"][mask]).abs().mean()
            regularization = (parameter - source_sh).square().mean()
            loss = roi_l1 + 1e-4 * regularization
            loss.backward()
            optimizer.step()
            losses.append(
                {
                    "iteration": step + 1,
                    "camera_id": item["camera_id"],
                    "total": float(loss.detach().item()),
                    "roi_l1": float(roi_l1.detach().item()),
                    "regularization": float(regularization.detach().item()),
                }
            )
        checkpoint = _write_f32(method_root / "shared-sh.f32", parameter)
        residual_record = None
        gradient_parameters = ["canonical_sh_coefficients"]
    else:
        shared_method = _SHARED_METHODS.get(method)
        if shared_method is None:
            raise StructuringError("compact method has no declared shared parent")
        shared = _load_shared(hybrid_root, hybrid.device, shared_method).detach()
        coefficient_count = 4 if method_kind == "sh1_q8" else 1
        residual_shape = (
            (len(instance_ids), 4, 3)
            if method_kind == "sh1_q8"
            else (len(instance_ids), 3)
        )
        float_residual = torch.zeros(
            residual_shape,
            dtype=torch.float32,
            device=hybrid.device,
            requires_grad=True,
        )
        learning_rate = 0.003 if method == SH1_Q8_METHOD else 0.005
        optimizer = torch.optim.Adam((float_residual,), lr=learning_rate)
        iterations = SH1_Q8_ITERATIONS if method_kind == "sh1_q8" else Q8_ITERATIONS
        mask_parameter = torch.ones(
            residual_shape, dtype=torch.float32, device=hybrid.device
        )
        mask_parameter[canonical_index] = 0
        for step in range(iterations):
            item = data[step % len(data)]
            optimizer.zero_grad(set_to_none=True)
            bounded = float_residual * mask_parameter
            panel = render_fields(
                hybrid.panels(appearance=shared, residuals=bounded),
                item["camera"],
            )
            prediction = composite_panel_over_background(
                panel, item["background"]
            ).clamp(0, 1)
            roi_l1 = (
                (prediction[item["union_mask"]] - item["target"][item["union_mask"]])
                .abs()
                .mean()
            )
            regularization = bounded.square().mean()
            loss = roi_l1 + 5e-3 * regularization
            loss.backward()
            optimizer.step()
            losses.append(
                {
                    "iteration": step + 1,
                    "camera_id": item["camera_id"],
                    "total": float(loss.detach().item()),
                    "roi_l1": float(roi_l1.detach().item()),
                    "regularization": float(regularization.detach().item()),
                }
            )
        fitted = (float_residual * mask_parameter).detach().cpu().tolist()
        training_gain = (
            sum(item["total"] for item in losses[:2]) / 2
            - sum(item["total"] for item in losses[-2:]) / 2
        )
        threshold = (
            SH1_Q8_MINIMUM_TRAINING_LOSS_GAIN
            if method_kind == "sh1_q8"
            else Q8_MINIMUM_TRAINING_LOSS_GAIN
        )
        accepted = training_gain >= threshold
        if method_kind == "sh1_q8":
            zero_offsets: Any = [
                [[0.0, 0.0, 0.0] for _ in range(4)] for _ in instance_ids
            ]
            encoded_offsets = fitted if accepted else zero_offsets
            residual_record = encode_sh1_q8(encoded_offsets)
            payload_values = [
                value
                for instance in residual_record["values"]
                for coefficient in instance
                for value in coefficient
            ]
        else:
            encoded_offsets = (
                fitted if accepted else [[0.0, 0.0, 0.0] for _ in instance_ids]
            )
            residuals = encode_q8(
                encoded_offsets,
                instance_ids=instance_ids,
                canonical_zero_instance=canonical_instance_id,
            )
            residual_record = residuals.to_dict()
            payload_values = [value for row in residuals.values for value in row]
        payload = bytes(value & 0xFF for value in payload_values)
        payload_path = method_root / "residuals.i8"
        payload_path.write_bytes(payload)
        residual_record["payload"] = {
            "path": payload_path.name,
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_file(payload_path),
        }
        residual_record["fitted_float_offsets"] = fitted
        residual_record["training_loss_gain"] = training_gain
        residual_record["minimum_training_loss_gain"] = threshold
        residual_record["fitted_offsets_accepted"] = accepted
        residual_record["acceptance_uses_evaluation_views"] = False
        residual_record["acceptance_reason"] = (
            "training_gain_met_threshold"
            if accepted
            else "training_gain_below_threshold_zero_payload_preserves_shared"
        )
        atomic_json(method_root / "residuals.json", residual_record)
        checkpoint = {
            "path": "residuals.i8",
            "shape": list(residual_shape),
            "dtype": "signed_int8",
            "bytes": len(payload),
            "sha256": sha256_file(payload_path),
        }
        gradient_parameters = [
            (
                "per_instance_degree1_sh_residual_rear_front"
                if coefficient_count == 4
                else "per_instance_sh0_residual_rear_front"
            )
        ]
    elapsed = time.perf_counter() - started
    background_after = background_integrity(hybrid_root)
    require_background_unchanged(background_before, background_after)
    initial = sum(item["total"] for item in losses[:2]) / min(2, len(losses))
    final = sum(item["total"] for item in losses[-2:]) / min(2, len(losses))
    if final >= initial:
        raise StructuringError("declared fitting loss did not decrease")
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    summary = {
        "pilot_version": pilot_version,
        "method_id": method,
        "status": "completed",
        "hybrid_digest": load_object(hybrid_root / "hybrid.json")["scientific_digest"],
        "configuration_digest": fitting_config(
            pilot_version,
            fitting_camera_ids=fitting_camera_ids,
            instance_ids=instance_ids,
            canonical_instance_id=canonical_instance_id,
        )["scientific_digest"],
        "real_photographs_used": True,
        "fitting_camera_ids": list(fitting_camera_ids),
        "evaluation_camera_ids_used_in_loss": [],
        "targets": [
            {
                "camera_id": item["camera_id"],
                "image_name": item["image_name"],
                "source_sha256": item["target_sha256"],
                "roi_pixel_count": item["roi_pixel_count"],
                "background_plate": item["plate"],
            }
            for item in data
        ],
        "iterations": iterations,
        "loss_initial": initial,
        "loss_final": final,
        "loss_decreased": final < initial,
        "loss_trace": losses,
        "checkpoint": checkpoint,
        "q8_residual": residual_record,
        "gradient_parameter_groups": gradient_parameters,
        "fixed_parameters": ["geometry", "opacity", "scale", "background"],
        "resource": {
            "elapsed_seconds": elapsed,
            "gpu_peak_allocated_bytes": peak_allocated,
            "gpu_peak_reserved_bytes": peak_reserved,
            "cpu_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "camera_batch_size": 1,
            "profile": "G14 standard",
        },
        "background_integrity_before": background_before,
        "background_integrity_after": background_after,
    }
    summary["scientific_digest"] = content_digest(summary)
    atomic_json(summary_path, summary)
    atomic_json(
        method_root / "fitting-config.json",
        fitting_config(
            pilot_version,
            fitting_camera_ids=fitting_camera_ids,
            instance_ids=instance_ids,
            canonical_instance_id=canonical_instance_id,
        ),
    )
    return summary


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise StructuringError("PyTorch is required for fitting") from error
    return torch
