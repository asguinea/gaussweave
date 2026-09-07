"""Held-out real-photograph evaluation for explicit and structured methods."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import content_digest
from gaussweave.gaussians.real_explicit import (
    RealGaussianTensorSource,
    _camera_from_record,
)
from gaussweave.real_structuring.compositing import render_fields
from gaussweave.real_structuring.fitting import (
    DRJOHNSON_Q8_METHOD,
    DRJOHNSON_SHARED_METHOD,
    Q8_METHOD,
    SH1_Q8_METHOD,
    SHARED_METHOD,
    TRUCK_BAY_Q8_METHOD,
    TRUCK_BAY_SHARED_METHOD,
    _load_shared,
)
from gaussweave.real_structuring.hybrid import HybridTensors, validate_hybrid
from gaussweave.real_structuring.models import (
    StructuringError,
    atomic_json,
    load_object,
)


def _source_image(path: Path, device: str) -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise StructuringError("Pillow is required for held-out targets") from error
    torch = _torch()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        values = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        return values.reshape(rgb.height, rgb.width, 3).to(device, torch.float32) / 255


def _write_png(path: Path, tensor: Any) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise StructuringError("Pillow is required for local render outputs") from error
    values = tensor.detach().clamp(0, 1).mul(255).round().to("cpu", _torch().uint8)
    Image.frombytes(
        "RGB", (values.shape[1], values.shape[0]), bytes(values.flatten())
    ).save(path)


def _psnr(prediction: Any, target: Any, mask: Any | None = None) -> tuple[float, float]:
    difference = prediction - target
    selected = difference if mask is None else difference[mask]
    mse = float(selected.square().mean().item())
    return (math.inf if mse == 0 else 10 * math.log10(1 / mse), mse)


def _global_masked_ssim(prediction: Any, target: Any, mask: Any) -> float:
    # Declared deterministic global-window RGB SSIM over strict visible ROI pixels.
    torch = _torch()
    x = prediction[mask].to(torch.float64)
    y = target[mask].to(torch.float64)
    mu_x, mu_y = x.mean(dim=0), y.mean(dim=0)
    dx, dy = x - mu_x, y - mu_y
    variance_x = dx.square().mean(dim=0)
    variance_y = dy.square().mean(dim=0)
    covariance = (dx * dy).mean(dim=0)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return float(score.mean().item())


def _strict_panel_mask(panel_fields: dict[str, Any], camera: Any) -> tuple[Any, int]:
    torch = _torch()
    with torch.no_grad():
        panel = render_fields(panel_fields, camera)
    visible = panel.alpha > 0.05
    if visible.any():
        kernel = torch.ones((1, 1, 5, 5), device=visible.device)
        count = torch.nn.functional.conv2d(
            visible.to(torch.float32)[None, None], kernel, padding=2
        )[0, 0]
        strict = count == 25
    else:
        strict = visible
    pixels = int(strict.sum().item())
    if pixels < 64:
        raise StructuringError("held-out panel ROI mask is missing or too small")
    return strict, pixels


def _q8_tensor(root: Path, device: str, method: str) -> Any:
    torch = _torch()
    record = load_object(root / "fits" / method / "residuals.json")
    values = torch.tensor(record["values"], dtype=torch.float32, device=device)
    return values * float(record["scale"])


def evaluate(*, hybrid_root: Path, overwrite: bool = False) -> dict[str, Any]:
    """Evaluate all validated cameras for three method conditions."""

    validate_hybrid(hybrid_root)
    manifest = load_object(hybrid_root / "hybrid.json")
    declared_methods = manifest.get("method_ids")
    if isinstance(declared_methods, dict):
        shared_method = str(declared_methods["shared"])
        compact_method = str(declared_methods["compact"])
    else:
        pairs = (
            (SHARED_METHOD, SH1_Q8_METHOD),
            (SHARED_METHOD, Q8_METHOD),
            (DRJOHNSON_SHARED_METHOD, DRJOHNSON_Q8_METHOD),
            (TRUCK_BAY_SHARED_METHOD, TRUCK_BAY_Q8_METHOD),
        )
        pair = next(
            (
                item
                for item in pairs
                if (hybrid_root / "fits" / item[0] / "fit-summary.json").is_file()
                and (hybrid_root / "fits" / item[1] / "fit-summary.json").is_file()
            ),
            None,
        )
        if pair is None:
            raise StructuringError("structured fitting checkpoints are missing")
        shared_method, compact_method = pair
    if not (hybrid_root / "fits" / shared_method / "fit-summary.json").is_file():
        raise StructuringError("shared fitting checkpoint is missing")
    if not (hybrid_root / "fits" / compact_method / "fit-summary.json").is_file():
        raise StructuringError("q8 fitting checkpoint is missing")
    output = hybrid_root / "evaluation"
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    dataset_root = Path(str(manifest["local_dataset_root"]))
    split = json.loads(
        (dataset_root / "annotations" / "evaluation-split.json").read_text(
            encoding="utf-8"
        )
    )
    records = [
        cast(Mapping[str, Any], record)
        for record in split["cameras"]
        if str(record["camera_id"]).startswith("cam-eval-")
    ]
    expected = tuple(str(value) for value in manifest["evaluation_camera_ids"])
    if tuple(str(record["camera_id"]) for record in records) != expected:
        raise StructuringError("held-out evaluation camera identity mismatch")
    if len(records) != 8:
        raise StructuringError("exactly eight frozen held-out cameras are required")
    hybrid = HybridTensors.load(hybrid_root)
    shared_sh = _load_shared(hybrid_root, hybrid.device, shared_method)
    q8 = _q8_tensor(hybrid_root, hybrid.device, compact_method)
    explicit = RealGaussianTensorSource.load(
        dataset_root / str(manifest["converted_root_relative"])
    ).tensors(hybrid.device)
    methods = ("official_explicit_gs", shared_method, compact_method)
    observations: list[dict[str, Any]] = []
    render_dir = output / "renders"
    render_dir.mkdir()
    torch.cuda.reset_peak_memory_stats()
    for record in records:
        camera = _camera_from_record(record)
        target = _source_image(
            dataset_root / "source" / "images" / str(record["image_name"]),
            hybrid.device,
        )
        with torch.no_grad():
            explicit_rgb = render_fields(explicit, camera).rgb.clamp(0, 1)
            shared_panels = hybrid.panels(appearance=shared_sh)
            roi, roi_pixels = _strict_panel_mask(shared_panels, camera)
            shared_rgb = render_fields(
                hybrid.materialize(appearance=shared_sh), camera
            ).rgb.clamp(0, 1)
            q8_rgb = render_fields(
                hybrid.materialize(appearance=shared_sh, residuals=q8), camera
            ).rgb.clamp(0, 1)
        predictions = {
            "official_explicit_gs": explicit_rgb,
            shared_method: shared_rgb,
            compact_method: q8_rgb,
        }
        view_dir = render_dir / str(record["camera_id"])
        view_dir.mkdir()
        for method, prediction in predictions.items():
            full_psnr, full_mse = _psnr(prediction, target)
            roi_psnr, roi_mse = _psnr(prediction, target, roi)
            explicit_psnr, _ = _psnr(prediction, explicit_rgb, roi)
            maximum = float((prediction[roi] - target[roi]).abs().max().item())
            observations.append(
                {
                    "camera_id": record["camera_id"],
                    "image_name": record["image_name"],
                    "method_id": method,
                    "render_complete": True,
                    "full_frame_psnr_db": full_psnr,
                    "full_frame_mse": full_mse,
                    "panel_roi_psnr_db": roi_psnr,
                    "panel_roi_mse": roi_mse,
                    "panel_roi_ssim": _global_masked_ssim(prediction, target, roi),
                    "panel_roi_maximum_rgb_error": maximum,
                    "valid_roi_pixel_count": roi_pixels,
                    "explicit_vs_structured_roi_psnr_db": (
                        "positive_infinity"
                        if method == "official_explicit_gs"
                        else explicit_psnr
                    ),
                    "mask_policy": (
                        "shared materialized panel alpha > 0.05 eroded by 2 px; "
                        "same frozen mask for every method"
                    ),
                    "warnings": [
                        "exploratory scene-specific held-out diagnostic",
                        "global-window strict-ROI RGB SSIM",
                    ],
                }
            )
            _write_png(view_dir / f"{method}.png", prediction)
        mask_path = view_dir / "strict-roi.u8"
        mask_path.write_bytes(bytes(roi.detach().to("cpu", torch.uint8).flatten()))
    aggregates: dict[str, Any] = {}
    for method in methods:
        selected = [item for item in observations if item["method_id"] == method]
        aggregates[method] = {
            "view_count": len(selected),
            "mean_full_frame_psnr_db": sum(
                float(item["full_frame_psnr_db"]) for item in selected
            )
            / len(selected),
            "mean_panel_roi_psnr_db": sum(
                float(item["panel_roi_psnr_db"]) for item in selected
            )
            / len(selected),
            "minimum_panel_roi_psnr_db": min(
                float(item["panel_roi_psnr_db"]) for item in selected
            ),
            "mean_panel_roi_ssim": sum(
                float(item["panel_roi_ssim"]) for item in selected
            )
            / len(selected),
        }
    improvement = (
        aggregates[compact_method]["mean_panel_roi_psnr_db"]
        - aggregates[shared_method]["mean_panel_roi_psnr_db"]
    )
    result = {
        "pilot_version": str(manifest["pilot_version"]),
        "status": "completed",
        "evidence_class": "local-only qualitative real-benchmark pilot",
        "oracle": True,
        "scene_specific": True,
        "target_type": "withheld real benchmark photographs",
        "evaluation_camera_ids": list(expected),
        "fitting_camera_ids": manifest["fitting_camera_ids"],
        "method_ids": list(methods),
        "compact_residual_method_id": compact_method,
        "observations": observations,
        "aggregates": aggregates,
        "q8_roi_psnr_improvement_db_over_shared": improvement,
        "compact_residual_roi_psnr_improvement_db_over_shared": improvement,
        "q8_improves_or_preserves_shared": improvement >= -1e-8,
        "hero_camera_id": manifest.get("hero_camera_id"),
        "resource": {
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "camera_batch_size": 1,
        },
        "local_outputs": (
            "renders contain restricted derived imagery and masks; never stage"
        ),
    }
    result["scientific_digest"] = content_digest(result)
    atomic_json(output / "evaluation-summary.json", result)
    return result


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise StructuringError("PyTorch is required for held-out evaluation") from error
    return torch
