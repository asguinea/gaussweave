"""GPU-backed G1 construction, evidence, and verification."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.rendering.gsplat_renderer import GsplatRenderer
from gaussweave.rendering.models import RenderSettings
from gaussweave.representation.accounting import account_representation
from gaussweave.representation.fixture import (
    build_fixture_representations,
    camera_digest,
    evaluation_cameras,
    fixture_summary,
)
from gaussweave.representation.models import materialize
from gaussweave.representation.serialization import (
    load_representation,
    save_representation,
    validate_representation,
)
from gaussweave.results.artifacts import (
    inventory_tree,
    load_inventory,
    verify_inventory,
    write_inventory,
)

HERO_CAMERA_ID = "cam-eval-0002"
G1_VERSION = "gw-g1-v1"


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
    )


def write_rgb_png(path: Path, image: Any) -> None:
    """Write a torch [H,W,3] RGB tensor as a compact deterministic PNG."""

    values = image.detach().clamp(0, 1).mul(255).round().to("cpu")
    height, width, channels = values.shape
    if channels != 3:
        raise ValueError("RGB PNG input must have exactly three channels")
    raw_values = bytes(int(value) for value in values.flatten().tolist())
    scanlines = b"".join(
        b"\x00" + raw_values[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    _write(path, payload)


def _metric_value(value: float) -> float | str:
    return "positive_infinity" if math.isinf(value) and value > 0 else value


def _fidelity(
    candidate: Any, reference: Any, camera_ids: tuple[str, ...]
) -> dict[str, Any]:
    import torch

    difference = candidate.double() - reference.double()
    mse = difference.square().mean(dim=(1, 2, 3))
    max_abs = difference.abs().amax(dim=(1, 2, 3))
    psnr = torch.where(
        mse == 0, torch.full_like(mse, torch.inf), 10.0 * torch.log10(1.0 / mse)
    )
    per_view = []
    for index, camera_id in enumerate(camera_ids):
        per_view.append(
            {
                "camera_id": camera_id,
                "mse": float(mse[index].item()),
                "psnr_db": _metric_value(float(psnr[index].item())),
                "max_abs_error": float(max_abs[index].item()),
                "exact_tensor_equality": bool(
                    torch.equal(candidate[index], reference[index])
                ),
            }
        )
    finite_psnr = [float(value.item()) for value in psnr if not torch.isinf(value)]
    aggregate: dict[str, Any]
    if len(finite_psnr) != len(camera_ids):
        aggregate = {
            "psnr_mean_db": "positive_infinity",
            "psnr_min_db": "positive_infinity",
            "perfect_view_count": len(camera_ids) - len(finite_psnr),
        }
    else:
        aggregate = {
            "psnr_mean_db": sum(finite_psnr) / len(finite_psnr),
            "psnr_min_db": min(finite_psnr),
            "perfect_view_count": 0,
        }
    aggregate["mse_mean"] = float(mse.mean().item())
    aggregate["max_abs_error"] = float(max_abs.max().item())
    return {"per_view": per_view, "aggregate": aggregate}


def _error_image(candidate: Any, reference: Any) -> Any:
    import torch

    error = (candidate - reference).abs().amax(dim=-1, keepdim=True)
    maximum = torch.clamp(error.max(), min=1e-8)
    normalized = torch.clamp(error / maximum, 0.0, 1.0)
    return torch.cat(
        (normalized, normalized.square(), torch.zeros_like(normalized)), dim=-1
    )


def _render(scene: Any, cameras: tuple[Any, ...], resource_path: Path) -> Any:
    settings = RenderSettings(
        output_buffers=("rgb", "alpha", "depth"),
        background_rgb=(0.025, 0.035, 0.055),
        camera_batch_size=8,
        warmup_count=1,
        repetition_count=3,
        device="cuda:0",
    )
    return GsplatRenderer().render(
        scene.to_renderable(),
        cameras,
        settings,
        profile_name="smoke",
        resource_output=resource_path,
    )


def _save_models(root: Path, exact: Any, low: Any) -> dict[str, Path]:
    models = {
        "explicit_exact": (exact.explicit, root / "representations" / "explicit-exact"),
        "shared_exact": (
            exact.shared,
            root / "representations" / "struct-shared-exact",
        ),
        "explicit_low_variation": (
            low.explicit,
            root / "representations" / "explicit-low-variation",
        ),
        "shared_low_variation": (
            low.shared,
            root / "representations" / "struct-shared-low-variation",
        ),
        "residual_q8_low_variation": (
            low.residual_q8,
            root / "representations" / "struct-residual-q8-low-variation",
        ),
    }
    paths: dict[str, Path] = {}
    for key, (model, path) in models.items():
        if model is None:
            raise ValueError(f"missing model for {key}")
        save_representation(model, path)
        paths[key] = path
    return paths


def _account_models(root: Path, paths: dict[str, Path]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    reference = paths["explicit_low_variation"]
    for key, path in paths.items():
        record = account_representation(
            path,
            reference_root=reference if key != "explicit_low_variation" else None,
        )
        results[key] = record
        _write(root / "accounting" / f"{key}.json", pretty_json_bytes(record))
    return results


def _representation_summaries(
    output_root: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for key, path in paths.items():
        record = validate_representation(path)
        record["path"] = path.relative_to(output_root).as_posix()
        summaries[key] = record
    return summaries


def _meaningful_view(rgb: Any) -> bool:
    # Foreground must differ from the fixed dark background in at least 0.5% of pixels.
    import torch

    background = torch.tensor((0.025, 0.035, 0.055), device=rgb.device)
    foreground = (rgb - background).abs().amax(dim=-1) > 0.02
    return bool(foreground.float().mean().item() >= 0.005)


def build_g1(
    *,
    repo_root: Path,
    output_root: Path,
    principal_seed: int = 17,
    publish_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"G1 output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    exact = build_fixture_representations(
        repo_root=repo_root,
        appearance_regime="exact",
        principal_seed=principal_seed,
    )
    low = build_fixture_representations(
        repo_root=repo_root,
        appearance_regime="low_variation",
        principal_seed=principal_seed,
    )
    if low.residual_q8 is None:
        raise ValueError("low-variation fixture must include q8 residuals")
    exact_materialized = materialize(exact.shared)
    low_shared_materialized = materialize(low.shared)
    low_q8_materialized = materialize(low.residual_q8)
    exact_array_match = exact_materialized.gaussians == exact.explicit.gaussians
    if not exact_array_match:
        raise ValueError(
            "exact structural materialization differs from explicit reference arrays"
        )
    cameras = evaluation_cameras(repo_root=repo_root, principal_seed=principal_seed)
    paths = _save_models(output_root, exact, low)
    accounting = _account_models(output_root, paths)
    resource_dir = output_root / "resources"
    renders = {
        "explicit_exact": _render(
            exact.explicit.gaussians, cameras, resource_dir / "explicit-exact.json"
        ),
        "shared_exact": _render(
            exact_materialized.gaussians, cameras, resource_dir / "shared-exact.json"
        ),
        "explicit_low_variation": _render(
            low.explicit.gaussians,
            cameras,
            resource_dir / "explicit-low-variation.json",
        ),
        "shared_low_variation": _render(
            low_shared_materialized.gaussians,
            cameras,
            resource_dir / "shared-low-variation.json",
        ),
        "residual_q8_low_variation": _render(
            low_q8_materialized.gaussians,
            cameras,
            resource_dir / "residual-q8-low-variation.json",
        ),
    }
    exact_metrics = _fidelity(
        renders["shared_exact"].rgb,
        renders["explicit_exact"].rgb,
        renders["explicit_exact"].camera_ids,
    )
    shared_metrics = _fidelity(
        renders["shared_low_variation"].rgb,
        renders["explicit_low_variation"].rgb,
        renders["explicit_low_variation"].camera_ids,
    )
    q8_metrics = _fidelity(
        renders["residual_q8_low_variation"].rgb,
        renders["explicit_low_variation"].rgb,
        renders["explicit_low_variation"].camera_ids,
    )
    exact_all = all(item["exact_tensor_equality"] for item in exact_metrics["per_view"])
    q8_improves_all = all(
        float(q8["psnr_db"]) > float(shared["psnr_db"])
        for q8, shared in zip(
            q8_metrics["per_view"],
            shared_metrics["per_view"],
            strict=True,
        )
    )
    q8_improves_aggregate = float(q8_metrics["aggregate"]["psnr_mean_db"]) > float(
        shared_metrics["aggregate"]["psnr_mean_db"]
    ) and float(q8_metrics["aggregate"]["psnr_min_db"]) > float(
        shared_metrics["aggregate"]["psnr_min_db"]
    )
    meaningful = {
        key: [
            camera_id
            for camera_id, image in zip(
                result.camera_ids,
                result.rgb,
                strict=True,
            )
            if _meaningful_view(image)
        ]
        for key, result in renders.items()
    }
    evidence = output_root / "evidence"
    hero_index = renders["explicit_exact"].camera_ids.index(HERO_CAMERA_ID)
    hero_images = {
        "hero-exact-explicit.png": renders["explicit_exact"].rgb[hero_index],
        "hero-exact-shared.png": renders["shared_exact"].rgb[hero_index],
        "hero-low-reference.png": renders["explicit_low_variation"].rgb[hero_index],
        "hero-low-shared.png": renders["shared_low_variation"].rgb[hero_index],
        "hero-low-residual-q8.png": renders["residual_q8_low_variation"].rgb[
            hero_index
        ],
        "hero-low-shared-error.png": _error_image(
            renders["shared_low_variation"].rgb[hero_index],
            renders["explicit_low_variation"].rgb[hero_index],
        ),
        "hero-low-q8-error.png": _error_image(
            renders["residual_q8_low_variation"].rgb[hero_index],
            renders["explicit_low_variation"].rgb[hero_index],
        ),
    }
    for name, image in hero_images.items():
        write_rgb_png(evidence / name, image)
    import torch

    separator = (
        torch.ones((256, 4, 3), device=next(iter(hero_images.values())).device) * 0.92
    )
    composite = torch.cat(
        [
            hero_images["hero-exact-explicit.png"],
            separator,
            hero_images["hero-low-reference.png"],
            separator,
            hero_images["hero-low-shared.png"],
            separator,
            hero_images["hero-low-residual-q8.png"],
            separator,
            hero_images["hero-low-q8-error.png"],
        ],
        dim=1,
    )
    write_rgb_png(evidence / "hero-composite.png", composite)
    renderer_metadata = {
        key: {
            "renderer": result.renderer_name,
            "renderer_version": result.renderer_version,
            "elapsed_seconds": result.elapsed_seconds,
            "camera_ids": list(result.camera_ids),
            "stored_gaussian_count": result.stored_gaussian_count,
            "submitted_gaussian_count": result.submitted_gaussian_count,
            "metadata": result.metadata,
        }
        for key, result in renders.items()
    }
    metrics = {
        "exact_shared_vs_reference": exact_metrics,
        "low_shared_vs_reference": shared_metrics,
        "low_residual_q8_vs_reference": q8_metrics,
    }
    _write(evidence / "g1-metrics.json", pretty_json_bytes(cast(Any, metrics)))
    q8_residuals = low.residual_q8.residuals
    if q8_residuals is None:
        raise ValueError("q8 representation unexpectedly lacks residuals")
    summary: dict[str, Any] = {
        "g1_version": G1_VERSION,
        "state": "passed"
        if exact_all and q8_improves_all and q8_improves_aggregate
        else "failed",
        "principal_seed": principal_seed,
        "fixture": fixture_summary(repo_root=repo_root, principal_seed=principal_seed),
        "camera_digest": camera_digest(cameras),
        "hero_camera_id": HERO_CAMERA_ID,
        "method_ids": ["explicit_gs", "struct_shared", "struct_residual_q8"],
        "appearance_regimes": ["exact", "low_variation"],
        "representations": _representation_summaries(output_root, paths),
        "accounting": accounting,
        "materialization": {
            "exact_array_match": exact_array_match,
            "exact_materialized_digest": exact_materialized.gaussians.scientific_digest,
            "low_shared_materialized_digest": (
                low_shared_materialized.gaussians.scientific_digest
            ),
            "low_q8_materialized_digest": (
                low_q8_materialized.gaussians.scientific_digest
            ),
            "stored_structural_count": exact.shared.stored_gaussian_count,
            "materialized_count": exact.shared.materialized_gaussian_count,
            "q8_saturation_count": q8_residuals.saturation_count,
            "q8_clipping_count": low_q8_materialized.clipping_count,
        },
        "metrics": metrics,
        "renderer": renderer_metadata,
        "meaningful_camera_ids": meaningful,
        "checks": {
            "exact_array_equality": exact_array_match,
            "exact_render_equality_all_8": exact_all,
            "q8_psnr_improves_every_camera": q8_improves_all,
            "q8_psnr_improves_mean_and_min": q8_improves_aggregate,
            "all_8_cameras_meaningful": all(
                len(value) == 8 for value in meaningful.values()
            ),
        },
        "limitations": [
            "direct synthetic Gaussian fixture, not a trained scene",
            "single terminal component and fixed procedural grid",
            "q8 appearance residuals only",
            "materializes ordinary Gaussians before gsplat rendering",
            "no pruning, edits, full matrix, induction, or real scenes in this gate",
        ],
    }
    summary["g1_digest"] = content_digest(summary)
    _write(output_root / "g1-summary.json", pretty_json_bytes(summary))
    inventory = inventory_tree(
        evidence,
        inventory_id="gw-g1-evidence-v1",
        artifact_type="other",
        generated_at="2026-07-29T00:00:00+00:00",
    )
    write_inventory(inventory, output_root / "g1-artifact-inventory.json")
    verification = verify_inventory(inventory, evidence, strict=True)
    if not verification.to_dict()["valid"]:
        raise ValueError("G1 evidence inventory verification failed")
    if publish_root is not None:
        publish_g1(output_root, publish_root, overwrite=overwrite)
    return summary


def _report_markdown(summary: dict[str, Any]) -> str:
    accounting = summary["accounting"]
    exact = summary["metrics"]["exact_shared_vs_reference"]["aggregate"]
    shared = summary["metrics"]["low_shared_vs_reference"]["aggregate"]
    q8 = summary["metrics"]["low_residual_q8_vs_reference"]["aggregate"]
    materialization = summary["materialization"]
    checks = summary["checks"]
    explicit_bytes = accounting["explicit_low_variation"]["complete_serialized_bytes"]
    shared_bytes = accounting["shared_low_variation"]["complete_serialized_bytes"]
    q8_bytes = accounting["residual_q8_low_variation"]["complete_serialized_bytes"]
    shared_ratio = accounting["shared_low_variation"][
        "compression_ratio_reference_over_method"
    ]
    q8_ratio = accounting["residual_q8_low_variation"][
        "compression_ratio_reference_over_method"
    ]
    exact_array_equality = checks["exact_array_equality"]
    shared_psnr = f"{shared['psnr_mean_db']:.6f} / {shared['psnr_min_db']:.6f}"
    return f"""# GaussWeave G1 Representation Report

## Outcome

- Gate state: **{summary["state"].upper()}**
- Fixture: `{summary["fixture"]["fixture_id"]}`
- Principal seed: `{summary["principal_seed"]}`
- Hero camera: `{summary["hero_camera_id"]}`
- Renderer: `gsplat {summary["renderer"]["explicit_exact"]["renderer_version"]}`

## Representation and materialization

- Unique Gaussians: {summary["fixture"]["counts"]["unique"]}
- Canonical terminal Gaussians: {summary["fixture"]["counts"]["terminal"]}
- Hero instances: {summary["fixture"]["counts"]["instances"]}
- Stored structural Gaussians: {materialization["stored_structural_count"]}
- Materialized Gaussians: {materialization["materialized_count"]}
- Exact arrays equal the independent explicit reference: {exact_array_equality}

## Fidelity

- Exact structural PSNR mean/min: `{exact["psnr_mean_db"]}` / `{exact["psnr_min_db"]}`
- Low shared PSNR mean/min: `{shared_psnr}`
- Low q8 PSNR mean/min: `{q8["psnr_mean_db"]:.6f}` / `{q8["psnr_min_db"]:.6f}`
- q8 improves every camera: {checks["q8_psnr_improves_every_camera"]}
- q8 improves mean and minimum PSNR: {checks["q8_psnr_improves_mean_and_min"]}
- All eight cameras are meaningful: {checks["all_8_cameras_meaningful"]}

## Actual complete serialized bytes

- Explicit low variation: {explicit_bytes}
- Shared low variation: {shared_bytes}
- q8 residual low variation: {q8_bytes}
- Shared compression ratio (explicit/shared): {shared_ratio:.6f}
- q8 compression ratio (explicit/q8): {q8_ratio:.6f}

## Limits

This is the bounded G1 direct-Gaussian fixture only. It does not claim training,
pruning, runtime memory reduction, edits, learned structure induction, real-scene
generalization, or coverage beyond the documented fixture.
"""


def publish_g1(output_root: Path, publish_root: Path, *, overwrite: bool) -> None:
    publish_root.mkdir(parents=True, exist_ok=True)
    evidence_target = publish_root / "evidence" / "g1"
    if evidence_target.exists():
        if not overwrite:
            raise FileExistsError(
                f"published evidence already exists: {evidence_target}"
            )
        shutil.rmtree(evidence_target)
    shutil.copytree(output_root / "evidence", evidence_target)
    summary = json.loads((output_root / "g1-summary.json").read_text(encoding="utf-8"))
    destinations = {
        "g1-summary.json": (output_root / "g1-summary.json").read_bytes(),
        "g1-artifact-inventory.json": (
            output_root / "g1-artifact-inventory.json"
        ).read_bytes(),
        "gaussweave_g1_report.md": _report_markdown(summary).encode("utf-8"),
    }
    for name, payload in destinations.items():
        target = publish_root / name
        if target.exists() and not overwrite:
            raise FileExistsError(f"published report already exists: {target}")
        _write(target, payload)


def check_g1(root: Path) -> dict[str, Any]:
    summary_path = root / "g1-summary.json"
    inventory_path = root / "g1-artifact-inventory.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    digest = summary.pop("g1_digest", None)
    digest_valid = digest == content_digest(summary)
    summary["g1_digest"] = digest
    inventory = load_inventory(inventory_path)
    evidence_root = root / "evidence"
    if (root / "evidence" / "g1").is_dir():
        evidence_root = root / "evidence" / "g1"
    verification = verify_inventory(inventory, evidence_root, strict=True)
    representations = {}
    representations_root = root / "representations"
    if representations_root.is_dir():
        for path in sorted(
            item for item in representations_root.iterdir() if item.is_dir()
        ):
            loaded = load_representation(path)
            representations[path.name] = loaded.scientific_digest
    checks = summary.get("checks", {})
    valid = (
        summary.get("state") == "passed"
        and digest_valid
        and verification.to_dict()["valid"]
        and all(bool(value) for value in checks.values())
    )
    return {
        "valid": valid,
        "state": "passed" if valid else "failed",
        "g1_digest_valid": digest_valid,
        "inventory": verification.to_dict(),
        "representations": representations,
        "checks": checks,
    }
