"""Lazy project adapter for native gsplat 1.5.3 rendering."""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from gaussweave.accounting.resources import load_profile, measure_operation
from gaussweave.config.resolution import pretty_json_bytes
from gaussweave.rendering.fixtures import separated_gaussians, smoke_cameras
from gaussweave.rendering.models import (
    Camera,
    GaussianTensorSource,
    RenderResult,
    RenderSettings,
)

EXPECTED_GSPLAT_VERSION = "1.5.3"
REGRESSION_MAX_ABS_TOLERANCE = 1e-5


class RenderingError(RuntimeError):
    """Native renderer environment or output is invalid."""


class Renderer(Protocol):
    def render(
        self,
        scene: GaussianTensorSource,
        cameras: Camera | Sequence[Camera],
        settings: RenderSettings,
        *,
        profile_name: str = "smoke",
        resource_output: Path | None = None,
    ) -> RenderResult: ...


class GsplatRenderer:
    """Concrete lazy native gsplat adapter returning project-owned results."""

    name = "gsplat"

    def render(
        self,
        scene: GaussianTensorSource,
        cameras: Camera | Sequence[Camera],
        settings: RenderSettings,
        *,
        profile_name: str = "smoke",
        resource_output: Path | None = None,
    ) -> RenderResult:
        torch, gsplat, rasterization = _dependencies()
        if not settings.device.startswith("cuda") or not torch.cuda.is_available():
            raise RenderingError(
                "native gsplat rendering requires an available CUDA device"
            )
        camera_batch = (cameras,) if isinstance(cameras, Camera) else tuple(cameras)
        if not camera_batch:
            raise RenderingError("at least one camera is required")
        if len(camera_batch) > settings.camera_batch_size:
            raise RenderingError("camera batch exceeds declared setting")
        dimensions = {(camera.width, camera.height) for camera in camera_batch}
        if len(dimensions) != 1:
            raise RenderingError("all cameras in a native batch need equal dimensions")
        width, height = dimensions.pop()
        tensors = scene.tensors(settings.device)
        matrices = [camera.gsplat_matrices(settings.device) for camera in camera_batch]
        viewmats = torch.stack([item[0] for item in matrices])
        intrinsics = torch.stack([item[1] for item in matrices])
        background = torch.tensor(
            settings.background_rgb,
            dtype=torch.float32,
            device=settings.device,
        )
        near = max(camera.near for camera in camera_batch)
        far = min(camera.far for camera in camera_batch)
        if near >= far:
            raise RenderingError("camera batch has incompatible clipping planes")
        depth_requested = "depth" in settings.output_buffers
        render_mode = (
            "RGB+ED"
            if depth_requested and settings.depth_mode == "expected"
            else "RGB+D"
            if depth_requested
            else "RGB"
        )
        latest: tuple[Any, Any, dict[str, Any]] | None = None

        def operation() -> None:
            nonlocal latest
            latest = rasterization(
                tensors["means"],
                tensors["quaternions"],
                tensors["scales"],
                tensors["opacities"],
                tensors["appearance"],
                viewmats,
                intrinsics,
                width=width,
                height=height,
                near_plane=near,
                far_plane=far,
                sh_degree=scene.sh_degree if scene.appearance_mode == "sh" else None,
                packed=settings.packed,
                backgrounds=None,
                render_mode=render_mode,
                rasterize_mode="antialiased" if settings.antialiasing else "classic",
            )

        resource = measure_operation(
            operation,
            operation_name="gsplat-render",
            profile=load_profile(profile_name),
            workspace_root=Path.cwd(),
            device=settings.device,
            warmup_count=settings.warmup_count,
            repetition_count=settings.repetition_count,
            output=resource_output,
        )
        if latest is None:
            raise RenderingError("gsplat produced no measured render")
        rendered, alpha, native_metadata = latest
        expected_shape = (len(camera_batch), height, width)
        if tuple(rendered.shape[:3]) != expected_shape:
            raise RenderingError(f"unexpected rendered shape: {tuple(rendered.shape)}")
        if tuple(alpha.shape) != (*expected_shape, 1):
            raise RenderingError(f"unexpected alpha shape: {tuple(alpha.shape)}")
        rgb = rendered[..., :3] + background * (1.0 - alpha)
        depth = rendered[..., 3:4] if depth_requested else None
        if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
            raise RenderingError("gsplat returned nonfinite RGB or alpha")
        unclamped_rgb_min = float(rgb.min().item())
        unclamped_rgb_max = float(rgb.max().item())
        if scene.appearance_mode == "sh":
            rgb = rgb.clamp(0.0, 1.0)
        elif rgb.min() < -1e-6 or rgb.max() > 1.000001:
            raise RenderingError("RGB output is outside documented [0, 1] range")
        if alpha.min() < -1e-6 or alpha.max() > 1.000001:
            raise RenderingError("alpha output is outside [0, 1]")
        if depth is not None:
            foreground = alpha[..., 0] > 1e-5
            if foreground.any() and not torch.isfinite(depth[..., 0][foreground]).all():
                raise RenderingError("foreground depth contains nonfinite values")
        metadata = {
            "appearance_mode": scene.appearance_mode,
            "sh_degree": scene.sh_degree,
            "coefficient_shape": list(scene.coefficient_shape),
            "color_activation": scene.color_activation,
            "display_clamp": scene.appearance_mode == "sh",
            "unclamped_rgb_range": [unclamped_rgb_min, unclamped_rgb_max],
            "depth_semantics": (
                "expected_z_depth=sum(w*z)/sum(w)"
                if settings.depth_mode == "expected"
                else "accumulated_z_depth=sum(w*z)"
            )
            if depth_requested
            else None,
            "packed": settings.packed,
            "antialiasing": settings.antialiasing,
            "deterministic_smoke": settings.deterministic_smoke,
            "coordinate_convention": "OpenCV world-to-camera, +z forward",
            "native_meta_keys": sorted(str(key) for key in native_metadata),
            "regression_max_abs_tolerance": REGRESSION_MAX_ABS_TOLERANCE,
        }
        return RenderResult(
            rgb,
            alpha,
            depth,
            width,
            height,
            tuple(camera.camera_id for camera in camera_batch),
            scene.count,
            scene.count,
            metadata,
            resource.timing.median_seconds or 0.0,
            resource,
            self.name,
            getattr(gsplat, "__version__", "unknown"),
        )


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import gsplat  # type: ignore[import-untyped]
        import torch
        from gsplat import rasterization
    except (ImportError, OSError) as error:
        raise RenderingError(
            "locked PyTorch/CUDA/gsplat environment is unavailable"
        ) from error
    version = getattr(gsplat, "__version__", "unknown")
    if version != EXPECTED_GSPLAT_VERSION:
        raise RenderingError(
            f"gsplat {EXPECTED_GSPLAT_VERSION} is required, found {version}"
        )
    return torch, gsplat, rasterization


def write_smoke_outputs(
    result: RenderResult, output: Path, *, overwrite: bool = False
) -> dict[str, str]:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise RenderingError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for index, camera_id in enumerate(result.camera_ids):
        rgb_path = output / f"{camera_id}-rgb.ppm"
        alpha_path = output / f"{camera_id}-alpha.pgm"
        _write_ppm(rgb_path, result.rgb[index])
        _write_pgm(alpha_path, result.alpha[index, ..., 0])
        paths[f"{camera_id}_rgb"] = rgb_path.name
        paths[f"{camera_id}_alpha"] = alpha_path.name
        if result.depth is not None:
            depth_path = output / f"{camera_id}-depth.pfm"
            _write_pfm(depth_path, result.depth[index, ..., 0])
            paths[f"{camera_id}_depth"] = depth_path.name
    metadata: Any = {
        "renderer": result.renderer_name,
        "renderer_version": result.renderer_version,
        "width": result.width,
        "height": result.height,
        "camera_ids": list(result.camera_ids),
        "stored_gaussian_count": result.stored_gaussian_count,
        "submitted_gaussian_count": result.submitted_gaussian_count,
        "elapsed_seconds": result.elapsed_seconds,
        "metadata": result.metadata,
        "outputs": paths,
    }
    _atomic_write(output / "render-metadata.json", pretty_json_bytes(metadata))
    resource = result.resource_record
    _atomic_write(
        output / "resource-record.json", pretty_json_bytes(resource.to_dict())
    )
    paths["metadata"] = "render-metadata.json"
    paths["resource"] = "resource-record.json"
    return paths


def _write_ppm(path: Path, image: Any) -> None:
    values = image.detach().clamp(0, 1).mul(255).round().to("cpu")
    payload = bytes(int(value) for value in values.flatten().tolist())
    _atomic_write(
        path, f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode() + payload
    )


def _write_pgm(path: Path, image: Any) -> None:
    values = image.detach().clamp(0, 1).mul(65535).round().to("cpu")
    payload = b"".join(
        struct.pack(">H", int(value)) for value in values.flatten().tolist()
    )
    _atomic_write(
        path, f"P5\n{image.shape[1]} {image.shape[0]}\n65535\n".encode() + payload
    )


def _write_pfm(path: Path, image: Any) -> None:
    rows = image.detach().to("cpu").flip(0).tolist()
    payload = b"".join(struct.pack("<f", float(value)) for row in rows for value in row)
    _atomic_write(
        path, f"Pf\n{image.shape[1]} {image.shape[0]}\n-1.0\n".encode() + payload
    )


def _atomic_write(path: Path, content: bytes) -> None:
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
        raise RenderingError(f"unable to write {path}: {error}") from error
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--profile", default="smoke")
    smoke.add_argument("--output", required=True, type=Path)
    smoke.add_argument("--no-depth", action="store_true")
    smoke.add_argument("--overwrite", action="store_true")
    smoke.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        if (
            options.output.exists()
            and (not options.output.is_dir() or any(options.output.iterdir()))
            and not options.overwrite
        ):
            raise RenderingError(
                f"refusing to overwrite nonempty output: {options.output}"
            )
        settings = RenderSettings(
            output_buffers=(
                ("rgb", "alpha") if options.no_depth else ("rgb", "alpha", "depth")
            )
        )
        result = GsplatRenderer().render(
            separated_gaussians(),
            smoke_cameras(),
            settings,
            profile_name=options.profile,
        )
        outputs = write_smoke_outputs(
            result, options.output, overwrite=options.overwrite
        )
    except (RenderingError, ValueError) as error:
        print(
            json.dumps({"valid": False, "error": str(error)}) if options.json else error
        )
        return 1
    summary = {
        "valid": True,
        "renderer": result.renderer_name,
        "version": result.renderer_version,
        "shape": list(result.rgb.shape),
        "alpha_shape": list(result.alpha.shape),
        "depth_shape": list(result.depth.shape) if result.depth is not None else None,
        "median_seconds": result.elapsed_seconds,
        "peak_allocated_bytes": result.resource_record.gpu_final.peak_allocated_bytes,
        "peak_reserved_bytes": result.resource_record.gpu_final.peak_reserved_bytes,
        "compliance": result.resource_record.compliance.state.value,
        "outputs": outputs,
    }
    print(
        json.dumps(summary, indent=2, sort_keys=True)
        if options.json
        else (
            f"gsplat {result.renderer_version}: {tuple(result.rgb.shape)}; "
            f"median={result.elapsed_seconds:.6f}s; "
            f"compliance={summary['compliance']}; output={options.output}"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
