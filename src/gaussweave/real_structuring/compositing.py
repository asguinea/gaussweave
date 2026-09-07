"""Native differentiable panel rendering and fixed-background compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gaussweave.real_structuring.models import StructuringError
from gaussweave.rendering.models import Camera


@dataclass(frozen=True)
class Plate:
    rgb: Any
    alpha: Any
    depth: Any


def render_fields(fields: dict[str, Any], camera: Camera) -> Plate:
    """Render semantic Gaussian fields with native gsplat, preserving gradients."""

    torch, rasterization = _dependencies()
    viewmat, intrinsics = camera.gsplat_matrices(str(fields["means"].device))
    rendered, alpha, _metadata = rasterization(
        fields["means"],
        fields["quaternions"],
        fields["scales"],
        fields["opacities"],
        fields["appearance"],
        viewmat.unsqueeze(0),
        intrinsics.unsqueeze(0),
        width=camera.width,
        height=camera.height,
        near_plane=camera.near,
        far_plane=camera.far,
        sh_degree=3,
        packed=True,
        backgrounds=None,
        render_mode="RGB+ED",
        rasterize_mode="classic",
    )
    rgb = rendered[0, ..., :3]
    depth = rendered[0, ..., 3]
    return Plate(rgb, alpha[0, ..., 0], depth)


def composite_panel_over_background(
    panel: Plate, background: Plate, *, depth_tolerance_m: float = 0.03
) -> Any:
    """Composite a differentiable panel plate with fixed depth-based visibility."""

    torch = _torch()
    finite_panel = torch.isfinite(panel.depth)
    finite_background = torch.isfinite(background.depth)
    visible = finite_panel & (
        (~finite_background)
        | (background.alpha < 1e-4)
        | (panel.depth <= background.depth + depth_tolerance_m)
    )
    visibility = visible.to(panel.rgb.dtype)
    alpha = panel.alpha * visibility
    return panel.rgb * visibility.unsqueeze(-1) + background.rgb * (
        1.0 - alpha.unsqueeze(-1)
    )


def crop_camera(camera: Camera, box: tuple[int, int, int, int]) -> Camera:
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > camera.width or y1 > camera.height:
        raise StructuringError("camera crop exceeds image bounds")
    return Camera(
        camera_id=camera.camera_id,
        width=x1 - x0,
        height=y1 - y0,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx - x0,
        cy=camera.cy - y0,
        world_from_camera=camera.world_from_camera,
        near=camera.near,
        far=camera.far,
    )


def _dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from gsplat import rasterization  # type: ignore[import-untyped]
    except (ImportError, OSError) as error:
        raise StructuringError(
            "locked CUDA/gsplat environment is unavailable"
        ) from error
    if not torch.cuda.is_available():
        raise StructuringError("CUDA is required for real panel fitting")
    return torch, rasterization


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise StructuringError("PyTorch is required for compositing") from error
    return torch
