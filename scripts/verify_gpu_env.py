"""Verify the locked PyTorch CUDA and gsplat environment with real GPU work."""

from __future__ import annotations

import json

import gsplat
import torch
from gsplat import rasterization


def main() -> None:
    """Run CUDA tensor and gsplat rasterization smoke operations."""
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")

    device = torch.device("cuda")
    value = (torch.tensor([1.0, 2.0], device=device) ** 2).sum()
    torch.cuda.synchronize()
    if value.item() != 5.0:
        raise RuntimeError(f"unexpected CUDA result: {value.item()}")

    means = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.full((1, 3), 0.1, device=device)
    opacities = torch.tensor([0.9], device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rendered, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        width=32,
        height=32,
    )
    torch.cuda.synchronize()
    if not rendered.is_cuda or not alpha.is_cuda or rendered.shape != (1, 32, 32, 3):
        raise RuntimeError("gsplat rasterization returned unexpected output")

    properties = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "cuda_computation": value.item(),
                "gpu": properties.name,
                "gsplat": getattr(gsplat, "__version__", "unknown"),
                "gsplat_render_shape": list(rendered.shape),
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "vram_bytes": properties.total_memory,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
