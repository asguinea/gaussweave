"""Live WSL GPU qualification tests for the locked optional environment."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.wsl]


def test_torch_cuda_computation_and_identity() -> None:
    import torch

    assert torch.cuda.is_available()
    value = (torch.tensor([1.0, 2.0], device="cuda") ** 2).sum()
    torch.cuda.synchronize()
    assert value.item() == 5.0
    properties = torch.cuda.get_device_properties(0)
    assert "RTX 5070 Laptop GPU" in properties.name
    assert properties.total_memory > 4 * 1024**3


def test_gsplat_native_rasterization() -> None:
    import torch
    from gsplat import rasterization

    device = torch.device("cuda")
    rendered, alpha, _ = rasterization(
        torch.tensor([[0.0, 0.0, 3.0]], device=device),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        torch.full((1, 3), 0.1, device=device),
        torch.tensor([0.9], device=device),
        torch.tensor([[1.0, 0.0, 0.0]], device=device),
        torch.eye(4, device=device).unsqueeze(0),
        torch.tensor(
            [[[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]],
            device=device,
        ),
        width=32,
        height=32,
    )
    torch.cuda.synchronize()
    assert rendered.shape == (1, 32, 32, 3)
    assert rendered.is_cuda and alpha.is_cuda


def test_package_import_remains_gpu_dependency_free() -> None:
    code = (
        "import sys; import gaussweave; "
        "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
