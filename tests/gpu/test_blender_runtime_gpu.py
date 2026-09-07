from __future__ import annotations

from pathlib import Path

import pytest

from gaussweave.data.blender import BlenderBackend, BlenderDevice, discover_blender
from gaussweave.data.blender_runtime import run_blender_scene_generation

pytestmark = [pytest.mark.gpu, pytest.mark.blender, pytest.mark.wsl]
REPOSITORY = Path(__file__).resolve().parents[2]


def test_runtime_probe_accepts_gpu_device_and_strictly_verifies(
    tmp_path: Path,
) -> None:
    try:
        executable = discover_blender(backend=BlenderBackend.WSL)
    except Exception as error:
        pytest.skip(f"Blender unavailable for GPU runtime probe: {error}")
    result = run_blender_scene_generation(
        REPOSITORY / "configs/scenes/colonnade-development-s303.json",
        tmp_path / "gpu-runtime",
        blender_executable=executable,
        device=BlenderDevice.GPU,
    )
    assert result.valid
    assert result.device == "gpu"
    assert result.inventory_state == "valid"
    assert result.preview_digest
