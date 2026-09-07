from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gaussweave.data.blender import BlenderBackend, BlenderDevice, discover_blender
from gaussweave.data.blender_runtime import run_blender_scene_generation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/scenes/facade-development-s101.json"

pytestmark = [pytest.mark.integration, pytest.mark.blender, pytest.mark.wsl]


@pytest.fixture(scope="module")
def blender_executable() -> Path:
    if not os.environ.get("WSL_DISTRO_NAME"):
        pytest.skip("camera installation probe requires WSL")
    return discover_blender(backend=BlenderBackend.WSL)


def test_blender_camera_install_projection_and_split_renders(
    tmp_path: Path, blender_executable: Path
) -> None:
    result = run_blender_scene_generation(
        CONFIG,
        tmp_path / "camera-probe",
        mode="camera-probe",
        backend=BlenderBackend.WSL,
        device=BlenderDevice.CPU,
        blender_executable=blender_executable,
    )
    assert result.valid
    installation = json.loads(
        (tmp_path / "camera-probe/source/camera_installation.json").read_text()
    )
    assert installation["installed_camera_count"] == 40
    assert len(installation["rendered_camera_ids"]) == 3
    assert max(item["error_px"] for item in installation["projection_checks"]) <= 1e-4
    assert len(list((tmp_path / "camera-probe/preview/cameras").glob("*.png"))) == 3


def test_blender_camera_probe_same_seed_is_scientifically_stable(
    tmp_path: Path, blender_executable: Path
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        run_blender_scene_generation(
            CONFIG,
            root,
            mode="camera-probe",
            backend=BlenderBackend.WSL,
            device=BlenderDevice.CPU,
            blender_executable=blender_executable,
        )
    first = json.loads((roots[0] / "cameras/cameras.json").read_text())
    second = json.loads((roots[1] / "cameras/cameras.json").read_text())
    assert first == second
