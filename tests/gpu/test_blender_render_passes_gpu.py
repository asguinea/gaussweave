from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gaussweave.data.blender import BlenderBackend, BlenderDevice, discover_blender
from gaussweave.data.blender_runtime import run_blender_scene_generation
from gaussweave.data.render_formats import DecodedArray, read_npy_f32, read_png
from gaussweave.data.render_passes import validate_render_root

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.gpu, pytest.mark.blender, pytest.mark.wsl]


def test_qualified_cpu_gpu_render_passes_agree_within_policy(tmp_path: Path) -> None:
    try:
        executable = discover_blender(backend=BlenderBackend.WSL)
    except Exception as error:
        pytest.skip(f"Blender unavailable for GPU render-pass probe: {error}")
    roots = {"cpu": tmp_path / "cpu", "gpu": tmp_path / "gpu"}
    for device, output in roots.items():
        result = run_blender_scene_generation(
            ROOT / "configs/scenes/facade-development-s101.json",
            output,
            mode="render-pass-probe",
            backend=BlenderBackend.WSL,
            device=BlenderDevice(device),
            blender_executable=executable,
        )
        assert result.valid
        assert result.inventory_state == "valid"
        assert validate_render_root(output).valid

    cpu_collection, cpu = _decode(roots["cpu"])
    gpu_collection, gpu = _decode(roots["gpu"])
    assert set(cpu) == set(gpu)
    assert cpu_collection["registry"] == gpu_collection["registry"]
    assert (roots["cpu"] / "cameras/cameras.json").read_bytes() == (
        roots["gpu"] / "cameras/cameras.json"
    ).read_bytes()
    assert (roots["cpu"] / "metadata/render_conventions.json").read_bytes() == (
        roots["gpu"] / "metadata/render_conventions.json"
    ).read_bytes()
    for key in cpu:
        if key[1] in {"semantic", "terminal", "instance", "object_id"}:
            assert cpu[key].decoded_digest == gpu[key].decoded_digest
    assert _max_difference(cpu, gpu, "rgb") <= 1.0
    assert _interior_max_difference(cpu, gpu, "depth") <= 1e-5
    assert _interior_max_difference(cpu, gpu, "normals") <= 0.02


def _decode(
    root: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], DecodedArray]]:
    collection = json.loads(
        (root / "metadata/render_collection.json").read_text(encoding="utf-8")
    )
    decoded = {}
    for view in collection["views"]:
        for artifact in view["pass_artifacts"]:
            path = root / artifact["path"]
            decoded[(view["camera_id"], artifact["pass_name"])] = (
                read_npy_f32(path) if path.suffix == ".npy" else read_png(path)
            )
    return collection, decoded


def _max_difference(
    left: dict[tuple[str, str], DecodedArray],
    right: dict[tuple[str, str], DecodedArray],
    pass_name: str,
) -> float:
    return max(
        abs(float(a) - float(b))
        for key, decoded in left.items()
        if key[1] == pass_name
        for a, b in zip(decoded.values, right[key].values, strict=True)
    )


def _interior_max_difference(
    left: dict[tuple[str, str], DecodedArray],
    right: dict[tuple[str, str], DecodedArray],
    pass_name: str,
) -> float:
    differences = []
    for key, decoded in left.items():
        if key[1] != pass_name:
            continue
        object_ids = left[(key[0], "object_id")]
        for row in range(1, decoded.height - 1):
            for column in range(1, decoded.width - 1):
                labels = {
                    int(object_ids.value(adjacent_row, adjacent_column))
                    for adjacent_row in range(row - 1, row + 2)
                    for adjacent_column in range(column - 1, column + 2)
                }
                if len(labels) != 1:
                    continue
                for channel in range(decoded.channels):
                    differences.append(
                        abs(
                            float(decoded.value(row, column, channel))
                            - float(right[key].value(row, column, channel))
                        )
                    )
    return max(differences)
