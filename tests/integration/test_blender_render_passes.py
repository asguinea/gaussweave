from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from gaussweave.config.resolution import content_digest, pretty_json_bytes
from gaussweave.data.blender import BlenderBackend, BlenderDevice, discover_blender
from gaussweave.data.blender_runtime import run_blender_scene_generation
from gaussweave.data.render_formats import (
    DecodedArray,
    read_npy_f32,
    read_png,
    write_npy_f32,
    write_png_u16,
)
from gaussweave.data.render_passes import RenderValidationError, validate_render_root
from gaussweave.results.artifacts import hash_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/scenes/facade-development-s101.json"

pytestmark = [pytest.mark.integration, pytest.mark.blender, pytest.mark.wsl]


@pytest.fixture(scope="module")
def rendered_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not os.environ.get("WSL_DISTRO_NAME"):
        pytest.skip("render-pass probe requires WSL")
    executable = discover_blender(backend=BlenderBackend.WSL)
    temporary = tmp_path_factory.mktemp("render-pass-probe")
    output = temporary / "root"
    document = _load_json(CONFIG)
    document["outputs"]["alpha"] = True
    document["outputs"]["material_ids"] = True
    config = temporary / "optional-passes.json"
    config.write_bytes(pretty_json_bytes(document))
    result = run_blender_scene_generation(
        config,
        output,
        mode="render-pass-probe",
        backend=BlenderBackend.WSL,
        device=BlenderDevice.CPU,
        blender_executable=executable,
    )
    assert result.valid
    assert result.inventory_state == "valid"
    return output


def test_real_blender_emits_aligned_analytic_render_passes(
    rendered_root: Path,
) -> None:
    output = rendered_root
    validation = validate_render_root(output)
    collection = json.loads(
        (output / "metadata/render_collection.json").read_text(encoding="utf-8")
    )

    assert validation.valid
    assert validation.camera_count == 3
    assert validation.artifact_count == 27
    assert len(validation.analytic_checks) == 3
    assert all(item["valid"] for item in validation.analytic_checks)
    assert [view["split"] for view in collection["views"]] == [
        "train",
        "validation",
        "test",
    ]
    assert all(len(view["pass_artifacts"]) == 9 for view in collection["views"])
    assert {
        artifact["pass_name"] for artifact in collection["views"][0]["pass_artifacts"]
    } == {
        "alpha",
        "depth",
        "instance",
        "material_id",
        "normals",
        "object_id",
        "rgb",
        "semantic",
        "terminal",
    }


@pytest.mark.parametrize(
    ("case", "expected_rule"),
    [
        ("missing", "decode"),
        ("dimensions", "dimensions"),
        ("unregistered_id", "registered_ids"),
        ("nonfinite_depth", "foreground_valid"),
        ("nonunit_normal", "foreground_unit"),
        ("stale_camera", "stale_camera"),
    ],
)
def test_deliberate_render_corruptions_have_structured_diagnostics(
    tmp_path: Path,
    rendered_root: Path,
    case: str,
    expected_rule: str,
) -> None:
    root = tmp_path / case
    shutil.copytree(rendered_root, root)
    collection = _load_json(root / "metadata/render_collection.json")
    view = collection["views"][0]
    by_name = {item["pass_name"]: item for item in view["pass_artifacts"]}

    if case == "missing":
        (root / by_name["depth"]["path"]).unlink()
    elif case == "stale_camera":
        view["camera_reference"] = "cameras/stale.json"
        _refresh_collection(root, collection)
    elif case == "dimensions":
        artifact = by_name["terminal"]
        decoded = read_png(root / artifact["path"])
        values = [
            int(decoded.value(row, column))
            for row in range(decoded.height)
            for column in range(decoded.width - 1)
        ]
        write_png_u16(
            root / artifact["path"],
            width=decoded.width - 1,
            height=decoded.height,
            values=values,
        )
        _refresh_artifact(root, artifact, read_png(root / artifact["path"]))
        _refresh_collection(root, collection)
    elif case == "unregistered_id":
        _mutate_mask(root, by_name["semantic"], by_name["semantic"], 65535)
        _refresh_collection(root, collection)
    elif case == "nonfinite_depth":
        _mutate_float(
            root,
            by_name["depth"],
            by_name["semantic"],
            (float("nan"),),
        )
        _refresh_collection(root, collection)
    else:
        _mutate_float(
            root,
            by_name["normals"],
            by_name["semantic"],
            (2.0, 0.0, 0.0),
        )
        _refresh_collection(root, collection)

    with pytest.raises(RenderValidationError) as captured:
        validate_render_root(root)
    assert expected_rule in {issue.rule for issue in captured.value.issues}


def _mutate_mask(
    root: Path,
    artifact: dict[str, Any],
    semantic_artifact: dict[str, Any],
    replacement: int,
) -> None:
    decoded = read_png(root / artifact["path"])
    semantic = read_png(root / semantic_artifact["path"])
    values = [int(value) for value in decoded.values]
    pixel = next(
        index for index, value in enumerate(semantic.values) if int(value) != 0
    )
    values[pixel] = replacement
    write_png_u16(
        root / artifact["path"],
        width=decoded.width,
        height=decoded.height,
        values=values,
    )
    _refresh_artifact(root, artifact, read_png(root / artifact["path"]))


def _mutate_float(
    root: Path,
    artifact: dict[str, Any],
    semantic_artifact: dict[str, Any],
    replacement: tuple[float, ...],
) -> None:
    decoded = read_npy_f32(root / artifact["path"])
    semantic = read_png(root / semantic_artifact["path"])
    values = [float(value) for value in decoded.values]
    pixel = next(
        index for index, value in enumerate(semantic.values) if int(value) != 0
    )
    offset = pixel * decoded.channels
    values[offset : offset + decoded.channels] = replacement
    write_npy_f32(
        root / artifact["path"],
        width=decoded.width,
        height=decoded.height,
        channels=decoded.channels,  # type: ignore[arg-type]
        values=values,
    )
    _refresh_artifact(root, artifact, read_npy_f32(root / artifact["path"]))


def _refresh_artifact(
    root: Path, artifact: dict[str, Any], decoded: DecodedArray
) -> None:
    artifact["file_digest"] = hash_file(root / artifact["path"]).digest
    artifact["decoded_digest"] = decoded.decoded_digest
    artifact["width"] = decoded.width
    artifact["height"] = decoded.height


def _refresh_collection(root: Path, collection: dict[str, Any]) -> None:
    for view in collection["views"]:
        scientific_view = {
            key: value
            for key, value in view.items()
            if key not in {"warnings", "scientific_digest", "execution_reference"}
        }
        scientific_view["pass_artifacts"] = [
            {**artifact, "file_digest": None}
            for artifact in scientific_view["pass_artifacts"]
        ]
        view["scientific_digest"] = content_digest(scientific_view)
    scientific = {
        key: value
        for key, value in collection.items()
        if key not in {"warnings", "scientific_digest", "execution_reference"}
    }
    scientific["views"] = [
        {
            key: value
            for key, value in view.items()
            if key not in {"warnings", "scientific_digest", "execution_reference"}
        }
        for view in collection["views"]
    ]
    for view in scientific["views"]:
        view["pass_artifacts"] = [
            {**artifact, "file_digest": None} for artifact in view["pass_artifacts"]
        ]
    collection["scientific_digest"] = content_digest(scientific)
    (root / "metadata/render_collection.json").write_bytes(
        pretty_json_bytes(collection)
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
