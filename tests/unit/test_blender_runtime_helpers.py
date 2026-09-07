from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blender_scripts.gaussweave_runtime import REQUIRED_SEED_STREAMS
from blender_scripts.gaussweave_runtime.arguments import parse_runtime_arguments
from blender_scripts.gaussweave_runtime.io import (
    RuntimeIOError,
    content_digest,
    portable_path,
)
from blender_scripts.gaussweave_runtime.naming import NamingError, NamingRegistry
from blender_scripts.gaussweave_runtime.seeding import SeedError, SeedRegistry
from blender_scripts.gaussweave_runtime.specs import (
    SpecificationError,
    validate_material_reuse,
    validate_material_values,
    validate_positive_finite,
)
from gaussweave.data.blender import BlenderBackend, BlenderInvocation
from gaussweave.data.blender_runtime import (
    RUNTIME_MARKER,
    _prepare_output_root,
    _validate_output_root,
)

pytestmark = pytest.mark.unit


def _seed_config() -> dict[str, object]:
    return {
        "master_seed": 101,
        "derived_seeds": {
            name: index * 101 + 7
            for index, name in enumerate(REQUIRED_SEED_STREAMS, start=1)
        },
    }


def test_runtime_argument_parser_accepts_complete_handoff_and_rejects_missing() -> None:
    options = parse_runtime_arguments(
        [
            "--config",
            "resolved.json",
            "--output-root",
            "output",
            "--mode",
            "runtime-probe",
            "--generator-version",
            "blender-runtime-v1",
            "--provenance",
            "configs/scenes/probe.json",
            "--scene-id",
            "syn-probe-test-s101",
            "--master-seed",
            "101",
            "--backend",
            "wsl",
            "--device",
            "cpu",
        ]
    )
    assert options.mode == "runtime-probe"
    assert options.master_seed == 101
    with pytest.raises(SystemExit):
        parse_runtime_arguments(["--mode", "runtime-probe"])


def test_seed_registry_matches_config_isolated_and_cross_process() -> None:
    config = _seed_config()
    first = SeedRegistry.from_config(config)
    second = SeedRegistry.from_config(config)
    expected = second.python("materials").random()
    first.python("geometry").random()
    first.python("geometry").random()
    assert first.python("materials").random() == expected
    assert first.metadata() == second.metadata()
    assert first.seeds == config["derived_seeds"]

    program = (
        "import json;"
        "from blender_scripts.gaussweave_runtime.seeding import SeedRegistry;"
        f"c=json.loads({json.dumps(json.dumps(config))});"
        "print(json.dumps(SeedRegistry.from_config(c).metadata(),sort_keys=True))"
    )
    one = subprocess.check_output([sys.executable, "-c", program], text=True)
    two = subprocess.check_output([sys.executable, "-c", program], text=True)
    assert one == two


def test_seed_registry_rejects_missing_extra_and_invalid_streams() -> None:
    config = _seed_config()
    missing = json.loads(json.dumps(config))
    missing["derived_seeds"].pop("edits")
    with pytest.raises(SeedError, match="missing seed"):
        SeedRegistry.from_config(missing)
    extra = json.loads(json.dumps(config))
    extra["derived_seeds"]["other"] = 1
    with pytest.raises(SeedError, match="unexpected seed"):
        SeedRegistry.from_config(extra)


def test_naming_is_stable_and_duplicate_ids_fail() -> None:
    first = NamingRegistry()
    second = NamingRegistry()
    expected = "SS_OBJ_TERMINAL_PROBE_A"
    assert first.reserve("OBJ", "terminal-probe-a") == expected
    assert second.reserve("OBJ", "terminal-probe-a") == expected
    with pytest.raises(NamingError, match="duplicate stable ID"):
        first.reserve("OBJ", "terminal-probe-a")
    assert first.reserve("MAT", "probe-base", allow_reuse=True) == first.reserve(
        "MAT", "probe-base", allow_reuse=True
    )


@pytest.mark.parametrize(
    "values",
    [(), (0.0,), (-1.0,), (float("inf"),), (float("nan"),)],
)
def test_metric_dimensions_reject_degenerate_or_nonfinite(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(SpecificationError):
        validate_positive_finite(values)
    validate_positive_finite((0.1, 2.0, 3.0))


def test_material_specification_and_portable_paths() -> None:
    assert validate_material_values(
        (0.1, 0.2, 0.3, 1.0), roughness=0.5, metallic=0.0, alpha=1.0
    ) == (0.1, 0.2, 0.3, 1.0)
    with pytest.raises(SpecificationError):
        validate_material_values(
            (0.1, 0.2, 0.3, 2.0), roughness=0.5, metallic=0.0, alpha=1.0
        )
    validate_material_reuse("stable-definition", "stable-definition")
    with pytest.raises(SpecificationError, match="conflicting material"):
        validate_material_reuse("stable-definition", "changed-definition")
    assert portable_path("source/generator_metadata.json") == (
        "source/generator_metadata.json"
    )
    for unsafe in ("../escape", "/absolute", "C:/drive", "a\\b", "a//b"):
        with pytest.raises(RuntimeIOError):
            portable_path(unsafe)


def test_content_digest_is_stable_and_rejects_nonfinite() -> None:
    assert content_digest({"b": 2, "a": 1}) == content_digest({"a": 1, "b": 2})
    with pytest.raises(RuntimeIOError):
        content_digest({"invalid": float("nan")})


def test_output_root_safety_and_marked_overwrite(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="traversal"):
        _validate_output_root(tmp_path / ".." / "escape")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(Exception, match="nonempty"):
        _prepare_output_root(occupied, scene_id="scene-a", overwrite=False)
    with pytest.raises(Exception, match="marked"):
        _prepare_output_root(occupied, scene_id="scene-a", overwrite=True)
    (occupied / RUNTIME_MARKER).write_text(
        "GaussWeave runtime-probe root\nscene_id=scene-a\n",
        encoding="utf-8",
    )
    _prepare_output_root(occupied, scene_id="scene-a", overwrite=True)
    assert list(occupied.iterdir()) == []

    target = tmp_path / "symlink-target"
    target.mkdir()
    symlink = tmp_path / "symlink-root"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(Exception, match="symlink"):
        _validate_output_root(symlink)


def test_project_runtime_imports_are_accelerator_and_bpy_independent() -> None:
    program = (
        "import sys;"
        "from gaussweave.data.blender import run_blender_scene_generation;"
        "import gaussweave.data.blender_runtime;"
        "assert 'bpy' not in sys.modules;"
        "assert 'torch' not in sys.modules;"
        "assert 'gsplat' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_invocation_supports_runtime_log_paths(tmp_path: Path) -> None:
    invocation = BlenderInvocation(
        tmp_path / "blender",
        tmp_path / "script.py",
        tmp_path,
        (),
        tmp_path,
        tmp_path,
        1.0,
        {},
        BlenderBackend.WSL,
        Path("logs/stdout.log"),
        Path("logs/stderr.log"),
    )
    assert invocation.stdout_path.as_posix() == "logs/stdout.log"
