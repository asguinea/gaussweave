from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.cli import ExitCode
from gaussweave.data.blender import (
    BlenderBackend,
    BlenderError,
    BlenderExecutionError,
    discover_blender,
)
from gaussweave.data.blender_runtime import (
    compare_runtime_probes,
    run_blender_scene_generation,
)
from gaussweave.data.scene_config import SEED_STREAMS, derive_seed
from gaussweave.results.artifacts import load_inventory, verify_inventory

pytestmark = [pytest.mark.integration, pytest.mark.blender, pytest.mark.wsl]
REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY / "configs/scenes/colonnade-development-s303.json"


@pytest.fixture(scope="module")
def blender_executable() -> Path:
    try:
        return discover_blender(backend=BlenderBackend.WSL)
    except Exception as error:
        pytest.skip(f"Blender unavailable for runtime integration: {error}")


@pytest.fixture(scope="module")
def deterministic_runs(
    tmp_path_factory: pytest.TempPathFactory, blender_executable: Path
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("blender-runtime")
    first = root / "first"
    second = root / "second"
    run_blender_scene_generation(CONFIG, first, blender_executable=blender_executable)
    run_blender_scene_generation(CONFIG, second, blender_executable=blender_executable)
    return first, second


def test_real_probe_scene_metadata_render_and_inventory(
    deterministic_runs: tuple[Path, Path],
) -> None:
    root = deterministic_runs[0]
    required = (
        "source/resolved_scene_config.json",
        "source/generator_metadata.json",
        "source/runtime_probe.blend",
        "source/execution_summary.json",
        "preview/runtime_probe.png",
        "logs/stdout.log",
        "logs/stderr.log",
        "logs/external-command.json",
        "logs/resource-record.json",
        "checksums.sha256",
        "artifact-inventory.json",
    )
    for portable in required:
        target = root / portable
        assert target.is_file()
        if portable != "logs/stderr.log":
            assert target.stat().st_size > 0
    metadata = json.loads(
        (root / "source/generator_metadata.json").read_text(encoding="utf-8")
    )
    scientific = metadata["scientific"]
    assert scientific["purpose"].startswith("runtime-probe engineering fixture")
    assert scientific["coordinates"]["up_axis"] == "+Z"
    assert scientific["coordinates"]["unit"] == "meters"
    assert [item["name"] for item in scientific["collections"]] == [
        "GAUSSWEAVE_ROOT",
        "STRUCTURE",
        "TERMINALS",
        "INSTANCES",
        "OCCLUDERS",
        "LIGHTING",
        "CAMERAS",
        "ANNOTATIONS",
        "RUNTIME_PROBE",
    ]
    roles = [item["role"] for item in scientific["objects"]]
    assert roles.count("structure") == 1
    assert roles.count("terminal-placeholder") == 1
    assert roles.count("instance-placeholder") == 2
    assert roles.count("annotation-helper") == 1
    assert len(scientific["materials"]) == 2
    assert scientific["camera"] and len(scientific["lights"]) >= 1
    assert scientific["render"]["resolution"] == {
        "height": 256,
        "percentage": 100,
        "width": 256,
    }
    assert scientific["render"]["engine"] == "BLENDER_EEVEE_NEXT"
    assert scientific["render"]["samples"] == 32
    assert scientific["render"]["denoise"] is False
    assert scientific["render"]["image"] == {
        "bit_depth": 8,
        "color_mode": "RGB",
        "format": "PNG",
    }
    assert scientific["render"]["color_management"]["view_transform"] == "Standard"
    assert scientific["render"]["background"]["rgba"] == [0.2, 0.24, 0.28, 1.0]
    assert scientific["render"]["frame"] == 1
    inventory = load_inventory(root / "artifact-inventory.json")
    assert verify_inventory(inventory, root, strict=True).to_dict()["valid"]
    assert inventory.artifact_count == 10


def test_same_config_scientific_metadata_and_pixels_are_exact(
    deterministic_runs: tuple[Path, Path],
) -> None:
    comparison = compare_runtime_probes(*deterministic_runs)
    assert comparison["scientific_metadata_exact"]
    assert comparison["scientific_digest_equal"]
    assert comparison["preview_pixels_exact"]


def test_changed_master_seed_changes_only_seeded_probe_values(
    deterministic_runs: tuple[Path, Path],
    tmp_path: Path,
    blender_executable: Path,
) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["master_seed"] = 304
    raw["derived_seeds"] = {stream: derive_seed(304, stream) for stream in SEED_STREAMS}
    raw["camera"]["seed"] = raw["derived_seeds"]["cameras"]
    raw["appearance"]["seed"] = raw["derived_seeds"]["materials"]
    raw["lighting"]["seed"] = raw["derived_seeds"]["lighting"]
    raw["occlusion"]["seed"] = raw["derived_seeds"]["occluders"]
    for edit in raw["edits"]:
        edit["edit_seed"] = raw["derived_seeds"]["edits"]
    raw["scene_id"] = "syn-colonnade-linear-development-s304"
    changed_config = tmp_path / "changed-seed.json"
    changed_config.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    changed_root = tmp_path / "changed-seed"
    run_blender_scene_generation(
        changed_config,
        changed_root,
        blender_executable=blender_executable,
    )
    original = json.loads(
        (deterministic_runs[0] / "source/generator_metadata.json").read_text(
            encoding="utf-8"
        )
    )["scientific"]
    changed = json.loads(
        (changed_root / "source/generator_metadata.json").read_text(encoding="utf-8")
    )["scientific"]
    assert original["seeds"] != changed["seeds"]
    assert original["objects"] != changed["objects"]
    assert original["materials"] != changed["materials"]
    for stable_key in ("coordinates", "collections", "render"):
        assert original[stable_key] == changed[stable_key]


def test_validation_failure_is_nonzero_and_preserves_evidence(
    tmp_path: Path, blender_executable: Path
) -> None:
    root = tmp_path / "validation-failure"
    with pytest.raises(BlenderExecutionError, match="failed with exit code"):
        run_blender_scene_generation(
            CONFIG,
            root,
            blender_executable=blender_executable,
            inject_failure="validation",
        )
    assert (root / "source/runtime_failure.json").is_file()
    assert list((root / "metadata").glob("failure-*.json"))
    assert (root / "logs/stderr.log").is_file()
    inventory = load_inventory(root / "artifact-inventory.json")
    assert verify_inventory(inventory, root, strict=True).to_dict()["valid"]


def test_cli_json_cpu_overwrite_and_timeout(
    tmp_path: Path, blender_executable: Path
) -> None:
    root = tmp_path / "cli"
    base = [
        sys.executable,
        "-m",
        "gaussweave.cli",
        "blender",
        "runtime-probe",
        "--config",
        str(CONFIG),
        "--root",
        str(root),
        "--blender-executable",
        str(blender_executable),
        "--json",
    ]
    success = subprocess.run(base, capture_output=True, text=True, check=False)
    assert success.returncode == 0, success.stderr
    payload = json.loads(success.stdout)
    assert payload["valid"] and payload["device"] == "cpu"
    refused = subprocess.run(base, capture_output=True, text=True, check=False)
    assert refused.returncode == ExitCode.OPERATION
    assert "nonempty" in json.loads(refused.stdout)["error"]
    overwritten = subprocess.run(
        [*base, "--overwrite"], capture_output=True, text=True, check=False
    )
    assert overwritten.returncode == 0, overwritten.stderr

    timeout = subprocess.run(
        [
            *base[: base.index("--root") + 1],
            str(tmp_path / "timeout"),
            *base[base.index("--blender-executable") :],
            "--timeout-seconds",
            "0.2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert timeout.returncode in {ExitCode.ENVIRONMENT, ExitCode.OPERATION}
    assert json.loads(timeout.stdout)["valid"] is False


def test_unmarked_occupied_root_is_preserved(
    tmp_path: Path, blender_executable: Path
) -> None:
    root = tmp_path / "occupied"
    root.mkdir()
    unrelated = root / "unrelated.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    with pytest.raises(BlenderError, match="matching marked"):
        run_blender_scene_generation(
            CONFIG,
            root,
            blender_executable=blender_executable,
            overwrite=True,
        )
    assert unrelated.read_text(encoding="utf-8") == "preserve"
