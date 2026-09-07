from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaussweave.data.blender import (
    BlenderBackend,
    BlenderDevice,
    BlenderError,
    BlenderExecutionError,
    BlenderInvocation,
    capture_blender_identity,
    compare_qualification_runs,
    discover_blender,
    execute_blender,
    qualify_blender,
    scientific_metadata_digest,
)
from gaussweave.results.artifacts import load_inventory, verify_inventory

pytestmark = [pytest.mark.integration, pytest.mark.blender, pytest.mark.wsl]


@pytest.fixture(scope="module")
def blender_executable() -> Path:
    try:
        return discover_blender(backend=BlenderBackend.WSL)
    except Exception as error:
        pytest.skip(f"Blender unavailable for integration qualification: {error}")


@pytest.fixture(scope="module")
def cpu_runs(
    tmp_path_factory: pytest.TempPathFactory, blender_executable: Path
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("blender-cpu")
    first = root / "same-seed-a"
    second = root / "same-seed-b"
    qualify_blender(first, blender_executable=blender_executable, seed=17)
    qualify_blender(second, blender_executable=blender_executable, seed=17)
    return first, second


def test_version_capture_and_embedded_python(
    blender_executable: Path, tmp_path: Path
) -> None:
    identity = capture_blender_identity(
        blender_executable,
        backend=BlenderBackend.WSL,
        working_directory=tmp_path,
    )
    assert identity.version == "4.5.12 LTS"
    assert identity.embedded_python.startswith("3.11.11")
    assert identity.build_hash == "84afd5f785f7"


def test_minimal_wsl_cpu_generation_and_scene_structure(
    cpu_runs: tuple[Path, Path],
) -> None:
    root = cpu_runs[0]
    for path in (
        "scene.blend",
        "render.png",
        "generation-metadata.json",
        "execution-summary.json",
    ):
        assert (root / path).is_file() and (root / path).stat().st_size > 0
    metadata = json.loads((root / "generation-metadata.json").read_text())
    assert metadata["status"] == "qualified"
    assert metadata["objects"] == {
        "camera": ["QualificationCamera"],
        "geometry": ["QualificationCube"],
        "lights": ["QualificationKeyLight"],
    }
    assert metadata["materials"] == ["QualificationMaterial"]
    assert metadata["coordinates"]["units"] == "meters"
    assert metadata["coordinates"]["up_axis"] == "+Z"
    assert metadata["render"]["engine"] == "CYCLES"
    assert metadata["render"]["resolution"] == {
        "height": 64,
        "percentage": 100,
        "width": 64,
    }
    assert metadata["render"]["image_format"] == "PNG"
    assert metadata["compute"] == {
        "backend": "CPU",
        "device_name": "CPU",
        "requested": "cpu",
    }


def test_same_seed_determinism_output_and_artifact_membership(
    cpu_runs: tuple[Path, Path],
) -> None:
    comparison = compare_qualification_runs(*cpu_runs)
    assert comparison["valid"]
    assert comparison["structural_metadata_exact"]
    assert comparison["rendered_pixel_payload_exact"]
    assert comparison["artifact_membership_exact"]
    assert (
        comparison["first_scientific_digest"] == comparison["second_scientific_digest"]
    )


def test_different_seed_changes_only_intended_variation(
    cpu_runs: tuple[Path, Path], tmp_path: Path, blender_executable: Path
) -> None:
    different = tmp_path / "different-seed"
    qualify_blender(different, blender_executable=blender_executable, seed=29)
    first = json.loads((cpu_runs[0] / "generation-metadata.json").read_text())
    changed = json.loads((different / "generation-metadata.json").read_text())
    assert first["seed"] != changed["seed"]
    assert first["seeded_value"] != changed["seeded_value"]
    for key in (
        "coordinates",
        "collections",
        "objects",
        "materials",
        "camera",
        "render",
    ):
        assert first[key] == changed[key]


def test_artifacts_strictly_verify_and_volatile_fields_are_excluded(
    cpu_runs: tuple[Path, Path],
) -> None:
    root = cpu_runs[0]
    inventory = load_inventory(root / "artifact-inventory.json")
    verification = verify_inventory(inventory, root, strict=True)
    assert verification.to_dict()["valid"]
    assert inventory.artifact_count == 8
    assert sorted(item.path for item in inventory.artifacts) == [
        "execution-summary.json",
        "generation-metadata.json",
        "render.png",
        "resource-record.json",
        "scene.blend",
        "source/qualify_blender.py",
        "stderr.log",
        "stdout.log",
    ]
    assert scientific_metadata_digest(
        cpu_runs[0] / "generation-metadata.json"
    ) == scientific_metadata_digest(cpu_runs[1] / "generation-metadata.json")


def test_real_blender_failure_preserves_command_evidence(
    blender_executable: Path, tmp_path: Path
) -> None:
    root = tmp_path / "real-failure"
    root.mkdir()
    script = root / "fail.py"
    script.write_text("raise RuntimeError('intentional qualification failure')\n")
    identity = capture_blender_identity(
        blender_executable,
        backend=BlenderBackend.WSL,
        working_directory=root,
    )
    result = execute_blender(
        BlenderInvocation(
            blender_executable,
            script,
            root,
            (),
            root,
            root,
            30.0,
            {},
            BlenderBackend.WSL,
        ),
        identity=identity,
    )
    assert not result.success and result.exit_code != 0
    assert result.failure_ref
    assert "intentional qualification failure" in (root / "stderr.log").read_text()


@pytest.mark.gpu
def test_real_gpu_probe_renders_or_reports_explicit_unsupported(
    blender_executable: Path, tmp_path: Path
) -> None:
    result = qualify_blender(
        tmp_path / "gpu",
        blender_executable=blender_executable,
        device=BlenderDevice.GPU,
    )
    assert result.status in {"qualified", "unsupported"}
    if result.status == "qualified":
        assert result.compute_backend in {"CUDA", "OPTIX"}
        assert result.compute_device
        assert result.image_bytes and result.image_bytes > 0
        assert result.artifact_count == 8
    else:
        assert result.compute_backend == "UNSUPPORTED"
        assert result.warnings
        assert result.artifact_count == 7


def test_qualification_refuses_unmarked_overwrite(
    blender_executable: Path, tmp_path: Path
) -> None:
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "unrelated.txt").write_text("preserve me")
    with pytest.raises(BlenderError, match="marked"):
        qualify_blender(root, blender_executable=blender_executable, overwrite=True)
    assert (root / "unrelated.txt").read_text() == "preserve me"


def test_missing_executable_is_environment_failure(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="not a file"):
        qualify_blender(
            tmp_path / "missing",
            blender_executable=tmp_path / "does-not-exist",
        )


def test_execution_failure_does_not_claim_qualification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "gaussweave.data.blender.discover_blender",
        lambda **_: Path("/missing/blender"),
    )
    with pytest.raises((BlenderExecutionError, OSError)):
        qualify_blender(tmp_path / "failed")
