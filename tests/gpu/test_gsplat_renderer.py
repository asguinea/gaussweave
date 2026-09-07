from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.rendering.fixtures import (
    centered_gaussian,
    separated_gaussians,
    smoke_cameras,
)
from gaussweave.rendering.gsplat_renderer import (
    REGRESSION_MAX_ABS_TOLERANCE,
    GsplatRenderer,
)
from gaussweave.rendering.models import RenderSettings
from gaussweave.results.artifacts import (
    ArtifactDeclaration,
    VerificationState,
    build_inventory,
    verify_inventory,
)

pytestmark = pytest.mark.gpu


def _settings(*, depth: bool = True, repetitions: int = 2) -> RenderSettings:
    return RenderSettings(
        output_buffers=("rgb", "alpha", "depth") if depth else ("rgb", "alpha"),
        warmup_count=1,
        repetition_count=repetitions,
    )


def test_one_gaussian_rgb_alpha_background_and_resources(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    result = GsplatRenderer().render(
        centered_gaussian(),
        smoke_cameras()[0],
        _settings(depth=False),
        resource_output=tmp_path / "resource.json",
    )
    assert tuple(result.rgb.shape) == (1, 64, 64, 3)
    assert tuple(result.alpha.shape) == (1, 64, 64, 1)
    assert torch.isfinite(result.rgb).all() and torch.isfinite(result.alpha).all()
    assert 0 <= result.alpha.min() <= result.alpha.max() <= 1
    assert result.alpha[0, 32, 32, 0] > 0.5
    assert result.rgb[0, 32, 32, 0] > result.rgb[0, 32, 32, 1]
    assert result.alpha[0, 0, 0, 0] < 1e-4
    assert torch.max(torch.abs(result.rgb[0, 0, 0])) < 1e-4
    resource = result.resource_record
    assert resource.timing.warmup_count == 1
    assert resource.timing.repetition_count == 2
    assert len(resource.timing.samples_seconds) == 2
    assert resource.timing.synchronized
    assert resource.timing.median_seconds is not None
    assert resource.timing.p95_seconds is not None
    assert resource.gpu_final.peak_allocated_bytes > 0
    assert resource.gpu_final.peak_reserved_bytes > 0
    assert resource.compliance.state.value.startswith("compliant")


def test_multi_gaussian_depth_batch_and_determinism() -> None:
    torch = pytest.importorskip("torch")
    renderer = GsplatRenderer()
    first = renderer.render(separated_gaussians(), smoke_cameras(), _settings())
    second = renderer.render(separated_gaussians(), smoke_cameras(), _settings())
    assert tuple(first.rgb.shape) == (2, 64, 64, 3)
    assert tuple(first.depth.shape) == (2, 64, 64, 1)
    assert first.camera_ids == ("camera-center", "camera-right")
    assert first.metadata["depth_semantics"] == "expected_z_depth=sum(w*z)/sum(w)"
    red = first.rgb[0, 32, 17]
    green = first.rgb[0, 32, 42]
    assert red[0] > red[1] and green[1] > green[0]
    assert first.depth[0, 32, 17, 0] < first.depth[0, 32, 42, 0]
    assert not torch.allclose(first.rgb[0], first.rgb[1])
    maximum = torch.max(torch.abs(first.rgb - second.rgb)).item()
    assert maximum <= REGRESSION_MAX_ABS_TOLERANCE


def test_smoke_cli_outputs_overwrite_and_inventory(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    arguments = [
        sys.executable,
        "-m",
        "gaussweave.rendering.gsplat_renderer",
        "smoke",
        "--profile",
        "smoke",
        "--output",
        str(output),
        "--json",
    ]
    completed = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["shape"] == [2, 64, 64, 3]
    assert summary["depth_shape"] == [2, 64, 64, 1]
    assert summary["compliance"].startswith("compliant")
    assert (output / "camera-center-rgb.ppm").is_file()
    assert (output / "camera-center-alpha.pgm").is_file()
    assert (output / "camera-center-depth.pfm").is_file()
    assert (output / "render-metadata.json").is_file()
    assert (output / "resource-record.json").is_file()
    refused = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert refused.returncode == 1
    paths = sorted(path.relative_to(output).as_posix() for path in output.iterdir())
    inventory = build_inventory(
        output,
        [
            ArtifactDeclaration(f"render-{index}", "render", path)
            for index, path in enumerate(paths)
        ],
        inventory_id="inventory-smoke-render",
    )
    assert (
        verify_inventory(inventory, output, strict=True).state
        is VerificationState.VALID
    )
