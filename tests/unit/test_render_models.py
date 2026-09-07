from __future__ import annotations

import builtins
import math
import subprocess
import sys

import pytest

from gaussweave.rendering import gsplat_renderer
from gaussweave.rendering.fixtures import centered_gaussian
from gaussweave.rendering.gsplat_renderer import RenderingError
from gaussweave.rendering.models import (
    Camera,
    RenderableGaussians,
    RenderModelError,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _camera(**changes: object) -> Camera:
    values = {
        "camera_id": "camera",
        "width": 32,
        "height": 32,
        "fx": 50.0,
        "fy": 50.0,
        "cx": 15.5,
        "cy": 15.5,
        "world_from_camera": IDENTITY,
    }
    values.update(changes)
    return Camera(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 0},
        {"fx": 0.0},
        {"fx": math.nan},
        {"near": 2.0, "far": 1.0},
        {"coordinate_convention": "unknown"},
        {"world_from_camera": ((1.0,),) * 4},
        {
            "world_from_camera": (
                (1.0, 0.0, 0.0, math.nan),
                *IDENTITY[1:],
            )
        },
        {
            "world_from_camera": (
                *IDENTITY[:3],
                (0.0, 0.0, 1.0, 1.0),
            )
        },
    ],
)
def test_camera_validation(changes: dict[str, object]) -> None:
    with pytest.raises(RenderModelError):
        _camera(**changes)


def test_known_camera_conversion_is_world_to_camera() -> None:
    torch = pytest.importorskip("torch")
    camera = _camera(
        world_from_camera=(
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 3.0),
            (0.0, 0.0, 1.0, 4.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    view, intrinsics = camera.gsplat_matrices("cpu")
    point_world = torch.tensor([2.0, 3.0, 5.0, 1.0])
    assert torch.allclose(view @ point_world, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert intrinsics.tolist() == [
        [50.0, 0.0, 15.5],
        [0.0, 50.0, 15.5],
        [0.0, 0.0, 1.0],
    ]


def test_gaussian_validation_and_modes() -> None:
    scene = centered_gaussian()
    assert scene.count == 1 and scene.coefficient_shape == (1, 3)
    sh = RenderableGaussians(
        scene.means,
        scene.quaternions,
        scene.scales,
        scene.opacities,
        (((0.5, 0.0, 0.0),),),
        appearance_mode="sh",
        sh_degree=0,
    )
    assert sh.coefficient_shape == (1, 1, 3)
    invalid_values = [
        {"quaternions": ((2.0, 0.0, 0.0, 0.0),)},
        {"scales": ((0.0, 0.1, 0.1),)},
        {"means": ((math.nan, 0.0, 3.0),)},
        {"appearance": ((1.0, 0.0),)},
        {"opacities": (2.0,)},
        {"stable_ids": ("same", "same")},
    ]
    for changes in invalid_values:
        values = {
            "means": scene.means,
            "quaternions": scene.quaternions,
            "scales": scene.scales,
            "opacities": scene.opacities,
            "appearance": scene.appearance,
        }
        values.update(changes)
        with pytest.raises(RenderModelError):
            RenderableGaussians(**values)  # type: ignore[arg-type]


def test_inconsistent_gaussian_counts_rejected() -> None:
    scene = centered_gaussian()
    with pytest.raises(RenderModelError):
        RenderableGaussians(
            scene.means,
            (),
            scene.scales,
            scene.opacities,
            scene.appearance,
        )


def test_lazy_dependency_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def controlled_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"torch", "gsplat"} or name.startswith("gsplat."):
            raise ImportError("controlled missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    with pytest.raises(RenderingError, match="environment is unavailable"):
        gsplat_renderer._dependencies()


def test_rendering_model_import_isolation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gaussweave.rendering; "
            "import gaussweave.rendering.models; "
            "assert 'torch' not in sys.modules; assert 'gsplat' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
