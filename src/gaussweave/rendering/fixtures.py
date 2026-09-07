"""Small deterministic project rendering fixtures."""

from __future__ import annotations

from gaussweave.rendering.models import Camera, RenderableGaussians

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def smoke_cameras() -> tuple[Camera, Camera]:
    """Return a forward OpenCV camera and a translated second viewpoint."""

    return (
        Camera("camera-center", 64, 64, 80.0, 80.0, 31.5, 31.5, IDENTITY),
        Camera(
            "camera-right",
            64,
            64,
            80.0,
            80.0,
            31.5,
            31.5,
            (
                (1.0, 0.0, 0.0, 0.35),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
    )


def centered_gaussian() -> RenderableGaussians:
    return RenderableGaussians(
        means=((0.0, 0.0, 3.0),),
        quaternions=((1.0, 0.0, 0.0, 0.0),),
        scales=((0.18, 0.18, 0.18),),
        opacities=(0.95,),
        appearance=((1.0, 0.1, 0.05),),
        stable_ids=("gaussian-center",),
    )


def separated_gaussians() -> RenderableGaussians:
    return RenderableGaussians(
        means=((-0.45, 0.0, 2.5), (0.45, 0.0, 3.5), (0.0, -0.45, 3.0)),
        quaternions=(
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
        scales=((0.16, 0.16, 0.16),) * 3,
        opacities=(0.95, 0.95, 0.9),
        appearance=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        stable_ids=("gaussian-red-near", "gaussian-green-far", "gaussian-blue"),
    )
