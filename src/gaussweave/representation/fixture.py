"""Deterministic GW direct-Gaussian hero fixture construction."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import canonical_json_bytes
from gaussweave.config.resolution import content_digest as _content_digest
from gaussweave.data.camera_math import look_at_world_from_camera
from gaussweave.rendering.models import Camera
from gaussweave.representation.models import (
    CANONICAL_FRAME,
    LOCAL_FRAME,
    AppearanceResiduals,
    CanonicalTerminal,
    ExplicitRepresentation,
    GaussianArrays,
    GridRepeat,
    StructuralRepresentation,
    UniqueComponent,
    Vector3,
    concatenate_explicit,
    f32,
)

FIXTURE_ID = "gw-direct-gaussian-panel-grid-v1"
SEED_POLICY_VERSION = "gw-seeds-v1"
PRINCIPAL_SEEDS = (17, 29, 43)
CAMERA_IDS = tuple(f"cam-eval-{index:04d}" for index in range(8))
CAMERA_WIDTH = 256
CAMERA_HEIGHT = 256


def content_digest(value: Any) -> str:
    return _content_digest(cast(Any, value))


def derive_seed(principal_seed: int, stream_name: str) -> int:
    payload = {
        "policy_version": SEED_POLICY_VERSION,
        "principal_seed": principal_seed,
        "stream_name": stream_name,
    }
    digest = hashlib.sha256(canonical_json_bytes(cast(Any, payload))).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


PROTOCOL_DIGEST = content_digest(
    {
        "fixture_id": FIXTURE_ID,
        "seed_policy_version": SEED_POLICY_VERSION,
        "principal_seeds": PRINCIPAL_SEEDS,
        "camera_ids": CAMERA_IDS,
        "image_dimensions": [CAMERA_WIDTH, CAMERA_HEIGHT],
        "camera_model": "pinhole-opencv",
    }
)


def _terminal(principal_seed: int) -> CanonicalTerminal:
    rng = random.Random(derive_seed(principal_seed, "terminal_appearance"))
    means: list[Vector3] = []
    colors: list[Vector3] = []
    scales: list[Vector3] = []
    opacities: list[float] = []
    for row in range(16):
        for column in range(16):
            x = -0.30 + column * 0.04
            z = -0.34 + row * (0.68 / 15.0)
            border = row in {0, 15} or column in {0, 15}
            key = (
                (row == column)
                or (row + column == 15)
                or (6 <= row <= 9 and 4 <= column <= 11)
            )
            if border:
                base = (0.76, 0.50, 0.18)
            elif key:
                base = (0.22, 0.72, 0.74)
            else:
                base = (0.18, 0.34, 0.62)
            jitter = tuple(rng.uniform(-0.015, 0.015) for _ in range(3))
            means.append((f32(x), f32(-0.015 if key else 0.0), f32(z)))
            colors.append(
                cast(
                    Vector3,
                    tuple(f32(base[channel] + jitter[channel]) for channel in range(3)),
                )
            )
            scales.append((f32(0.027), f32(0.018), f32(0.030)))
            opacities.append(f32(0.94 if border or key else 0.86))
    gaussians = GaussianArrays(
        means=tuple(means),
        quaternions=((1.0, 0.0, 0.0, 0.0),) * 256,
        scales=tuple(scales),
        opacities=tuple(opacities),
        colors=tuple(colors),
        stable_ids=tuple(f"terminal-g{index:04d}" for index in range(256)),
        coordinate_frame=LOCAL_FRAME,
    )
    return CanonicalTerminal(
        component_id="terminal-panel-v1",
        gaussians=gaussians,
        provenance={
            "generator": "gaussweave.representation.fixture._terminal",
            "principal_seed": principal_seed,
            "stream": "terminal_appearance",
            "derived_seed": derive_seed(principal_seed, "terminal_appearance"),
        },
    )


def _unique(principal_seed: int) -> UniqueComponent:
    rng = random.Random(derive_seed(principal_seed, "fixture_geometry"))
    means: list[Vector3] = []
    colors: list[Vector3] = []
    scales: list[Vector3] = []
    opacities: list[float] = []
    # 384-Gaussian dark exhibition wall behind every frozen repeat count.
    for row in range(16):
        for column in range(24):
            x = -3.75 + column * (7.50 / 23.0)
            z = 0.02 + row * (3.96 / 15.0)
            jitter = rng.uniform(-0.008, 0.008)
            means.append((f32(x), f32(0.24), f32(z)))
            colors.append(
                (f32(0.075 + jitter), f32(0.105 + jitter), f32(0.155 + jitter))
            )
            scales.append((f32(0.18), f32(0.035), f32(0.15)))
            opacities.append(f32(0.72))
    # 128-Gaussian frame: top/bottom rails plus left/right uprights.
    for index in range(32):
        x = -3.85 + index * (7.70 / 31.0)
        for z in (-0.10, 4.10):
            means.append((f32(x), f32(0.02), f32(z)))
            colors.append((f32(0.72), f32(0.56), f32(0.20)))
            scales.append((f32(0.14), f32(0.05), f32(0.055)))
            opacities.append(f32(0.95))
        z_side = 0.02 + index * (3.96 / 31.0)
        for x_side in (-3.86, 3.86):
            means.append((f32(x_side), f32(0.02), f32(z_side)))
            colors.append((f32(0.72), f32(0.56), f32(0.20)))
            scales.append((f32(0.055), f32(0.05), f32(0.12)))
            opacities.append(f32(0.95))
    gaussians = GaussianArrays(
        means=tuple(means),
        quaternions=((1.0, 0.0, 0.0, 0.0),) * 512,
        scales=tuple(scales),
        opacities=tuple(opacities),
        colors=tuple(colors),
        stable_ids=tuple(f"unique-g{index:04d}" for index in range(512)),
        coordinate_frame=CANONICAL_FRAME,
    )
    return UniqueComponent(
        component_id="unique-exhibition-frame-v1",
        gaussians=gaussians,
        provenance={
            "generator": "gaussweave.representation.fixture._unique",
            "principal_seed": principal_seed,
            "stream": "fixture_geometry",
            "derived_seed": derive_seed(principal_seed, "fixture_geometry"),
            "contains_repeated_terminal_content": False,
        },
    )


def grid_for_repeat_count(repeat_count: int) -> GridRepeat:
    if repeat_count not in {4, 8, 16, 32}:
        raise ValueError("frozen GW repeat_count must be one of 4, 8, 16, 32")
    rows = 4
    columns = repeat_count // rows
    return GridRepeat(
        rows=rows,
        columns=columns,
        origin=(f32(-0.5 * (columns - 1) * 0.90), f32(0.0), f32(0.48)),
        row_step=(f32(0.0), f32(0.0), f32(0.96)),
        column_step=(f32(0.90), f32(0.0), f32(0.0)),
    )


def instance_offsets(principal_seed: int, grid: GridRepeat) -> dict[str, Vector3]:
    rng = random.Random(derive_seed(principal_seed, "instance_variation"))
    offsets: dict[str, Vector3] = {}
    for flat_index in grid.active_indices:
        offsets[grid.instance_id(flat_index)] = tuple(
            f32(rng.uniform(-0.08, 0.08)) for _ in range(3)
        )  # type: ignore[assignment]
    return offsets


@dataclass(frozen=True)
class FixtureRepresentations:
    explicit: ExplicitRepresentation
    shared: StructuralRepresentation
    residual_q8: StructuralRepresentation | None
    reference_offsets: dict[str, Vector3]
    protocol_digest: str

    @property
    def scientific_digest(self) -> str:
        return content_digest(
            {
                "explicit": self.explicit.scientific_digest,
                "shared": self.shared.scientific_digest,
                "residual_q8": (
                    None
                    if self.residual_q8 is None
                    else self.residual_q8.scientific_digest
                ),
                "reference_offsets": self.reference_offsets,
                "protocol_digest": self.protocol_digest,
            }
        )


def build_fixture_representations(
    *,
    repo_root: Path,
    appearance_regime: str,
    repeat_count: int = 16,
    principal_seed: int = 17,
) -> FixtureRepresentations:
    _ = repo_root
    if appearance_regime not in {"exact", "low_variation"}:
        raise ValueError("appearance_regime must be exact or low_variation")
    if principal_seed not in PRINCIPAL_SEEDS:
        raise ValueError("principal_seed is not part of the reproducibility seed set")
    terminal = _terminal(principal_seed)
    unique = _unique(principal_seed)
    grid = grid_for_repeat_count(repeat_count)
    offsets = (
        {} if appearance_regime == "exact" else instance_offsets(principal_seed, grid)
    )
    explicit = ExplicitRepresentation(
        gaussians=concatenate_explicit(unique, terminal, grid, offsets or None),
        fixture_id=FIXTURE_ID,
        appearance_regime=appearance_regime,
        principal_seed=principal_seed,
        grid=grid,
    )
    shared = StructuralRepresentation(
        unique=unique,
        terminal=terminal,
        grid=grid,
        fixture_id=FIXTURE_ID,
        appearance_regime=appearance_regime,
        principal_seed=principal_seed,
    )
    residual_q8 = None
    if appearance_regime == "low_variation":
        instance_ids = tuple(grid.instance_id(index) for index in grid.active_indices)
        residuals = AppearanceResiduals.encode(
            [offsets[instance_id] for instance_id in instance_ids],
            instance_ids,
        )
        residual_q8 = StructuralRepresentation(
            unique=unique,
            terminal=terminal,
            grid=grid,
            fixture_id=FIXTURE_ID,
            appearance_regime=appearance_regime,
            principal_seed=principal_seed,
            residuals=residuals,
            method_id="struct_residual_q8",
        )
    return FixtureRepresentations(
        explicit=explicit,
        shared=shared,
        residual_q8=residual_q8,
        reference_offsets=offsets,
        protocol_digest=PROTOCOL_DIGEST,
    )


def _matrix4(
    values: tuple[float, ...],
) -> tuple[tuple[float, float, float, float], ...]:
    return cast(
        tuple[tuple[float, float, float, float], ...],
        tuple(
            tuple(f32(value) for value in values[offset : offset + 4])
            for offset in range(0, 16, 4)
        ),
    )


def evaluation_cameras(
    *, repo_root: Path, principal_seed: int = 17
) -> tuple[Camera, ...]:
    _ = repo_root
    if principal_seed not in PRINCIPAL_SEEDS:
        raise ValueError("principal_seed is not part of the reproducibility seed set")
    positions = (
        (-2.2, -7.3, 2.4),
        (2.2, -7.3, 2.4),
        (0.0, -7.0, 2.0),
        (0.0, -7.5, 3.6),
        (-3.0, -7.8, 1.6),
        (3.0, -7.8, 1.6),
        (-1.4, -6.8, 3.0),
        (1.4, -6.8, 3.0),
    )
    target = (0.0, 0.0, 2.0)
    cameras: list[Camera] = []
    for camera_id, position in zip(
        CAMERA_IDS,
        positions,
        strict=True,
    ):
        world_from_camera = look_at_world_from_camera(position, target)
        cameras.append(
            Camera(
                camera_id=camera_id,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fx=230.4,
                fy=230.4,
                cx=127.5,
                cy=127.5,
                world_from_camera=_matrix4(world_from_camera),
            )
        )
    return tuple(cameras)


def camera_digest(cameras: tuple[Camera, ...]) -> str:
    return content_digest([asdict(camera) for camera in cameras])


def fixture_summary(*, repo_root: Path, principal_seed: int = 17) -> dict[str, Any]:
    exact = build_fixture_representations(
        repo_root=repo_root,
        appearance_regime="exact",
        principal_seed=principal_seed,
    )
    low = build_fixture_representations(
        repo_root=repo_root,
        appearance_regime="low_variation",
        principal_seed=principal_seed,
    )
    cameras = evaluation_cameras(repo_root=repo_root, principal_seed=principal_seed)
    return {
        "fixture_id": FIXTURE_ID,
        "principal_seed": principal_seed,
        "protocol_digest": exact.protocol_digest,
        "exact_digest": exact.scientific_digest,
        "low_variation_digest": low.scientific_digest,
        "camera_digest": camera_digest(cameras),
        "camera_ids": [camera.camera_id for camera in cameras],
        "counts": {
            "unique": exact.shared.unique.gaussians.count,
            "terminal": exact.shared.terminal.gaussians.count,
            "instances": exact.shared.grid.instance_count,
            "stored_structural": exact.shared.stored_gaussian_count,
            "materialized": exact.shared.materialized_gaussian_count,
        },
    }
