from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from gaussweave.data.render_passes import (
    AlphaConvention,
    DepthConvention,
    IntegerConvention,
    NormalConvention,
    RenderPassError,
    RGBConvention,
    StableIdRegistry,
    build_render_pass_policy,
    default_conventions,
    engineering_probe_registry,
)
from gaussweave.data.scene_config import load_scene_configuration

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/scenes/facade-development-s101.json"


def test_convention_record_is_explicit_and_self_validating() -> None:
    conventions = default_conventions()

    assert conventions.rgb.channel_order == "RGB"
    assert conventions.rgb.transfer_function == "sRGB"
    assert conventions.depth.units == "meters"
    assert conventions.depth.axis_semantics == "distance_along_normalized_camera_ray"
    assert conventions.normals.coordinate_space == "world"
    assert conventions.masks_and_ids.interpolation == "none_nearest"
    assert conventions.alpha.meaning == "straight_surface_coverage"
    assert conventions.scientific_digest.startswith("sha256:")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RGBConvention(bit_depth=12),
        lambda: DepthConvention(invalid_value=-1.0),
        lambda: NormalConvention(coordinate_space="camera"),
        lambda: IntegerConvention(bit_depth=8),
        lambda: AlphaConvention(numeric_range=(-1.0, 1.0)),
    ],
)
def test_contradictory_conventions_fail(factory: object) -> None:
    with pytest.raises(RenderPassError):
        factory()  # type: ignore[operator]


def test_engineering_registry_has_stable_distinct_identity_domains() -> None:
    registry = engineering_probe_registry()

    assert registry.semantic_id("background") == 0
    assert (
        registry.instance_ids["instance-probe-01"]
        != registry.instance_ids["instance-probe-02"]
    )
    assert registry.instance_to_terminal[1] == registry.instance_to_terminal[2] == 1
    assert registry.object_ids["probe-structure"] == 1
    assert registry.material_ids["probe-base"] == 1
    with pytest.raises(RenderPassError, match="unregistered"):
        registry.semantic_id("unknown")


def test_duplicate_and_overflowing_registry_ids_fail() -> None:
    base = engineering_probe_registry()
    with pytest.raises(RenderPassError, match="unique"):
        StableIdRegistry(
            base.registry_version,
            base.background_id,
            base.semantic_categories,
            base.terminal_ids,
            {"one": 1, "two": 1},
            {1: 1},
            base.object_ids,
            base.material_ids,
        )
    with pytest.raises(RenderPassError, match="uint16"):
        StableIdRegistry(
            base.registry_version,
            base.background_id,
            base.semantic_categories,
            base.terminal_ids,
            {"overflow": 65536},
            {65536: 1},
            base.object_ids,
            base.material_ids,
        )


def test_pass_enablement_follows_scene_output_configuration() -> None:
    config = load_scene_configuration(CONFIG)
    base = build_render_pass_policy(config)
    assert "object_id" in base.enabled_passes
    assert "alpha" not in base.enabled_passes
    assert "material_id" not in base.enabled_passes

    optional_config = replace(
        config,
        outputs=replace(
            config.outputs,
            alpha=True,
            object_ids=False,
            material_ids=True,
        ),
    )
    optional = build_render_pass_policy(optional_config)
    assert "alpha" in optional.enabled_passes
    assert "object_id" not in optional.enabled_passes
    assert "material_id" in optional.enabled_passes
    assert optional.color_management["stored_rgb_is_display_referred"] is True
    assert optional.conventions.rgb.background_composition.startswith("transparent")
    with pytest.raises(RenderPassError, match="NPY depth"):
        build_render_pass_policy(
            replace(
                config,
                outputs=replace(config.outputs, depth_format="exr"),
            )
        )


def test_render_validation_import_is_cpu_safe_in_fresh_process() -> None:
    code = """
import sys
import gaussweave.data.render_passes
forbidden = {"bpy", "torch", "gsplat"} & set(sys.modules)
assert not forbidden, sorted(forbidden)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
