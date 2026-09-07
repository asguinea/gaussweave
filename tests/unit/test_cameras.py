from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from gaussweave.config.resolution import content_digest
from gaussweave.data.camera_math import (
    angular_distance_degrees,
    focal_from_horizontal_fov,
    horizontal_fov_degrees,
    look_at_world_from_camera,
    pose_axes,
    project_point,
    rotation_determinant,
    validate_rigid_pose,
)
from gaussweave.data.cameras import (
    CAMERA_FORMAT,
    CameraValidationError,
    generate_camera_collection,
    load_camera_collection,
    validate_camera_collection,
    write_camera_artifacts,
)
from gaussweave.data.scene_config import load_scene_configuration
from gaussweave.results.artifacts import (
    VerificationState,
    load_inventory,
    verify_inventory,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "facade": ROOT / "configs/scenes/facade-development-s101.json",
    "corridor": ROOT / "configs/scenes/corridor-development-s202.json",
    "colonnade": ROOT / "configs/scenes/colonnade-development-s303.json",
}


@pytest.mark.unit
def test_look_at_uses_project_opencv_axes() -> None:
    pose = look_at_world_from_camera((0.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    right, down, forward, position = pose_axes(pose)
    assert right == pytest.approx((1.0, 0.0, 0.0))
    assert down == pytest.approx((0.0, 1.0, 0.0))
    assert forward == pytest.approx((0.0, 0.0, 1.0))
    assert position == (0.0, 0.0, 0.0)


@pytest.mark.unit
def test_look_at_parallel_up_has_stable_fallback() -> None:
    first = look_at_world_from_camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    second = look_at_world_from_camera((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert first == second
    validate_rigid_pose(first)


@pytest.mark.unit
def test_look_at_is_rigid_and_right_handed() -> None:
    pose = look_at_world_from_camera((2.0, -3.0, 1.0), (-1.0, 4.0, 2.5))
    validate_rigid_pose(pose)
    assert rotation_determinant(pose) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_focal_and_fov_round_trip() -> None:
    focal = focal_from_horizontal_fov(640, 61.5)
    assert horizontal_fov_degrees(640, focal) == pytest.approx(61.5)


@pytest.mark.unit
def test_projection_places_target_at_principal_point() -> None:
    pose = look_at_world_from_camera((3.0, -2.0, 1.0), (0.0, 0.0, 1.5))
    x, y, depth = project_point(
        (0.0, 0.0, 1.5), pose, fx=500.0, fy=500.0, cx=320.0, cy=240.0
    )
    assert (x, y) == pytest.approx((320.0, 240.0), abs=1e-10)
    assert depth > 0.0


@pytest.mark.unit
def test_angular_distance() -> None:
    assert angular_distance_degrees((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == 90.0


@pytest.mark.unit
@pytest.mark.parametrize("family", ("facade", "corridor", "colonnade"))
def test_family_policy_generates_exact_counts(family: str) -> None:
    config = load_scene_configuration(CONFIGS[family])
    collection = generate_camera_collection(config)
    assert collection.counts == {
        "train": config.camera.train_count,
        "validation": config.camera.validation_count,
        "test": config.camera.test_count,
    }
    assert len(collection.records) == sum(collection.counts.values())
    assert collection.family == family


@pytest.mark.unit
@pytest.mark.parametrize("family", ("facade", "corridor", "colonnade"))
def test_family_policy_passes_coverage_and_separation(family: str) -> None:
    result = validate_camera_collection(
        generate_camera_collection(load_scene_configuration(CONFIGS[family]))
    )
    assert result["valid"] is True
    assert result["minimum_structure_coverage"] >= 0.65


@pytest.mark.unit
def test_same_seed_is_byte_stable() -> None:
    config = load_scene_configuration(CONFIGS["facade"])
    first = generate_camera_collection(config)
    second = generate_camera_collection(config)
    assert first.to_dict() == second.to_dict()


@pytest.mark.unit
def test_changed_camera_seed_changes_scientific_identity() -> None:
    config = load_scene_configuration(CONFIGS["facade"])
    first = generate_camera_collection(config)
    second = generate_camera_collection(config, camera_seed=config.camera.seed + 1)
    assert first.scientific_digest != second.scientific_digest
    assert first.records[0].position_m != second.records[0].position_m


@pytest.mark.unit
def test_non_camera_seed_does_not_change_camera_draws() -> None:
    config = load_scene_configuration(CONFIGS["facade"])
    changed_seeds = dict(config.derived_seeds)
    changed_seeds["geometry"] += 1
    changed = replace(config, derived_seeds=changed_seeds)
    assert (
        generate_camera_collection(config).records
        == generate_camera_collection(changed).records
    )


@pytest.mark.unit
def test_family_intent_mapping_order_does_not_change_output() -> None:
    config = load_scene_configuration(CONFIGS["facade"])
    reordered_policy = replace(
        config.camera,
        family_intent=dict(reversed(tuple(config.camera.family_intent.items()))),
    )
    reordered = replace(config, camera=reordered_policy)
    assert generate_camera_collection(config) == generate_camera_collection(reordered)


@pytest.mark.unit
def test_split_ids_use_val_token_but_internal_validation_name() -> None:
    collection = generate_camera_collection(
        load_scene_configuration(CONFIGS["corridor"]),
        train_count=3,
        validation_count=2,
        test_count=1,
    )
    assert all(item.startswith("cam-val-") for item in collection.splits["validation"])
    assert all(
        record.split == "validation"
        for record in collection.records
        if record.camera_id.startswith("cam-val-")
    )


@pytest.mark.unit
def test_split_holdouts_span_trajectory() -> None:
    collection = generate_camera_collection(load_scene_configuration(CONFIGS["facade"]))
    for split in ("train", "validation", "test"):
        parameters = [
            record.trajectory_parameter
            for record in collection.records
            if record.split == split
        ]
        assert max(parameters) - min(parameters) > 0.7


@pytest.mark.unit
def test_exact_cross_split_duplicate_is_rejected() -> None:
    original = generate_camera_collection(
        load_scene_configuration(CONFIGS["colonnade"])
    )
    train = next(record for record in original.records if record.split == "train")
    validation = next(
        record for record in original.records if record.split == "validation"
    )
    duplicate = replace(
        validation,
        world_from_camera=train.world_from_camera,
        position_m=train.position_m,
        right=train.right,
        down=train.down,
        forward=train.forward,
        target_m=train.target_m,
    )
    records = tuple(
        duplicate if record.camera_id == validation.camera_id else record
        for record in original.records
    )
    changed = replace(original, records=records, scientific_digest="")
    changed = replace(
        changed,
        scientific_digest=content_digest(changed.scientific_projection()),
    )
    with pytest.raises(CameraValidationError, match="split leakage"):
        validate_camera_collection(changed)


@pytest.mark.unit
def test_stricter_coverage_threshold_has_clear_failure() -> None:
    collection = generate_camera_collection(
        load_scene_configuration(CONFIGS["corridor"])
    )
    with pytest.raises(CameraValidationError, match="coverage below threshold"):
        validate_camera_collection(collection, minimum_structure_coverage=1.01)


@pytest.mark.unit
def test_record_rejects_wrong_split_id() -> None:
    record = generate_camera_collection(
        load_scene_configuration(CONFIGS["facade"])
    ).records[0]
    with pytest.raises(CameraValidationError, match="camera ID"):
        replace(record, camera_id="cam-test-9999")


@pytest.mark.unit
def test_record_rejects_nonrigid_transform() -> None:
    record = generate_camera_collection(
        load_scene_configuration(CONFIGS["facade"])
    ).records[0]
    matrix = list(record.world_from_camera)
    matrix[0] *= 2.0
    with pytest.raises(ValueError, match="unit length"):
        replace(record, world_from_camera=tuple(matrix))


@pytest.mark.unit
@pytest.mark.serialization
def test_collection_round_trip(tmp_path: Path) -> None:
    collection = generate_camera_collection(
        load_scene_configuration(CONFIGS["colonnade"])
    )
    result = write_camera_artifacts(collection, tmp_path)
    loaded = load_camera_collection(tmp_path / "cameras/cameras.json")
    assert loaded == collection
    assert result.inventory_state == VerificationState.VALID.value


@pytest.mark.unit
def test_camera_artifact_paths_and_split_membership(tmp_path: Path) -> None:
    collection = generate_camera_collection(load_scene_configuration(CONFIGS["facade"]))
    write_camera_artifacts(collection, tmp_path)
    expected = {
        "cameras.json",
        "train.txt",
        "validation.txt",
        "test.txt",
        "diagnostics.json",
        "checksums.sha256",
    }
    assert {path.name for path in (tmp_path / "cameras").iterdir()} == expected
    validation_ids = (tmp_path / "cameras/validation.txt").read_text().splitlines()
    assert validation_ids == list(collection.splits["validation"])


@pytest.mark.unit
def test_camera_artifact_inventory_is_strict(tmp_path: Path) -> None:
    collection = generate_camera_collection(load_scene_configuration(CONFIGS["facade"]))
    write_camera_artifacts(collection, tmp_path)
    parsed = load_inventory(tmp_path / "artifact-inventory.json")
    assert (
        verify_inventory(parsed, tmp_path, strict=True).state is VerificationState.VALID
    )
    (tmp_path / "unexpected.txt").write_text("unexpected")
    assert (
        verify_inventory(parsed, tmp_path, strict=True).state
        is VerificationState.INVALID
    )


@pytest.mark.unit
def test_collection_declares_accepted_format_and_no_occlusion_claim() -> None:
    collection = generate_camera_collection(load_scene_configuration(CONFIGS["facade"]))
    assert collection.format == CAMERA_FORMAT
    assert collection.coverage["exact_occlusion_claim"] is False
    assert all(
        record.coverage["exact_occlusion_claim"] is False
        for record in collection.records
    )


@pytest.mark.unit
def test_all_numeric_record_fields_are_finite() -> None:
    collection = generate_camera_collection(
        load_scene_configuration(CONFIGS["corridor"])
    )
    for record in collection.records:
        assert all(math.isfinite(value) for value in record.world_from_camera)
        assert all(math.isfinite(value) for value in record.position_m)
