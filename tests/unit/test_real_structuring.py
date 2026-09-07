from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gaussweave.real_structuring.frames import build_frames
from gaussweave.real_structuring.models import (
    CORE_COUNTS,
    INSTANCE_IDS,
    PILOT_VERSION,
    SOURCE_COUNT,
    OwnershipStatus,
    PanelFrame,
    SimilarityRegistration,
    StructuringError,
    validate_similarity_matrix,
)
from gaussweave.real_structuring.regions import validate_ownership
from gaussweave.real_structuring.residuals import encode_q8


def _frame(
    instance_id: str = "panel-middle",
    axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
) -> PanelFrame:
    return PanelFrame(
        instance_id=instance_id,
        origin_world=(0.0, 0.0, 0.0),
        axes_world=axes,
        local_lower=(-1.0, -1.0, -0.1),
        local_upper=(1.0, 1.0, 0.1),
        support_plane_normal_world=(0.0, 0.0, 1.0),
        support_plane_offset=0.0,
        source_annotation="fixture",
        confidence="medium",
        method="fixture",
    )


def test_t3_frame_is_right_handed_orthonormal_and_finite() -> None:
    frame = _frame()
    assert frame.width == 2
    assert frame.height == 2
    assert frame.depth == pytest.approx(0.2)
    assert frame.world_from_local == (
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
    )


@pytest.mark.parametrize(
    "axes,match",
    [
        (((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "right"),
        (((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "orthonormal"),
    ],
)
def test_t3_t8_invalid_frames_fail(
    axes: tuple[tuple[float, float, float], ...], match: str
) -> None:
    with pytest.raises(StructuringError, match=match):
        _frame(axes=axes)


def test_t3_t4_annotation_frames_are_deterministic_and_consistent() -> None:
    annotation = {
        "instances": [
            {
                "instance_id": instance_id,
                "annotation_confidence": "medium",
                "provisional_3d_box": {
                    "center_world": [float(index), 0.0, 0.0],
                    "axes_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "support_plane": {
                        "normal_world": [0, 0, 1],
                        "offset": 0,
                    },
                },
            }
            for index, instance_id in enumerate(INSTANCE_IDS)
        ]
    }
    bounds = {
        instance_id: ((-1.0, -0.5, -0.1), (1.0, 0.5, 0.1))
        for instance_id in INSTANCE_IDS
    }
    left = build_frames(annotation, bounds)
    right = build_frames(annotation, bounds)
    assert left == right
    assert left[1].instance_id == "panel-middle"
    assert len({frame.axes_world for frame in left}) == 1


def test_t8_reflection_singular_shear_and_excessive_scale_fail() -> None:
    with pytest.raises(StructuringError, match="reflections"):
        validate_similarity_matrix(
            (-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            maximum_scale_change=0.75,
        )
    with pytest.raises(StructuringError, match="nonsingular"):
        validate_similarity_matrix(
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            maximum_scale_change=0.75,
        )
    with pytest.raises(StructuringError, match="uniform"):
        validate_similarity_matrix(
            (1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            maximum_scale_change=0.75,
        )
    with pytest.raises(StructuringError, match="exceeds"):
        validate_similarity_matrix(
            (2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1),
            maximum_scale_change=0.75,
        )


def test_t18_t19_q8_is_deterministic_canonical_zero_and_isolated() -> None:
    offsets = ((0.02, -0.01, 0.0), (0.1, 0.1, 0.1), (-0.03, 0.01, 0.02))
    left = encode_q8(offsets)
    right = encode_q8(offsets)
    assert left == right
    assert left.values[1] == (0, 0, 0)
    assert left.saturation_count == 0
    assert left.decode("panel-middle") == (0, 0, 0)
    assert left.decode("panel-rear") != left.decode("panel-front")


def test_t18_q8_saturation_is_recorded_with_declared_scale() -> None:
    encoded = encode_q8(
        ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
        scale=0.001,
    )
    assert encoded.saturation_count == 2
    assert encoded.values[0][0] == 127
    assert encoded.values[2][0] == -128


def test_t18_q8_policy_mismatch_fails() -> None:
    encoded = encode_q8(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
    with pytest.raises(StructuringError, match="policy mismatch"):
        type(encoded)(
            encoded.instance_ids,
            encoded.values,
            encoded.scale,
            encoded.saturation_count,
            rounding="ties_to_even",
        )


def test_t1_t2_ownership_partition_and_background_complement(tmp_path: Path) -> None:
    payload = bytearray([int(OwnershipStatus.EXPLICIT_BACKGROUND)] * SOURCE_COUNT)
    start = 0
    for status, instance_id in zip(
        (
            OwnershipStatus.PANEL_REAR_CORE,
            OwnershipStatus.PANEL_MIDDLE_CORE,
            OwnershipStatus.PANEL_FRONT_CORE,
        ),
        INSTANCE_IDS,
        strict=True,
    ):
        count = CORE_COUNTS[instance_id]
        payload[start : start + count] = bytes([int(status)]) * count
        start += count
    payload[start : start + 17] = bytes([int(OwnershipStatus.BOUNDARY_GUARD)]) * 17
    path = tmp_path / "ownership.u8"
    path.write_bytes(payload)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "ownership.json").write_text(
        json.dumps(
            {
                "pilot_version": PILOT_VERSION,
                "payload": "ownership.u8",
                "payload_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    result = validate_ownership(tmp_path.resolve())
    assert result["source_gaussian_count"] == SOURCE_COUNT
    assert result["core_total"] == 39_791
    assert result["background_complement_count"] == 2_501_435
    assert sum(result["status_counts"].values()) == SOURCE_COUNT


def test_t1_overlapping_or_duplicated_core_count_fails(tmp_path: Path) -> None:
    payload = bytearray([int(OwnershipStatus.EXPLICIT_BACKGROUND)] * SOURCE_COUNT)
    payload[: CORE_COUNTS["panel-rear"] + 1] = bytes(
        [int(OwnershipStatus.PANEL_REAR_CORE)]
    ) * (CORE_COUNTS["panel-rear"] + 1)
    path = tmp_path / "ownership.u8"
    path.write_bytes(payload)
    import hashlib

    (tmp_path / "ownership.json").write_text(
        json.dumps(
            {
                "pilot_version": PILOT_VERSION,
                "payload": "ownership.u8",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StructuringError, match="core count mismatch"):
        validate_ownership(tmp_path.resolve())


def test_t3_nonfinite_frame_fails() -> None:
    with pytest.raises(StructuringError, match="finite"):
        PanelFrame(
            "panel-middle",
            (math.nan, 0, 0),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            (-1, -1, -1),
            (1, 1, 1),
            (0, 0, 1),
            0,
            "fixture",
            "medium",
            "fixture",
        )


def test_t14_missing_roi_mask_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    from gaussweave.real_structuring import evaluation
    from gaussweave.real_structuring.compositing import Plate

    empty = torch.zeros((8, 8), dtype=torch.float32)
    monkeypatch.setattr(
        evaluation,
        "render_fields",
        lambda _fields, _camera: Plate(
            torch.zeros((8, 8, 3)), empty, torch.ones((8, 8))
        ),
    )
    with pytest.raises(StructuringError, match="ROI mask"):
        evaluation._strict_panel_mask({}, object())


def test_t22_incomplete_accounting_fails(tmp_path: Path) -> None:
    from gaussweave.real_structuring.accounting import account

    with pytest.raises(StructuringError, match="invalid JSON"):
        account(tmp_path)


def test_t24_staged_restricted_render_is_detected() -> None:
    from gaussweave.real_structuring.models import detect_restricted_artifact_paths

    paths = [
        "src/gaussweave/real_structuring/models.py",
        "artifacts/real-scene/hero.png",
        "datasets/tnt-truck/model.ply",
    ]
    assert detect_restricted_artifact_paths(paths) == paths[1:]


def test_t4_excessive_registration_correction_fails() -> None:
    identity = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    with pytest.raises(StructuringError, match="correction exceeds"):
        SimilarityRegistration(
            instance_id="panel-rear",
            matrix=identity,
            scale=1,
            initialization_matrix=identity,
            objective_before=1,
            objective_after=0.5,
            correction_translation_m=0.36,
            rotation_angle_degrees=0,
            diagnostics={},
        )


def test_t9_altered_background_fails() -> None:
    from gaussweave.real_structuring.background import require_background_unchanged

    before = {"integrity_digest": "sha256:before"}
    after = {"integrity_digest": "sha256:after"}
    with pytest.raises(StructuringError, match="fixed background changed"):
        require_background_unchanged(before, after)
