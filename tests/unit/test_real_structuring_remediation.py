from __future__ import annotations

import pytest

from gaussweave.real_structuring.models import StructuringError
from gaussweave.real_structuring.remediation import (
    EQUAL_INTERIOR_HALF_EXTENTS,
    REGION_VERSION,
    compare_dimensions,
    remediation_stop_decision,
    select_candidate,
    transform_anisotropic_gaussians,
    validate_diagonal_scale_transform,
)
from gaussweave.real_structuring.residuals import (
    SH1_CODEC_VERSION,
    encode_sh1_q8,
)


def _candidate(candidate_id: str, *, score: float, passing: bool = True) -> dict:
    return {
        "candidate_id": candidate_id,
        "semantic_description": candidate_id,
        "geometry_gate": passing,
        "visibility_fitting_gate": passing,
        "separation_gate": passing,
        "representation_gate": passing,
        "appeal_gate": passing,
        "training_only_score": score,
        "transform_class": "rigid",
        "held_out_metric_that_must_not_select": -999,
    }


def test_remediation_candidate_dimension_comparison() -> None:
    equal = {
        key: (0.48, 0.36, 0.24) for key in ("panel-rear", "panel-middle", "panel-front")
    }
    assert compare_dimensions(equal)["rigid_dimension_gate"] is True
    unequal = dict(equal)
    unequal["panel-rear"] = (0.60, 0.36, 0.24)
    assert compare_dimensions(unequal)["rigid_dimension_gate"] is False


def test_equal_core_dimensions_are_identical_by_construction() -> None:
    dimensions = tuple(2 * value for value in EQUAL_INTERIOR_HALF_EXTENTS)
    assert dimensions == pytest.approx((0.48, 0.36, 0.20))
    assert REGION_VERSION == "truck-bed-panel-interiors-v2"


def test_candidate_selection_ignores_held_out_values() -> None:
    candidates = [_candidate("a", score=0.8), _candidate("c", score=0.7)]
    first = select_candidate(candidates)
    candidates[0]["held_out_metric_that_must_not_select"] = -1e9
    candidates[1]["held_out_metric_that_must_not_select"] = 1e9
    second = select_candidate(candidates)
    assert first == second
    assert first["selected_candidate_id"] == "a"
    assert first["held_out_evidence_used"] is False


def test_candidate_hard_gate_excludes_higher_scoring_failure() -> None:
    candidates = [
        _candidate("failed", score=1.0, passing=False),
        _candidate("pass", score=0.5),
    ]
    assert select_candidate(candidates)["selected_candidate_id"] == "pass"


def test_diagonal_scale_transforms_anisotropic_gaussian_covariance() -> None:
    torch = pytest.importorskip("torch")
    means = torch.tensor([[1.0, 2.0, 3.0]])
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    scales = torch.tensor([[1.0, 2.0, 3.0]])
    matrix = (
        1.2,
        0,
        0,
        4,
        0,
        0.8,
        0,
        5,
        0,
        0,
        1.1,
        6,
        0,
        0,
        0,
        1,
    )
    new_means, new_quaternions, new_scales = transform_anisotropic_gaussians(
        means, quaternions, scales, matrix
    )
    assert new_means == pytest.approx(torch.tensor([[5.2, 6.6, 9.3]]))
    assert sorted(new_scales[0].tolist()) == pytest.approx([1.2, 1.6, 3.3])
    assert torch.linalg.vector_norm(new_quaternions, dim=1) == pytest.approx(
        torch.ones(1)
    )


def test_diagonal_scale_rejects_shear_and_out_of_bounds() -> None:
    shear = (1, 0.2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    with pytest.raises(StructuringError, match="shear"):
        validate_diagonal_scale_transform(shear)
    oversized = (1.3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    with pytest.raises(StructuringError, match="bounds"):
        validate_diagonal_scale_transform(oversized)


def test_compact_sh1_q8_codec_is_deterministic_and_canonical_zero() -> None:
    offsets = [
        [[0.01 * (coefficient + 1), -0.02, 0.03] for coefficient in range(4)],
        [[0.5, 0.5, 0.5] for _ in range(4)],
        [[-0.04, 0.02, 0.01] for _ in range(4)],
    ]
    first = encode_sh1_q8(offsets)
    second = encode_sh1_q8(offsets)
    assert first == second
    assert first["codec_version"] == SH1_CODEC_VERSION
    assert first["values"][1] == [[0, 0, 0]] * 4
    assert first["saturation_count"] == 0
    assert (
        len(
            bytes(
                value & 0xFF
                for row in first["values"]
                for coefficient in row
                for value in coefficient
            )
        )
        == 36
    )


def test_remediation_stop_rule_activates_fallback() -> None:
    stopped = remediation_stop_decision(
        selected_candidate="candidate-a-equal-interiors",
        visual_decision="not_credible",
        residual_gate=False,
    )
    assert stopped["status"] == "activate_explicit_fallback"
    assert stopped["truck_suitable_for_evaluation"] is False
    assert stopped["next_step"] == "select_an_alternative_scene"


def test_remediation_stop_rule_allows_credible_result() -> None:
    result = remediation_stop_decision(
        selected_candidate="candidate-a-equal-interiors",
        visual_decision="credible_with_limitations",
        residual_gate=True,
    )
    assert result["status"] == "truck_remediation_passed"
    assert result["truck_suitable_for_evaluation"] is True
