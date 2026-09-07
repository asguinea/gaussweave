from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from gaussweave.representation.accounting import account_representation
from gaussweave.representation.fixture import (
    build_fixture_representations,
    derive_seed,
    evaluation_cameras,
    fixture_summary,
    grid_for_repeat_count,
    instance_offsets,
)
from gaussweave.representation.models import (
    AppearanceResiduals,
    GaussianArrays,
    SimilarityTransform,
    materialize,
    quantize_q8,
)
from gaussweave.representation.serialization import (
    RepresentationIOError,
    load_representation,
    save_representation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def exact():
    return build_fixture_representations(
        repo_root=REPO_ROOT,
        appearance_regime="exact",
        repeat_count=16,
        principal_seed=17,
    )


@pytest.fixture(scope="module")
def low():
    return build_fixture_representations(
        repo_root=REPO_ROOT,
        appearance_regime="low_variation",
        repeat_count=16,
        principal_seed=17,
    )


@pytest.mark.unit
def test_t01_gaussian_array_validation_rejects_bad_scale(exact) -> None:
    values = exact.explicit.gaussians
    with pytest.raises(ValueError, match="strictly positive"):
        GaussianArrays(
            means=values.means,
            quaternions=values.quaternions,
            scales=((0.0, 1.0, 1.0),) + values.scales[1:],
            opacities=values.opacities,
            colors=values.colors,
            stable_ids=values.stable_ids,
        )


@pytest.mark.unit
def test_t02_direct_rgb_and_parameter_conventions_are_explicit(exact) -> None:
    assert exact.explicit.gaussians.sh_degree == 0
    assert exact.explicit.gaussians.to_renderable().appearance_mode == "direct_rgb"
    assert all(
        math.isclose(sum(q * q for q in row), 1.0, abs_tol=2e-5)
        for row in exact.explicit.gaussians.quaternions
    )


@pytest.mark.unit
def test_t03_terminal_contract(exact) -> None:
    terminal = exact.shared.terminal
    assert terminal.component_id == "terminal-panel-v1"
    assert terminal.gaussians.count == 256
    assert terminal.local_bounds[0] != terminal.local_bounds[1]
    assert terminal.provenance["stream"] == "terminal_appearance"


@pytest.mark.unit
def test_t04_unique_component_contains_no_repeated_terminal(exact) -> None:
    unique = exact.shared.unique
    assert unique.gaussians.count == 512
    assert unique.provenance["contains_repeated_terminal_content"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("count", "shape"), [(4, (4, 1)), (8, (4, 2)), (16, (4, 4)), (32, (4, 8))]
)
def test_t05_grid_order_and_active_instances(
    count: int, shape: tuple[int, int]
) -> None:
    grid = grid_for_repeat_count(count)
    assert (grid.rows, grid.columns) == shape
    assert grid.instance_count == count
    assert grid.instance_id(grid.active_indices[0]) == "instance-r00-c00"


@pytest.mark.unit
def test_t06_q8_rounding_boundaries_and_saturation() -> None:
    assert quantize_q8(0.5, 1.0) == (1, False)
    assert quantize_q8(-0.5, 1.0) == (-1, False)
    assert quantize_q8(1000.0, 1.0) == (127, True)
    assert quantize_q8(-1000.0, 1.0) == (-128, True)
    residual = AppearanceResiduals.encode([(0.08, -0.08, 0.0)], ["instance-r00-c00"])
    assert residual.decode("instance-r00-c00")[0] == pytest.approx(0.08, abs=1e-6)


@pytest.mark.unit
def test_t07_identity_and_translation_transform() -> None:
    identity = SimilarityTransform((0.0, 0.0, 0.0))
    translated = SimilarityTransform((1.0, 2.0, 3.0))
    assert identity.apply_mean((0.25, -0.5, 1.0)) == pytest.approx((0.25, -0.5, 1.0))
    assert translated.apply_mean((0.25, -0.5, 1.0)) == pytest.approx((1.25, 1.5, 4.0))


@pytest.mark.unit
def test_t08_rotation_and_uniform_scale_from_matrix() -> None:
    transform = SimilarityTransform.from_matrix(
        (0, -2, 0, 1, 2, 0, 0, 2, 0, 0, 2, 3, 0, 0, 0, 1)
    )
    assert transform.uniform_scale == pytest.approx(2.0)
    assert transform.apply_mean((1.0, 0.0, 0.0)) == pytest.approx((1.0, 4.0, 3.0))


@pytest.mark.unit
@pytest.mark.parametrize(
    "matrix",
    [
        (-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        (1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ],
)
def test_t09_invalid_affine_transforms_are_rejected(matrix) -> None:
    with pytest.raises(ValueError):
        SimilarityTransform.from_matrix(matrix)


@pytest.mark.unit
def test_t10_exact_materialization_matches_independent_reference(exact) -> None:
    expanded = materialize(exact.shared)
    assert expanded.gaussians == exact.explicit.gaussians
    assert (
        expanded.gaussians.scientific_digest
        == exact.explicit.gaussians.scientific_digest
    )
    assert expanded.stored_gaussian_count == 768
    assert expanded.materialized_gaussian_count == 4608


@pytest.mark.unit
def test_t11_low_variation_changes_only_appearance(low) -> None:
    shared = materialize(low.shared).gaussians
    reference = low.explicit.gaussians
    assert shared.means == reference.means
    assert shared.quaternions == reference.quaternions
    assert shared.scales == reference.scales
    assert shared.opacities == reference.opacities
    assert shared.colors != reference.colors


@pytest.mark.unit
def test_t12_q8_residuals_reduce_array_appearance_error(low) -> None:
    shared = materialize(low.shared).gaussians
    q8 = materialize(low.residual_q8).gaussians
    reference = low.explicit.gaussians
    shared_error = sum(
        (a - b) ** 2
        for x, y in zip(shared.colors, reference.colors, strict=True)
        for a, b in zip(x, y, strict=True)
    )
    q8_error = sum(
        (a - b) ** 2
        for x, y in zip(q8.colors, reference.colors, strict=True)
        for a, b in zip(x, y, strict=True)
    )
    assert q8_error < shared_error
    assert low.residual_q8.residuals.saturation_count == 0


@pytest.mark.unit
def test_t13_explicit_reference_has_ordinary_full_arrays(exact) -> None:
    arrays = exact.explicit.gaussians
    assert arrays.count == 512 + 16 * 256
    assert arrays.stable_ids[512] == "instance-r00-c00/terminal-g0000"


@pytest.mark.unit
@pytest.mark.serialization
def test_t14_explicit_serialization_round_trip(exact, tmp_path: Path) -> None:
    root = tmp_path / "explicit"
    save_representation(exact.explicit, root)
    loaded = load_representation(root)
    assert loaded == exact.explicit


@pytest.mark.unit
@pytest.mark.serialization
def test_t15_structural_serialization_round_trip(low, tmp_path: Path) -> None:
    root = tmp_path / "structural"
    save_representation(low.residual_q8, root)
    loaded = load_representation(root)
    assert loaded == low.residual_q8
    assert materialize(loaded).gaussians == materialize(low.residual_q8).gaussians


@pytest.mark.unit
@pytest.mark.serialization
def test_t16_corruption_is_rejected(low, tmp_path: Path) -> None:
    root = tmp_path / "corrupt"
    save_representation(low.residual_q8, root)
    path = root / "arrays" / "terminal_means.npy"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(RepresentationIOError, match="checksum"):
        load_representation(root)


@pytest.mark.unit
@pytest.mark.serialization
def test_t17_extra_files_and_unsafe_manifest_paths_are_rejected(
    exact, tmp_path: Path
) -> None:
    root = tmp_path / "extra"
    save_representation(exact.explicit, root)
    (root / "unexpected.tmp").write_text("not allowed", encoding="utf-8")
    with pytest.raises(RepresentationIOError, match="file set mismatch"):
        load_representation(root)


@pytest.mark.unit
@pytest.mark.resource
def test_t18_complete_accounting_is_actual_and_additive(low, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    structural = tmp_path / "structural"
    save_representation(low.explicit, explicit)
    save_representation(low.residual_q8, structural)
    record = account_representation(structural, reference_root=explicit)
    assert sum(record["category_bytes"].values()) == record["complete_serialized_bytes"]
    assert record["compression_ratio_reference_over_method"] > 1.0
    assert record["file_count"] == len(record["files"])


@pytest.mark.unit
def test_t19_repeated_clean_process_fixture_digests_match() -> None:
    script = (
        "from pathlib import Path; import json; "
        "from gaussweave.representation.fixture import fixture_summary; "
        "print(json.dumps(fixture_summary(repo_root=Path('.')),sort_keys=True))"
    )
    first = subprocess.check_output(
        [sys.executable, "-c", script], cwd=REPO_ROOT, text=True
    )
    second = subprocess.check_output(
        [sys.executable, "-c", script], cwd=REPO_ROOT, text=True
    )
    assert first == second


@pytest.mark.unit
def test_t20_seed_streams_are_isolated_and_sensitive() -> None:
    grid = grid_for_repeat_count(16)
    assert instance_offsets(17, grid) != instance_offsets(29, grid)
    assert derive_seed(17, "instance_variation") != derive_seed(17, "cameras")
    assert [
        asdict(camera)
        for camera in evaluation_cameras(repo_root=REPO_ROOT, principal_seed=17)
    ] == [
        asdict(camera)
        for camera in evaluation_cameras(repo_root=REPO_ROOT, principal_seed=29)
    ]


@pytest.mark.unit
def test_t21_cpu_import_does_not_load_torch_or_gsplat() -> None:
    script = (
        "import sys; import gaussweave; import gaussweave.representation.models; "
        "print('torch' in sys.modules, 'gsplat' in sys.modules)"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", script], cwd=REPO_ROOT, text=True
    )
    assert output.strip() == "False False"


@pytest.mark.unit
def test_t22_fixture_summary_has_frozen_counts_and_cameras() -> None:
    summary = fixture_summary(repo_root=REPO_ROOT)
    assert summary["counts"] == {
        "unique": 512,
        "terminal": 256,
        "instances": 16,
        "stored_structural": 768,
        "materialized": 4608,
    }
    assert summary["camera_ids"] == [f"cam-eval-{index:04d}" for index in range(8)]
