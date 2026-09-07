"""scene configuration typed scene configuration, validation, and identity tests."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from gaussweave.config import SchemaValidationError
from gaussweave.data.scene_config import (
    SEED_STREAMS,
    ColonnadeParameters,
    CorridorParameters,
    EditOperation,
    FacadeParameters,
    SceneFamily,
    build_blender_handoff,
    derive_seed,
    derive_seeds,
    load_scene_configuration,
    scientific_projection,
    write_resolved_scene_configuration,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[2]
SCENES = ROOT / "configs" / "scenes"
INVALID_CASES = ROOT / "tests" / "fixtures" / "scene_config" / "invalid_cases.json"
VALID = (
    SCENES / "facade-development-s101.json",
    SCENES / "corridor-development-s202.json",
    SCENES / "colonnade-development-s303.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _assign(document: dict[str, Any], path: list[str | int], value: object) -> None:
    target: Any = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "family", "parameter_type"),
    [
        (VALID[0], SceneFamily.FACADE, FacadeParameters),
        (VALID[1], SceneFamily.CORRIDOR, CorridorParameters),
        (VALID[2], SceneFamily.COLONNADE, ColonnadeParameters),
    ],
)
def test_valid_family_configurations_are_typed(
    path: Path,
    family: SceneFamily,
    parameter_type: type[object],
) -> None:
    config = load_scene_configuration(path)

    assert config.family is family
    assert isinstance(config.family_parameters, parameter_type)
    assert tuple(config.derived_seeds) == SEED_STREAMS
    assert config.generator.implementation_status == "configuration_only"


def test_invalid_fixture_catalog_covers_required_semantic_failures(
    tmp_path: Path,
) -> None:
    catalog = _load_json(INVALID_CASES)
    base = _load_json(ROOT / cast(str, catalog["base"]))
    seen: set[str] = set()

    for raw_case in cast(list[dict[str, Any]], catalog["cases"]):
        case = deepcopy(raw_case)
        document = deepcopy(base)
        path = cast(list[str | int], case["path"])
        if case.get("copy_first_edit"):
            edits = cast(list[dict[str, Any]], document["edits"])
            edits.append(deepcopy(edits[0]))
        else:
            _assign(document, path, case["value"])
        fixture = tmp_path / f"{case['id']}.json"
        _write_json(fixture, document)
        with pytest.raises(SchemaValidationError) as captured:
            load_scene_configuration(fixture)
        assert cast(str, case["expected"]) in str(captured.value)
        seen.add(cast(str, case["id"]))

    assert seen == {
        "wrong-family-payload",
        "impossible-geometry",
        "empty-camera-split",
        "unsupported-output-format",
        "invalid-edit-parameter",
        "duplicate-edit-id",
        "out-of-range-missing-index",
        "unsafe-output-path",
        "resource-profile-overflow",
        "unsupported-resource-profile",
    }


def test_seed_derivation_is_stable_complete_and_order_independent() -> None:
    first = derive_seeds(101)
    second = derive_seeds(101)

    assert first == second
    assert tuple(first) == SEED_STREAMS
    assert first["geometry"] == 3613843789
    assert derive_seed(101, "annotations") == 1286660697

    script = (
        "from gaussweave.data.scene_config import derive_seed;"
        "print(derive_seed(101, 'annotations'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1286660697"


def test_seed_stream_isolation_and_single_override() -> None:
    baseline = derive_seeds(42)
    changed_master = derive_seeds(43)
    overridden = derive_seeds(42, {"materials": 7})

    assert all(baseline[name] != changed_master[name] for name in SEED_STREAMS)
    assert overridden["materials"] == 7
    assert all(
        overridden[name] == baseline[name]
        for name in SEED_STREAMS
        if name != "materials"
    )


def test_round_trip_preserves_fully_resolved_configuration(tmp_path: Path) -> None:
    original = load_scene_configuration(VALID[0])
    path = tmp_path / "round-trip.json"
    _write_json(path, original.to_dict())

    restored = load_scene_configuration(path)

    assert restored == original
    assert restored.to_dict() == original.to_dict()
    assert restored.full_digest == original.full_digest


def test_scientific_digest_sensitivity_and_operational_stability(
    tmp_path: Path,
) -> None:
    original = _load_json(VALID[0])
    base_path = tmp_path / "base.json"
    _write_json(base_path, original)
    base = load_scene_configuration(base_path)

    operational = deepcopy(original)
    operational["outputs"]["output_root"] = "datasets/generated/other-root"
    operational["provenance"]["created_at"] = "2026-01-01T00:00:00Z"
    operational["provenance"]["source_config"] = "configs/scenes/renamed.json"
    operational["notes"] = "different operational note"
    operational_path = tmp_path / "operational.json"
    _write_json(operational_path, operational)
    operational_config = load_scene_configuration(operational_path)

    scientific = deepcopy(original)
    scientific["appearance"]["variation_amplitude"] = 0.09
    scientific_path = tmp_path / "scientific.json"
    _write_json(scientific_path, scientific)
    scientific_config = load_scene_configuration(scientific_path)

    assert operational_config.scientific_digest == base.scientific_digest
    assert operational_config.full_digest != base.full_digest
    assert scientific_config.scientific_digest != base.scientific_digest
    assert scientific_projection(base)["scientific_projection_version"] == "scene-v1"


def test_resolution_outputs_are_deterministic_and_non_overwriting(
    tmp_path: Path,
) -> None:
    config = load_scene_configuration(VALID[1])
    pretty = tmp_path / "resolved.json"
    canonical = tmp_path / "resolved.canonical.json"
    digest = tmp_path / "digests.json"

    record = write_resolved_scene_configuration(
        config,
        pretty,
        canonical_output=canonical,
        digest_output=digest,
    )

    assert json.loads(pretty.read_text(encoding="utf-8")) == config.to_dict()
    assert canonical.read_bytes().endswith(b"}")
    assert not canonical.read_bytes().endswith(b"\n")
    assert record["scientific_digest"] == config.scientific_digest
    second = tmp_path / "second.json"
    second_canonical = tmp_path / "second.canonical.json"
    second_record = write_resolved_scene_configuration(
        config,
        second,
        canonical_output=second_canonical,
    )
    assert canonical.read_bytes() == second_canonical.read_bytes()
    assert record == second_record
    with pytest.raises(FileExistsError):
        write_resolved_scene_configuration(config, pretty)


def test_blender_handoff_is_explicit_and_does_not_generate(tmp_path: Path) -> None:
    config = load_scene_configuration(VALID[2])
    handoff = build_blender_handoff(
        config,
        resolved_config_path="runs/config/resolved.json",
        output_root="datasets/generated/colonnade-development-s303",
        backend="wsl",
        timeout_seconds=180.0,
        provenance_ref="runs/config/provenance.json",
    )

    arguments = handoff.ordered_script_arguments()
    assert handoff.scene_id == config.scene_id
    assert handoff.master_seed == 303
    assert handoff.generator_version == "scene-config-v1"
    assert arguments[:2] == ("--config", "runs/config/resolved.json")
    assert "--output-root" in arguments
    assert list(tmp_path.iterdir()) == []


def test_unknown_fields_and_unsupported_seed_policy_are_rejected(
    tmp_path: Path,
) -> None:
    base = _load_json(VALID[0])
    base["unexpected"] = True
    unknown = tmp_path / "unknown.json"
    _write_json(unknown, base)
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        load_scene_configuration(unknown)

    policy = _load_json(VALID[0])
    policy["seed_policy"]["version"] = "unstable"
    invalid_policy = tmp_path / "policy.json"
    _write_json(invalid_policy, policy)
    with pytest.raises(SchemaValidationError, match="sha256-v1"):
        load_scene_configuration(invalid_policy)

    scene_id = _load_json(VALID[0])
    scene_id["scene_id"] = "syn-corridor-wrong-s101"
    invalid_id = tmp_path / "scene-id.json"
    _write_json(invalid_id, scene_id)
    with pytest.raises(SchemaValidationError, match="selected family"):
        load_scene_configuration(invalid_id)


def test_all_variation_regimes_are_accepted_and_unknown_values_fail(
    tmp_path: Path,
) -> None:
    base = _load_json(VALID[0])
    enum_cases = {
        ("structure", "repetition_level"): ("low", "moderate", "high"),
        ("structure", "spacing_regime"): ("compact", "standard", "wide"),
        ("structure", "terminal_complexity"): ("simple", "compound"),
        ("structure", "regularity"): (
            "strict",
            "mildly_irregular",
            "disrupted",
            "weak",
        ),
        ("lighting", "regime"): (
            "diffuse",
            "directional",
            "mixed_interior",
            "spatially_varying",
        ),
    }
    for path, values in enum_cases.items():
        for accepted in values:
            document = deepcopy(base)
            document[path[0]][path[1]] = accepted
            fixture = tmp_path / f"{path[1]}-{accepted}.json"
            _write_json(fixture, document)
            load_scene_configuration(fixture)

    for regime in (
        "identical",
        "deterministic_low_variation",
        "alternating",
        "grouped",
        "instance_specific",
    ):
        document = deepcopy(base)
        appearance = document["appearance"]
        appearance["regime"] = regime
        if regime == "identical":
            appearance["palette"] = [appearance["base_color"]]
            appearance["variation_amplitude"] = 0.0
            appearance["exact_canonical_reuse"] = True
        if regime == "alternating":
            appearance["alternate_period"] = 2
        if regime == "grouped":
            appearance["group_size"] = 2
        fixture = tmp_path / f"appearance-{regime}.json"
        _write_json(fixture, document)
        load_scene_configuration(fixture)

    for regime in ("clear", "partial_foreground", "truncation", "inter_instance"):
        document = deepcopy(base)
        occlusion = document["occlusion"]
        occlusion["regime"] = regime
        if regime == "clear":
            occlusion.update(
                {
                    "occluder_categories": [],
                    "target_fraction": [0.0, 0.0],
                    "qualitative_level": "none",
                    "occluder_count": 0,
                    "allow_camera_truncation": False,
                }
            )
        elif regime == "truncation":
            occlusion["allow_camera_truncation"] = True
        fixture = tmp_path / f"occlusion-{regime}.json"
        _write_json(fixture, document)
        load_scene_configuration(fixture)

    invalid = deepcopy(base)
    invalid["lighting"]["regime"] = "unknown"
    fixture = tmp_path / "unknown-regime.json"
    _write_json(fixture, invalid)
    with pytest.raises(SchemaValidationError, match="must be one of"):
        load_scene_configuration(fixture)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (["camera", "near_m"], 50.0, "near plane"),
        (["camera", "horizontal_fov_degrees"], 200.0, "at most 179"),
        (
            ["camera", "minimum_angular_separation_degrees"],
            0.0,
            "greater than",
        ),
        (["outputs", "rgb_bit_depth"], 12, "incompatible"),
        (["outputs", "mask_bit_depth"], 8, "required"),
    ],
)
def test_camera_and_output_contradictions_fail(
    tmp_path: Path,
    path: list[str],
    value: object,
    message: str,
) -> None:
    document = _load_json(VALID[1])
    _assign(document, path, value)
    fixture = tmp_path / f"invalid-{path[-1]}.json"
    _write_json(fixture, document)

    with pytest.raises(SchemaValidationError, match=message):
        load_scene_configuration(fixture)


def test_every_edit_operation_accepts_exact_parameters_and_rejects_extra(
    tmp_path: Path,
) -> None:
    base = _load_json(VALID[0])
    parameters: dict[EditOperation, dict[str, object]] = {
        EditOperation.CHANGE_REPEAT_COUNT: {"new_count": 5},
        EditOperation.REMOVE_REGION: {"region_role": "facade-grid"},
        EditOperation.CHANGE_SPACING: {"spacing_m": 2.4},
        EditOperation.SCALE_TERMINAL: {"scale": [1.1, 1.0, 1.0]},
        EditOperation.REPLACE_TERMINAL: {"replacement_category": "window-terminal"},
        EditOperation.REMOVE_INSTANCES: {"indices": [1, 2]},
        EditOperation.EXTEND_REGION: {"distance_m": 1.0},
    }
    edits = []
    seed = base["derived_seeds"]["edits"]
    for index, (operation, params) in enumerate(parameters.items()):
        edits.append(
            {
                "edit_id": f"edit-operation-{index}",
                "operation": operation.value,
                "target_role": "facade-grid",
                "parameters": params,
                "applicability": "required",
                "camera_reuse": "reuse_all",
                "changed_region_path": f"edits/operation-{index}/changed.png",
                "unchanged_region_path": f"edits/operation-{index}/unchanged.png",
                "edit_seed": seed,
            }
        )
    base["edits"] = edits
    valid = tmp_path / "all-edits.json"
    _write_json(valid, base)
    loaded = load_scene_configuration(valid)
    assert {edit.operation for edit in loaded.edits} == set(EditOperation)

    invalid = deepcopy(base)
    invalid["edits"][0]["parameters"]["unexpected"] = True
    invalid_path = tmp_path / "invalid-edit.json"
    _write_json(invalid_path, invalid)
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        load_scene_configuration(invalid_path)

    inapplicable = deepcopy(base)
    inapplicable["edits"][0]["target_role"] = "corridor-bays"
    inapplicable_path = tmp_path / "inapplicable-edit.json"
    _write_json(inapplicable_path, inapplicable)
    with pytest.raises(SchemaValidationError, match="must name a declared"):
        load_scene_configuration(inapplicable_path)


def test_all_resource_profiles_and_extended_override(tmp_path: Path) -> None:
    base = _load_json(VALID[2])
    for profile in ("smoke", "quick", "standard", "extended"):
        document = deepcopy(base)
        document["expected_resource_profile"] = profile
        fixture = tmp_path / f"{profile}.json"
        _write_json(fixture, document)
        load_scene_configuration(fixture)

    extended = deepcopy(base)
    extended["expected_resource_profile"] = "extended"
    extended["extended_mode"] = True
    extended["render"]["width"] = 5000
    extended["camera"]["image_width"] = 5000
    fixture = tmp_path / "extended-override.json"
    _write_json(fixture, extended)
    load_scene_configuration(fixture)


def test_all_scientific_sections_change_identity(tmp_path: Path) -> None:
    base_json = _load_json(VALID[0])
    base_path = tmp_path / "base-scientific.json"
    _write_json(base_path, base_json)
    baseline = load_scene_configuration(base_path).scientific_digest
    changes: list[tuple[list[str | int], object]] = [
        (["family_parameters", "wall_depth_m"], 0.45),
        (["camera", "minimum_structure_coverage"], 0.71),
        (["appearance", "roughness"], 0.56),
        (["lighting", "key_energy"], 1201.0),
        (["occlusion", "minimum_visible_fraction"], 0.61),
        (["outputs", "generate_preview"], False),
        (["edits", 0, "parameters", "spacing_m"], 2.6),
    ]
    for index, (path, value) in enumerate(changes):
        document = deepcopy(base_json)
        _assign(document, path, value)
        fixture = tmp_path / f"scientific-{index}.json"
        _write_json(fixture, document)
        assert load_scene_configuration(fixture).scientific_digest != baseline

    seed_document = deepcopy(base_json)
    seed_document["master_seed"] = 102
    seed_document["scene_id"] = "syn-facade-grid-development-s102"
    seeds = derive_seeds(102)
    seed_document["derived_seeds"] = seeds
    seed_document["camera"]["seed"] = seeds["cameras"]
    seed_document["appearance"]["seed"] = seeds["materials"]
    seed_document["lighting"]["seed"] = seeds["lighting"]
    seed_document["occlusion"]["seed"] = seeds["occluders"]
    seed_document["edits"][0]["edit_seed"] = seeds["edits"]
    seed_fixture = tmp_path / "scientific-seed.json"
    _write_json(seed_fixture, seed_document)
    assert load_scene_configuration(seed_fixture).scientific_digest != baseline


def test_definition_schemas_are_not_changed_by_scene_loading() -> None:
    schemas = tuple((ROOT / "definitions" / "schemas").glob("*.json"))
    before = {path: path.read_bytes() for path in schemas}

    for path in VALID:
        load_scene_configuration(path)

    assert {path: path.read_bytes() for path in schemas} == before
