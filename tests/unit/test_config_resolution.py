"""Deterministic configuration resolution, digest, and run-ID tests."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaussweave.config import SchemaValidationError
from gaussweave.config.resolution import (
    RUN_DIGEST_PREFIX_LENGTH,
    ResolutionError,
    apply_safe_defaults,
    canonical_json_bytes,
    content_digest,
    generate_run_id,
    merge_layers,
    resolve_layers,
    scientific_digest,
    scientific_projection,
    write_resolution_outputs,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[2]
VALIDATION_FIXTURES = ROOT / "tests" / "fixtures" / "validation"
RESOLUTION_FIXTURES = ROOT / "tests" / "fixtures" / "resolution"
BASE = VALIDATION_FIXTURES / "valid_experiment.json"
OPERATIONAL = RESOLUTION_FIXTURES / "operational-overlay.json"
SCIENTIFIC = RESOLUTION_FIXTURES / "scientific-overlay.json"
FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def load(path: Path) -> dict[str, object]:
    """Load one test fixture."""

    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_recursive_merge_semantics_and_source_immutability() -> None:
    earlier = {
        "object": {"kept": 1, "changed": 1},
        "array": [1, 2],
        "scalar": "old",
        "nullable": "value",
    }
    later = {
        "object": {"changed": 2, "added": 3},
        "array": [3],
        "scalar": "new",
        "nullable": None,
    }
    original_earlier = deepcopy(earlier)
    original_later = deepcopy(later)

    merged = merge_layers([earlier, later])

    assert merged == {
        "object": {"kept": 1, "changed": 2, "added": 3},
        "array": [3],
        "scalar": "new",
        "nullable": None,
    }
    assert earlier == original_earlier
    assert later == original_later
    assert merge_layers([later, earlier]) != merged


def test_resolution_validates_and_preserves_source_files() -> None:
    before = BASE.read_bytes()
    resolved = resolve_layers(
        [BASE, OPERATIONAL],
        repository_root=ROOT,
        timestamp=FIXED_TIME,
    )

    assert BASE.read_bytes() == before
    assert resolved.experiment_id == "exp-hardware-fixture-v1"
    assert resolved.resource_profile == "smoke"
    assert resolved.seeds == (0,)
    assert resolved.provenance.validation_status == "valid"
    assert [layer.order for layer in resolved.provenance.layers] == [1, 2]
    assert resolved.provenance.layers[0].path == (
        "tests/fixtures/validation/valid_experiment.json"
    )
    assert all(
        layer.digest.startswith("sha256:") for layer in resolved.provenance.layers
    )
    assert "/home/" not in json.dumps(resolved.provenance.to_dict())


def test_incomplete_composition_fails_structured_validation() -> None:
    with pytest.raises(SchemaValidationError) as captured:
        resolve_layers([RESOLUTION_FIXTURES / "incomplete-layer.json"])

    assert captured.value.issues
    assert all(issue.stage.value == "schema" for issue in captured.value.issues)


def test_explicit_safe_defaults_only() -> None:
    source = load(BASE)
    source_method = source["method"]
    assert isinstance(source_method, dict)
    source_method.pop("privileged_information", None)

    defaulted = apply_safe_defaults(source)

    assert defaulted["description"] is None
    assert defaulted["candidate_claims"] == []
    assert defaulted["baseline"] is None
    assert defaulted["matrix"] == {}
    assert defaulted["notes"] is None
    method = defaulted["method"]
    assert isinstance(method, dict)
    assert method["privileged_information"] == []
    assert defaulted["dataset"] == source["dataset"]
    assert defaulted["execution"] == source["execution"]
    assert defaulted["rendering"] == source["rendering"]


def test_canonical_serialization_is_key_stable_and_process_stable() -> None:
    left = {"z": [3, 2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "z": [3, 2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    code = (
        "from gaussweave.config.resolution import canonical_json_bytes;"
        "print(canonical_json_bytes("
        "{'z':[3,2,1],'a':{'y':2,'x':1}}"
        ").hex())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == canonical_json_bytes(left).hex()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value: float) -> None:
    with pytest.raises(ResolutionError, match="non-finite"):
        canonical_json_bytes({"value": value})


def test_full_digest_is_complete_and_canonical() -> None:
    document = load(BASE)
    reordered = dict(reversed(list(document.items())))

    assert content_digest(document) == content_digest(reordered)
    changed = deepcopy(document)
    changed["description"] = "changed"
    assert content_digest(document) != content_digest(changed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("outputs", "run_root"), "runs/other"),
        (("execution", "attempt"), 99),
        (("description",), "other description"),
        (("notes",), "other notes"),
        (("freeze", "frozen_at"), "2027-01-01T00:00:00Z"),
        (("freeze", "configuration_digest"), "sha256:" + "1" * 64),
    ],
)
def test_scientific_digest_ignores_operational_fields(
    path: tuple[str, ...], value: object
) -> None:
    document = apply_safe_defaults(load(BASE))
    changed = deepcopy(document)
    target = changed
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value

    assert scientific_digest(document) == scientific_digest(changed)
    assert content_digest(document) != content_digest(changed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dataset", "scene_selection", "scene_ids"), ["syn-other"]),
        (("execution", "seeds"), [1]),
        (("execution", "resource_profile"), "quick"),
        (("method", "operating_point"), {"gaussian_cap": 256}),
        (("rendering", "width"), 2),
        (("metrics", "metric_ids"), ["ssim"]),
        (("stopping", "rules"), [{"kind": "iteration_count", "value": 2}]),
    ],
)
def test_scientific_digest_changes_for_scientific_fields(
    path: tuple[str, ...], value: object
) -> None:
    document = apply_safe_defaults(load(BASE))
    changed = deepcopy(document)
    target = changed
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value

    assert scientific_digest(document) != scientific_digest(changed)


def test_projection_is_versioned_and_excludes_output_policy() -> None:
    projection = scientific_projection(apply_safe_defaults(load(BASE)))

    assert projection["projection_version"] == "1.0"
    experiment = projection["experiment"]
    assert isinstance(experiment, dict)
    assert "outputs" not in experiment
    assert "freeze" not in experiment
    execution = experiment["execution"]
    assert isinstance(execution, dict)
    assert "attempt" not in execution


def test_run_id_is_deterministic_and_attempt_aware() -> None:
    digest = scientific_digest(apply_safe_defaults(load(BASE)))
    arguments = {
        "experiment_id": "exp-hardware-fixture-v1",
        "scene_id": "syn-fixture",
        "seed": 0,
        "scientific_configuration_digest": digest,
    }

    first = generate_run_id(**arguments, attempt=1)
    assert first == generate_run_id(**arguments, attempt=1)
    assert first != generate_run_id(**arguments, attempt=2)
    assert len(digest.removeprefix("sha256:")[:RUN_DIGEST_PREFIX_LENGTH]) == 12
    assert first.startswith("run-exp-hardware-fixture-v1-syn-fixture-s0-")


@pytest.mark.parametrize(
    "changes",
    [
        {"experiment_id": "bad"},
        {"scene_id": "bad"},
        {"seed": -1},
        {"scientific_configuration_digest": "sha256:bad"},
        {"attempt": 0},
    ],
)
def test_invalid_run_identity_inputs_fail(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "experiment_id": "exp-hardware-fixture-v1",
        "scene_id": "syn-fixture",
        "seed": 0,
        "scientific_configuration_digest": "sha256:" + "0" * 64,
        "attempt": 1,
    }
    arguments.update(changes)
    with pytest.raises(ResolutionError):
        generate_run_id(**arguments)  # type: ignore[arg-type]


def test_atomic_outputs_have_newlines_and_refuse_overwrite(tmp_path: Path) -> None:
    resolved = resolve_layers([BASE], repository_root=ROOT, timestamp=FIXED_TIME)
    output = tmp_path / "resolved.json"

    paths = write_resolution_outputs(resolved, output)

    for path in paths.values():
        assert Path(path).read_bytes().endswith(b"\n")
    canonical = output.with_suffix(".canonical.json").read_bytes()
    assert canonical == canonical_json_bytes(dict(resolved.document)) + b"\n"
    with pytest.raises(ResolutionError, match="refusing to overwrite"):
        write_resolution_outputs(resolved, output)
