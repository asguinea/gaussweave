"""Tests for accepted-schema discovery, loading, and validation."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from gaussweave.config import (
    DocumentLoadError,
    SchemaDiscoveryError,
    SchemaKind,
    SchemaRegistry,
    SchemaValidationError,
)
from gaussweave.config.validation import (
    load_and_validate_json,
    load_experiment_configuration,
    load_json_document,
    validate_document,
)

pytestmark = [pytest.mark.unit, pytest.mark.schema]

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "validation"
VALID_FIXTURES = {
    SchemaKind.SCENE_MANIFEST: "valid_scene_manifest.json",
    SchemaKind.GRAMMAR: "valid_grammar.json",
    SchemaKind.EXPERIMENT: "valid_experiment.json",
    SchemaKind.RESULT: "valid_result.json",
}


def fixture_data(name: str) -> dict[str, object]:
    """Read one deterministic JSON fixture."""

    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_registry_discovers_and_self_checks_all_accepted_schemas() -> None:
    loaded = SchemaRegistry(repository_root=ROOT).load_all()

    assert [item.descriptor.kind for item in loaded] == list(SchemaKind)
    assert all(
        item.descriptor.draft == "https://json-schema.org/draft/2020-12/schema"
        for item in loaded
    )
    assert all(item.descriptor.schema_version == "1.0.0" for item in loaded)
    assert all(item.descriptor.title and item.descriptor.schema_id for item in loaded)


def test_explicit_invalid_definitions_root_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(SchemaDiscoveryError, match="does not contain schemas"):
        SchemaRegistry(definitions_root=tmp_path)


@pytest.mark.parametrize(("kind", "filename"), VALID_FIXTURES.items())
def test_valid_fixture_passes(kind: SchemaKind, filename: str) -> None:
    validated = load_and_validate_json(
        FIXTURES / filename,
        kind,
        registry=SchemaRegistry(repository_root=ROOT),
    )

    assert validated.kind is kind
    assert validated.schema_version == "1.0.0"


def test_missing_required_fields_collect_structured_errors() -> None:
    data = fixture_data("valid_experiment.json")
    del data["dataset"]
    del data["method"]

    with pytest.raises(SchemaValidationError) as captured:
        validate_document(data, SchemaKind.EXPERIMENT)

    assert len(captured.value.issues) == 2
    assert all(issue.validator == "required" for issue in captured.value.issues)
    assert all(issue.schema_path for issue in captured.value.issues)


def test_unknown_field_is_rejected() -> None:
    data = fixture_data("valid_experiment.json")
    data["unexpected"] = True

    with pytest.raises(SchemaValidationError) as captured:
        validate_document(data, SchemaKind.EXPERIMENT)

    assert any(
        issue.validator == "additionalProperties" for issue in captured.value.issues
    )


@pytest.mark.parametrize(
    ("field", "value", "validator"),
    [
        ("experiment_id", "bad-id", "pattern"),
        ("family", "unsupported-family", "enum"),
    ],
)
def test_invalid_identifier_and_enum_are_readable(
    field: str, value: str, validator: str
) -> None:
    data = fixture_data("valid_experiment.json")
    data[field] = value

    with pytest.raises(SchemaValidationError) as captured:
        validate_document(data, SchemaKind.EXPERIMENT)

    issue = next(item for item in captured.value.issues if item.validator == validator)
    assert issue.instance_path == (field,)
    assert issue.invalid_value == repr(value)


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute/path", r"C:\escape"])
def test_unsafe_paths_are_rejected(unsafe: str) -> None:
    data = fixture_data("valid_experiment.json")
    outputs = data["outputs"]
    assert isinstance(outputs, dict)
    outputs["run_root"] = unsafe

    with pytest.raises(SchemaValidationError) as captured:
        validate_document(data, SchemaKind.EXPERIMENT)

    assert any(
        issue.validator in {"pattern", "safeRelativePath"}
        for issue in captured.value.issues
    )


def test_malformed_json_has_bounded_source_and_location(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(DocumentLoadError) as captured:
        load_json_document(path, kind=SchemaKind.EXPERIMENT)

    issue = captured.value.issue
    assert issue.source == "broken.json"
    assert issue.instance_path == (1, 19)
    assert "malformed JSON" in issue.message


def test_non_regular_and_oversized_documents_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="not a regular file"):
        load_json_document(tmp_path, kind=SchemaKind.RESULT)
    path = tmp_path / "large.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="exceeds 1 byte limit"):
        load_json_document(path, kind=SchemaKind.RESULT, max_bytes=1)


def test_grammar_node_map_key_must_equal_node_id() -> None:
    data = fixture_data("valid_grammar.json")
    nodes = data["nodes"]
    assert isinstance(nodes, dict)
    nodes["wrong-key"] = nodes.pop("root-main")
    data["root_node_id"] = "wrong-key"

    with pytest.raises(SchemaValidationError) as captured:
        validate_document(data, SchemaKind.GRAMMAR)

    issue = next(
        item for item in captured.value.issues if item.validator == "nodeMapKey"
    )
    assert issue.instance_path == ("nodes", "wrong-key", "node_id")
    assert issue.stage.value == "semantic"


def test_typed_experiment_loader_retains_null_and_source() -> None:
    configuration = load_experiment_configuration(
        FIXTURES / "valid_experiment.json",
        registry=SchemaRegistry(repository_root=ROOT),
    )

    assert configuration.experiment_id == "exp-hardware-fixture-v1"
    assert configuration.source == "valid_experiment.json"
    freeze = configuration.data["freeze"]
    assert isinstance(freeze, dict)
    assert freeze["frozen_at"] is None


def test_error_value_summary_is_bounded() -> None:
    data = fixture_data("valid_experiment.json")
    data["family"] = "x" * 1000

    with pytest.raises(SchemaValidationError) as captured:
        validate_document(data, SchemaKind.EXPERIMENT)

    assert max(len(issue.invalid_value or "") for issue in captured.value.issues) <= 160


def test_validation_import_is_cpu_only() -> None:
    code = (
        "import sys; import gaussweave.config; "
        "blocked={'torch','gsplat','bpy','pycolmap'}; "
        "assert blocked.isdisjoint(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_custom_semantic_validator_extension_is_called() -> None:
    data = fixture_data("valid_experiment.json")
    called: list[SchemaKind] = []

    def custom(
        document: object, kind: SchemaKind, source: str | None
    ) -> tuple[object, ...]:
        del document, source
        called.append(kind)
        return ()

    validate_document(
        deepcopy(data),
        SchemaKind.EXPERIMENT,
        semantic_validators=(custom,),  # type: ignore[arg-type]
    )
    assert called == [SchemaKind.EXPERIMENT]
