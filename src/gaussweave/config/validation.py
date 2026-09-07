"""CPU-only JSON Schema and lightweight semantic validation."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from gaussweave.config.errors import (
    DocumentLoadError,
    SchemaValidationError,
    ValidationIssue,
    ValidationStage,
    display_source,
    summarize_value,
)
from gaussweave.config.schema import SchemaKind, SchemaRegistry

MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
type SemanticValidator = Callable[
    [Mapping[str, JsonValue], SchemaKind, str | None],
    tuple[ValidationIssue, ...],
]


@dataclass(frozen=True)
class ValidatedDocument:
    """A schema-checked document with retained source representation."""

    kind: SchemaKind
    data: Mapping[str, JsonValue]
    source: str | None
    schema_id: str
    schema_version: str


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Initial typed wrapper for a validated experiment configuration."""

    data: Mapping[str, JsonValue]
    source: str | None
    schema_id: str
    schema_version: str

    @property
    def experiment_id(self) -> str:
        """Return the required validated experiment identifier."""

        value = self.data["experiment_id"]
        assert isinstance(value, str)
        return value


def load_json_document(
    path: Path, *, kind: SchemaKind, max_bytes: int = MAX_DOCUMENT_BYTES
) -> dict[str, JsonValue]:
    """Load one bounded UTF-8 regular JSON object."""

    source = display_source(path)
    try:
        if not path.exists():
            raise OSError("file does not exist")
        if not path.is_file():
            raise OSError("path is not a regular file")
        size = path.stat().st_size
        if size > max_bytes:
            raise OSError(f"file exceeds {max_bytes} byte limit")
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
    except json.JSONDecodeError as error:
        issue = ValidationIssue(
            document_kind=kind.value,
            stage=ValidationStage.LOADING,
            source=source,
            message=(
                f"malformed JSON at line {error.lineno}, column {error.colno}: "
                f"{error.msg}"
            ),
            instance_path=(error.lineno, error.colno),
        )
        raise DocumentLoadError(issue) from error
    except (OSError, UnicodeError) as error:
        issue = ValidationIssue(
            document_kind=kind.value,
            stage=ValidationStage.LOADING,
            source=source,
            message=str(error),
        )
        raise DocumentLoadError(issue) from error
    if not isinstance(value, dict):
        issue = ValidationIssue(
            document_kind=kind.value,
            stage=ValidationStage.LOADING,
            source=source,
            message="top-level JSON value must be an object",
            invalid_value=summarize_value(value),
        )
        raise DocumentLoadError(issue)
    return value


def validate_document(
    data: Mapping[str, JsonValue],
    kind: SchemaKind,
    *,
    registry: SchemaRegistry | None = None,
    source: str | None = None,
    semantic_validators: Sequence[SemanticValidator] = (),
) -> ValidatedDocument:
    """Validate a mapping against an accepted schema and semantic extensions."""

    loaded = (registry or SchemaRegistry()).load(kind)
    validator = Draft202012Validator(
        loaded.document,
        format_checker=FormatChecker(),
    )
    errors = sorted(
        validator.iter_errors(dict(data)),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    issues = tuple(
        ValidationIssue(
            document_kind=kind.value,
            stage=ValidationStage.SCHEMA,
            source=source,
            instance_path=tuple(error.absolute_path),
            schema_path=tuple(error.absolute_schema_path),
            validator=str(error.validator) if error.validator else None,
            message=error.message,
            invalid_value=summarize_value(error.instance),
        )
        for error in errors
    )
    semantic_issues: list[ValidationIssue] = []
    builtin_checks: tuple[SemanticValidator, ...] = (
        _validate_safe_paths,
        _validate_grammar_node_keys,
    )
    for check in builtin_checks:
        semantic_issues.extend(check(data, kind, source))
    for extension in semantic_validators:
        semantic_issues.extend(extension(data, kind, source))
    all_issues = (*issues, *semantic_issues)
    if all_issues:
        raise SchemaValidationError(all_issues)

    return ValidatedDocument(
        kind=kind,
        data=_freeze_mapping(data),
        source=source,
        schema_id=loaded.descriptor.schema_id,
        schema_version=loaded.descriptor.schema_version,
    )


def validate_json_file(
    path: Path,
    kind: SchemaKind,
    *,
    registry: SchemaRegistry | None = None,
) -> ValidatedDocument:
    """Load and validate one schema-governed JSON file."""

    return load_and_validate_json(path, kind, registry=registry)


def load_and_validate_json(
    path: Path,
    kind: SchemaKind,
    *,
    registry: SchemaRegistry | None = None,
) -> ValidatedDocument:
    """Load bounded JSON and return a typed validated document."""

    data = load_json_document(path, kind=kind)
    return validate_document(
        data,
        kind,
        registry=registry,
        source=display_source(path),
    )


def validate_scene_manifest(
    data: Mapping[str, JsonValue], **kwargs: Any
) -> ValidatedDocument:
    """Validate a scene manifest."""

    return validate_document(data, SchemaKind.SCENE_MANIFEST, **kwargs)


def validate_grammar(data: Mapping[str, JsonValue], **kwargs: Any) -> ValidatedDocument:
    """Validate a grammar document."""

    return validate_document(data, SchemaKind.GRAMMAR, **kwargs)


def validate_experiment(
    data: Mapping[str, JsonValue], **kwargs: Any
) -> ValidatedDocument:
    """Validate an experiment configuration."""

    return validate_document(data, SchemaKind.EXPERIMENT, **kwargs)


def validate_result(data: Mapping[str, JsonValue], **kwargs: Any) -> ValidatedDocument:
    """Validate a result record."""

    return validate_document(data, SchemaKind.RESULT, **kwargs)


def load_experiment_configuration(
    path: Path, *, registry: SchemaRegistry | None = None
) -> ExperimentConfiguration:
    """Load a JSON experiment into its initial typed configuration wrapper."""

    validated = load_and_validate_json(
        path,
        SchemaKind.EXPERIMENT,
        registry=registry,
    )
    return ExperimentConfiguration(
        data=validated.data,
        source=validated.source,
        schema_id=validated.schema_id,
        schema_version=validated.schema_version,
    )


def _validate_safe_paths(
    data: Mapping[str, JsonValue], kind: SchemaKind, source: str | None
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    for path, key, value in _walk_strings(data):
        if not _is_path_key(key):
            continue
        pure = PurePosixPath(value)
        is_windows_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", value))
        if (
            pure.is_absolute()
            or is_windows_absolute
            or "\\" in value
            or ".." in pure.parts
        ):
            issues.append(
                ValidationIssue(
                    document_kind=kind.value,
                    stage=ValidationStage.SEMANTIC,
                    source=source,
                    instance_path=path,
                    validator="safeRelativePath",
                    message="path must be relative POSIX syntax without '..'",
                    invalid_value=summarize_value(value),
                )
            )
    return tuple(issues)


def _validate_grammar_node_keys(
    data: Mapping[str, JsonValue], kind: SchemaKind, source: str | None
) -> tuple[ValidationIssue, ...]:
    if kind is not SchemaKind.GRAMMAR:
        return ()
    nodes = data.get("nodes")
    if not isinstance(nodes, dict):
        return ()
    issues: list[ValidationIssue] = []
    for key, node in nodes.items():
        if isinstance(node, dict) and node.get("node_id") != key:
            issues.append(
                ValidationIssue(
                    document_kind=kind.value,
                    stage=ValidationStage.SEMANTIC,
                    source=source,
                    instance_path=("nodes", key, "node_id"),
                    validator="nodeMapKey",
                    message=f"node map key {key!r} must equal node_id",
                    invalid_value=summarize_value(node.get("node_id")),
                )
            )
    return tuple(issues)


def _walk_strings(
    value: Mapping[str, JsonValue],
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], str, str]]:
    found: list[tuple[tuple[str | int, ...], str, str]] = []
    for key, item in value.items():
        item_path = (*path, key)
        if isinstance(item, str):
            found.append((item_path, key, item))
        elif isinstance(item, dict):
            found.extend(_walk_strings(item, item_path))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if isinstance(child, dict):
                    found.extend(_walk_strings(child, (*item_path, index)))
    return found


def _is_path_key(key: str) -> bool:
    return (
        key == "path"
        or key.endswith("_path")
        or key.endswith("_ref")
        or key.endswith("_root")
    )


def _freeze_mapping(
    data: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return MappingProxyType(dict(data))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=[kind.value for kind in SchemaKind],
    )
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    parser.add_argument("path", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate one document for the temporary pre-CLI entry point."""

    options = _build_parser().parse_args(arguments)
    kind = SchemaKind(options.kind)
    try:
        validated = load_and_validate_json(options.path, kind)
    except DocumentLoadError as error:
        payload = {"valid": False, "errors": [error.issue.to_dict()]}
        print(json.dumps(payload, indent=2) if options.json else str(error))
        return 2
    except SchemaValidationError as error:
        print(json.dumps(error.to_dict(), indent=2) if options.json else str(error))
        return 1
    payload = {
        "valid": True,
        "kind": kind.value,
        "schema_id": validated.schema_id,
        "schema_version": validated.schema_version,
        "source": validated.source,
    }
    print(
        json.dumps(payload, indent=2)
        if options.json
        else f"valid {kind.value} document: {validated.source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
