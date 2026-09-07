"""Discovery and loading of the accepted packaged JSON Schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from gaussweave.config.errors import SchemaDiscoveryError


class SchemaKind(StrEnum):
    """Stable identifiers for schema-governed project documents."""

    SCENE_MANIFEST = "scene"
    GRAMMAR = "grammar"
    EXPERIMENT = "experiment"
    RESULT = "result"


@dataclass(frozen=True)
class SchemaDescriptor:
    """Static and loaded identity for one accepted schema."""

    kind: SchemaKind
    filename: str
    title: str
    schema_id: str
    schema_version: str
    draft: str


@dataclass(frozen=True)
class LoadedSchema:
    """A checked accepted schema and its descriptor."""

    descriptor: SchemaDescriptor
    path: Path
    document: dict[str, Any]


_FILENAMES = {
    SchemaKind.SCENE_MANIFEST: "scene_manifest.schema.json",
    SchemaKind.GRAMMAR: "grammar.schema.json",
    SchemaKind.EXPERIMENT: "experiment.schema.json",
    SchemaKind.RESULT: "result.schema.json",
}


class SchemaRegistry:
    """Locate, parse, and self-check authoritative local schemas."""

    def __init__(
        self,
        *,
        definitions_root: Path | None = None,
        repository_root: Path | None = None,
    ) -> None:
        self.definitions_root = discover_definitions_root(
            definitions_root=definitions_root,
            repository_root=repository_root,
        )

    def load(self, kind: SchemaKind) -> LoadedSchema:
        """Load and self-check one schema without network resolution."""

        path = self.definitions_root / "schemas" / _FILENAMES[kind]
        if not path.is_file():
            raise SchemaDiscoveryError(
                f"accepted {kind.value} schema is missing: {path.name}"
            )
        try:
            raw = path.read_text(encoding="utf-8")
            document = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SchemaDiscoveryError(
                f"accepted {kind.value} schema is unreadable or malformed: {error}"
            ) from error
        if not isinstance(document, dict):
            raise SchemaDiscoveryError(
                f"accepted {kind.value} schema must be a JSON object"
            )
        try:
            Draft202012Validator.check_schema(document)
        except SchemaError as error:
            raise SchemaDiscoveryError(
                f"accepted {kind.value} schema failed self-check: {error.message}"
            ) from error
        descriptor = SchemaDescriptor(
            kind=kind,
            filename=path.name,
            title=_required_text(document, "title", kind),
            schema_id=_required_text(document, "$id", kind),
            schema_version=_schema_version(document),
            draft=_required_text(document, "$schema", kind),
        )
        return LoadedSchema(descriptor, path, document)

    def load_all(self) -> tuple[LoadedSchema, ...]:
        """Load all accepted schemas in stable enumeration order."""

        return tuple(self.load(kind) for kind in SchemaKind)


def discover_definitions_root(
    *,
    definitions_root: Path | None = None,
    repository_root: Path | None = None,
) -> Path:
    """Find a root containing the accepted schema directory."""

    if definitions_root is not None:
        candidate = definitions_root.expanduser().resolve()
        if _is_definitions_root(candidate):
            return candidate
        raise SchemaDiscoveryError(
            f"explicit definitions root does not contain schemas: {candidate.name}"
        )
    if repository_root is not None:
        candidate = repository_root.expanduser().resolve() / "src" / "gaussweave"
        if _is_definitions_root(candidate):
            return candidate
        raise SchemaDiscoveryError(
            "explicit repository root does not contain packaged schemas"
        )

    packaged_root = Path(__file__).resolve().parents[1]
    if _is_definitions_root(packaged_root):
        return packaged_root

    for start in (Path.cwd(), Path(__file__).resolve()):
        for parent in (start, *start.parents):
            candidate = parent / "src" / "gaussweave"
            if _is_definitions_root(candidate):
                return candidate
    raise SchemaDiscoveryError(
        "unable to discover packaged schemas; pass definitions_root or repository_root"
    )


def _is_definitions_root(path: Path) -> bool:
    return path.is_dir() and (path / "schemas").is_dir()


def _required_text(document: dict[str, Any], key: str, kind: SchemaKind) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise SchemaDiscoveryError(
            f"accepted {kind.value} schema has invalid {key} metadata"
        )
    return value


def _schema_version(document: dict[str, Any]) -> str:
    properties = document.get("properties")
    if not isinstance(properties, dict):
        return "unknown"
    version = properties.get("schema_version")
    if not isinstance(version, dict):
        return "unknown"
    value = version.get("const")
    return value if isinstance(value, str) else "unknown"
