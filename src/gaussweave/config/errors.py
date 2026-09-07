"""Structured errors for document loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ValidationStage(StrEnum):
    """Stage at which validation failed."""

    DISCOVERY = "discovery"
    LOADING = "loading"
    SCHEMA = "schema"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class ValidationIssue:
    """One bounded, machine-readable validation failure."""

    document_kind: str
    stage: ValidationStage
    message: str
    source: str | None = None
    instance_path: tuple[str | int, ...] = ()
    schema_path: tuple[str | int, ...] = ()
    validator: str | None = None
    invalid_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        value = asdict(self)
        value["stage"] = self.stage.value
        value["instance_path"] = list(self.instance_path)
        value["schema_path"] = list(self.schema_path)
        return value

    def format(self) -> str:
        """Return a concise human-readable representation."""

        location = _format_path(self.instance_path)
        prefix = f"{self.document_kind} {self.stage.value}"
        return f"{prefix} at {location}: {self.message}"


class ConfigurationError(Exception):
    """Base class for project-owned configuration failures."""


class SchemaDiscoveryError(ConfigurationError):
    """The authoritative definitions schema directory cannot be located."""


class DocumentLoadError(ConfigurationError):
    """A document could not be safely loaded."""

    def __init__(self, issue: ValidationIssue) -> None:
        super().__init__(issue.format())
        self.issue = issue


class SchemaValidationError(ConfigurationError):
    """A document failed schema or lightweight semantic validation."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        if not issues:
            raise ValueError("at least one validation issue is required")
        super().__init__("\n".join(issue.format() for issue in issues))
        self.issues = issues

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible error envelope."""

        return {
            "valid": False,
            "errors": [issue.to_dict() for issue in self.issues],
        }


def display_source(path: Path) -> str:
    """Return a source name without exposing a personal absolute path."""

    return path.name


def summarize_value(value: object, *, limit: int = 160) -> str:
    """Return a bounded repr suitable for public validation output."""

    text = repr(value)
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _format_path(path: tuple[str | int, ...]) -> str:
    if not path:
        return "$"
    result = "$"
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result
