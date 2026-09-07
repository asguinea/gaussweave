"""CPU-only schema validation and typed configuration loading."""

from gaussweave.config.errors import (
    DocumentLoadError,
    SchemaDiscoveryError,
    SchemaValidationError,
    ValidationIssue,
    ValidationStage,
)
from gaussweave.config.schema import SchemaKind, SchemaRegistry

__all__ = [
    "DocumentLoadError",
    "SchemaDiscoveryError",
    "SchemaKind",
    "SchemaRegistry",
    "SchemaValidationError",
    "ValidationIssue",
    "ValidationStage",
]
