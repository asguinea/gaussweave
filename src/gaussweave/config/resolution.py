"""Deterministic experiment configuration resolution and identity primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from gaussweave.config.errors import (
    ConfigurationError,
    DocumentLoadError,
    SchemaValidationError,
)
from gaussweave.config.schema import SchemaKind, SchemaRegistry
from gaussweave.config.validation import (
    JsonValue,
    load_json_document,
    validate_document,
)

RESOLVER_VERSION = "1.0"
SCIENTIFIC_PROJECTION_VERSION = "1.0"
RUN_DIGEST_PREFIX_LENGTH = 12
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPERIMENT_ID_PATTERN = re.compile(r"^exp-[a-z0-9][a-z0-9._-]*-v[0-9]+$")
SCENE_ID_PATTERN = re.compile(r"^(?:syn|real)-[a-z0-9][a-z0-9-]*$")


class ResolutionError(ConfigurationError):
    """Configuration composition, identity, or output failed."""


@dataclass(frozen=True)
class LayerProvenance:
    """Portable identity of one ordered source layer."""

    order: int
    path: str
    digest: str


@dataclass(frozen=True)
class ResolutionProvenance:
    """Portable provenance for one configuration resolution."""

    layers: tuple[LayerProvenance, ...]
    composed_at: str
    definitions_version: str
    resolver_version: str
    warnings: tuple[str, ...]
    validation_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible provenance record."""

        value = asdict(self)
        value["layers"] = [asdict(layer) for layer in self.layers]
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class ResolvedConfiguration:
    """Typed envelope around one schema-valid resolved experiment."""

    document: Mapping[str, JsonValue]
    provenance: ResolutionProvenance
    scientific_digest: str
    full_digest: str
    definitions_version: str
    experiment_id: str
    mode: str
    research_questions: tuple[str, ...]
    resource_profile: str
    seeds: tuple[int, ...]
    scientific_projection_version: str = SCIENTIFIC_PROJECTION_VERSION

    def to_summary(self) -> dict[str, JsonValue]:
        """Return a concise machine-readable resolution summary."""

        return {
            "definitions_version": self.definitions_version,
            "experiment_id": self.experiment_id,
            "full_digest": self.full_digest,
            "mode": self.mode,
            "resource_profile": self.resource_profile,
            "scientific_digest": self.scientific_digest,
            "scientific_projection_version": self.scientific_projection_version,
            "seeds": list(self.seeds),
        }


def merge_layers(layers: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """Recursively merge ordered layers without mutating source mappings."""

    result: dict[str, JsonValue] = {}
    for layer in layers:
        result = _merge_objects(result, layer)
    return result


def apply_safe_defaults(document: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Insert only explicitly approved non-scientific defaults."""

    resolved = deepcopy(dict(document))
    defaults: dict[str, JsonValue] = {
        "description": None,
        "candidate_claims": [],
        "baseline": None,
        "matrix": {},
        "notes": None,
    }
    for key, value in defaults.items():
        resolved.setdefault(key, value)
    method = resolved.get("method")
    if isinstance(method, dict):
        method.setdefault("privileged_information", [])
    return resolved


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize canonical UTF-8 JSON bytes without a terminal newline."""

    _reject_non_finite(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ResolutionError(f"value is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def pretty_json_bytes(value: JsonValue) -> bytes:
    """Serialize readable deterministic JSON with one terminal newline."""

    _reject_non_finite(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ResolutionError(f"value is not JSON serializable: {error}") from error
    return f"{text}\n".encode()


def content_digest(value: JsonValue) -> str:
    """Return a SHA-256 digest of canonical JSON bytes."""

    return _digest_bytes(canonical_json_bytes(value))


def scientific_projection(
    document: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Project an experiment onto versioned scientific identity fields.

    Output paths and saving flags affect artifact persistence, not model or
    evaluation behavior, so the complete ``outputs`` object is excluded.
    Freeze bookkeeping identifies an execution state rather than parameters.
    """

    projected = deepcopy(dict(document))
    for key in ("description", "notes", "outputs", "freeze"):
        projected.pop(key, None)
    execution = projected.get("execution")
    if isinstance(execution, dict):
        execution.pop("attempt", None)
    return {
        "projection_version": SCIENTIFIC_PROJECTION_VERSION,
        "experiment": projected,
    }


def scientific_digest(document: Mapping[str, JsonValue]) -> str:
    """Return the versioned scientific configuration digest."""

    return content_digest(scientific_projection(document))


def generate_run_id(
    *,
    experiment_id: str,
    scene_id: str,
    seed: int,
    scientific_configuration_digest: str,
    attempt: int,
) -> str:
    """Generate a safe deterministic run ID using a 12-hex digest prefix."""

    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ResolutionError(f"invalid experiment ID: {experiment_id!r}")
    if not SCENE_ID_PATTERN.fullmatch(scene_id):
        raise ResolutionError(f"invalid scene ID: {scene_id!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ResolutionError("seed must be a nonnegative integer")
    if not DIGEST_PATTERN.fullmatch(scientific_configuration_digest):
        raise ResolutionError("scientific digest must be sha256:<64 lowercase hex>")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ResolutionError("attempt must be an integer beginning at 1")
    prefix = scientific_configuration_digest.removeprefix("sha256:")[
        :RUN_DIGEST_PREFIX_LENGTH
    ]
    return f"run-{experiment_id}-{scene_id}-s{seed}-{prefix}-a{attempt}"


def resolve_layers(
    layer_paths: Sequence[Path],
    *,
    repository_root: Path | None = None,
    registry: SchemaRegistry | None = None,
    timestamp: datetime | None = None,
) -> ResolvedConfiguration:
    """Load, compose, default, validate, and identify ordered JSON layers."""

    if not layer_paths:
        raise ResolutionError("at least one configuration layer is required")
    root = (repository_root or Path.cwd()).resolve()
    source_documents: list[dict[str, JsonValue]] = []
    layer_records: list[LayerProvenance] = []
    warnings: list[str] = []
    for order, path in enumerate(layer_paths, start=1):
        resolved_path = path.expanduser().resolve()
        source_documents.append(
            load_json_document(resolved_path, kind=SchemaKind.EXPERIMENT)
        )
        portable_path, warning = _portable_path(resolved_path, root)
        if warning:
            warnings.append(warning)
        try:
            file_digest = _digest_bytes(resolved_path.read_bytes())
        except OSError as error:
            raise ResolutionError(
                f"unable to checksum layer {resolved_path.name}: {error}"
            ) from error
        layer_records.append(LayerProvenance(order, portable_path, file_digest))

    composed = apply_safe_defaults(merge_layers(source_documents))
    validated = validate_document(
        composed,
        SchemaKind.EXPERIMENT,
        registry=registry,
        source="resolved configuration",
    )
    document = dict(validated.data)
    definitions_version = _required_string(document, "definitions_version")
    provenance = ResolutionProvenance(
        layers=tuple(layer_records),
        composed_at=(timestamp or datetime.now(UTC)).astimezone(UTC).isoformat(),
        definitions_version=definitions_version,
        resolver_version=RESOLVER_VERSION,
        warnings=tuple(warnings),
        validation_status="valid",
    )
    execution = _required_mapping(document, "execution")
    questions = _required_string_list(document, "research_questions")
    seeds = _required_integer_list(execution, "seeds")
    return ResolvedConfiguration(
        document=MappingProxyType(document),
        provenance=provenance,
        scientific_digest=scientific_digest(document),
        full_digest=content_digest(document),
        definitions_version=definitions_version,
        experiment_id=_required_string(document, "experiment_id"),
        mode=_required_string(document, "mode"),
        research_questions=questions,
        resource_profile=_required_string(execution, "resource_profile"),
        seeds=seeds,
    )


def write_resolution_outputs(
    resolved: ResolvedConfiguration,
    output: Path,
    *,
    provenance_output: Path | None = None,
    canonical_output: Path | None = None,
    digest_output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Atomically write resolved, canonical, provenance, and digest files."""

    output = output.resolve()
    canonical = (canonical_output or output.with_suffix(".canonical.json")).resolve()
    provenance = (provenance_output or output.with_suffix(".provenance.json")).resolve()
    digests = (digest_output or output.with_suffix(".digests.json")).resolve()
    targets = (output, canonical, provenance, digests)
    if len(set(targets)) != len(targets):
        raise ResolutionError("resolution output paths must be distinct")
    if not overwrite:
        existing = [path.name for path in targets if path.exists()]
        if existing:
            raise ResolutionError(
                f"refusing to overwrite existing output: {', '.join(existing)}"
            )

    summary = resolved.to_summary()
    _atomic_write(output, pretty_json_bytes(dict(resolved.document)))
    _atomic_write(canonical, canonical_json_bytes(dict(resolved.document)) + b"\n")
    _atomic_write(provenance, pretty_json_bytes(resolved.provenance.to_dict()))
    _atomic_write(digests, pretty_json_bytes(summary))
    return {
        "canonical_output": canonical.as_posix(),
        "digest_output": digests.as_posix(),
        "provenance_output": provenance.as_posix(),
        "resolved_output": output.as_posix(),
    }


def _merge_objects(
    earlier: Mapping[str, JsonValue],
    later: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    result = deepcopy(dict(earlier))
    for key, later_value in later.items():
        earlier_value = result.get(key)
        if isinstance(earlier_value, dict) and isinstance(later_value, dict):
            result[key] = _merge_objects(earlier_value, later_value)
        else:
            result[key] = deepcopy(later_value)
    return result


def _reject_non_finite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ResolutionError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _portable_path(path: Path, repository_root: Path) -> tuple[str, str | None]:
    try:
        return path.relative_to(repository_root).as_posix(), None
    except ValueError:
        return (
            path.name,
            f"layer {path.name} is outside repository root; basename recorded",
        )


def _required_mapping(
    document: Mapping[str, JsonValue], key: str
) -> Mapping[str, JsonValue]:
    value = document[key]
    if not isinstance(value, dict):
        raise ResolutionError(f"validated {key} must be an object")
    return value


def _required_string(document: Mapping[str, JsonValue], key: str) -> str:
    value = document[key]
    if not isinstance(value, str):
        raise ResolutionError(f"validated {key} must be a string")
    return value


def _required_string_list(
    document: Mapping[str, JsonValue], key: str
) -> tuple[str, ...]:
    value = document[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ResolutionError(f"validated {key} must be a string list")
    return tuple(item for item in value if isinstance(item, str))


def _required_integer_list(
    document: Mapping[str, JsonValue], key: str
) -> tuple[int, ...]:
    value = document[key]
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ResolutionError(f"validated {key} must be an integer list")
    return tuple(
        item for item in value if isinstance(item, int) and not isinstance(item, bool)
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    except OSError as error:
        raise ResolutionError(f"unable to write {path.name}: {error}") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--canonical-output", type=Path)
    parser.add_argument("--digest-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    parser.add_argument("--scene-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int, default=1)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Resolve ordered layers for the temporary pre-CLI entry point."""

    options = _build_parser().parse_args(arguments)
    try:
        resolved = resolve_layers(options.layer)
        outputs = write_resolution_outputs(
            resolved,
            options.output,
            provenance_output=options.provenance_output,
            canonical_output=options.canonical_output,
            digest_output=options.digest_output,
            overwrite=options.overwrite,
        )
        summary: dict[str, Any] = {
            **resolved.to_summary(),
            "outputs": outputs,
        }
        if options.scene_id is not None or options.seed is not None:
            if options.scene_id is None or options.seed is None:
                raise ResolutionError("--scene-id and --seed must be supplied together")
            summary["run_id"] = generate_run_id(
                experiment_id=resolved.experiment_id,
                scene_id=options.scene_id,
                seed=options.seed,
                scientific_configuration_digest=resolved.scientific_digest,
                attempt=options.attempt,
            )
    except (DocumentLoadError, ResolutionError, SchemaValidationError) as error:
        payload = {"valid": False, "error": str(error)}
        print(json.dumps(payload, indent=2) if options.json else str(error))
        return 1
    if options.json:
        print(json.dumps({"valid": True, **summary}, indent=2, sort_keys=True))
    else:
        print(f"resolved {resolved.experiment_id}")
        print(f"scientific digest: {resolved.scientific_digest}")
        print(f"full digest: {resolved.full_digest}")
        if "run_id" in summary:
            print(f"run ID: {summary['run_id']}")
        print(f"resolved output: {outputs['resolved_output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
