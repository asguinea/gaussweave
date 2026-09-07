"""Safe, deterministic on-disk serialization for Gaussian representations."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

from gaussweave.config.resolution import canonical_json_bytes
from gaussweave.representation.models import (
    CANONICAL_FRAME,
    LOCAL_FRAME,
    AppearanceResiduals,
    CanonicalTerminal,
    ExplicitRepresentation,
    GaussianArrays,
    GridRepeat,
    PrunedRepresentation,
    StructuralRepresentation,
    UniqueComponent,
)

SERIALIZATION_VERSION = "gaussweave-representation-serialization-v1"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ARRAY_VALUES = 1_000_000


class RepresentationIOError(ValueError):
    """Raised when a representation is unsafe, corrupt, or incomplete."""


def _canonical_json(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RepresentationIOError(f"unsafe representation path: {value!r}")
    if "\\" in value or ":" in value:
        raise RepresentationIOError(f"non-portable representation path: {value!r}")
    return Path(*pure.parts)


def _npy_payload(
    values: Sequence[Sequence[float]] | Sequence[float] | Sequence[Sequence[int]],
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> bytes:
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise RepresentationIOError("negative array dimensions are invalid")
        count *= dimension
    if count > MAX_ARRAY_VALUES:
        raise RepresentationIOError("array exceeds bounded GW element count")
    if dtype not in {"<f4", "<u4", "|i1"}:
        raise RepresentationIOError(f"unsupported array dtype {dtype}")
    header_dict = {"descr": dtype, "fortran_order": False, "shape": shape}
    header = repr(header_dict).encode("latin1")
    preamble_length = 10
    padding = (16 - ((preamble_length + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    preamble = b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header))
    flattened: list[float | int] = []
    if len(shape) == 1:
        flattened.extend(cast(Sequence[float | int], values))
    else:
        for row in cast(Sequence[Sequence[float | int]], values):
            flattened.extend(row)
    if len(flattened) != count:
        raise RepresentationIOError("array value count does not match declared shape")
    if dtype == "<f4":
        body = b"".join(struct.pack("<f", float(value)) for value in flattened)
    elif dtype == "<u4":
        body = b"".join(struct.pack("<I", int(value)) for value in flattened)
    else:
        body = bytes((int(value) & 0xFF) for value in flattened)
    return preamble + header + body


def _read_npy(
    path: Path, *, expected_dtype: str, expected_shape: tuple[int, ...]
) -> tuple[Any, ...]:
    payload = path.read_bytes()
    if len(payload) > MAX_FILE_BYTES:
        raise RepresentationIOError(f"array file is too large: {path.name}")
    if (
        len(payload) < 10
        or payload[:6] != b"\x93NUMPY"
        or payload[6:8] != bytes((1, 0))
    ):
        raise RepresentationIOError(f"unsupported or corrupt NPY header: {path.name}")
    header_length = struct.unpack("<H", payload[8:10])[0]
    if header_length <= 0 or header_length > 4096 or 10 + header_length > len(payload):
        raise RepresentationIOError(f"invalid NPY header length: {path.name}")
    try:
        header = ast.literal_eval(
            payload[10 : 10 + header_length].decode("latin1").strip()
        )
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise RepresentationIOError(f"invalid NPY header: {path.name}") from error
    if not isinstance(header, dict):
        raise RepresentationIOError(f"NPY header must be a mapping: {path.name}")
    if (
        header.get("descr") != expected_dtype
        or header.get("fortran_order") is not False
    ):
        raise RepresentationIOError(f"NPY dtype/order mismatch: {path.name}")
    if tuple(header.get("shape", ())) != expected_shape:
        raise RepresentationIOError(f"NPY shape mismatch: {path.name}")
    count = 1
    for dimension in expected_shape:
        count *= dimension
    item_size = 4 if expected_dtype in {"<f4", "<u4"} else 1
    body = payload[10 + header_length :]
    if len(body) != count * item_size:
        raise RepresentationIOError(f"NPY payload length mismatch: {path.name}")
    if expected_dtype == "<f4":
        return tuple(value[0] for value in struct.iter_unpack("<f", body))
    if expected_dtype == "<u4":
        return tuple(value[0] for value in struct.iter_unpack("<I", body))
    return tuple(value if value < 128 else value - 256 for value in body)


def _reshape(values: tuple[Any, ...], width: int) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple(values[offset : offset + width])
        for offset in range(0, len(values), width)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_specs(
    prefix: str, arrays: GaussianArrays
) -> list[tuple[str, bytes, str, list[int], str]]:
    return [
        (
            f"arrays/{prefix}means.npy",
            _npy_payload(arrays.means, shape=(arrays.count, 3), dtype="<f4"),
            "<f4",
            [arrays.count, 3],
            "gaussian_means",
        ),
        (
            f"arrays/{prefix}quaternions.npy",
            _npy_payload(arrays.quaternions, shape=(arrays.count, 4), dtype="<f4"),
            "<f4",
            [arrays.count, 4],
            "gaussian_quaternions",
        ),
        (
            f"arrays/{prefix}scales.npy",
            _npy_payload(arrays.scales, shape=(arrays.count, 3), dtype="<f4"),
            "<f4",
            [arrays.count, 3],
            "gaussian_scales",
        ),
        (
            f"arrays/{prefix}opacities.npy",
            _npy_payload(arrays.opacities, shape=(arrays.count,), dtype="<f4"),
            "<f4",
            [arrays.count],
            "gaussian_opacities",
        ),
        (
            f"arrays/{prefix}colors.npy",
            _npy_payload(arrays.colors, shape=(arrays.count, 3), dtype="<f4"),
            "<f4",
            [arrays.count, 3],
            "gaussian_appearance",
        ),
    ]


def _grid_dict(grid: GridRepeat) -> dict[str, Any]:
    return {
        "rows": grid.rows,
        "columns": grid.columns,
        "origin": list(grid.origin),
        "row_step": list(grid.row_step),
        "column_step": list(grid.column_step),
        "active_indices": list(grid.active_indices),
        "ordering_policy": grid.ordering_policy,
        "scientific_digest": grid.scientific_digest,
    }


def _write_files(
    root: Path,
    files: Iterable[tuple[str, bytes, str | None, list[int] | None, str]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative, payload, dtype, shape, category in sorted(
        files, key=lambda item: item[0]
    ):
        path = root / _safe_relative(relative)
        _atomic_write(path, payload)
        entry: dict[str, Any] = {
            "path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "category": category,
        }
        if dtype is not None:
            entry["dtype"] = dtype
        if shape is not None:
            entry["shape"] = shape
        entries.append(entry)
    return entries


def _prepare_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"representation output must be empty: {root}")


def save_representation(
    representation: ExplicitRepresentation
    | PrunedRepresentation
    | StructuralRepresentation,
    root: Path,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Write a representation using JSON and NPY only."""

    root = root.resolve()
    _prepare_root(root)
    files: list[tuple[str, bytes, str | None, list[int] | None, str]] = []
    if isinstance(representation, (ExplicitRepresentation, PrunedRepresentation)):
        pruned = isinstance(representation, PrunedRepresentation)
        metadata: dict[str, Any] = {
            "serialization_version": SERIALIZATION_VERSION,
            "representation_kind": "pruned_explicit" if pruned else "explicit",
            "format_version": representation.format_version,
            "method_id": representation.method_id,
            "fixture_id": representation.fixture_id,
            "appearance_regime": representation.appearance_regime,
            "principal_seed": representation.principal_seed,
            "grid": _grid_dict(representation.grid),
            "gaussian_count": representation.gaussians.count,
            "coordinate_frame": representation.gaussians.coordinate_frame,
            "ordering_policy": representation.gaussians.ordering_policy,
            "stable_id_policy": "unique-gNNNN_then_instance-rRR-cCC/terminal-gNNNN",
            "scientific_digest": representation.scientific_digest,
            "gaussian_digest": representation.gaussians.scientific_digest,
            "limitations": list(representation.limitations),
        }
        files.extend(_array_specs("", representation.gaussians))
        if isinstance(representation, PrunedRepresentation):
            metadata["source_representation_digest"] = (
                representation.source_representation_digest
            )
            metadata["target_budget_bytes"] = representation.target_budget_bytes
            metadata["actual_complete_bytes"] = representation.actual_complete_bytes
            metadata["source_gaussian_count"] = representation.source_gaussian_count
            metadata["stored_gaussian_count"] = representation.stored_gaussian_count
            metadata["materialized_gaussian_count"] = (
                representation.materialized_gaussian_count
            )
            metadata["importance_policy_version"] = (
                representation.importance_policy_version
            )
            metadata["score_summary"] = dict(representation.score_summary)
            metadata["retained_index_ordering"] = representation.ordering_policy
            files.append(
                (
                    "arrays/retained_original_indices.npy",
                    _npy_payload(
                        representation.retained_original_indices,
                        shape=(representation.gaussians.count,),
                        dtype="<u4",
                    ),
                    "<u4",
                    [representation.gaussians.count],
                    "pruning_indices",
                )
            )
    else:
        metadata = {
            "serialization_version": SERIALIZATION_VERSION,
            "representation_kind": "structural",
            "format_version": representation.format_version,
            "method_id": representation.method_id,
            "fixture_id": representation.fixture_id,
            "appearance_regime": representation.appearance_regime,
            "principal_seed": representation.principal_seed,
            "stored_gaussian_count": representation.stored_gaussian_count,
            "materialized_gaussian_count": representation.materialized_gaussian_count,
            "coordinate_frame": CANONICAL_FRAME,
            "ordering_policy": "unique_then_grid_instance_then_terminal",
            "decoder_id": representation.decoder_id,
            "terminal": {
                "component_id": representation.terminal.component_id,
                "role": representation.terminal.role,
                "local_frame": representation.terminal.local_frame,
                "provenance": dict(representation.terminal.provenance),
                "scientific_digest": representation.terminal.scientific_digest,
            },
            "unique": {
                "component_id": representation.unique.component_id,
                "role": representation.unique.role,
                "provenance": dict(representation.unique.provenance),
                "scientific_digest": representation.unique.scientific_digest,
            },
            "scientific_digest": representation.scientific_digest,
            "limitations": list(representation.limitations),
        }
        files.extend(_array_specs("unique_", representation.unique.gaussians))
        files.extend(_array_specs("terminal_", representation.terminal.gaussians))
        grid_payload = _canonical_json(_grid_dict(representation.grid))
        files.append(("repeat/grid.json", grid_payload, None, None, "repeat_rule"))
        binding = {
            "component_id": representation.terminal.component_id,
            "instance_ids": [
                representation.grid.instance_id(index)
                for index in representation.grid.active_indices
            ],
            "binding_policy": (
                "each_active_instance_references_the_canonical_terminal_once"
            ),
        }
        files.append(
            ("repeat/binding.json", _canonical_json(binding), None, None, "binding")
        )
        if representation.residuals is not None:
            residuals = representation.residuals
            residual_metadata = {
                "path": "repeat/residuals_int8.npy",
                "instance_ids": list(residuals.instance_ids),
                "scale": residuals.scale,
                "zero_point": residuals.zero_point,
                "dtype": residuals.dtype,
                "channels": list(residuals.channels),
                "clipping_policy": residuals.clipping_policy,
                "ordering_policy": residuals.ordering_policy,
                "saturation_count": residuals.saturation_count,
                "scientific_digest": residuals.scientific_digest,
            }
            metadata["residuals"] = residual_metadata
            files.append(
                (
                    "repeat/residuals_int8.npy",
                    _npy_payload(
                        residuals.values,
                        shape=(len(residuals.values), 3),
                        dtype="|i1",
                    ),
                    "|i1",
                    [len(residuals.values), 3],
                    "appearance_residuals",
                )
            )
        else:
            metadata["residuals"] = None
    files.append(
        (
            "representation.json",
            _canonical_json(metadata),
            None,
            None,
            "method_and_version_metadata",
        )
    )
    entries = _write_files(root, files)
    manifest = {
        "serialization_version": SERIALIZATION_VERSION,
        "entry_count": len(entries),
        "entries": entries,
        "self_entry_policy": "manifest_not_self_hashed",
    }
    _atomic_write(root / "manifest.json", _canonical_json(manifest))
    return validate_representation(root) if validate else manifest


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise RepresentationIOError(f"missing or oversized JSON file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RepresentationIOError(f"invalid JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise RepresentationIOError(f"JSON root must be an object: {path.name}")
    return value


def _load_manifest(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _load_json(root / "manifest.json")
    if manifest.get("serialization_version") != SERIALIZATION_VERSION:
        raise RepresentationIOError("unsupported representation serialization version")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or manifest.get("entry_count") != len(
        raw_entries
    ):
        raise RepresentationIOError("manifest entry count is invalid")
    entries: dict[str, dict[str, Any]] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise RepresentationIOError("manifest contains an invalid entry")
        relative = raw["path"]
        _safe_relative(relative)
        if relative in entries:
            raise RepresentationIOError("manifest contains duplicate paths")
        entries[relative] = raw
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    expected = set(entries) | {"manifest.json"}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RepresentationIOError(
            f"representation file set mismatch; missing={missing}, extra={extra}"
        )
    for relative, entry in entries.items():
        path = root / _safe_relative(relative)
        if path.is_symlink():
            raise RepresentationIOError(f"symlinks are not allowed: {relative}")
        if path.stat().st_size != entry.get("bytes") or _sha256(path) != entry.get(
            "sha256"
        ):
            raise RepresentationIOError(f"manifest checksum/size mismatch: {relative}")
    return manifest, entries


def _arrays(
    root: Path,
    entries: dict[str, dict[str, Any]],
    prefix: str,
    count: int,
    ids: tuple[str, ...],
    frame: str,
    ordering_policy: str = "stable_ids_lexical_by_construction",
) -> GaussianArrays:
    def read(field: str, width: int | None) -> tuple[Any, ...]:
        relative = f"arrays/{prefix}{field}.npy"
        entry = entries.get(relative)
        shape = (count,) if width is None else (count, width)
        if (
            entry is None
            or entry.get("dtype") != "<f4"
            or tuple(entry.get("shape", ())) != shape
        ):
            raise RepresentationIOError(
                f"manifest array declaration mismatch: {relative}"
            )
        values = _read_npy(root / relative, expected_dtype="<f4", expected_shape=shape)
        return values if width is None else _reshape(values, width)

    return GaussianArrays(
        means=read("means", 3),
        quaternions=read("quaternions", 4),
        scales=read("scales", 3),
        opacities=read("opacities", None),
        colors=read("colors", 3),
        stable_ids=ids,
        coordinate_frame=frame,
        ordering_policy=ordering_policy,
    )


def _grid(value: dict[str, Any]) -> GridRepeat:
    try:
        grid = GridRepeat(
            rows=int(value["rows"]),
            columns=int(value["columns"]),
            origin=tuple(value["origin"]),
            row_step=tuple(value["row_step"]),
            column_step=tuple(value["column_step"]),
            active_indices=tuple(int(index) for index in value["active_indices"]),
            ordering_policy=str(value["ordering_policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RepresentationIOError("invalid grid metadata") from error
    if grid.scientific_digest != value.get("scientific_digest"):
        raise RepresentationIOError("grid scientific digest mismatch")
    return grid


def load_representation(
    root: Path,
) -> ExplicitRepresentation | PrunedRepresentation | StructuralRepresentation:
    root = root.resolve()
    _, entries = _load_manifest(root)
    metadata = _load_json(root / "representation.json")
    if metadata.get("serialization_version") != SERIALIZATION_VERSION:
        raise RepresentationIOError("representation metadata version mismatch")
    kind = metadata.get("representation_kind")
    grid_value = metadata.get("grid")
    if kind == "structural":
        grid_value = _load_json(root / "repeat/grid.json")
    if not isinstance(grid_value, dict):
        raise RepresentationIOError("representation is missing grid metadata")
    grid = _grid(grid_value)
    if kind in {"explicit", "pruned_explicit"}:
        count = int(metadata.get("gaussian_count", 0))
        full_ids = tuple(
            [f"unique-g{index:04d}" for index in range(512)]
            + [
                f"{grid.instance_id(flat_index)}/terminal-g{terminal_index:04d}"
                for flat_index in grid.active_indices
                for terminal_index in range(256)
            ]
        )
        retained_indices: tuple[int, ...] | None = None
        if kind == "pruned_explicit":
            retained_indices = tuple(
                int(value)
                for value in _read_npy(
                    root / "arrays/retained_original_indices.npy",
                    expected_dtype="<u4",
                    expected_shape=(count,),
                )
            )
            try:
                ids = tuple(full_ids[index] for index in retained_indices)
            except IndexError as error:
                raise RepresentationIOError(
                    "retained index is outside the source explicit array"
                ) from error
        else:
            ids = full_ids
        arrays = _arrays(
            root,
            entries,
            "",
            count,
            ids,
            CANONICAL_FRAME,
            str(metadata["ordering_policy"]),
        )
        representation: (
            ExplicitRepresentation | PrunedRepresentation | StructuralRepresentation
        )
        if kind == "pruned_explicit":
            if retained_indices is None:
                raise RepresentationIOError("pruned retained indices are missing")
            representation = PrunedRepresentation(
                gaussians=arrays,
                retained_original_indices=retained_indices,
                source_representation_digest=str(
                    metadata["source_representation_digest"]
                ),
                fixture_id=str(metadata["fixture_id"]),
                appearance_regime=str(metadata["appearance_regime"]),
                principal_seed=int(metadata["principal_seed"]),
                grid=grid,
                target_budget_bytes=int(metadata["target_budget_bytes"]),
                actual_complete_bytes=int(metadata["actual_complete_bytes"]),
                source_gaussian_count=int(metadata["source_gaussian_count"]),
                score_summary=dict(metadata["score_summary"]),
                method_id=str(metadata["method_id"]),
                format_version=str(metadata["format_version"]),
                importance_policy_version=str(metadata["importance_policy_version"]),
                ordering_policy=str(metadata["retained_index_ordering"]),
                limitations=tuple(metadata["limitations"]),
            )
        else:
            representation = ExplicitRepresentation(
                gaussians=arrays,
                fixture_id=str(metadata["fixture_id"]),
                appearance_regime=str(metadata["appearance_regime"]),
                principal_seed=int(metadata["principal_seed"]),
                grid=grid,
                method_id=str(metadata["method_id"]),
                format_version=str(metadata["format_version"]),
                limitations=tuple(metadata["limitations"]),
            )
        if arrays.scientific_digest != metadata.get("gaussian_digest"):
            raise RepresentationIOError("explicit Gaussian scientific digest mismatch")
    elif kind == "structural":
        terminal_meta = metadata.get("terminal")
        unique_meta = metadata.get("unique")
        if not isinstance(terminal_meta, dict) or not isinstance(unique_meta, dict):
            raise RepresentationIOError("structural component metadata is missing")
        unique_arrays = _arrays(
            root,
            entries,
            "unique_",
            512,
            tuple(f"unique-g{index:04d}" for index in range(512)),
            CANONICAL_FRAME,
        )
        terminal_arrays = _arrays(
            root,
            entries,
            "terminal_",
            256,
            tuple(f"terminal-g{index:04d}" for index in range(256)),
            LOCAL_FRAME,
        )
        unique = UniqueComponent(
            component_id=str(unique_meta["component_id"]),
            gaussians=unique_arrays,
            provenance=dict(unique_meta["provenance"]),
            role=str(unique_meta["role"]),
        )
        terminal = CanonicalTerminal(
            component_id=str(terminal_meta["component_id"]),
            gaussians=terminal_arrays,
            provenance=dict(terminal_meta["provenance"]),
            role=str(terminal_meta["role"]),
            local_frame=str(terminal_meta["local_frame"]),
        )
        residual_meta = metadata.get("residuals")
        residuals = None
        if residual_meta is not None:
            if not isinstance(residual_meta, dict):
                raise RepresentationIOError("invalid residual metadata")
            instance_ids = tuple(str(value) for value in residual_meta["instance_ids"])
            values = _read_npy(
                root / "repeat/residuals_int8.npy",
                expected_dtype="|i1",
                expected_shape=(len(instance_ids), 3),
            )
            residuals = AppearanceResiduals(
                values=_reshape(values, 3),
                instance_ids=instance_ids,
                scale=float(residual_meta["scale"]),
                zero_point=int(residual_meta["zero_point"]),
                dtype=str(residual_meta["dtype"]),
                channels=tuple(residual_meta["channels"]),
                clipping_policy=str(residual_meta["clipping_policy"]),
                ordering_policy=str(residual_meta["ordering_policy"]),
                saturation_count=int(residual_meta["saturation_count"]),
            )
            if residuals.scientific_digest != residual_meta.get("scientific_digest"):
                raise RepresentationIOError("residual scientific digest mismatch")
        representation = StructuralRepresentation(
            unique=unique,
            terminal=terminal,
            grid=grid,
            fixture_id=str(metadata["fixture_id"]),
            appearance_regime=str(metadata["appearance_regime"]),
            principal_seed=int(metadata["principal_seed"]),
            residuals=residuals,
            method_id=str(metadata["method_id"]),
            format_version=str(metadata["format_version"]),
            decoder_id=str(metadata["decoder_id"]),
            limitations=tuple(metadata["limitations"]),
        )
        if unique.scientific_digest != unique_meta.get("scientific_digest"):
            raise RepresentationIOError("unique component scientific digest mismatch")
        if terminal.scientific_digest != terminal_meta.get("scientific_digest"):
            raise RepresentationIOError("terminal component scientific digest mismatch")
    else:
        raise RepresentationIOError(f"unsupported representation kind: {kind!r}")
    if representation.scientific_digest != metadata.get("scientific_digest"):
        raise RepresentationIOError("representation scientific digest mismatch")
    return representation


def validate_representation(root: Path) -> dict[str, Any]:
    representation = load_representation(root)
    if (
        isinstance(representation, PrunedRepresentation)
        and representation.actual_complete_bytes > representation.target_budget_bytes
    ):
        raise RepresentationIOError(
            "pruned representation exceeds its target byte budget"
        )
    _, entries = _load_manifest(root.resolve())
    return {
        "valid": True,
        "path": str(root.resolve()),
        "method_id": representation.method_id,
        "representation_kind": (
            "explicit"
            if isinstance(
                representation,
                (ExplicitRepresentation, PrunedRepresentation),
            )
            else "structural"
        ),
        "scientific_digest": representation.scientific_digest,
        "file_count": len(entries) + 1,
        "complete_bytes": sum(
            (root.resolve() / path).stat().st_size for path in entries
        )
        + (root.resolve() / "manifest.json").stat().st_size,
    }
