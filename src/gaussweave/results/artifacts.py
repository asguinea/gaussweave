"""CPU-only artifact path, checksum, inventory, and archive safety primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from gaussweave.config.resolution import canonical_json_bytes, pretty_json_bytes

INVENTORY_VERSION = "1.0"
DIRECTORY_DIGEST_POLICY_VERSION = "1.0"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
DRIVE_RE = re.compile(r"^[A-Za-z]:")
ARTIFACT_TYPES = frozenset(
    {
        "resolved_config",
        "environment",
        "checkpoint",
        "grammar",
        "binding",
        "render",
        "metric_observations",
        "accounting_record",
        "resource_record",
        "failure_report",
        "qualitative_panel",
        "video",
        "table",
        "figure",
        "other",
    }
)


class ArtifactError(ValueError):
    """Artifact data or filesystem state is unsafe or invalid."""


class HiddenPolicy(StrEnum):
    """Deterministic hidden-entry handling policy."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


class VerificationState(StrEnum):
    """Overall inventory verification state."""

    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


@dataclass(frozen=True)
class ArtifactPath:
    """A portable path bound safely to a declared local root."""

    root: Path
    portable: str
    local: Path

    @classmethod
    def resolve(
        cls, root: Path, portable: str, *, must_exist: bool = False
    ) -> ArtifactPath:
        normalized = normalize_portable_path(portable)
        resolved_root = root.resolve(strict=must_exist)
        if must_exist and not resolved_root.is_dir():
            raise ArtifactError(f"artifact root is not a directory: {root}")
        local = resolved_root.joinpath(*PurePosixPath(normalized).parts)
        _reject_symlink_components(resolved_root, local)
        try:
            local.resolve(strict=must_exist).relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise ArtifactError(f"path escapes artifact root: {portable!r}") from error
        return cls(resolved_root, normalized, local)


@dataclass(frozen=True)
class FileIdentity:
    bytes: int
    digest: str


@dataclass(frozen=True)
class DirectoryEntry:
    path: str
    bytes: int
    digest: str


@dataclass(frozen=True)
class DirectoryIdentity:
    policy_version: str
    digest: str
    entries: tuple[DirectoryEntry, ...]


@dataclass(frozen=True)
class ArtifactDeclaration:
    artifact_id: str
    artifact_type: str
    path: str
    required: bool = True
    description: str | None = None
    producing_run_id: str | None = None
    created_at: str | None = None
    format_version: str | None = None


@dataclass(frozen=True)
class ArtifactRecord(ArtifactDeclaration):
    bytes: int = 0
    digest: str = ""


@dataclass(frozen=True)
class ArtifactInventory:
    inventory_version: str
    inventory_id: str
    artifact_root: str
    artifacts: tuple[ArtifactRecord, ...]
    total_required_bytes: int
    total_optional_bytes: int
    artifact_count: int
    directory_digest: str
    directory_digest_policy_version: str
    hidden_policy: str
    excluded_paths: tuple[str, ...]
    generated_at: str
    warnings: tuple[str, ...]
    content_digest: str

    def to_dict(self, *, include_content_digest: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["artifacts"] = [asdict(item) for item in self.artifacts]
        value["excluded_paths"] = list(self.excluded_paths)
        value["warnings"] = list(self.warnings)
        if not include_content_digest:
            value.pop("content_digest")
        return value


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    severity: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    state: VerificationState
    issues: tuple[VerificationIssue, ...]
    checked_artifacts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "valid": self.state is not VerificationState.INVALID,
            "checked_artifacts": self.checked_artifacts,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    kind: str = "file"
    link_target: str | None = None


def normalize_portable_path(value: str) -> str:
    """Validate an already-POSIX portable relative path."""

    if not isinstance(value, str) or not value or value == ".":
        raise ArtifactError("portable path must be non-empty and not '.'")
    if "\\" in value:
        raise ArtifactError("backslashes are forbidden in portable paths")
    if value.startswith("/") or DRIVE_RE.match(value):
        raise ArtifactError(f"portable path must be relative: {value!r}")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ArtifactError(f"ambiguous or traversing portable path: {value!r}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ArtifactError("portable path is empty")
    return normalized


def hash_file(
    path: Path, *, expected_digest: str | None = None, chunk_size: int = 1024 * 1024
) -> FileIdentity:
    """Stream a regular file into a bounded-memory SHA-256 identity."""

    if chunk_size < 1:
        raise ArtifactError("chunk size must be positive")
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
                count += len(chunk)
    except OSError as error:
        raise ArtifactError(f"unable to hash {path}: {error}") from error
    identity = FileIdentity(count, f"sha256:{digest.hexdigest()}")
    if expected_digest is not None and identity.digest != expected_digest:
        raise ArtifactError(f"digest mismatch for {path}")
    return identity


def directory_identity(
    root: Path,
    *,
    hidden_policy: HiddenPolicy = HiddenPolicy.INCLUDE,
    excluded_paths: Iterable[str] = (),
) -> DirectoryIdentity:
    """Hash a bounded tree using sorted [path, bytes, digest] records."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ArtifactError(f"artifact root is not a directory: {root}")
    excluded = {normalize_portable_path(item) for item in excluded_paths}
    entries: list[DirectoryEntry] = []
    for portable, local in _walk_files(resolved_root, hidden_policy):
        if portable in excluded:
            continue
        identity = hash_file(local)
        entries.append(DirectoryEntry(portable, identity.bytes, identity.digest))
    entries.sort(key=lambda item: item.path)
    payload: Any = {
        "policy_version": DIRECTORY_DIGEST_POLICY_VERSION,
        "files": [[item.path, item.bytes, item.digest] for item in entries],
    }
    digest = f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"
    return DirectoryIdentity(DIRECTORY_DIGEST_POLICY_VERSION, digest, tuple(entries))


def build_inventory(
    root: Path,
    declarations: Sequence[ArtifactDeclaration],
    *,
    inventory_id: str,
    artifact_root: str = ".",
    hidden_policy: HiddenPolicy = HiddenPolicy.INCLUDE,
    excluded_paths: Iterable[str] = (),
    generated_at: str | None = None,
    warnings: Iterable[str] = (),
) -> ArtifactInventory:
    """Build an inventory from caller-classified explicit file declarations."""

    if not ID_RE.fullmatch(inventory_id):
        raise ArtifactError(f"invalid inventory ID: {inventory_id!r}")
    ids: set[str] = set()
    paths: set[str] = set()
    records: list[ArtifactRecord] = []
    for declaration in declarations:
        path = normalize_portable_path(declaration.path)
        if not ID_RE.fullmatch(declaration.artifact_id):
            raise ArtifactError(f"invalid artifact ID: {declaration.artifact_id!r}")
        if declaration.artifact_type not in ARTIFACT_TYPES:
            raise ArtifactError(f"invalid artifact type: {declaration.artifact_type!r}")
        if declaration.artifact_id in ids:
            raise ArtifactError(f"duplicate artifact ID: {declaration.artifact_id}")
        if path in paths:
            raise ArtifactError(f"duplicate artifact path: {path}")
        ids.add(declaration.artifact_id)
        paths.add(path)
        target = ArtifactPath.resolve(root, path, must_exist=True)
        identity = hash_file(target.local)
        values = asdict(declaration)
        values["path"] = path
        records.append(
            ArtifactRecord(**values, bytes=identity.bytes, digest=identity.digest)
        )
    records.sort(key=lambda item: item.path)
    exclusions = tuple(sorted({normalize_portable_path(p) for p in excluded_paths}))
    tree = directory_identity(
        root, hidden_policy=hidden_policy, excluded_paths=exclusions
    )
    inventory = ArtifactInventory(
        inventory_version=INVENTORY_VERSION,
        inventory_id=inventory_id,
        artifact_root=artifact_root,
        artifacts=tuple(records),
        total_required_bytes=sum(item.bytes for item in records if item.required),
        total_optional_bytes=sum(item.bytes for item in records if not item.required),
        artifact_count=len(records),
        directory_digest=tree.digest,
        directory_digest_policy_version=tree.policy_version,
        hidden_policy=hidden_policy.value,
        excluded_paths=exclusions,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        warnings=tuple(warnings),
        content_digest="",
    )
    return replace(inventory, content_digest=_inventory_digest(inventory))


def inventory_tree(
    root: Path,
    *,
    inventory_id: str,
    artifact_type: str = "other",
    optional_paths: Iterable[str] = (),
    hidden_policy: HiddenPolicy = HiddenPolicy.INCLUDE,
    excluded_paths: Iterable[str] = (),
    generated_at: str | None = None,
) -> ArtifactInventory:
    """Build a bounded-tree inventory using caller-supplied classification."""

    excluded = {normalize_portable_path(item) for item in excluded_paths}
    optional = {normalize_portable_path(item) for item in optional_paths}
    declarations = [
        ArtifactDeclaration(
            artifact_id=f"artifact-{index:04d}",
            artifact_type=artifact_type,
            path=portable,
            required=portable not in optional,
        )
        for index, (portable, _) in enumerate(
            (
                pair
                for pair in _walk_files(root.resolve(strict=True), hidden_policy)
                if pair[0] not in excluded
            ),
            start=1,
        )
    ]
    return build_inventory(
        root,
        declarations,
        inventory_id=inventory_id,
        hidden_policy=hidden_policy,
        excluded_paths=excluded,
        generated_at=generated_at,
    )


def write_inventory(
    inventory: ArtifactInventory, output: Path, *, overwrite: bool = False
) -> None:
    """Atomically write deterministic pretty UTF-8 JSON."""

    if output.exists() and not overwrite:
        raise ArtifactError(f"refusing to overwrite existing output: {output}")
    _atomic_write(output, pretty_json_bytes(inventory.to_dict()))


def load_inventory(path: Path) -> ArtifactInventory:
    """Load and structurally validate an inventory and its self digest."""

    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"inventory is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        records = tuple(ArtifactRecord(**item) for item in value["artifacts"])
        scalar_values = {
            key: item
            for key, item in value.items()
            if key not in {"artifacts", "excluded_paths", "warnings"}
        }
        inventory = ArtifactInventory(
            **scalar_values,
            artifacts=records,
            excluded_paths=tuple(value["excluded_paths"]),
            warnings=tuple(value["warnings"]),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ArtifactError(f"invalid inventory {path}: {error}") from error
    _validate_loaded_inventory(inventory)
    if inventory.content_digest != _inventory_digest(inventory):
        raise ArtifactError("inventory content digest mismatch")
    return inventory


def verify_inventory(
    inventory: ArtifactInventory, root: Path, *, strict: bool = False
) -> VerificationResult:
    """Return structured validation of inventory records and tree identity."""

    issues: list[VerificationIssue] = []
    for record in inventory.artifacts:
        try:
            target = ArtifactPath.resolve(root, record.path)
            if not target.local.exists():
                severity = "error" if record.required or strict else "warning"
                issues.append(
                    VerificationIssue(
                        "missing_required" if record.required else "missing_optional",
                        severity,
                        "artifact is missing",
                        record.path,
                    )
                )
                continue
            identity = hash_file(target.local)
            if identity.bytes != record.bytes:
                issues.append(
                    VerificationIssue(
                        "size_mismatch",
                        "error",
                        f"expected {record.bytes} bytes, found {identity.bytes}",
                        record.path,
                    )
                )
            if identity.digest != record.digest:
                issues.append(
                    VerificationIssue(
                        "digest_mismatch",
                        "error",
                        "SHA-256 digest differs",
                        record.path,
                    )
                )
        except ArtifactError as error:
            issues.append(
                VerificationIssue("unsafe_artifact", "error", str(error), record.path)
            )
    try:
        tree = directory_identity(
            root,
            hidden_policy=HiddenPolicy(inventory.hidden_policy),
            excluded_paths=inventory.excluded_paths,
        )
        missing_optional = {
            issue.path
            for issue in issues
            if issue.code == "missing_optional" and issue.path is not None
        }
        expected_present = {
            item.path: (item.bytes, item.digest)
            for item in inventory.artifacts
            if item.path not in missing_optional
        }
        actual = {item.path: (item.bytes, item.digest) for item in tree.entries}
        optional_only_change = (
            bool(missing_optional) and actual == expected_present and not strict
        )
        if tree.digest != inventory.directory_digest and not optional_only_change:
            issues.append(
                VerificationIssue(
                    "directory_digest_mismatch", "error", "directory digest differs"
                )
            )
        if strict:
            expected = {item.path for item in inventory.artifacts}
            for entry in tree.entries:
                if entry.path not in expected:
                    issues.append(
                        VerificationIssue(
                            "unexpected_artifact",
                            "error",
                            "unexpected artifact",
                            entry.path,
                        )
                    )
    except (ArtifactError, ValueError) as error:
        issues.append(VerificationIssue("directory_error", "error", str(error)))
    errors = any(item.severity == "error" for item in issues)
    state = (
        VerificationState.INVALID
        if errors
        else VerificationState.VALID_WITH_WARNINGS
        if issues
        else VerificationState.VALID
    )
    return VerificationResult(state, tuple(issues), len(inventory.artifacts))


def validate_archive_members(members: Iterable[ArchiveMember]) -> tuple[str, ...]:
    """Validate normalized ZIP/tar destinations and exposed link targets."""

    destinations: set[str] = set()
    result: list[str] = []
    for member in members:
        raw = member.name[:-1] if member.name.endswith("/") else member.name
        destination = normalize_portable_path(raw)
        if destination in destinations:
            raise ArtifactError(f"duplicate archive destination: {destination}")
        destinations.add(destination)
        if member.kind in {"symlink", "hardlink"}:
            if not member.link_target:
                raise ArtifactError(f"archive link has no target: {destination}")
            target = member.link_target
            if "\\" in target or target.startswith("/") or DRIVE_RE.match(target):
                raise ArtifactError(f"unsafe archive link target: {target!r}")
            combined = PurePosixPath(destination).parent.joinpath(target)
            depth = 0
            for part in combined.parts:
                depth = depth - 1 if part == ".." else depth + (part != ".")
                if depth < 0:
                    raise ArtifactError(
                        f"archive link escapes destination: {destination}"
                    )
        elif member.kind not in {"file", "directory"}:
            raise ArtifactError(f"unsupported archive member type: {member.kind}")
        result.append(destination)
    return tuple(result)


def _walk_files(root: Path, hidden_policy: HiddenPolicy) -> Iterable[tuple[str, Path]]:
    def visit(directory: Path) -> Iterable[tuple[str, Path]]:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ArtifactError(f"unable to inspect {directory}: {error}") from error
        for entry in entries:
            if hidden_policy is HiddenPolicy.EXCLUDE and entry.name.startswith("."):
                continue
            path = Path(entry.path)
            if entry.is_symlink():
                raise ArtifactError(f"symlink rejected during tree inventory: {path}")
            if entry.is_dir(follow_symlinks=False):
                yield from visit(path)
            elif entry.is_file(follow_symlinks=False):
                yield path.relative_to(root).as_posix(), path
            else:
                raise ArtifactError(f"non-regular tree entry: {path}")

    return visit(root)


def _reject_symlink_components(root: Path, local: Path) -> None:
    current = root
    for part in local.relative_to(root).parts:
        current /= part
        try:
            if current.is_symlink():
                raise ArtifactError(f"symlink component rejected: {current}")
        except OSError as error:
            raise ArtifactError(
                f"unable to inspect path component: {current}"
            ) from error


def _inventory_digest(inventory: ArtifactInventory) -> str:
    payload = inventory.to_dict(include_content_digest=False)
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _validate_loaded_inventory(inventory: ArtifactInventory) -> None:
    if inventory.inventory_version != INVENTORY_VERSION:
        raise ArtifactError(
            f"unsupported inventory version: {inventory.inventory_version}"
        )
    if not ID_RE.fullmatch(inventory.inventory_id):
        raise ArtifactError("invalid inventory ID")
    ids: set[str] = set()
    paths: set[str] = set()
    for item in inventory.artifacts:
        normalize_portable_path(item.path)
        if not ID_RE.fullmatch(item.artifact_id) or item.artifact_type not in ARTIFACT_TYPES:
            raise ArtifactError("invalid artifact ID or type")
        if item.artifact_id in ids or item.path in paths:
            raise ArtifactError("duplicate inventory artifact ID or path")
        if not DIGEST_RE.fullmatch(item.digest) or item.bytes < 0:
            raise ArtifactError("invalid artifact identity")
        ids.add(item.artifact_id)
        paths.add(item.path)
    if tuple(sorted(paths)) != tuple(item.path for item in inventory.artifacts):
        raise ArtifactError("inventory artifacts are not deterministically ordered")
    if (
        not DIGEST_RE.fullmatch(inventory.directory_digest)
        or not DIGEST_RE.fullmatch(inventory.content_digest)
        or inventory.directory_digest_policy_version != DIRECTORY_DIGEST_POLICY_VERSION
    ):
        raise ArtifactError("invalid or unsupported inventory digest")
    try:
        HiddenPolicy(inventory.hidden_policy)
    except ValueError as error:
        raise ArtifactError("invalid hidden-file policy") from error
    if inventory.artifact_count != len(inventory.artifacts):
        raise ArtifactError("artifact count mismatch")
    if inventory.total_required_bytes != sum(
        item.bytes for item in inventory.artifacts if item.required
    ) or inventory.total_optional_bytes != sum(
        item.bytes for item in inventory.artifacts if not item.required
    ):
        raise ArtifactError("inventory byte totals mismatch")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)
    except OSError as error:
        raise ArtifactError(f"unable to write inventory: {error}") from error
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("inventory")
    create.add_argument("--root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--inventory-id", default="inventory-artifacts")
    create.add_argument(
        "--artifact-type", choices=sorted(ARTIFACT_TYPES), default="other"
    )
    create.add_argument("--optional", action="append", default=[])
    create.add_argument("--exclude-hidden", action="store_true")
    create.add_argument("--overwrite", action="store_true")
    create.add_argument("--json", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--inventory", required=True, type=Path)
    verify.add_argument("--strict", action="store_true")
    verify.add_argument("--json", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        if options.command == "inventory":
            root = options.root.resolve(strict=True)
            output = options.output.resolve()
            excluded: list[str] = []
            try:
                excluded.append(output.relative_to(root).as_posix())
            except ValueError:
                pass
            inventory = inventory_tree(
                root,
                inventory_id=options.inventory_id,
                artifact_type=options.artifact_type,
                optional_paths=options.optional,
                hidden_policy=(
                    HiddenPolicy.EXCLUDE
                    if options.exclude_hidden
                    else HiddenPolicy.INCLUDE
                ),
                excluded_paths=excluded,
            )
            write_inventory(inventory, output, overwrite=options.overwrite)
            summary = {
                "artifact_count": inventory.artifact_count,
                "content_digest": inventory.content_digest,
                "directory_digest": inventory.directory_digest,
                "output": output.as_posix(),
                "state": "created",
            }
            code = 0
        else:
            inventory = load_inventory(options.inventory)
            result = verify_inventory(inventory, options.root, strict=options.strict)
            summary = result.to_dict()
            code = 0 if result.state is not VerificationState.INVALID else 1
    except (ArtifactError, OSError) as error:
        summary = {"state": "error", "valid": False, "error": str(error)}
        code = 2
    if options.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif options.command == "inventory" and code == 0:
        print(f"created inventory with {summary['artifact_count']} artifacts")
        print(f"directory digest: {summary['directory_digest']}")
        print(f"content digest: {summary['content_digest']}")
    elif options.command == "verify" and code in {0, 1}:
        print(f"verification: {summary['state']}")
        print(f"checked artifacts: {summary['checked_artifacts']}")
        for issue in summary["issues"]:
            suffix = f" ({issue['path']})" if issue["path"] else ""
            print(f"{issue['severity']}: {issue['code']}{suffix}")
    else:
        print(summary["error"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
