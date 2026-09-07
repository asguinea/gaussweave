from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from gaussweave.results.artifacts import (
    ArchiveMember,
    ArtifactDeclaration,
    ArtifactError,
    ArtifactPath,
    HiddenPolicy,
    VerificationState,
    build_inventory,
    directory_identity,
    hash_file,
    load_inventory,
    normalize_portable_path,
    validate_archive_members,
    verify_inventory,
    write_inventory,
)


def _files(root: Path, values: dict[str, bytes]) -> None:
    for name, content in values.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _inventory(root: Path, *, excluded: tuple[str, ...] = ()):
    return build_inventory(
        root,
        [
            ArtifactDeclaration("required", "other", "a.txt"),
            ArtifactDeclaration("optional", "table", "nested/b.txt", required=False),
        ],
        inventory_id="inventory-test",
        excluded_paths=excluded,
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_portable_paths_accept_nested_and_reject_unsafe(tmp_path: Path) -> None:
    _files(tmp_path, {"nested/file.txt": b"x"})
    path = ArtifactPath.resolve(tmp_path, "nested/file.txt", must_exist=True)
    assert path.portable == "nested/file.txt"
    assert path.local.is_relative_to(tmp_path.resolve())
    for value in ("", ".", "/x", "C:/x", "../x", "a/../x", r"a\..\x", "a//x"):
        with pytest.raises(ArtifactError):
            normalize_portable_path(value)


def test_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-artifact.txt"
    outside.write_text("outside")
    link = tmp_path / "link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("filesystem cannot create symlinks")
    with pytest.raises(ArtifactError):
        ArtifactPath.resolve(tmp_path, "link", must_exist=True)
    with pytest.raises(ArtifactError):
        directory_identity(tmp_path)


def test_stream_hash_exact_bytes_and_change(tmp_path: Path) -> None:
    target = tmp_path / "known"
    target.write_bytes(b"abc")
    first = hash_file(target, chunk_size=1)
    assert first.bytes == 3
    assert first.digest == f"sha256:{hashlib.sha256(b'abc').hexdigest()}"
    target.write_bytes(b"abd")
    assert hash_file(target).digest != first.digest
    with pytest.raises(ArtifactError):
        hash_file(target, expected_digest=first.digest)


def test_directory_digest_determinism_and_sensitivity(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _files(left, {"z": b"2", "a/x": b"1"})
    _files(right, {"a/x": b"1", "z": b"2"})
    baseline = directory_identity(left).digest
    assert baseline == directory_identity(right).digest
    (right / "z").write_bytes(b"3")
    assert baseline != directory_identity(right).digest
    (right / "z").rename(right / "y")
    assert baseline != directory_identity(right).digest
    (right / "extra").write_bytes(b"x")
    assert baseline != directory_identity(right).digest


def test_hidden_policy_is_explicit(tmp_path: Path) -> None:
    _files(tmp_path, {"visible": b"x", ".hidden": b"x"})
    assert len(directory_identity(tmp_path).entries) == 2
    assert (
        len(directory_identity(tmp_path, hidden_policy=HiddenPolicy.EXCLUDE).entries)
        == 1
    )


def test_inventory_totals_duplicates_and_roundtrip(tmp_path: Path) -> None:
    _files(tmp_path, {"a.txt": b"abc", "nested/b.txt": b"xy"})
    inventory = _inventory(tmp_path, excluded=("inventory.json",))
    assert inventory.artifact_count == 2
    assert inventory.total_required_bytes == 3
    assert inventory.total_optional_bytes == 2
    output = tmp_path / "inventory.json"
    write_inventory(inventory, output)
    assert output.read_bytes().endswith(b"\n")
    loaded = load_inventory(output)
    assert loaded == inventory
    assert loaded.content_digest == inventory.content_digest
    with pytest.raises(ArtifactError):
        write_inventory(inventory, output)
    duplicate = [
        ArtifactDeclaration("same", "other", "a.txt"),
        ArtifactDeclaration("same", "other", "nested/b.txt"),
    ]
    with pytest.raises(ArtifactError):
        build_inventory(tmp_path, duplicate, inventory_id="inventory-x")
    duplicate[1] = ArtifactDeclaration("different", "other", "a.txt")
    with pytest.raises(ArtifactError):
        build_inventory(tmp_path, duplicate, inventory_id="inventory-x")


def test_verification_states(tmp_path: Path) -> None:
    _files(tmp_path, {"a.txt": b"abc", "nested/b.txt": b"xy"})
    inventory = _inventory(tmp_path)
    assert verify_inventory(inventory, tmp_path).state is VerificationState.VALID
    (tmp_path / "nested/b.txt").unlink()
    result = verify_inventory(inventory, tmp_path)
    assert result.state is VerificationState.VALID_WITH_WARNINGS
    assert any(issue.code == "missing_optional" for issue in result.issues)
    assert (
        verify_inventory(inventory, tmp_path, strict=True).state
        is VerificationState.INVALID
    )


def test_verification_detects_mismatch_and_unexpected(tmp_path: Path) -> None:
    _files(tmp_path, {"a.txt": b"abc", "nested/b.txt": b"xy"})
    inventory = _inventory(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"changed")
    result = verify_inventory(inventory, tmp_path)
    assert {item.code for item in result.issues} >= {
        "digest_mismatch",
        "size_mismatch",
        "directory_digest_mismatch",
    }
    (tmp_path / "a.txt").write_bytes(b"abc")
    (tmp_path / "unexpected").write_bytes(b"x")
    result = verify_inventory(inventory, tmp_path, strict=True)
    assert any(item.code == "unexpected_artifact" for item in result.issues)


def test_missing_required_is_invalid(tmp_path: Path) -> None:
    _files(tmp_path, {"a.txt": b"abc", "nested/b.txt": b"xy"})
    inventory = _inventory(tmp_path)
    (tmp_path / "a.txt").unlink()
    result = verify_inventory(inventory, tmp_path)
    assert result.state is VerificationState.INVALID
    assert any(item.code == "missing_required" for item in result.issues)


def test_tampered_inventory_content_digest_is_rejected(tmp_path: Path) -> None:
    _files(tmp_path, {"a.txt": b"abc", "nested/b.txt": b"xy"})
    output = tmp_path / "inventory.json"
    write_inventory(_inventory(tmp_path, excluded=("inventory.json",)), output)
    output.write_text(
        output.read_text().replace('"artifact_count": 2', '"artifact_count": 3')
    )
    with pytest.raises(ArtifactError):
        load_inventory(output)


@pytest.mark.parametrize(
    "name", ["/abs", "C:/drive", "../escape", "a/../../escape", r"a\..\escape", "", "."]
)
def test_archive_unsafe_names(name: str) -> None:
    with pytest.raises(ArtifactError):
        validate_archive_members([ArchiveMember(name)])


def test_archive_safe_duplicates_and_links() -> None:
    assert validate_archive_members(
        [ArchiveMember("a/file"), ArchiveMember("a/link", "symlink", "file")]
    ) == ("a/file", "a/link")
    with pytest.raises(ArtifactError):
        validate_archive_members([ArchiveMember("a"), ArchiveMember("a")])
    with pytest.raises(ArtifactError):
        validate_archive_members([ArchiveMember("a/link", "symlink", "../../outside")])


def test_cpu_import_isolation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import gaussweave.results.artifacts; "
                "assert 'torch' not in sys.modules; "
                "assert 'gsplat' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
