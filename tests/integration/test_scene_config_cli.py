"""scene configuration unified CLI and lightweight import integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.cli]

ROOT = Path(__file__).parents[2]
FACADE = ROOT / "configs" / "scenes" / "facade-development-s101.json"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gaussweave.cli", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_unified_cli_validates_scene_as_structured_json() -> None:
    result = _run("scene", "validate", str(FACADE), "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["family"] == "facade"
    assert payload["scientific_digest"].startswith("sha256:")
    assert set(payload["derived_seeds"]) == {
        "geometry",
        "materials",
        "lighting",
        "cameras",
        "occluders",
        "edits",
        "annotations",
    }


def test_unified_cli_human_summary_includes_family_digest_and_seeds() -> None:
    result = _run("scene", "validate", str(FACADE))

    assert result.returncode == 0, result.stderr
    assert "family=facade" in result.stdout
    assert "scientific_digest=sha256:" in result.stdout
    assert "derived_seeds=" in result.stdout


def test_unified_cli_resolves_scene_and_writes_identity_outputs(
    tmp_path: Path,
) -> None:
    result = _run(
        "scene",
        "resolve",
        "--config",
        str(FACADE),
        "--output",
        str(tmp_path / "resolved.json"),
        "--canonical-output",
        str(tmp_path / "canonical.json"),
        "--digest-output",
        str(tmp_path / "digests.json"),
        "--repository-root",
        str(ROOT),
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert (tmp_path / "resolved.json").is_file()
    assert (tmp_path / "canonical.json").is_file()
    assert (tmp_path / "digests.json").is_file()

    repeated = _run(
        "scene",
        "resolve",
        "--config",
        str(FACADE),
        "--output",
        str(tmp_path / "resolved.json"),
        "--json",
    )
    assert repeated.returncode == 3
    assert "refusing to overwrite" in repeated.stdout


def test_unified_cli_returns_validation_exit_for_invalid_scene(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")

    result = _run("scene", "validate", str(invalid), "--json")

    assert result.returncode == 3
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert payload["exit_code"] == 3


def test_scene_import_has_no_heavy_optional_side_effects() -> None:
    script = """
import json
import sys
import gaussweave.data.scene_config
prefixes = ("torch", "gsplat", "bpy", "blender", "pycolmap")
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
