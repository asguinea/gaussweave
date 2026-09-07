"""CPU-only tests for the initial package import contract."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SUBPACKAGES = (
    "accounting",
    "baselines",
    "cli",
    "config",
    "data",
    "editing",
    "experiments",
    "gaussians",
    "grammar",
    "induction",
    "metrics",
    "rendering",
    "results",
    "runtime",
)

HEAVY_MODULE_PREFIXES = (
    "torch",
    "gsplat",
    "bpy",
    "blender",
    "pycolmap",
    "gaussweave_native",
)


def test_package_import_exposes_version() -> None:
    import gaussweave

    assert isinstance(gaussweave.__version__, str)
    assert gaussweave.__version__


def test_architectural_subpackages_import() -> None:
    for name in SUBPACKAGES:
        module = importlib.import_module(f"gaussweave.{name}")
        assert module.__name__ == f"gaussweave.{name}"


@pytest.mark.integration
def test_fresh_import_has_no_heavy_optional_side_effects() -> None:
    script = """
import json
import sys
import gaussweave

prefixes = (
    "torch",
    "gsplat",
    "bpy",
    "blender",
    "pycolmap",
    "gaussweave_native",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
)
print(json.dumps({"version": gaussweave.__version__, "loaded": loaded}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["version"]
    assert result["loaded"] == []
