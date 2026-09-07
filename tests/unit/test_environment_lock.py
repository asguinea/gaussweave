"""Static reproducibility checks that do not import ML dependencies."""

from __future__ import annotations

from pathlib import Path


def test_gpu_environment_lock_artifacts_are_present() -> None:
    root = Path(__file__).parents[2]
    for relative in (
        "uv.lock",
        "environment/wsl-qualification.json",
        "environment/wsl-system-tools.json",
        "scripts/bootstrap_wsl_env.sh",
        "scripts/capture_wsl_environment.sh",
        "scripts/verify_gpu_env.py",
    ):
        assert (root / relative).is_file(), relative


def test_lock_and_manifests_have_no_personal_windows_path() -> None:
    root = Path(__file__).parents[2]
    for relative in (
        "uv.lock",
        "environment/wsl-qualification.json",
        "environment/wsl-system-tools.json",
    ):
        text = (root / relative).read_text(encoding="utf-8").lower()
        assert "c:\\users\\" not in text
        assert "/home/gaussweave/" not in text
