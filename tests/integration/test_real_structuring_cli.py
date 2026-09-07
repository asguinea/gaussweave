from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gaussweave.cli.main import main


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["real-region", "extract", "--help"], "--dataset-root"),
        (["real-region", "register", "--help"], "--region-root"),
        (["real-region", "build-hybrid", "--help"], "--output"),
        (["real-region", "fit", "--help"], "real_struct_shared"),
        (["real-region", "evaluate", "--help"], "--hybrid-root"),
        (["real-region", "edit", "--help"], "--instance"),
        (["real-region", "validate-edit", "--help"], "path"),
    ],
)
def test_real_region_cli_help(
    arguments: list[str],
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 0
    assert expected in capsys.readouterr().out


def test_t25_cpu_validation_import_isolation() -> None:
    code = (
        "import json,sys; "
        "import gaussweave.real_structuring.models; "
        "import gaussweave.real_structuring.regions; "
        "import gaussweave.real_structuring.frames; "
        "print(json.dumps(sorted(set(sys.modules)&{'bpy','torch','gsplat'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_t13_controlled_evaluation_camera_in_fit_set_fails() -> None:
    from gaussweave.real_structuring import fitting

    assert not any(
        camera_id.startswith("cam-eval-") for camera_id in fitting.FITTING_CAMERA_IDS
    )


def test_remediation_cpu_import_isolation() -> None:
    code = (
        "import json,sys; "
        "import gaussweave.real_structuring.remediation; "
        "print(json.dumps(sorted(set(sys.modules)&{'bpy','torch','gsplat'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
