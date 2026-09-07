from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaussweave.real_structuring.editing import (
    BACKGROUND_COUNT,
    CANONICAL_COUNT,
    EDIT_ID,
    EDIT_VERSION,
    EDITED_ACTIVE_INSTANCE_IDS,
    EDITED_MATERIALIZED_TOTAL,
    RealEditRequest,
    _file_delta,
    _render_scientific_digest,
    inclusion_decision,
    privacy_issues,
    validate_active_instances,
)
from gaussweave.real_structuring.models import (
    StructuringError,
    detect_restricted_artifact_paths,
)


def _request() -> RealEditRequest:
    digest = "sha256:" + "1" * 64
    return RealEditRequest(
        edit_id=EDIT_ID,
        branch_id="gw-real-pilot",
        edit_version=EDIT_VERSION,
        operation="remove-instance",
        dataset_id="tnt-truck",
        scene_id="truck",
        source_pilot_version="gw-truck-panel-interiors-v2",
        region_version="truck-bed-panel-interiors-v2",
        source_representation_digest=digest,
        canonical_terminal_digest=digest,
        fixed_background_digest=digest,
        fixed_background_integrity_digest=digest,
        removed_instance_id="panel-middle",
        active_instance_ids=EDITED_ACTIVE_INSTANCE_IDS,
        camera_policy_digest=digest,
        expected_instance_count=2,
        stored_gaussian_count=BACKGROUND_COUNT + CANONICAL_COUNT,
        materialized_gaussian_count=EDITED_MATERIALIZED_TOTAL,
        changed_field_paths=(
            "active_instance_ids",
            "expected_instance_count",
            "materialized_gaussian_count",
        ),
    )


def test_t1_edit_identity_is_stable_and_excludes_paths_and_timings() -> None:
    first = _request()
    second = _request()
    assert first.scientific_digest == second.scientific_digest
    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert "output" not in serialized
    assert "latency" not in serialized


def test_t1_render_scientific_digest_excludes_timing_and_gpu_peaks() -> None:
    record = {
        "camera_id": "cam-eval-000001",
        "source_render_latency_seconds": 1.0,
        "edited_render_latency_seconds": 2.0,
        "gpu_peak_allocated_bytes": 3,
        "gpu_peak_reserved_bytes": 4,
        "source_render_digest": "sha256:" + "a" * 64,
    }
    first = _render_scientific_digest([record])  # type: ignore[list-item]
    record["source_render_latency_seconds"] = 100.0
    record["gpu_peak_reserved_bytes"] = 400
    second = _render_scientific_digest([record])  # type: ignore[list-item]
    assert first == second


def test_t2_t13_active_instance_set_is_exact_and_unique() -> None:
    assert validate_active_instances(EDITED_ACTIVE_INSTANCE_IDS) is None
    for invalid in (
        ("panel-rear", "panel-middle"),
        ("panel-rear", "panel-rear"),
        ("panel-front",),
    ):
        with pytest.raises(StructuringError):
            validate_active_instances(invalid)


def test_t7_request_enforces_edited_counts() -> None:
    request = _request()
    values = request.to_dict()
    values.pop("scientific_digest")
    values["materialized_gaussian_count"] += 1
    with pytest.raises(StructuringError, match="count"):
        RealEditRequest(**values)


def test_t9_serialization_delta_matches_direct_byte_comparison(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    edited = tmp_path / "edited"
    source.mkdir()
    edited.mkdir()
    (source / "same.bin").write_bytes(b"same")
    (edited / "same.bin").write_bytes(b"same")
    (source / "changed.json").write_bytes(b'{"active":3}')
    (edited / "changed.json").write_bytes(b'{"active":2}')
    (edited / "added.json").write_bytes(b"edit")
    delta = _file_delta(source, edited)
    assert [record["path"] for record in delta["changed_files"]] == [
        "added.json",
        "changed.json",
    ]
    assert delta["unchanged_files"] == ["same.bin"]
    assert delta["changed_serialized_bytes"] > 0
    assert delta["added_bytes"] >= 4


def test_t17_t20_permission_is_independent_and_requires_written_evidence() -> None:
    conditional = inclusion_decision(
        visual_decision="credible_with_limitations",
        localization_passed=True,
        artifact_complete=True,
        permission_evidence=None,
    )
    assert conditional["decision"] == "include_if_cleared"
    assert conditional["release_use_allowed"] is False
    attempted = inclusion_decision(
        visual_decision="credible",
        localization_passed=True,
        artifact_complete=True,
        permission_evidence={"status": "requested"},
    )
    assert attempted["release_use_allowed"] is False


def test_t15_t20_visual_and_localization_gates_are_mandatory() -> None:
    rejected = inclusion_decision(
        visual_decision="not_credible",
        localization_passed=True,
        artifact_complete=True,
        permission_evidence=None,
    )
    assert rejected["decision"] == "reject_visual"
    held = inclusion_decision(
        visual_decision="credible",
        localization_passed=False,
        artifact_complete=True,
        permission_evidence=None,
    )
    assert held["decision"] == "hold_for_later"


def test_t18_restricted_staging_detector_catches_visual_and_model_payloads() -> None:
    staged = [
        "reports/evaluation/gw/truck-edit/panel.png",
        "local/model.ply",
        "reports/evaluation/gw/truck-edit/edit-summary.json",
    ]
    assert detect_restricted_artifact_paths(staged) == staged[:2]


@pytest.mark.parametrize(
    "text,code",
    [
        (r'{"path":"C:\\Users\\person\\Truck"}', "absolute_windows_path"),
        ('{"path":"/home/person/Truck"}', "absolute_home_path"),
        ('{"contact":"person@example.org"}', "email_address"),
        ('{"hostname":"workstation"}', "machine_hostname_field"),
    ],
)
def test_t19_privacy_scan_rejects_nonportable_identifiers(
    text: str,
    code: str,
) -> None:
    assert code in privacy_issues(text)


def test_cpu_edit_and_panel_imports_do_not_load_heavy_modules() -> None:
    import subprocess
    import sys

    code = (
        "import json,sys;"
        "import gaussweave.real_structuring.editing;"
        "print(json.dumps(sorted(set(sys.modules)&{'torch','gsplat','bpy'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
