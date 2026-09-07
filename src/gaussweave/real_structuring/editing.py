"""Deterministic local-only instance removal for the accepted Truck refit.

The CPU contracts and validators in this module do not import PyTorch or
``gsplat``. Heavy dependencies are loaded only while rendering the requested
local evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from gaussweave.config.resolution import content_digest
from gaussweave.real_structuring.background import background_integrity
from gaussweave.real_structuring.models import (
    INSTANCE_IDS,
    StructuringError,
    atomic_json,
    load_object,
    sha256_file,
)

BRANCH_ID = "gw-real-pilot"
EDIT_VERSION = "gw-truck-remove-middle-panel-v1"
EDIT_ID = "truck-remove-middle-panel-interior-v1"
DATASET_ID = "tnt-truck"
REGION_VERSION = "truck-bed-panel-interiors-v2"
SOURCE_PILOT_VERSION = "gw-truck-panel-interiors-v2"
METHOD_ID = "real_struct_residual_sh1_q8"
REMOVED_INSTANCE_ID = "panel-middle"
SOURCE_ACTIVE_INSTANCE_IDS = INSTANCE_IDS
EDITED_ACTIVE_INSTANCE_IDS = ("panel-rear", "panel-front")
EVALUATION_CAMERA_IDS = (
    "cam-eval-000001",
    "cam-eval-000033",
    "cam-eval-000065",
    "cam-eval-000097",
    "cam-eval-000129",
    "cam-eval-000161",
    "cam-eval-000193",
    "cam-eval-000249",
)
HERO_CAMERA_ID = "cam-eval-000249"
BACKGROUND_COUNT = 2_533_367
CANONICAL_COUNT = 2_953
SOURCE_MATERIALIZED_CANONICAL = 8_859
SOURCE_MATERIALIZED_TOTAL = 2_542_226
EDITED_MATERIALIZED_CANONICAL = 5_906
EDITED_MATERIALIZED_TOTAL = 2_539_273
CHANGE_THRESHOLD = 1.0 / 255.0
OUTSIDE_TOLERANCE = 1.0 / 255.0
FOOTPRINT_ALPHA_THRESHOLD = 1e-4
SUPPORT_ALPHA_THRESHOLD = 1e-8
UNCERTAINTY_DILATION_PIXELS = 4


class ChangedArtifactRecord(TypedDict):
    """One exact direct-file comparison in the representation delta."""

    path: str
    source_bytes: int
    edited_bytes: int
    changed_byte_positions_plus_length_delta: int
    added_bytes: int
    removed_bytes: int
    source_sha256: str
    edited_sha256: str


def inclusion_decision(
    *,
    visual_decision: str,
    localization_passed: bool,
    artifact_complete: bool,
    permission_evidence: object | None,
) -> dict[str, Any]:
    """Combine technical, visual, and third-party permission release gates."""

    permitted_visual = visual_decision in {"credible", "credible_with_limitations"}
    if visual_decision == "not_credible":
        decision = "reject_visual"
    elif permitted_visual and localization_passed and artifact_complete:
        decision = "include_if_cleared"
    else:
        decision = "hold_for_later"
    permission_cleared = (
        isinstance(permission_evidence, dict)
        and permission_evidence.get("status") == "written_permission_received"
        and str(permission_evidence.get("evidence_digest", "")).startswith("sha256:")
    )
    return {
        "decision": decision,
        "visual_gate": permitted_visual,
        "localization_gate": localization_passed,
        "artifact_complete_gate": artifact_complete,
        "permission_gate": permission_cleared,
        "release_use_allowed": decision == "include_if_cleared" and permission_cleared,
        "permission_evidence": permission_evidence,
    }


class PerViewRenderResult(TypedDict):
    """Portable local render evidence for one frozen evaluation camera."""

    camera_id: str
    image_name: str
    width: int
    height: int
    source_render_digest: str
    edited_render_digest: str
    source_alpha_digest: str
    edited_alpha_digest: str
    source_depth_digest: str
    edited_depth_digest: str
    source_render_latency_seconds: float
    edited_render_latency_seconds: float
    materialized_source_gaussian_count: int
    materialized_edited_gaussian_count: int
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int
    complete: bool
    warnings: list[str]


class GeometricChangeMaskRecord(TypedDict):
    """Portable digest and pixel counts for geometry-derived local masks."""

    camera_id: str
    width: int
    height: int
    removed_footprint_digest: str
    conservative_support_digest: str
    uncertainty_band_digest: str
    stable_outside_digest: str
    background_digest: str
    removed_footprint_pixels: int
    conservative_support_pixels: int
    uncertainty_band_pixels: int
    stable_outside_pixels: int
    background_pixels: int
    generation_parameters: dict[str, Any]


class ImageChangeRecord(TypedDict):
    """Image-space localization diagnostic without edited photographic truth."""

    camera_id: str
    changed_pixel_count: int
    changed_pixel_fraction: float
    maximum_absolute_rgb_difference: float
    mean_absolute_rgb_change: float
    changed_pixels_inside_geometric_support: int
    changed_pixels_outside_geometric_support: int
    stable_outside_region_pixel_count: int
    maximum_outside_region_difference: float
    outside_region_tolerance: float
    outside_region_stable: bool


class UnchangedRegionValidation(TypedDict):
    """Aggregate stability result outside every conservative edit support."""

    camera_count: int
    all_outside_regions_stable: bool
    maximum_outside_region_difference: float
    tolerance: float
    changed_pixels_outside_support: int


class QualitativeVisualPackage(TypedDict):
    """Traceability record for the local hero sequence and fixed crop."""

    camera_id: str
    image_name: str
    crop_xyxy: list[int]
    image_dimensions: list[int]
    selection_policy: str
    instance_overlay_masks: dict[str, str]


class EditEvidenceRecord(TypedDict):
    """Complete top-level edit contract shared by portable evidence."""

    edit_id: str
    source_pilot_version: str
    region_version: str
    source_representation_digest: str
    canonical_terminal_digest: str
    fixed_background_digest: str
    removed_instance_id: str
    active_instance_ids: list[str]
    camera_policy_digest: str
    stored_gaussian_count: int
    materialized_gaussian_count: int
    changed_field_paths: list[str]
    complete_bytes: int
    changed_serialized_bytes: int
    edit_request_bytes: int
    status: str
    warnings: list[str]
    limitations: list[str]


class ReleaseUseStatus(TypedDict):
    """Release gate for local restricted evidence."""

    status: str
    release_use_allowed: bool
    permission_evidence: None
    source_image_use: str
    derived_render_use: str


@dataclass(frozen=True)
class RealEditRequest:
    """Stable representation-level request for removing the middle instance."""

    edit_id: str
    branch_id: str
    edit_version: str
    operation: str
    dataset_id: str
    scene_id: str
    source_pilot_version: str
    region_version: str
    source_representation_digest: str
    canonical_terminal_digest: str
    fixed_background_digest: str
    fixed_background_integrity_digest: str
    removed_instance_id: str
    active_instance_ids: tuple[str, ...]
    camera_policy_digest: str
    expected_instance_count: int
    stored_gaussian_count: int
    materialized_gaussian_count: int
    changed_field_paths: tuple[str, ...]
    status: str = "validated"
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = (
        "manual_oracle_annotation",
        "local_only_real_benchmark_demonstration",
        "materialized_rendering",
        "no_edited_photographic_ground_truth",
        "not_cleared_for_release",
    )

    def __post_init__(self) -> None:
        if self.edit_id != EDIT_ID or self.edit_version != EDIT_VERSION:
            raise StructuringError("edit identity differs from the frozen request")
        if self.operation != "remove-instance":
            raise StructuringError("only remove-instance is supported")
        if self.dataset_id != DATASET_ID or self.region_version != REGION_VERSION:
            raise StructuringError("edit source identity mismatch")
        validate_active_instances(self.active_instance_ids)
        if (
            self.removed_instance_id != REMOVED_INSTANCE_ID
            or self.expected_instance_count != 2
        ):
            raise StructuringError("only panel-middle may be removed")
        if (
            self.stored_gaussian_count != BACKGROUND_COUNT + CANONICAL_COUNT
            or self.materialized_gaussian_count != EDITED_MATERIALIZED_TOTAL
        ):
            raise StructuringError("edited count contract mismatch")
        for digest in (
            self.source_representation_digest,
            self.canonical_terminal_digest,
            self.fixed_background_digest,
            self.fixed_background_integrity_digest,
            self.camera_policy_digest,
        ):
            if not digest.startswith("sha256:"):
                raise StructuringError("edit identity requires SHA-256 digests")

    @property
    def scientific_digest(self) -> str:
        """Digest scientific fields only; paths and timings are absent."""

        return content_digest(cast(Any, asdict(self)))

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic serialized request."""

        value = asdict(self)
        value["scientific_digest"] = self.scientific_digest
        return value


@dataclass(frozen=True)
class SourceHybridRepresentation:
    """Accepted source representation identity and counts."""

    representation_digest: str
    canonical_terminal_digest: str
    fixed_background_digest: str
    fixed_background_integrity_digest: str
    active_instance_ids: tuple[str, ...]
    stored_gaussian_count: int
    materialized_gaussian_count: int
    complete_bytes: int


@dataclass(frozen=True)
class EditedActiveInstanceSet:
    """Authoritative ordered active-instance specification."""

    active_instance_ids: tuple[str, ...]
    removed_instance_id: str
    expected_instance_count: int

    def __post_init__(self) -> None:
        validate_active_instances(self.active_instance_ids)
        if self.removed_instance_id != REMOVED_INSTANCE_ID:
            raise StructuringError("unexpected removed instance")
        if self.expected_instance_count != len(self.active_instance_ids):
            raise StructuringError("active-instance count mismatch")

    @property
    def scientific_digest(self) -> str:
        return content_digest(cast(Any, asdict(self)))


@dataclass(frozen=True)
class EditedHybridRepresentation:
    """Derived representation identity, counts, bytes, and provenance."""

    representation_digest: str
    parent_representation_digest: str
    edit_request_digest: str
    active_instance_set_digest: str
    canonical_terminal_digest: str
    fixed_background_digest: str
    fixed_background_integrity_digest: str
    active_instance_ids: tuple[str, ...]
    stored_gaussian_count: int
    materialized_gaussian_count: int
    complete_bytes: int


@dataclass(frozen=True)
class RemovedInstance:
    """Removed instance and independently validated structural effect."""

    instance_id: str
    source_position: int
    canonical_gaussian_count: int
    materialized_count_delta: int
    source_core_reintroduced_by_background: bool


@dataclass(frozen=True)
class MaterializationResult:
    """Source and edited stored/materialized count evidence."""

    fixed_explicit_complement: int
    canonical_stored: int
    source_active_instances: int
    source_canonical_materialized: int
    source_hybrid_materialized: int
    edited_active_instances: int
    edited_canonical_materialized: int
    edited_hybrid_materialized: int


def validate_active_instances(active_instance_ids: tuple[str, ...]) -> None:
    """Require the exact rear/front order with no duplicate or middle instance."""

    if active_instance_ids != EDITED_ACTIVE_INSTANCE_IDS:
        if len(active_instance_ids) != len(set(active_instance_ids)):
            raise StructuringError("duplicate retained instance")
        if REMOVED_INSTANCE_ID in active_instance_ids:
            raise StructuringError("panel-middle remains active")
        raise StructuringError("edited active instances must be rear then front")


def _hybrid_root(refit_root: Path) -> Path:
    candidate = refit_root / "hybrid"
    root = candidate if (candidate / "hybrid.json").is_file() else refit_root
    if not (root / "hybrid.json").is_file():
        raise StructuringError("accepted refit root does not contain hybrid.json")
    return root.resolve()


def _camera_records(hybrid_root: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = load_object(hybrid_root / "hybrid.json")
    dataset_root = Path(str(manifest["local_dataset_root"]))
    split_path = dataset_root / "annotations" / "evaluation-split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    by_id = {
        str(record["camera_id"]): cast(dict[str, Any], record)
        for record in split["cameras"]
    }
    try:
        records = [by_id[camera_id] for camera_id in EVALUATION_CAMERA_IDS]
    except KeyError as error:
        raise StructuringError("frozen evaluation camera is missing") from error
    digest = content_digest(
        cast(
            Any,
            {
                "policy": "eight validated evaluation cameras in deterministic order",
                "records": records,
            },
        )
    )
    return records, digest


def build_edit_request(refit_root: Path) -> RealEditRequest:
    """Build and validate the path-independent frozen edit identity."""

    hybrid_root = _hybrid_root(refit_root)
    hybrid = load_object(hybrid_root / "hybrid.json")
    canonical = load_object(hybrid_root / "canonical" / "metadata.json")
    background = background_integrity(hybrid_root)
    _, camera_digest = _camera_records(hybrid_root)
    if hybrid.get("region_version") != REGION_VERSION:
        raise StructuringError("edit requires truck-bed-panel-interiors-v2")
    counts = hybrid.get("counts", {})
    expected_counts = {
        "fixed_explicit_background": BACKGROUND_COUNT,
        "stored_canonical_panel": CANONICAL_COUNT,
        "materialized_canonical_panel": SOURCE_MATERIALIZED_CANONICAL,
        "hybrid_materialized": SOURCE_MATERIALIZED_TOTAL,
    }
    if any(int(counts.get(key, -1)) != value for key, value in expected_counts.items()):
        raise StructuringError("accepted source count contract mismatch")
    return RealEditRequest(
        edit_id=EDIT_ID,
        branch_id=BRANCH_ID,
        edit_version=EDIT_VERSION,
        operation="remove-instance",
        dataset_id=DATASET_ID,
        scene_id="real-tnt-truck",
        source_pilot_version=str(hybrid["pilot_version"]),
        region_version=str(hybrid["region_version"]),
        source_representation_digest=str(hybrid["scientific_digest"]),
        canonical_terminal_digest=str(canonical["scientific_digest"]),
        fixed_background_digest=str(background["scientific_digest"]),
        fixed_background_integrity_digest=str(background["integrity_digest"]),
        removed_instance_id=REMOVED_INSTANCE_ID,
        active_instance_ids=EDITED_ACTIVE_INSTANCE_IDS,
        camera_policy_digest=camera_digest,
        expected_instance_count=2,
        stored_gaussian_count=BACKGROUND_COUNT + CANONICAL_COUNT,
        materialized_gaussian_count=EDITED_MATERIALIZED_TOTAL,
        changed_field_paths=(
            "active_instances.active_instance_ids",
            "active_instances.expected_instance_count",
            "binding.terminal.instance_ids",
            "representation.active_instance_count",
            "representation.materialized_gaussian_count",
            "representation.derivation",
        ),
    )


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_payloads(hybrid_root: Path, target: Path) -> None:
    for directory in ("background", "canonical"):
        for source in sorted((hybrid_root / directory).rglob("*")):
            if source.is_file():
                _link_or_copy(source, target / source.relative_to(hybrid_root))
    _link_or_copy(
        hybrid_root / "instance-transforms.json",
        target / "instance-transforms.json",
    )
    shared_summary = load_object(
        hybrid_root / "fits" / "real_struct_shared" / "fit-summary.json"
    )
    checkpoint = cast(dict[str, Any], shared_summary["checkpoint"])
    _link_or_copy(
        hybrid_root / "fits" / "real_struct_shared" / str(checkpoint["path"]),
        target / "appearance" / "shared-sh.f32",
    )
    atomic_json(
        target / "appearance" / "metadata.json",
        {
            "method_id": "real_struct_shared",
            "checkpoint": {
                **checkpoint,
                "path": "shared-sh.f32",
            },
            "source_fit_scientific_digest": shared_summary["scientific_digest"],
        },
    )
    residual_root = hybrid_root / "fits" / METHOD_ID
    for name in ("residuals.i8", "residuals.json"):
        _link_or_copy(residual_root / name, target / "residuals" / name)


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _representation_manifest(
    *,
    request: RealEditRequest,
    active_instance_ids: tuple[str, ...],
    source: bool,
) -> dict[str, Any]:
    active_count = len(active_instance_ids)
    materialized = BACKGROUND_COUNT + active_count * CANONICAL_COUNT
    value: dict[str, Any] = {
        "format_version": "gaussweave-real-active-instance-edit-v1",
        "representation_id": (
            "gw-truck-panel-interiors-v2-source" if source else EDIT_VERSION
        ),
        "parent_representation_digest": request.source_representation_digest,
        "edit_request_digest": None if source else request.scientific_digest,
        "region_version": REGION_VERSION,
        "method_id": METHOD_ID,
        "active_instance_ids": list(active_instance_ids),
        "active_instance_count": active_count,
        "removed_instance_id": None if source else REMOVED_INSTANCE_ID,
        "fixed_explicit_complement_count": BACKGROUND_COUNT,
        "canonical_stored_count": CANONICAL_COUNT,
        "stored_gaussian_count": BACKGROUND_COUNT + CANONICAL_COUNT,
        "materialized_canonical_count": active_count * CANONICAL_COUNT,
        "materialized_gaussian_count": materialized,
        "canonical_terminal_digest": request.canonical_terminal_digest,
        "fixed_background_digest": request.fixed_background_digest,
        "fixed_background_integrity_digest": request.fixed_background_integrity_digest,
        "camera_policy_digest": request.camera_policy_digest,
        "materialization": "reference_full_tensor_materialization",
        "ordering": "background source order then frozen active instance order",
        "derivation": (
            None
            if source
            else {
                "operation": "remove-instance",
                "removed_instance_id": REMOVED_INSTANCE_ID,
                "changed_field_paths": list(request.changed_field_paths),
                "optimization_performed": False,
            }
        ),
        "warnings": [],
        "limitations": list(request.limitations),
    }
    value["scientific_digest"] = content_digest(cast(Any, value))
    return value


def _serialize_representation(
    *,
    hybrid_root: Path,
    output: Path,
    request: RealEditRequest,
    active_instance_ids: tuple[str, ...],
    source: bool,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    _copy_payloads(hybrid_root, output)
    transforms = load_object(output / "instance-transforms.json")
    by_id = {
        str(record["instance_id"]): record
        for record in cast(list[dict[str, Any]], transforms["instances"])
    }
    if tuple(by_id) != INSTANCE_IDS:
        raise StructuringError("source instance transforms are reordered or missing")
    active = {
        "format_version": "gaussweave-active-instance-set-v1",
        "active_instance_ids": list(active_instance_ids),
        "expected_instance_count": len(active_instance_ids),
        "removed_instance_id": None if source else REMOVED_INSTANCE_ID,
        "transform_record_digests": {
            instance_id: content_digest(cast(Any, by_id[instance_id]))
            for instance_id in active_instance_ids
        },
    }
    active["scientific_digest"] = content_digest(cast(Any, active))
    atomic_json(output / "active-instances.json", active)
    accepted_binding = load_object(hybrid_root / "binding.json")
    binding = {
        **accepted_binding,
        "binding_id": (
            f"{accepted_binding['binding_id']}-source-active"
            if source
            else f"{accepted_binding['binding_id']}-remove-middle"
        ),
        "terminal": {
            **cast(dict[str, Any], accepted_binding["terminal"]),
            "instance_ids": list(active_instance_ids),
        },
        "active_instance_set": "active-instances.json",
        "edit_request_digest": None if source else request.scientific_digest,
    }
    binding.pop("scientific_digest", None)
    binding["scientific_digest"] = content_digest(cast(Any, binding))
    atomic_json(output / "binding.json", binding)
    manifest = _representation_manifest(
        request=request,
        active_instance_ids=active_instance_ids,
        source=source,
    )
    atomic_json(output / "representation.json", manifest)
    return manifest


def _file_delta(source_root: Path, edited_root: Path) -> dict[str, Any]:
    source_files = {
        path.relative_to(source_root).as_posix(): path
        for path in source_root.rglob("*")
        if path.is_file()
    }
    edited_files = {
        path.relative_to(edited_root).as_posix(): path
        for path in edited_root.rglob("*")
        if path.is_file()
    }
    records: list[ChangedArtifactRecord] = []
    for relative in sorted(set(source_files) | set(edited_files)):
        source_payload = (
            source_files[relative].read_bytes() if relative in source_files else b""
        )
        edited_payload = (
            edited_files[relative].read_bytes() if relative in edited_files else b""
        )
        shared = min(len(source_payload), len(edited_payload))
        differing = sum(
            source_payload[index] != edited_payload[index] for index in range(shared)
        )
        changed = differing + abs(len(source_payload) - len(edited_payload))
        if changed:
            records.append(
                {
                    "path": relative,
                    "source_bytes": len(source_payload),
                    "edited_bytes": len(edited_payload),
                    "changed_byte_positions_plus_length_delta": changed,
                    "added_bytes": max(0, len(edited_payload) - len(source_payload)),
                    "removed_bytes": max(0, len(source_payload) - len(edited_payload)),
                    "source_sha256": hashlib.sha256(source_payload).hexdigest(),
                    "edited_sha256": hashlib.sha256(edited_payload).hexdigest(),
                }
            )
    changed_paths = {record["path"] for record in records}
    unchanged = sorted((set(source_files) & set(edited_files)) - changed_paths)
    return {
        "byte_delta_policy": (
            "aligned differing byte positions plus absolute file-length delta"
        ),
        "changed_files": records,
        "changed_file_count": len(records),
        "changed_serialized_bytes": sum(
            record["changed_byte_positions_plus_length_delta"] for record in records
        ),
        "added_bytes": sum(record["added_bytes"] for record in records),
        "removed_bytes": sum(record["removed_bytes"] for record in records),
        "unchanged_files": unchanged,
        "unchanged_file_count": len(unchanged),
    }


def _hash_tree(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _validate_declared_payloads(root: Path) -> None:
    metadata = load_object(root / "metadata.json")
    for record in cast(dict[str, dict[str, Any]], metadata["files"]).values():
        payload = root / str(record["path"])
        if payload.stat().st_size != int(record["bytes"]) or sha256_file(
            payload
        ) != str(record["sha256"]):
            raise StructuringError(
                f"declared payload integrity mismatch: {payload.name}"
            )


def _validate_fixed_content(
    *,
    source_root: Path,
    edited_root: Path,
    hybrid_root: Path,
    request: RealEditRequest,
) -> dict[str, Any]:
    source_canonical = _hash_tree(source_root / "canonical")
    edited_canonical = _hash_tree(edited_root / "canonical")
    source_background = _hash_tree(source_root / "background")
    edited_background = _hash_tree(edited_root / "background")
    if source_canonical != edited_canonical:
        raise StructuringError("canonical terminal changed during edit")
    if source_background != edited_background:
        raise StructuringError("fixed background changed during edit")
    _validate_declared_payloads(source_root / "canonical")
    _validate_declared_payloads(edited_root / "canonical")
    canonical_meta = load_object(edited_root / "canonical" / "metadata.json")
    background = background_integrity(edited_root)
    if canonical_meta["scientific_digest"] != request.canonical_terminal_digest:
        raise StructuringError("canonical scientific digest mismatch")
    if (
        background["scientific_digest"] != request.fixed_background_digest
        or background["integrity_digest"] != request.fixed_background_integrity_digest
    ):
        raise StructuringError("background scientific or integrity digest mismatch")
    source_transforms = (source_root / "instance-transforms.json").read_bytes()
    edited_transforms = (edited_root / "instance-transforms.json").read_bytes()
    if source_transforms != edited_transforms:
        raise StructuringError("retained transforms changed")
    source_residuals = _hash_tree(source_root / "residuals")
    edited_residuals = _hash_tree(edited_root / "residuals")
    if source_residuals != edited_residuals:
        raise StructuringError("retained SH1 q8 residual payload changed")
    camera_records, camera_digest = _camera_records(hybrid_root)
    if camera_digest != request.camera_policy_digest:
        raise StructuringError("camera records changed")
    return {
        "canonical_byte_identical": True,
        "canonical_scientific_digest": request.canonical_terminal_digest,
        "background_byte_identical": True,
        "background_scientific_digest": request.fixed_background_digest,
        "background_integrity_digest": request.fixed_background_integrity_digest,
        "instance_transforms_byte_identical": True,
        "retained_transform_ids": list(EDITED_ACTIVE_INSTANCE_IDS),
        "residual_payload_byte_identical": True,
        "camera_records_unchanged": True,
        "camera_count": len(camera_records),
        "camera_policy_digest": camera_digest,
    }


def _validate_middle_absent_from_background(hybrid_root: Path) -> bool:
    manifest = load_object(hybrid_root / "hybrid.json")
    region_root = Path(str(manifest["local_region_root"]))
    ownership = (region_root / "ownership.u8").read_bytes()
    index_payload = (region_root / "background-indices.u32").read_bytes()
    indexes = array("I")
    indexes.frombytes(index_payload)
    if indexes.itemsize != 4:
        raise StructuringError("platform u32 width mismatch")
    if sys.byteorder != "little":
        indexes.byteswap()
    middle_code = int(
        load_object(region_root / "ownership.json")["status_codes"]["panel_middle_core"]
    )
    return not any(ownership[index] == middle_code for index in indexes)


def _tensor_digest(value: Any) -> str:
    payload = value.detach().contiguous().to("cpu").numpy().tobytes(order="C")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_rgb(path: Path, value: Any) -> None:
    from PIL import Image

    torch = _torch()
    pixels = value.detach().clamp(0, 1).mul(255).round().to("cpu", torch.uint8)
    Image.frombytes(
        "RGB",
        (int(pixels.shape[1]), int(pixels.shape[0])),
        bytes(pixels.flatten()),
    ).save(path)


def _write_mask(path: Path, mask: Any) -> dict[str, Any]:
    from PIL import Image

    torch = _torch()
    pixels = mask.detach().to("cpu", torch.uint8).mul(255)
    Image.frombytes(
        "L",
        (int(pixels.shape[1]), int(pixels.shape[0])),
        bytes(pixels.flatten()),
    ).save(path)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_count": int(mask.sum().item()),
    }


def _dilate(mask: Any, pixels: int) -> Any:
    torch = _torch()
    if pixels <= 0:
        return mask.bool()
    return torch.nn.functional.max_pool2d(
        mask.to(torch.float32)[None, None],
        pixels * 2 + 1,
        stride=1,
        padding=pixels,
    )[0, 0].bool()


def _render_evaluation(
    *,
    hybrid_root: Path,
    output: Path,
    camera_filter: set[str] | None,
) -> tuple[
    list[PerViewRenderResult],
    list[GeometricChangeMaskRecord],
    list[ImageChangeRecord],
    dict[str, Any],
]:
    from gaussweave.gaussians.real_explicit import _camera_from_record
    from gaussweave.real_structuring.compositing import render_fields
    from gaussweave.real_structuring.evaluation import _q8_tensor
    from gaussweave.real_structuring.fitting import _load_shared
    from gaussweave.real_structuring.hybrid import HybridTensors

    torch = _torch()
    camera_records, _ = _camera_records(hybrid_root)
    if camera_filter is not None:
        unknown = camera_filter - set(EVALUATION_CAMERA_IDS)
        if unknown:
            raise StructuringError(f"unknown debug cameras: {sorted(unknown)}")
        camera_records = [
            record for record in camera_records if record["camera_id"] in camera_filter
        ]
    hybrid = HybridTensors.load(hybrid_root)
    shared = _load_shared(hybrid_root, hybrid.device)
    residuals = _q8_tensor(hybrid_root, hybrid.device, METHOD_ID)
    source_fields = hybrid.materialize(
        appearance=shared,
        residuals=residuals,
        active_instance_ids=SOURCE_ACTIVE_INSTANCE_IDS,
    )
    edited_fields = hybrid.materialize(
        appearance=shared,
        residuals=residuals,
        active_instance_ids=EDITED_ACTIVE_INSTANCE_IDS,
    )
    removed_fields = hybrid.panels(
        appearance=shared,
        residuals=residuals,
        active_instance_ids=(REMOVED_INSTANCE_ID,),
    )
    render_records: list[PerViewRenderResult] = []
    mask_records: list[GeometricChangeMaskRecord] = []
    localization_records: list[ImageChangeRecord] = []
    hero: dict[str, Any] = {}
    for record in camera_records:
        camera_id = str(record["camera_id"])
        camera = _camera_from_record(record)
        view_root = output / "renders" / camera_id
        view_root.mkdir(parents=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            source = render_fields(source_fields, camera)
        torch.cuda.synchronize()
        source_latency = time.perf_counter() - started
        source_peak_allocated = int(torch.cuda.max_memory_allocated())
        source_peak_reserved = int(torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            edited = render_fields(edited_fields, camera)
        torch.cuda.synchronize()
        edited_latency = time.perf_counter() - started
        edited_peak_allocated = int(torch.cuda.max_memory_allocated())
        edited_peak_reserved = int(torch.cuda.max_memory_reserved())
        with torch.no_grad():
            removed = render_fields(removed_fields, camera)
        source_rgb = source.rgb.clamp(0, 1)
        edited_rgb = edited.rgb.clamp(0, 1)
        difference = (source_rgb - edited_rgb).abs()
        changed = difference.amax(dim=-1) > CHANGE_THRESHOLD
        removed_footprint = torch.isfinite(removed.depth) & (
            removed.alpha > FOOTPRINT_ALPHA_THRESHOLD
        )
        raw_support = torch.isfinite(removed.depth) & (
            removed.alpha > SUPPORT_ALPHA_THRESHOLD
        )
        support = _dilate(raw_support, UNCERTAINTY_DILATION_PIXELS)
        uncertainty = support & ~removed_footprint
        stable_outside = ~support
        background = edited.alpha < FOOTPRINT_ALPHA_THRESHOLD
        inside_changed = changed & support
        outside_changed = changed & stable_outside
        outside_values = difference[stable_outside]
        outside_maximum = (
            float(outside_values.max().item()) if outside_values.numel() else 0.0
        )
        _write_rgb(view_root / "source-structured.png", source_rgb)
        _write_rgb(view_root / "edited-remove-middle.png", edited_rgb)
        effect = difference.amax(dim=-1, keepdim=True)
        effect_rgb = torch.cat(
            (
                effect.clamp(0, 1),
                (effect * 0.25).clamp(0, 1),
                torch.zeros_like(effect),
            ),
            dim=-1,
        )
        _write_rgb(view_root / "edit-effect.png", effect_rgb)
        masks = {
            "removed-footprint": removed_footprint,
            "conservative-support": support,
            "uncertainty-band": uncertainty,
            "stable-outside": stable_outside,
            "background": background,
        }
        mask_artifacts = {
            name: _write_mask(view_root / f"{name}.png", mask)
            for name, mask in masks.items()
        }
        render_records.append(
            {
                "camera_id": camera_id,
                "image_name": str(record["image_name"]),
                "width": camera.width,
                "height": camera.height,
                "source_render_digest": _tensor_digest(source_rgb),
                "edited_render_digest": _tensor_digest(edited_rgb),
                "source_alpha_digest": _tensor_digest(source.alpha),
                "edited_alpha_digest": _tensor_digest(edited.alpha),
                "source_depth_digest": _tensor_digest(source.depth),
                "edited_depth_digest": _tensor_digest(edited.depth),
                "source_render_latency_seconds": source_latency,
                "edited_render_latency_seconds": edited_latency,
                "materialized_source_gaussian_count": SOURCE_MATERIALIZED_TOTAL,
                "materialized_edited_gaussian_count": EDITED_MATERIALIZED_TOTAL,
                "gpu_peak_allocated_bytes": max(
                    source_peak_allocated, edited_peak_allocated
                ),
                "gpu_peak_reserved_bytes": max(
                    source_peak_reserved, edited_peak_reserved
                ),
                "complete": True,
                "warnings": [
                    "materialized renderer; no runtime-memory reduction claim",
                    "edited scene has no photographic ground truth",
                ],
            }
        )
        mask_records.append(
            {
                "camera_id": camera_id,
                "width": camera.width,
                "height": camera.height,
                "removed_footprint_digest": (
                    f"sha256:{mask_artifacts['removed-footprint']['sha256']}"
                ),
                "conservative_support_digest": (
                    f"sha256:{mask_artifacts['conservative-support']['sha256']}"
                ),
                "uncertainty_band_digest": (
                    f"sha256:{mask_artifacts['uncertainty-band']['sha256']}"
                ),
                "stable_outside_digest": (
                    f"sha256:{mask_artifacts['stable-outside']['sha256']}"
                ),
                "background_digest": (
                    f"sha256:{mask_artifacts['background']['sha256']}"
                ),
                "removed_footprint_pixels": int(removed_footprint.sum().item()),
                "conservative_support_pixels": int(support.sum().item()),
                "uncertainty_band_pixels": int(uncertainty.sum().item()),
                "stable_outside_pixels": int(stable_outside.sum().item()),
                "background_pixels": int(background.sum().item()),
                "generation_parameters": {
                    "source": "removed panel-middle canonical geometry projection",
                    "footprint_alpha_threshold": FOOTPRINT_ALPHA_THRESHOLD,
                    "support_alpha_threshold": SUPPORT_ALPHA_THRESHOLD,
                    "uncertainty_dilation_pixels": UNCERTAINTY_DILATION_PIXELS,
                    "antialiasing_policy": "nonzero-alpha support plus fixed dilation",
                    "hand_painted": False,
                },
            }
        )
        localization_records.append(
            {
                "camera_id": camera_id,
                "changed_pixel_count": int(changed.sum().item()),
                "changed_pixel_fraction": float(changed.float().mean().item()),
                "maximum_absolute_rgb_difference": float(difference.max().item()),
                "mean_absolute_rgb_change": float(difference.mean().item()),
                "changed_pixels_inside_geometric_support": int(
                    inside_changed.sum().item()
                ),
                "changed_pixels_outside_geometric_support": int(
                    outside_changed.sum().item()
                ),
                "stable_outside_region_pixel_count": int(stable_outside.sum().item()),
                "maximum_outside_region_difference": outside_maximum,
                "outside_region_tolerance": OUTSIDE_TOLERANCE,
                "outside_region_stable": outside_maximum <= OUTSIDE_TOLERANCE,
            }
        )
        if camera_id == HERO_CAMERA_ID:
            all_panels = hybrid.panels(
                appearance=shared,
                residuals=residuals,
                active_instance_ids=SOURCE_ACTIVE_INSTANCE_IDS,
            )
            with torch.no_grad():
                panel_plate = render_fields(all_panels, camera)
            panel_mask = panel_plate.alpha > FOOTPRINT_ALPHA_THRESHOLD
            rows, columns = torch.where(panel_mask)
            if rows.numel() == 0:
                raise StructuringError("hero panel projection is empty")
            padding = 24
            x0 = max(0, int(columns.min().item()) - padding)
            y0 = max(0, int(rows.min().item()) - padding)
            x1 = min(camera.width, int(columns.max().item()) + padding + 1)
            y1 = min(camera.height, int(rows.max().item()) + padding + 1)
            instance_overlay_masks = {}
            for instance_id in SOURCE_ACTIVE_INSTANCE_IDS:
                instance_fields = hybrid.panels(
                    appearance=shared,
                    residuals=residuals,
                    active_instance_ids=(instance_id,),
                )
                with torch.no_grad():
                    instance_plate = render_fields(instance_fields, camera)
                artifact = _write_mask(
                    view_root / f"oracle-{instance_id}.png",
                    instance_plate.alpha > FOOTPRINT_ALPHA_THRESHOLD,
                )
                instance_overlay_masks[instance_id] = {
                    "sha256": artifact["sha256"],
                    "pixel_count": artifact["pixel_count"],
                }
            hero = {
                "camera_id": HERO_CAMERA_ID,
                "image_name": str(record["image_name"]),
                "crop_xyxy": [x0, y0, x1, y1],
                "image_dimensions": [camera.width, camera.height],
                "selection_policy": (
                    "fixed validated hero camera; deterministic panel-union "
                    "bounding box plus 24 px"
                ),
                "instance_overlay_masks": instance_overlay_masks,
            }
    return render_records, mask_records, localization_records, hero


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise StructuringError(
            "PyTorch is required for local Truck rendering"
        ) from error
    if not torch.cuda.is_available():
        raise StructuringError("CUDA is required for local Truck rendering")
    return torch


def _aggregate_localization(records: list[ImageChangeRecord]) -> dict[str, Any]:
    total_pixels = sum(
        record["stable_outside_region_pixel_count"] for record in records
    )
    return {
        "camera_count": len(records),
        "changed_pixel_count_total": sum(
            record["changed_pixel_count"] for record in records
        ),
        "changed_pixel_fraction_mean": sum(
            record["changed_pixel_fraction"] for record in records
        )
        / len(records),
        "maximum_absolute_rgb_difference": max(
            record["maximum_absolute_rgb_difference"] for record in records
        ),
        "mean_absolute_rgb_change_mean": sum(
            record["mean_absolute_rgb_change"] for record in records
        )
        / len(records),
        "changed_pixels_outside_geometric_support_total": sum(
            record["changed_pixels_outside_geometric_support"] for record in records
        ),
        "stable_outside_region_pixel_count_total": total_pixels,
        "maximum_outside_region_difference": max(
            record["maximum_outside_region_difference"] for record in records
        ),
        "outside_region_tolerance": OUTSIDE_TOLERANCE,
        "all_outside_regions_stable": all(
            record["outside_region_stable"] for record in records
        ),
    }


def _render_scientific_digest(records: list[PerViewRenderResult]) -> str:
    excluded = {
        "source_render_latency_seconds",
        "edited_render_latency_seconds",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
    }
    projection = [
        {key: value for key, value in record.items() if key not in excluded}
        for record in records
    ]
    return content_digest(cast(Any, projection))


def execute_edit(
    *,
    refit_root: Path,
    operation: str,
    instance: str,
    output: Path,
    camera_filter: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Serialize, render, and diagnose the one bounded Truck removal edit."""

    if operation != "remove-instance" or instance != REMOVED_INSTANCE_ID:
        raise StructuringError("only remove-instance panel-middle is permitted")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"edit output must be empty: {output}")
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    hybrid_root = _hybrid_root(refit_root)
    request = build_edit_request(hybrid_root)
    atomic_json(output / "edit-request.json", request.to_dict())
    hybrid_manifest = load_object(hybrid_root / "hybrid.json")
    atomic_json(
        output / "runtime-local.json",
        {
            "classification": "local_only_never_stage",
            "hybrid_root": str(hybrid_root),
            "dataset_root": str(hybrid_manifest["local_dataset_root"]),
        },
    )
    source_root = output / "source-representation"
    edited_root = output / "edited-representation"
    source_manifest = _serialize_representation(
        hybrid_root=hybrid_root,
        output=source_root,
        request=request,
        active_instance_ids=SOURCE_ACTIVE_INSTANCE_IDS,
        source=True,
    )
    edited_manifest = _serialize_representation(
        hybrid_root=hybrid_root,
        output=edited_root,
        request=request,
        active_instance_ids=EDITED_ACTIVE_INSTANCE_IDS,
        source=False,
    )
    fixed_content = _validate_fixed_content(
        source_root=source_root,
        edited_root=edited_root,
        hybrid_root=hybrid_root,
        request=request,
    )
    middle_absent = _validate_middle_absent_from_background(hybrid_root)
    if not middle_absent:
        raise StructuringError("middle core reappears through fixed background")
    source_bytes = _tree_bytes(source_root)
    edited_bytes = _tree_bytes(edited_root)
    delta = _file_delta(source_root, edited_root)
    request_bytes = (output / "edit-request.json").stat().st_size
    accounting = {
        "accounting_level": "serialized_uncompressed",
        "representation_boundary": (
            "fixed explicit background, canonical terminal, frozen transforms, "
            "shared appearance, SH1 q8 payload, active-instance set, binding, "
            "and representation metadata"
        ),
        "source_representation_bytes": source_bytes,
        "edited_representation_bytes": edited_bytes,
        "edit_request_bytes": request_bytes,
        "changed_serialized_bytes": delta["changed_serialized_bytes"],
        "added_bytes": delta["added_bytes"],
        "removed_bytes": delta["removed_bytes"],
        "changed_files": delta["changed_files"],
        "unchanged_files": delta["unchanged_files"],
        "instance_record_delta": -1,
        "metadata_byte_delta": edited_bytes - source_bytes,
        "materialized_count_delta": -CANONICAL_COUNT,
        "exclusions": [
            "source photographs",
            "renders",
            "masks",
            "panel composites",
            "videos",
            "renderer workspace",
        ],
        "claim_boundary": (
            "actual local serialized edit delta; no whole-scene compression, "
            "runtime-memory, or rendering-speed claim"
        ),
    }
    accounting["scientific_digest"] = content_digest(cast(Any, accounting))
    atomic_json(output / "edit-accounting.json", accounting)
    render_ledger, mask_ledger, localization, hero = _render_evaluation(
        hybrid_root=hybrid_root,
        output=output,
        camera_filter=camera_filter,
    )
    full_evaluation = camera_filter is None
    localization_summary = _aggregate_localization(localization)
    geometry = {
        "removed_instance_absent": True,
        "removed_instance_id": REMOVED_INSTANCE_ID,
        "remaining_instance_counts": {"panel-rear": 1, "panel-front": 1},
        "duplicate_retained_instances": False,
        "middle_source_core_reintroduced_by_background": False,
        "body_and_frame_preserved_by_background_identity": True,
        "camera_dependent_edit_state": False,
    }
    materialization = MaterializationResult(
        fixed_explicit_complement=BACKGROUND_COUNT,
        canonical_stored=CANONICAL_COUNT,
        source_active_instances=3,
        source_canonical_materialized=SOURCE_MATERIALIZED_CANONICAL,
        source_hybrid_materialized=SOURCE_MATERIALIZED_TOTAL,
        edited_active_instances=2,
        edited_canonical_materialized=EDITED_MATERIALIZED_CANONICAL,
        edited_hybrid_materialized=EDITED_MATERIALIZED_TOTAL,
    )
    source_record = SourceHybridRepresentation(
        representation_digest=str(source_manifest["scientific_digest"]),
        canonical_terminal_digest=request.canonical_terminal_digest,
        fixed_background_digest=request.fixed_background_digest,
        fixed_background_integrity_digest=request.fixed_background_integrity_digest,
        active_instance_ids=SOURCE_ACTIVE_INSTANCE_IDS,
        stored_gaussian_count=BACKGROUND_COUNT + CANONICAL_COUNT,
        materialized_gaussian_count=SOURCE_MATERIALIZED_TOTAL,
        complete_bytes=source_bytes,
    )
    active_set = EditedActiveInstanceSet(
        active_instance_ids=EDITED_ACTIVE_INSTANCE_IDS,
        removed_instance_id=REMOVED_INSTANCE_ID,
        expected_instance_count=2,
    )
    edited_record = EditedHybridRepresentation(
        representation_digest=str(edited_manifest["scientific_digest"]),
        parent_representation_digest=request.source_representation_digest,
        edit_request_digest=request.scientific_digest,
        active_instance_set_digest=active_set.scientific_digest,
        canonical_terminal_digest=request.canonical_terminal_digest,
        fixed_background_digest=request.fixed_background_digest,
        fixed_background_integrity_digest=request.fixed_background_integrity_digest,
        active_instance_ids=EDITED_ACTIVE_INSTANCE_IDS,
        stored_gaussian_count=BACKGROUND_COUNT + CANONICAL_COUNT,
        materialized_gaussian_count=EDITED_MATERIALIZED_TOTAL,
        complete_bytes=edited_bytes,
    )
    removed_record = RemovedInstance(
        instance_id=REMOVED_INSTANCE_ID,
        source_position=1,
        canonical_gaussian_count=CANONICAL_COUNT,
        materialized_count_delta=-CANONICAL_COUNT,
        source_core_reintroduced_by_background=False,
    )
    status = (
        "completed"
        if full_evaluation
        and len(render_ledger) == 8
        and localization_summary["all_outside_regions_stable"]
        else "debug_subset"
    )
    edit_record: EditEvidenceRecord = {
        "edit_id": request.edit_id,
        "source_pilot_version": request.source_pilot_version,
        "region_version": request.region_version,
        "source_representation_digest": request.source_representation_digest,
        "canonical_terminal_digest": request.canonical_terminal_digest,
        "fixed_background_digest": request.fixed_background_digest,
        "removed_instance_id": request.removed_instance_id,
        "active_instance_ids": list(request.active_instance_ids),
        "camera_policy_digest": request.camera_policy_digest,
        "stored_gaussian_count": request.stored_gaussian_count,
        "materialized_gaussian_count": request.materialized_gaussian_count,
        "changed_field_paths": list(request.changed_field_paths),
        "complete_bytes": edited_bytes,
        "changed_serialized_bytes": int(accounting["changed_serialized_bytes"]),
        "edit_request_bytes": request_bytes,
        "status": status,
        "warnings": [
            "oracle annotation",
            "local-only qualitative real-benchmark demonstration",
            "no photographic ground truth exists for the edited scene",
        ],
        "limitations": list(request.limitations),
    }
    summary: dict[str, Any] = {
        "edit_id": EDIT_ID,
        "edit_version": EDIT_VERSION,
        "branch_id": BRANCH_ID,
        "status": status,
        "source_pilot_version": SOURCE_PILOT_VERSION,
        "region_version": REGION_VERSION,
        "source_representation": asdict(source_record),
        "edited_active_instance_set": {
            **asdict(active_set),
            "scientific_digest": active_set.scientific_digest,
        },
        "edited_representation": asdict(edited_record),
        "edit_record": edit_record,
        "removed_instance": asdict(removed_record),
        "fixed_content_integrity": fixed_content,
        "geometry_sanity": geometry,
        "materialization": asdict(materialization),
        "accounting_digest": accounting["scientific_digest"],
        "render_ledger_digest": _render_scientific_digest(render_ledger),
        "mask_ledger_digest": content_digest(cast(Any, mask_ledger)),
        "localization_digest": content_digest(cast(Any, localization)),
        "localization_summary": localization_summary,
        "hero": hero,
        "camera_filter": sorted(camera_filter) if camera_filter else None,
        "all_evaluation_views_complete": full_evaluation
        and len(render_ledger) == len(EVALUATION_CAMERA_IDS),
        "elapsed_seconds": time.perf_counter() - started,
        "changed_field_paths": list(request.changed_field_paths),
        "warnings": [
            "oracle annotation",
            "local-only qualitative real-benchmark demonstration",
            "no photographic ground truth exists for the edited scene",
        ],
        "limitations": list(request.limitations),
        "release_use_status": "not_cleared_for_release",
    }
    identity_projection = {
        key: value
        for key, value in summary.items()
        if key not in {"elapsed_seconds", "camera_filter"}
    }
    summary["scientific_digest"] = content_digest(cast(Any, identity_projection))
    atomic_json(output / "render-ledger.json", {"views": render_ledger})
    atomic_json(output / "mask-ledger.json", {"views": mask_ledger})
    atomic_json(
        output / "localization-summary.json",
        {"views": localization, "aggregate": localization_summary},
    )
    atomic_json(output / "hero-view.json", hero)
    atomic_json(output / "edit-summary.json", summary)
    return summary


_WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[\"'\\s])(?:[a-z]:[\\/])")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


def privacy_issues(text: str) -> list[str]:
    """Reject machine-local paths and personal identifiers from portable text."""

    issues: list[str] = []
    checks = {
        "absolute_windows_path": bool(_WINDOWS_ABSOLUTE.search(text)),
        "absolute_home_path": any(
            token in text for token in ("/home/", "/Users/", "/mnt/c/", "\\Users\\")
        ),
        "email_address": bool(_EMAIL.search(text)),
        "wsl_network_path": "\\\\wsl" in text.lower(),
        "machine_hostname_field": bool(
            re.search(r'(?i)"(?:host|hostname|machine_name)"\s*:', text)
        ),
    }
    issues.extend(name for name, present in checks.items() if present)
    return issues


def validate_edit_root(root: Path, *, require_full: bool = True) -> dict[str, Any]:
    """CPU-only validation of identity, integrity, counts, masks, and localization."""

    request = load_object(root / "edit-request.json")
    summary = load_object(root / "edit-summary.json")
    accounting = load_object(root / "edit-accounting.json")
    renders = load_object(root / "render-ledger.json").get("views", [])
    masks = load_object(root / "mask-ledger.json").get("views", [])
    localization = load_object(root / "localization-summary.json")
    issues: list[dict[str, Any]] = []
    if request.get("edit_id") != EDIT_ID:
        issues.append({"code": "invalid_edit_id"})
    if request.get("scientific_digest") != content_digest(
        cast(
            Any,
            {
                key: value
                for key, value in request.items()
                if key != "scientific_digest"
            },
        )
    ):
        issues.append({"code": "invalid_edit_request_digest"})
    runtime_path = root / "runtime-local.json"
    if not runtime_path.is_file():
        issues.append({"code": "missing_local_runtime_record"})
    else:
        runtime = load_object(runtime_path)
        try:
            request_record = dict(request)
            request_record.pop("scientific_digest", None)
            for field in (
                "active_instance_ids",
                "changed_field_paths",
                "warnings",
                "limitations",
            ):
                request_record[field] = tuple(request_record[field])
            typed_request = RealEditRequest(**request_record)
            _validate_fixed_content(
                source_root=root / "source-representation",
                edited_root=root / "edited-representation",
                hybrid_root=Path(str(runtime["hybrid_root"])),
                request=typed_request,
            )
        except (KeyError, OSError, StructuringError, TypeError, ValueError) as error:
            issues.append(
                {
                    "code": "fixed_content_integrity_mismatch",
                    "message": str(error),
                }
            )
    active_record = load_object(
        root / "edited-representation" / "active-instances.json"
    )
    active = tuple(active_record.get("active_instance_ids", ()))
    try:
        validate_active_instances(cast(tuple[str, ...], active))
    except StructuringError as error:
        issues.append({"code": "invalid_active_instance_set", "message": str(error)})
    materialization = summary.get("materialization", {})
    expected_counts = {
        "fixed_explicit_complement": BACKGROUND_COUNT,
        "canonical_stored": CANONICAL_COUNT,
        "source_active_instances": 3,
        "source_canonical_materialized": SOURCE_MATERIALIZED_CANONICAL,
        "source_hybrid_materialized": SOURCE_MATERIALIZED_TOTAL,
        "edited_active_instances": 2,
        "edited_canonical_materialized": EDITED_MATERIALIZED_CANONICAL,
        "edited_hybrid_materialized": EDITED_MATERIALIZED_TOTAL,
    }
    if any(materialization.get(key) != value for key, value in expected_counts.items()):
        issues.append({"code": "invalid_materialized_counts"})
    integrity = summary.get("fixed_content_integrity", {})
    for field in (
        "canonical_byte_identical",
        "background_byte_identical",
        "instance_transforms_byte_identical",
        "residual_payload_byte_identical",
        "camera_records_unchanged",
    ):
        if integrity.get(field) is not True:
            issues.append({"code": f"failed_{field}"})
    if accounting.get("changed_serialized_bytes", 0) <= 0:
        issues.append({"code": "missing_serialization_delta"})
    expected_cameras = set(EVALUATION_CAMERA_IDS)
    render_ids = {record.get("camera_id") for record in renders}
    mask_ids = {record.get("camera_id") for record in masks}
    if require_full and render_ids != expected_cameras:
        issues.append({"code": "missing_camera_render"})
    if require_full and mask_ids != expected_cameras:
        issues.append({"code": "missing_camera_mask"})
    for record in masks:
        width, height = int(record.get("width", 0)), int(record.get("height", 0))
        if width <= 0 or height <= 0:
            issues.append(
                {
                    "code": "invalid_mask_dimensions",
                    "camera_id": record.get("camera_id"),
                }
            )
        if (
            int(record.get("conservative_support_pixels", -1))
            + int(record.get("stable_outside_pixels", -1))
            != width * height
        ):
            issues.append(
                {
                    "code": "mask_partition_mismatch",
                    "camera_id": record.get("camera_id"),
                }
            )
        camera_id = str(record.get("camera_id"))
        for name in (
            "removed-footprint.png",
            "conservative-support.png",
            "uncertainty-band.png",
            "stable-outside.png",
            "background.png",
        ):
            path = root / "renders" / camera_id / name
            try:
                from PIL import Image

                with Image.open(path) as image:
                    actual_dimensions = image.size
                if actual_dimensions != (width, height):
                    issues.append(
                        {
                            "code": "mask_dimension_mismatch",
                            "camera_id": camera_id,
                            "mask": name,
                        }
                    )
            except (OSError, ValueError):
                issues.append(
                    {
                        "code": "missing_camera_mask",
                        "camera_id": camera_id,
                        "mask": name,
                    }
                )
    for record in renders:
        camera_id = str(record.get("camera_id"))
        for name in (
            "source-structured.png",
            "edited-remove-middle.png",
            "edit-effect.png",
        ):
            if not (root / "renders" / camera_id / name).is_file():
                issues.append(
                    {
                        "code": "missing_camera_render",
                        "camera_id": camera_id,
                        "render": name,
                    }
                )
    aggregate = localization.get("aggregate", {})
    if not aggregate.get("all_outside_regions_stable"):
        issues.append({"code": "unlocalized_changes"})
    privacy_text = "\n".join(
        json.dumps(value, sort_keys=True)
        for value in (request, summary, accounting, renders, masks, localization)
    )
    issues.extend({"code": code} for code in privacy_issues(privacy_text))
    return {
        "valid": not issues,
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "edit_id": request.get("edit_id"),
        "camera_count": len(renders),
        "mask_count": len(masks),
        "changed_serialized_bytes": accounting.get("changed_serialized_bytes"),
        "scientific_digest": summary.get("scientific_digest"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    edit = commands.add_parser("edit")
    edit.add_argument("--refit-root", type=Path, required=True)
    edit.add_argument("--operation", required=True, choices=("remove-instance",))
    edit.add_argument("--instance", required=True, choices=(REMOVED_INSTANCE_ID,))
    edit.add_argument("--output", type=Path, required=True)
    edit.add_argument("--camera", action="append")
    edit.add_argument("--overwrite", action="store_true")
    edit.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate-edit")
    validate.add_argument("path", type=Path)
    validate.add_argument("--allow-debug-subset", action="store_true")
    validate.add_argument("--json", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    """Execute or validate the frozen edit with structured output and exits."""

    options = _parser().parse_args(arguments)
    try:
        if options.action == "edit":
            result = execute_edit(
                refit_root=options.refit_root,
                operation=options.operation,
                instance=options.instance,
                output=options.output,
                camera_filter=set(options.camera) if options.camera else None,
                overwrite=options.overwrite,
            )
            valid = result["status"] == "completed" or bool(options.camera)
        else:
            result = validate_edit_root(
                options.path,
                require_full=not options.allow_debug_subset,
            )
            valid = bool(result["valid"])
    except (StructuringError, FileExistsError, OSError, ValueError) as error:
        result = {"valid": False, "error": str(error)}
        valid = False
    if options.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    elif valid:
        print(
            f"{EDIT_ID}: {result.get('status', 'completed')}; "
            f"artifact={getattr(options, 'output', getattr(options, 'path', ''))}"
        )
    else:
        print(result.get("error", f"{EDIT_ID}: validation failed"))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
