"""Local-only extraction of panel ownership, frames, and canonical arrays."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import content_digest
from gaussweave.gaussians.real_explicit import RealGaussianTensorSource
from gaussweave.real_structuring.frames import build_frames
from gaussweave.real_structuring.models import (
    CORE_COUNTS,
    INSTANCE_IDS,
    PILOT_VERSION,
    SOURCE_COUNT,
    OwnershipStatus,
    StructuringError,
    atomic_json,
    sha256_file,
)
from gaussweave.real_structuring.regions import INSTANCE_VIEWS
from gaussweave.real_structuring.registration import register_instances


def _read_pfm(path: Path) -> Any:
    torch = _torch()
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise StructuringError("primary depth must be a grayscale PFM")
        width, height = (int(value) for value in stream.readline().split())
        if float(stream.readline()) >= 0:
            raise StructuringError("primary PFM must use little-endian float32")
        payload = bytearray(stream.read())
    if len(payload) != width * height * 4:
        raise StructuringError("primary depth PFM size mismatch")
    return torch.frombuffer(payload, dtype=torch.float32).reshape(height, width).flip(0)


def _raw(path: Path, tensor: Any) -> dict[str, Any]:
    array = tensor.detach().to("cpu").contiguous().numpy()
    array.tofile(path)
    return {
        "path": path.name,
        "shape": list(tensor.shape),
        "dtype": "little_endian_float32",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _indices(path: Path, tensor: Any) -> dict[str, Any]:
    values = tensor.detach().to("cpu").to(_torch().int32).contiguous().numpy()
    values.tofile(path)
    return {
        "path": path.name,
        "shape": [int(tensor.numel())],
        "dtype": "little_endian_uint32",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _quat_multiply(left: Any, right: Any) -> Any:
    torch = _torch()
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _matrix_to_quaternion(rotation: Any) -> Any:
    # The qualified panel frames share one proper rotation. This stable closed form
    # follows the same scalar-first convention as the Gaussian representation.
    torch = _torch()
    values = [
        [float(rotation[row, col].item()) for col in range(3)] for row in range(3)
    ]
    r00, r01, r02 = values[0]
    r10, r11, r12 = values[1]
    r20, r21, r22 = values[2]
    trace = r00 + r11 + r22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quaternion = (
            0.25 * scale,
            (r21 - r12) / scale,
            (r02 - r20) / scale,
            (r10 - r01) / scale,
        )
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2
        quaternion = (
            (r21 - r12) / scale,
            0.25 * scale,
            (r01 + r10) / scale,
            (r02 + r20) / scale,
        )
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2
        quaternion = (
            (r02 - r20) / scale,
            (r01 + r10) / scale,
            0.25 * scale,
            (r12 + r21) / scale,
        )
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2
        quaternion = (
            (r10 - r01) / scale,
            (r02 + r20) / scale,
            (r12 + r21) / scale,
            0.25 * scale,
        )
    result = torch.tensor(quaternion, dtype=torch.float32)
    result /= torch.linalg.vector_norm(result)
    if result[0] < 0:
        result *= -1
    return result


def extract_region(
    *,
    qualification: Path,
    dataset_root: Path,
    output: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reproduce validated cores and serialize one local canonical terminal."""

    torch = _torch()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = dataset_root.resolve()
    converted = dataset_root / "converted" / "graphdeco-30000"
    annotation_path = dataset_root / "annotations" / "truck-bed-side-panels.json"
    split_path = dataset_root / "annotations" / "evaluation-split.json"
    depth_path = (
        dataset_root
        / "renders"
        / "graphdeco-30000-selected"
        / "cam-train-000251"
        / "cam-train-000251-depth.pfm"
    )
    required = (qualification, annotation_path, split_path, depth_path)
    if any(not path.is_file() for path in required):
        raise StructuringError("qualified source input is missing")
    qualification_raw = json.loads(qualification.read_text(encoding="utf-8"))
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if qualification_raw.get("qualification_version") != "gw-truck-explicit-v1":
        raise StructuringError("source qualification identity mismatch")
    records = cast(list[Mapping[str, Any]], split["cameras"])
    primary = next(
        (record for record in records if record["camera_id"] == "cam-train-000251"),
        None,
    )
    if primary is None:
        raise StructuringError("qualified primary fitting camera is missing")
    source = RealGaussianTensorSource.load(converted)
    if source.count != SOURCE_COUNT:
        raise StructuringError("source Gaussian count mismatch")
    tensors = source.tensors("cpu")
    means = tensors["means"].to(torch.float64)
    world_from_camera = torch.tensor(
        primary["world_from_camera"], dtype=torch.float64
    ).reshape(4, 4)
    camera_from_world = torch.linalg.inv(world_from_camera)
    homogeneous = torch.cat(
        (means, torch.ones((means.shape[0], 1), dtype=torch.float64)), dim=1
    )
    camera_points = (camera_from_world @ homogeneous.T).T[:, :3]
    positive = camera_points[:, 2] > 1e-6
    x_pixels = float(primary["fx"]) * camera_points[:, 0] / camera_points[:, 2]
    y_pixels = float(primary["fy"]) * camera_points[:, 1] / camera_points[:, 2]
    x_pixels += float(primary["cx"])
    y_pixels += float(primary["cy"])
    depth = _read_pfm(depth_path)
    ownership = torch.full(
        (source.count,),
        int(OwnershipStatus.EXPLICIT_BACKGROUND),
        dtype=torch.uint8,
    )
    masks: dict[str, Any] = {}
    core_indices: dict[str, Any] = {}
    guard_candidates = torch.zeros(source.count, dtype=torch.bool)
    status_by_id = {
        "panel-rear": OwnershipStatus.PANEL_REAR_CORE,
        "panel-middle": OwnershipStatus.PANEL_MIDDLE_CORE,
        "panel-front": OwnershipStatus.PANEL_FRONT_CORE,
    }
    for instance_id in INSTANCE_IDS:
        x0, y0, x1, y1 = INSTANCE_VIEWS[instance_id]["000251.jpg"]
        patch = depth[y0:y1, x0:x1]
        valid = patch[torch.isfinite(patch) & (patch > 0)]
        median = float(torch.median(valid).item())
        mad = float(torch.median((valid - median).abs()).item())
        half = max(0.18, 4.0 * mad)
        inside = (
            positive
            & (x_pixels >= x0)
            & (x_pixels <= x1)
            & (y_pixels >= y0)
            & (y_pixels <= y1)
        )
        selected = inside & ((camera_points[:, 2] - median).abs() <= half)
        if int(selected.sum().item()) != CORE_COUNTS[instance_id]:
            raise StructuringError(f"core reproduction failed: {instance_id}")
        expanded = (
            positive
            & (x_pixels >= x0 - 8)
            & (x_pixels <= x1 + 8)
            & (y_pixels >= y0 - 8)
            & (y_pixels <= y1 + 8)
            & ((camera_points[:, 2] - median).abs() <= half * 1.25)
        )
        guard_candidates |= expanded & ~selected
        masks[instance_id] = selected
        core_indices[instance_id] = torch.nonzero(selected).flatten()
        ownership[selected] = int(status_by_id[instance_id])
    if any(
        bool((masks[left] & masks[right]).any().item())
        for i, left in enumerate(INSTANCE_IDS)
        for right in INSTANCE_IDS[i + 1 :]
    ):
        raise StructuringError("panel core ownership overlaps")
    ownership[
        guard_candidates & (ownership == int(OwnershipStatus.EXPLICIT_BACKGROUND))
    ] = int(OwnershipStatus.BOUNDARY_GUARD)
    ownership_path = output / "ownership.u8"
    ownership.numpy().tofile(ownership_path)
    counts = {
        status.name.lower(): int((ownership == int(status)).sum().item())
        for status in OwnershipStatus
    }
    ownership_record = {
        "pilot_version": PILOT_VERSION,
        "source_gaussian_count": source.count,
        "payload": ownership_path.name,
        "payload_bytes": ownership_path.stat().st_size,
        "payload_sha256": sha256_file(ownership_path),
        "status_codes": {
            status.name.lower(): int(status) for status in OwnershipStatus
        },
        "status_counts": counts,
        "policy": {
            "core": "exact validated depth-gated primary-view selections",
            "guard": (
                "non-core points in an 8 px expanded footprint and 1.25x depth "
                "gate; retained explicit"
            ),
            "ambiguous": "depth-rejected and all other non-core points remain explicit",
            "complement": "all source rows except the three exact core sets",
            "ordering": "official source PLY row order",
        },
    }
    ownership_record["scientific_digest"] = content_digest(cast(Any, ownership_record))
    atomic_json(output / "ownership.json", ownership_record)
    axes = torch.tensor(
        annotation["instances"][0]["provisional_3d_box"]["axes_world"],
        dtype=torch.float64,
    ).T
    local_points: dict[str, Any] = {}
    local_bounds: dict[str, Any] = {}
    opacities: dict[str, Any] = {}
    for instance_id, item in zip(INSTANCE_IDS, annotation["instances"], strict=True):
        origin = torch.tensor(
            item["provisional_3d_box"]["center_world"], dtype=torch.float64
        )
        points = (means[core_indices[instance_id]] - origin) @ axes
        local_points[instance_id] = points
        opacities[instance_id] = tensors["opacities"][core_indices[instance_id]]
        local_bounds[instance_id] = (
            [float(value) for value in points.amin(dim=0).tolist()],
            [float(value) for value in points.amax(dim=0).tolist()],
        )
    frames = build_frames(annotation, local_bounds)
    registrations = register_instances(frames, local_points, opacities)
    frames_record = {
        "pilot_version": PILOT_VERSION,
        "canonical_instance_id": "panel-middle",
        "coordinate_convention": (
            "right-handed local xyz; columns of world_from_local are semantic axes"
        ),
        "frames": [frame.to_dict() for frame in frames],
    }
    frames_record["scientific_digest"] = content_digest(cast(Any, frames_record))
    atomic_json(output / "frames.json", frames_record)
    registration_record = {
        "pilot_version": PILOT_VERSION,
        "method": (
            "oracle annotation frame initialization plus deterministic "
            "opacity-weighted robust geometry-only bounded similarity refinement"
        ),
        "canonical_instance_id": "panel-middle",
        "evaluation_images_or_losses_used": False,
        "instances": [item.to_dict() for item in registrations],
    }
    registration_record["scientific_digest"] = content_digest(
        cast(Any, registration_record)
    )
    atomic_json(output / "registration.json", registration_record)
    canonical_dir = output / "canonical"
    canonical_dir.mkdir(exist_ok=True)
    middle_indices = core_indices["panel-middle"]
    middle_frame = frames[1]
    rotation = torch.tensor(middle_frame.axes_world, dtype=torch.float32).T
    frame_quaternion = _matrix_to_quaternion(rotation)
    inverse_frame_quaternion = frame_quaternion.clone()
    inverse_frame_quaternion[1:] *= -1
    local_quaternions = _quat_multiply(
        inverse_frame_quaternion.expand(middle_indices.numel(), -1),
        tensors["quaternions"][middle_indices],
    )
    local_quaternions /= torch.linalg.vector_norm(
        local_quaternions, dim=1, keepdim=True
    )
    canonical_files = {
        "means": _raw(
            canonical_dir / "means.f32",
            local_points["panel-middle"].to(torch.float32),
        ),
        "quaternions": _raw(canonical_dir / "quaternions.f32", local_quaternions),
        "scales": _raw(canonical_dir / "scales.f32", tensors["scales"][middle_indices]),
        "opacities": _raw(
            canonical_dir / "opacities.f32", tensors["opacities"][middle_indices]
        ),
        "sh_coefficients": _raw(
            canonical_dir / "sh-coefficients.f32",
            tensors["appearance"][middle_indices],
        ),
        "source_indices": _indices(
            canonical_dir / "source-indices.u32", middle_indices
        ),
    }
    canonical_meta = {
        "format_version": "gaussweave-real-canonical-v1",
        "pilot_version": PILOT_VERSION,
        "terminal_id": "truck-panel-middle-canonical",
        "source_instance_id": "panel-middle",
        "gaussian_count": CORE_COUNTS["panel-middle"],
        "sh_degree": 3,
        "coordinate_frame": "panel-middle local frame",
        "ordering_policy": "official source PLY row order retained",
        "source_stable_id_policy": (
            f"g-{source.metadata['source_checkpoint']['sha256'][:12]}-"
            "<zero-padded-source-row>"
        ),
        "activation_conventions": {
            "scale": "positive activated standard deviation",
            "opacity": "activated alpha",
            "appearance": "GRAPHDECO real SH coefficient-major [N,16,3]",
        },
        "appearance_policy": "world_locked; frame rotations are identical in pilot",
        "clipping_policy": "explicit_membership",
        "local_bounds": [
            list(middle_frame.local_lower),
            list(middle_frame.local_upper),
        ],
        "files": canonical_files,
    }
    canonical_meta["scientific_digest"] = content_digest(cast(Any, canonical_meta))
    atomic_json(canonical_dir / "metadata.json", canonical_meta)
    background_indices = torch.nonzero(
        (ownership != int(OwnershipStatus.PANEL_REAR_CORE))
        & (ownership != int(OwnershipStatus.PANEL_MIDDLE_CORE))
        & (ownership != int(OwnershipStatus.PANEL_FRONT_CORE))
    ).flatten()
    background_record = _indices(output / "background-indices.u32", background_indices)
    manifest = {
        "pilot_version": PILOT_VERSION,
        "local_dataset_root": str(dataset_root),
        "qualification_sha256": sha256_file(qualification),
        "annotation_sha256": sha256_file(annotation_path),
        "split_sha256": sha256_file(split_path),
        "source_model_digest": source.metadata["model_digest"],
        "converted_root_relative": "converted/graphdeco-30000",
        "ownership": "ownership.json",
        "frames": "frames.json",
        "registration": "registration.json",
        "canonical": "canonical/metadata.json",
        "background_indices": background_record,
        "counts": {
            "source": source.count,
            "source_explicit_panel": sum(CORE_COUNTS.values()),
            "stored_canonical_panel": CORE_COUNTS["panel-middle"],
            "materialized_canonical_panel": 3 * CORE_COUNTS["panel-middle"],
            "fixed_explicit_background": int(background_indices.numel()),
            "hybrid_materialized": int(background_indices.numel())
            + 3 * CORE_COUNTS["panel-middle"],
        },
        "fitting_camera_ids": ["cam-train-000251", "cam-train-000180"],
        "evaluation_camera_ids": [
            str(record["camera_id"])
            for record in records
            if str(record["camera_id"]).startswith("cam-eval-")
        ],
    }
    manifest["scientific_digest"] = content_digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    return {
        "valid": True,
        "output": str(output),
        "ownership_counts": counts,
        "canonical_count": CORE_COUNTS["panel-middle"],
        "canonical_digest": canonical_meta["scientific_digest"],
        "background_count": int(background_indices.numel()),
        "registration": registration_record,
        "manifest_digest": manifest["scientific_digest"],
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise StructuringError("PyTorch is required for local extraction") from error
    return torch
