"""Hybrid fixed-background plus materialized-canonical construction."""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from gaussweave.config.resolution import content_digest
from gaussweave.gaussians.real_explicit import RealGaussianTensorSource
from gaussweave.real_structuring.models import (
    StructuringError,
    atomic_json,
    load_object,
    sha256_file,
)
from gaussweave.real_structuring.regions import validate_region_root

FIELDS: Mapping[str, tuple[str, tuple[int, ...]]] = {
    "means": ("means.f32", (3,)),
    "quaternions": ("quaternions.f32", (4,)),
    "scales": ("scales.f32", (3,)),
    "opacities": ("opacities.f32", ()),
    "appearance": ("sh-coefficients.f32", (16, 3)),
}


def _load_indices(path: Path, count: int) -> Any:
    torch = _torch()
    return torch.from_file(str(path), shared=False, size=count, dtype=torch.int32).to(
        torch.int64
    )


def _write_tensor(path: Path, tensor: Any) -> dict[str, Any]:
    tensor.detach().to("cpu").contiguous().numpy().tofile(path)
    return {
        "path": path.name,
        "shape": list(tensor.shape),
        "dtype": "little_endian_float32",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_hybrid(
    *, region_root: Path, output: Path, overwrite: bool = False
) -> dict[str, Any]:
    """Serialize a fixed explicit complement and one canonical terminal."""

    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validate_region_root(region_root)
    region_manifest = load_object(region_root / "manifest.json")
    dataset_root = Path(str(region_manifest["local_dataset_root"])).resolve()
    source = RealGaussianTensorSource.load(
        dataset_root / str(region_manifest["converted_root_relative"])
    )
    background_count = int(region_manifest["counts"]["fixed_explicit_background"])
    indices = _load_indices(region_root / "background-indices.u32", background_count)
    tensors = source.tensors("cpu")
    background_dir = output / "background"
    background_dir.mkdir()
    files: dict[str, Any] = {}
    for name, tensor in tensors.items():
        files[name] = _write_tensor(background_dir / FIELDS[name][0], tensor[indices])
    canonical_dir = output / "canonical"
    shutil.copytree(region_root / "canonical", canonical_dir)
    frames = load_object(region_root / "frames.json")
    registration = load_object(region_root / "registration.json")
    registrations = cast(list[dict[str, Any]], registration["instances"])
    instance_ids = tuple(str(item["instance_id"]) for item in registrations)
    if not instance_ids or len(instance_ids) != len(set(instance_ids)):
        raise StructuringError("hybrid instance IDs must be unique and nonempty")
    canonical_meta = load_object(region_root / "canonical" / "metadata.json")
    pilot_version = str(region_manifest["pilot_version"])
    region_version = str(
        region_manifest.get("region_version", "truck-bed-side-panels-v1")
    )
    atomic_json(output / "instance-transforms.json", registration)
    binding = {
        "format_version": "gaussweave-real-binding-v2",
        "pilot_version": pilot_version,
        "binding_id": (
            f"{region_manifest.get('scene_id', 'real-scene')}-"
            f"{region_version}-oracle-binding"
        ),
        "scene_id": region_manifest.get("scene_id", "tnt-truck"),
        "region_id": region_version,
        "grammar_provenance": "manual_oracle",
        "source_checkpoint_id": source.metadata["representation_id"],
        "unique_subset": "background/",
        "terminal": {
            "terminal_id": canonical_meta["terminal_id"],
            "canonical_subset": "canonical/",
            "source_instance_id": canonical_meta["source_instance_id"],
            "instance_ids": list(instance_ids),
            "residual_policy": "external_method_condition",
        },
        "ordering": (
            "background source order then canonical instances in declared "
            "registration order"
        ),
        "appearance_policy": "world_locked",
        "materialization": "reference_full_tensor_materialization",
        "limitations": [
            "manual oracle panels",
            "single scene-specific repeated region",
            "materialized rendering only",
        ],
    }
    binding["scientific_digest"] = content_digest(binding)
    atomic_json(output / "binding.json", binding)
    background_meta = {
        "format_version": "gaussweave-real-background-v1",
        "gaussian_count": background_count,
        "source_model_digest": source.metadata["model_digest"],
        "ordering_policy": "official source PLY order with core rows removed",
        "includes_boundary_guard": True,
        "files": files,
    }
    background_meta["scientific_digest"] = content_digest(background_meta)
    atomic_json(background_dir / "metadata.json", background_meta)
    manifest = {
        "format_version": "gaussweave-real-hybrid-v1",
        "pilot_version": pilot_version,
        "region_version": region_version,
        "local_dataset_root": str(dataset_root),
        "local_region_root": str(region_root.resolve()),
        "background": "background/metadata.json",
        "canonical": "canonical/metadata.json",
        "transforms": "instance-transforms.json",
        "binding": "binding.json",
        "frames_digest": frames["scientific_digest"],
        "counts": region_manifest["counts"],
        "source_model_digest": source.metadata["model_digest"],
        "converted_root_relative": region_manifest["converted_root_relative"],
        "fitting_camera_ids": region_manifest["fitting_camera_ids"],
        "evaluation_camera_ids": region_manifest["evaluation_camera_ids"],
        "method_ids": region_manifest.get("method_ids"),
        "hero_camera_id": region_manifest.get("hero_camera_id"),
        "instance_views": region_manifest.get("instance_views"),
    }
    manifest["scientific_digest"] = content_digest(manifest)
    atomic_json(output / "hybrid.json", manifest)
    return {
        "valid": True,
        "output": str(output),
        "background_count": background_count,
        "background_digest": background_meta["scientific_digest"],
        "canonical_count": int(canonical_meta["gaussian_count"]),
        "counts": manifest["counts"],
        "hybrid_digest": manifest["scientific_digest"],
    }


def _load_raw(root: Path, record: Mapping[str, Any], device: str) -> Any:
    torch = _torch()
    shape = tuple(int(value) for value in record["shape"])
    return (
        torch.from_file(
            str(root / str(record["path"])),
            shared=False,
            size=math.prod(shape),
            dtype=torch.float32,
        )
        .reshape(shape)
        .to(device)
    )


def _quaternion_from_matrix(matrix: Any) -> Any:
    torch = _torch()
    # Stable eigen-based conversion is unnecessary here: all registered rotations
    # are identical. Use PyTorch's largest-diagonal branches deterministically.
    r = matrix
    trace = float(torch.trace(r).item())
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        values = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + float(r[0, 0] - r[1, 1] - r[2, 2])) * 2
        values = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + float(r[1, 1] - r[0, 0] - r[2, 2])) * 2
        values = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + float(r[2, 2] - r[0, 0] - r[1, 1])) * 2
        values = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        )
    result = torch.stack(
        [
            value if torch.is_tensor(value) else torch.tensor(value, device=r.device)
            for value in values
        ]
    ).to(torch.float32)
    result /= torch.linalg.vector_norm(result)
    if result[0] < 0:
        result *= -1
    return result


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


@dataclass
class HybridTensors:
    """Loaded hybrid arrays with a differentiable canonical appearance override."""

    root: Path
    device: str
    background: dict[str, Any]
    canonical: dict[str, Any]
    transforms: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, root: Path, device: str = "cuda:0") -> HybridTensors:
        manifest = load_object(root / "hybrid.json")
        if manifest.get("format_version") != "gaussweave-real-hybrid-v1":
            raise StructuringError("unsupported hybrid artifact")
        background_meta = load_object(root / str(manifest["background"]))
        canonical_meta = load_object(root / str(manifest["canonical"]))
        background_root = root / "background"
        canonical_root = root / "canonical"
        background = {
            name: _load_raw(background_root, background_meta["files"][name], device)
            for name in FIELDS
        }
        canonical = {
            name: _load_raw(
                canonical_root,
                canonical_meta["files"][
                    "sh_coefficients" if name == "appearance" else name
                ],
                device,
            )
            for name in FIELDS
        }
        transforms = tuple(
            cast(dict[str, Any], item)
            for item in load_object(root / "instance-transforms.json")["instances"]
        )
        return cls(root.resolve(), device, background, canonical, transforms)

    def panels(
        self,
        *,
        appearance: Any | None = None,
        residuals: Any | None = None,
        active_instance_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Materialize selected canonical instances in frozen transform order."""

        torch = _torch()
        declared_ids = tuple(str(item["instance_id"]) for item in self.transforms)
        if active_instance_ids is None:
            active_instance_ids = declared_ids
        if len(active_instance_ids) != len(set(active_instance_ids)):
            raise StructuringError("active instance IDs must be unique")
        unknown = set(active_instance_ids) - set(declared_ids)
        if unknown:
            raise StructuringError(f"unknown active instance IDs: {sorted(unknown)}")
        source_appearance = (
            self.canonical["appearance"] if appearance is None else appearance
        )
        fields: dict[str, list[Any]] = {name: [] for name in FIELDS}
        for index, record in enumerate(self.transforms):
            if record["instance_id"] not in active_instance_ids:
                continue
            matrix = torch.tensor(
                record["matrix"], dtype=torch.float32, device=self.device
            ).reshape(4, 4)
            linear = matrix[:3, :3]
            scale = torch.linalg.vector_norm(linear[:, 0])
            rotation = linear / scale
            means = self.canonical["means"] @ linear.T + matrix[:3, 3]
            frame_quaternion = _quaternion_from_matrix(rotation)
            quaternions = _quat_multiply(
                frame_quaternion.expand(self.canonical["quaternions"].shape[0], -1),
                self.canonical["quaternions"],
            )
            quaternions /= torch.linalg.vector_norm(quaternions, dim=1, keepdim=True)
            sh = source_appearance
            if residuals is not None:
                offset = residuals[index]
                if offset.ndim == 1:
                    offset = offset.unsqueeze(0)
                coefficient_count = int(offset.shape[0])
                if coefficient_count not in {1, 4} or offset.shape[-1] != 3:
                    raise StructuringError("unsupported compact SH residual shape")
                sh = torch.cat(
                    (
                        sh[:, :coefficient_count, :] + offset,
                        sh[:, coefficient_count:, :],
                    ),
                    dim=1,
                )
            fields["means"].append(means)
            fields["quaternions"].append(quaternions)
            fields["scales"].append(self.canonical["scales"] * scale)
            fields["opacities"].append(self.canonical["opacities"])
            fields["appearance"].append(sh)
        if not fields["means"]:
            raise StructuringError("at least one active instance is required")
        return {name: torch.cat(values, dim=0) for name, values in fields.items()}

    def materialize(
        self,
        *,
        appearance: Any | None = None,
        residuals: Any | None = None,
        active_instance_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        torch = _torch()
        panels = self.panels(
            appearance=appearance,
            residuals=residuals,
            active_instance_ids=active_instance_ids,
        )
        return {
            name: torch.cat((self.background[name], panels[name]), dim=0)
            for name in FIELDS
        }


def validate_hybrid(root: Path) -> dict[str, Any]:
    manifest = load_object(root / "hybrid.json")
    background = load_object(root / "background" / "metadata.json")
    canonical = load_object(root / "canonical" / "metadata.json")
    transform_count = len(load_object(root / "instance-transforms.json")["instances"])
    if (
        background["gaussian_count"] + transform_count * canonical["gaussian_count"]
        != manifest["counts"]["hybrid_materialized"]
    ):
        raise StructuringError("hybrid materialized count mismatch")
    for metadata_root, metadata in (
        (root / "background", background),
        (root / "canonical", canonical),
    ):
        for record in metadata["files"].values():
            path = metadata_root / record["path"]
            if (
                path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                raise StructuringError("hybrid array integrity mismatch")
    return {
        "valid": True,
        "hybrid_digest": manifest["scientific_digest"],
        "background_digest": background["scientific_digest"],
        "canonical_digest": canonical["scientific_digest"],
        "counts": manifest["counts"],
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise StructuringError(
            "PyTorch is required for hybrid materialization"
        ) from error
    return torch
