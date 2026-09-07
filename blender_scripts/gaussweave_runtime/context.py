"""Generator context shared by Blender-side runtime operations."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import content_digest
from .naming import NamingRegistry
from .seeding import SeedRegistry


@dataclass
class GeneratorContext:
    """Mutable registries plus immutable resolved configuration identity."""

    config: dict[str, Any]
    output_root: Path
    generator_version: str
    backend: str
    render_device: str
    blender_version: str
    python_version: str
    naming: NamingRegistry = field(default_factory=NamingRegistry)
    seeds: SeedRegistry | None = None
    collections: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    materials: list[dict[str, Any]] = field(default_factory=list)
    camera: dict[str, Any] | None = None
    cameras: list[dict[str, Any]] = field(default_factory=list)
    lights: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compute: dict[str, Any] = field(default_factory=dict)
    purpose: str = "runtime-probe engineering fixture; not a benchmark family scene"

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        output_root: Path,
        *,
        generator_version: str,
        backend: str,
        render_device: str,
        blender_version: str,
    ) -> GeneratorContext:
        context = cls(
            config,
            output_root,
            generator_version,
            backend,
            render_device,
            blender_version,
            sys.version,
        )
        context.seeds = SeedRegistry.from_config(config)
        return context

    @property
    def scene_id(self) -> str:
        return str(self.config["scene_id"])

    @property
    def family(self) -> str:
        return str(self.config["family"])

    @property
    def master_seed(self) -> int:
        return int(self.config["master_seed"])

    @property
    def configuration_digest(self) -> str:
        return content_digest(self.config)

    def register_artifact(self, artifact_id: str, path: str, kind: str) -> None:
        if any(item["path"] == path for item in self.artifacts):
            raise ValueError(f"duplicate artifact path: {path}")
        self.artifacts.append({"artifact_id": artifact_id, "path": path, "type": kind})
