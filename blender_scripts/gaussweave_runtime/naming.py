"""Deterministic Blender data-block naming."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
KIND_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class NamingError(ValueError):
    """A stable identity or generated name is invalid or duplicated."""


@dataclass
class NamingRegistry:
    """Reserve deterministic names for stable IDs without UUIDs or hash()."""

    names: dict[tuple[str, str], str] = field(default_factory=dict)
    stable_ids: dict[str, str] = field(default_factory=dict)
    owners: dict[str, tuple[str, str]] = field(default_factory=dict)

    def reserve(
        self,
        kind: str,
        stable_id: str,
        *,
        explicit_name: str | None = None,
        allow_reuse: bool = False,
    ) -> str:
        if not KIND_RE.fullmatch(kind):
            raise NamingError(f"invalid naming kind: {kind!r}")
        if not TOKEN_RE.fullmatch(stable_id):
            raise NamingError(f"invalid stable ID: {stable_id!r}")
        key = (kind, stable_id)
        if stable_id in self.stable_ids and self.stable_ids[stable_id] != kind:
            raise NamingError(f"duplicate stable ID: {stable_id}")
        expected = explicit_name or f"SS_{kind}_{stable_id.upper().replace('-', '_')}"
        existing = self.names.get(key)
        if existing is not None:
            if existing != expected:
                raise NamingError(f"conflicting name for stable ID: {stable_id}")
            if allow_reuse:
                return existing
            raise NamingError(f"duplicate stable ID: {stable_id}")
        owner = self.owners.get(expected)
        if owner is not None and owner != key:
            raise NamingError(f"duplicate generated name: {expected}")
        self.names[key] = expected
        self.stable_ids[stable_id] = kind
        self.owners[expected] = key
        return expected

    def records(self) -> list[dict[str, str]]:
        """Return stable sorted registry records."""

        return [
            {"kind": kind, "stable_id": stable_id, "name": name}
            for (kind, stable_id), name in sorted(self.names.items())
        ]
