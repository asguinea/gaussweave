"""Independent named seed streams accepted from scene configuration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from . import REQUIRED_SEED_STREAMS

UINT32_MAX = 2**32 - 1


class SeedError(ValueError):
    """Resolved seed metadata is incomplete or invalid."""


@dataclass
class SeedRegistry:
    """Own independent Python and optional NumPy RNGs for every stream."""

    master_seed: int
    seeds: dict[str, int]
    python_rngs: dict[str, random.Random]
    numpy_rngs: dict[str, Any]
    numpy_version: str | None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SeedRegistry:
        master = config.get("master_seed")
        derived = config.get("derived_seeds")
        if not _uint32(master):
            raise SeedError("master_seed must be an unsigned 32-bit integer")
        if not isinstance(derived, dict):
            raise SeedError("derived_seeds must be an object")
        missing = [name for name in REQUIRED_SEED_STREAMS if name not in derived]
        extra = sorted(set(derived) - set(REQUIRED_SEED_STREAMS))
        if missing:
            raise SeedError("missing seed streams: " + ", ".join(missing))
        if extra:
            raise SeedError("unexpected seed streams: " + ", ".join(extra))
        seeds: dict[str, int] = {}
        for name in REQUIRED_SEED_STREAMS:
            value = derived[name]
            if not _uint32(value):
                raise SeedError(f"invalid seed for {name}")
            seeds[name] = value
        python_rngs = {name: random.Random(seed) for name, seed in seeds.items()}
        numpy_rngs: dict[str, Any] = {}
        numpy_version: str | None = None
        try:
            import numpy

            numpy_version = str(numpy.__version__)
            numpy_rngs = {
                name: numpy.random.default_rng(seed) for name, seed in seeds.items()
            }
        except ImportError:
            pass
        return cls(master, seeds, python_rngs, numpy_rngs, numpy_version)

    def python(self, stream: str) -> random.Random:
        """Return the mutable isolated Python generator for one stream."""

        try:
            return self.python_rngs[stream]
        except KeyError as error:
            raise SeedError(f"unknown seed stream: {stream}") from error

    def numpy(self, stream: str) -> Any:
        """Return the isolated NumPy generator when bundled NumPy is available."""

        if not self.numpy_rngs:
            raise SeedError("NumPy randomness is unavailable")
        try:
            return self.numpy_rngs[stream]
        except KeyError as error:
            raise SeedError(f"unknown seed stream: {stream}") from error

    def metadata(self) -> dict[str, Any]:
        """Return reproducible initial-state metadata without consuming live RNGs."""

        return {
            "master_seed": self.master_seed,
            "derived_seeds": dict(sorted(self.seeds.items())),
            "python_algorithm": "random.Random/MT19937",
            "numpy_algorithm": (
                "numpy.random.default_rng/PCG64" if self.numpy_version else None
            ),
            "numpy_version": self.numpy_version,
            "initial_samples": {
                name: {
                    "python_uniform": round(random.Random(seed).random(), 15),
                    "python_uint32": random.Random(seed).getrandbits(32),
                }
                for name, seed in sorted(self.seeds.items())
            },
        }


def _uint32(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= UINT32_MAX
    )
