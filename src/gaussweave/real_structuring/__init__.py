"""Reusable oracle real-region structural refitting primitives.

CPU validators in this package intentionally avoid importing PyTorch and gsplat.
GPU dependencies are loaded only inside extraction, fitting, and evaluation calls.
"""

from gaussweave.real_structuring.models import (
    INSTANCE_IDS,
    PILOT_VERSION,
    OwnershipStatus,
    PanelFrame,
    Q8Residuals,
    SimilarityRegistration,
)

__all__ = [
    "INSTANCE_IDS",
    "PILOT_VERSION",
    "OwnershipStatus",
    "PanelFrame",
    "Q8Residuals",
    "SimilarityRegistration",
]
