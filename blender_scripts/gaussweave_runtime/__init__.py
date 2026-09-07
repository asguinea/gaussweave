"""Shared deterministic runtime for GaussWeave Blender generators.

This package is loaded by Blender's embedded Python.  The package initializer
intentionally avoids importing :mod:`bpy` so pure helpers remain testable in an
ordinary Python process.
"""

from __future__ import annotations

COLLECTION_ROLES = (
    "GAUSSWEAVE_ROOT",
    "STRUCTURE",
    "TERMINALS",
    "INSTANCES",
    "OCCLUDERS",
    "LIGHTING",
    "CAMERAS",
    "ANNOTATIONS",
    "RUNTIME_PROBE",
)
GENERATOR_METADATA_VERSION = "1.0.0"
REQUIRED_SEED_STREAMS = (
    "geometry",
    "materials",
    "lighting",
    "cameras",
    "occluders",
    "edits",
    "annotations",
)
RUNTIME_VERSION = "blender-runtime-v1"
