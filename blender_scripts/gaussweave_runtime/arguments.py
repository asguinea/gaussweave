"""Pure argument parsing for the generic Blender entry point."""

from __future__ import annotations

import argparse


def parse_runtime_arguments(arguments: list[str]) -> argparse.Namespace:
    """Parse the explicit argument vector following Blender's separator."""

    parser = argparse.ArgumentParser(description="GaussWeave Blender generator runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("runtime-probe", "camera-probe", "render-pass-probe"),
    )
    parser.add_argument("--cameras")
    parser.add_argument("--render-policy")
    parser.add_argument("--generator-version", required=True)
    parser.add_argument(
        "--provenance",
        "--provenance-ref",
        dest="provenance",
        required=True,
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--master-seed", required=True, type=int)
    parser.add_argument("--backend", required=True, choices=("wsl", "windows"))
    parser.add_argument("--device", required=True, choices=("cpu", "gpu"))
    parser.add_argument(
        "--inject-failure",
        choices=("after-reset", "before-validation", "validation"),
    )
    return parser.parse_args(arguments)
