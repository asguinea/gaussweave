"""Formal CLI for reusable real-region structural operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gaussweave.real_structuring.accounting import account
from gaussweave.real_structuring.extraction import extract_region
from gaussweave.real_structuring.fitting import fit
from gaussweave.real_structuring.hybrid import build_hybrid, validate_hybrid
from gaussweave.real_structuring.models import StructuringError, load_object
from gaussweave.real_structuring.regions import validate_region_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--qualification", required=True, type=Path)
    extract.add_argument("--dataset-root", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--json", action="store_true")
    register = commands.add_parser("register")
    register.add_argument("--region-root", required=True, type=Path)
    register.add_argument("--json", action="store_true")
    build = commands.add_parser("build-hybrid")
    build.add_argument("--region-root", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--json", action="store_true")
    fitting = commands.add_parser("fit")
    fitting.add_argument("--hybrid-root", required=True, type=Path)
    fitting.add_argument(
        "--method",
        required=True,
        choices=("real_struct_shared", "real_struct_residual_q8"),
    )
    fitting.add_argument("--resume", action="store_true")
    fitting.add_argument("--overwrite", action="store_true")
    fitting.add_argument("--json", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--hybrid-root", required=True, type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--kind", choices=("region", "hybrid", "accounting"), default="region"
    )
    validate.add_argument("--json", action="store_true")
    return parser


def _run(options: argparse.Namespace) -> dict[str, Any]:
    if options.command == "extract":
        return extract_region(
            qualification=options.qualification,
            dataset_root=options.dataset_root,
            output=options.output,
            overwrite=options.overwrite,
        )
    if options.command == "register":
        validation = validate_region_root(options.region_root)
        return {
            "valid": True,
            "region": validation,
            "registration": load_object(options.region_root / "registration.json"),
        }
    if options.command == "build-hybrid":
        return build_hybrid(
            region_root=options.region_root,
            output=options.output,
            overwrite=options.overwrite,
        )
    if options.command == "fit":
        return fit(
            hybrid_root=options.hybrid_root,
            method=options.method,
            resume=options.resume,
            overwrite=options.overwrite,
        )
    if options.command == "evaluate":
        from gaussweave.real_structuring.evaluation import evaluate

        result = evaluate(hybrid_root=options.hybrid_root, overwrite=options.overwrite)
        result["accounting"] = account(options.hybrid_root)
        return result
    if options.kind == "region":
        return validate_region_root(options.path)
    if options.kind == "hybrid":
        return validate_hybrid(options.path)
    return load_object(options.path / "accounting-summary.json")


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        result = _run(options)
    except (StructuringError, FileExistsError, OSError, ValueError) as error:
        value = {"valid": False, "error": str(error), "command": options.command}
        print(json.dumps(value, sort_keys=True) if options.json else str(error))
        return 1
    print(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if options.json
        else f"{options.command}: valid; output={result.get('output', 'validated')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
