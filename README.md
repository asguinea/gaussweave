# GaussWeave

[![CI](https://github.com/asguinea/gaussweave/actions/workflows/ci.yml/badge.svg)](https://github.com/asguinea/gaussweave/actions/workflows/ci.yml)
[![CodeQL](https://github.com/asguinea/gaussweave/actions/workflows/codeql.yml/badge.svg)](https://github.com/asguinea/gaussweave/actions/workflows/codeql.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

GaussWeave is a typed Python toolkit for reproducible experiments with structured
3D Gaussian scene representations. It combines deterministic scene and camera
generation, Gaussian rendering adapters, resource accounting, artifact integrity
checks, experiment lifecycle records, and structured-region operations behind one
command-line interface.

The repository is an early public release. APIs and file formats may evolve before
version 1.0.

## Highlights

- Deterministic synthetic scenes, cameras, and Blender runtime helpers.
- Typed Gaussian data models, serialization, rendering, and fidelity metrics.
- Reproducible configuration resolution with JSON Schema validation.
- Checksummed artifact inventories and durable success/failure records.
- CPU-only contract tests plus opt-in CUDA, Blender, and real-scene workflows.
- Locked Python dependencies and recorded reference-environment metadata.

## Requirements

- Linux or WSL 2 on x86-64.
- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/) 0.11.x.
- Optional GPU path: a CUDA-capable NVIDIA GPU, CUDA Toolkit 13.0, and a compatible
  driver.
- Optional generation path: Blender 4.5 LTS.

Native Windows is not currently a supported execution environment. For GPU work in
WSL, keep the checkout on the Linux filesystem rather than under `/mnt`.

## Quick start

```bash
git clone https://github.com/asguinea/gaussweave.git
cd gaussweave
uv sync --frozen --extra dev
uv run gaussweave --help
uv run gaussweave config validate configs/experiments/smoke.json
uv run pytest
```

The default test command excludes cases marked `gpu`, `slow`, or `blender`. Run the quality
checks used in continuous integration with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

## GPU environment

The locked GPU extra uses PyTorch 2.13.0 with CUDA 13.0 and gsplat 1.5.3:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
uv sync --frozen --extra dev --extra gpu --python 3.12
uv run --frozen --extra dev --extra gpu python scripts/verify_gpu_env.py
uv run --frozen --extra dev --extra gpu pytest -m gpu tests/gpu
```

[`scripts/bootstrap_wsl_env.sh`](scripts/bootstrap_wsl_env.sh) performs the same
environment checks and synchronization for WSL.

## Reproducibility

Start with [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the supported validation
tiers, exact commands, expected artifacts, and known limitations. The committed
[`uv.lock`](uv.lock), schemas, deterministic seeds, hardware profile, and reference
environment records make dependency and execution assumptions explicit.

No external datasets, pretrained models, generated results, or other restricted
artifacts are included. See [`DATA.md`](DATA.md) before running a real-scene
workflow.

## Command-line interface

`gaussweave --help` lists the top-level command groups. Each group provides its own
help, for example:

```bash
uv run gaussweave scene --help
uv run gaussweave camera --help
uv run gaussweave artifact --help
uv run gaussweave experiment --help
uv run gaussweave real-region --help
```

Machine-readable commands accept `--json` where appropriate and use stable exit
codes for validation, environment, resource, operation, and artifact-integrity
failures.

## Repository layout

```text
blender_scripts/   deterministic headless Blender runtime
configs/           experiment, hardware, and scene examples
environment/       reference execution-environment records
scripts/           environment bootstrap and verification helpers
src/gaussweave/    installable Python package and JSON Schemas
tests/              unit, integration, and opt-in GPU tests
```

## Contributing and security

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report vulnerabilities privately by
following [`SECURITY.md`](SECURITY.md).

## Citation

If GaussWeave supports your work, cite the software metadata in
[`CITATION.cff`](CITATION.cff). GitHub also exposes this through **Cite this
repository**.

## License

Copyright 2026 Alejandro Sanchez Guinea.

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
