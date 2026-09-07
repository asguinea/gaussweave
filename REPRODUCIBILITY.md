# Reproducibility guide

GaussWeave separates lightweight software verification from workflows that require
specialized hardware or third-party assets. Begin with the lowest applicable tier
and record every local deviation from the locked setup.

## 1. CPU contract suite

This tier validates configuration, schemas, deterministic camera math, artifact
integrity, lifecycle records, serialization, metrics, and lightweight adapters.

```bash
uv sync --frozen --extra dev --python 3.12
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Expected outcome: all commands exit with code zero. The pytest configuration
excludes `gpu`, `slow`, and `blender` markers by default.

## 2. CUDA and gsplat smoke suite

Use Linux or WSL 2 with the checkout on a native Linux filesystem. The locked GPU
environment expects CUDA Toolkit 13.0 and an NVIDIA driver compatible with the
PyTorch CUDA 13.0 wheel.

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
./scripts/bootstrap_wsl_env.sh
uv run --frozen --extra dev --extra gpu python scripts/verify_gpu_env.py
uv run --frozen --extra dev --extra gpu pytest -m gpu tests/gpu
```

The verification script performs real CUDA tensor work and one gsplat
rasterization. Capture the live environment before retaining performance results:

```bash
./scripts/capture_wsl_environment.sh artifacts/environment/wsl-live.json
```

Compare the result with [`environment/wsl-qualification.json`](environment/wsl-qualification.json)
and document meaningful differences. The hardware limits in
[`configs/hardware/g14-standard.json`](configs/hardware/g14-standard.json) are a
reference profile, not a portable performance guarantee.

## 3. Deterministic synthetic generation

Install Blender 4.5 LTS and make `blender` available on `PATH`, or pass an explicit
executable to the command. A minimal deterministic qualification is:

```bash
uv run gaussweave blender qualify \
  --root artifacts/blender-qualification \
  --backend wsl \
  --device cpu \
  --seed 17 \
  --json
```

The scene configurations under [`configs/scenes`](configs/scenes) define fixed
seeds and camera intent. Outputs belong under the ignored `artifacts/` directory.
Use the artifact inventory commands to checksum a retained run:

```bash
uv run gaussweave artifact inventory \
  --root artifacts/blender-qualification \
  --output artifacts/blender-qualification/inventory.json \
  --exclude-hidden \
  --overwrite \
  --json
uv run gaussweave artifact verify \
  --root artifacts/blender-qualification \
  --inventory artifacts/blender-qualification/inventory.json \
  --strict \
  --json
```

## 4. Real-scene workflows

Real-scene commands require assets that are intentionally absent from this
repository. Obtain each dataset and pretrained representation from its upstream
source, review its terms, and keep it under the ignored `datasets/` tree on a Linux
filesystem. See [`DATA.md`](DATA.md).

These workflows encode strict dataset identities, file digests, camera conventions,
and structured-region validations. Treat a failed identity or digest check as a
reproducibility failure; do not bypass it silently.

## Determinism and provenance

- Principal synthetic seeds are `17`, `29`, and `43` where a workflow requires a
  multi-seed protocol.
- Configuration resolution writes canonical JSON and SHA-256 digests.
- Artifact inventories record relative paths, byte counts, and SHA-256 digests.
- Run lifecycle records preserve both successful and failed attempts.
- The lockfile is authoritative. Use `--frozen` and disclose any dependency change.
- Generated data, renders, checkpoints, and environment captures remain untracked.

For a durable record, retain the Git commit, resolved configuration, random seeds,
environment capture, command line, logs, result records, and artifact inventory
together.
