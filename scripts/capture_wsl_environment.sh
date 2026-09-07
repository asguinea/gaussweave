#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
output="${1:-artifacts/environment/wsl-live.json}"
cd "$repo_root"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
uv run --frozen --extra dev --extra gpu \
  python -m gaussweave.runtime.environment --json --output "$output"
echo "environment record written to $output"
