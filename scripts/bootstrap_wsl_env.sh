#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
case "$repo_root" in
  /mnt/*)
    echo "error: clone the repository onto the WSL Linux filesystem, not under /mnt" >&2
    exit 1
    ;;
esac

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "error: this bootstrap script must run inside WSL" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi
if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "error: Python 3.12 is required" >&2
  exit 1
fi
if [[ ! -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  echo "error: toolkit-only cuda-toolkit-13-0 is required for gsplat" >&2
  exit 1
fi

cd "$repo_root"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
uv sync --frozen --extra dev --extra gpu --python 3.12
echo "locked WSL GPU environment is synchronized at $repo_root/.venv"
