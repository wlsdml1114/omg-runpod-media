#!/usr/bin/env bash
# Pod 기본 도구와 캐시 위치를 준비합니다. 비밀정보를 다루지 않습니다.
# 사용: source scripts/setup_pod.sh
set -euo pipefail

export WORK_ROOT="${WORK_ROOT:-/workspace/ablation}"
export UV_CACHE_DIR=/workspace/cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/cache/uv-python
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=4
export HF_HOME=/workspace/cache/hf
export HF_XET_CACHE=/workspace/cache/xet
export TRITON_CACHE_DIR=/workspace/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/root/.local/bin:$PATH"

mkdir -p "$WORK_ROOT" /workspace/cache

apt-get update
apt-get install -y ffmpeg git curl rsync
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.9.0/install.sh | sh
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv --version
ffmpeg -hide_banner -version | head -1
