#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: MINIMAX_H3_LICENSE_APPROVED=1 $0 /path/to/ComfyUI" >&2
  exit 2
fi

if [[ "${MINIMAX_H3_LICENSE_APPROVED:-}" != "1" ]]; then
  echo "Stopped: this script does not grant a MiniMax H3 license." >&2
  echo "For deployment in Korea, submit https://platform.minimax.io/h3-license and obtain separate authorization from MiniMax first." >&2
  echo "Form submission alone is not approval. After approval, rerun with MINIMAX_H3_LICENSE_APPROVED=1." >&2
  exit 3
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI ('hf') is required. Install it with: pip install -U huggingface_hub" >&2
  exit 1
fi

COMFY_DIR="$(cd "$1" && pwd)"
if [[ ! -f "$COMFY_DIR/main.py" ]]; then
  echo "ComfyUI main.py was not found in: $COMFY_DIR" >&2
  exit 1
fi

hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
  --local-dir "$COMFY_DIR/models"

echo "Model files downloaded into $COMFY_DIR/models"
