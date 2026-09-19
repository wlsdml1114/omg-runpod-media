#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/workspace/ComfyUI}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face 'hf' CLI is required." >&2
  exit 1
fi

hf download Lightricks/LTX-2.5 \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir "$COMFY_ROOT/models"

hf download Comfy-Org/gemma-4 \
  text_encoders/gemma4_e2b_it_bf16.safetensors \
  --local-dir "$COMFY_ROOT/models"
