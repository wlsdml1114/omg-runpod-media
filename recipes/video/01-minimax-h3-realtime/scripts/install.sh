#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/ComfyUI" >&2
  exit 2
fi

COMFY_DIR="$(cd "$1" && pwd)"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$COMFY_DIR/main.py" ]]; then
  echo "ComfyUI main.py was not found in: $COMFY_DIR" >&2
  exit 1
fi

TARGET_DIR="$COMFY_DIR/custom_nodes/oh_my_gpu_h3_nvenc"
mkdir -p "$TARGET_DIR"
cp "$PACKAGE_DIR/custom_nodes/oh_my_gpu_h3_nvenc/__init__.py" "$TARGET_DIR/"
cp "$PACKAGE_DIR/custom_nodes/oh_my_gpu_h3_nvenc/nodes.py" "$TARGET_DIR/"

echo "Installed the NVENC output node to $TARGET_DIR"
echo "Restart ComfyUI, then load workflows/minimax_h3_fl2va_4step_nvenc_ui.json"
