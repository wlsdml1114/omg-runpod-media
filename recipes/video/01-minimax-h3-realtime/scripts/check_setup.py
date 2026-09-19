#!/usr/bin/env python3
"""Validate the template files and, optionally, a ComfyUI installation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_WORKFLOW = ROOT / "workflows/minimax_h3_fl2va_4step_nvenc_api.json"
UI_WORKFLOW = ROOT / "workflows/minimax_h3_fl2va_4step_nvenc_ui.json"
NODE_FILE = ROOT / "custom_nodes/oh_my_gpu_h3_nvenc/nodes.py"

MODELS = (
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
    "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def check_static() -> None:
    for path in (API_WORKFLOW, UI_WORKFLOW, NODE_FILE):
        if not path.is_file():
            fail(f"missing file: {path.relative_to(ROOT)}")

    api = json.loads(API_WORKFLOW.read_text(encoding="utf-8"))
    ui = json.loads(UI_WORKFLOW.read_text(encoding="utf-8"))
    if api.get("92", {}).get("class_type") != "H3NVENCSaveVideo":
        fail("API workflow does not use H3NVENCSaveVideo at node 92")
    if api.get("140:138", {}).get("inputs", {}).get("value") != 4:
        fail("API workflow is not configured for four sampling steps")
    if api.get("140:139", {}).get("inputs", {}).get("value") is not True:
        fail("API workflow does not enable the turbo LoRA path")

    ui_nodes = {node["id"]: node for node in ui.get("nodes", [])}
    if ui_nodes.get(92, {}).get("type") != "H3NVENCSaveVideo":
        fail("UI workflow does not use H3NVENCSaveVideo at node 92")
    print("OK: template JSON and required node files")


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        fail(f"command failed: {' '.join(command)}\n{result.stderr[-1000:]}")
    return result.stdout + result.stderr


def check_runtime(comfyui: Path) -> None:
    if not (comfyui / "main.py").is_file():
        fail(f"ComfyUI main.py not found in {comfyui}")
    for command in ("ffmpeg", "nvidia-smi"):
        if shutil.which(command) is None:
            fail(f"required command not found: {command}")
    if "h264_nvenc" not in command_output(["ffmpeg", "-hide_banner", "-encoders"]):
        fail("this FFmpeg build does not expose h264_nvenc")
    command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])

    missing = [str(comfyui / "models" / relative) for relative in MODELS if not (comfyui / "models" / relative).is_file()]
    if missing:
        fail("missing model files:\n  " + "\n  ".join(missing))
    installed_node = comfyui / "custom_nodes/oh_my_gpu_h3_nvenc/nodes.py"
    if not installed_node.is_file():
        fail(f"custom node is not installed: {installed_node}")
    print("OK: ComfyUI, model paths, NVIDIA GPU and NVENC encoder")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="check package structure only")
    parser.add_argument("--comfyui", type=Path, help="path to the ComfyUI directory")
    args = parser.parse_args()
    check_static()
    if args.static:
        return
    if args.comfyui is None:
        parser.error("--comfyui is required unless --static is used")
    check_runtime(args.comfyui.resolve())


if __name__ == "__main__":
    main()
