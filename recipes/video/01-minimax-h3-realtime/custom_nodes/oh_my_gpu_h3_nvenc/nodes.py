"""Minimal ComfyUI output node for H.264 encoding through NVIDIA NVENC."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from fractions import Fraction

import folder_paths
import numpy as np
import torch


class H3NVENCSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "minimax_h3_nvenc/output"}),
                "qp": ("INT", {"default": 18, "min": 0, "max": 51}),
                "preset": (["p1", "p2", "p3", "p4", "p5", "p6", "p7"], {"default": "p1"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "OhMyGPU/video"

    def save(self, video, filename_prefix: str, qp: int, preset: str):
        total_start = time.monotonic()
        components = video.get_components()
        images = components.images
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError(f"Expected an FHWC RGB tensor, got {tuple(images.shape)}")

        frame_count, height, width = images.shape[:3]
        fps = float(Fraction(components.frame_rate))
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        os.makedirs(full_output_folder, exist_ok=True)
        output_name = f"{filename}_{counter:05}_.mp4"
        output_path = os.path.join(full_output_folder, output_name)

        convert_start = time.monotonic()
        rgb = (
            images[..., :3]
            .detach()
            .float()
            .mul(255.0)
            .clamp_(0.0, 255.0)
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
            .numpy()
        )
        tensor_convert_seconds = time.monotonic() - convert_start

        audio_path = None
        audio_prepare_start = time.monotonic()
        audio = components.audio
        audio_sample_rate = None
        audio_channels = None
        if audio:
            audio_sample_rate = int(audio["sample_rate"])
            waveform = audio["waveform"][0].detach().float().to("cpu")
            audio_channels = int(waveform.shape[0])
            max_samples = int(np.ceil((frame_count / fps) * audio_sample_rate))
            waveform = waveform[:, :max_samples].clamp(-1.0, 1.0)
            pcm = (
                waveform.transpose(0, 1)
                .mul(32767.0)
                .round()
                .to(dtype=torch.int16)
                .contiguous()
                .numpy()
            )
            handle = tempfile.NamedTemporaryFile(prefix="h3-audio-", suffix=".s16le", delete=False)
            audio_path = handle.name
            try:
                handle.write(pcm.tobytes())
            finally:
                handle.close()
        audio_prepare_seconds = time.monotonic() - audio_prepare_start

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "pipe:0",
        ]
        if audio_path is not None:
            command.extend([
                "-f", "s16le", "-ar", str(audio_sample_rate),
                "-ac", str(audio_channels), "-i", audio_path,
            ])
        command.extend([
            "-map", "0:v:0",
            "-c:v", "h264_nvenc", "-preset", preset, "-tune", "ll",
            "-rc", "constqp", "-qp", str(qp), "-pix_fmt", "yuv420p",
        ])
        if audio_path is not None:
            command.extend(["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"])
        command.extend(["-movflags", "+faststart", output_path])

        encode_start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=rgb.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=180,
            )
        finally:
            if audio_path is not None:
                try:
                    os.unlink(audio_path)
                except FileNotFoundError:
                    pass
        ffmpeg_seconds = time.monotonic() - encode_start
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"FFmpeg NVENC failed ({completed.returncode}): {detail}")

        timing = {
            "schema_version": "1.0.0",
            "encoder": "ffmpeg h264_nvenc",
            "preset": preset,
            "tune": "ll",
            "rate_control": "constqp",
            "qp": qp,
            "input_tensor_device": str(images.device),
            "width": width,
            "height": height,
            "frames": frame_count,
            "fps": fps,
            "audio_sample_rate": audio_sample_rate,
            "audio_channels": audio_channels,
            "tensor_convert_seconds": tensor_convert_seconds,
            "audio_prepare_seconds": audio_prepare_seconds,
            "ffmpeg_encode_mux_seconds": ffmpeg_seconds,
            "output_node_total_seconds": time.monotonic() - total_start,
            "output_bytes": os.path.getsize(output_path),
        }
        with open(output_path + ".timing.json", "w", encoding="utf-8") as handle:
            json.dump(timing, handle, indent=2)
            handle.write("\n")
        print("[H3_NVENC_TIMING] " + json.dumps(timing, sort_keys=True), flush=True)

        return {
            "ui": {
                "gifs": [{
                    "filename": output_name,
                    "subfolder": subfolder,
                    "type": "output",
                    "format": "video/mp4",
                }]
            },
            "result": (video,),
        }


NODE_CLASS_MAPPINGS = {"H3NVENCSaveVideo": H3NVENCSaveVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"H3NVENCSaveVideo": "MiniMax H3 NVENC Save Video"}
