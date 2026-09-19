#!/usr/bin/env python3
"""Run the included MiniMax H3 API workflow and record backend completion time."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import uuid
from copy import deepcopy
from pathlib import Path
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = ROOT / "workflows/minimax_h3_fl2va_4step_nvenc_api.json"


def request_json(session: requests.Session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=60, **kwargs)
    response.raise_for_status()
    return response.json()


def upload_image(session: requests.Session, server: str, image: Path) -> str:
    with image.open("rb") as handle:
        payload = request_json(
            session,
            "POST",
            urljoin(server + "/", "upload/image"),
            files={"image": (image.name, handle, "application/octet-stream")},
            data={"type": "input", "overwrite": "true"},
        )
    return payload.get("name", image.name)


def status_timestamps(history: dict) -> tuple[float | None, float | None]:
    start = end = None
    for message in history.get("status", {}).get("messages", []):
        if not isinstance(message, list) or len(message) != 2:
            continue
        event, data = message
        timestamp = data.get("timestamp") if isinstance(data, dict) else None
        if event == "execution_start" and timestamp is not None:
            start = float(timestamp) / 1000.0
        if event == "execution_success" and timestamp is not None:
            end = float(timestamp) / 1000.0
    return start, end


def wait_for_history(session: requests.Session, server: str, prompt_id: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = request_json(session, "GET", urljoin(server + "/", f"history/{prompt_id}"))
        if prompt_id in payload:
            history = payload[prompt_id]
            status = history.get("status", {})
            if status.get("completed"):
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False))
                return history
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI did not finish prompt {prompt_id} within {timeout}s")


def output_video(history: dict) -> dict:
    output = history.get("outputs", {}).get("92", {})
    for field in ("gifs", "videos"):
        values = output.get(field, [])
        if values:
            return values[0]
    raise RuntimeError("output node 92 did not return a video")


def download_video(session: requests.Session, server: str, item: dict, target: Path) -> None:
    response = session.get(
        urljoin(server + "/", "view"),
        params={
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        },
        timeout=120,
    )
    response.raise_for_status()
    target.write_bytes(response.content)


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration,size",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr[-1000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="A cinematic product shot with gentle camera movement and natural ambient sound.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-output")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    workflow_source = json.loads(args.workflow.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    uploaded_name = upload_image(session, args.server, args.image.resolve())
    results = []

    for index in range(args.runs):
        workflow = deepcopy(workflow_source)
        workflow["141"]["inputs"]["image"] = uploaded_name
        workflow["140:131"]["inputs"].update({
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
        })
        workflow["140:133"]["inputs"]["value"] = args.duration
        workflow["140:129"]["inputs"]["noise_seed"] = args.seed + index
        workflow["92"]["inputs"]["filename_prefix"] = f"minimax_h3_nvenc/run_{index + 1:02d}"

        queued_at = time.time()
        payload = request_json(
            session,
            "POST",
            urljoin(args.server + "/", "prompt"),
            json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        )
        prompt_id = payload["prompt_id"]
        history = wait_for_history(session, args.server, prompt_id, args.timeout)
        finished_at = time.time()
        start, end = status_timestamps(history)
        item = output_video(history)
        target = args.output_dir / f"run-{index + 1:02d}-seed-{args.seed + index}.mp4"
        download_video(session, args.server, item, target)
        backend_seconds = end - start if start is not None and end is not None else None
        results.append({
            "run": index + 1,
            "seed": args.seed + index,
            "prompt_id": prompt_id,
            "backend_seconds": backend_seconds,
            "request_wall_seconds": finished_at - queued_at,
            "output": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "ffprobe": ffprobe(target),
        })
        print(f"run {index + 1}/{args.runs}: backend={backend_seconds}s output={target}")

    measured = [item["backend_seconds"] for item in results if item["backend_seconds"] is not None]
    summary = {
        "server": args.server,
        "workflow": str(args.workflow.resolve()),
        "input_image": str(args.image.resolve()),
        "settings": {
            "width": args.width,
            "height": args.height,
            "duration_seconds": args.duration,
            "steps": 4,
            "fps": 24,
            "encoder": "h264_nvenc",
            "nvenc_preset": "p1",
            "nvenc_qp": 18,
        },
        "median_backend_seconds": statistics.median(measured) if measured else None,
        "runs": results,
    }
    summary_path = args.output_dir / "benchmark-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
