#!/usr/bin/env python3
"""Submit one LTX 2.5 ComfyUI run and block until its verified output exists."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websocket


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def select_prompt(path: Path, prompt_id: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    for item in document["prompts"]:
        if item["prompt_id"] == prompt_id:
            return item
    raise ValueError(f"prompt_id not found: {prompt_id}")


def build_prompt(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = json.loads(args.base.read_text(encoding="utf-8"))
    prompt_spec = select_prompt(args.prompts, args.prompt_id)
    prompt["405:376"]["inputs"]["value"] = prompt_spec["prompt"]
    prompt["405:383"]["inputs"]["value"] = args.prompt_enhance
    prompt["405:362"]["inputs"]["value"] = args.duration
    prompt["405:361"]["inputs"]["value"] = args.fps
    prompt["405:339"]["inputs"]["noise_seed"] = args.seed
    prompt["405:372"]["inputs"]["value"] = args.width
    prompt["405:360"]["inputs"]["value"] = args.height
    prompt["75"]["inputs"]["filename_prefix"] = args.filename_prefix
    request_summary = {
        "prompt_id": args.prompt_id,
        "prompt": prompt_spec["prompt"],
        "negative_prompt_reference": prompt_spec.get("negative_prompt"),
        "seed": args.seed,
        "stage_2_seed": prompt["405:338"]["inputs"]["noise_seed"],
        "width": args.width,
        "height": args.height,
        "duration_seconds": args.duration,
        "fps": args.fps,
        "expected_frames": round(args.duration * args.fps) + 1,
        "prompt_enhance": args.prompt_enhance,
        "first_stage_sigmas": prompt["405:404"]["inputs"]["sigmas"],
        "second_stage_sigmas": prompt["405:395"]["inputs"]["sigmas"],
        "filename_prefix": args.filename_prefix,
    }
    return prompt, request_summary


def find_output(history_entry: dict[str, Any], output_root: Path) -> Path:
    candidates: list[Path] = []
    for node_output in history_entry.get("outputs", {}).values():
        if not isinstance(node_output, dict):
            continue
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                subfolder = str(item.get("subfolder", ""))
                candidates.append(output_root / subfolder / str(item["filename"]))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"no output file found in history: {candidates}")
    videos = [path for path in existing if path.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}]
    return videos[0] if videos else existing[0]


def parse_backend_times(history_entry: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for message in history_entry.get("status", {}).get("messages", []):
        if not isinstance(message, list) or len(message) < 2 or not isinstance(message[1], dict):
            continue
        name, data = message[0], message[1]
        if name == "execution_start" and data.get("timestamp") is not None:
            result["execution_start_epoch_ms"] = data["timestamp"]
        if name == "execution_success" and data.get("timestamp") is not None:
            result["execution_success_epoch_ms"] = data["timestamp"]
    if "execution_start_epoch_ms" in result and "execution_success_epoch_ms" in result:
        result["execution_seconds"] = (
            result["execution_success_epoch_ms"] - result["execution_start_epoch_ms"]
        ) / 1000
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="127.0.0.1:7860")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--prompt-enhance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filename-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/ComfyUI/output"))
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt, request_summary = build_prompt(args)
    client_id = str(uuid.uuid4())
    submitted_at = utc_now()
    start = time.monotonic()
    events: list[dict[str, Any]] = []
    ws = websocket.create_connection(f"ws://{args.server}/ws?clientId={client_id}", timeout=1)
    try:
        response = http_json(
            f"http://{args.server}/prompt",
            {"prompt": prompt, "client_id": client_id},
        )
        prompt_id = response["prompt_id"]
        accepted_at = utc_now()
        while time.monotonic() - start < args.timeout:
            try:
                raw = ws.recv()
                if isinstance(raw, str):
                    event = json.loads(raw)
                    if event.get("data", {}).get("prompt_id") in {None, prompt_id}:
                        events.append({"observed_at_utc": utc_now(), "event": event})
                    if event.get("type") == "execution_error" and event.get("data", {}).get("prompt_id") == prompt_id:
                        raise RuntimeError(json.dumps(event, ensure_ascii=False))
            except websocket.WebSocketTimeoutException:
                pass
            history = http_json(f"http://{args.server}/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False))
                if status.get("completed"):
                    break
        else:
            raise TimeoutError(f"ComfyUI prompt timed out after {args.timeout}s: {prompt_id}")
    finally:
        ws.close()

    source = find_output(entry, args.output_root)
    shutil.copy2(source, args.output)
    completed_at = utc_now()
    evidence = {
        "client_id": client_id,
        "prompt_id": prompt_id,
        "submitted_at_utc": submitted_at,
        "accepted_at_utc": accepted_at,
        "completed_at_utc": completed_at,
        "client_wall_seconds": time.monotonic() - start,
        "backend_timing": parse_backend_times(entry),
        "request_summary": request_summary,
        "source_output": str(source),
        "copied_output": str(args.output),
    }
    (args.output.parent / "request.json").write_text(
        json.dumps({"prompt": prompt, "client_id": client_id}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output.parent / "api-response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output.parent / "events.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events), encoding="utf-8"
    )
    (args.output.parent / "history.json").write_text(
        json.dumps({prompt_id: entry}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output.parent / "client-summary.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
