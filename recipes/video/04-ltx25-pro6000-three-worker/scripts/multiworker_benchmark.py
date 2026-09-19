#!/usr/bin/env python3
"""Run one or more fixed LTX-2.5 I2V jobs against independent ComfyUI workers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_workflow(base: dict, *, seed: int, label: str) -> dict:
    workflow = copy.deepcopy(base)
    workflow["398:339"]["inputs"]["noise_seed"] = seed
    workflow["75"]["inputs"]["filename_prefix"] = f"bench/{label}"
    return workflow


def validate_environment(stats: dict) -> None:
    system = stats.get("system", {})
    devices = stats.get("devices", [])
    device_name = devices[0].get("name", "") if devices else ""
    failures = []
    if system.get("comfyui_version") != "0.34.0":
        failures.append(f"ComfyUI={system.get('comfyui_version')!r}")
    if system.get("pytorch_version") != "2.13.0+cu130":
        failures.append(f"PyTorch={system.get('pytorch_version')!r}")
    if "RTX PRO 6000" not in device_name:
        failures.append(f"device={device_name!r}; expected RTX PRO 6000")
    if failures:
        raise RuntimeError("Unexpected benchmark environment: " + ", ".join(failures))


def execution_seconds(history_entry: dict) -> float | None:
    timestamps = {}
    for name, detail in history_entry.get("status", {}).get("messages", []):
        if name in {"execution_start", "execution_success"}:
            timestamps[name] = detail.get("timestamp")
    if not all(timestamps.get(name) for name in ("execution_start", "execution_success")):
        return None
    return round(
        (timestamps["execution_success"] - timestamps["execution_start"]) / 1000,
        3,
    )


def summarize_batch(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("At least one completed run is required")
    first_submit = min(run["submitted_monotonic"] for run in runs)
    last_submit = max(run["submitted_monotonic"] for run in runs)
    last_complete = max(run["completed_monotonic"] for run in runs)
    wall_seconds = last_complete - first_submit
    comfy_seconds = [
        run["comfy_execution_seconds"]
        for run in runs
        if run.get("comfy_execution_seconds") is not None
    ]
    return {
        "worker_count": len(runs),
        "batch_wall_seconds": round(wall_seconds, 3),
        "start_skew_seconds": round(last_submit - first_submit, 3),
        "videos_per_minute": round(len(runs) * 60 / wall_seconds, 3),
        "mean_comfy_execution_seconds": (
            round(statistics.mean(comfy_seconds), 3) if comfy_seconds else None
        ),
    }


def build_view_path(descriptor: dict) -> str:
    return "/view?" + urllib.parse.urlencode(
        {
            "filename": descriptor["filename"],
            "subfolder": descriptor.get("subfolder", ""),
            "type": descriptor["type"],
        }
    )


def extract_output(history_entry: dict) -> dict:
    for output in history_entry.get("outputs", {}).values():
        for descriptor in output.get("images", []):
            if descriptor.get("type") == "output":
                return descriptor
    raise RuntimeError("Comfy history did not contain an output descriptor")


def ffprobe(path: Path) -> dict | None:
    if not shutil.which("ffprobe"):
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _request(self, path: str, *, payload: dict | None = None, timeout: int = 60):
        data = None
        headers = {"User-Agent": "Mozilla/5.0"}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"Comfy HTTP {exc.code}: {body}") from exc

    def get_json(self, path: str) -> dict:
        with self._request(path, timeout=30) as response:
            return json.load(response)

    def post_json(self, path: str, payload: dict) -> dict:
        with self._request(path, payload=payload) as response:
            return json.load(response)

    def download(self, path: str, destination: Path) -> None:
        with self._request(path, timeout=300) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)


def run_worker(
    *,
    worker_index: int,
    base_url: str,
    workflow: dict,
    input_path: Path,
    artifact_dir: Path,
    label: str,
    seed: int,
    barrier: threading.Barrier,
    timeout: int,
) -> dict:
    worker_dir = artifact_dir / f"worker-{worker_index}"
    worker_dir.mkdir()
    client = ComfyClient(base_url)
    stats = client.get_json("/system_stats")
    validate_environment(stats)
    (worker_dir / "system-stats-initial.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    )

    prepared = prepare_workflow(workflow, seed=seed, label=label)
    (worker_dir / "submitted-workflow.json").write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2) + "\n"
    )
    remote_input_name = prepared["395"]["inputs"]["image"]
    remote_input = worker_dir / f"remote-{Path(remote_input_name).name}"
    client.download(
        build_view_path(
            {"filename": remote_input_name, "subfolder": "", "type": "input"}
        ),
        remote_input,
    )
    input_sha256 = sha256_file(input_path)
    if sha256_file(remote_input) != input_sha256:
        raise RuntimeError(f"Worker {worker_index} remote input hash mismatch")

    client_id = str(uuid.uuid4())
    barrier.wait()
    submitted_at = utc_now()
    submitted_monotonic = time.perf_counter()
    queued = client.post_json("/prompt", {"prompt": prepared, "client_id": client_id})
    if queued.get("node_errors"):
        raise RuntimeError(json.dumps(queued["node_errors"], ensure_ascii=False))
    prompt_id = queued["prompt_id"]
    history_entry = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = client.get_json(f"/history/{prompt_id}")
        if prompt_id in history:
            history_entry = history[prompt_id]
            status = history_entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "error":
                break
        time.sleep(0.25)
    else:
        raise TimeoutError(f"Worker {worker_index} exceeded {timeout} seconds")

    completed_monotonic = time.perf_counter()
    completed_at = utc_now()
    (worker_dir / "history.json").write_text(
        json.dumps(history_entry, ensure_ascii=False, indent=2) + "\n"
    )
    status = (history_entry or {}).get("status", {})
    if status.get("status_str") != "success":
        raise RuntimeError(f"Worker {worker_index} failed: {status}")

    descriptor = extract_output(history_entry)
    output_path = worker_dir / Path(descriptor["filename"]).name
    client.download(build_view_path(descriptor), output_path)
    result = {
        "worker_index": worker_index,
        "base_url": base_url,
        "prompt_id": prompt_id,
        "client_id": client_id,
        "label": label,
        "seed": seed,
        "submitted_at": submitted_at,
        "completed_at": completed_at,
        "submitted_monotonic": submitted_monotonic,
        "completed_monotonic": completed_monotonic,
        "wall_seconds": round(completed_monotonic - submitted_monotonic, 3),
        "comfy_execution_seconds": execution_seconds(history_entry),
        "status": status,
        "input": {"remote_filename": remote_input_name, "sha256": input_sha256},
        "workflow_sha256": sha256_file(worker_dir / "submitted-workflow.json"),
        "output": {
            "descriptor": descriptor,
            "local_path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "ffprobe": ffprobe(output_path),
        },
    }
    (worker_dir / "benchmark-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


def monitor_global_vram(client: ComfyClient, stop: threading.Event, path: Path) -> None:
    with path.open("w") as handle:
        while not stop.is_set():
            try:
                stats = client.get_json("/system_stats")
                device = stats.get("devices", [{}])[0]
                sample = {
                    "captured_at": utc_now(),
                    "vram_total": device.get("vram_total"),
                    "vram_free": device.get("vram_free"),
                    "torch_vram_total": device.get("torch_vram_total"),
                    "torch_vram_free": device.get("torch_vram_free"),
                }
                handle.write(json.dumps(sample) + "\n")
                handle.flush()
            except Exception as exc:  # Preserve monitoring failures without aborting jobs.
                handle.write(json.dumps({"captured_at": utc_now(), "error": str(exc)}) + "\n")
                handle.flush()
            stop.wait(0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    if args.artifact_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable artifact directory: {args.artifact_dir}"
        )
    args.artifact_dir.mkdir(parents=True)
    workflow = json.loads(args.workflow.read_text())
    worker_count = len(args.base_url)
    barrier = threading.Barrier(worker_count)
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_global_vram,
        args=(
            ComfyClient(args.base_url[0]),
            monitor_stop,
            args.artifact_dir / "global-vram-samples.jsonl",
        ),
        daemon=True,
    )
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    run_worker,
                    worker_index=index,
                    base_url=base_url,
                    workflow=workflow,
                    input_path=args.input,
                    artifact_dir=args.artifact_dir,
                    label=f"{args.label}-w{index}",
                    seed=args.seed_base + index,
                    barrier=barrier,
                    timeout=args.timeout,
                )
                for index, base_url in enumerate(args.base_url)
            ]
            runs = [future.result() for future in futures]
    finally:
        monitor_stop.set()
        monitor.join(timeout=5)

    summary = {
        "label": args.label,
        "created_at": utc_now(),
        "workflow_source": {
            "path": str(args.workflow.resolve()),
            "sha256": sha256_file(args.workflow),
        },
        "input_source": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
        },
        "batch": summarize_batch(runs),
        "runs": runs,
    }
    (args.artifact_dir / "batch-result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
