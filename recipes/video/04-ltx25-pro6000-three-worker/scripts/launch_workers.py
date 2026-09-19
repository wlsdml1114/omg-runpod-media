#!/usr/bin/env python3
"""Launch three isolated ComfyUI API processes that share one GPU and model store."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PYTHON = os.environ.get("COMFY_PYTHON", "/venv/bin/python")
COMFY_MAIN = os.environ.get("COMFY_MAIN", "/opt/ComfyUI/main.py")
SHARED_ROOT = Path(os.environ.get("COMFY_SHARED_ROOT", "/workspace/ComfyUI"))
WORKERS_ROOT = SHARED_ROOT / "workers"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_worker_command(*, worker_index: int, port: int, attention: str) -> list[str]:
    if attention not in {"pytorch", "sage"}:
        raise ValueError(f"Unknown attention mode: {attention}")
    worker_root = WORKERS_ROOT / str(worker_index)
    command = [
        PYTHON,
        COMFY_MAIN,
        "--listen",
        "0.0.0.0",
        "--port",
        str(port),
        "--models-directory",
        str(SHARED_ROOT / "models"),
        "--input-directory",
        str(SHARED_ROOT / "input"),
        "--output-directory",
        str(worker_root / "output"),
        "--temp-directory",
        str(worker_root / "temp"),
        "--user-directory",
        str(worker_root / "user"),
        "--preview-method",
        "none",
        "--cache-ram",
        "--reserve-vram",
        "1",
        "--async-offload",
        "2",
        "--disable-auto-launch",
        "--log-stdout",
    ]
    command.append(
        "--use-sage-attention"
        if attention == "sage"
        else "--use-pytorch-cross-attention"
    )
    return command


def stop_recorded_worker(pid_path: Path) -> None:
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid PID file: {pid_path}") from exc
    proc_path = Path(f"/proc/{pid}/cmdline")
    if not proc_path.exists():
        pid_path.unlink()
        return
    command = proc_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    if COMFY_MAIN not in command:
        raise RuntimeError(f"Refusing to stop PID {pid}; not a managed Comfy worker")
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not proc_path.exists():
            break
        time.sleep(0.25)
    else:
        raise RuntimeError(f"Managed Comfy worker PID {pid} did not stop")
    pid_path.unlink(missing_ok=True)


def wait_healthy(port: int, timeout: int = 180) -> dict:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/system_stats"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.load(response)
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Comfy worker on port {port} was not healthy after {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", choices=("pytorch", "sage"), required=True)
    parser.add_argument("--worker-count", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=3000)
    args = parser.parse_args()
    if args.worker_count < 1:
        raise ValueError("worker-count must be positive")

    workers = []
    for index in range(args.worker_count):
        worker_root = WORKERS_ROOT / str(index)
        worker_root.mkdir(parents=True, exist_ok=True)
        for name in ("output", "temp", "user"):
            (worker_root / name).mkdir(exist_ok=True)
        stop_recorded_worker(worker_root / "comfy.pid")

    for index in range(args.worker_count):
        port = args.base_port + index
        worker_root = WORKERS_ROOT / str(index)
        command = build_worker_command(
            worker_index=index, port=port, attention=args.attention
        )
        log_path = worker_root / f"comfy-{args.attention}.log"
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=str(Path(COMFY_MAIN).parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (worker_root / "comfy.pid").write_text(f"{process.pid}\n")
        workers.append(
            {
                "worker_index": index,
                "port": port,
                "pid": process.pid,
                "command": command,
                "log_path": str(log_path),
            }
        )

    for worker in workers:
        worker["system_stats"] = wait_healthy(worker["port"])

    manifest = {
        "launched_at": utc_now(),
        "attention": args.attention,
        "worker_count": args.worker_count,
        "workers": workers,
    }
    manifest_path = WORKERS_ROOT / f"launch-{args.attention}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
