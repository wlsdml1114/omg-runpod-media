#!/usr/bin/env python3
"""Run one plan-step-2 round: create a fresh Pod, wait for its first-read results, save, delete.

  python3 step2_round.py eu-r1 --dc EU-RO-1 --gpu "NVIDIA GeForce RTX 4090" --targets "nv=/workspace gv=/workspace-global"
  python3 step2_round.py us-r1 --dc US-NC-2 --gpu "NVIDIA RTX PRO 6000 Blackwell Server Edition" --targets "gv=/workspace-global" --no-nv

RUNPOD_API_KEY from env. Volume ids come from NV_ID and GV_ID in env. The Pod is always
deleted at the end (also on failure or timeout), and the delete is confirmed by a 404 read.
Writes results/step2-<round>.jsonl (RB_RESULT lines) and results/step2-<round>-pod.json (Pod record, no env).
"""
import argparse, base64, io, json, os, sys, tarfile, time, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://api.runpod.io/v2"
NV_ID, GV_ID = os.environ["NV_ID"], os.environ["GV_ID"]
START_TIMEOUT_S, RUN_TIMEOUT_S = 5 * 60, 25 * 60


def call(method, path, body=None, timeout=60):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:1000]


def logs(pod_id, seconds=15):
    """Pod logs are an SSE stream; read for a few seconds and return container lines."""
    req = urllib.request.Request(f"{API}/pods/{pod_id}/logs", headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]})
    lines, end = [], time.monotonic() + seconds
    try:
        with urllib.request.urlopen(req, timeout=seconds) as r:
            while time.monotonic() < end:
                raw = r.readline()
                if not raw:
                    break
                if raw.startswith(b"data: "):
                    try:
                        d = json.loads(raw[6:])
                    except ValueError:
                        continue
                    if d.get("source") != "system":
                        lines.append(d["line"])
    except (TimeoutError, OSError):
        pass
    return lines


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in ("pod_step2.sh", "read_bench.sh", "manifests"):
            t.add(HERE / name, arcname=name)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round")
    ap.add_argument("--dc", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--no-nv", action="store_true", help="global volume only (other data centers)")
    a = ap.parse_args()
    out = HERE / "results" / f"step2-{a.round}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; use a new round label")

    mounts = {"global": [{"volumeId": GV_ID, "path": "/workspace-global"}]}
    if not a.no_nv:
        mounts["network"] = [{"volumeId": NV_ID, "path": "/workspace"}]
    body = {"name": f"omg-gv-{a.round}", "cloud": "SECURE", "dataCenterIds": [a.dc],
            "gpu": {"id": a.gpu, "count": 1, "minCudaVersion": "12.8"},
            "image": "python:3.11-slim", "disk": 20, "mounts": mounts,
            "env": {"BENCH_TGZ_B64": bundle(), "ROUND": a.round, "TARGETS": a.targets},
            "entrypoint": ["bash", "-c", "mkdir -p /root/bench && cd /root/bench && printf %s \"$BENCH_TGZ_B64\" | base64 -d | tar xz && exec bash pod_step2.sh"]}
    status, pod = call("POST", "/pods", body, timeout=120)
    if status >= 300:
        sys.exit(f"create failed {status}: {pod}")
    pod_id = pod["id"]
    print(f"created {pod_id} {a.round} {pod.get('dataCenterId')} ${pod.get('cost')}/h", flush=True)
    t0, result_lines, outcome = time.monotonic(), [], "timeout"
    try:
        while time.monotonic() - t0 < RUN_TIMEOUT_S:
            lines = logs(pod_id)
            if not lines and time.monotonic() - t0 > START_TIMEOUT_S:
                outcome = "no_start"
                break
            done = [l for l in lines if l.startswith("RB_STEP2")]
            if done:
                outcome = done[-1].split()[-1]
                result_lines = [l.split("RB_RESULT ", 1)[1] for l in lines if l.startswith("RB_RESULT ")]
                for l in lines:
                    if "FAILED" in l or "read_bench:" in l:
                        print("  " + l, flush=True)
                break
            time.sleep(20)
        _, info = call("GET", f"/pods/{pod_id}")
        if isinstance(info, dict):
            info.get("env", {}).pop("BENCH_TGZ_B64", None)
            (HERE / "results" / f"step2-{a.round}-pod.json").write_text(json.dumps(info, indent=1))
        if result_lines:
            out.write_text("".join(l + "\n" for l in result_lines))
    finally:
        st, _ = call("DELETE", f"/pods/{pod_id}")
        gone, _ = call("GET", f"/pods/{pod_id}")
        print(f"deleted {pod_id}: delete {st}, get {gone}, ran {time.monotonic() - t0:.0f}s, outcome {outcome}", flush=True)
    for l in result_lines:
        d = json.loads(l)
        print(f"  {d['kind']:10} {d['volume']:3} {d.get('file', '-')[:44]:44} "
              f"{d.get('mb_per_s', '')} MB/s {d.get('seconds', d.get('read_seconds'))} s", flush=True)
    if outcome != "done":
        sys.exit(1)


if __name__ == "__main__":
    main()
