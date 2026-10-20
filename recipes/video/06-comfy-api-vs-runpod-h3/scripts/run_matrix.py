#!/usr/bin/env python3
"""Run the measurement matrix in order, checking that workers are at zero before every cold run.

Order (each warm burst follows its own cold run immediately, so Runpod's 5 s idle timeout
cannot turn a warm run into a cold one):
  runpod cold-1, runpod warm-1..5, comfy cold-1, comfy warm-1..5,
  runpod cold-2, comfy cold-2, runpod cold-3, comfy cold-3

Zero check before a cold run:
  runpod: /health shows running 0, initializing 0, inProgress 0, inQueue 0
  comfy:  a `comfy deploy status` snapshot sampled >= 30 s after our last job finished shows
          running 0, initializing 0, jobsInQueue 0 (snapshots refresh about every 5 min)
Idle slots are recorded, not required to be zero.

  python3 scripts/run_matrix.py            # run everything not already in benchmark-output/runs/
  python3 scripts/run_matrix.py --dry-run  # print the plan
Env: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, COMFY_API_KEY, COMFY_CLOUD_API_KEY, COMFY_BASE_URL
"""
import argparse, json, os, subprocess, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE.parent
OUT = RECIPE / "benchmark-output"
LOG = OUT / "matrix-log.jsonl"
MIN_GAP_S = 60          # never start a cold run sooner than this after the previous job on that side
MAX_ZERO_WAIT_S = 20 * 60

PLAN = (
    [("runpod", "cold-1", 42101)] + [("runpod", f"warm-{i}", 42110 + i) for i in range(1, 6)]
    + [("comfy", "cold-1", 42101)] + [("comfy", f"warm-{i}", 42110 + i) for i in range(1, 6)]
    + [("runpod", "cold-2", 42102), ("comfy", "cold-2", 42102),
       ("runpod", "cold-3", 42103), ("comfy", "cold-3", 42103)]
)


def now():
    return datetime.now(timezone.utc)


def log(**kw):
    kw["at"] = now().isoformat(timespec="seconds")
    line = json.dumps(kw, ensure_ascii=False)
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def runpod_health():
    ep, key = os.environ["RUNPOD_ENDPOINT_ID"], os.environ["RUNPOD_API_KEY"]
    req = urllib.request.Request(f"https://api.runpod.ai/v2/{ep}/health", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def comfy_status():
    out = subprocess.run(["comfy", "deploy", "status"], cwd=RECIPE / "comfy-build", capture_output=True, text=True, timeout=120).stdout
    s = json.loads(out)["data"]["serving"]
    return s


def parse_ts(s):
    s = s.replace("Z", "+00:00")
    head, _, tz = s.partition("+")
    if "." in head:
        whole, frac = head.split(".")
        head = f"{whole}.{frac[:6].ljust(6, '0')}"
    return datetime.fromisoformat(f"{head}+{tz}")


def wait_zero(backend, last_done):
    t0 = time.monotonic()
    if last_done:
        gap = MIN_GAP_S - (now() - last_done).total_seconds()
        if gap > 0:
            time.sleep(gap)
    while True:
        try:
            if backend == "runpod":
                h = runpod_health()
                w, j = h["workers"], h["jobs"]
                ok = w["running"] == 0 and w["initializing"] == 0 and j["inProgress"] == 0 and j["inQueue"] == 0
                snap = {"workers": w, "jobs": {k: j[k] for k in ("inProgress", "inQueue")}}
            else:
                s = comfy_status()
                w = s["workers"]
                fresh = last_done is None or parse_ts(s["sampledAt"]) >= last_done + timedelta(seconds=30)
                ok = fresh and w["running"] == 0 and w["initializing"] == 0 and s["jobsInQueue"] == 0
                snap = {"workers": w, "jobsInQueue": s["jobsInQueue"], "sampledAt": s["sampledAt"], "fresh": fresh}
        except Exception as e:  # transient API/CLI errors: retry
            ok, snap = False, {"error": str(e)[:200]}
        if ok:
            log(event="zero_confirmed", backend=backend, waited_s=round(time.monotonic() - t0), snapshot=snap)
            return snap
        if time.monotonic() - t0 > MAX_ZERO_WAIT_S:
            log(event="zero_timeout", backend=backend, snapshot=snap)
            raise SystemExit(f"{backend}: workers did not reach zero within {MAX_ZERO_WAIT_S}s")
        time.sleep(20 if backend == "runpod" else 45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    last_done = {"runpod": None, "comfy": None}
    for backend, label, seed in PLAN:
        if (OUT / "runs" / backend / label).exists():
            print(f"skip {backend} {label} (already in runs/)")
            continue
        if args.dry_run:
            print(backend, label, seed)
            continue
        note = ""
        if label.startswith("cold"):
            snap = wait_zero(backend, last_done[backend])
            note = "zero check: " + json.dumps(snap)
        log(event="start", backend=backend, label=label, seed=seed)
        p = subprocess.run([sys.executable, str(HERE / "bench.py"), backend, "--label", label, "--seed", str(seed), "--note", note],
                           capture_output=True, text=True)
        last_done[backend] = now()
        rec = json.loads((OUT / "runs" / backend / label / "run.json").read_text()) if (OUT / "runs" / backend / label / "run.json").exists() else {}
        log(event="done", backend=backend, label=label, status=rec.get("status"), client_done_s=rec.get("client_done_s"),
            provider=rec.get("provider"), error=rec.get("error") or (p.stderr[-300:] if p.returncode else None))
        if rec.get("status") not in ("COMPLETED", "succeeded"):
            log(event="stop", reason="run did not succeed; matrix halted for a look")
            raise SystemExit(1)
    if not args.dry_run:
        log(event="matrix_complete")


if __name__ == "__main__":
    main()
