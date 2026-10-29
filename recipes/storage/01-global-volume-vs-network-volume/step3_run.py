#!/usr/bin/env python3
"""Plan step 3 driver: send the bench.py run order and collect worker logs + data center per run.

  python3 step3_run.py --only warmup          # warmup-0 on every endpoint (image pull, smoke check)
  python3 step3_run.py --workload llm         # LLM cold/warm matrix
  python3 step3_run.py --workload video

Endpoint ids: EP_L_S0 ... EP_V_S2 in env (results/endpoints.json is written at create time).
For a cold label it waits until /health shows no idle/ready/running/initializing worker (up to
COLD_WAIT_S); if a slot stays, it records that and sends with forced_cold=true instead of
touching the endpoint. Right after each job it reads the worker list (data center) and the
worker's logs (SSE, a few seconds) while the worker is still alive (idle timeout 30 s).
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

REST = "https://api.runpod.io/v2"
COLD_WAIT_S = 6 * 60


def rest(path, timeout=60):
    req = urllib.request.Request(REST + path, headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode(errors="replace")[:500]}


def sse_lines(path, seconds):
    req = urllib.request.Request(REST + path, headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]})
    out, end = [], time.monotonic() + seconds
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
                    out.append(f"{d.get('ts', '')} {d.get('source', '')} {d.get('line', '')}")
    except (TimeoutError, OSError):
        pass
    return out


def wait_cold(ep):
    t0 = time.monotonic()
    while time.monotonic() - t0 < COLD_WAIT_S:
        h = bench.http("GET", f"{bench.API}/{ep}/health", os.environ["RUNPOD_API_KEY"])
        if not bench.busy_workers(h):
            return True, round(time.monotonic() - t0, 1)
        time.sleep(10)
    return False, round(time.monotonic() - t0, 1)


def collect_worker(workload, cond, label, ep):
    run_dir = bench.RUNS / f"{workload}-{cond}" / label
    rec = json.loads((run_dir / "run.json").read_text())
    wid = rec.get("worker_id")
    workers = rest(f"/serverless/{ep}/workers")
    (run_dir / "workers.json").write_text(json.dumps(workers, indent=1))
    dc = None
    for w in (workers or {}).get("workers", []) if isinstance(workers, dict) else []:
        if w.get("id") == wid:
            dc = w.get("dataCenterId") or (w.get("machine") or {}).get("dataCenterId") or w.get("location")
    if not wid:
        return None
    # the log API backfills only 100 lines unless tail is set (max 5000)
    lines = sse_lines(f"/serverless/{ep}/workers/{wid}/logs?tail=5000", 15)
    log = run_dir / "worker-raw.log"
    log.write_text("\n".join(lines) + "\n")
    bench.cmd_attach_log(SimpleNamespace(workload=workload, cond=cond, label=label, log=str(log),
                                         worker_dc=dc, worker_id=wid))
    return dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["llm", "video"])
    ap.add_argument("--only", choices=["warmup"])
    ap.add_argument("--skip", nargs="*", default=[], metavar="WL:COND:LABEL",
                    help="planned runs to leave out, e.g. video:s2:warm-12 (budget cut, recorded in results/skipped.json)")
    ap.add_argument("--extra", nargs="*", default=[], metavar="WL:COND:LABEL",
                    help="runs to append after the plan, e.g. llm:s1:cold-2b")
    a = ap.parse_args()
    rows = [r for r in bench.run_order() if (a.workload is None or r[0] == a.workload)]
    if a.only == "warmup":
        rows = [r for r in rows if r[2] == "warmup-0"]
    else:
        rows = [r for r in rows if r[2] != "warmup-0"]
    skip = {tuple(s.split(":")) for s in a.skip}
    if skip:
        rows = [r for r in rows if (r[0], r[1], r[2]) not in skip]
        log = HERE / "results" / "skipped.json"
        prev = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(sorted({*map(tuple, prev), *skip}), indent=1))
    for e in a.extra:
        wl, cond, label = e.split(":")
        rows.append((wl, cond, label, None))
    for wl, cond, label, seed in rows:
        ep = os.environ[bench.endpoint_env(wl, cond)]
        if (bench.RUNS / f"{wl}-{cond}" / label / "run.json").exists():
            print(f"skip {wl} {cond} {label} (done)", flush=True)
            continue
        note = ""
        if label.startswith("cold"):
            ok, waited = wait_cold(ep)
            note = f"waited {waited}s for 0 workers" if ok else f"slot still up after {waited}s, sent anyway"
        args = SimpleNamespace(workload=wl, cond=cond, label=label, seed=seed, note=note,
                               force=label.startswith("cold"), dry_run=False)
        print(f"--- {wl} {cond} {label} {note}", flush=True)
        bench.cmd_send(args)
        try:
            dc = collect_worker(wl, cond, label, ep)
            print(f"    worker dc: {dc}", flush=True)
        except Exception as e:  # logs are extra evidence; a miss must not stop the matrix
            print(f"    worker log collection failed: {e}", flush=True)


if __name__ == "__main__":
    main()
