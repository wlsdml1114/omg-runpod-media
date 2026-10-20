#!/usr/bin/env python3
"""Send one H3 Ref2VA Turbo4 job to Runpod Serverless or Comfy API and record timing.

Both backends get the same prepared inputs (source.mp4 1344x768 + identity.png) and the
same graph (workflows/h3-ref2va-turbo4.api.json). Only the seed changes between runs.

  python3 scripts/bench.py runpod --label cold-1 --seed 42101
  python3 scripts/bench.py comfy  --label cold-1 --seed 42101
  python3 scripts/bench.py comfy  --label warm-1 --seed 42102 --dry-run

Env: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, COMFY_API_KEY, COMFY_BASE_URL (https://<dep>.run.comfy.app).
Every run appends one line to benchmark-output/runs.jsonl and keeps raw responses under
benchmark-output/runs/<backend>/<label>/. Inputs come from inputs/ (override with BENCH_INPUTS).
The Runpod side expects a handler that takes source_video_base64 / reference_image_base64 /
mode / scope / prompt / seed / source_audio and returns base64 videos (see README).
Keys are read from the environment only and never written to disk.
"""
import argparse, base64, copy, hashlib, json, mimetypes, os, sys, time, urllib.error, urllib.parse, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
OUT = RECIPE / "benchmark-output"
GRAPH = RECIPE / "workflows" / "h3-ref2va-turbo4.api.json"
INPUTS = Path(os.environ.get("BENCH_INPUTS", RECIPE / "inputs"))
PROMPT = ("Preserve the original spoken delivery, mouth motion, pauses, gaze and microphone gestures. "
          "Keep the white polo shirt and original spacecraft interior. "
          "Replace only the face and hair with the fictional adult woman in the identity reference.")
POLL_S = 2.0
TIMEOUT_S = 40 * 60

# Graph node ids (see README)
N_IMAGE, N_VIDEO, N_SEED, N_SAVE = "5", "6", "9", "17"


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def span(a, b):
    """Seconds between two RFC 3339 timestamps (nanosecond precision is cut to microseconds)."""
    if not a or not b:
        return None
    def parse(s):
        s = s.replace("Z", "+00:00")
        head, _, tz = s.partition("+")
        if "." in head:
            whole, frac = head.split(".")
            head = f"{whole}.{frac[:6].ljust(6, '0')}"
        return datetime.fromisoformat(f"{head}+{tz}")
    return round((parse(b) - parse(a)).total_seconds(), 3)


def env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env {name}")
    return v


def http(method, url, *, token=None, body=None, headers=None, raw=False, timeout=120):
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    data = body
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = r.read()
            return r.status, (out if raw else json.loads(out or b"null"))
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"{method} {urllib.parse.urlsplit(url).path} -> {e.code}: {text}") from None


class Recorder:
    """Client-side clock: wall time from submit, plus first time each status was seen."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.first_seen = {}

    def elapsed(self):
        return round(time.monotonic() - self.t0, 3)

    def saw(self, status):
        self.first_seen.setdefault(status, self.elapsed())


# ---------------------------------------------------------------- Runpod

def run_runpod(args, run_dir, rec):
    key, ep = env("RUNPOD_API_KEY"), env("RUNPOD_ENDPOINT_ID")
    base = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
    body = {"input": {
        "source_video_base64": base64.b64encode((INPUTS / "source.mp4").read_bytes()).decode(),
        "reference_image_base64": base64.b64encode((INPUTS / "identity.png").read_bytes()).decode(),
        "mode": "turbo4", "scope": "head", "prompt": PROMPT, "seed": args.seed, "source_audio": True,
    }}
    if args.dry_run:
        shown = copy.deepcopy(body)
        for k in ("source_video_base64", "reference_image_base64"):
            shown["input"][k] = f"<{len(body['input'][k])} chars>"
        return {"dry_run": True, "request": shown}

    rec.t0 = time.monotonic()
    submitted_at = now_iso()
    _, sub = http("POST", f"{base}/{ep}/run", token=key, body=body)
    job_id = sub["id"]
    rec.saw("SUBMITTED")
    while True:
        _, st = http("GET", f"{base}/{ep}/status/{job_id}", token=key)
        rec.saw(st.get("status"))
        if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        if rec.elapsed() > TIMEOUT_S:
            http("POST", f"{base}/{ep}/cancel/{job_id}", token=key)
            raise RuntimeError(f"timeout after {TIMEOUT_S}s, cancel sent for {job_id}")
        time.sleep(POLL_S)
    t_done = rec.elapsed()

    out = st.get("output") or {}
    saved = []
    for v in out.get("videos", []):
        b = base64.b64decode(v.pop("video_base64"))
        p = run_dir / v["filename"]
        p.write_bytes(b)
        saved.append({"role": v.get("role"), "file": p.name, "bytes": len(b), "sha256": sha256(b),
                      "sha256_matches_provider": sha256(b) == v.get("sha256")})
    meta = out.get("metadata", {})
    meta.pop("workflow", None)
    (run_dir / "status.json").write_text(json.dumps(st, indent=1))
    return {
        "job_id": job_id, "status": st.get("status"), "submitted_at": submitted_at,
        "client_done_s": t_done, "client_saved_s": rec.elapsed(),
        "provider": {"delayTime_ms": st.get("delayTime"), "executionTime_ms": st.get("executionTime"),
                     "workerId": st.get("workerId"),
                     "generation_request_seconds": meta.get("generation_request_seconds"),
                     "handler_seconds": meta.get("handler_seconds")},
        "outputs": saved, "error": st.get("error"),
    }


# ---------------------------------------------------------------- Comfy API

def comfy_assets(base, key, state_file):
    """Upload the two inputs once and reuse their asset ids across runs."""
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    for name in ("source.mp4", "identity.png"):
        data = (INPUTS / name).read_bytes()
        digest = sha256(data)
        if state.get(name, {}).get("sha256") == digest:
            continue
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        parts = []
        for field, value in (("content_type", ctype), ("file_path", name), ("tags", "input")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        _, asset = http("POST", f"{base}/api/v2/assets", token=key, body=b"".join(parts),
                        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                 "Idempotency-Key": str(uuid.uuid4())}, timeout=600)
        state[name] = {"sha256": digest, "id": asset["id"], "hash": asset.get("hash"), "file_path": name}
        state_file.write_text(json.dumps(state, indent=1))
    return state


def asset_ref(a):
    info = {"id": a["id"], "file_path": a["file_path"]}
    if a.get("hash"):
        info["hash"] = a["hash"]
    return {"__type": "core/ASSET", "info": info}


def build_graph(seed, assets):
    g = json.loads(GRAPH.read_text())
    g[N_SEED]["inputs"]["noise_seed"] = seed
    g[N_SAVE]["inputs"]["filename_prefix"] = "bench/h3-turbo4"
    if assets:
        g[N_IMAGE]["inputs"]["image"] = asset_ref(assets["identity.png"])
        g[N_VIDEO]["inputs"]["file"] = asset_ref(assets["source.mp4"])
    return g


def run_comfy(args, run_dir, rec):
    key, base = env("COMFY_API_KEY"), env("COMFY_BASE_URL").rstrip("/")
    if args.dry_run:
        return {"dry_run": True, "request": {"workflow": build_graph(args.seed, None)}}

    assets = comfy_assets(base, key, OUT / "comfy-assets.json")
    body = {"workflow": build_graph(args.seed, assets), "metadata": {"bench": "comfy-api-vs-runpod", "label": args.label}}
    rec.t0 = time.monotonic()
    submitted_at = now_iso()
    _, job = http("POST", f"{base}/api/v2/jobs", token=key, body=body, headers={"Idempotency-Key": str(uuid.uuid4())})
    job_id = job["id"]
    rec.saw("submitted")
    self_url = (job.get("urls") or {}).get("self") or f"{base}/api/v2/jobs/{job_id}"
    if self_url.startswith("/"):
        self_url = base + self_url
    while True:
        _, st = http("GET", self_url, token=key)
        rec.saw(st.get("status"))
        if st.get("status") in ("succeeded", "failed", "canceled", "expired"):
            break
        if rec.elapsed() > TIMEOUT_S:
            cancel = (st.get("urls") or {}).get("cancel") or f"{base}/api/v2/jobs/{job_id}/cancel"
            http("POST", base + cancel if cancel.startswith("/") else cancel, token=key)
            raise RuntimeError(f"timeout after {TIMEOUT_S}s, cancel sent for {job_id}")
        time.sleep(POLL_S)
    t_done = rec.elapsed()

    saved = []
    base_host = urllib.parse.urlsplit(base).netloc
    for o in st.get("outputs", []):
        # outputs also echo the LoadVideo input (node 6); keep only the SaveVideo result
        if o.get("node_id") != N_SAVE or o.get("type") != "video" or not o.get("url"):
            continue
        url = base + o["url"] if o["url"].startswith("/") else o["url"]
        same_host = urllib.parse.urlsplit(url).netloc == base_host
        _, b = http("GET", url, token=key if same_host else None, raw=True, timeout=600)
        p = run_dir / Path(o.get("name") or f"{o['id']}.mp4").name
        p.write_bytes(b)
        saved.append({"node_id": o.get("node_id"), "file": p.name, "bytes": len(b), "sha256": sha256(b)})
    (run_dir / "status.json").write_text(json.dumps(st, indent=1))
    return {
        "job_id": job_id, "status": st.get("status"), "submitted_at": submitted_at,
        "client_done_s": t_done, "client_saved_s": rec.elapsed(),
        # metrics was absent on the deployment (2026-10-19), and status reads "running" long before
        # started_at, so derive the split from the job's own timestamps
        "provider": {"metrics": st.get("metrics"), "created_at": st.get("created_at"),
                     "started_at": st.get("started_at"), "completed_at": st.get("completed_at"),
                     "created_to_started_s": span(st.get("created_at"), st.get("started_at")),
                     "started_to_completed_s": span(st.get("started_at"), st.get("completed_at")),
                     "deployment_id": st.get("deployment_id"), "release_version": st.get("release_version")},
        "outputs": saved, "error": st.get("error"),
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["runpod", "comfy"])
    ap.add_argument("--label", required=True, help="e.g. cold-1, warm-3")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--note", default="", help="free text, e.g. 'workers=0 confirmed in console'")
    ap.add_argument("--dry-run", action="store_true", help="print the request without sending")
    args = ap.parse_args()

    for f in (GRAPH, INPUTS / "source.mp4", INPUTS / "identity.png"):
        if not f.exists():
            sys.exit(f"missing {f}")
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "runs" / args.backend / args.label
    if run_dir.exists() and not args.dry_run:
        sys.exit(f"{run_dir} exists; use a new --label")

    rec = Recorder()
    record = {"backend": args.backend, "label": args.label, "seed": args.seed, "note": args.note,
              "graph_sha256": sha256(GRAPH.read_bytes()),
              "input_sha256": {n: sha256((INPUTS / n).read_bytes()) for n in ("source.mp4", "identity.png")}}
    if args.dry_run:
        fn = run_runpod if args.backend == "runpod" else run_comfy
        print(json.dumps(fn(args, None, rec), indent=1, ensure_ascii=False)[:4000])
        return

    run_dir.mkdir(parents=True)
    try:
        result = (run_runpod if args.backend == "runpod" else run_comfy)(args, run_dir, rec)
        record.update(result)
    except Exception as e:  # keep failures in the log; they are part of the comparison
        record.update({"status": "client_error", "error": str(e), "client_failed_s": rec.elapsed()})
    record["first_seen_s"] = rec.first_seen
    record["recorded_at"] = now_iso()
    (run_dir / "run.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    with open(OUT / "runs.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({k: record.get(k) for k in ("backend", "label", "status", "client_done_s", "provider", "error")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
