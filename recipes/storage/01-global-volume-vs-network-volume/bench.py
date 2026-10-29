#!/usr/bin/env python3
"""Serverless cold/warm requests for the network volume vs global volume benchmark (B1-B5).

Plan: docs/global-volume-v1/benchmark-plan.md, step 3.

  python3 bench.py order                                   # print the run order
  python3 bench.py send llm s1 cold-1                      # one request to EP_L_S1
  python3 bench.py send video s2 cold-1 --seed 52101       # one H3 Turbo4 request to EP_V_S2
  python3 bench.py send video s2 warm-1 --seed 52111 --dry-run
  python3 bench.py attach-log video s2 cold-1 worker.log --worker-dc US-NC-2

Endpoints come from env: EP_L_S0, EP_L_S1, EP_L_S2, EP_V_S1, EP_V_S2 (EP_V_S0 only after the user
agrees to send jobs to the existing endpoint). RUNPOD_API_KEY is read from env and never written.

Before a cold request, /health must show no worker that could take the job right away
(idle, ready, running or initializing). Recipe 06 showed that idle slots turn a "cold" request into
a 1-second wait, so a cold label is refused unless --force is given; the health snapshot is kept.

Each run writes runs/<workload>-<cond>/<label>/run.json and appends one line to runs.jsonl.
attach-log appends a separate {"kind": "log"} line instead of rewriting earlier lines.
"""
import argparse, base64, copy, hashlib, json, os, re, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
LOG = HERE / "runs.jsonl"
API = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
POLL_S = 2.0
TIMEOUT_S = {"llm": 20 * 60, "video": 40 * 60}

# LLM request: OMG ep04 smoke-input.json. S1/S2 serve a local path, so their endpoints set
# OPENAI_SERVED_MODEL_NAME_OVERRIDE to this id and the same body works on all three.
LLM_MODEL = "PassingByPixels/Qwen3.8-27B-NVFP4"
LLM_BODY = {"input": {
    "openai_route": "/v1/chat/completions",
    "openai_input": {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
    },
}}

# Video request: recipe 06 Runpod body (H3 Ref2VA Turbo4, 1344x768, 124 frames).
VIDEO_INPUTS = Path(os.environ.get("BENCH_INPUTS", str(Path(__file__).resolve().parent / "inputs")))
VIDEO_PROMPT = ("Preserve the original spoken delivery, mouth motion, pauses, gaze and microphone gestures. "
                "Keep the white polo shirt and original spacecraft interior. "
                "Replace only the face and hair with the fictional adult woman in the identity reference.")

WORKLOADS = {"llm": "L", "video": "V"}
CONDS = ("s0", "s1", "s2")
BUSY = ("idle", "ready", "running", "initializing")

# Worker log lines worth keeping (B2). vLLM wording differs by version, so match loosely.
LOG_PATTERNS = {
    "h3_model_ready": re.compile(r"H3_MODEL_READY (?P<file>\S+) source=(?P<source>\S+) bytes=(?P<bytes>\d+) verification_seconds=(?P<seconds>[\d.]+)"),
    # the original H3 image and the variant print milliseconds; older builds printed seconds
    "h3_models_ready": re.compile(r"H3_MODELS_READY .*?(?:verify_workers=(?P<workers>\d+) .*?)?verification_(?:seconds=(?P<seconds>[\d.]+)|ms=(?P<ms>\d+))"),
    "h3_endpoint_ready": re.compile(r"H3_ENDPOINT_READY (?:startup_seconds=(?P<seconds>[\d.]+)|model_cache_verify_ms=\S+ comfy_ready_ms=(?P<comfy_ms>\d+) startup_ms=(?P<ms>\d+))"),
    "vllm_weights": re.compile(r"Loading weights took (?P<seconds>[\d.]+) ?s"),
    "vllm_model_loading": re.compile(r"Model loading took (?P<gib>[\d.]+) ?GiB(?: memory)? and (?P<seconds>[\d.]+) ?s"),
    "vllm_init_engine": re.compile(r"init engine .*? took (?P<seconds>[\d.]+) ?s"),
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env {name}")
    return v


def endpoint_env(workload, cond):
    return f"EP_{WORKLOADS[workload]}_{cond.upper()}"


def http(method, url, token, body=None, timeout=120, retries=None):
    """GETs retry transient 5xx/network errors (a single 500 on /status ended L-S1 cold-2 on 2026-10-22).
    POSTs are not retried, so a job is never submitted twice."""
    retries = (5 if method == "GET" else 0) if retries is None else retries
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            msg = f"{method} {url.split('/v2/')[-1]} -> {e.code}: {e.read().decode(errors='replace')[:1000]}"
            if e.code < 500 or attempt == retries:
                raise RuntimeError(msg) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise RuntimeError(f"{method} {url.split('/v2/')[-1]} -> {e}") from None
        time.sleep(3 * (attempt + 1))


def request_body(workload, seed):
    if workload == "llm":
        return copy.deepcopy(LLM_BODY)
    if seed is None:
        sys.exit("video requests need --seed")
    return {"input": {
        "source_video_base64": base64.b64encode((VIDEO_INPUTS / "source.mp4").read_bytes()).decode(),
        "reference_image_base64": base64.b64encode((VIDEO_INPUTS / "identity.png").read_bytes()).decode(),
        "mode": "turbo4", "scope": "head", "prompt": VIDEO_PROMPT, "seed": seed, "source_audio": True,
    }}


def redact(body):
    shown = copy.deepcopy(body)
    for k, v in shown["input"].items():
        if k.endswith("_base64"):
            shown["input"][k] = f"<{len(v)} chars>"
    return shown


def busy_workers(health):
    w = (health or {}).get("workers") or {}
    return {k: w.get(k, 0) for k in BUSY if w.get(k, 0)}


def save_outputs(workload, out, run_dir):
    """LLM: keep the text. Video: decode the MP4s like recipe 06 and keep hashes."""
    if workload == "llm":
        return {"output_preview": json.dumps(out, ensure_ascii=False)[:400]}
    saved = []
    for v in (out or {}).get("videos", []):
        b = base64.b64decode(v.pop("video_base64"))
        p = run_dir / v["filename"]
        p.write_bytes(b)
        saved.append({"role": v.get("role"), "file": p.name, "bytes": len(b), "sha256": sha256(b),
                      "sha256_matches_provider": sha256(b) == v.get("sha256")})
    return {"outputs": saved}


def cmd_send(a):
    key = env("RUNPOD_API_KEY")
    ep = env(endpoint_env(a.workload, a.cond))
    body = request_body(a.workload, a.seed)
    if a.dry_run:
        print(json.dumps({"endpoint_env": endpoint_env(a.workload, a.cond), "request": redact(body)}, indent=1, ensure_ascii=False))
        return
    run_dir = RUNS / f"{a.workload}-{a.cond}" / a.label
    if run_dir.exists():
        sys.exit(f"{run_dir} exists; use a new label")

    health = http("GET", f"{API}/{ep}/health", key)
    busy = busy_workers(health)
    if a.label.startswith("cold") and busy and not a.force:
        sys.exit(f"not cold: workers {busy}. Wait for 0 (or scale max to 0 and back), then retry.")

    record = {"kind": "request", "workload": a.workload, "cond": a.cond, "label": a.label, "seed": a.seed,
              "note": a.note, "endpoint_env": endpoint_env(a.workload, a.cond),
              "health_before": health, "forced_cold": bool(a.label.startswith("cold") and busy),
              "request_sha256": sha256(json.dumps(body, sort_keys=True).encode())}
    if a.workload == "video":
        record["input_sha256"] = {n: sha256((VIDEO_INPUTS / n).read_bytes()) for n in ("source.mp4", "identity.png")}
    run_dir.mkdir(parents=True)
    first_seen, t0 = {}, time.monotonic()
    try:
        record["submitted_at"] = now_iso()
        job_id = http("POST", f"{API}/{ep}/run", key, body)["id"]
        record["job_id"] = job_id
        while True:
            st = http("GET", f"{API}/{ep}/status/{job_id}", key)
            first_seen.setdefault(st.get("status"), round(time.monotonic() - t0, 3))
            if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break
            if time.monotonic() - t0 > TIMEOUT_S[a.workload]:
                http("POST", f"{API}/{ep}/cancel/{job_id}", key)
                raise RuntimeError(f"timeout after {TIMEOUT_S[a.workload]}s, cancel sent for {job_id}")
            time.sleep(POLL_S)
        record["client_done_s"] = round(time.monotonic() - t0, 3)
        out = st.pop("output", None)
        record.update({"status": st.get("status"), "delay_s": ms(st.get("delayTime")),
                       "execution_s": ms(st.get("executionTime")), "worker_id": st.get("workerId"),
                       "error": st.get("error")})
        record.update(save_outputs(a.workload, out, run_dir))
        (run_dir / "status.json").write_text(json.dumps(st, indent=1))
    except Exception as e:  # failures stay in the log; they are part of the result
        record.update({"status": "client_error", "error": str(e), "client_failed_s": round(time.monotonic() - t0, 3)})
    record["first_seen_s"] = first_seen
    record["recorded_at"] = now_iso()
    write(run_dir / "run.json", record)
    print(json.dumps({k: record.get(k) for k in ("workload", "cond", "label", "status", "delay_s", "execution_s",
                                                  "client_done_s", "worker_id", "error")}, indent=1, ensure_ascii=False))
    if record.get("worker_id"):
        print(f"next: look up worker {record['worker_id']} (list-endpoint-workers), save its log, then "
              f"bench.py attach-log {a.workload} {a.cond} {a.label} <log> --worker-dc <DC>")


def ms(v):
    return round(v / 1000, 3) if isinstance(v, (int, float)) else None


def parse_log(text):
    found = {k: [] for k in LOG_PATTERNS}
    for line in text.splitlines():
        for k, pat in LOG_PATTERNS.items():
            m = pat.search(line)
            if m:
                g = {n: v for n, v in m.groupdict().items() if v is not None}
                rec = {n: (float(v) if n in ("seconds", "gib") else int(v) if n in ("bytes", "ms", "comfy_ms", "workers") else v)
                       for n, v in g.items()}
                if "ms" in rec:  # normalise to seconds
                    rec["seconds"] = rec.pop("ms") / 1000
                if "comfy_ms" in rec:
                    rec["comfy_seconds"] = rec.pop("comfy_ms") / 1000
                found[k].append(rec)
    summary = {k: v for k, v in found.items() if v}
    files = summary.get("h3_model_ready", [])
    if files:
        total = sum(f["bytes"] for f in files)
        slowest = max(f["seconds"] for f in files)  # bootstrap verifies 4 files at a time
        summary["h3_read_bytes"] = total
        summary["h3_sources"] = sorted({f["source"] for f in files})
        if summary.get("h3_models_ready"):
            wall = summary["h3_models_ready"][-1]["seconds"]
            summary["h3_effective_mb_per_s"] = round(total / 1e6 / wall, 1) if wall else None
        summary["h3_slowest_file_s"] = slowest
    return summary


def cmd_attach_log(a):
    run_dir = RUNS / f"{a.workload}-{a.cond}" / a.label
    if not run_dir.exists():
        sys.exit(f"no run {run_dir}")
    text = Path(a.log).read_text(errors="replace")
    (run_dir / "worker.log").write_text(text)
    summary = parse_log(text)
    if a.workload == "video" and summary.get("h3_sources") and summary["h3_sources"] != ["runpod_hf_cache"]:
        print(f"warning: model source {summary['h3_sources']} - the worker did not read from the volume", file=sys.stderr)
    entry = {"kind": "log", "workload": a.workload, "cond": a.cond, "label": a.label,
             "worker_dc": a.worker_dc, "worker_id": a.worker_id, "log_sha256": sha256(text.encode()),
             "summary": summary, "recorded_at": now_iso()}
    write(run_dir / "log-summary.json", entry)
    print(json.dumps(entry, indent=1, ensure_ascii=False))


def write(path, record):
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    with open(LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_order():
    """Plan step 3: LLM first, then video. Within a workload, conditions alternate per round."""
    rows = []
    for wl, conds, warm in (("llm", CONDS, 3), ("video", ("s1", "s2"), 2)):
        base = 52000 if wl == "video" else None
        for c in conds:
            rows.append((wl, c, "warmup-0", base and base + 100))
        for r in (1, 2, 3):
            for c in conds:
                rows.append((wl, c, f"cold-{r}", base and base + 100 + r))
                for w in range(1, warm + 1):
                    rows.append((wl, c, f"warm-{r}{w}", base and base + 110 + r * 10 + w))
    return rows


def cmd_order(_):
    for i, (wl, c, label, seed) in enumerate(run_order(), 1):
        seed_arg = f" --seed {seed}" if seed else ""
        print(f"{i:3d}  python3 bench.py send {wl} {c} {label}{seed_arg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send", help="send one request and record it")
    s.add_argument("workload", choices=list(WORKLOADS))
    s.add_argument("cond", choices=CONDS)
    s.add_argument("label", help="warmup-0, cold-1, warm-11, ...")
    s.add_argument("--seed", type=int, help="video only")
    s.add_argument("--note", default="")
    s.add_argument("--force", action="store_true", help="send a cold label even if workers are up (recorded)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_send)
    l = sub.add_parser("attach-log", help="parse a saved worker log into the run")
    l.add_argument("workload", choices=list(WORKLOADS))
    l.add_argument("cond", choices=CONDS)
    l.add_argument("label")
    l.add_argument("log")
    l.add_argument("--worker-dc", default=None)
    l.add_argument("--worker-id", default=None)
    l.set_defaults(fn=cmd_attach_log)
    o = sub.add_parser("order", help="print the planned run order")
    o.set_defaults(fn=cmd_order)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
