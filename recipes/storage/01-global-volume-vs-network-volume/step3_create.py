#!/usr/bin/env python3
"""Create the five step-3 endpoints (min 0, max 1, idle 30 s, FlashBoot off) and write results/endpoints.json.
Run once; refuses if endpoints.json already lists ids. RUNPOD_API_KEY from env."""
import json, os, sys, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "endpoints.json"
API = "https://api.runpod.io/v2"
NV, GV = os.environ["NV_ID"], os.environ["GV_ID"]
VLLM = "runpod/worker-v1-vllm:v2.28.0@sha256:6d8e4ff417e09aeffe6a55e547ed81203b02228533c5305e726caaa7dab34571"
H3 = os.environ["H3_IMAGE"]  # your H3 ComfyUI worker image (not included in this recipe)


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:800]


GPU = {"pools": ["BLACKWELL_96"], "count": 1, "minCudaVersion": "13.0",
       "excludedTypes": ["NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]}
COMMON = {"type": "QUEUE", "gpu": GPU, "flashboot": "OFF", "workers": {"min": 0, "max": 1, "idleTimeout": 30},
          "scaling": {"type": "QUEUE_DELAY", "queueDelay": 1}}
# OMG ep04 env.example (worker v2.28.0)
LLM_ENV = {"MODEL_NAME": "PassingByPixels/Qwen3.8-27B-NVFP4", "MODEL_REVISION": "1e5f29f3212294efdef42a873672c9bc399cae3c",
           "QUANTIZATION": "modelopt", "KV_CACHE_DTYPE": "fp8", "MAX_MODEL_LEN": "32768", "GPU_MEMORY_UTILIZATION": "0.88",
           "ENFORCE_EAGER": "true", "MAX_NUM_SEQS": "8", "MAX_CONCURRENCY": "1", "TRUST_REMOTE_CODE": "true",
           "REASONING_PARSER": "qwen3", "ENABLE_AUTO_TOOL_CHOICE": "true", "TOOL_CALL_PARSER": "qwen3_xml"}
VOL_LLM_ENV = {k: v for k, v in LLM_ENV.items() if k != "MODEL_REVISION"}
VOL_LLM_ENV.update({"MODEL_NAME": "/runpod-volume/models/qwen38-27b-nvfp4", "HF_HUB_OFFLINE": "1",
                    "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "PassingByPixels/Qwen3.8-27B-NVFP4"})
H3_ENV = {"H3_CACHE_HUB": "/runpod-volume/h3-hub", "H3_CACHE_REPO": os.environ["H3_REPO"],
          "H3_CACHE_REVISION": os.environ["H3_REV"], "H3_CACHE_REQUIRED": "1"}
SPECS = {
    "EP_L_S0": {"name": "omg-gv-L-S0-hf", "image": VLLM, "disk": 60, "env": LLM_ENV},
    "EP_L_S1": {"name": "omg-gv-L-S1-nv", "image": VLLM, "disk": 30, "env": VOL_LLM_ENV, "dataCenterIds": ["EU-RO-1"], "mounts": {"network": [{"volumeId": NV}]}},
    "EP_L_S2": {"name": "omg-gv-L-S2-gv", "image": VLLM, "disk": 30, "env": VOL_LLM_ENV, "mounts": {"global": [{"volumeId": GV}]}},
    "EP_V_S1": {"name": "omg-gv-V-S1-nv", "image": H3, "disk": 40, "env": H3_ENV, "timeout": 1800000, "dataCenterIds": ["EU-RO-1"], "mounts": {"network": [{"volumeId": NV}]}},
    "EP_V_S2": {"name": "omg-gv-V-S2-gv", "image": H3, "disk": 40, "env": H3_ENV, "timeout": 1800000, "mounts": {"global": [{"volumeId": GV}]}},
}

if __name__ == "__main__":
    made = json.loads(OUT.read_text()) if OUT.exists() else {}
    if made:
        sys.exit(f"{OUT} already lists {made}")
    try:
        for key, spec in SPECS.items():
            st, r = call("POST", "/serverless", {**COMMON, **spec})
            if st >= 300:
                print(key, "FAILED", st, r)
                break
            made[key] = r["id"]
            print(key, r["id"], r.get("name"), "mounts", r.get("mounts"), "dc", r.get("dataCenterIds"), "flashboot", r.get("flashboot"))
    finally:
        OUT.write_text(json.dumps(made, indent=1))
