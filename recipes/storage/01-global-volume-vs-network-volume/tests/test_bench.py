"""Offline checks for bench.py and read_bench.sh. No Runpod calls, no API key needed.

  python3 -m unittest discover -s bench/global-volume/tests
"""
import importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

H3_LOG = """\
H3_MODEL_READY models/vae/minimax_h3_audio_vae_fp32.safetensors source=runpod_hf_cache bytes=605254808 verification_seconds=2.100
H3_MODEL_READY models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors source=runpod_hf_cache bytes=20970379616 verification_seconds=70.500
H3_MODELS_READY repo=example/h3-models revision=697cbe10 cache_required=True verification_seconds=100.000
H3_ENDPOINT_READY startup_seconds=131.250
"""

VLLM_LOG = """\
INFO 10-22 01:02:03 [default_loader.py:262] Loading weights took 21.37 seconds
INFO 10-22 01:02:04 [gpu_model_runner.py:2007] Model loading took 18.4 GiB and 23.91 seconds
INFO 10-22 01:02:40 [core.py:193] init engine (profile, create kv cache, warmup model) took 35.20 seconds
"""


class ParseLog(unittest.TestCase):
    def test_h3(self):
        s = bench.parse_log(H3_LOG)
        self.assertEqual(len(s["h3_model_ready"]), 2)
        self.assertEqual(s["h3_read_bytes"], 605254808 + 20970379616)
        self.assertEqual(s["h3_sources"], ["runpod_hf_cache"])
        self.assertEqual(s["h3_endpoint_ready"][0]["seconds"], 131.25)
        self.assertAlmostEqual(s["h3_effective_mb_per_s"], 215.8, places=1)

    def test_h3_millisecond_format(self):
        # format printed by the original H3 image and the variant
        s = bench.parse_log(
            "H3_MODEL_READY vae/a.safetensors source=runpod_hf_cache bytes=1000000000 verification_seconds=5.000\n"
            "H3_MODELS_READY repo=example/h3-models revision=697c cache_required=True verify_workers=1 verify_chunk_mb=8 verification_ms=10000\n"
            "H3_ENDPOINT_READY model_cache_verify_ms=10000 comfy_ready_ms=31500 startup_ms=42100\n")
        self.assertEqual(s["h3_models_ready"][0], {"workers": 1, "seconds": 10.0})
        self.assertEqual(s["h3_endpoint_ready"][0], {"seconds": 42.1, "comfy_seconds": 31.5})
        self.assertEqual(s["h3_effective_mb_per_s"], 100.0)
        old = bench.parse_log("H3_MODELS_READY repo=x revision=y cache_required=True verification_ms=2500\n")
        self.assertEqual(old["h3_models_ready"][0], {"seconds": 2.5})

    def test_vllm(self):
        s = bench.parse_log(VLLM_LOG)
        self.assertEqual(s["vllm_weights"][0]["seconds"], 21.37)
        self.assertEqual(s["vllm_model_loading"][0], {"gib": 18.4, "seconds": 23.91})
        self.assertEqual(s["vllm_init_engine"][0]["seconds"], 35.2)

    def test_unrelated_lines_ignored(self):
        self.assertEqual(bench.parse_log("hello\nworld\n"), {})


class Helpers(unittest.TestCase):
    def test_busy_workers(self):
        self.assertEqual(bench.busy_workers({"workers": {"idle": 0, "running": 0, "throttled": 2}}), {})
        self.assertEqual(bench.busy_workers({"workers": {"idle": 1, "ready": 0}}), {"idle": 1})

    def test_endpoint_env(self):
        self.assertEqual(bench.endpoint_env("llm", "s0"), "EP_L_S0")
        self.assertEqual(bench.endpoint_env("video", "s2"), "EP_V_S2")

    def test_order(self):
        rows = bench.run_order()
        llm = [r for r in rows if r[0] == "llm"]
        video = [r for r in rows if r[0] == "video"]
        # 3 conds x (warmup + 3 cold + 9 warm) and 2 conds x (warmup + 3 cold + 6 warm)
        self.assertEqual(len(llm), 3 * 13)
        self.assertEqual(len(video), 2 * 10)
        self.assertTrue(all(r[3] is None for r in llm))
        seeds = [r[3] for r in video]
        self.assertEqual(len(seeds), len(set(seeds)) * 2)  # same seed for s1 and s2 in a round
        self.assertNotIn("s0", {r[1] for r in video})
        self.assertLess(rows.index(llm[-1]), rows.index(video[0]))

    def test_llm_body_matches_ep04(self):
        self.assertEqual(bench.LLM_BODY["input"]["openai_input"]["model"], "PassingByPixels/Qwen3.8-27B-NVFP4")

    def test_dry_run_hides_key_and_payload(self):
        env = dict(os.environ, RUNPOD_API_KEY="test-key-not-real", EP_L_S1="ep-test")
        out = subprocess.run([sys.executable, str(HERE / "bench.py"), "send", "llm", "s1", "cold-1", "--dry-run"],
                             env=env, capture_output=True, text=True, check=True).stdout
        self.assertNotIn("test-key-not-real", out)
        self.assertEqual(json.loads(out)["endpoint_env"], "EP_L_S1")


class ReadBench(unittest.TestCase):
    """Run read_bench.sh against two local dirs standing in for /workspace and /workspace-global."""

    def run_sh(self, *args, env=None, check=True):
        return subprocess.run(["bash", str(HERE / "read_bench.sh"), *args], env=env,
                              capture_output=True, text=True, check=check)

    def test_copy_run_and_cache_guard(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            nv, gv = t / "nv", t / "gv"
            big = nv / "models/qwen38-27b-nvfp4/model-00001-of-00003.safetensors"
            big.parent.mkdir(parents=True)
            big.write_bytes(os.urandom(3 * 1024 * 1024))
            env = dict(os.environ, RESULTS=str(t / "r.jsonl"), SEEN=str(t / "seen"), SMALL_COUNT="20", SMALL_KB="4")
            self.run_sh("copy", "a1-test", str(nv), str(gv), env=env)
            self.assertEqual(big.read_bytes(), (gv / big.relative_to(nv)).read_bytes())
            for root in (nv, gv):
                self.run_sh("small-prepare", str(root), env=env)
            self.run_sh("run", "eu-r1", f"nv={nv}", f"gv={gv}", env=env)
            lines = [json.loads(l) for l in (t / "r.jsonl").read_text().splitlines()]
            kinds = [(l["kind"], l.get("volume")) for l in lines]
            self.assertEqual(kinds, [("copy", None), ("read_large", "nv"), ("read_small", "nv"),
                                     ("read_large", "gv"), ("read_small", "gv")])
            self.assertEqual(lines[1]["bytes"], 3 * 1024 * 1024)
            self.assertEqual(lines[2]["files"], 20)
            again = self.run_sh("run", "eu-r1-again", f"nv={nv}", env=env, check=False)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("already read in this Pod", again.stderr)


if __name__ == "__main__":
    unittest.main()
