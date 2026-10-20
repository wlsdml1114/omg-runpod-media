import csv
import importlib.util
import json
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecipeTest(unittest.TestCase):
    def test_graph_is_turbo4_with_known_node_ids(self):
        graph = json.loads((ROOT / "workflows/h3-ref2va-turbo4.api.json").read_text())
        self.assertEqual(len(graph), 18)
        self.assertEqual(graph["9"]["class_type"], "RandomNoise")
        self.assertEqual(graph["12"]["inputs"]["steps"], 4)
        self.assertEqual(graph["17"]["class_type"], "SaveVideo")
        self.assertEqual(graph["18"]["inputs"]["lora_name"], "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")

    def test_bench_swaps_only_seed_inputs_and_prefix(self):
        bench = load("bench")
        assets = {
            "identity.png": {"id": "img", "file_path": "identity.png"},
            "source.mp4": {"id": "vid", "file_path": "source.mp4"},
        }
        built = bench.build_graph(123, assets)
        original = json.loads(bench.GRAPH.read_text())
        self.assertEqual(built["9"]["inputs"]["noise_seed"], 123)
        self.assertEqual(built["6"]["inputs"]["file"]["__type"], "core/ASSET")
        changed = {k for k in original if original[k] != built[k]}
        self.assertEqual(changed, {"5", "6", "9", "17"})

    def test_span_handles_nanosecond_timestamps(self):
        bench = load("bench")
        self.assertEqual(bench.span("2026-10-19T11:39:39.689565Z", "2026-10-19T11:45:05.316621Z"), 325.627)
        self.assertEqual(bench.span("2026-10-19T12:48:02.569192505Z", "2026-10-19T12:48:03.0Z"), 0.431)

    def test_matrix_order_puts_warm_right_after_each_cold(self):
        plan = load("run_matrix").PLAN
        labels = [f"{backend}:{label}" for backend, label, _ in plan]
        self.assertEqual(len(plan), 16)
        self.assertEqual(labels[0:6], ["runpod:cold-1"] + [f"runpod:warm-{i}" for i in range(1, 6)])
        self.assertEqual(labels[6:12], ["comfy:cold-1"] + [f"comfy:warm-{i}" for i in range(1, 6)])
        seeds = {(b, l): s for b, l, s in plan}
        self.assertEqual(seeds[("runpod", "warm-3")], seeds[("comfy", "warm-3")])

    def test_results_csv_matches_reported_medians(self):
        with (ROOT / "results/runs.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        warm = lambda backend: [float(r["client_done_s"]) for r in rows if r["backend"] == backend and r["label"].startswith("warm")]
        self.assertEqual(round(statistics.median(warm("runpod")), 1), 196.3)
        self.assertEqual(round(statistics.median(warm("comfy")), 1), 165.1)
        self.assertNotIn("job_id", rows[0])


if __name__ == "__main__":
    unittest.main()
