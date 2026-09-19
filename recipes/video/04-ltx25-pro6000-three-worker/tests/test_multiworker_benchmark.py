import copy
import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import multiworker_benchmark


class MultiworkerBenchmarkTest(unittest.TestCase):
    def test_prepare_workflow_changes_only_seed_and_output_prefix(self):
        base = {
            "398:339": {"inputs": {"noise_seed": 42}},
            "75": {"inputs": {"filename_prefix": "bench/original"}},
            "395": {"inputs": {"image": "i2v-source-832x448.png"}},
        }
        original = copy.deepcopy(base)

        prepared = multiworker_benchmark.prepare_workflow(
            base, seed=500101, label="pytorch-b01-w1"
        )

        self.assertEqual(prepared["398:339"]["inputs"]["noise_seed"], 500101)
        self.assertEqual(
            prepared["75"]["inputs"]["filename_prefix"],
            "bench/pytorch-b01-w1",
        )
        self.assertEqual(base, original)

    def test_validate_environment_accepts_fixed_pro6000_stack(self):
        stats = {
            "system": {
                "comfyui_version": "0.34.0",
                "pytorch_version": "2.13.0+cu130",
            },
            "devices": [{"name": "cuda:0 NVIDIA RTX PRO 6000 : cudaMallocAsync"}],
        }

        multiworker_benchmark.validate_environment(stats)

    def test_validate_environment_rejects_wrong_gpu(self):
        stats = {
            "system": {
                "comfyui_version": "0.34.0",
                "pytorch_version": "2.13.0+cu130",
            },
            "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5090"}],
        }

        with self.assertRaisesRegex(RuntimeError, "RTX PRO 6000"):
            multiworker_benchmark.validate_environment(stats)

    def test_execution_seconds_uses_comfy_timestamps(self):
        history = {
            "status": {
                "messages": [
                    ["execution_start", {"timestamp": 1000}],
                    ["execution_success", {"timestamp": 17345}],
                ]
            }
        }

        self.assertEqual(multiworker_benchmark.execution_seconds(history), 16.345)

    def test_summarize_batch_reports_throughput_and_start_skew(self):
        runs = [
            {
                "submitted_monotonic": 10.000,
                "completed_monotonic": 30.000,
                "comfy_execution_seconds": 18.0,
            },
            {
                "submitted_monotonic": 10.075,
                "completed_monotonic": 31.000,
                "comfy_execution_seconds": 19.0,
            },
            {
                "submitted_monotonic": 10.125,
                "completed_monotonic": 32.000,
                "comfy_execution_seconds": 20.0,
            },
        ]

        self.assertEqual(
            multiworker_benchmark.summarize_batch(runs),
            {
                "worker_count": 3,
                "batch_wall_seconds": 22.0,
                "start_skew_seconds": 0.125,
                "videos_per_minute": 8.182,
                "mean_comfy_execution_seconds": 19.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
