import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import launch_workers


class LaunchWorkersTest(unittest.TestCase):
    def test_build_worker_command_isolates_mutable_directories(self):
        command = launch_workers.build_worker_command(
            worker_index=2, port=3002, attention="pytorch"
        )

        self.assertIn("--models-directory", command)
        self.assertIn("/workspace/ComfyUI/models", command)
        self.assertIn("--input-directory", command)
        self.assertIn("/workspace/ComfyUI/input", command)
        self.assertIn("/workspace/ComfyUI/workers/2/output", command)
        self.assertIn("/workspace/ComfyUI/workers/2/temp", command)
        self.assertIn("/workspace/ComfyUI/workers/2/user", command)
        self.assertIn("--cache-ram", command)
        self.assertNotIn("--cache-none", command)
        self.assertNotIn("--use-sage-attention", command)

    def test_build_worker_command_enables_sage_only_for_sage_cell(self):
        command = launch_workers.build_worker_command(
            worker_index=0, port=3000, attention="sage"
        )

        self.assertIn("--use-sage-attention", command)

    def test_build_worker_command_rejects_unknown_attention(self):
        with self.assertRaisesRegex(ValueError, "attention"):
            launch_workers.build_worker_command(
                worker_index=0, port=3000, attention="mystery"
            )


if __name__ == "__main__":
    unittest.main()
