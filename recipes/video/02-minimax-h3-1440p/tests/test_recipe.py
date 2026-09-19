import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RecipeTest(unittest.TestCase):
    def test_workflow_uses_official_turbo_8_step_path(self):
        workflow = json.loads((ROOT / "workflows/minimax_h3_turbo8_api.json").read_text())
        self.assertEqual(workflow["140:138"]["inputs"]["value"], 8)
        self.assertTrue(workflow["140:139"]["inputs"]["value"])
        self.assertIn("turbo_8step", workflow["140:134"]["inputs"]["lora_name"])
        self.assertEqual(workflow["140:134"]["inputs"]["strength_model"], 1)

    def test_matrix_contains_all_published_resolution_cells(self):
        with (ROOT / "inputs/run-matrix.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        sizes = {(int(row["requested_width"]), int(row["requested_height"])) for row in rows}
        self.assertTrue({(768, 448), (1920, 1088), (2560, 1440)} <= sizes)


if __name__ == "__main__":
    unittest.main()
