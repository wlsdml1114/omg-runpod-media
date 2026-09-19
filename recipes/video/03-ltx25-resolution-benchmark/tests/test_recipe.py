import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RecipeTest(unittest.TestCase):
    def test_api_workflow_has_fixed_duration_fps_and_output(self):
        workflow = json.loads((ROOT / "workflows/ltx25_t2v_api.json").read_text())
        self.assertEqual(workflow["405:362"]["inputs"]["value"], 5)
        self.assertEqual(workflow["405:361"]["inputs"]["value"], 24)
        self.assertEqual(workflow["75"]["class_type"], "SaveVideo")

    def test_matrix_has_five_warm_runs_per_resolution(self):
        with (ROOT / "inputs/run-matrix.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        counts = {}
        for row in rows:
            if row["phase"] != "warm" or row["include_in_aggregate"].lower() != "true":
                continue
            key = (int(row["requested_width"]), int(row["requested_height"]))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts, {(832, 448): 5, (1280, 704): 5, (1984, 1088): 5})


if __name__ == "__main__":
    unittest.main()
