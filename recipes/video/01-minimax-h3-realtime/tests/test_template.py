import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TemplateTests(unittest.TestCase):
    def test_api_workflow_uses_four_step_nvenc_path(self):
        workflow = json.loads((ROOT / "workflows/minimax_h3_fl2va_4step_nvenc_api.json").read_text())
        self.assertEqual(workflow["92"]["class_type"], "H3NVENCSaveVideo")
        self.assertEqual(workflow["140:138"]["inputs"]["value"], 4)
        self.assertTrue(workflow["140:139"]["inputs"]["value"])
        self.assertIn("turbo_4step", workflow["140:134"]["inputs"]["lora_name"])
        self.assertEqual(workflow["140:131"]["inputs"]["width"], 608)
        self.assertEqual(workflow["140:131"]["inputs"]["height"], 352)

    def test_ui_workflow_uses_nvenc_node(self):
        workflow = json.loads((ROOT / "workflows/minimax_h3_fl2va_4step_nvenc_ui.json").read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(nodes[92]["type"], "H3NVENCSaveVideo")
        self.assertEqual(nodes[92]["widgets_values"], ["minimax_h3_nvenc/output", 18, "p1"])

    def test_python_scripts_parse(self):
        for relative in (
            "scripts/check_setup.py",
            "scripts/run_benchmark.py",
            "custom_nodes/oh_my_gpu_h3_nvenc/nodes.py",
        ):
            path = ROOT / relative
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")

    def test_model_files_are_not_bundled(self):
        forbidden = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
        bundled = [path for path in ROOT.rglob("*") if path.suffix.lower() in forbidden]
        self.assertEqual(bundled, [])


if __name__ == "__main__":
    unittest.main()
