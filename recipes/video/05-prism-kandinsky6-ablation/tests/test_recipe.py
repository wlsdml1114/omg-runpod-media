import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runs():
    return {r['label']: r for r in json.loads((ROOT / 'configs/runs.json').read_text())['runs']}


class RecipeTest(unittest.TestCase):
    def test_prism_runs_change_one_factor_at_a_time(self):
        r = runs()
        self.assertFalse(r['A0']['dynamic_block'])
        self.assertTrue(r['A2']['dynamic_block'])
        self.assertEqual(r['A0']['bsa_sparsity'], r['A2']['bsa_sparsity'])
        self.assertEqual(r['A2']['bsa_sparsity'], 0.75)
        self.assertEqual(r['A4']['bsa_sparsity'], 0.93)
        ignore = {'run_id', 'label', 'dynamic_block', 'bsa_sparsity', 'record_shapes'}
        for a, b in [('A0', 'A2'), ('A2', 'A4')]:
            diff = {k for k in r[a].keys() | r[b].keys() if k not in ignore and r[a].get(k) != r[b].get(k)}
            self.assertEqual(diff, set(), (a, b))
        self.assertTrue(r['A2'].get('record_shapes'))

    def test_shared_seed_size_and_negative_prompt(self):
        for label, r in runs().items():
            self.assertEqual(r['seed'], 42001, label)
            self.assertEqual(r['negative_prompt'], '', label)
            if r['model'] == 'prism':
                self.assertEqual((r['width'], r['height'], r['num_frames'], r['steps'], r['cfg']),
                                 (1280, 720, 121, 50, 5))
            else:
                self.assertEqual((r['checkpoint'], r['width'], r['height'], r['steps'], r['guidance']),
                                 ('pro-distill', 768, 432, 10, 1.0))
                self.assertTrue(r['sr'])

    def test_beautifier_only_differs_between_c2_and_c3(self):
        r = runs()
        self.assertEqual(r['C2']['beautifier'], 'none')
        self.assertEqual(r['C3']['beautifier'], 'qwen25')
        self.assertEqual(r['C2']['prompt'], r['C3']['prompt'])

    def test_inputs_match_frozen_hashes(self):
        for label, r in runs().items():
            data = (ROOT / r['input']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), r['input_sha256'], label)

    def test_manifest_pins_revisions(self):
        manifest = {m['key']: m for m in json.loads((ROOT / 'configs/model-manifest.json').read_text())}
        self.assertEqual(manifest['prism']['revision'], '0ff9e33dc323a9a97ca3defcc019d745e93f3a26')
        self.assertEqual(manifest['k6distill']['revision'], '80d5e6c7f646c5f512b684e2da0765618d805894')
        self.assertEqual(manifest['k6sr']['revision'], 'f58a2158fdec454c5265088543e2b0eae5c1be3c')
        for m in manifest.values():
            self.assertTrue(all(re.fullmatch(r'[0-9a-f]{64}', f['sha256_lfs'] or '0' * 64) for f in m['files']))

    def test_scripts_parse_and_contain_no_secrets(self):
        for path in (ROOT / 'scripts').glob('*.py'):
            ast.parse(path.read_text(), str(path))
        pattern = re.compile(r'hf_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{20,}|' + '/Users/')
        for path in ROOT.rglob('*'):
            if path.is_file() and path.parent.name != 'tests' and path.suffix in {'.py', '.sh', '.json', '.md', '.txt'}:
                self.assertIsNone(pattern.search(path.read_text(errors='ignore')), str(path))

    def test_block_shape_summary_matches_readme_numbers(self):
        s = json.loads((ROOT / 'evidence/A2-block-shape-summary.json').read_text())
        self.assertEqual(s['grid_padded_THW_tokens'], [32, 48, 80])
        self.assertEqual(s['calls'], 4000)
        self.assertAlmostEqual(s['shape_usage_fraction_all_calls']['(4, 4, 4)'], 0.6225)
        per_step = s['per_step_shape_fraction_posi']
        self.assertAlmostEqual(per_step[0][0], 0.7528)
        self.assertAlmostEqual(per_step[-1][0], 0.4391, places=3)
        self.assertAlmostEqual(s['selected_kblocks_per_qblock_mean_over_calls'], 480, delta=1)


if __name__ == '__main__':
    unittest.main()
