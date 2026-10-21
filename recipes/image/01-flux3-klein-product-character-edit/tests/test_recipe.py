import ast
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import klein_graph  # noqa: E402


def load(path):
    return json.loads((ROOT / path).read_text())


class RecipeTest(unittest.TestCase):
    def test_manifest_pins_revision_and_hashes(self):
        m = load('configs/model-manifest.json')
        self.assertEqual(m['repo'], 'black-forest-labs/flux-3-image-klein-comfy')
        self.assertRegex(m['revision'], r'^[0-9a-f]{40}$')
        names = {Path(f['path']).name for f in m['files']}
        for needed in (klein_graph.UNET['bf16'], klein_graph.UNET['int8'], klein_graph.TEXT_ENCODER,
                       klein_graph.UPSAMPLER, klein_graph.VAE):
            self.assertIn(needed, names)
        for f in m['files']:
            self.assertRegex(f['sha256'], r'^[0-9a-f]{64}$')
            self.assertIn(f['comfy_dir'], ('diffusion_models', 'text_encoders', 'vae'))

    def test_edit_graph_matches_template_settings(self):
        g = klein_graph.build('put the bottle from the first image on the desk in the second image',
                              ['a.png', 'b.png'], upsample=True)
        ks = g['ks']['inputs']
        self.assertEqual((ks['steps'], ks['cfg'], ks['sampler_name'], ks['scheduler']), (16, 1.0, 'euler', 'simple'))
        self.assertAlmostEqual(g['ms']['inputs']['max_shift'], 2.7725887, places=6)
        self.assertEqual(g['ref']['inputs']['resolution.megapixels'], 1.0)
        self.assertEqual(g['scale']['inputs']['image'], ['img1', 0], 'output size follows the background (last ref)')
        self.assertEqual(g['pos']['inputs']['text'], ['upgen', 0])
        self.assertEqual(g['upgen']['inputs']['sampling_mode.top_k'], 1)

    def test_upsampler_off_uses_prompt_directly(self):
        g = klein_graph.build('edit', ['a.png'])
        self.assertEqual(g['pos']['inputs']['text'], 'edit')
        self.assertNotIn('upgen', g)

    def test_example_workflows_are_current(self):
        cfg = load('configs/runs.json')
        c = next(x for x in cfg['products'] if x['id'] == 'R2-serum-desk')
        g = klein_graph.build(c['prompt'], [r + '.png' for r in c['refs']], upsample=True)
        self.assertEqual(g, load('workflows/product_edit_upsampler_api.json'))
        c = next(x for x in cfg['people'] if x['id'] == 'R2-C2-woman-rain')
        g = klein_graph.build(c['prompt'], [r + '.png' for r in c['refs']], megapixels=2.0, size_ref=0, upsample=True)
        self.assertEqual(g, load('workflows/person_edit_upsampler_api.json'))

    def test_runs_reference_known_materials(self):
        cfg = load('configs/runs.json')
        mats = {m['id'] for m in cfg['materials']}
        for c in cfg['products'] + cfg['people']:
            self.assertTrue(set(c['refs']) <= mats, c['id'])
        for c in cfg['chain']:
            self.assertIn(c['person'], mats)

    def test_system_prompt_is_from_template(self):
        text = (ROOT / 'configs/upsampler-edit-system-prompt.txt').read_text()
        self.assertTrue(text.startswith('You are the prompt upsampler of an image-editing model.'))
        self.assertIn('Comfy Org', (ROOT / 'configs/LICENSE.upstream').read_text())

    def test_scripts_parse_and_contain_no_secrets(self):
        secret = re.compile(r'(hf_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})')
        for path in list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py')):
            src = path.read_text()
            ast.parse(src)
            self.assertIsNone(secret.search(src), path.name)
        for path in (ROOT / 'configs').glob('*.json'):
            self.assertIsNone(secret.search(path.read_text()), path.name)


if __name__ == '__main__':
    unittest.main()
