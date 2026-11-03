import csv
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clone_ko  # noqa: E402  (imports torch/Ming only inside main())

MING_COMMIT = "028c1eace193e7acdb256453da89b6e9207e052b"
MODEL_REV = "8d66b70a00f05ad183fa8a863b4ce609fc83cbc0"


class RecipeTest(unittest.TestCase):
    def test_files_exist(self):
        for p in ("README.md", "inputs/README.md", "inputs/sentences.tsv", "scripts/setup_pod.sh",
                  "scripts/clone_ko.py", "evidence/results-summary.json", ".gitignore"):
            self.assertTrue((ROOT / p).is_file(), p)

    def test_setup_pins_code_model_and_hashes(self):
        s = (ROOT / "scripts/setup_pod.sh").read_text()
        self.assertIn(f'MING_COMMIT="{MING_COMMIT}"', s)
        self.assertIn(f'MODEL_REV="{MODEL_REV}"', s)
        self.assertIn('MODEL_ID="inclusionAI/Ming-3.0-tts-0.5B"', s)
        self.assertEqual(len(re.findall(r'_SHA256="[0-9a-f]{64}"', s)), 2)
        self.assertIn("sha256sum -c", s)
        self.assertIn("tools/preflight.py --model_path", s)
        self.assertIn("set -euo pipefail", s)

    def test_readme_and_evidence_agree_on_pins(self):
        readme = (ROOT / "README.md").read_text()
        ev = json.loads((ROOT / "evidence/results-summary.json").read_text())
        self.assertIn(MING_COMMIT, readme)
        self.assertIn(MODEL_REV, readme)
        self.assertEqual(ev["ming_commit"], MING_COMMIT)
        self.assertTrue(ev["model"].endswith("@" + MODEL_REV))

    def test_sentences(self):
        with open(ROOT / "inputs/sentences.tsv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(len(rows), 23)
        self.assertEqual(len({r["id"] for r in rows}), 23)
        self.assertTrue(all(r["text"].strip() for r in rows))
        groups = {}
        for r in rows:
            groups[r["group"]] = groups.get(r["group"], 0) + 1
        self.assertEqual(sum(groups.values()), 23)

    def test_tag_conversion(self):
        self.assertEqual(clone_ko.render_text("[웃음] 됐네요."), "[Laughter] 됐네요.")
        self.assertEqual(clone_ko.render_text("하… [한숨] 또요?"), "하… [Sigh] 또요?")
        self.assertEqual(clone_ko.render_text("[웃음] 됐네요.", "strip"), "됐네요.")

    def test_args(self):
        base = ["--repo", "r", "--model-path", "m", "--ref-wav", "a.wav", "--ref-text", "t"]
        a = clone_ko.parse_args(base + ["--text", "안녕하세요."])
        self.assertEqual((a.takes, a.seed_base, a.raw_text), (1, 1000, False))
        a = clone_ko.parse_args(base + ["--sentences", "inputs/sentences.tsv", "--only", "S01", "N03"])
        a.text = None
        self.assertEqual([r["id"] for r in clone_ko.load_jobs(a)], ["S01", "N03"])
        with self.assertRaises(SystemExit):
            clone_ko.parse_args(base)  # needs --text or --sentences

    def test_evidence_numbers_match_readme(self):
        ev = json.loads((ROOT / "evidence/results-summary.json").read_text())
        s = ev["ming_summary"]
        self.assertEqual((s["files"], s["failed"]), (69, 0))
        self.assertEqual(len(ev["runs"]), 69)
        self.assertAlmostEqual(s["gen_s_median"], 5.9, delta=0.05)
        self.assertAlmostEqual(s["rtf_median"], 1.46, delta=0.005)
        self.assertEqual({r["seed"] for r in ev["runs"]}, {1001, 1002, 1003})
        self.assertTrue(all(r["sample_rate"] == 44100 for r in ev["runs"]))
        c = ev["comparison_same_sentences"]
        self.assertEqual(c["speaker_similarity_position_pct"]["ming"], 95)
        self.assertEqual(c["blind_round2_most_like_me"]["ming"], 17)
        readme = (ROOT / "README.md").read_text()
        for needle in ("5.9초", "1.46", "| **95** |", "| **17** |", "1,466초"):
            self.assertIn(needle, readme)

    def test_no_voice_files_or_secrets_committed(self):
        ignore = (ROOT / ".gitignore").read_text()
        for p in ("ref/", "out/", "runs.jsonl", "*.wav"):
            self.assertIn(p, ignore)
        self.assertFalse(list(ROOT.rglob("*.wav")))
        for p in ROOT.rglob("*"):
            if p.is_file() and p.suffix in (".md", ".py", ".sh", ".json", ".tsv"):
                t = p.read_text(encoding="utf-8")
                self.assertNotRegex(t, r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{20,})", p.name)


if __name__ == "__main__":
    unittest.main()
