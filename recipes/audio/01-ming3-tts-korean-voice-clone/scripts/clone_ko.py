"""Korean voice cloning with Ming-3.0-tts 0.5B (runs inside the Pod, venv active).

Korean has no built-in voice in Ming, so every request clones a reference
recording: task=MULTILINGUAL, language_id="ko", plus the reference wav and its
exact transcript. Settings match the 2026-10-22 run (Ming defaults, the text is
split into pieces and each piece gets the reference in front).

One sentence:
  python scripts/clone_ko.py --repo /workspace/Ming-3.0-tts \
    --model-path /workspace/models/Ming-3.0-tts-0.5b \
    --ref-wav ref/my_voice.wav --ref-text-file ref/my_voice.txt \
    --text "근데 이번엔 목소리를 바꿔 볼 거예요."

The 23 test sentences, 3 takes each (69 files, what the video compared):
  python scripts/clone_ko.py ... --sentences inputs/sentences.tsv --takes 3
  python scripts/clone_ko.py ... --sentences inputs/sentences.tsv --only S01 N03 --takes 1

Outputs: out/<id>_t<take>.wav (44.1 kHz) and one JSON line per request in
runs.jsonl (generation seconds, audio seconds, failures too). Existing wavs are
skipped unless --force, so an interrupted run can simply be started again.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
# sentences.tsv keeps neutral Korean tags; Ming's own syntax is [Laughter] / [Sigh]
# (officially Chinese/English only — whether they work in Korean is part of the test).
TAGS = {"웃음": "[Laughter]", "한숨": "[Sigh]"}
TAG_RE = re.compile(r"\[(웃음|한숨)\]\s*")


def render_text(text, tags="convert"):
    if tags == "strip":
        return TAG_RE.sub("", text).strip()
    return TAG_RE.sub(lambda m: TAGS[m.group(1)] + " ", text).strip()


def load_jobs(a):
    if a.text:
        return [{"id": a.id, "text": a.text}]
    with open(a.sentences, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if a.only:
        wanted = set(a.only)
        rows = [r for r in rows if r["id"] in wanted or r["group"] in wanted]
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="Ming-3.0-tts checkout (setup_pod.sh pins the commit)")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ref-wav", required=True, help="10–15 s of your own voice, one speaker, no music")
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument("--ref-text", help="exact transcript of --ref-wav")
    ref.add_argument("--ref-text-file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="one sentence to read")
    src.add_argument("--sentences", help="TSV with id, group, text columns (inputs/sentences.tsv)")
    ap.add_argument("--id", default="text", help="output name for --text")
    ap.add_argument("--only", nargs="*", help="sentence ids or groups from --sentences")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000, help="take N uses seed-base + N (video: 1001–1003)")
    ap.add_argument("--tags", choices=["convert", "strip"], default="convert")
    ap.add_argument("--raw-text", action="store_true",
                    help="send each sentence in one piece instead of Ming's default splitting (not used in the video)")
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    ap.add_argument("--runs", default=str(HERE / "runs.jsonl"))
    ap.add_argument("--force", action="store_true", help="overwrite existing wavs")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    ref_text = a.ref_text if a.ref_text is not None else Path(a.ref_text_file).read_text(encoding="utf-8").strip()
    ref_wav = str(Path(a.ref_wav).resolve())
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = Path(a.runs).resolve()
    rows = load_jobs(a)

    sys.path.insert(0, a.repo)
    os.chdir(a.repo)  # Ming loads some assets by relative path
    import torch
    from ming_tts import MingTTS

    t0 = time.perf_counter()
    model = MingTTS(a.model_path)
    load_s = round(time.perf_counter() - t0, 2)
    gpu = torch.cuda.get_device_name(0)
    print(f"loaded in {load_s}s on {gpu}, sample_rate={model.sample_rate}")

    for row in rows:
        for take in range(1, a.takes + 1):
            path = out_dir / f"{row['id']}_t{take}.wav"
            if path.exists() and not a.force:
                continue
            text = render_text(row["text"], a.tags)
            seed = a.seed_base + take
            torch.manual_seed(seed)
            rec = {"model": "ming-3.0-tts-0.5b", "id": row["id"], "take": take, "seed": seed,
                   "text_sent": text, "gpu": gpu, "load_s": load_s, "raw_text": a.raw_text,
                   "out": path.name}
            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                wav = model.synthesize(
                    task="MULTILINGUAL",
                    language_id="ko",
                    text=text,
                    prompt_wav_path=ref_wav,
                    prompt_text=ref_text,
                    use_spk_emb=True,
                    raw_text=a.raw_text,
                    output_path=str(path),
                )
                torch.cuda.synchronize()
                gen_s = time.perf_counter() - t
                audio_s = wav.shape[-1] / model.sample_rate
                rec.update(ok=True, gen_s=round(gen_s, 3), audio_s=round(audio_s, 3),
                           rtf=round(gen_s / audio_s, 3) if audio_s else None,
                           sample_rate=model.sample_rate)
                print(f"{row['id']} t{take}: {gen_s:.2f}s -> {audio_s:.2f}s audio")
            except Exception as e:  # keep failures in runs.jsonl
                rec.update(ok=False, error=f"{type(e).__name__}: {e}")
                print(f"{row['id']} t{take}: FAILED {e}")
            rec["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            with open(runs, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
