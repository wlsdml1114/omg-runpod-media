"""Run the recipe stages against a local ComfyUI started from WORK_ROOT.

usage: python scripts/run_recipe.py materials|products|people|chain [--ids ID ...] [--no-upsampler]

Every run writes benchmark-output/<run_id>/{out.png, workflow.api.json, caption.txt} and appends one line
to benchmark-output/runs.jsonl. Time is measured from POST /prompt to the saved PNG, so it includes the
prompt upsampler and VAE decode, and also model loading on the first run of a session.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import klein_graph  # noqa: E402

RECIPE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('WORK_ROOT', '/workspace/klein'))
COMFY = WORK / 'ComfyUI'
OUT = Path(os.environ.get('OUTPUT_DIR', RECIPE / 'benchmark-output'))
SERVER = 'http://127.0.0.1:8188'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def call(path, payload=None):
    req = urllib.request.Request(SERVER + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'} if payload else {})
    with urllib.request.urlopen(req, timeout=120) as f:
        return json.load(f)


def ensure_server():
    try:
        call('/system_stats')
        return
    except Exception:
        pass
    log = open(WORK / 'comfy-server.log', 'a')
    subprocess.Popen([str(WORK / 'env/bin/python'), 'main.py', '--listen', '127.0.0.1', '--port', '8188',
                      '--disable-auto-launch', '--preview-method', 'none'],
                     cwd=COMFY, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(300):
        try:
            call('/system_stats')
            return
        except Exception:
            time.sleep(2)
    sys.exit('ComfyUI did not start; see comfy-server.log')


def run(run_id, graph, keep_as_input=None, limit=1800):
    out = OUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / 'workflow.api.json').write_text(json.dumps(graph, ensure_ascii=False, indent=1))
    rec = {'run_id': run_id, 'started_at': now()}
    t0 = time.monotonic()
    try:
        accepted = call('/prompt', {'prompt': graph})
        if accepted.get('node_errors'):
            raise RuntimeError(json.dumps(accepted['node_errors'])[:2000])
        pid = accepted['prompt_id']
        hist = None
        while time.monotonic() - t0 < limit:
            h = call('/history/' + pid).get(pid)
            if h and (h['status'].get('completed') or h['status'].get('status_str') == 'error'):
                hist = h
                break
            time.sleep(0.5)
        if hist is None:
            raise TimeoutError('timeout')
        if hist['status'].get('status_str') != 'success':
            raise RuntimeError('execution error: ' + json.dumps(hist['status'].get('messages', []))[-1500:])
        image = [i for o in hist['outputs'].values() for i in o.get('images', [])][-1]
        src = COMFY / 'output' / image.get('subfolder', '') / image['filename']
        shutil.copy2(src, out / 'out.png')
        if keep_as_input:
            shutil.copy2(src, COMFY / 'input' / f'{keep_as_input}.png')
        texts = [t for o in hist['outputs'].values() for t in (o.get('text') or [])]
        if texts:
            (out / 'caption.txt').write_text(texts[-1] if isinstance(texts[-1], str) else json.dumps(texts[-1]))
        rec.update(status='success', seconds=round(time.monotonic() - t0, 2))
    except Exception as e:  # keep going so one failure does not stop a batch
        rec.update(status='failed', error=str(e), seconds=round(time.monotonic() - t0, 2))
    rec['finished_at'] = now()
    with (OUT / 'runs.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    print(rec['status'].upper(), run_id, rec['seconds'], rec.get('error', ''), flush=True)
    return rec


def done(run_id):
    return (OUT / run_id / 'out.png').exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['materials', 'products', 'people', 'chain'])
    p.add_argument('--ids', nargs='*', default=[])
    p.add_argument('--no-upsampler', action='store_true', help='turn the prompt upsampler off for edits')
    p.add_argument('--model', choices=list(klein_graph.UNET), default='bf16')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RECIPE / 'configs/runs.json').read_text())
    want = lambda i: not a.ids or i in a.ids  # noqa: E731
    up = not a.no_upsampler
    ensure_server()

    def material(name):
        src = OUT / name / 'out.png'
        if not src.exists():
            sys.exit(f'missing material {name}: run the materials stage first')
        shutil.copy2(src, COMFY / 'input' / f'{name}.png')
        return f'{name}.png'

    if a.stage == 'materials':
        for m in cfg['materials']:
            if want(m['id']) and not done(m['id']):
                run(m['id'], klein_graph.build(m['prompt'], width=m['w'], height=m['h'], seed=m['seed'],
                                               model=a.model), keep_as_input=m['id'])
    elif a.stage == 'products':
        for c in cfg['products']:
            for seed in c.get('seeds', [42]):
                rid = c['id'] + ('' if seed == 42 else f'-s{seed}')
                if want(c['id']) and not done(rid):
                    refs = [material(r) for r in c['refs']]
                    run(rid, klein_graph.build(c['prompt'], refs, upsample=up, seed=seed, model=a.model))
    elif a.stage == 'people':
        for c in cfg['people']:
            for seed in c.get('seeds', [42]):
                rid = c['id'] + ('' if seed == 42 else f'-s{seed}')
                if want(c['id']) and not done(rid):
                    refs = [material(r) for r in c['refs']]
                    run(rid, klein_graph.build(c['prompt'], refs, megapixels=2.0, size_ref=0, upsample=up,
                                               seed=seed, model=a.model))
    else:
        for c in cfg['chain']:
            if not want(c['id']):
                continue
            first = material(c['person'])
            if c.get('also_original'):
                for s in c['scenes']:
                    rid = f"{c['id']}-orig-{s['id']}"
                    if not done(rid):
                        run(rid, klein_graph.build(s['prompt'], [first], megapixels=2.0, size_ref=0, upsample=up,
                                                   model=a.model))
            prev = first
            for s in c['scenes']:
                rid = f"{c['id']}-chain-{s['id']}"
                if done(rid):
                    shutil.copy2(OUT / rid / 'out.png', COMFY / 'input' / f'{rid}.png')
                else:
                    run(rid, klein_graph.build(s['prompt'], [prev], megapixels=2.0, size_ref=0, upsample=up,
                                               model=a.model), keep_as_input=rid)
                prev = f'{rid}.png'


if __name__ == '__main__':
    main()
