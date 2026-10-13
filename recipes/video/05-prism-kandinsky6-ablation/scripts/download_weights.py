"""고정 revision의 필요한 파일만 받고 SHA-256을 확인합니다.

사용: HF_TOKEN=... python scripts/download_weights.py --model prism k6distill k6sr
토큰은 환경 변수로만 받습니다. 스크립트는 토큰을 출력하거나 저장하지 않습니다.
Kandinsky 6 저장소는 Hugging Face에서 연락처 공유에 동의해야 받을 수 있는 gated 저장소입니다.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('WORK_ROOT', '/workspace/ablation'))
TARGET = {
    'prism': {'local_dir': WORK / 'weights/prism'},
    'k6sr': {'local_dir': WORK / 'weights/k6-sr'},
    # Kandinsky 런타임은 KANDINSKY_HOME/weights 아래 HF 캐시 구조에서 체크포인트를 찾습니다.
    'k6distill': {'cache_dir': WORK / 'weights/k6-cache/weights'},
    'k6pro': {'cache_dir': WORK / 'weights/k6-cache/weights'},
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec):
    from huggingface_hub import snapshot_download
    paths = [f['path'] for f in spec['files']]
    root = Path(snapshot_download(repo_id=spec['repository'], revision=spec['revision'],
                                  allow_patterns=paths, token=os.environ.get('HF_TOKEN'),
                                  **TARGET[spec['key']]))
    checked = 0
    for f in spec['files']:
        path = root / f['path']
        if not path.is_file():
            raise FileNotFoundError(f['path'])
        if f.get('sha256_lfs'):
            if sha256(path) != f['sha256_lfs']:
                raise ValueError('Weight hash mismatch: ' + f['path'])
            checked += 1
    return {'repository': spec['repository'], 'revision': spec['revision'], 'root': str(root),
            'files': len(paths), 'sha256_verified': checked}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', nargs='+', choices=list(TARGET), required=True)
    keys = p.parse_args().model
    manifest = {m['key']: m for m in json.loads((RECIPE / 'configs/model-manifest.json').read_text())}
    for key in keys:
        print(json.dumps(fetch(manifest[key]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
