"""Install pinned ComfyUI and download the pinned FLUX 3 Klein files into WORK_ROOT (default /workspace/klein).

Run inside the Pod. Needs HF_TOKEN in the environment, and the license accepted on
https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy for that account.
"""
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('WORK_ROOT', '/workspace/klein'))
COMFY = WORK / 'ComfyUI'
COMFY_TAG = 'v0.42.0'
COMFY_COMMIT = '355cd6b14c92c19ac3188fe528e47a3f21ef5178'


def run(args, cwd=None):
    print('RUN', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    env = WORK / 'env'
    if sys.prefix == sys.base_prefix:
        if not env.exists():
            run([sys.executable, '-m', 'venv', '--system-site-packages', env])
        os.execv(str(env / 'bin/python'), [str(env / 'bin/python'), str(Path(__file__).resolve())])

    if not os.environ.get('HF_TOKEN'):
        sys.exit('HF_TOKEN is not set. Accept the license on Hugging Face, then: read -rs HF_TOKEN && export HF_TOKEN')

    if not COMFY.exists():
        run(['git', 'clone', 'https://github.com/Comfy-Org/ComfyUI.git', COMFY])
    run(['git', '-C', COMFY, 'fetch', '--tags'])
    run(['git', '-C', COMFY, 'checkout', '--detach', COMFY_TAG])
    head = subprocess.check_output(['git', '-C', str(COMFY), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMFY_COMMIT:
        sys.exit(f'{COMFY_TAG} resolved to {head}, expected {COMFY_COMMIT}')

    import torch
    pins = WORK / 'torch-constraints.txt'
    lines = [f'torch=={torch.__version__}']
    for name in ('torchvision', 'torchaudio'):
        try:
            lines.append(f'{name}=={__import__(name).__version__}')
        except ImportError:
            pass
    pins.write_text('\n'.join(lines) + '\n')
    run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-r', COMFY / 'requirements.txt',
         '-r', RECIPE / 'requirements.txt', '-c', pins])
    for name in ('input', 'output', 'user'):
        (COMFY / name).mkdir(exist_ok=True)

    from huggingface_hub import hf_hub_download
    manifest = json.loads((RECIPE / 'configs/model-manifest.json').read_text())

    def download(item):
        start = time.time()
        path = hf_hub_download(repo_id=manifest['repo'], filename=item['path'],
                               revision=manifest['revision'], local_dir=COMFY / 'models')
        if os.environ.get('SKIP_SHA256') != '1' and sha256(path) != item['sha256']:
            raise SystemExit(f"SHA-256 mismatch: {item['path']}")
        print('OK', item['path'], f'{time.time() - start:.1f}s', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, manifest['files']))
    print('SETUP_COMPLETE', json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                                        'gpu': torch.cuda.get_device_name(0), 'comfyui': COMFY_TAG}), flush=True)


if __name__ == '__main__':
    main()
