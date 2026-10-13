"""고정 커밋의 Prism·Kandinsky 6 런타임을 설치합니다. 인프라를 만들거나 지우지 않습니다.

사용: python3 scripts/install.py --model prism|k6
WORK_ROOT(기본 /workspace/ablation) 아래 source/, envs/ 를 만듭니다.
2026-10-13 실험의 bootstrap.py에서 옮겼습니다.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request

RECIPE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('WORK_ROOT', '/workspace/ablation'))
PINS = {
    'Prism': ('https://github.com/Tencent-Hunyuan/Prism.git', '8f0c2633f40ccf89e26f928c6d1771616caeb510'),
    'kandinsky-6': ('https://github.com/kandinskylab/kandinsky-6.git', 'f19c9b8ec2fcea5d391f5391983a540fc5e004da'),
}
# Prism requirements.txt는 descript-audiotools>=0.7.4를 요구하지만 PyPI는 0.7.2까지만 있습니다.
# 0.7.4를 선언한 공식 소스 커밋으로 고정합니다.
AUDIOTOOLS = ('descript-audiotools @ git+https://github.com/descriptinc/audiotools.git'
              '@348ebf2034ce24e2a91a553e3171cb00c0c71678')
FLASH_ATTN_TAG = 'v2.8.3'


def call(args, cwd=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def source(name):
    url, commit = PINS[name]
    path = WORK / 'source' / name
    if not path.exists():
        path.mkdir(parents=True)
        call(['git', 'init', path])
        call(['git', '-C', path, 'remote', 'add', 'origin', url])
    call(['git', '-C', path, 'fetch', '--depth', '1', 'origin', commit])
    call(['git', '-C', path, 'checkout', '--detach', commit])
    return path


def install_k6():
    repo = source('kandinsky-6')
    # Hopper(H100/H200)용 extra: FlashAttention 3. kandinsky_sr도 uv.lock에 고정돼 함께 설치됩니다.
    call(['uv', 'sync', '--python', '3.13.8', '--frozen', '--extra', 'hopper', '--no-dev'], cwd=repo)
    python = repo / '.venv/bin/python'
    call([python, '-c', 'import torch, flash_attn_interface, kandinsky, kandinsky_sr; print(torch.__version__)'])
    return python


def install_prism():
    repo = source('Prism')
    env = WORK / 'envs/prism'
    if not (env / 'bin/python').exists():
        call(['uv', 'venv', '--python', '3.11.13', env])
    python = env / 'bin/python'
    call(['uv', 'pip', 'install', '--python', python, 'torch==2.7.1', 'torchvision==0.22.1',
          'torchaudio==2.7.1', '--index-url', 'https://download.pytorch.org/whl/cu128'])
    # flash-attn은 소스 빌드 대신 PyTorch C++ ABI에 맞는 공식 릴리스 wheel을 설치합니다.
    abi = subprocess.check_output([str(python), '-c',
                                   'import torch;print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())'],
                                  text=True).strip()
    name = f'flash_attn-2.8.3+cu12torch2.7cxx11abi{abi}-cp311-cp311-linux_x86_64.whl'
    api = f'https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/{FLASH_ATTN_TAG}'
    release = json.load(urllib.request.urlopen(api))
    wheel = next(a['browser_download_url'] for a in release['assets'] if a['name'] == name)
    call(['uv', 'pip', 'install', '--python', python, wheel])
    lines = (repo / 'requirements.txt').read_text().splitlines()
    req = WORK / 'envs/prism-requirements-patched.txt'
    req.write_text('\n'.join(AUDIOTOOLS if x.startswith('descript-audiotools') else x
                             for x in lines if not x.startswith('flash-attn')) + '\n')
    runtime = WORK / 'envs/prism-runtime-constraints.txt'
    runtime.write_text('torch==2.7.1\ntorchvision==0.22.1\ntorchaudio==2.7.1\nflash-attn==2.8.3\n')
    call(['uv', 'pip', 'install', '--python', python, '-r', req, '-c', runtime,
          '-c', RECIPE / 'configs/prism-constraints.txt'])
    call([python, '-c', 'import torch, diffusers, transformers, flash_attn, triton, audiotools, decord; '
                        'print(torch.__version__)'])
    return python


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['prism', 'k6'], required=True)
    model = p.parse_args().model
    call(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'])
    python = install_prism() if model == 'prism' else install_k6()
    freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(python)], text=True)
    (WORK / f'envs/{model}-freeze.txt').write_text(freeze)


if __name__ == '__main__':
    main()
