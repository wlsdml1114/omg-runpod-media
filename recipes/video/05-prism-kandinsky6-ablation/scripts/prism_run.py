"""Prism Alpha를 고정 커밋 그대로 호출해 여러 설정을 한 프로세스에서 실행합니다.

사용 (Prism 소스 디렉터리에서, Prism venv의 python으로):
  $WORK_ROOT/envs/prism/bin/python scripts/prism_run.py --runs A0
  $WORK_ROOT/envs/prism/bin/python scripts/prism_run.py --runs A2 A2s A4

업스트림 파일은 수정하지 않습니다. 메모리 안에서만 바꾸는 것(수치 계산에는 영향 없음):
  * 빈 negative prompt 강제: 업스트림 CLI는 ''를 받으면 기본 부정 프롬프트로 바꿉니다.
  * 시간 측정 훅: 로드 / 디노이즈(첫 forward 시작 → 마지막 forward 반환) / 디코드 / 저장.
  * PRISM_EXPERT_SWAP=1(기본): MoE 영상 전문가(video_dit, video_dit_2, 각 28.6GB) 중 지금 쓰는
    하나만 GPU에 둡니다. H100 80GB에서 둘을 함께 올리면 OOM 위험이 큽니다.
  * 가중치 한 번만 로드: 두 번째 실행부터 같은 MOVABridge를 재사용합니다.
    인자 파싱, configure_bsa / configure_ivpq, 시드 설정은 실행마다 업스트림 main()이 다시 합니다.
  * record_shapes가 켜진 실행은 prism_shape_hook으로 동적 블록 모양을 기록합니다.
2026-10-13 실험의 prism_one.py / prism_multi.py / run_job.py에서 옮겼습니다.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

RECIPE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('WORK_ROOT', '/workspace/ablation'))
PRISM = WORK / 'source/Prism'
sys.path.insert(0, str(PRISM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

p = argparse.ArgumentParser()
p.add_argument('--runs', nargs='+', required=True, help='configs/runs.json의 label (A0, A2, A2s, A4)')
p.add_argument('--out', default=str(WORK / 'runs'))
ARGS = p.parse_args()
ALL = {r['label']: r for r in json.loads((RECIPE / 'configs/runs.json').read_text())['runs']}
RUNS = [ALL[x] for x in ARGS.runs]
assert all(r['model'] == 'prism' for r in RUNS), 'Prism 실행만 넣으세요'


def prism_argv(r, run_dir):
    q = [str(x) for x in r['bsa_chunk_3d_shape']]
    return ['--ckpt', str(WORK / 'weights/prism/pretrained_models/MOVA-360p'),
            '--resume_ckpt', str(WORK / 'weights/prism/preview_alpha/diffusion_pytorch_model.safetensors'),
            '--config', str(PRISM / 'configs/train/t2va_config/mova_infer.yaml'),
            '--ref_path', str(RECIPE / r['input']), '--output_path', str(run_dir / 'native.mp4'),
            '--height', str(r['height']), '--width', str(r['width']), '--num_frames', str(r['num_frames']),
            '--fps', '24', '--num_inference_steps', str(r['steps']), '--cfg_scale', str(r['cfg']),
            '--sp_size', '1', '--offload', r['offload'],
            '--visual_shift', str(r['visual_shift']), '--audio_shift', str(r['audio_shift']),
            '--seed', str(r['seed']), '--prompt', r['prompt'], '--audio_prompt', r['prompt'],
            '--negative_prompt', '', '--enable_bsa', 'true', '--bsa_sparsity', str(r['bsa_sparsity']),
            '--bsa_chunk_3d_shape_q', *q, '--bsa_chunk_3d_shape_k', *q,
            '--bsa_cdf_threshold', str(r['bsa_cdf_threshold']),
            # 중요: CLI 기본값은 false입니다. README 표의 기본값(true)과 다릅니다.
            '--enable_ivpq_dynamic_block', 'true' if r['dynamic_block'] else 'false']


T0 = time.perf_counter()
import torch  # noqa: E402
import hymm.sample.sample_mova_single as sampler  # noqa: E402
from hymm.diffusion.pipelines.mova_pipeline import MOVAPipeline  # noqa: E402
from hymm.models.modules.mova import MOVABridge  # noqa: E402
import prism_shape_hook  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - T0
sampler.NEGATIVE_PROMPT = ''
EXPERT_SWAP = os.environ.get('PRISM_EXPERT_SWAP', '1') == '1'
state = {}
obs = {}


def now():
    return time.perf_counter() - T0


def reset_state():
    state.clear()
    state.update(first_fwd=None, last_fwd_end=None, call_start=None, call_end=None,
                 expert=None, fwd_idx=-1, layer=0)
    obs.clear()
    obs.update(forward_calls=0, swaps=[])


reset_state()

# ---- pipeline 시간
_orig_call = MOVAPipeline.__call__


def _call(self, *a, **k):
    torch.cuda.synchronize()
    state['call_start'] = now()
    try:
        return _orig_call(self, *a, **k)
    finally:
        torch.cuda.synchronize()
        state['call_end'] = now()


MOVAPipeline.__call__ = _call


# ---- 전문가 교체 + forward 시간
def _expert_modules(bridge, second):
    if second:
        return list(bridge.video_dit_2.blocks)
    return [fb.video_block for fb in bridge.fusion_blocks] + list(bridge.remaining_video_blocks)


def _ensure_expert(bridge, second):
    if state['expert'] == second:
        return
    t = time.perf_counter()
    dev = next(bridge.audio_dit.parameters()).device
    for m in _expert_modules(bridge, not second):
        m.to('cpu')
    torch.cuda.empty_cache()
    for m in _expert_modules(bridge, second):
        m.to(dev)
    torch.cuda.synchronize()
    obs['swaps'].append({'to_expert_2': second, 'seconds': time.perf_counter() - t,
                         'at_forward': state['fwd_idx']})
    state['expert'] = second


_orig_fwd = MOVABridge.forward


def _fwd(self, *a, **k):
    state['fwd_idx'] += 1
    state['layer'] = 0
    if state['first_fwd'] is None:
        torch.cuda.synchronize()
        state['first_fwd'] = now()
    if EXPERT_SWAP and next(self.audio_dit.parameters()).is_cuda:
        _ensure_expert(self, bool(k.get('use_video_dit_2', False)) and self.video_dit_2 is not None)
    out = _orig_fwd(self, *a, **k)
    obs['forward_calls'] += 1
    state['last_fwd_end'] = now()
    return out


MOVABridge.forward = _fwd
_orig_to = MOVABridge.to


def _to(self, *a, **k):
    r = _orig_to(self, *a, **k)
    state['expert'] = None  # 디노이즈 뒤 파이프라인이 트랜스포머 전체를 CPU로 옮김
    return r


MOVABridge.to = _to
prism_shape_hook.install(state)

# ---- 가중치 한 번만 로드
CACHE = {}
_orig_from = MOVABridge.from_mova_pretrained


def _from(*a, **k):
    if 'bridge' not in CACHE:
        CACHE['bridge'] = _orig_from(*a, **k)
        return CACHE['bridge']
    CACHE['reused'] = True
    return CACHE['bridge']


MOVABridge.from_mova_pretrained = staticmethod(_from)
_SKIP = {}
_orig_sload = sampler.safetensors_load_file


def _sload(path, *a, **k):
    return _SKIP if CACHE.get('reused') else _orig_sload(path, *a, **k)


sampler.safetensors_load_file = _sload
_orig_lsd = MOVABridge.load_state_dict


def _lsd(self, sd, *a, **k):
    return ([], []) if sd is _SKIP else _orig_lsd(self, sd, *a, **k)


MOVABridge.load_state_dict = _lsd

# ---- 실행
results = []
for r in RUNS:
    run_dir = Path(ARGS.out) / r['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'request.json').write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
    reset_state()
    prism_shape_hook.start(str(run_dir / 'dynamic-block-shapes.npz') if r.get('record_shapes') else None)
    sys.argv = [sys.argv[0]] + prism_argv(r, run_dir)
    run_t0 = now()
    status = 'ok'
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        sampler.main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = type(e).__name__ + ': ' + str(e)[:500]
    end = now()

    def span(a, b):
        return (state[b] - state[a]) if state.get(a) is not None and state.get(b) is not None else None

    o = {
        'run_id': r['run_id'], 'label': r['label'], 'status': status,
        'import_seconds': IMPORT_SECONDS, 'process_reused_weights': bool(CACHE.get('reused')),
        'run_total_seconds': end - run_t0,
        'load_seconds': (state['call_start'] - run_t0) if state['call_start'] else None,
        'pre_denoise_in_call_seconds': span('call_start', 'first_fwd'),
        'denoise_seconds': span('first_fwd', 'last_fwd_end'),
        'post_denoise_decode_seconds': span('last_fwd_end', 'call_end'),
        'save_after_call_seconds': (end - state['call_end']) if state['call_end'] else None,
        'forward_calls': obs['forward_calls'], 'expert_swap': EXPERT_SWAP, 'swaps': obs['swaps'],
        'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
    }
    if o['denoise_seconds'] and r['steps']:
        o['denoise_seconds_per_step'] = o['denoise_seconds'] / r['steps']
    if r.get('record_shapes'):
        try:
            o['shape_record'] = prism_shape_hook.save()
        except Exception as e:  # noqa: BLE001
            o['shape_record'] = {'error': repr(e)}
    (run_dir / 'prism-observation.json').write_text(json.dumps(o, indent=2) + '\n')
    results.append(status)
    print(json.dumps({'run_id': r['run_id'], 'status': status,
                      'denoise_seconds': o['denoise_seconds']}), flush=True)
sys.exit(0 if all(s == 'ok' for s in results) else 5)
