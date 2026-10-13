"""Kandinsky 6 Pro-distill(또는 Pro)을 한 번만 로드해 여러 클립을 만들고, 이어서 공식 VSR ×2를 적용합니다.

사용 (kandinsky-6 venv의 python으로):
  $WORK_ROOT/source/kandinsky-6/.venv/bin/python scripts/k6_run.py --checkpoint pro-distill --runs C1 C1s C2 C3

나누어 재는 구간: 모델 로드 / 클립별 생성(프롬프트 보강 · 디노이즈 · 디코드) / SR 로드 + 커널 워밍업 / 클립별 SR.
업스트림 파일은 수정하지 않습니다. 고정 커밋의 SR 헬퍼는 `warmup` 인자 이름이 함수 이름을 가려
warmup=False가 무시되는 문제가 있어, 여기서는 워밍업 커널을 한 번 직접 호출합니다.
2026-10-13 실험의 k6_batch.py에서 옮겼습니다.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

RECIPE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('WORK_ROOT', '/workspace/ablation'))
REVISIONS = {'pro': '295ca976810e318fb45c870ba4fc72c9a37fd548',
             'pro-distill': '80d5e6c7f646c5f512b684e2da0765618d805894'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, choices=list(REVISIONS))
    p.add_argument('--runs', nargs='+', required=True, help='configs/runs.json의 label (C1, C1s, C2, C3)')
    p.add_argument('--out', default=str(WORK / 'runs'))
    p.add_argument('--batch-id', default='k6-batch')
    a = p.parse_args()
    out_root = Path(a.out)
    batch_dir = out_root / a.batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    os.environ['KANDINSKY_HOME'] = str(WORK / 'weights/k6-cache')
    os.environ['KANDINSKY_LOG_DIR'] = str(batch_dir / 'k6-logs')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    all_runs = {r['label']: r for r in json.loads((RECIPE / 'configs/runs.json').read_text())['runs']}
    reqs = [all_runs[x] for x in a.runs]
    assert all(r['model'] == 'k6' and r['checkpoint'] == a.checkpoint for r in reqs), 'checkpoint가 다른 실행이 섞였습니다'

    t0 = time.perf_counter()
    import torch
    import yaml
    import kandinsky.runtime.weights as weights
    import kandinsky.cli as cli
    import kandinsky.pipeline.pipeline as kpipe
    from kandinsky.pipeline.factory import get_pipeline
    from kandinsky.pipeline.config import load_config
    from kandinsky.core.components.beautifier import build_beautifier, NullBeautifier
    weights.CHECKPOINT_REVISION = REVISIONS[a.checkpoint]

    first = reqs[0]
    config = {
        'checkpoint': a.checkpoint,
        'generation': {'height': first['height'], 'width': first['width'], 'sample_frames': 121,
                       'latent_frames': first['latent_frames'], 'num_steps': first['steps'],
                       'guidance_weight': first['guidance'],
                       'visual_cond_scheme': first['visual_cond_scheme']},
        'cache': {'mode': 'none'}, 'compile': {'strategy': 'none'},
        'offload': {'strategy': 'module'}, 'attention': {'engine': 'flash_attention_3'},
        'beautifier': {'name': 'none'}, 'text_embedder': {'quantized_qwen': False},
        'sr': {'enabled': False, 'checkpoint_path': str(WORK / 'weights/k6-sr'),
               'resolution_scale': 2, 'num_steps': 5, 'seed': first['seed']}}
    config_path = batch_dir / 'k6-config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (batch_dir / 'k6-resolved-config.json').write_text(load_config(config_path).model_dump_json(indent=2) + '\n')
    cli.configure_logging()

    phase = {}

    def timed(mod, name, key):
        orig = getattr(mod, name)

        def wrapped(*x, **k):
            torch.cuda.synchronize()
            s = time.perf_counter()
            try:
                return orig(*x, **k)
            finally:
                torch.cuda.synchronize()
                phase[key] = phase.get(key, 0.0) + time.perf_counter() - s
        setattr(mod, name, wrapped)

    for n in ['denoise_loop', 'piflow_denoise_loop']:
        timed(kpipe, n, 'denoise_seconds')
    timed(kpipe, 'postprocess_video', 'video_decode_seconds')
    timed(kpipe, 'postprocess_audio', 'audio_decode_seconds')
    timed(kpipe, 'mux_video_audio', 'mux_write_seconds')

    summary = {'checkpoint': a.checkpoint, 'revision': REVISIONS[a.checkpoint], 'runs': {},
               'import_seconds': time.perf_counter() - t0}
    s = time.perf_counter()
    pipe = get_pipeline(str(config_path), device='cuda', attention_engine='flash_attention_3',
                        cache_mode='none', offload_strategy='module')
    torch.cuda.synchronize()
    summary['model_load_seconds'] = time.perf_counter() - s

    held = []
    for r in reqs:
        run = out_root / r['run_id']
        run.mkdir(parents=True, exist_ok=True)
        (run / 'request.json').write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
        rec = {'run_id': r['run_id'], 'label': r['label']}
        try:
            pipe.beautifier = build_beautifier('qwen25') if r['beautifier'] == 'qwen25' else NullBeautifier()
            phase.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            s = time.perf_counter()
            res = pipe(text=r['prompt'], height=r['height'], width=r['width'], latent_frames=r['latent_frames'],
                       num_steps=r['steps'], guidance_weight=r['guidance'], seed=r['seed'],
                       negative_text=r['negative_prompt'], save_path=str(run / 'native.mp4'),
                       image=str(RECIPE / r['input']), sample_audio=True, show_progress=True,
                       visual_cond_scheme=r['visual_cond_scheme'])
            torch.cuda.synchronize()
            rec['generation_seconds'] = time.perf_counter() - s
            rec.update(phase)
            rec['max_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
            (run / 'prompt-used.txt').write_text('\n\n'.join(res.prompts) + '\n', encoding='utf-8')
            rec['status'] = 'native_ok'
            if r.get('sr'):
                held.append((r, res.frames[0].permute(1, 0, 2, 3).contiguous().cpu(), res.audio, run))
            del res
        except Exception as e:  # noqa: BLE001
            rec['status'] = 'failed'
            rec['error'] = type(e).__name__ + ': ' + str(e)[:800]
            (run / 'error.txt').write_text(traceback.format_exc())
            torch.cuda.empty_cache()
        summary['runs'][r['run_id']] = rec
        (batch_dir / 'batch-observation.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')

    fps, audio_fps = int(pipe.fps), pipe.audio_fps
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    if held:
        from kandinsky.pipeline.sr import load_sr_pipeline
        from kandinsky_sr.pipeline.components import scale_factor_for
        from kandinsky_sr.pipeline.warmup import clip_base_resolution, compile_vae_decode, warmup as warmup_kernels
        sr_cfg = dict(config, sr=dict(config['sr'], enabled=True))
        sr_cfg_path = batch_dir / 'k6-sr-config.yaml'
        sr_cfg_path.write_text(yaml.safe_dump(sr_cfg, sort_keys=False))
        s = time.perf_counter()
        sr_pipe = load_sr_pipeline(str(sr_cfg_path), 'cuda', force=True, offload_strategy='module')
        torch.cuda.synchronize()
        summary['sr_load_seconds'] = time.perf_counter() - s
        s = time.perf_counter()
        with sr_pipe.offload.use('dit'):
            warmup_kernels(sr_pipe.dit, scale_factor_for(sr_pipe), 'cuda')
        with sr_pipe.offload.use('vae'):
            compile_vae_decode(sr_pipe, 'cuda', bases=[clip_base_resolution(sr_pipe, tuple(held[0][1].shape[-2:]))])
        torch.cuda.synchronize()
        summary['sr_kernel_warmup_seconds'] = time.perf_counter() - s
        for r, frames, audio, run in held:
            rec = summary['runs'][r['run_id']]
            try:
                s = time.perf_counter()
                sr_pipe(video=frames, audio=audio, fps=fps, audio_sample_rate=audio_fps,
                        show_progress=True, save_path=str(run / 'native_sr.mp4'))
                torch.cuda.synchronize()
                rec['sr_seconds'] = time.perf_counter() - s
                rec['status'] = 'sr_ok'
            except Exception as e:  # noqa: BLE001
                rec['status'] = 'sr_failed'
                rec['sr_error'] = type(e).__name__ + ': ' + str(e)[:800]
                (run / 'sr-error.txt').write_text(traceback.format_exc())
            (batch_dir / 'batch-observation.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')

    summary['total_seconds'] = time.perf_counter() - t0
    (batch_dir / 'batch-observation.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    if any(v['status'] not in ('native_ok', 'sr_ok') for v in summary['runs'].values()):
        raise SystemExit(4)


if __name__ == '__main__':
    main()
