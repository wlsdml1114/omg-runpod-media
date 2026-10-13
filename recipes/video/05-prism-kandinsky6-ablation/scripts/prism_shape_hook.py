"""Prism IVPQ 동적 블록 모양 기록 훅 (업스트림 파일을 수정하지 않는 래퍼).

dynamic_block_attention.flash_attn_bsa_3d_dynamic 호출마다
  * 영역(macro-zone)별 shape_ids
  * 쿼리 블록별 선택된 키 블록 수(헤드 평균)
를 메모리에 모았다가 npz로 저장합니다. 계산 결과에는 손대지 않습니다.

prism_run.py가 import 후 install(state)로 켭니다. state는 현재 forward 번호와 층 번호를 담는 dict입니다.
"""
from pathlib import Path

REC = {'enabled': False, 'path': None, 'calls': []}


def install(state):
    import torch
    import hymm.models.modules.block_sparse_attention.dynamic_block_attention as dba
    import hymm.models.modules.block_sparse_attention.bsa_interface as bsi

    acc = {'active': False, 'sum': None, 'heads': 0}

    def wrap_select(name):
        orig = getattr(bsi, name)

        def wrapped(*a, **k):
            res = orig(*a, **k)
            if acc['active'] and REC['enabled']:
                lens = res[1].detach().float().reshape(-1)  # 헤드 하나의 쿼리 블록별 선택 수
                acc['sum'] = lens if acc['sum'] is None else acc['sum'] + lens
                acc['heads'] += 1
            return res
        setattr(bsi, name, wrapped)

    for name in ['get_select_indices_cdf_topk_from_score', 'get_select_indices_topk_from_score',
                 'get_select_indices_cdf_from_score']:
        wrap_select(name)

    orig_dyn = dba.flash_attn_bsa_3d_dynamic

    def dyn(q, k, v, grid_padded, **kw):
        if not REC['enabled']:
            return orig_dyn(q, k, v, grid_padded, **kw)
        acc.update(active=True, sum=None, heads=0)
        try:
            out = orig_dyn(q, k, v, grid_padded, **kw)
        finally:
            acc['active'] = False
        sids = out[1] if isinstance(out, tuple) else kw.get('shape_ids')
        REC['calls'].append({
            'forward': state['fwd_idx'], 'layer': state['layer'], 'grid': tuple(grid_padded),
            'shape_ids': None if sids is None else sids.detach().to(torch.int8).cpu().clone(),
            'sel_mean': None if acc['sum'] is None else (acc['sum'] / max(acc['heads'], 1)).half().cpu(),
        })
        state['layer'] += 1
        return out

    dba.flash_attn_bsa_3d_dynamic = dyn


def start(path):
    REC.update(enabled=bool(path), path=path, calls=[])


def save():
    import numpy as np
    from hymm.models.modules.block_sparse_attention.dynamic_block_shape import ALL_CANDIDATE_SHAPES, ZONE_SIZE
    calls = REC['calls']
    REC['enabled'] = False
    if not calls or not REC['path']:
        return {'calls': 0}
    sids = np.stack([c['shape_ids'].numpy() for c in calls])
    sel = [c['sel_mean'].numpy() if c['sel_mean'] is not None else np.zeros(0, np.float16) for c in calls]
    Path(REC['path']).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REC['path'], shape_ids=sids,
                        forward=np.array([c['forward'] for c in calls]),
                        layer=np.array([c['layer'] for c in calls]),
                        grid=np.array([c['grid'] for c in calls]),
                        sel_mean_flat=np.concatenate(sel),
                        sel_offsets=np.cumsum([0] + [len(s) for s in sel]),
                        candidate_shapes=np.array(ALL_CANDIDATE_SHAPES), zone_size=ZONE_SIZE)
    summary = {'calls': len(calls), 'grid': list(calls[0]['grid']), 'zones': int(sids.shape[1])}
    REC['calls'] = []
    return summary
