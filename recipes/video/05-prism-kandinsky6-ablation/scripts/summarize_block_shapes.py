"""prism_shape_hook이 남긴 npz를 요약합니다 (numpy만 필요, GPU 불필요).

사용: python scripts/summarize_block_shapes.py <dynamic-block-shapes.npz> [--out summary.json]
CFG를 쓰면 forward가 스텝마다 두 번(조건부, 무조건부) 호출됩니다. 짝수 forward를 조건부(양의) 분기로 봅니다.
"""
import argparse
import json

import numpy as np


def summarize(path):
    z = np.load(path)
    sids, fwd = z['shape_ids'], z['forward']
    shapes = [tuple(int(v) for v in s) for s in z['candidate_shapes']]
    zs = int(z['zone_size'])
    T, H, W = (int(v) for v in z['grid'][0])
    posi = fwd % 2 == 0
    steps = sorted(set((fwd[posi] // 2).tolist()))
    default = shapes.index((4, 4, 4))
    per_step = [round(float((sids[posi & (fwd // 2 == s)] == default).mean()), 4) for s in steps]
    out = {
        'grid_padded_THW_tokens': [T, H, W],
        'zones_THW': [T // zs, H // zs, W // zs],
        'forwards': int(fwd.max()) + 1,
        'layers_per_forward': int(z['layer'].max()) + 1,
        'calls': int(len(sids)),
        'shape_usage_fraction_all_calls': {'x'.join(map(str, shapes[i])): round(float((sids == i).mean()), 4)
                                           for i in range(len(shapes))},
        'default_444_fraction_per_step_positive_branch': per_step,
    }
    off, flat = z['sel_offsets'], z['sel_mean_flat']
    if len(flat):
        means = [float(flat[off[i]:off[i + 1]].mean()) for i in range(len(off) - 1) if off[i + 1] > off[i]]
        out['selected_kblocks_per_qblock_mean'] = round(float(np.mean(means)), 3)
        out['selected_kblocks_per_qblock_range'] = [round(min(means), 3), round(max(means), 3)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz')
    p.add_argument('--out')
    a = p.parse_args()
    s = summarize(a.npz)
    text = json.dumps(s, indent=1)
    if a.out:
        open(a.out, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
