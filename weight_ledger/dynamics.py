"""Training dynamics of a LoRA run from its checkpoints (CPU, small RAM): how the update grows, where its energy goes,
and whether its directions stabilise. Part B of SPEC.md.

usage: python weight_ledger/dynamics.py --run /content/drive/MyDrive/ZaloAI_VECNA/ft1/run [--out DIR]
Writes DYNAMICS.md + dynamics.json into --out (default: the run directory).
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger as L  # noqa: E402


def summarize(directory):
    _, scale, factors = L.load_adapter(directory)
    mods, energy = {}, defaultdict(float)
    for key, f in factors.items():
        if 'A' not in f or 'B' not in f:
            continue
        left, s, right = L.decompose(f['A'], f['B'], scale)
        e = float(np.sum(s ** 2))
        mods[key] = (left, right, e)
        layer, block, module = key
        for g in (f'band:{L.band_of(layer)}', f'block:{block}', f'module:{module}',
                  f'attn:{"full" if layer in L.FULL_ATTN else "linear"}'):
            energy[g] += e
    total = sum(m[2] for m in mods.values())
    return mods, {k: v / total for k, v in sorted(energy.items())}, math.sqrt(total)


def overlap(a, b):
    common = sorted(set(a) & set(b))
    if not common:
        return None
    lo = np.mean([L.principal_overlap(a[k][0], b[k][0]) for k in common])
    ro = np.mean([L.principal_overlap(a[k][1], b[k][1]) for k in common])
    return dict(left=float(lo), right=float(ro), modules=len(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    run = Path(a.run)
    ckpts = sorted(run.glob('ckpt_step*'), key=lambda p: int(p.name.replace('ckpt_step', '')))
    if (run / 'final' / 'adapter_model.safetensors').exists():
        ckpts.append(run / 'final')
    rows, prev, summaries = [], None, []
    for c in ckpts:
        mods, share, frob = summarize(c)
        summaries.append((c.name, mods))
        row = dict(checkpoint=c.name, total_frob=frob, share=share,
                   overlap_prev=overlap(prev, mods) if prev else None)
        rows.append(row)
        prev = mods
        print(c.name, f'frob={frob:.3f}', flush=True)
    last = summaries[-1][1]
    for row, (_, mods) in zip(rows, summaries):
        row['overlap_last'] = overlap(mods, last)
    out = Path(a.out) if a.out else run
    out.mkdir(parents=True, exist_ok=True)
    (out / 'dynamics.json').write_text(json.dumps(rows, indent=1), encoding='utf8')
    groups = [g for g in rows[-1]['share'] if g.startswith(('band:', 'block:', 'attn:'))]
    md = ['# Training dynamics', '', f'Run: `{run}` — {len(rows)} checkpoints. Overlap = mean principal-subspace overlap of each '
          'module\'s LoRA update (1 = same directions, ~0.004 = random r=16 subspaces in 4096-d).', '',
          '| Checkpoint | Update Frobenius | ' + ' | '.join(groups) + ' | overlap with previous (left/right) | overlap with last (left/right) |',
          '|---|---:|' + '---:|' * len(groups) + '---|---|']
    for r in rows:
        op = r['overlap_prev']
        ol = r['overlap_last']
        md.append(f"| {r['checkpoint']} | {r['total_frob']:.3f} | " + ' | '.join(f"{r['share'].get(g, 0):.3f}" for g in groups)
                  + f" | {f'{op['left']:.2f}/{op['right']:.2f}' if op else '—'} | {f'{ol['left']:.2f}/{ol['right']:.2f}' if ol else '—'} |")
    md += ['', 'Read: if the Frobenius norm still rises steeply at the end, training had not converged; band/block shares show where '
           'the update concentrates; rising overlap with the last checkpoint shows the directions settling. Static only (DESCRIPTIVE); '
           'causal claims need the band/module ablations of SPEC.md part C.']
    (out / 'DYNAMICS.md').write_text('\n'.join(md), encoding='utf8')
    print('wrote', out / 'DYNAMICS.md')


if __name__ == '__main__':
    main()
