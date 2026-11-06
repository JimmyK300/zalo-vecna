"""Weight-space ledger: document WHERE and HOW a LoRA fine-tune changed Qwen3.5-9B.

Static (CPU, <1 GB): per-module update norms/spectra, layer/band/module-type energy, readout alignment of
residual-writing updates (down_proj, o_proj) with probe-token unembedding contrasts, and subspace overlap with
reference adapters. Writes <out>/ledger.json + LEDGER.md. Behavioural probes (edit-service margin battery,
band ablations, random controls) are listed in SPEC.md and recorded separately.

usage: python ledger.py --adapter DIR --name RUN_ID [--ref NAME=DIR ...] [--probe NAME=GOOD_ID,BAD_ID ...]
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

STOCK = Path('D:/astra/models/stock')   # overridden by --stock (local dir or HF repo id, e.g. Qwen/Qwen3.5-9B)
ROOT = Path('D:/astra/research/weight_ledger')   # overridden by --out-root
KEY = re.compile(r'layers\.(\d+)\.(self_attn|mlp|linear_attn)\.(\w+)\.lora_([AB])\.weight$')
FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31}  # Qwen3.5-9B: every 4th layer is full attention
BANDS = [(0, 7), (8, 15), (16, 23), (24, 31)]
DEFAULT_PROBES = {'cast_vs_r': (59606, 427)}


def load_adapter(directory):
    directory = Path(directory)
    cfg = json.loads((directory/'adapter_config.json').read_text(encoding='utf8'))
    scale = cfg['lora_alpha']/cfg['r']
    factors = defaultdict(dict)
    with safe_open(str(directory/'adapter_model.safetensors'), 'pt') as f:
        for k in f.keys():
            m = KEY.search(k)
            if m:
                layer, block, module, ab = m.groups()
                factors[(int(layer), block, module)][ab] = f.get_tensor(k).double().numpy()
    return cfg, scale, factors


def decompose(A, B, scale):
    """SVD of scale*B@A via QR (never materializes the full matrix)."""
    Qb, Rb = np.linalg.qr(B)
    Qa, Ra = np.linalg.qr(A.T)
    U, s, Vt = np.linalg.svd(scale*(Rb@Ra.T))
    return Qb@U, s, Qa@Vt.T  # left (out-dim), singular values, right (in-dim)


def _stock_file(name):
    """Path to a file of the base model: local directory, or the HF cache for a repo id (downloads only that file)."""
    if Path(str(STOCK)).exists():
        return Path(str(STOCK)) / name
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(str(STOCK), name))


def unembed_rows(ids):
    index = json.loads(_stock_file('model.safetensors.index.json').read_text())['weight_map']
    name = 'lm_head.weight' if 'lm_head.weight' in index else next(k for k in index if k.endswith('embed_tokens.weight'))
    with safe_open(str(_stock_file(index[name])), 'pt') as f:
        sl = f.get_slice(name)
        return {i: sl[i:i+1].double().numpy()[0] for i in ids}


def principal_overlap(U1, U2):
    """Mean squared cosine of principal angles between column spaces (0 = orthogonal, 1 = identical)."""
    s = np.linalg.svd(U1.T@U2, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)**2))


def band_of(layer):
    return next(f'{a}-{b}' for a, b in BANDS if a <= layer <= b)


def run(adapter, name, refs, probes):
    out = ROOT/name
    out.mkdir(parents=True, exist_ok=True)
    cfg, scale, factors = load_adapter(adapter)
    rows = unembed_rows(sorted({i for pair in probes.values() for i in pair}))
    contrasts = {}
    for pname, (good, bad) in probes.items():
        v = rows[good]-rows[bad]
        contrasts[pname] = v/np.linalg.norm(v)
    ref_svd = {}
    for rname, rdir in refs.items():
        _, rscale, rf = load_adapter(rdir)
        ref_svd[rname] = {k: decompose(v['A'], v['B'], rscale) for k, v in rf.items() if 'A' in v and 'B' in v}
    modules, energy = [], defaultdict(float)
    for key in sorted(factors):
        layer, block, module = key
        f = factors[key]
        if 'A' not in f or 'B' not in f:
            continue
        L, s, R = decompose(f['A'], f['B'], scale)
        e = float(np.sum(s**2))
        p = s**2/e if e > 0 else s
        row = dict(layer=layer, block=block, module=module, band=band_of(layer),
                   attn_type='full' if layer in FULL_ATTN else 'linear',
                   frob=math.sqrt(e), energy=e, top1_energy_frac=float(p[0]) if e else 0.0,
                   effective_rank=float(np.exp(-np.sum(p*np.log(p+1e-30)))) if e else 0.0,
                   singular_values=[float(x) for x in s])
        if L.shape[0] == len(next(iter(contrasts.values()))) and module in ('down_proj', 'o_proj', 'out_proj'):
            # Energy-weighted |cos| between residual-writing directions and each probe readout contrast.
            row['readout'] = {pn: dict(top1_cos=float(L[:, 0]@c), weighted_abs_cos=float(np.sum(p*np.abs(L.T@c))))
                              for pn, c in contrasts.items()}
        for rname, rs in ref_svd.items():
            if key in rs:
                row.setdefault('ref_overlap', {})[rname] = dict(left=principal_overlap(L, rs[key][0]),
                                                               right=principal_overlap(R, rs[key][2]))
        modules.append(row)
        for agg in (f'layer:{layer}', f'band:{row["band"]}', f'block:{block}', f'module:{module}', f'attn:{row["attn_type"]}'):
            energy[agg] += e
    total = sum(r['energy'] for r in modules)
    shares = {k: v/total for k, v in sorted(energy.items())}
    top = sorted(modules, key=lambda r: -r['energy'])[:15]
    ledger = dict(run=name, adapter=str(adapter), lora=dict(r=cfg['r'], alpha=cfg['lora_alpha'], targets=cfg.get('target_modules')),
                  total_frob=math.sqrt(total), energy_share=shares, probes={k: list(v) for k, v in probes.items()},
                  references=list(refs), top_modules=[(r['layer'], r['block'], r['module'], round(r['energy']/total, 4)) for r in top],
                  modules=modules)
    (out/'ledger.json').write_text(json.dumps(ledger, indent=1), encoding='utf8')
    md = [f'# Weight ledger: {name}', '', f'Adapter: `{adapter}` (r={cfg["r"]}, alpha={cfg["lora_alpha"]}); total update Frobenius {math.sqrt(total):.3f}.', '',
          '## Energy share (fraction of total squared update norm)', '', '| group | share |', '|---|---:|']
    md += [f'| {k} | {v:.3f} |' for k, v in shares.items() if not k.startswith('layer:')]
    md += ['', '| layer | share |', '|---|---:|'] + [f'| {k[6:]} | {v:.3f} |' for k, v in shares.items() if k.startswith('layer:')]
    md += ['', '## Top 15 modules by energy', '', '| layer | block | module | share | eff. rank | top-1 frac |', '|---|---|---|---:|---:|---:|']
    md += [f'| {r["layer"]} | {r["block"]} | {r["module"]} | {r["energy"]/total:.3f} | {r["effective_rank"]:.1f} | {r["top1_energy_frac"]:.2f} |' for r in top]
    ro = [r for r in modules if 'readout' in r]
    if ro:
        md += ['', '## Readout alignment (residual-writing modules, top 10 by |weighted cos| per probe)', '']
        for pn in probes:
            best = sorted(ro, key=lambda r: -r['readout'][pn]['weighted_abs_cos'])[:10]
            md += [f'**{pn}** (random-direction baseline ~ {1/math.sqrt(4096):.3f}):', '', '| layer | module | weighted abs cos | top-1 cos |', '|---|---|---:|---:|']
            md += [f'| {r["layer"]} | {r["module"]} | {r["readout"][pn]["weighted_abs_cos"]:.4f} | {r["readout"][pn]["top1_cos"]:+.4f} |' for r in best]
            md.append('')
    for rname in refs:
        ov = [r['ref_overlap'][rname] for r in modules if rname in r.get('ref_overlap', {})]
        if ov:
            md += [f'## Overlap with reference `{rname}`', '',
                   f'Mean principal-subspace overlap: left {np.mean([o["left"] for o in ov]):.3f}, right {np.mean([o["right"] for o in ov]):.3f} '
                   f'over {len(ov)} shared modules (random r=16 subspaces in 4096-d ~ 0.004).', '']
    (out/'LEDGER.md').write_text('\n'.join(md), encoding='utf8')
    print(json.dumps(dict(out=str(out), total_frob=math.sqrt(total), top=ledger['top_modules'][:5])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--ref', action='append', default=[], help='NAME=DIR')
    ap.add_argument('--probe', action='append', default=[], help='NAME=GOOD_ID,BAD_ID')
    ap.add_argument('--stock', default=None, help='base model: local dir or HF repo id')
    ap.add_argument('--out-root', default=None, help='where <name>/LEDGER.md is written')
    a = ap.parse_args()
    if a.stock:
        STOCK = a.stock
    if a.out_root:
        ROOT = Path(a.out_root)
    probes = dict(DEFAULT_PROBES)
    for p in a.probe:
        n, ids = p.split('=')
        probes[n] = tuple(int(x) for x in ids.split(','))
    run(a.adapter, a.name, dict(r.split('=', 1) for r in a.ref), probes)
