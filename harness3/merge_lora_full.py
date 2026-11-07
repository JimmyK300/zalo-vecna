"""Merge a PEFT LoRA adapter (trained on the text model, keys base_model.model.model.layers.N.<block>.<module>) into a
FULL Qwen3.5-9B checkpoint (keys model.language_model.layers.N.<block>.<module>.weight), keeping the original
multimodal architecture/config so vLLM loads it exactly like the base. W' = W + (alpha/r) * B @ A, computed in fp32.

  python -m harness3.merge_lora_full --base BASE_DIR --adapter s128.safetensors --config adapter_config.json --out OUT [--scale 1.0]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

KEY = re.compile(r'layers\.(\d+)\.(self_attn|mlp|linear_attn)\.(\w+)\.lora_([AB])\.weight$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=1.0, help='dose multiplier on top of alpha/r (1.0 = as trained)')
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    s = cfg['lora_alpha'] / cfg['r'] * a.scale
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lora = {}
    with safe_open(a.adapter, 'pt') as f:
        for k in f.keys():
            m = KEY.search(k)
            if m:
                L, b, mod, ab = m.groups()
                lora.setdefault(f'model.language_model.layers.{L}.{b}.{mod}.weight', {})[ab] = f.get_tensor(k)
    idx = json.loads((base / 'model.safetensors.index.json').read_text())
    shards = sorted(set(idx['weight_map'].values()))
    used, norm2 = 0, 0.0
    for sh in shards:
        tensors = {}
        with safe_open(str(base / sh), 'pt') as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in lora:
                    A, B = lora[k]['A'].to(dev, torch.float32), lora[k]['B'].to(dev, torch.float32)
                    d = s * (B @ A)
                    assert d.shape == t.shape, (k, d.shape, t.shape)
                    t = (t.to(dev, torch.float32) + d).to(t.dtype).cpu()
                    used += 1
                    norm2 += float(d.norm()) ** 2
                tensors[k] = t.contiguous()
        save_file(tensors, str(out / sh), metadata=meta or {'format': 'pt'})
        print('wrote', sh, flush=True)
    assert used == len(lora), f'merged {used} of {len(lora)} LoRA modules'
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith('.safetensors'):
            shutil.copy(p, out / p.name)
    (out / 'merge_info.json').write_text(json.dumps(dict(adapter=a.adapter, modules=used, scale=s,
                                                         delta_frobenius=norm2 ** 0.5), indent=1))
    print(f'merged {used} modules, scale {s}, total delta Frobenius {norm2 ** 0.5:.3f}', flush=True)


if __name__ == '__main__':
    main()
