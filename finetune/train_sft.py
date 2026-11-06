"""FT1: LoRA SFT of Qwen3.5-9B on H3b traces, with weight-ledger logging (Colab: A100 80GB or RTX PRO 6000 96GB).

Colab setup (one cell):
  !pip -q install "transformers>=5.2" peft accelerate datasets flash-linear-attention causal-conv1d
  # flash-linear-attention / causal-conv1d give Qwen3.5's 24 linear-attention layers fast kernels; without them
  # training silently falls back to a much slower path (check the first-step time in the log).

Run:
  !python train_sft.py --data sft_nothink.jsonl --out /content/drive/MyDrive/ft1_nothink --epochs 2

Writes to --out: adapter checkpoints (ckpt_step*/), final adapter (final/), train_log.jsonl (step loss, lr, time),
example_losses.jsonl (per-example loss each epoch = dataset cartography), grad_sketch_*.npy + grad_index.json
(per-example CountSketch of LoRA gradients at 2 checkpoints; attribution), run_config.json.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch


def assistant_mask_tokens(tok, messages, tools, thinking, max_len):
    """Token ids for the whole conversation and labels that keep ONLY assistant turns (incl. their end token)."""
    full = tok.apply_chat_template(messages, tools=tools, tokenize=True, enable_thinking=thinking)
    if hasattr(full, 'input_ids'):
        full = full['input_ids']
    labels = [-100] * len(full)
    for k, m in enumerate(messages):
        if m['role'] != 'assistant':
            continue
        pre = tok.apply_chat_template(messages[:k], tools=tools, tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        upto = tok.apply_chat_template(messages[:k + 1], tools=tools, tokenize=True, enable_thinking=thinking)
        pre = pre['input_ids'] if hasattr(pre, 'input_ids') else pre
        upto = upto['input_ids'] if hasattr(upto, 'input_ids') else upto
        for i in range(len(pre), min(len(upto), len(full))):
            labels[i] = full[i]
    if len(full) > max_len:
        return None
    return full, labels


class CountSketch:
    """Fixed random CountSketch of the flattened LoRA gradient into `dim` buckets (seeded => comparable across runs)."""
    def __init__(self, params, dim=4096, seed=20261106):
        g = torch.Generator().manual_seed(seed)
        self.params, self.dim = params, dim
        self.buckets, self.signs = [], []
        for p in params:
            n = p.numel()
            self.buckets.append(torch.randint(0, dim, (n,), generator=g).to(p.device))
            self.signs.append((torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(p.device, torch.float32))

    def __call__(self):
        out = torch.zeros(self.dim, device=self.params[0].device, dtype=torch.float32)
        for p, b, s in zip(self.params, self.buckets, self.signs):
            if p.grad is not None:
                out.index_add_(0, b, p.grad.detach().float().reshape(-1) * s)
        return out.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--thinking', action='store_true', help='use for sft_think.jsonl')
    ap.add_argument('--epochs', type=float, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--grad-accum', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=30000)
    ap.add_argument('--checkpoints', type=int, default=10)
    ap.add_argument('--sketch-examples', type=int, default=512, help='examples per gradient-sketch pass (0 = off)')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: use only N examples')
    ap.add_argument('--seed', type=int, default=20261106)
    ap.add_argument('--resume', action='store_true', help='continue from the latest ckpt_step* in --out')
    ap.add_argument('--four-bit', dest='four_bit', action='store_true', help='QLoRA (nf4) for 24 GB GPUs')
    ap.add_argument('--ckpt-over', dest='ckpt_over', type=int, default=0,
                    help='gradient checkpointing only for examples with >= N tokens (0 = always). 12000 suits 96 GB')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data, encoding='utf8') if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    data = []
    for r in rows:
        enc = assistant_mask_tokens(tok, r['messages'], r['tools'], a.thinking, a.max_len)
        if enc:
            data.append(dict(id=r['id'], ids=enc[0], labels=enc[1], gold_type=r.get('gold_type'), tags=r.get('tags') or {}))
    print(f'{len(data)} examples, {sum(len(d["ids"]) for d in data):,} tokens, '
          f'{sum(sum(l != -100 for l in d["labels"]) for d in data):,} trained (assistant) tokens', flush=True)

    load = dict(dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    if a.four_bit:  # QLoRA for 24 GB-class GPUs (L4); slower, slightly different numerics
        from transformers import BitsAndBytesConfig
        load['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **load)
    if a.four_bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM')
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    def example_loss(d):
        """Next-token loss computing logits ONLY at positions that predict assistant tokens (identical loss, far less memory:
        full-sequence logits over a 248k vocabulary would need ~20 GB at 22k tokens)."""
        ids = torch.tensor([d['ids']], device=model.device)
        lab = torch.tensor(d['labels'], device=model.device)
        keep = (lab[1:] != -100).nonzero(as_tuple=True)[0]          # position t predicts token t+1
        logits = model(input_ids=ids, logits_to_keep=keep).logits[0]  # [len(keep), vocab]
        return torch.nn.functional.cross_entropy(logits.float(), lab[keep + 1])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    total_micro = int(math.ceil(len(data) * a.epochs))
    total_steps = max(1, total_micro // a.grad_accum)
    warm = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    ckpt_every = max(1, total_steps // max(1, a.checkpoints))
    revision = getattr(model.config, '_commit_hash', None)
    json.dump(dict(vars(a), n_examples=len(data), total_steps=total_steps, base_revision=revision,
                   lora=dict(r=a.rank, alpha=a.alpha, targets='all-linear')),
              open(out / 'run_config.json', 'w'), indent=1)
    sketch = CountSketch(params, seed=a.seed) if a.sketch_examples else None

    def sketch_pass(tag):
        if not sketch:
            return
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        # Stratified by (answer type, metric tag) so every skill group gets gradient sketches; same ids at mid and final.
        groups = {}
        for j, d in enumerate(data):
            groups.setdefault((d['gold_type'], d['tags'].get('metric')), []).append(j)
        rng = random.Random(a.seed)
        for g in groups.values():
            rng.shuffle(g)
        idx, k = [], 0
        while len(idx) < min(a.sketch_examples, len(data)):
            for g in groups.values():
                if k < len(g) and len(idx) < a.sketch_examples:
                    idx.append(g[k])
            k += 1
        idx = sorted(idx)
        vecs = []
        for i in idx:
            model.zero_grad(set_to_none=True)
            example_loss(data[i]).backward()
            vecs.append(sketch())
        model.zero_grad(set_to_none=True)
        import numpy as np
        np.save(out / f'grad_sketch_{tag}.npy', np.stack(vecs).astype('float16'))
        json.dump([dict(id=data[i]['id'], gold_type=data[i]['gold_type'], tags=data[i]['tags']) for i in idx],
                  open(out / 'grad_index.json', 'w'), ensure_ascii=False)
        print(f'gradient sketch {tag}: {len(idx)} examples', flush=True)

    log = open(out / 'train_log.jsonl', 'a')
    exlog = open(out / 'example_losses.jsonl', 'a')
    step, micro, t0 = 0, 0, time.time()
    order = []
    epoch = 0
    if a.resume:
        # Continue after a disconnect: load the latest checkpoint adapter, fast-forward step/scheduler and the data order
        # (same seed => same shuffles). Optimizer moments restart (documented approximation).
        ckpts = sorted(out.glob('ckpt_step*'))
        if ckpts:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(ckpts[-1] / 'adapter_model.safetensors')))
            step = int(ckpts[-1].name.replace('ckpt_step', ''))
            micro = step * a.grad_accum
            for _ in range(step):
                sched.step()
            for _ in range(micro):  # replay the order generator without training
                if not order:
                    order = list(range(len(data)))
                    random.shuffle(order)
                    epoch += 1
                order.pop()
            print(f'resumed from {ckpts[-1].name} (step {step}, epoch {epoch})', flush=True)
    ckpt_on, oom_redo = [True], [0]
    model.train()
    while micro < total_micro:
        if not order:
            order = list(range(len(data)))
            random.shuffle(order)
            epoch += 1
        i = order.pop()
        d = data[i]
        # Gradient checkpointing only for long examples (same math, ~30% less recompute); OOM -> redo with checkpointing.
        want_ckpt = a.ckpt_over <= 0 or len(d['ids']) >= a.ckpt_over
        if want_ckpt != ckpt_on[0]:
            if want_ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            else:
                model.gradient_checkpointing_disable()
            ckpt_on[0] = want_ckpt
        # Without checkpointing an OOM may hit mid-backward after part of this example's gradient was accumulated:
        # stash the accumulated LoRA grads (~43M floats) so the redo starts from exactly the pre-example state.
        stash = None if ckpt_on[0] else [None if p.grad is None else p.grad.detach().clone() for p in params]
        oom = False
        try:
            loss = example_loss(d)
            (loss / a.grad_accum).backward()
        except torch.cuda.OutOfMemoryError:
            if ckpt_on[0]:
                raise
            oom = True  # redo OUTSIDE the except block, so the traceback no longer pins the failed activations
        if oom:
            loss = None
            for p, g in zip(params, stash):
                p.grad = g
            torch.cuda.empty_cache()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            ckpt_on[0] = True
            oom_redo[0] += 1
            loss = example_loss(d)
            (loss / a.grad_accum).backward()
        exlog.write(json.dumps(dict(id=d['id'], epoch=epoch, micro=micro, loss=float(loss.detach()), gold_type=d['gold_type'])) + '\n')
        micro += 1
        if micro % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps(dict(step=step, epoch=epoch, loss=float(loss.detach()), lr=sched.get_last_lr()[0], sec=time.time() - t0,
                                      oom_redo=oom_redo[0], gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))) + '\n')
            log.flush()
            exlog.flush()
            if step == 1:
                print(f'first optimizer step after {time.time() - t0:.0f}s (if this is very slow, the linear-attention kernels are missing)', flush=True)
            if step % ckpt_every == 0:
                model.save_pretrained(out / f'ckpt_step{step:05d}')
                print(f'step {step}/{total_steps} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s) '
                      f'peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB, oom-redo {oom_redo[0]}', flush=True)
            if step == max(1, total_steps // 2):
                sketch_pass('mid')
                ckpt_on[0] = True  # sketch_pass leaves checkpointing on
    model.save_pretrained(out / 'final')
    tok.save_pretrained(out / 'final')
    sketch_pass('final')
    print('done', flush=True)


if __name__ == '__main__':
    main()
