"""Evaluate base or fine-tuned Qwen3.5-9B in H3b with Transformers (Colab), on a question-id list, graded against gold.

  !python eval_h3b_hf.py --data devkit/train --ids eval80.json --adapter /content/drive/MyDrive/ft1_nothink/final --out ft1_eval.jsonl
  !python eval_h3b_hf.py --data devkit/train --ids eval80.json --out base_eval.jsonl            # base model, same harness

Prompts are rendered with tokenizer.apply_chat_template(messages, tools=h3.TOOLS_B, enable_thinking=...) exactly as in
our evaluations; tool calls are parsed from Qwen XML by harness3.h3. Reports PA as-is and with unanswerable->abstain.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness3 import h3  # noqa: E402
from harness3.workspace import build  # noqa: E402


class ContextExhausted(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ids', required=True, help='JSON file {"ids": [...]}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--adapter', default='')
    ap.add_argument('--thinking', action='store_true')
    ap.add_argument('--cache', default='.h3_cache')
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()

    def chat(messages, tools):
        prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=a.thinking)
        ids = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
        n = ids['input_ids'].shape[1]
        if n + 2048 > 32768:
            raise ContextExhausted('context')
        with torch.inference_mode():
            g = model.generate(**ids, max_new_tokens=2048, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = g[0, n:]
        text = tok.decode(new, skip_special_tokens=False).replace(tok.eos_token or '<|im_end|>', '').replace('<|im_end|>', '').strip()
        return dict(text=text, output_cap=len(new) >= 2048)

    data = Path(a.data)
    qs = {json.loads(l)['id']: json.loads(l) for l in open(data / 'questions.jsonl', encoding='utf8') if l.strip()}
    gold = {json.loads(l)['id']: json.loads(l) for l in open(data / 'answers.jsonl', encoding='utf8') if l.strip()}
    rulebook = (data / 'rulebook.md').read_text(encoding='utf8')
    ids = json.load(open(a.ids))['ids']
    from datamate.evaluate import score_one
    done = {json.loads(l)['id'] for l in open(a.out, encoding='utf8')} if Path(a.out).exists() else set()
    with open(a.out, 'a', encoding='utf8') as f:
        for k, qid in enumerate(ids, 1):
            if qid in done:
                continue
            q = qs[qid]
            db, docs = build(data, q['business_id'], a.cache)
            ex = h3.SQL3(db)
            try:
                rec = h3.solve(q, h3.messages_for(q, db, docs, rulebook, 'H3b'), chat, ex, 'H3b')
            except h3.TechnicalFailure:
                rec = dict(id=qid, prediction=dict(id=qid, type='abstain', answer=None), stop='technical', counts={})
            finally:
                ex.close()
            outcome = score_one(rec['prediction'], gold[qid])
            f.write(json.dumps(dict(id=qid, gold_type=gold[qid]['type'], prediction=rec['prediction'], stop=rec['stop'],
                                    outcome=outcome, counts=rec.get('counts', {})), ensure_ascii=False, default=str) + '\n')
            f.flush()
            print(f'[{k}/{len(ids)}] {qid} {outcome} {rec["stop"]}', flush=True)
    recs = [json.loads(l) for l in open(a.out, encoding='utf8')]
    A = [r for r in recs if r['gold_type'] != 'unanswerable']
    U = [r for r in recs if r['gold_type'] == 'unanswerable']

    def pa(x, policy=False):
        o = ['abstain' if policy and r['prediction'].get('type') == 'unanswerable' else r['outcome'] for r in x]
        return (o.count('correct') - o.count('wrong')) / max(len(o), 1)
    for pol in (False, True):
        print(('unanswerable->abstain ' if pol else 'as-is ') + json.dumps(dict(
            answerable_PA=round(pa(A, pol), 3), unanswerable_PA=round(pa(U, pol), 3),
            real_mix_PA=round(0.974 * pa(A, pol) + 0.026 * pa(U, pol), 3),
            answerable_correct=sum(r['outcome'] == 'correct' for r in A), n=len(recs))))


if __name__ == '__main__':
    main()
