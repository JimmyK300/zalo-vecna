"""Paired H3b evaluation against a vLLM server using RAW prompts (exact Qwen chat template, thinking on/off).

The prompt is rendered locally with the model's own tokenizer chat template (tools=h3.TOOLS_B), sent to
/v1/completions, and the raw generation (reasoning + '</think>' + tool call) is handed to h3.solve unchanged, so
every arm sees byte-identical prompts for identical histories. No answers/gold are needed here; grade afterwards.

  python -m harness3.raw_eval --pack /content/pack --model Qwen/Qwen3.5-9B --tag stock --out /content/out \
        --base-url http://127.0.0.1:8000/v1 --thinking --max-tokens 8192 --ctx 32768 --workers 28
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from . import h3

LOCK = threading.Lock()


class ContextExhausted(RuntimeError):
    """Name matters: h3.solve converts a later-turn ContextExhausted into a counted outcome."""


class RawClient:
    def __init__(self, base_url, model, tokenizer, thinking, max_tokens, ctx):
        self.url, self.model, self.tok = base_url.rstrip('/') + '/completions', model, tokenizer
        self.thinking, self.max_tokens, self.ctx = thinking, max_tokens, ctx

    def render(self, messages, tools):
        return self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.thinking)

    def __call__(self, messages, tools):
        prompt = self.render(messages, tools)
        n = len(self.tok(prompt, add_special_tokens=False)['input_ids'])
        room = self.ctx - n
        if room < 256:
            raise ContextExhausted(f'prompt {n} tokens, context {self.ctx}')
        body = dict(model=self.model, prompt=prompt, max_tokens=min(self.max_tokens, room), temperature=0, top_p=1,
                    seed=47, stop=['<|im_end|>', '<|endoftext|>'], skip_special_tokens=False)
        err = ''
        for attempt in range(6):
            try:
                r = requests.post(self.url, json=body, timeout=1800)
                if r.ok:
                    break
                err = f'{r.status_code} {r.text[:300]}'
            except Exception as e:
                err = repr(e)
            time.sleep(min(60, 5 * 2 ** attempt))
        else:
            raise RuntimeError('vLLM failed: ' + err)
        c = r.json()['choices'][0]
        text = c['text']
        if self.thinking and '</think>' not in text:
            # Sol review: without a closing tag h3.visible() would parse calls inside unfinished reasoning.
            text = text + '\n</think>\n\n'
        cap = c.get('finish_reason') == 'length'
        u = r.json().get('usage', {})
        return dict(text=text, finish_reason=c.get('finish_reason'), incomplete=cap, output_cap=cap,
                    prompt_tokens=u.get('prompt_tokens', n), completion_tokens=u.get('completion_tokens'))


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf8').splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True, help='questions.jsonl, rulebook.md, cache/<biz>.duckdb + .docs.txt')
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    ap.add_argument('--thinking', action='store_true')
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--variant', default='H3b')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    pack, out = Path(a.pack), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f'{a.tag}.jsonl'
    done = {r['id'] for r in rows(dest)} if dest.exists() else set()
    qs = [q for q in rows(pack / 'questions.jsonl') if q['id'] not in done]
    rulebook = (pack / 'rulebook.md').read_text(encoding='utf8')
    client = RawClient(a.base_url, a.model, tok, a.thinking, a.max_tokens, a.ctx)
    meta = dict(tag=a.tag, model=a.model, thinking=a.thinking, max_tokens=a.max_tokens, ctx=a.ctx, variant=a.variant,
                temperature=0, seed=47)
    (out / f'{a.tag}.meta.json').write_text(json.dumps(meta, indent=1))

    def one(q):
        db = pack / 'cache' / f"{q['business_id']}.duckdb"
        docs = (pack / 'cache' / f"{q['business_id']}.docs.txt").read_text(encoding='utf8')
        for attempt in range(3):
            ex = h3.SQL3(db)
            try:
                t0 = time.time()
                rec = h3.solve(q, h3.messages_for(q, db, docs, rulebook, a.variant), client, ex, a.variant)
                rec['seconds'] = round(time.time() - t0, 1)
                return rec
            except h3.TechnicalFailure as e:
                last = repr(e)
            except ContextExhausted as e:   # first-turn overflow: counted outcome, same policy for every arm
                return dict(id=q['id'], prediction=dict(id=q['id'], type='abstain', answer=None),
                            stop='context_exhausted_first_turn', trace=[], counts={}, error=repr(e))
            finally:
                ex.close()
        return dict(id=q['id'], prediction=dict(id=q['id'], type='abstain', answer=None), stop='technical->abstain',
                    trace=[], counts={}, error=last)

    print(f'{a.tag}: {len(qs)} questions ({len(done)} done)', flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(one, q): q for q in qs}
        for i, f in enumerate(as_completed(futs), 1):
            q = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                rec = dict(id=q['id'], prediction=dict(id=q['id'], type='abstain', answer=None), stop='error->abstain',
                           trace=[], counts={}, error=repr(e)[:400])
            rec.update(tag=a.tag, model=a.model, business_id=q['business_id'])
            rec.pop('messages', None)
            with LOCK, dest.open('a', encoding='utf8') as s:
                s.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            print(f"[{i}/{len(qs)}] {q['id']} {rec['stop']} {rec.get('seconds', '')}s", flush=True)


if __name__ == '__main__':
    main()
