"""Run the H3/H3b harness over a DataMate data folder and write a submission file.

Current client: OpenAI-compatible server with NATIVE tool calling (vLLM, OpenRouter, or an Ollama tag whose template
renders tools). See README.md: a bare-template Ollama tag (TEMPLATE {{ .Prompt }}) silently drops the tools, so the
model never calls them. For Colab/Transformers, render with tokenizer.apply_chat_template(messages, tools=h3.TOOLS_B,
add_generation_prompt=True, enable_thinking=False) and pass the generated text to h3.solve (it parses Qwen XML calls).

  python -m harness3.run --data devkit/train --limit 50 --gold devkit/train/answers.jsonl      # quick local score
  python -m harness3.run --base-url https://openrouter.ai/api/v1 --model qwen/qwen3.5-9b \
        --api-key-env OPENROUTER_API_KEY --extra '{"reasoning":{"enabled":false}}' --workers 16 --data devkit/public_test
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import h3
from .workspace import build

LOCK = threading.Lock()


def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf8').splitlines() if l.strip()]


class Client:
    """OpenAI-compatible chat client with native tool calls; keeps the transcript in Qwen tool-call text form."""
    def __init__(self, base_url, model, api_key, extra, max_tokens):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key=api_key or 'none', timeout=600)
        self.model, self.extra, self.max_tokens = model, extra, max_tokens

    @staticmethod
    def convert(messages):
        out, pending = [], []
        for m in messages:
            if m['role'] == 'assistant':
                calls = h3.parse_calls(m['content'] or '')
                if calls:
                    name, args = calls[0]
                    cid = f'call_{len(out)}'
                    pending = [cid]
                    out.append(dict(role='assistant', content=None, tool_calls=[dict(id=cid, type='function',
                               function=dict(name=name, arguments=json.dumps(args, ensure_ascii=False)))]))
                else:
                    out.append(dict(role='assistant', content=m['content']))
            elif m['role'] == 'tool':
                out.append(dict(role='tool', tool_call_id=pending.pop(0) if pending else 'call_x', content=m['content']))
            else:
                out.append(dict(role=m['role'], content=m['content']))
        return out

    def __call__(self, messages, tools):
        err = ''
        for attempt in range(6):
            try:
                r = self.c.chat.completions.create(model=self.model, messages=self.convert(messages), tools=tools,
                                                   max_tokens=self.max_tokens, temperature=0, extra_body=self.extra)
                if r.choices:
                    break
            except Exception as e:
                err = repr(e)
            time.sleep(min(60, 5 * 2 ** attempt))
        else:
            raise RuntimeError('model server failed: ' + err[:300])
        msg = r.choices[0].message
        text = (msg.content or '').strip()
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or '{}')
            except ValueError:
                args = {'query': tc.function.arguments}
            args = {k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in args.items()}
            text = '<tool_call>\n<function=%s>\n' % tc.function.name + ''.join(
                '<parameter=%s>\n%s\n</parameter>\n' % (k, v) for k, v in args.items()) + '</function>\n</tool_call>'
        cap = r.choices[0].finish_reason == 'length'
        return dict(text=text, finish_reason=r.choices[0].finish_reason, incomplete=cap, output_cap=cap)


def solve_one(q, data_dir, cache, rulebook, client, variant):
    db, docs = build(data_dir, q['business_id'], cache)
    ex = h3.SQL3(db)
    try:
        for attempt in range(3):
            try:
                return h3.solve(q, h3.messages_for(q, db, docs, rulebook, variant), client, ex, variant)
            except h3.TechnicalFailure:
                continue
        return dict(id=q['id'], prediction=dict(id=q['id'], type='abstain', answer=None), stop='technical->abstain', trace=[], counts={})
    finally:
        ex.close()


def to_submission(pred):
    from datamate.io_contract import format_record
    return format_record(pred['id'], pred['type'], pred.get('answer'), pred.get('reason'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder with questions.jsonl, rulebook.md, calendar.csv, data/')
    ap.add_argument('--out', default='submission.jsonl')
    ap.add_argument('--variant', choices=['H3', 'H3b'], default='H3b')
    ap.add_argument('--base-url', default='http://127.0.0.1:11434/v1')
    ap.add_argument('--model', default='qwen3.5:9b')
    ap.add_argument('--api-key-env', default='')
    ap.add_argument('--extra', default='{"think": false, "reasoning_effort": "none"}',
                    help='JSON extra_body; default disables thinking on Ollama')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--workers', type=int, default=1, help='parallel questions (keep 1-2 for a single local GPU)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cache', default='.h3_cache')
    ap.add_argument('--gold', default='', help='optional answers.jsonl to score the run')
    a = ap.parse_args()

    data_dir = Path(a.data)
    questions = rows(data_dir / 'questions.jsonl')
    if a.limit:
        questions = questions[:a.limit]
    rulebook = (data_dir / 'rulebook.md').read_text(encoding='utf8')
    client = Client(a.base_url, a.model, os.environ.get(a.api_key_env, '') if a.api_key_env else '', json.loads(a.extra), a.max_tokens)
    out, traces = Path(a.out), Path(a.out).with_suffix('.traces.jsonl')
    done = {r['id'] for r in rows(out)} if out.exists() else set()
    todo = [q for q in questions if q['id'] not in done]
    # Build workspaces serially first (one per business), then answer in parallel.
    for biz in sorted({q['business_id'] for q in todo}):
        build(data_dir, biz, a.cache)
    print(f'{len(todo)} questions to answer ({len(done)} already in {out})', flush=True)
    with ThreadPoolExecutor(max(1, a.workers)) as pool:
        futs = {pool.submit(solve_one, q, data_dir, a.cache, rulebook, client, a.variant): q for q in todo}
        for i, f in enumerate(as_completed(futs), 1):
            q = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                print(f'{q["id"]} ERROR {e!r}', file=sys.stderr, flush=True)
                rec = dict(id=q['id'], prediction=dict(id=q['id'], type='abstain', answer=None), stop='error->abstain', trace=[], counts={})
            with LOCK:
                with out.open('a', encoding='utf8') as s:
                    s.write(json.dumps(to_submission(rec['prediction']), ensure_ascii=False) + '\n')
                with traces.open('a', encoding='utf8') as s:
                    s.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            print(f'[{i}/{len(todo)}] {q["id"]} {rec["stop"]}', flush=True)
    if a.gold:
        from datamate.evaluate import score
        gold = {g['id']: g for g in rows(a.gold)}
        pred = {r['id']: r for r in rows(out)}
        res = score(pred, gold, only_pred=True)
        res.pop('rows', None)
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
