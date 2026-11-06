# harness3 — minimal tool harness for DataMate (H3 / H3b)

A deliberately small "instruction set" for a 9B model, built from the failure analysis of stock Qwen3.5-9B in the
previous simple harness (H1). No recipes, no Python tool, no plan cards: one SQL tool plus structured endings.
**Status: experimental idea, not yet validated** (see Evidence). Adapt freely.

## Design
**Four tools**
- `sql(query)` — one read-only DuckDB query; `SELECT`/`WITH` plus `DESCRIBE`, `SHOW TABLES`, `SUMMARIZE` so the model
  can inspect the schema instead of guessing. Returns columns, up to 50 rows, total row count. Results over ~6,000
  chars are cut with a note to filter/aggregate.
- `answer(type, value)` — structured final answer, type is an enum; value parsed per type (number, percent 0-100,
  boolean, date `YYYY-MM[-DD]`, string, list = JSON array, table = `{"columns","rows"}`). A bad value gets an exact
  error and a retry instead of a lost question.
- `unanswerable(reason)` and `abstain()`.

**Prompt** (`SPEC` / `SPEC_B` in `h3.py`): scoring rule stated (wrong -1, abstain 0), one line per answer type,
"Query ONLY these tables" + typed columns + 3 sample rows per table (no source-file mapping layer), then the
rulebook and business docs unchanged, then the question JSON.

**Runtime**: `[calls left: N]` after every SQL result (budget 8); at 0 only an ending is allowed. Table/column errors
list the real tables/columns. A final answer written as prose + JSON is accepted if the JSON is valid. A run that
never ends validly abstains (0) instead of submitting junk (-1). Mid-episode context overflow ends with abstain.

**H3b** (default) = H3 + calibration of the endings, because H3's explicit `unanswerable` tool invited false
"unanswerable" on answerable questions: `unanswerable` is refused until at least one SQL query succeeded and must cite
the evidence; the prompt says almost every question is answerable, and abstain is described as "checked but cannot
pin down", not a safe default (an unnecessary abstain also loses a point). Unanswerable gold is only 70/3000 (2.3%)
of train, so false unanswerable/abstain on the other 97.7% dominates.

## Why (failure analysis that motivated it)
Stock Qwen3.5-9B in H1 (33-48% correct): ~20% of questions ended with no valid final (prose before JSON, nested
unanswerable object, tool call after force-final); hallucinated table names from the two naming layers; half the runs
exhausted the 8-call budget; wrong answer type (percent vs number, list vs table); 0/5 unanswerable detected.
DeepSeek V4.1 Flash gets 97% in the same H1 harness, with SHORTER SQL (median 60 tokens vs stock 106), fewer CTE/
subqueries and 3.8 turns vs 6-7: the gap is choosing the right prepared columns (`orders.in_revenue`,
`revenue_lines`), not capacity. So the harness should remove mechanical failure modes, not add tools.

## Evidence (stock Qwen3.5-9B via OpenRouter bf16, thinking off, greedy; 2026-11-06)
80 questions = 60 from 3 held-out validation businesses (all answerable) + 20 unanswerable from train businesses.
Same model/settings for all arms; paired.

| Arm | All 80: right/wrong/abstain, PA | Answerable 60: right/wrong/abstain, PA | False "unanswerable" | Unanswerable detected | SQL errors (60) |
|---|---|---|---:|---:|---:|
| H1 (previous simple harness) | 22/58/0, -0.45 | 22/38/0, -0.267 | 0 | 0/20 | 42 |
| H3 | 36/38/6, -0.025 | 20/36/4, -0.267 | 11 | 16/20 | 25 |
| **H3b** | 34/38/8, -0.05 | **22/32/6, -0.167** | **6** | 12/20 | **16** |

- Paired vs H1 on all 80: H3 23 vs 9 discordant (exact McNemar p=0.02), H3b 20 vs 8 (p=0.036); H3b vs H3 n.s.
- The 80-mix is 25% unanswerable vs 2.3% in train, which flatters H3. Reweighted to the real 97.7/2.3 mix:
  H1 ~ -0.28, H3 ~ -0.25, **H3b ~ -0.16 PA** (best: same accuracy as H1, fewer wrong via abstain, 60% fewer SQL errors).
- Remaining weakness: 6/60 false "unanswerable" on answerable questions (each -1). Candidate next step: allow
  `unanswerable` only when a query shows the requested period is outside the data range, or drop the tool.
- Stock is only ~37% accurate here, so PA stays negative: a calibrated abstention policy and fine-tuning matter more
  than the harness. The harness's job is removing mechanical failures, which it does.

## Update: H3c (negative) and the recommended submission policy
- H3c showed per-table date coverage and accepted `unanswerable` only for a period fully outside it (otherwise
  "REJECTED: query or abstain"). Result: false unanswerable 6 -> 0, but answerable correct 22 -> 14; the rejected
  model committed to wrong answers instead of abstaining. Real-mix PA -0.42 (worse than H1). Not recommended.
- **Recommended: H3b + submit any `unanswerable` as `abstain`.** At the real ~2.6% unanswerable rate (public reference
  26/1000), the model's unanswerable calls are right only ~14% of the time (EV -0.72 vs 0 for abstain). Re-scoring the
  same runs: real-mix PA H1 -0.29, H3b -0.155, **H3b + unanswerable->abstain -0.073** (chosen post hoc on these 80;
  principled by base rates, still to be confirmed on fresh questions).

## Running
`python -m harness3.run --data <folder with questions.jsonl, rulebook.md, calendar.csv, data/> --out submission.jsonl`
Workspaces are built once per business with the repository loader (`harness3/workspace.py`) into `.h3_cache/`.

**Important runtime caveat**: `run.py` currently uses an OpenAI-compatible client with native tool calling. An Ollama
tag created with a bare template (`TEMPLATE {{ .Prompt }}`) silently ignores `tools`, so the model never calls them
(observed: 0 SQL calls). Either use a server whose chat template renders tools (vLLM, OpenRouter, a properly
templated Ollama tag) or render the prompt yourself:
`tokenizer.apply_chat_template(messages, tools=h3.TOOLS_B, add_generation_prompt=True, enable_thinking=False)` and pass
the generated text to `h3.solve` as `{"text": ...}` (it parses Qwen XML `<tool_call><function=...>` calls). That is
how our own evaluations ran, and matches `datamate/hf_backend.py`'s Transformers path.
