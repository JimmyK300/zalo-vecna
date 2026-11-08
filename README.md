# DataMate agent — first working version (Zalo AI Challenge 2026)

Answers Vietnamese questions over messy business data (CSV / Excel / SQLite / docs).
The LLM decides **how** to compute; DuckDB / pandas do the computing.

**FT2 research update (2026-11-08):** [overnight results, checkpoint selection,
weight-analysis preview and CPU limitations](docs/ft2/README.md).

**Colab submission (GitHub + Drive, no new fine-tuning):** open
[`notebooks/DataMate_Submission_Colab.ipynb`](notebooks/DataMate_Submission_Colab.ipynb)
and follow [`SUBMISSION_GUIDE_VI.md`](notebooks/SUBMISSION_GUIDE_VI.md).
This branch adds local Transformers inference, conservative auxiliary question expansion,
and stricter numeric scoring. Historical PA below used the older scorer; GPU quality/latency
of the new backend has not been benchmarked.

```
question + business folder
  -> workspace.py   load every file into DataFrames, detect messy Excel headers,
                    build a schema profile (+ README / rulebook text)
  -> agent.py       tool loop: run_sql (DuckDB) / run_python (pandas) / final_answer
                    evidence check: an answer that never appeared in a tool output is
                    sent back once for recomputation
                    confidence < threshold -> abstain
  -> io_contract.py submission.jsonl, written line by line (survives a timeout), resumable
```

## Status

| Piece | State |
|---|---|
| Loader, profiler, tools, agent loop, runner, logs | done, tested end to end with the offline `scripted` backend |
| Codex CLI backend (`codex exec`, ChatGPT login) | **working**: 14/14 on the synthetic sample set, 1.1 min with 4 workers |
| Claude backend (`claude-haiku-5-5`) | written, **not run yet** (no API key on this machine) |
| OpenAI-compatible backend (GPT or local vLLM) | written, not run yet |
| Input / output format | **official** (devkit in `devkit/`): `{id, business_id, question, as_of_date}` → `{id, type, answer}` |
| Scoring | re-implementation of rulebook §9 in `datamate/evaluate.py` (no official scorer is shipped) |
| Rulebook | read from `<data>/rulebook.md` at run time and put in the system prompt. **It differs between train and public test** (public adds industries and rules), so never hard-code it |
| Data cleaning | monthly sheets merged (`_sheet` column), text numbers like `"84,000"` converted, mixed date columns parsed into `<col>__date` |

Devkit: `devkit/train` (30 businesses, 3000 questions + `answers.jsonl`) and `devkit/public_test`
(14 businesses, 1000 questions, no answers). Run on real data:
```bash
.venv/Scripts/python -m datamate.run --data devkit/train --out out/x/submission.jsonl --backend codex --sample 30 --fresh --workers 6
.venv/Scripts/python -m datamate.evaluate --pred out/x/submission.jsonl --gold devkit/train/answers.jsonl -v
```
`--sample N` picks a seeded sample spread over businesses; `--businesses a,b` filters.
`--stratify-gold devkit/train/answers.jsonl` (with `--sample`) balances the sample by answer type; gold only picks ids.
`--consistency 2` re-solves each answered question once more (sampled) and abstains when the two disagree.
A rerun without `--fresh` resumes and retries abstains; a quota / rate-limit error stops the run instead of burning questions.

Results (train). **Cost note:** the $ figures in the OpenRouter rows are list price × tokens and are
~8× too high. OpenRouter routes to cheaper providers and bills cached prompt prefixes at a discount; the
real bill for all OpenRouter runs on 2026-11-01 was **$1.45** (Qwen3.5-9B $1.01, Ministral 14B $0.44).
Runs now log the real charge (`tokens.cost_usd` per question, total in the "done" log line):
about $0.001 per question for Qwen3.5-9B without thinking. Before 2026-11-02 the logged total left
out the second attempt of `--consistency 2`, so v4–v7 costs are undercounted (v8 logs every attempt).

| run | n | PA | correct / wrong / abstain | notes |
|---|---|---|---|---|
| `--sample 30 --seed 1` | 30 | 1.00 | 30 / 0 / 0 | only number, percent, list, date |
| `--sample 150 --seed 2 --stratify-gold` | 150 | 0.96 | 147 / 3 / 0 | 141 with gpt-6.1-terra (139 right; 2 counts off by one, biz_105, biz_118), 9 resumed with `--model gpt-6-luna` after a quota stop (8 right; Zyrtec count 251 vs 389, units of "hộp" lines) |
| same 150, `--model gpt-6-luna` | 150 | 0.67 | 125 / 25 / 0 | 10.4 min. Lists 8/18, strings 13/18. Errors: wrong answer type (3), walk-in "kl" counted as a customer (2), top-N order / members (~7), other calculations; never abstains (confidence always ≥ 0.9) |
| same 150, `--backend openrouter --model qwen/qwen3.5-9b`, `DATAMATE_REASONING=on` | 150 | 0.26 | 72 / 33 / 45 | 63 min, $4.66 (19.1M in / 5.7M out tokens). All 45 abstains = ran out of 16 steps |
| same, `DATAMATE_REASONING=off` | 150 | 0.22 | 78 / 45 / 27 | 20 min, $2.87. 21% of tool calls error (mostly DuckDB syntax on Vietnamese column names) |
| same, `--model mistralai/ministral-14b-2512` | 150 | -0.12 | 61 / 79 / 10 | 31 min, $2.15. 42% of tool calls error; tables 0/17; worse than abstaining on everything |
| Qwen3.5-9B no-think **+ helpers** (ASCII columns, README roles, order_status / in_revenue / walk-in / dup-line columns, SQL hints, rank(), step budget) | 150 | 0.36 | 98 / 44 / 8 | 12 min, $2.08; tool-error rate 20% → 12%; abstains (out of steps) 27 → 8 |
| same answers re-scored with the new string→date coercion (offline, no new calls) | 150 | 0.43 | 103 / 39 / 8 | remaining big miss: debt tables (4/13), percent changes |
| same setup rerun live with date coercion + table prompt rule (v3) | 150 | 0.37 | 101 / 46 / 3 | 16 min, **real cost $0.21**. Dates 10→16 right, but tables 4→3 and percent 10→8; 11 questions flipped right and 8 flipped wrong vs v2 (provider-side nondeterminism, ±0.05 PA noise) |
| **v4: three layers** (rule guard, primitives + `orders` table, relationship graph + plan card), first attempt only | 150 | 0.37 | 98 / 42 / 10 | numbers 17→19, percent 8→11, booleans 18/18; tables still 1/12; model used `orders` in 133/150 questions but `as_of_balance` only twice |
| v4 + `--consistency 2` (second sampled attempt, abstain on disagreement) | 150 | **0.53** | 87 / 8 / 55 | 21 min, real cost $0.42. 45 disagreements: 34 of them were wrong first answers, 11 right. Accuracy on answered questions 92% |
| **v5: + debt recipe** (ledger + ready call on the plan card + recipe candidate), first attempt only | 150 | **0.67** | 122 / 22 / 6 | tables 1→18 of 18; debt questions 20/22; model called `debt.*` in 21 questions. Same level as GPT-6 Luna (0.67) |
| v5 + `--consistency 2` + recipe check | 150 | 0.68 | 109 / 7 / 34 | 13.5 min, real cost $0.18. Consistency now abstains 13 right vs 15 wrong (barely positive); the recipe check never fired (model always agreed) |
| v6: + order-metric engine (ready call + candidate check) | 150 | 0.73 | 112 / 2 / 36 | first attempt 0.60; $0.36 |
| **held-out natural 300** (seed 3, tuning ids excluded), v6 | 300 | 0.56 | 183 / 16 / 101 | first attempt 0.51; $0.69 |
| **v7: recipes answer their questions directly** (`--candidate-policy answer`), held-out natural 300 | 300 | **0.69** | 215 / 7 / 78 | recipes answered 123 (122 right); model 177 (93 / 6 / 78); $0.60 |
| v7, balanced 150 | 150 | **0.87** | 132 / 2 / 16 | recipes answered 97 (96 right); $0.10 |
| v7 setup, `--model qwen/qwen3-14b` (thinking off), held-out 300 | 300 | 0.49 | 186 / 39 / 75 | $1.58; worse than Qwen3.5-9B and pricier; 1 question over its 40k context |
| same, balanced 150 | 150 | 0.67 | 115 / 15 / 20 | stopped after the last question hung 40 min (counted as abstain) |
| semantic model A/B, model-only 177 held-out (no recipe), one attempt | 177 | off 0.40 / on 0.54 | 121/50/6 vs 132/36/9 | replicate: off 0.52 / on 0.48. Same-arm runs differ by 0.12: **effect not proven** (pooled +0.05) |
| **v8: + semantic model + `--verify`** (third attempt adjudicates disagreements), held-out 300 | 300 | **0.82** | 265 / 20 / 15 | $1.23 (now includes every attempt). Verify answered 52 of 59 disagreements: 38 right / 14 wrong. Without verify the same run is 0.737 |
| v8, balanced 150 | 150 | **0.89** | 137 / 4 / 9 | $0.35 |
| the 59 questions that disagreed in v8, rerun: `--verify` vs `--verify --verify-new` | 59 | 0.39 vs 0.36 | 38/15/6 vs 37/16/6 | $0.28 + $0.30. Only 22–25 disagreed this time (the rest agreed). The new-answer path fired twice and both were rejected by the 4th attempt: **no effect**, flag left off. The old arm reproduces v8 on these 59 (38/14/7 then, 38/15/6 now) |
| local Q4 (Ollama), 60 recipe-free held-out, one attempt: Qwen3.5-9B-Agentic-GRPO vs stock Qwen3.5-9B | 60 | 0.42 vs 0.28 | 41/16/3 vs 32/15/13 | tie (32 right each) on the 47 both answered; stock lost 11 to repetition loops ("token repeat limit"). Fine-tune: 2 SQL errors vs 22, median 35 s vs 50 s |
| v8 pipeline with the fine-tune, local, held-out 300 | 71 done | 0.45 (stock v8 on same 71: 0.86) | 33 / 1 / 37 | **PC crashed at 03:52** (sensors normal: hotspot 80°C). 36 of the 40 model questions aborted on repetition loops under the sampled 2nd attempt / verify prompts |
| Docker image | skeleton only (`Dockerfile`, `docker/entrypoint.sh`) |

## Run

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/make_sample_data.py      # 14 synthetic questions, 2 businesses
```

Codex CLI (uses your `codex login`; model from `~/.codex/config.toml` unless `--model` is given):
```bash
.venv/Scripts/python -m datamate.run --data sample_data/data --out out/codex/submission.jsonl --backend codex --fresh --workers 4
```
Each step is one stateless `codex exec` call (read-only sandbox, `--output-schema` forces one JSON action);
the host runs the action and feeds the result back. `DATAMATE_EFFORT` sets Codex reasoning effort (default `medium`).

Claude (proof of concept):
```bash
export ANTHROPIC_API_KEY=...
.venv/Scripts/python -m datamate.run --data sample_data/data --out out/haiku/submission.jsonl --backend anthropic --fresh
.venv/Scripts/python -m datamate.evaluate --pred out/haiku/submission.jsonl --gold sample_data/gold.jsonl -v
```

OpenAI / local vLLM:
```bash
export OPENAI_API_KEY=...  OPENAI_BASE_URL=...   # base URL only for vLLM
.venv/Scripts/python -m datamate.run --data sample_data/data --out out/gpt/submission.jsonl --backend openai --model <model-id> --fresh
```
Hosted reasoning models may reject `temperature`; set `DATAMATE_TEMPERATURE=` (empty) to omit it.

Offline smoke test (no LLM): `--backend scripted`.

Per-question traces (every tool call and output) go to `<out dir>/logs/<id>.json`, plus `run.log`.

## Settings (env var / flag)

| | default |
|---|---|
| `DATAMATE_BACKEND` / `--backend` | `anthropic` |
| `DATAMATE_MODEL` / `--model` | `claude-haiku-5-5` / `gpt-6-luna` |
| `DATAMATE_MIN_CONF` / `--min-confidence` | 0.5 |
| `DATAMATE_WORKERS` / `--workers` | 8 |
| `DATAMATE_TIME_BUDGET_MIN` | 330 (rule 7: 6 h including model load) |
| `DATAMATE_EFFORT` (Claude) | `medium` |

## Before the final submission (rules)

- Rule 2: ≤ 15B parameters in total across all models → e.g. one ~14B instruct model with tool calling, served by vLLM.
- Rule 3: no network → weights baked into the image; the Claude/GPT backends are for development only.
- Rule 8: re-run must agree within 0.03 PA → temperature 0, fixed seed, run the image twice locally and compare.

## Next steps

1. Get the devkit: fix `io_contract.py` (question fields, dataset mapping, output schema) and replace `evaluate.py` with the official scorer.
2. Run Haiku on the official train/public split to get a baseline and failure list.
3. Swap in an open-weight model ≤ 15B via vLLM and compare against the Haiku baseline.
4. Tune the abstain threshold against the real penalty; add an experiment log and a held-out split grouped by business.
