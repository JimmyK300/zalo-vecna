# Backlog

Ordered by expected gain per effort. Each item says what evidence exists today and how to test it.
Every change is compared against the current pipeline on the same questions (held-out 300, the
177 recipe-free held-out questions, the balanced 150), plus the altered-rulebook check
(`scripts/check_rule_changes.py`, must stay at 0 stale answers).

**Measurement rule learned on 2026-11-02:** two single-attempt runs of the same setup on the 177
recipe-free questions differed by 0.12 PA. A change worth less than ~0.10 needs at least 3 paired
runs per arm (or a larger sample) before it counts.

## Done (2026-11-02) — the ideas from the Valentine / Wren / CHESS / Vanna review

| Idea | What was built | Evidence |
|---|---|---|
| **Wren-style semantic model** | `datamate/semantic_model.py`: per shop, concepts (orders, payments, customers, items, sale lines, debt) mapped to source columns with evidence (README wording vs column name) and confidence; **unknowns** listed explicitly; data facts kept apart from rulebook decisions. Inspect: `python scripts/dump_semantic.py devkit/train devkit/public_test` → `out/semantic/` | A/B on 177 recipe-free held-out questions: 0.40→0.54, replicate 0.52→0.48. **Not proven** (pooled +0.05, noise 0.12). Kept on: no sign of harm, and it surfaces real traps (biz_106: customer names only 44% match the customer list, codes 100%; unrecognised "Sổ nợ" debt books, price-history sheets) |
| **Valentine-style join evidence** | Joins scored by key uniqueness, coverage both ways, cardinality, row fan-out, number of shared values; same-layout tables (monthly files) pooled as one family; N:M overlap is not a key | Public shops: "probable" joins 645 → 19 after family pooling and the N:M rule. No separate A/B (it ships inside the semantic model) |
| **CHESS-style selective verification** | `--verify`: when the two attempts disagree, a third attempt sees both answers and their last SQL/Python, must re-check with a query and match one of them, else abstain | Held-out 300: 0.737 → **0.817**. Answered 52 of 59 disagreements, 38 right / 14 wrong. When one candidate was right it picked it 38/42. Its errors are almost all "both candidates wrong" |
| Rule-change safety | Recipes now stay silent when a rule they rely on is missing: week start, same-day month comparison, month without year, walk-in, debt definition | `scripts/check_rule_changes.py`: dropping each of 12 rules → **0 stale** recipe answers across 3,000 train questions. Recipe accuracy unchanged (1,246 / 13) |

## Next

1. ~~Verify when both candidates are wrong~~ — tried 2026-11-02 (`--verify-new`: the verifier may
   give its own answer, accepted only if a 4th attempt agrees). On the 59 disagreement questions it
   fired twice, both rejected: no effect, so the flag stays off. Even when told it may, the verifier
   almost always picks A or B, and on this rerun its picks were only ~52% right (12/22, 13/25) vs
   90% last night when one candidate was right. Prompting the same 9B model again does not find the
   missing answer; the fix has to come from better first attempts (recipes, filtered metrics) or a
   stronger model, not more verification rounds.
2. **Filtered metrics** (size, item group, channel, payment method, customer group) on the existing
   metric engine: about a third of the remaining model-side errors. Must pass the rule-change check.
3. **Prove or drop the semantic model**: 3 paired runs per arm on the 177 recipe-free questions
   (~$2), or a per-question analysis on join-heavy questions (customer, payment, debt) where it
   should matter most.
4. **Vanna-style verified examples** (reviewed as risky): only store examples whose answer was
   checked against gold *and* that generalise across shops (keyed by concept, not by column name).
   A query that merely ran is not evidence. Low priority until 1–3 are done.
5. **Unseen debt layouts** ("Sổ nợ" books in 3 public shops, 19 questions): the semantic model
   already lists them as unknowns; a ledger reader for Nợ / Đã trả columns.
6. **Import / price-on-date / stock-as-of recipes** (same ledger pattern).
7. **Distillation** of good traces into Qwen3.5-9B for recipe-free questions; the
   Qwen3.5-9B-Agentic-GRPO results below say a generic SQL fine-tune is not a substitute.
8. **Docker / vLLM image** with the final model and settings; reproduce twice within 0.03 PA.

## Models tested and rejected

- **Qwen3-14B** (OpenRouter, thinking off): held-out 0.49 vs 0.69 for Qwen3.5-9B under the same
  v7 pipeline, balanced 0.67 vs 0.87, and twice the cost. Its 40k context also overflowed once.
- **Qwen3.5-9B-Agentic-GRPO** (local, Q4_K_M, converted with llama.cpp; tool calls work through
  Ollama's stock Qwen3.5 parser, no adapter needed). One attempt on 60 recipe-free questions: 0.42
  vs 0.28 for stock Qwen3.5-9B, but equal accuracy (32/47) where both answered; its edge was not
  looping and cleaner SQL. In the full v8 pipeline it looped on 36 of 40 model questions before the
  PC crashed (71/300 done, 0.45 vs 0.86 for stock v8 on the same questions). Not a replacement.
  Untested: thinking mode (it was trained with chain of thought), bf16/vLLM serving.
