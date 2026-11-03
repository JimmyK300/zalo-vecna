# DataMate: findings and how we got here

Model under test: Qwen3.5-9B (open weights, ≤15B) on OpenRouter, thinking off, temperature 0.
Scores are PA (+1 right, −1 wrong, 0 abstain, divided by questions) on the train split, using the
re-implemented §9 scorer. "Held-out" = a natural-mix sample of 300 questions that no tuning ever saw.

## Headline

| Stage | Balanced 150 | Held-out natural 300 |
|---|---|---|
| Qwen3.5-9B, plain agent | 0.22 | – |
| + data helpers (ASCII names, roles, helper columns) | 0.36 | – |
| + 3 layers (rule guard, primitives, graph + plan card) + consistency | 0.53 | – |
| + debt recipe | 0.68 | – |
| + order-metric engine | 0.73 | 0.56 |
| **+ recipes answer their own questions (final)** | **0.87** | **0.69** |
| *GPT-6 Luna, plain agent (reference)* | *0.67* | – |
| *GPT-6.1 Terra, plain agent (reference)* | *0.96* | – |

Final held-out: 215 right, 7 wrong, 78 abstained. Recipes answered 123 of 300 questions with
122 right; the model answered the other 177 (93 right, 6 wrong, 78 abstained). Cost $0.60.

## Findings

### 1. The bottleneck was never the model's reasoning; it was assembling the computation
- In most wrong answers the correct value never appeared in any tool output: the model computed
  the wrong thing (raw status filters, wrong amount column, cost not × quantity, wrong week),
  not right numbers misread.
- Thinking mode added +0.04 for 1.6× cost and 11× latency. Not worth it at this stage.
- Giving the model primitives was not enough (`as_of_balance` was used 2× in 150 questions).
  Giving it a **ready-to-run call** filled with this shop's tables and the question's dates was
  (debt questions 1/12 → 18/18 tables).

### 2. Deterministic recipes are ~99% right where they apply, the model ~67–85%
- Validated on the whole train split (gold only tests code): 1,246 right / 13 wrong over 1,259
  questions (42% of train) for debt, order metrics and item questions.
- On the unseen held-out sample: 122 / 123.
- So the best policy is: **if a validated recipe recognises the question, answer from it**; use
  the model only for the rest. Veto-only (abstain on disagreement) left most of the gain unused.

### 3. "Candidates disagree → abstain" is the right uncertainty signal for a small model
- The model's self-reported confidence was useless (Ministral answered 79 wrong at high confidence).
- Two sampled attempts that disagree → abstain: +0.15 PA when the model was weak; on the final
  held-out it removed 41 wrong and 25 right answers among model-answered questions (net +16).

### 4. Most of the work was data normalisation that generalises (layer 1)
Each of these was found from a concrete miss and fixed generically, then checked on all 44 shops:
- Subtotal / total rows inside the data ("Cộng ngày …", "TỔNG NGÀY 01/10", "Tổng tháng", bare
  "Cộng"): they doubled or tripled revenue in 8 shops.
- Accent stripping is dangerous: the surname "Tống" became "tổng" and deleted real sales rows;
  totals are now matched on the original text with diacritics, as whole-cell labels.
- Money written by hand in mixed styles in one column: "38.000đ", "2tr300", "71,114tr", "3696k";
  a column 99% one style silently turned the odd "1tr245" into blanks.
- Detail books split by year across files; README file-name patterns with placeholders
  (`BanHang_T<tháng>_<năm>.xlsx`) and per-sheet column sections; books without order ids
  (rulebook §8: every row is a record), without a status column (status in the note), and with
  "các dòng cùng **STT** là một đơn" (order key = sheet + STT).
- Base-unit quantities from 333 README lines ("1 thùng = 24 lon") and conversion tables.
- Customer codes mapped to names; customer identity by code (rulebook §3) — repayments were
  missed when one code had several name spellings.
Coverage on the unseen public shops went from 6/14 to 14/14 shops building a complete
`orders` table (12/14 build `lines`, 4/14 the debt ledger).

### 5. Guarding against overfitting
- Business definitions live in the rulebook; every helper that encodes a rule names the rulebook
  wording it relies on (`rules.py`) and switches off if a new rulebook drops it (tested by editing
  a copy of the rulebook).
- Recipes only fire when the question has no condition they do not understand (shop vocabulary of
  item / group / customer names, filter words); otherwise the model handles it.
- Every recipe is validated on all 3,000 train questions, and the final number comes from a
  held-out sample excluded from all tuning.

### 6. Measurement lessons
- One 150-question run moves ±0.05 PA between identical runs (provider nondeterminism); small
  changes need repeated or larger runs.
- The type-balanced sample over-represents rare question types; the natural mix is ~38% revenue,
  14% order counts, 6% customers, 4% debt — stock is ~1%.
- List-price cost estimates were ~8× too high; real OpenRouter charges are logged per question.

## Where it stops generalising (honest limits)
- **The model-side 59%**: questions no recipe covers are answered ~67% right on the first attempt;
  the consistency check then abstains on 37% of them. This is now the biggest gap (78 abstains in 300).
- **Long-tail filters**: revenue / orders restricted to a size (M/L), an item group, a channel or
  payment method, a customer group, a shift, Tết; imports, prices on a date, stock as of a date.
  Each is a recipe-shaped extension of what exists (the ledger and metric engine take filters),
  but each covers 1–3% of questions and needs its own validation.
- **Unseen debt layouts**: 3 of 5 public shops with debt questions keep debt in their own books
  (a "Sổ nợ" with Nợ / Đã trả columns); the ledger does not recognise them yet (19 questions).
- **Not done**: a local GPU run (the user's RX 6900 XT overheated; now safe with power/clock caps
  and a 95 °C guard), the Docker image with vLLM, and fine-tuning / distillation.

## Next levers, by expected gain
1. Filtered metrics (size, item group, channel, payment method, customer group) on the existing
   engine — roughly a third of the remaining model-side abstains.
2. Import / price-on-date / stock-as-of recipes (same ledger pattern).
3. Distil good traces (Terra was 96% right) into Qwen3.5-9B for the questions recipes do not cover.
4. Recognise "Sổ nợ"-style debt books for the unseen public layouts.
