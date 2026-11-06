# Weight-space ledger: protocol for every fine-tuning run

Goal: every training run leaves a durable, comparable record of HOW it changed the weights and WHAT each change does,
so Astra can form hypotheses from accumulated evidence instead of re-deriving it. One folder per run:
`D:/astra/research/weight_ledger/<NNN_run_id>/`. Index: `INDEX.md`.

## Lesson from entry 000 (why both halves are mandatory)
The agentic GRPO adapter spreads its update energy almost evenly across layer bands (0-7: 22%, 8-15: 25%, 16-23: 26%,
24-31: 27%; MLP 88%), yet its behavioural effect on the CAST decision is concentrated: layers 0-7 and MLP cause it,
8-15 opposes it (B2/LOCALIZATION_CONTROLS.md), and removing 0-7 + 16-23 repairs it on HELDOUT. Static readout alignment
only flags L31 down_proj (4.5x random) -- the causal early-layer effects act through downstream computation and are
invisible to direct readout. => Energy and readout are descriptive; causal claims need behavioural ablations + random controls.

## A. Static ledger (CPU, minutes) -- `ledger.py`
`python ledger.py --adapter <dir> --name <NNN_run> --ref agentic=D:/astra/adapters/checkpoint-1150/checkpoint-1150 [--ref prev=<dir>] [--probe name=good_id,bad_id]`
Records per module: update Frobenius norm, singular spectrum, effective rank, top-1 energy fraction; energy share by
layer / band / block / module / attention type (24 linear vs 8 full); readout alignment of residual-writing modules
(down_proj, o_proj) with probe-token unembedding contrasts; principal-subspace overlap with reference adapters
(does this run reuse the agentic run's directions, or a previous run's?).
Required probes: cast_vs_r (59606,427) plus one per targeted skill (e.g. the `unanswerable` type token vs the run's most
common wrong type token), chosen BEFORE looking at results.

## B. Training dynamics (logged in the training notebook)
- Save the adapter every N steps (~10 checkpoints/run). Run ledger A on each: norm growth curve, where energy accumulates
  first, whether directions stabilise (overlap between consecutive checkpoints).
- Per-example loss per epoch (dataset cartography: never-learned / instantly-learned / ambiguous examples).
- Per-example gradient sketches on LoRA params at 2-3 checkpoints, random-projected to 4096 dims (TRAK/LESS style,
  fixed projection seed recorded). Enables: "which training examples pushed validation failure X", data reweighting,
  targeted unlearning. Store `grads_<ckpt>.npy` (n_examples x 4096 float16) + `grad_index.jsonl` (example id, source, skill tags).

## C. Behavioural attribution (GPU, edit service, margin mode)
For each run's final adapter (base = stock), submit as coordinator:
1. full adapter, and its norm-matched random adapter;
2. band ablations: keep only 0-7 / 8-15 / 16-23 / 24-31; and remove each band;
3. module splits: MLP-only, attention-only;
4. top-k SVD truncations (k = 1, 4) to test low-rank sufficiency;
5. dose 0.25 / 0.5 / 0.75 / 1.0.
Probe sets: the CAST margin battery (regression guard) + a skill margin set per target skill (e.g. unanswerable questions:
logprob of `"unanswerable"` at the final-JSON type slot vs the model's top alternative), built on train/val businesses only.
Specific effect = variant shift minus its norm-matched random control's shift; > 0.05 nats and outside jitter.

## D. Task-level evaluation
B5 H1 harness on the run's validation businesses (never lockbox until final): exact accuracy, PA, abstention-gated PA,
behaviour counts (invalid_final, output caps, context overflow, SQL errors, unanswerable recall). Paired vs stock and vs
the previous run, same questions.

## E. Record for Astra (per run `RUN.md`)
Data (sources, counts, skill tags, hashes), training config, A-D results, and a short "weight-space facts" list:
claims with evidence paths, each labelled DESCRIPTIVE (static) or CAUSAL (ablation with random control). Then
`hypotheses.md`: what this run suggests trying next, each with a prediction and kill criterion (FAST_RULES).
Cross-run: `INDEX.md` keeps one line per run + a cumulative "known facts about Qwen3.5-9B weight space" section.
