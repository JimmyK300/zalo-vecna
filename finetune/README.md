# FT1 — one locked fine-tuning run (Colab)

**Method (locked):** LoRA SFT of `Qwen/Qwen3.5-9B` on correct DeepSeek V4.1 Flash traces collected in the **H3b**
harness (replayed to exact prompts/tool results), **no reasoning** (thinking off, as at inference), unanswerable
examples downsampled to ~2.5%, loss on assistant turns only. LoRA r=16, alpha=32, dropout 0.05, all linear layers;
2 epochs, lr 1e-4 cosine, 3% warmup, 16 conversations per step, max 30k tokens, bf16 + gradient checkpointing.
Final checkpoint is used. The 3 validation businesses and the 80 eval questions (`eval80.json`) are never trained on.

**Data file** (not in git: contains competition data rows): `sft_nothink.jsonl` (each example has `tags`: period, metric, op, SQL features; used for stratified gradient sketches and weight-space analysis) — put it on Google Drive.

## Colab
```bash
!git clone -b harness-h3b https://github.com/JimmyK300/zalo-vecna.git && cd zalo-vecna
!pip -q install "transformers>=5.2" peft accelerate datasets flash-linear-attention causal-conv1d
# 1) 10-minute smoke test: confirms the Qwen3.5 linear-attention kernels load (check "first optimizer step after Ns")
!python finetune/train_sft.py --data /content/drive/MyDrive/ft1/sft_nothink.jsonl --out /content/drive/MyDrive/ft1/smoke --limit 50 --epochs 1 --sketch-examples 8
# 2) the real run (~4-5 h on RTX PRO 6000 Blackwell, ~5-7 h on A100 80GB)
!python finetune/train_sft.py --data /content/drive/MyDrive/ft1/sft_nothink.jsonl --out /content/drive/MyDrive/ft1/run
```
Outputs (Drive): `ckpt_step*/` (10 checkpoints), `final/`, `train_log.jsonl`, `example_losses.jsonl` (per-example loss each
epoch), `grad_sketch_mid.npy` / `grad_sketch_final.npy` + `grad_index.json` (per-example gradient CountSketch for
attribution), `run_config.json`. Bring `final/` + logs back for the weight ledger (`D:/astra/research/weight_ledger`).

## Pass/fail evaluation (same harness, same questions, same stack)
Needs the DataMate train folder (`devkit/train`: questions.jsonl, answers.jsonl, rulebook.md, calendar.csv, data/).
```bash
!python finetune/eval_h3b_hf.py --data /content/drive/MyDrive/devkit/train --ids finetune/eval80.json --out base_eval.jsonl
!python finetune/eval_h3b_hf.py --data /content/drive/MyDrive/devkit/train --ids finetune/eval80.json --adapter /content/drive/MyDrive/ft1/run/final --out ft1_eval.jsonl
```
Keep the fine-tune only if **real-mix PA with unanswerable->abstain** improves over the base on these questions
(reference from the OpenRouter bf16 base: H3b + unanswerable->abstain real-mix PA = -0.073).

## If the RTX PRO 6000 (G4) is not available
The loss computes logits only at assistant positions (full-sequence logits over Qwen's 248k vocabulary would need ~20 GB
at 22k tokens), so bf16 LoRA fits a 40 GB A100.
| GPU | flags | 2 epochs, opt2 (28M tokens) |
|---|---|---|
| RTX PRO 6000 96 GB (G4) | (none) | ~5 h |
| A100 40/80 GB | (none) | ~6-8 h — near the Colab session limit: rerun the same command with `--resume` after a disconnect |
| L4 24 GB | `--four-bit` (QLoRA) | ~20 h — use opt1 and `--epochs 1` (~8-10 h) or rent a GPU |
| Rented A100 80GB / H100 (RunPod, Vast, Lambda) | (none) | ~3-5 h, ~$8-12, no session limit |
`--resume` reloads the newest `ckpt_step*` adapter in `--out`, fast-forwards the schedule and data order (same seed);
optimizer moments restart (small approximation). Always write `--out` to Drive.
