"""Cleaner-only smoke test tool.
Reads small sample of questions from dataset (default 20-30), runs cleaner ONLY (no solver, no recipes, no submission).
Checkpoints per question for interruption resilience, and produces review CSV/JSONL with blank auditor columns.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.cleaning import VERSION, clean_question, QuestionCleanResult
from datamate.cleaner_llm import CleanerLLM
from datamate.workspace import load_workspace


def run_cleaner_smoke(
    data_dir: Path,
    out_dir: Path,
    modes: list[str] = ("rules",),
    limit: int = 20,
    seed: int = 3407,
    fold: str = "dev",
    model: str | None = None,
    cleaner_url: str = "http://127.0.0.1:11434/v1",
    cleaner_budget_sec: float = 120,
    cleaner_timeout_sec: float = 8,
    cleaner_max_calls: int = 50,
    resume: bool = True,
    skip_entities: bool = False
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    questions_file = data_dir / "questions.jsonl"
    if not questions_file.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_file}")

    data_fingerprint = hashlib.sha256(questions_file.read_bytes()).hexdigest()
    questions = [json.loads(s) for s in questions_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    businesses = sorted({r["business_id"] for r in questions})
    random.Random(seed).shuffle(businesses)

    groups = dict(train=businesses[:20], dev=businesses[20:25], audit=businesses[25:])
    if fold not in groups:
        raise ValueError(f"Invalid fold {fold}; choose train, dev, or audit")

    chosen_questions = [r for r in questions if r["business_id"] in groups[fold]]
    if limit:
        random.Random(seed).shuffle(chosen_questions)
        chosen_questions = chosen_questions[:limit]

    # Sort chosen by business_id then id
    chosen_questions.sort(key=lambda x: (x["business_id"], x["id"]))

    manifest_path = out_dir / "smoke_manifest.json"
    manifest_split = dict(
        version=VERSION,
        seed=seed,
        fold=fold,
        limit=limit,
        ids=[q["id"] for q in chosen_questions],
        dataset_fingerprint=data_fingerprint
    )

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("split") != manifest_split:
                if not resume:
                    raise ValueError("Manifest split mismatch; specify new out directory or enable resume")
        except Exception:
            pass

    manifest_doc = {
        "split": manifest_split,
        "config": {
            "modes": list(modes),
            "model": model,
            "cleaner_url": cleaner_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    }
    manifest_path.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional CleanerLLM setup
    llm_cleaner = None
    if "rules+llm" in modes:
        if not model:
            raise ValueError("Mode rules+llm requires --model")
        cache_dir = out_dir / "cleaner_cache"
        llm_cleaner = CleanerLLM(
            model=model,
            endpoint=cleaner_url,
            cache_dir=cache_dir,
            budget=cleaner_budget_sec,
            timeout=cleaner_timeout_sec,
            max_calls=cleaner_max_calls
        )

    # Cache entities per shop to avoid repeated loading
    shop_entities: dict[str, list[str]] = {}
    shop_dir = data_dir / "data"

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    results = []
    latencies = []
    decisions = Counter()
    rejection_reasons = Counter()
    changes_count = 0

    t_start_total = time.monotonic()

    for idx, q in enumerate(chosen_questions, 1):
        qid = q["id"]
        biz_id = q["business_id"]
        orig_text = q["question"]

        cp_file = checkpoint_dir / f"{qid}.json"
        if resume and cp_file.exists():
            try:
                row_data = json.loads(cp_file.read_text(encoding="utf-8"))
                results.append(row_data)
                decisions[row_data.get("decision", "none")] += 1
                if row_data.get("rules_normalized") != orig_text or row_data.get("final_normalized") != orig_text:
                    changes_count += 1
                lat = row_data.get("cleaner_latency_ms", 0) / 1000.0
                if lat:
                    latencies.append(lat)
                if row_data.get("decision") == "rejected":
                    rejection_reasons[row_data.get("reason", "unknown")] += 1
                continue
            except Exception:
                pass

        is_off_mode = ("off" in modes and "rules" not in modes and "rules+llm" not in modes)
        if is_off_mode or skip_entities:
            entities = []
        else:
            # Load shop entities lazily if shop directory exists
            if biz_id not in shop_entities:
                entities = []
                biz_path = shop_dir / biz_id
                if biz_path.exists():
                    try:
                        ws = load_workspace(biz_path, cleaner=True)
                        for table in ws.tables.values():
                            for role, column in table.roles.items():
                                if role in ("customer", "customer_code", "item", "item_code", "item_group") and column in table.df:
                                    entities.extend(str(v) for v in table.df[column].dropna().unique()[:500])
                    except Exception:
                        pass
                shop_entities[biz_id] = sorted(set(entities))
            entities = shop_entities[biz_id]

        t0 = time.monotonic()
        if is_off_mode:
            rules_norm = orig_text
            final_norm = orig_text
            proposal = ""
            decision = "off"
            source = "none"
            reason = "cleaner mode off"
        else:
            # 1. Rules pass
            clean_res = clean_question(orig_text, entities)
            rules_norm = clean_res.normalized

            proposal = ""
            decision = "rules_applied" if rules_norm != orig_text else "rules_unchanged"
            source = "none"
            reason = ""
            final_norm = rules_norm

            # 2. LLM pass if requested and ambiguous
            if "rules+llm" in modes and llm_cleaner is not None:
                if clean_res.ambiguous or clean_res.ambiguous_spans:
                    clean_res = llm_cleaner.question(clean_res, context_hash="smoke", deadline=time.time() + cleaner_timeout_sec)
                    llm_meta = getattr(clean_res, "llm", {}) or {}
                    proposal = llm_meta.get("proposal", "")
                    decision = llm_meta.get("decision", "none")
                    source = llm_meta.get("source", "none")
                    reason = llm_meta.get("reason", llm_meta.get("validation", ""))
                    final_norm = clean_res.normalized
                else:
                    decision = "skipped"
                    reason = "no ambiguity detected by rules"

        elapsed = time.monotonic() - t0
        latencies.append(elapsed)

        if rules_norm != orig_text or final_norm != orig_text:
            changes_count += 1

        decisions[decision] += 1
        if decision == "rejected":
            rejection_reasons[reason] += 1

        row = {
            "id": qid,
            "business_id": biz_id,
            "fold": fold,
            "original_question": orig_text,
            "rules_normalized": rules_norm,
            "llm_proposed": proposal,
            "final_normalized": final_norm,
            "decision": decision,
            "reason": reason,
            "source": source,
            "cleaner_latency_ms": round(elapsed * 1000, 2),
            # Blank columns for human auditor
            "review_label": "",  # giu_nghia | doi_nghia | khong_ket_luan
            "error_type": "",    # time_drop | intent_flip | attr_swap | entity_loss | relation_inversion | other
            "reviewer_notes": ""
        }
        # Write checkpoint
        cp_file.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(row)

    total_wall = time.monotonic() - t_start_total

    # Write review CSV
    review_csv = out_dir / "cleaner_review.csv"
    if results:
        fieldnames = list(results[0].keys())
        with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Write review JSONL
    review_jsonl = out_dir / "cleaner_review.jsonl"
    with review_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p90 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.90))] if latencies else 0
    avg_lat = sum(latencies) / max(1, len(latencies))

    summary = {
        "total_questions": len(results),
        "questions_with_changes": changes_count,
        "decisions": dict(decisions),
        "rejection_reasons": dict(rejection_reasons),
        "latencies": {
            "avg_ms": round(avg_lat * 1000, 2),
            "median_ms": round(p50 * 1000, 2),
            "p90_ms": round(p90 * 1000, 2),
            "total_wall_sec": round(total_wall, 2)
        },
        "model_load_time": "not_measured (server runs independently)",
        "notice": "These statistics measure syntactic / rule decisions. They do NOT represent semantic accuracy without manual review."
    }

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "review_csv": str(review_csv),
        "review_jsonl": str(review_jsonl)
    }


def main():
    p = argparse.ArgumentParser(description="Cleaner-only smoke test tool.")
    p.add_argument("--data", default=Path("dataset/train"), type=Path, help="Path to train directory.")
    p.add_argument("--out", default=Path("out/smoke-cleaner"), type=Path, help="Output directory.")
    p.add_argument("--modes", nargs="+", choices=["off", "rules", "rules+llm"], default=["rules"], help="Cleaner modes.")
    p.add_argument("--limit", type=int, default=20, help="Number of questions to test (default 20, max recommended 30).")
    p.add_argument("--seed", type=int, default=3407, help="Random seed for shop and question selection.")
    p.add_argument("--fold", choices=["train", "dev", "audit"], default="dev", help="Split fold to sample from.")
    p.add_argument("--model", help="Local model name for rules+llm.")
    p.add_argument("--cleaner-url", default="http://127.0.0.1:11434/v1", help="URL of local OpenAI-compatible endpoint.")
    p.add_argument("--cleaner-budget-sec", type=float, default=120, help="Total LLM budget seconds.")
    p.add_argument("--cleaner-timeout-sec", type=float, default=8, help="Per-request timeout seconds.")
    p.add_argument("--cleaner-max-calls", type=int, default=50, help="Max LLM calls.")
    p.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    p.add_argument("--skip-entities", action="store_true", help="Skip loading entity catalogs from shop data for fast question-only smoke testing.")
    args = p.parse_args()

    res = run_cleaner_smoke(
        data_dir=args.data,
        out_dir=args.out,
        modes=args.modes,
        limit=args.limit,
        seed=args.seed,
        fold=args.fold,
        model=args.model,
        cleaner_url=args.cleaner_url,
        cleaner_budget_sec=args.cleaner_budget_sec,
        cleaner_timeout_sec=args.cleaner_timeout_sec,
        cleaner_max_calls=args.cleaner_max_calls,
        resume=not args.no_resume,
        skip_entities=args.skip_entities
    )
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    print(f"\nReview files written to:\n  - {res['review_csv']}\n  - {res['review_jsonl']}")


if __name__ == "__main__":
    main()
