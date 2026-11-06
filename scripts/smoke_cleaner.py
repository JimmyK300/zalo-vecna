"""Cleaner-only smoke test tool.
Reads a small sample of questions from dataset (default 20), runs cleaner ONLY
(no solver, no recipes, no submission).
Checkpoints atomically per question with strict config/code hash verification,
and produces review CSV/JSONL with blank human auditor columns.
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
from datamate.cleaning import VERSION, clean_question
from datamate.cleaner_llm import CleanerLLM
from datamate.workspace import load_workspace

REQUIRED_CHECKPOINT_KEYS = {
    "id", "business_id", "fold", "business_group", "mode",
    "original_question", "rules_normalized", "llm_proposed",
    "final_normalized", "decision", "reason", "source",
    "cleaner_latency_ms", "review_label", "error_type",
    "reviewer_notes", "config_hash"
}


def compute_cleaner_code_hash(extra_code_tag: str = "") -> str:
    """Compute SHA-256 over cleaning.py, cleaner_llm.py, and smoke_cleaner.py (including prompt & guard)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in ("datamate/cleaning.py", "datamate/cleaner_llm.py", "scripts/smoke_cleaner.py"):
        p = root / rel
        if p.exists():
            h.update(rel.encode("utf-8"))
            h.update(p.read_bytes())
    if extra_code_tag:
        h.update(str(extra_code_tag).encode("utf-8"))
    return h.hexdigest()


def _validate_checkpoint_row(row_data: object, expected_qid: str, expected_mode: str, expected_config_hash: str, cp_path: Path) -> dict:
    if not isinstance(row_data, dict):
        raise ValueError(f"Corrupted checkpoint {cp_path}: expected JSON object")
    missing = REQUIRED_CHECKPOINT_KEYS - set(row_data.keys())
    if missing:
        raise ValueError(f"Corrupted checkpoint {cp_path}: missing required keys {sorted(missing)}")
    if row_data.get("id") != expected_qid:
        raise ValueError(f"Checkpoint ID mismatch in {cp_path}: expected {expected_qid}, found {row_data.get('id')}")
    if row_data.get("mode") != expected_mode:
        raise ValueError(
            f"Checkpoint mode mismatch in {cp_path}: checkpoint was created with mode='{row_data.get('mode')}', "
            f"cannot resume with mode='{expected_mode}'. Use a new --out directory."
        )
    if row_data.get("config_hash") != expected_config_hash:
        raise ValueError(
            f"Checkpoint config_hash mismatch in {cp_path}: configuration, input, or code changed. "
            f"Use a new --out directory."
        )
    return row_data


def run_cleaner_smoke(
    data_dir: Path,
    out_dir: Path,
    modes: list[str] | tuple[str, ...] = ("rules",),
    limit: int = 20,
    seed: int = 3407,
    fold: str = "dev",
    model: str | None = None,
    cleaner_url: str = "http://127.0.0.1:11434/v1",
    cleaner_budget_sec: float = 120,
    cleaner_timeout_sec: float = 8,
    cleaner_max_calls: int = 50,
    resume: bool = True,
    skip_entities: bool = False,
    code_hash_override: str | None = None,
) -> dict:
    modes_list = list(dict.fromkeys(modes))
    if not modes_list:
        raise ValueError("At least one mode must be specified")
    for m in modes_list:
        if m not in ("off", "rules", "rules+llm"):
            raise ValueError(f"Invalid cleaner mode: {m}")
    if "rules+llm" in modes_list and not model:
        raise ValueError("Mode rules+llm requires --model")

    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions_file = data_dir / "questions.jsonl"
    if not questions_file.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_file}")

    data_fingerprint = hashlib.sha256(questions_file.read_bytes()).hexdigest()
    questions = [json.loads(s) for s in questions_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    businesses = sorted({r["business_id"] for r in questions})
    random.Random(seed).shuffle(businesses)

    if len(businesses) >= 25:
        groups = dict(train=businesses[:20], dev=businesses[20:25], audit=businesses[25:])
    else:
        # Support small synthetic fixtures in unit tests while keeping deterministic group split
        n = len(businesses)
        n_train = max(1, int(n * 0.65))
        n_dev = max(1, int(n * 0.2))
        groups = dict(
            train=businesses[:n_train],
            dev=businesses[n_train:n_train + n_dev] or businesses[:1],
            audit=businesses[n_train + n_dev:] or businesses[-1:]
        )

    if fold not in groups:
        raise ValueError(f"Invalid fold {fold}; choose train, dev, or audit")

    chosen_questions = [r for r in questions if r["business_id"] in groups[fold]]
    if limit:
        random.Random(seed).shuffle(chosen_questions)
        chosen_questions = chosen_questions[:limit]

    chosen_questions.sort(key=lambda x: (x["business_id"], x["id"]))
    chosen_ids = [q["id"] for q in chosen_questions]

    code_hash = code_hash_override if code_hash_override is not None else compute_cleaner_code_hash()

    manifest_split = dict(
        version=VERSION,
        seed=seed,
        fold=fold,
        limit=limit,
        ids=chosen_ids,
        dataset_fingerprint=data_fingerprint,
    )
    manifest_config = dict(
        modes=modes_list,
        model=model,
        cleaner_url=cleaner_url,
        skip_entities=bool(skip_entities),
        cleaner_budget_sec=float(cleaner_budget_sec),
        cleaner_timeout_sec=float(cleaner_timeout_sec),
        cleaner_max_calls=int(cleaner_max_calls),
        version=VERSION,
        code_hash=code_hash,
    )
    config_hash = hashlib.sha256(
        json.dumps({"split": manifest_split, "config": manifest_config}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    manifest_path = out_dir / "smoke_manifest.json"
    checkpoint_dir = out_dir / "checkpoints"

    # Strict manifest check before running or resuming: never overwrite or mix mismatched runs
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Corrupted smoke_manifest.json in {out_dir}: {exc}. Use a new --out directory.") from exc
        if not isinstance(existing, dict) or "split" not in existing or "config" not in existing:
            raise ValueError(f"Invalid smoke_manifest.json schema in {out_dir}. Use a new --out directory.")
        mismatches = []
        for k, exp_v in manifest_split.items():
            act_v = existing["split"].get(k)
            if act_v != exp_v:
                mismatches.append(f"split.{k} (expected {exp_v!r}, found {act_v!r})")
        for k, exp_v in manifest_config.items():
            act_v = existing["config"].get(k)
            if act_v != exp_v:
                mismatches.append(f"config.{k} (expected {exp_v!r}, found {act_v!r})")
        if mismatches:
            raise ValueError(
                f"Smoke manifest mismatch in {out_dir}: {'; '.join(mismatches)}. "
                f"Refusing to mix runs or overwrite checkpoints; please specify a new --out directory."
            )
    elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        # Even if smoke_manifest.json was removed, verify existing checkpoints match current config_hash
        for cp_existing in sorted(checkpoint_dir.glob("*.json")):
            try:
                raw_cp = json.loads(cp_existing.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Corrupted checkpoint {cp_existing}: {exc}") from exc
            if not isinstance(raw_cp, dict) or raw_cp.get("config_hash") != config_hash:
                raise ValueError(
                    f"Existing checkpoint {cp_existing} does not match current configuration/mode. "
                    f"Use a new --out directory."
                )

    checkpoint_dir.mkdir(exist_ok=True)

    # Write running manifest state
    manifest_doc = {
        "split": manifest_split,
        "config": manifest_config,
        "config_hash": config_hash,
        "status": "running",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_tmp = manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_tmp.replace(manifest_path)

    # Optional CleanerLLM setup
    llm_cleaner = None
    if "rules+llm" in modes_list:
        cache_dir = out_dir / "cleaner_cache"
        llm_cleaner = CleanerLLM(
            model=model,
            endpoint=cleaner_url,
            cache_dir=cache_dir,
            budget=cleaner_budget_sec,
            timeout=cleaner_timeout_sec,
            max_calls=cleaner_max_calls,
        )

    shop_entities: dict[str, list[str]] = {}
    shop_dir = data_dir / "data"

    all_results: list[dict] = []
    by_mode_summary: dict[str, dict] = {}
    t_start_total = time.monotonic()

    for mode in modes_list:
        mode_results: list[dict] = []
        latencies: list[float] = []
        decisions: Counter = Counter()
        rejection_reasons: Counter = Counter()
        changes_count = 0
        t_start_mode = time.monotonic()

        for q in chosen_questions:
            qid = q["id"]
            biz_id = q["business_id"]
            orig_text = q["question"]

            cp_file = checkpoint_dir / (f"{qid}.json" if len(modes_list) == 1 else f"{mode}_{qid}.json")
            legacy_cp_file = checkpoint_dir / f"{qid}.json"

            # Check if any incompatible single-file checkpoint exists when running single mode
            if resume and cp_file.exists():
                try:
                    raw_cp = json.loads(cp_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ValueError(f"Corrupted checkpoint JSON in {cp_file}: {exc}") from exc
                row_data = _validate_checkpoint_row(raw_cp, qid, mode, config_hash, cp_file)
                mode_results.append(row_data)
                all_results.append(row_data)
                decisions[row_data.get("decision", "none")] += 1
                if row_data.get("rules_normalized") != orig_text or row_data.get("final_normalized") != orig_text:
                    changes_count += 1
                lat = float(row_data.get("cleaner_latency_ms", 0.0)) / 1000.0
                latencies.append(lat)
                if row_data.get("decision") == "rejected":
                    rejection_reasons[row_data.get("reason", "unknown")] += 1
                continue
            elif resume and legacy_cp_file.exists() and legacy_cp_file != cp_file:
                # Validate legacy checkpoint isn't from another config
                try:
                    raw_cp = json.loads(legacy_cp_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ValueError(f"Corrupted checkpoint JSON in {legacy_cp_file}: {exc}") from exc
                _validate_checkpoint_row(raw_cp, qid, mode, config_hash, legacy_cp_file)

            is_off_mode = (mode == "off")
            if is_off_mode or skip_entities:
                entities = []
            else:
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
                clean_res = clean_question(orig_text, entities)
                rules_norm = clean_res.normalized
                proposal = ""
                decision = "rules_applied" if rules_norm != orig_text else "rules_unchanged"
                source = "none"
                reason = ""
                final_norm = rules_norm

                if mode == "rules+llm" and llm_cleaner is not None:
                    if clean_res.ambiguous or clean_res.ambiguous_spans:
                        clean_res = llm_cleaner.question(clean_res, context_hash=f"smoke:{config_hash}", deadline=time.time() + cleaner_timeout_sec)
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
                "business_group": fold,
                "mode": mode,
                "original_question": orig_text,
                "rules_normalized": rules_norm,
                "llm_proposed": proposal,
                "final_normalized": final_norm,
                "decision": decision,
                "reason": reason,
                "source": source,
                "cleaner_latency_ms": round(elapsed * 1000, 2),
                "review_label": "",
                "error_type": "",
                "reviewer_notes": "",
                "config_hash": config_hash,
            }
            # Atomic checkpoint write
            cp_tmp = cp_file.with_suffix(".tmp")
            cp_tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            cp_tmp.replace(cp_file)

            mode_results.append(row)
            all_results.append(row)

        mode_wall = time.monotonic() - t_start_mode
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0
        p90 = latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.90))] if latencies_sorted else 0.0
        avg_lat = sum(latencies_sorted) / max(1, len(latencies_sorted))

        by_mode_summary[mode] = {
            "mode": mode,
            "total_questions": len(mode_results),
            "questions_with_changes": changes_count,
            "decisions": dict(decisions),
            "rejection_reasons": dict(rejection_reasons),
            "latencies": {
                "avg_ms": round(avg_lat * 1000, 2),
                "median_ms": round(p50 * 1000, 2),
                "p90_ms": round(p90 * 1000, 2),
                "total_wall_sec": round(mode_wall, 2),
            },
        }

        if len(modes_list) > 1 and mode_results:
            mode_csv = out_dir / f"cleaner_review_{mode}.csv"
            review_cols = [k for k in mode_results[0].keys() if k != "config_hash"]
            with mode_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=review_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(mode_results)

    total_wall = time.monotonic() - t_start_total

    # Write combined review CSV & JSONL (omitting internal config_hash from CSV columns for clean auditing)
    review_csv = out_dir / "cleaner_review.csv"
    if all_results:
        fieldnames = [k for k in all_results[0].keys() if k != "config_hash"]
        with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

    review_jsonl = out_dir / "cleaner_review.jsonl"
    with review_jsonl.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    primary_mode = modes_list[-1]
    primary_stats = by_mode_summary[primary_mode]

    summary = {
        "modes_run": modes_list,
        "fold": fold,
        "business_group": fold,
        "total_questions": primary_stats["total_questions"],
        "questions_with_changes": primary_stats["questions_with_changes"],
        "decisions": primary_stats["decisions"],
        "rejection_reasons": primary_stats["rejection_reasons"],
        "latencies": {
            **primary_stats["latencies"],
            "total_wall_sec": round(total_wall, 2),
        },
        "by_mode": by_mode_summary,
        "config_hash": config_hash,
        "model_load_time": "not_measured (server runs independently)",
        "notice": "These statistics measure syntactic / rule decisions. They do NOT represent semantic accuracy without manual review.",
    }

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Mark manifest completed only after all outputs are flushed
    manifest_doc["status"] = "completed"
    manifest_doc["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest_tmp = manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_tmp.replace(manifest_path)

    return {
        "summary": summary,
        "review_csv": str(review_csv),
        "review_jsonl": str(review_jsonl),
    }


def main():
    p = argparse.ArgumentParser(description="Cleaner-only smoke test tool (never invokes solver or generates submissions).")
    p.add_argument("--data", default=Path("dataset/train"), type=Path, help="Path to train directory.")
    p.add_argument("--out", default=Path("out/smoke-cleaner"), type=Path, help="Output directory.")
    p.add_argument("--modes", nargs="+", choices=["off", "rules", "rules+llm"], default=["rules"], help="Cleaner modes to run.")
    p.add_argument("--limit", type=int, default=20, help="Number of questions to test (default 20, max recommended 30).")
    p.add_argument("--seed", type=int, default=3407, help="Random seed for shop and question selection.")
    p.add_argument("--fold", choices=["train", "dev", "audit"], default="dev", help="Split fold to sample from.")
    p.add_argument("--model", help="Local model name for rules+llm.")
    p.add_argument("--cleaner-url", default="http://127.0.0.1:11434/v1", help="URL of local OpenAI-compatible endpoint.")
    p.add_argument("--cleaner-budget-sec", type=float, default=120, help="Total LLM budget seconds.")
    p.add_argument("--cleaner-timeout-sec", type=float, default=8, help="Per-request timeout seconds.")
    p.add_argument("--cleaner-max-calls", type=int, default=50, help="Max LLM calls.")
    p.add_argument("--no-resume", action="store_true", help="Re-run questions within the same compatible configuration.")
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
        skip_entities=args.skip_entities,
    )
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    print(f"\nReview files written to:\n  - {res['review_csv']}\n  - {res['review_jsonl']}")


if __name__ == "__main__":
    main()
