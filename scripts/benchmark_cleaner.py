"""Paired cleaner runs on a frozen business split. No gold is sent to inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.evaluate import score, match


def check_or_save_manifest(out_dir: Path, split_data: dict, run_config: dict) -> tuple[bool, str]:
    """Decouple frozen split data from run configurations.
    Allows running modes sequentially in the same directory without split collision."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split.json"
    existing_data = None
    if manifest_path.exists():
        try:
            existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to read existing manifest: {e}"

    if existing_data is not None:
        target_split = existing_data.get("split") if isinstance(existing_data, dict) and "split" in existing_data else existing_data
        mismatches = []
        for k in ("seed", "fold", "ids", "dataset_fingerprint"):
            exp = split_data.get(k)
            actual = target_split.get(k)
            if actual is None and k == "dataset_fingerprint":
                actual = target_split.get("question_hash")
            if exp != actual:
                mismatches.append(f"{k} (expected {exp}, found {actual})")
        if mismatches:
            return False, f"Frozen split mismatch: {', '.join(mismatches)}; use a new output directory"
        runs = existing_data.get("runs", []) if isinstance(existing_data, dict) and "split" in existing_data else []
    else:
        runs = []

    run_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **run_config
    }
    runs.append(run_entry)

    doc = {
        "split": split_data,
        "runs": runs
    }
    manifest_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, ""


def validate_comparison_compatibility(run_a: dict, run_b: dict) -> tuple[bool, str]:
    """Verify that two runs share identical solver backend, model, and dataset fingerprint,
    ensuring that measured performance differences stem solely from the cleaner mode."""
    fp_a = run_a.get("dataset_fingerprint")
    fp_b = run_b.get("dataset_fingerprint")
    if fp_a and fp_b and fp_a != fp_b:
        return False, f"Incompatible dataset fingerprints ({fp_a} vs {fp_b})"

    back_a = run_a.get("backend")
    back_b = run_b.get("backend")
    if back_a != back_b:
        return False, f"Incompatible solver backends ({back_a} vs {back_b})"

    mod_a = run_a.get("model")
    mod_b = run_b.get("model")
    if mod_a != mod_b:
        return False, f"Incompatible solver models ({mod_a} vs {mod_b})"

    return True, "Compatible solver configurations"


def format_review_row(
    qid: str,
    original: str,
    rules_norm: str,
    llm_result: dict | None = None,
    business_id: str = "",
    fold: str = "",
    gold_type: str = "",
    gold_answer: str = "",
    extra_scores: dict | None = None
) -> dict:
    """Format a comprehensive row for human audit review, accurately capturing
    original proposal, rejection reason, cached acceptance, and final normalized question."""
    llm = llm_result or {}
    proposal = llm.get("proposal", "")
    decision = llm.get("decision", "none")
    source = llm.get("source", "none")
    status = llm.get("status", decision)
    reason = llm.get("reason", llm.get("validation", ""))

    if not proposal and decision not in ("rejected", "fallback"):
        proposal = llm.get("normalized", "")

    if decision == "accepted" or (status in ("accepted", "generated") and decision != "rejected" and not llm.get("uncertain_spans")):
        final_norm = proposal if proposal else (llm.get("normalized") or rules_norm)
    else:
        final_norm = rules_norm

    row = {
        "id": qid,
        "business_id": business_id,
        "fold": fold,
        "original_question": original,
        "rules_normalized": rules_norm,
        "llm_proposed": proposal,
        "decision": decision,
        "source": source,
        "guard_status": status,
        "guard_reason": reason,
        "final_normalized": final_norm,
        "latency_seconds": llm.get("seconds", 0.0),
        "gold_type": gold_type,
        "gold_answer": gold_answer,
        "human_review": ""  # Explicit blank column for human auditor
    }
    if extra_scores:
        row.update(extra_scores)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backend", default="scripted")
    p.add_argument("--model")
    p.add_argument("--modes", nargs="+", choices=["off", "rules", "rules+llm"], default=["off", "rules"])
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--fold", choices=["train", "dev", "audit"], default="dev")
    p.add_argument("--open-audit", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--cleaner-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--cleaner-budget-sec", type=float, default=900)
    p.add_argument("--cleaner-timeout-sec", type=float, default=8)
    p.add_argument("--cleaner-max-calls", type=int, default=100)
    args = p.parse_args()

    if args.fold == "audit" and not args.open_audit:
        p.error("freeze configuration first; --open-audit explicitly records holdout access")

    questions_file = args.data / "questions.jsonl"
    questions = [json.loads(s) for s in questions_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    businesses = sorted({r["business_id"] for r in questions})
    random.Random(args.seed).shuffle(businesses)
    if len(businesses) != 30:
        p.error("Expected the organizer train set with 30 businesses, not public/private test")

    groups = dict(train=businesses[:20], dev=businesses[20:25], audit=businesses[25:])
    chosen = sorted(r["id"] for r in questions if r["business_id"] in groups[args.fold])
    if args.limit:
        random.Random(args.seed).shuffle(chosen)
        chosen = sorted(chosen[:args.limit])

    split_data = dict(
        seed=args.seed,
        groups=groups,
        fold=args.fold,
        ids=chosen,
        dataset_fingerprint=hashlib.sha256(questions_file.read_bytes()).hexdigest(),
    )
    run_config = dict(
        modes=args.modes,
        backend=args.backend,
        model=args.model,
        audit_opened=args.fold == "audit",
        cleaner_url=args.cleaner_url,
    )
    ok, err_msg = check_or_save_manifest(args.out, split_data, run_config)
    if not ok:
        p.error(err_msg)

    ids_path = args.out / "ids.txt"
    ids_path.write_text("\n".join(chosen), encoding="utf-8")

    answers_file = args.data / "answers.jsonl"
    gold = {}
    if answers_file.exists():
        gold = {
            r["id"]: r
            for r in (json.loads(s) for s in answers_file.read_text(encoding="utf-8").splitlines() if s.strip())
            if r["id"] in chosen
        }

    q_lookup = {q["id"]: q for q in questions if q["id"] in chosen}

    # Execute requested modes
    for mode in args.modes:
        directory = args.out / mode
        directory.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "datamate.run",
            "--data", str(args.data),
            "--out", str(directory / "submission.jsonl"),
            "--backend", args.backend,
            "--cleaner", mode,
            "--only-ids", str(ids_path),
            "--workers", "1",
            "--cleaner-url", args.cleaner_url,
            "--cleaner-budget-sec", str(args.cleaner_budget_sec),
            "--cleaner-timeout-sec", str(args.cleaner_timeout_sec),
            "--cleaner-max-calls", str(args.cleaner_max_calls),
        ]
        if args.model:
            cmd += ["--model", args.model]

        start = time.monotonic()
        subprocess.run(cmd, check=True)

    # Evaluate all available modes in output directory (including previously run modes)
    candidate_modes = ["off", "rules", "rules+llm"]
    all_available_modes = [m for m in candidate_modes if (args.out / m / "submission.jsonl").exists()]

    result: dict[str, dict] = {}
    mode_evaluations: dict[str, dict] = {}
    mode_traces: dict[str, dict[str, dict]] = {}

    for mode in all_available_modes:
        directory = args.out / mode
        pred_file = directory / "submission.jsonl"
        pred = {}
        if pred_file.exists():
            pred = {
                r["id"]: r
                for r in (json.loads(s) for s in pred_file.read_text(encoding="utf-8").splitlines() if s.strip())
            }

        eval_res = {}
        if gold:
            raw_metrics = score(pred, gold, only_pred=False)
            detailed_rows = raw_metrics.pop("rows", [])
            eval_res = raw_metrics
            per_q_score = {}
            for row in detailed_rows:
                qid, kind, p_item, g_item = row
                s_val = 1 if kind == "correct" else (-1 if kind == "wrong" else 0)
                per_q_score[qid] = {
                    "score": s_val,
                    "kind": kind,
                    "prediction": p_item,
                    "gold": g_item,
                }
            mode_evaluations[mode] = per_q_score
        else:
            eval_res = {"status": "no_gold_labels"}

        traces_dict: dict[str, dict] = {}
        for f in (directory / "logs").glob("q_*.json"):
            qid = f.stem
            try:
                traces_dict[qid] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
        mode_traces[mode] = traces_dict

        llm_statuses = []
        reject_reasons = Counter()
        changes_count = 0
        latencies = []

        for qid in chosen:
            tr = traces_dict.get(qid, {})
            lat = tr.get("total_seconds", tr.get("seconds", 0))
            if lat:
                latencies.append(lat)
            cleaner_data = tr.get("cleaner") or {}
            if cleaner_data.get("changes"):
                changes_count += 1
            llm_info = cleaner_data.get("llm") or {}
            st = llm_info.get("status")
            if st:
                llm_statuses.append(st)
                if st == "rejected":
                    reject_reasons[llm_info.get("reason", "unknown")] += 1

        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else None
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None

        result[mode] = dict(
            metrics=eval_res,
            questions_evaluated=len(chosen),
            questions_with_cleaner_changes=changes_count,
            question_latency_p50=round(p50, 3) if p50 is not None else None,
            question_latency_p95=round(p95, 3) if p95 is not None else None,
            llm_calls_generated=sum(st == "generated" for st in llm_statuses),
            llm_calls_cached=sum(st == "cache" for st in llm_statuses),
            llm_calls_rejected=sum(st == "rejected" for st in llm_statuses),
            llm_calls_fallback=sum(st == "fallback" for st in llm_statuses),
            rejection_reasons=dict(reject_reasons),
            warning="Historical holdout is not pristine; prior recipes used whole train. Timings include resume if reused."
        )

    # Compute comparative flips between pairs of all available modes
    flips = {}
    manifest_doc = {}
    manifest_file = args.out / "split.json"
    if manifest_file.exists():
        try:
            manifest_doc = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            manifest_doc = {}
    all_runs = manifest_doc.get("runs", [])
    current_fp = split_data.get("dataset_fingerprint")

    def get_run_for_mode(m_name: str) -> dict:
        for r in reversed(all_runs):
            if m_name in r.get("modes", []):
                cfg = dict(r)
                cfg.setdefault("dataset_fingerprint", current_fp)
                return cfg
        return dict(backend=args.backend, model=args.model, dataset_fingerprint=current_fp)

    for i in range(len(all_available_modes)):
        for j in range(i + 1, len(all_available_modes)):
            m_a, m_b = all_available_modes[i], all_available_modes[j]
            pair_key = f"{m_a}_vs_{m_b}"
            run_a = get_run_for_mode(m_a)
            run_b = get_run_for_mode(m_b)
            compat, compat_reason = validate_comparison_compatibility(run_a, run_b)
            if not compat:
                flips[pair_key] = dict(
                    compatible=False,
                    incompatibility_reason=compat_reason,
                    warning="PA delta not evaluated due to solver mismatch; runs must share backend, model, and dataset."
                )
                continue

            c_to_w = []
            w_to_c = []
            a_to_c = []
            c_to_a = []
            scores_a = mode_evaluations.get(m_a, {})
            scores_b = mode_evaluations.get(m_b, {})
            for qid in chosen:
                sa = scores_a.get(qid, {}).get("score")
                sb = scores_b.get(qid, {}).get("score")
                if sa is not None and sb is not None:
                    if sa == 1 and sb == -1:
                        c_to_w.append(qid)
                    elif sa == -1 and sb == 1:
                        w_to_c.append(qid)
                    elif sa == 0 and sb == 1:
                        a_to_c.append(qid)
                    elif sa == 1 and sb == 0:
                        c_to_a.append(qid)
            pa_a = result.get(m_a, {}).get("metrics", {}).get("PA")
            pa_b = result.get(m_b, {}).get("metrics", {}).get("PA")
            flips[pair_key] = dict(
                compatible=True,
                correct_to_wrong=c_to_w,
                wrong_to_correct=w_to_c,
                abstain_to_correct=a_to_c,
                correct_to_abstain=c_to_a,
                net_correct_change=len(w_to_c) + len(a_to_c) - len(c_to_w) - len(c_to_a),
                pa_delta=round(pa_b - pa_a, 4) if pa_a is not None and pa_b is not None else None
            )

    result["comparisons"] = flips
    (args.out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate Human Review files (cleaner_review.jsonl & cleaner_review.csv)
    review_rows = []
    for qid in chosen:
        q_item = q_lookup.get(qid, {})
        gold_item = gold.get(qid, {})

        # Collect traces across modes
        trace_rules = mode_traces.get("rules", {}).get(qid, {})
        trace_llm = mode_traces.get("rules+llm", {}).get(qid, {})
        cleaner_rules = trace_rules.get("cleaner") or {}
        cleaner_llm = trace_llm.get("cleaner") or {}

        llm_meta = cleaner_llm.get("llm") or {}
        rules_norm = cleaner_rules.get("normalized", q_item.get("question", ""))

        extra_scores = {}
        for m in all_available_modes:
            m_score = mode_evaluations.get(m, {}).get(qid, {})
            extra_scores[f"pred_{m}"] = json.dumps(m_score.get("prediction"), ensure_ascii=False) if m_score else ""
            extra_scores[f"score_{m}"] = m_score.get("score", "")

        row_dict = format_review_row(
            qid=qid,
            original=q_item.get("question", ""),
            rules_norm=rules_norm,
            llm_result=llm_meta,
            business_id=q_item.get("business_id", ""),
            fold=args.fold,
            gold_type=gold_item.get("type", ""),
            gold_answer=json.dumps(gold_item.get("answer"), ensure_ascii=False) if "answer" in gold_item else "",
            extra_scores=extra_scores
        )
        review_rows.append(row_dict)

    review_jsonl = args.out / "cleaner_review.jsonl"
    with review_jsonl.open("w", encoding="utf-8") as f:
        for r in review_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    review_csv = args.out / "cleaner_review.csv"
    if review_rows:
        fieldnames = list(review_rows[0].keys())
        with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nReview files written to:\n  - {review_jsonl}\n  - {review_csv}")


if __name__ == "__main__":
    main()
