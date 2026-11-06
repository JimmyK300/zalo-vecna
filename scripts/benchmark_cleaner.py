"""Paired cleaner runs on a frozen business split. No gold is sent to inference.
Supports modes:
  - off:           cleaner off + baseline table processing (A)
  - rules:         rules cleaner + baseline table processing (B)
  - rules+llm:     rules+llm cleaner + baseline table processing (C)
  - rules+profile: rules cleaner + extended profiling/grounding context (D)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.evaluate import score, match

BENCHMARK_MODES = ("off", "rules", "rules+llm", "rules+profile")


def compute_solver_code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in ("datamate/cleaning.py", "datamate/cleaner_llm.py", "datamate/workspace.py", "datamate/agent.py", "datamate/run.py"):
        p = root / rel
        if p.exists():
            h.update(rel.encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


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


def verify_mode_output(mode_dir: Path, expected_split: dict) -> tuple[bool, str, dict]:
    """Verify that a mode subdirectory contains a completed, non-stale submission
    matching the expected split and bound to its own run_manifest.json."""
    run_manifest_path = mode_dir / "run_manifest.json"
    pred_file = mode_dir / "submission.jsonl"

    if not pred_file.exists():
        return False, f"Missing submission.jsonl in {mode_dir}", {"status": "missing", "output_verified": False}
    if not run_manifest_path.exists():
        return False, f"Missing run_manifest.json in {mode_dir} (stale or unverified output)", {
            "status": "stale", "stale_output": True, "output_verified": False
        }

    try:
        rm = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Corrupted run_manifest.json in {mode_dir}: {exc}", {"status": "failed", "output_verified": False}

    if not isinstance(rm, dict):
        return False, f"Invalid run_manifest.json in {mode_dir}", {"status": "failed", "output_verified": False}

    status = rm.get("status", "unknown")
    if status != "completed":
        return False, f"Run status is '{status}' (not 'completed') in {mode_dir}", {
            **rm, "output_verified": False
        }

    # Check submission file hash against run_manifest to detect stale/modified files
    actual_sha = hashlib.sha256(pred_file.read_bytes()).hexdigest()
    recorded_sha = rm.get("submission_sha256")
    if recorded_sha and recorded_sha != actual_sha:
        return False, f"Stale or modified submission.jsonl in {mode_dir} (sha256 mismatch with run_manifest)", {
            **rm, "status": "stale", "stale_output": True, "output_verified": False
        }

    # Check split match
    for k in ("seed", "fold", "ids", "dataset_fingerprint"):
        if k in expected_split and k in rm and rm.get(k) != expected_split.get(k):
            return False, f"Run manifest {k} mismatch in {mode_dir} ({rm.get(k)} != {expected_split.get(k)})", {
                **rm, "output_verified": False
            }

    # Parse submission.jsonl and verify all expected IDs are present and valid
    try:
        preds = {
            str(r["id"]): r
            for r in (json.loads(line) for line in pred_file.read_text(encoding="utf-8").splitlines() if line.strip())
        }
    except Exception as exc:
        return False, f"Corrupted submission.jsonl in {mode_dir}: {exc}", {**rm, "status": "failed", "output_verified": False}

    expected_ids = list(expected_split.get("ids") or rm.get("ids") or [])
    missing_ids = [qid for qid in expected_ids if qid not in preds]
    if missing_ids:
        return False, f"Incomplete submission in {mode_dir}: missing {len(missing_ids)} IDs (e.g. {missing_ids[:3]})", {
            **rm, "status": "incomplete", "missing_ids": missing_ids, "output_verified": False
        }

    error_ids = [qid for qid in expected_ids if preds[qid].get("type") in ("error", "skipped")]
    if error_ids:
        return False, f"Run had execution errors on {len(error_ids)} IDs in {mode_dir}", {
            **rm, "status": "failed", "error_ids": error_ids, "output_verified": False
        }

    enriched = {
        **rm,
        "completed_ids": sorted(preds.keys()),
        "missing_ids": [],
        "stale_output": False,
        "output_verified": True,
    }
    return True, "Verified completed mode output", enriched


def validate_comparison_compatibility(
    run_a: dict,
    run_b: dict,
    expected_split: dict | None = None
) -> tuple[bool, str]:
    """Verify that two runs share identical dataset/split/IDs, solver backend, solver model,
    solver configuration, code/config hash, and completed non-stale output status,
    ensuring that measured PA differences or flips stem solely from the cleaner mode."""
    for label, r in (("run_a", run_a), ("run_b", run_b)):
        if not isinstance(r, dict):
            return False, f"Invalid run metadata for {label}"
        st = r.get("status", "completed")
        if st != "completed":
            return False, f"Incompatible run status: {label} has status='{st}' (must be 'completed')"
        if r.get("stale_output") is True or r.get("output_verified") is False:
            return False, f"Unverified or stale output in {label}"
        if r.get("missing_ids"):
            return False, f"Incomplete output in {label}: missing IDs {r.get('missing_ids')[:3]}"
        if r.get("error_ids"):
            return False, f"Failed questions in {label}: {r.get('error_ids')[:3]}"

    # 1. Dataset fingerprint
    fp_a = run_a.get("dataset_fingerprint")
    fp_b = run_b.get("dataset_fingerprint")
    if fp_a != fp_b:
        return False, f"Incompatible dataset fingerprints ({fp_a} vs {fp_b})"
    if expected_split and expected_split.get("dataset_fingerprint") and fp_a and fp_a != expected_split["dataset_fingerprint"]:
        return False, f"Dataset fingerprint does not match frozen split ({fp_a} vs {expected_split['dataset_fingerprint']})"

    # 2. Split fold, seed, and question IDs
    for k in ("fold", "seed", "ids"):
        va, vb = run_a.get(k), run_b.get(k)
        if va is not None and vb is not None and va != vb:
            return False, f"Incompatible split {k} ({va} vs {vb})"
        if expected_split and expected_split.get(k) is not None:
            exp_v = expected_split[k]
            if (va is not None and va != exp_v) or (vb is not None and vb != exp_v):
                return False, f"Run {k} does not match expected frozen split"

    # Check completed_ids if provided
    comp_a = run_a.get("completed_ids")
    comp_b = run_b.get("completed_ids")
    target_ids = (expected_split or {}).get("ids") or run_a.get("ids") or run_b.get("ids")
    if target_ids:
        target_set = set(target_ids)
        if comp_a is not None and not target_set.issubset(set(comp_a)):
            return False, "run_a is missing required question IDs from split"
        if comp_b is not None and not target_set.issubset(set(comp_b)):
            return False, "run_b is missing required question IDs from split"
    elif comp_a is not None and comp_b is not None and set(comp_a) != set(comp_b):
        return False, "Incompatible completed question ID sets between runs"

    # 3. Solver backend and model
    back_a = run_a.get("backend")
    back_b = run_b.get("backend")
    if back_a != back_b:
        return False, f"Incompatible solver backends ({back_a} vs {back_b})"

    mod_a = run_a.get("model")
    mod_b = run_b.get("model")
    if mod_a != mod_b:
        return False, f"Incompatible solver models ({mod_a} vs {mod_b})"

    # 4. Solver configuration parameters
    cfg_a = run_a.get("solver_config") or run_a.get("inference")
    cfg_b = run_b.get("solver_config") or run_b.get("inference")
    if cfg_a is not None and cfg_b is not None and cfg_a != cfg_b:
        return False, f"Incompatible solver configurations ({cfg_a} vs {cfg_b})"

    # 5. Code / rulebook hashes if present on both
    for hk in ("rulebook_sha256", "data_hash", "code_hash", "cleaner_code_hash"):
        ha, hb = run_a.get(hk), run_b.get(hk)
        if ha is not None and hb is not None and ha != hb:
            return False, f"Incompatible {hk} ({ha} vs {hb})"

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
    p.add_argument("--modes", nargs="+", choices=list(BENCHMARK_MODES), default=["off", "rules"])
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--fold", choices=["train", "dev", "audit"], default="dev")
    p.add_argument("--open-audit", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--cleaner-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--cleaner-budget-sec", type=float, default=900)
    p.add_argument("--cleaner-timeout-sec", type=float, default=8)
    p.add_argument("--cleaner-max-calls", type=int, default=100)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--consistency", type=int, default=1)
    p.add_argument("--candidate-policy", choices=["answer", "veto", "off"], default="answer")
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

    data_fp = hashlib.sha256(questions_file.read_bytes()).hexdigest()
    rulebook_file = args.data / "rulebook.md"
    rulebook_sha = hashlib.sha256(rulebook_file.read_bytes()).hexdigest() if rulebook_file.exists() else ""
    code_hash = compute_solver_code_hash()

    solver_config = dict(
        min_confidence=args.min_confidence,
        consistency=args.consistency,
        candidate_policy=args.candidate_policy,
        seed=args.seed,
    )

    split_data = dict(
        seed=args.seed,
        groups=groups,
        fold=args.fold,
        ids=chosen,
        dataset_fingerprint=data_fp,
        rulebook_sha256=rulebook_sha,
    )
    run_config = dict(
        modes=args.modes,
        backend=args.backend,
        model=args.model,
        solver_config=solver_config,
        code_hash=code_hash,
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

    # Execute requested modes and bind each mode output to its own run_manifest.json
    for mode in args.modes:
        directory = args.out / mode
        directory.mkdir(exist_ok=True)
        run_manifest_path = directory / "run_manifest.json"
        base_mode_manifest = dict(
            mode=mode,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            seed=args.seed,
            fold=args.fold,
            ids=chosen,
            dataset_fingerprint=data_fp,
            rulebook_sha256=rulebook_sha,
            backend=args.backend,
            model=args.model,
            solver_config=solver_config,
            code_hash=code_hash,
            cleaner_url=args.cleaner_url if mode == "rules+llm" else None,
        )
        run_manifest_path.write_text(json.dumps(base_mode_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        pred_path = directory / "submission.jsonl"
        cmd = [
            sys.executable, "-m", "datamate.run",
            "--data", str(args.data),
            "--out", str(pred_path),
            "--backend", args.backend,
            "--cleaner", mode,
            "--only-ids", str(ids_path),
            "--workers", "1",
            "--min-confidence", str(args.min_confidence),
            "--consistency", str(args.consistency),
            "--candidate-policy", args.candidate_policy,
            "--seed", str(args.seed),
            "--fresh",
            "--cleaner-url", args.cleaner_url,
            "--cleaner-budget-sec", str(args.cleaner_budget_sec),
            "--cleaner-timeout-sec", str(args.cleaner_timeout_sec),
            "--cleaner-max-calls", str(args.cleaner_max_calls),
        ]
        if args.model:
            cmd += ["--model", args.model]

        env = os.environ.copy()
        if mode == "rules+profile":
            env["DATAMATE_PROFILE_GROUNDING"] = "1"

        try:
            subprocess.run(cmd, check=True, env=env)
            if pred_path.exists():
                sub_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
                preds_written = [
                    str(json.loads(line)["id"])
                    for line in pred_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                is_complete = set(chosen).issubset(set(preds_written))
                base_mode_manifest.update(
                    status="completed" if is_complete else "incomplete",
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    submission_sha256=sub_sha,
                    completed_ids=sorted(set(preds_written)),
                )
            else:
                base_mode_manifest.update(status="failed", error="submission.jsonl not created")
        except Exception as exc:
            base_mode_manifest.update(status="failed", error=str(exc))
            run_manifest_path.write_text(json.dumps(base_mode_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        run_manifest_path.write_text(json.dumps(base_mode_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Evaluate all available modes in output directory (including previously run modes)
    all_available_modes = [m for m in BENCHMARK_MODES if (args.out / m).is_dir() and ((args.out / m / "submission.jsonl").exists() or (args.out / m / "run_manifest.json").exists())]

    result: dict[str, dict] = {}
    mode_evaluations: dict[str, dict] = {}
    mode_traces: dict[str, dict[str, dict]] = {}
    verified_runs: dict[str, dict] = {}

    for mode in all_available_modes:
        directory = args.out / mode
        pred_file = directory / "submission.jsonl"
        ok_out, out_reason, run_meta = verify_mode_output(directory, split_data)
        verified_runs[mode] = run_meta

        pred = {}
        if pred_file.exists():
            try:
                pred = {
                    r["id"]: r
                    for r in (json.loads(s) for s in pred_file.read_text(encoding="utf-8").splitlines() if s.strip())
                }
            except Exception:
                pred = {}

        eval_res = {}
        if not ok_out:
            eval_res = {"status": "invalid_or_incomplete_output", "reason": out_reason}
        elif gold:
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
        if (directory / "logs").exists():
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
        p95 = latencies[min(len(latencies) - 1, int(latencies * 0.95) if isinstance(latencies, int) else int(len(latencies) * 0.95))] if latencies else None

        result[mode] = dict(
            output_verified=ok_out,
            verification_reason=out_reason,
            solver_backend=run_meta.get("backend", args.backend),
            solver_model=run_meta.get("model", args.model),
            metrics=eval_res,
            questions_evaluated=len(chosen) if ok_out else 0,
            questions_with_cleaner_changes=changes_count,
            question_latency_p50=round(p50, 3) if p50 is not None else None,
            question_latency_p95=round(p95, 3) if p95 is not None else None,
            llm_calls_generated=sum(st == "generated" for st in llm_statuses),
            llm_calls_cached=sum(st == "cache" for st in llm_statuses),
            llm_calls_rejected=sum(st == "rejected" for st in llm_statuses),
            llm_calls_fallback=sum(st == "fallback" for st in llm_statuses),
            rejection_reasons=dict(reject_reasons),
            warning="Historical holdout is not pristine; prior recipes used whole train. Local PA is not official BTC score."
        )

    # Compute comparative flips between pairs of all available modes
    flips = {}
    for i in range(len(all_available_modes)):
        for j in range(i + 1, len(all_available_modes)):
            m_a, m_b = all_available_modes[i], all_available_modes[j]
            pair_key = f"{m_a}_vs_{m_b}"
            run_a = verified_runs.get(m_a, {})
            run_b = verified_runs.get(m_b, {})
            compat, compat_reason = validate_comparison_compatibility(run_a, run_b, expected_split=split_data)
            if not compat:
                flips[pair_key] = dict(
                    compatible=False,
                    incompatibility_reason=compat_reason,
                    warning="PA delta and flips not evaluated due to incompatible or incomplete runs."
                )
                continue

            c_to_w, w_to_c, a_to_c, c_to_a = [], [], [], []
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

        trace_rules = mode_traces.get("rules", {}).get(qid, {}) or mode_traces.get("rules+profile", {}).get(qid, {})
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
