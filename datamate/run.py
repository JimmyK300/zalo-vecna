"""Entry point: python -m datamate.run --data /data --out /result/submission.jsonl"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .agent import build_system_prompt, solve
from .io_contract import (ABSTAIN, UNANSWERABLE, normalize_answer, SubmissionWriter, find_dataset_dir, find_questions_file, format_record,
                          load_questions, load_rulebook)
from .evaluate import match
from .filters import filter_candidate
from .graph import parse_rulebook
from .items import item_candidate, item_intent
from .metrics import metric_candidate
from .recipes import debt_candidate, debt_mentioned
from .rules import active_rules
from .workspace import WorkspaceCache
from .cleaning import clean_question, VERSION

log = logging.getLogger("datamate")


def sample_questions(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample spread evenly over businesses (round robin)."""
    rng = random.Random(seed)
    by_biz: dict[str, list[dict]] = {}
    for q in sorted(questions, key=lambda q: q["id"]):
        by_biz.setdefault(q["business_id"], []).append(q)
    for qs in by_biz.values():
        rng.shuffle(qs)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by_biz.values()):
        for b in sorted(by_biz):
            if i < len(by_biz[b]) and len(out) < n:
                out.append(by_biz[b][i])
        i += 1
    return out


def stratified_sample(questions: list[dict], n: int, seed: int, gold_path: Path) -> list[dict]:
    """Eval only: equal share per gold answer type (gold is used to pick ids, never shown to the agent).
    Within a type, spread over businesses; leftover quota goes to the biggest types."""
    gold_type = {str(json.loads(l)["id"]): json.loads(l)["type"]
                 for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        if q["id"] in gold_type:
            by_type.setdefault(gold_type[q["id"]], []).append(q)
    pools = {t: sample_questions(qs, len(qs), seed) for t, qs in by_type.items()}
    quota = {t: min(len(p), n // len(pools)) for t, p in pools.items()}
    for t in sorted(pools, key=lambda t: -len(pools[t])):
        extra = min(n - sum(quota.values()), len(pools[t]) - quota[t])
        quota[t] += max(0, extra)
    return [q for t in sorted(pools) for q in pools[t][:quota[t]]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATAMATE_DATA", "/data"))
    ap.add_argument("--out", default=os.environ.get("DATAMATE_OUT", "/result/submission.jsonl"))
    ap.add_argument("--backend", default=os.environ.get("DATAMATE_BACKEND", "anthropic"),
                    choices=["anthropic", "openai", "openrouter", "ollama", "codex", "opencode", "scripted"])
    ap.add_argument("--model", default=os.environ.get("DATAMATE_MODEL"))
    ap.add_argument("--cleaner", choices=["off", "rules", "rules+llm"], default="rules")
    ap.add_argument("--cleaner-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--cleaner-budget-sec", type=float, default=900)
    ap.add_argument("--cleaner-timeout-sec", type=float, default=8)
    ap.add_argument("--cleaner-max-calls", type=int, default=100)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DATAMATE_WORKERS", "8")))
    ap.add_argument("--min-confidence", type=float, default=float(os.environ.get("DATAMATE_MIN_CONF", "0.5")))
    ap.add_argument("--time-budget-min", type=float, default=float(os.environ.get("DATAMATE_TIME_BUDGET_MIN", "330")),
                    help="stop starting new questions after this many minutes (rule: 6h incl. loading)")
    ap.add_argument("--sample", type=int, help="only N questions, spread over businesses (debug/eval)")
    ap.add_argument("--stratify-gold", help="with --sample: answers.jsonl to balance the sample by answer type (eval only)")
    ap.add_argument("--businesses", help="comma-separated business_ids to keep")
    ap.add_argument("--exclude-ids", help="jsonl/txt file of question ids to leave out (held-out evaluation)")
    ap.add_argument("--only-ids", help="jsonl/txt file of question ids to keep (same questions across models)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing submission instead of resuming")
    ap.add_argument("--consistency", type=int, default=int(os.environ.get("DATAMATE_CONSISTENCY", "1")),
                    help="N>1: re-solve each answered question N-1 more times (sampled); abstain if any differs")
    ap.add_argument("--candidate-policy", default=os.environ.get("DATAMATE_CANDIDATE_POLICY", "answer"),
                    choices=["answer", "veto", "off"],
                    help="answer: a validated recipe answers its questions directly (default); veto: the model "
                         "answers and is abstained when it disagrees with the recipe; off: model only")
    ap.add_argument("--verify", action="store_true",
                    help="with --consistency 2: when the attempts disagree, a third attempt sees both candidates, "
                         "re-checks them with a query and must match one of them (else abstain)")
    ap.add_argument("--verify-new", action="store_true",
                    help="with --verify: the verifier may give its own answer when both candidates are wrong; "
                         "it counts only if one more independent attempt reaches the same answer")
    ap.add_argument("--thermal-guard", action="store_true",
                    help="local AMD GPU: slow down >=88C, pause >=92C until <=82C, kill Ollama and stop >=95C")
    ap.add_argument("--temp-limits", default="88,92,82,95", help="slow,pause,resume,stop in C")
    args = ap.parse_args()

    out = Path(args.out)
    log_dir = out.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "run.log", encoding="utf-8")])

    t_start = time.time()
    deadline = t_start + args.time_budget_min * 60
    data = Path(args.data)
    questions = load_questions(find_questions_file(data))
    if args.businesses:
        keep = set(args.businesses.split(","))
        questions = [q for q in questions if q["business_id"] in keep]
    if args.exclude_ids:
        import re as _re
        skip = set(_re.findall(r"q_[0-9a-f]+", Path(args.exclude_ids).read_text(encoding="utf-8")))
        questions = [q for q in questions if q["id"] not in skip]
    if args.only_ids:
        import re as _re
        keep_ids = set(_re.findall(r"q_[0-9a-f]+", Path(args.only_ids).read_text(encoding="utf-8")))
        questions = [q for q in questions if q["id"] in keep_ids]
    if args.sample and args.stratify_gold:
        questions = stratified_sample(questions, args.sample, args.seed, Path(args.stratify_gold))
    elif args.sample:
        questions = sample_questions(questions, args.sample, args.seed)
    system = build_system_prompt(load_rulebook(data))
    cal_path = data / "calendar.csv"
    shared = {"calendar": pd.read_csv(cal_path)} if cal_path.exists() else {}
    log.info("questions=%d data=%s backend=%s model=%s rulebook=%d chars",
             len(questions), data, args.backend, args.model, len(system))

    if args.fresh and out.exists():
        out.unlink()
    writer = SubmissionWriter(out)
    todo = [q for q in questions if q["id"] not in writer.done]
    # group by business so each workspace is loaded once and reused while hot
    todo.sort(key=lambda q: (q["business_id"], q["id"]))
    log.info("resuming: %d already done, %d to go", len(questions) - len(todo), len(todo))
    rulebook = load_rulebook(data)
    rule_rows = parse_rulebook(rulebook)
    rules = active_rules(rulebook)
    log.info("rule helpers on: %s", sorted(rules))
    cache = WorkspaceCache(shared, rules, cleaner=args.cleaner != "off")
    cleaner_llm = None
    if args.cleaner == "rules+llm":
        if args.backend not in ("openai", "ollama") or not args.model:
            ap.error("rules+llm requires an explicit --model and a local openai/ollama solver backend")
        from urllib.parse import urlparse
        solver_url = (os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1") if args.backend == "ollama"
                      else os.environ.get("OPENAI_BASE_URL", ""))
        if urlparse(solver_url).hostname not in ("localhost", "127.0.0.1", "::1"):
            ap.error("rules+llm requires a local solver URL too; set OPENAI_BASE_URL/OLLAMA_BASE_URL")
        if args.cleaner_budget_sec < 0 or args.cleaner_timeout_sec <= 0 or args.cleaner_max_calls < 0:
            ap.error("cleaner budgets must be nonnegative; timeout must be positive")
        from .cleaner_llm import CleanerLLM
        cleaner_llm = CleanerLLM(args.model, args.cleaner_url, out.parent / "cleaner_cache",
            args.cleaner_budget_sec, args.cleaner_timeout_sec, args.cleaner_max_calls)
    shop_lock = threading.Lock()
    manifest_path = out.parent / "cleaner_manifest.json"
    manifest = dict(version=VERSION, baseline="7a2a5e65f6bbbd89318563ac49fedf571101701e",
                    cleaner=args.cleaner, backend=args.backend, model=args.model, rulebook_sha256=hashlib.sha256(rulebook.encode()).hexdigest(),
                    endpoint=args.cleaner_url, budget=args.cleaner_budget_sec, timeout=args.cleaner_timeout_sec,
                    max_calls=args.cleaner_max_calls)
    data_hash = hashlib.sha256()
    for path in sorted([data / "questions.jsonl", data / "calendar.csv", *list((data / "data").rglob("*"))]):
        if path.is_file():
            data_hash.update(str(path.relative_to(data)).encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    data_hash.update(chunk)
    manifest["data_hash"] = data_hash.hexdigest()
    manifest["cleaner_code_hash"] = hashlib.sha256(b"".join(Path(__file__).with_name(n).read_bytes()
        for n in ("cleaning.py", "cleaner_llm.py", "workspace.py", "agent.py", "run.py"))).hexdigest()
    manifest["inference"] = dict(min_confidence=args.min_confidence, consistency=args.consistency,
        verify=args.verify, verify_new=args.verify_new, candidate_policy=args.candidate_policy, seed=args.seed)
    if manifest_path.exists() and not args.fresh:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cleaner config changed: use a new --out directory (do not mix predictions)")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.thermal_guard:
        from . import thermal
        slow, soft, resume, hard = map(float, args.temp_limits.split(","))
        if args.fresh:
            (log_dir / "gpu_temps.csv").unlink(missing_ok=True)
        model = args.model or ""
        thermal.GUARD = thermal.ThermalGuard(soft=soft, resume=resume, hard=hard, slow=slow,
                                             csv=log_dir / "gpu_temps.csv", interval=0.25,
                                             on_trip=lambda: thermal.unload_ollama(model)).start()
        log.info("thermal guard on: %s, now %s", args.temp_limits, thermal.GUARD.gpu.read())
    halted: list[str] = []  # set once the backend is unusable (e.g. quota hit): stop burning questions

    def work(q: dict) -> dict:
        q = {**q, "_work_started": time.monotonic()}
        if halted:
            return {"type": "skipped", "answer": None, "note": halted[0]}
        if time.time() > deadline:
            res = {"type": ABSTAIN, "answer": None, "reason": None, "trace": [], "note": "time budget exhausted"}
        else:
            ws = cache.get(find_dataset_dir(data, q["business_id"]))
            q = dict(q)
            if args.cleaner != "off":
                with shop_lock:
                    if not hasattr(ws, "cleaner_context"):
                        context = hashlib.sha256(rulebook.encode())
                        for path in sorted(ws.root.rglob("*")):
                            if path.is_file():
                                context.update(str(path.relative_to(ws.root)).encode())
                                with path.open("rb") as stream:
                                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                        context.update(chunk)
                        ws.cleaner_context = context.hexdigest()
                        ws.cleaner_entities = []
                        for table in ws.tables.values():
                            for role, column in table.roles.items():
                                if role in ("customer", "customer_code", "item", "item_code", "item_group") and column in table.df:
                                    ws.cleaner_entities.extend(str(v) for v in table.df[column].dropna().unique()[:500])
                        ws.cleaner_entities = sorted(set(ws.cleaner_entities))
                    report_path = log_dir / f"shop_{_safe(q['business_id'])}_cleaning.json"
                    if not getattr(ws, "cleaner_report_written", False):
                        report_path.write_text(json.dumps(ws.cleaning_reports, ensure_ascii=False, indent=2), encoding="utf-8")
                        ws.cleaner_report_written = True
                q["cleaner"] = clean_question(q["question"], ws.cleaner_entities).to_dict()
            cand = None if args.candidate_policy == "off" else recipe_candidate(q, ws)
            if cand is not None and args.candidate_policy == "answer":
                # a validated deterministic recipe covers this question: answer from it (no LLM call)
                return finish(q, {"type": cand[0], "answer": cand[1], "reason": "recipe", "trace": [],
                                  "final_call": {"recipe": True}, "seconds": 0, "tokens": {}, "recipe_candidate": list(cand)})
            if cleaner_llm is not None:
                with shop_lock:
                    if not hasattr(ws, "cleaner_shop_result"):
                        ws.cleaner_shop_result = cleaner_llm.shop(ws, ws.cleaner_context, deadline - 60)
                q["shop_cleaner_hints"], q["shop_cleaner_report"] = ws.cleaner_shop_result
                if q["cleaner"]["ambiguous"]:
                    from .cleaning import QuestionCleanResult
                    result = cleaner_llm.question(QuestionCleanResult(**q["cleaner"]), ws.cleaner_context, deadline - 60)
                    q["cleaner"] = result.to_dict()
            res = solve(q, ws, system, args.backend, args.model, args.min_confidence, deadline, rule_rows)
            if args.consistency > 1 and res["type"] not in (ABSTAIN,) and res.get("final_call"):
                # candidates disagree -> abstain: a second independent attempt must reach the same answer
                for v in range(1, args.consistency):
                    alt = solve(q, ws, system, args.backend, args.model, args.min_confidence, deadline,
                                rule_rows, variant=v)
                    res.setdefault("alternatives", []).append({"type": alt["type"], "answer": alt["answer"]})
                    add_cost(res, alt)
                    if not same_answer(res, alt):
                        res["disagreement"] = {"first": [res["type"], res["answer"]], "other": [alt["type"], alt["answer"]]}
                        if args.verify:
                            # CHESS-style: one more attempt sees both candidates and how each was computed;
                            # it must re-check with a query and its answer must match one of them
                            ver = solve(q, ws, system, args.backend, args.model, args.min_confidence, deadline,
                                        rule_rows, note=verify_note(res, alt, args.verify_new))
                            res["verify"] = {"type": ver["type"], "answer": ver["answer"], "reason": ver.get("reason"),
                                             "cost_usd": (ver.get("tokens") or {}).get("cost_usd", 0)}
                            add_cost(res, ver)
                            if ver["type"] != ABSTAIN and (same_answer(ver, res) or same_answer(ver, alt)):
                                res["type"], res["answer"] = ver["type"], ver["answer"]
                                break
                            if args.verify_new and ver["type"] != ABSTAIN:
                                # the verifier's own answer counts only if a fresh independent attempt agrees
                                conf = solve(q, ws, system, args.backend, args.model, args.min_confidence, deadline,
                                             rule_rows, variant=args.consistency)
                                add_cost(res, conf)
                                res["verify"]["confirm"] = {"type": conf["type"], "answer": conf["answer"]}
                                if same_answer(ver, conf):
                                    res["type"], res["answer"] = ver["type"], ver["answer"]
                                    break
                        res["type"], res["answer"] = ABSTAIN, None
                        break
            if res["type"] != ABSTAIN and cand is not None and args.candidate_policy == "veto":
                # a deterministic recipe candidate must agree with the model
                res["recipe_candidate"] = list(cand)
                if not same_answer(res, {"type": cand[0], "answer": cand[1]}):
                    res["disagreement"] = {"first": [res["type"], res["answer"]], "recipe": list(cand)}
                    res["type"], res["answer"] = ABSTAIN, None
        return finish(q, res)

    def finish(q: dict, res: dict) -> dict:
        res["total_seconds"] = round(time.monotonic() - q["_work_started"], 3)
        err = next((t["error"] for t in res["trace"] if "error" in t), None) if not res.get("final_call") else None
        if err:
            # backend failure, not a model decision: leave it out so a resumed run retries it
            # (finalize() still writes abstain for it in this run's file)
            if any(k in err.lower() for k in ("usage limit", "rate limit", "quota", "insufficient", "free-limit", "thermal guard")):
                halted.append(err[:300])
            log.warning("%s backend error, will retry on resume: %s", q["id"], err[:300])
            res["type"] = "error"
        else:
            writer.write(format_record(q["id"], res["type"], res["answer"], res.get("reason")))
        (log_dir / f"{_safe(q['id'])}.json").write_text(
            json.dumps({"id": q["id"], "business_id": q["business_id"], "as_of_date": q["as_of_date"],
                        "question": q["question"], "cleaner": q.get("cleaner"),
                        "shop_cleaner_hints": q.get("shop_cleaner_hints"),
                        "shop_cleaner_report": q.get("shop_cleaner_report"), **res}, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        return res

    counts: dict[str, int] = {}
    spent = [0.0]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, q): q for q in todo}
        for i, f in enumerate(as_completed(futures), 1):
            q = futures[f]
            try:
                res = f.result()
            except Exception as e:  # noqa: BLE001
                log.exception("question %s crashed: %s", q["id"], e)
                writer.write(format_record(q["id"], ABSTAIN))
                res = {"type": ABSTAIN, "answer": None}
            counts[res["type"]] = counts.get(res["type"], 0) + 1
            spent[0] += (res.get("tokens") or {}).get("cost_usd", 0) or 0
            log.info("[%d/%d] %s -> %s %s (%ss)", i, len(todo), q["id"], res["type"],
                     json.dumps(res["answer"], ensure_ascii=False)[:120], res.get("seconds"))

    if halted:
        log.error("backend halted, rerun without --fresh to resume: %s", halted[0])
    if args.thermal_guard:
        from . import thermal
        g = thermal.GUARD
        log.info("GPU max %.0fC, paused %.0fs, tripped=%s", g.max_seen, g.paused_s, g.tripped.is_set())
        g.stop()
    writer.finalize([q["id"] for q in questions])
    if cleaner_llm is not None:
        (out.parent / "cleaner_usage.json").write_text(json.dumps(dict(calls=cleaner_llm.calls,
            seconds=cleaner_llm.spent, budget=cleaner_llm.budget), indent=2), encoding="utf-8")
    log.info("done in %.1f min: %s -> %s (API cost reported: $%.4f)", (time.time() - t_start) / 60, counts, out, spent[0])


def recipe_candidate(q: dict, ws) -> tuple[str, object] | None:
    """The deterministic answer of the first recipe that recognises the question (debt ledger,
    order metrics, item recipes), or None. Every recipe is validated on the whole train split."""
    try:
        ledger = getattr(ws, "debt", None)
        debt_rule_off = ledger is not None and ws.rules is not None and "debt_ledger" not in ws.rules
        c = None if debt_rule_off else debt_candidate(q["question"], q["as_of_date"], ledger, debt_mentioned(ws.docs))
        if c is None:
            c = metric_candidate(q["question"], q["as_of_date"], ws)
        if c is None:
            c = filter_candidate(q["question"], q["as_of_date"], ws)
        if c is None and item_intent(q["question"], q["as_of_date"], ws):
            c = item_candidate(q["question"], q["as_of_date"], ws)
    except Exception:  # noqa: BLE001 - a candidate is optional
        return None
    if c is not None and c[0] not in ("unanswerable",):
        try:
            c = (c[0], normalize_answer(c[0], c[1]))
        except (ValueError, TypeError):
            return None
    return c


def same_answer(a: dict, b: dict) -> bool:
    """Do two attempts agree? Uses the scorer's own matching (aliases aside)."""
    if a["type"] != b["type"]:
        return False
    if a["type"] == UNANSWERABLE:
        return True
    try:
        return match(a["type"], b["answer"], a["answer"], None)
    except Exception:  # noqa: BLE001 - malformed answer: treat as disagreement
        return False


def add_cost(res: dict, other: dict) -> None:
    """Fold another attempt's token usage into the record, so the run's cost total is the real bill."""
    t, o = res.setdefault("tokens", {}), other.get("tokens") or {}
    for k in ("input", "output", "cached", "cost_usd"):
        t[k] = (t.get(k) or 0) + (o.get(k) or 0)


def _computations(res: dict, k: int = 2) -> str:
    calls = [c for step in res.get("trace", []) for c in step.get("tool_calls", []) if c[0] in ("run_sql", "run_python")]
    out = []
    for name, args in calls[-k:]:
        code = args.get("query") or args.get("code") or ""
        out.append(f"    {name}: {code.strip()[:700]}")
    return "\n".join(out) or "    (no computation recorded)"


def verify_note(a: dict, b: dict, allow_new: bool = False) -> str:
    ending = ("If you find that BOTH are wrong, compute the correct answer yourself and give that. "
              "If you cannot verify any answer, answer 'abstain'." if allow_new else
              "If neither can be verified, answer 'abstain'.")
    return ("# Verification\nTwo independent attempts at this question disagree:\n"
            f"A: {a['type']} {json.dumps(a['answer'], ensure_ascii=False)[:300]}\n  last computations:\n{_computations(a)}\n"
            f"B: {b['type']} {json.dumps(b['answer'], ensure_ascii=False)[:300]}\n  last computations:\n{_computations(b)}\n"
            "Find where they differ (filter, date range, amount column, status rule, join, unit). Check the rulebook "
            "and the data with a query, then call final_answer with the answer you verified. " + ending)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:100]


if __name__ == "__main__":
    main()
