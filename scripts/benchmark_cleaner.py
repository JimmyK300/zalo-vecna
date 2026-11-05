"""Paired cleaner runs on a frozen business split. No gold is sent to inference."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.evaluate import score


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
    args = p.parse_args()
    if args.fold == "audit" and not args.open_audit:
        p.error("freeze configuration first; --open-audit explicitly records holdout access")
    questions = [json.loads(s) for s in (args.data / "questions.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    businesses = sorted({r["business_id"] for r in questions})
    random.Random(args.seed).shuffle(businesses)
    if len(businesses) != 30:
        p.error("Expected the organizer train set with 30 businesses, not public/private test")
    groups = dict(train=businesses[:20], dev=businesses[20:25], audit=businesses[25:])
    chosen = sorted(r["id"] for r in questions if r["business_id"] in groups[args.fold])
    if args.limit:
        random.Random(args.seed).shuffle(chosen)
        chosen = sorted(chosen[:args.limit])
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(seed=args.seed, groups=groups, fold=args.fold, ids=chosen, backend=args.backend, model=args.model,
                  question_hash=hashlib.sha256((args.data / "questions.jsonl").read_bytes()).hexdigest(),
                  audit_opened=args.fold == "audit", cleaner_url=args.cleaner_url)
    manifest = args.out / "split.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        p.error("Frozen split/backend changed; use a new output directory")
    manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    ids = args.out / "ids.txt"
    ids.write_text("\n".join(chosen), encoding="utf-8")
    # Load labels only for scoring after subprocess inference; never create per-question prompts from gold.
    result = {}
    for mode in args.modes:
        directory = args.out / mode
        directory.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "datamate.run", "--data", str(args.data), "--out", str(directory / "submission.jsonl"),
               "--backend", args.backend, "--cleaner", mode, "--only-ids", str(ids), "--workers", "1", "--cleaner-url", args.cleaner_url]
        if args.model:
            cmd += ["--model", args.model]
        start = time.monotonic()
        subprocess.run(cmd, check=True)
        elapsed = time.monotonic() - start
        gold = {r["id"]: r for r in (json.loads(s) for s in (args.data / "answers.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()) if r["id"] in chosen}
        pred = {r["id"]: r for r in (json.loads(s) for s in (directory / "submission.jsonl").read_text(encoding="utf-8").splitlines() if s.strip())}
        metrics = score(pred, gold, only_pred=False)
        metrics.pop("rows")
        traces = [json.loads(f.read_text(encoding="utf-8")) for f in (directory / "logs").glob("q_*.json")]
        llm = [(r.get("cleaner") or {}).get("llm", {}) for r in traces]
        latencies = sorted(r.get("total_seconds", r.get("seconds", 0)) for r in traces)
        result[mode] = dict(metrics=metrics, wall_seconds=elapsed,
            question_p95=latencies[min(len(latencies)-1, int(len(latencies)*.95))] if latencies else None,
            llm_generated=sum(x.get("status") == "generated" for x in llm),
            llm_rejected=sum(x.get("status") == "rejected" for x in llm),
            warning="Historical holdout is not pristine; prior recipes used whole train. Timings include resume if reused.")
        (args.out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
