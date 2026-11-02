import argparse
import datetime as dt
import json
import hashlib
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .agent import solve
from .client import LocalModel
from .data import ReadOnlySQL, build_database
from .normalized import build_normalized
from .preflight import inspect_hardware, write_report
from .scoring import score, summarize
from .suite import load_suite, make_suite, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = ROOT.parent / "train"
DEFAULT_SUITE = ROOT / "benchmarks" / "train10.json"


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str), flush=True)


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_fingerprints():
    paths=sorted((ROOT/'datamate_bench').glob('*.py'))+sorted((ROOT/'datamate_bench').glob('*.json'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def safe_business(split, business_id):
    base = (Path(split) / "data").resolve()
    business = (base / business_id).resolve()
    if business.parent != base or not business.is_dir():
        raise ValueError("Business must be a direct child directory of split/data")
    return business


def write_submission(path, entry):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def report_markdown(summary, details):
    scorecard = summary["score"]
    lines = ["# DataMate local benchmark", "", f"Model: `{summary['model']}`", "",
             f"Đúng {scorecard['correct']}, sai {scorecard['wrong']}, abstain {scorecard['abstain']}; PA = {scorecard['pa']:.3f}.",
             "", "Đây là smoke benchmark phát triển, không ước lượng đáng tin cậy điểm public/private.",
             "", f"Tổng thời gian: {summary['wall_seconds']:.1f}s; đọc/cache dữ liệu: {summary['ingestion_seconds']:.1f}s; chuẩn hóa: {summary['normalization_seconds']:.1f}s; giải: {summary['question_seconds_total']:.1f}s.",
             "", "Thời gian khởi động server/tải weights nằm ngoài phép đo này. Xem speed.json/server log để đo riêng.",
             "", "| Câu | Type nhãn | Điểm | Giây giải | Lượt model |", "|---|---|---:|---:|---:|"]
    for d in details:
        lines.append(f"| {d['id']} | {d['reference']['type']} | {d['score']} | {d['question_seconds']:.1f} | {d['model_calls']} |")
    lines += ["", "Các giá trị tham chiếu chỉ được đọc để chấm sau khi inference kết thúc.",
              "Trạng thái CPU/GPU phải đối chiếu preflight và log server; endpoint HTTP không tự chứng minh GPU offload."]
    return "\n".join(lines) + "\n"


def run_benchmark(args):
    start = time.perf_counter()
    split = args.train.resolve()
    manifest, questions = load_suite(split, args.suite)
    if args.limit:
        questions = questions[:args.limit]
    out = args.output or ROOT / "artifacts" / ("run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Output directory must be empty; use a new directory for each run")
    config = {"version": __version__, "git_revision": git_revision(),
              "source_sha256": source_fingerprints(),
              "suite_purpose": manifest.get('purpose'),
              "input_fingerprints": {k: v for k, v in manifest.items() if k.endswith("sha256")},
              "endpoint": args.endpoint, "model_requested": args.model,
              "question_timeout": args.question_timeout, "max_steps": args.max_steps,
              "max_tokens": args.max_tokens, "seed": args.seed,
              "questions": [q["id"] for q in questions]}
    # Preserve the actual quantization/runtime identities instead of just an HTTP alias.
    for label, path in {
        "model_artifact": ROOT / "models" / "model_manifest.json",
        "runtime_artifact": ROOT / "tools" / "llama.cpp" / "runtime_manifest.json",
        "server_session": ROOT / "artifacts" / "server_session.json",
    }.items():
        if path.exists():
            config[label] = json.loads(path.read_text(encoding="utf-8-sig"))
    write_report(out / "config.json", config)
    hardware = inspect_hardware()
    write_report(out / "preflight.json", hardware)
    if args.require_cuda and not hardware["cuda_available"]:
        raise RuntimeError("CUDA unavailable. See preflight.json; do not label this as an RTX GPU benchmark.")
    client = LocalModel(args.endpoint, args.model, args.max_tokens, args.seed)
    # Fail early, before loading large spreadsheets, if no model is listening.
    client.connect()
    rulebook = (split / "rulebook.md").read_text(encoding="utf-8")
    prepared, pending = {}, []
    for index, q in enumerate(questions, 1):
        business_id = q["business_id"]
        if business_id not in prepared:
            business = safe_business(split, business_id)
            print(f"[{index}/{len(questions)}] Loading {business_id}...", flush=True)
            db, schema, ingest = build_database(business, split / "calendar.csv", args.cache)
            readme = (business / "README.md").read_text(encoding="utf-8")
            db, schema, typed_timing = build_normalized(db, schema, readme, args.cache / "typed")
            ingest.update(typed_timing)
            prepared[business_id] = (db, schema, ingest, readme)
        db, schema, ingest, readme = prepared[business_id]
        sql = ReadOnlySQL(db, seconds=args.sql_timeout)
        before = len(client.records)
        try:
            result = solve(q, readme, rulebook, schema, sql, client, args.max_steps, args.question_timeout)
        finally:
            sql.close()
        calls = client.records[before:]
        result.update({"id": q["id"], "business_id": business_id, "question": q["question"],
                       "as_of_date": q["as_of_date"], "data_fingerprint": schema["fingerprint"],
                       "model_calls": len(calls), "requests": calls})
        pending.append(result)
        write_submission(out / "submission.jsonl", {"id": q["id"], **result["prediction"]})
        write_report(out / "logs" / f"{q['id']}.json", result)
        print(f"[{index}/{len(questions)}] {q['id']}: {result['prediction']['type']} "
              f"({result['question_seconds']:.1f}s, {len(calls)} model calls)", flush=True)
    # Labels never enter the model/tool environment. Load them after all questions.
    references = {r["id"]: r for r in read_jsonl(split / "answers.jsonl")}
    for result in pending:
        result["reference"] = references[result["id"]]
        result["score"] = score(result["prediction"], result["reference"])
    durations = [r["question_seconds"] for r in pending]
    summary = {"model": client.model, "score": summarize([r["score"] for r in pending]),
               "wall_seconds": time.perf_counter() - start,
               "ingestion_seconds": sum(v[2]["ingest_seconds"] for v in prepared.values()),
               "normalization_seconds": sum(v[2].get("normalization_seconds",0) for v in prepared.values()),
               "question_seconds_total": sum(durations),
               "question_seconds_median": statistics.median(durations) if durations else None,
               "model_calls": len(client.records), "hardware_cuda_visible": hardware["cuda_available"],
               "gpu_execution_verified": False,
               "note": "Confirm GPU offload from server logs. No GPU speed claim is inferred from HTTP.",
               "ingestion": {b: p[2] for b, p in prepared.items()}}
    write_report(out / "scores.json", pending)
    write_report(out / "summary.json", summary)
    (out / "report.md").write_text(report_markdown(summary, pending), encoding="utf-8")
    print_json({"output": str(out), **summary})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fresh DataMate local benchmark (no old solvers)")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Inspect CUDA without loading weights")
    preflight.add_argument("--output", type=Path, default=ROOT / "artifacts" / "preflight.json")
    preflight.add_argument("--require-cuda", action="store_true")
    suite = commands.add_parser("make-suite", help="Select 10 train questions across all 8 reference types")
    suite.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    suite.add_argument("--output", type=Path, default=DEFAULT_SUITE)
    suite.add_argument("--seed", type=int, default=20261102)
    suite.add_argument("--size", type=int, default=10)
    suite.add_argument("--exclude-suite", type=Path)
    suite.add_argument("--exclude-businesses", action="store_true")
    profile = commands.add_parser("profile", help="Time real ingestion only; no model accuracy claim")
    profile.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    profile.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    profile.add_argument("--cache", type=Path, default=ROOT / ".cache" / "ingestion")
    profile.add_argument("--output", type=Path, default=ROOT / "artifacts" / "ingestion.json")
    for name in ["run", "speed"]:
        cmd = commands.add_parser(name, help="Run train QA" if name == "run" else "Measure local model HTTP generation")
        cmd.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
        cmd.add_argument("--model")
        cmd.add_argument("--max-tokens", type=int, default=1024 if name == "run" else 256)
        cmd.add_argument("--seed", type=int, default=42)
        if name == "speed":
            cmd.add_argument("--output", type=Path, default=ROOT / "artifacts" / "speed.json")
            continue
        cmd.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
        cmd.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
        cmd.add_argument("--cache", type=Path, default=ROOT / ".cache" / "ingestion")
        cmd.add_argument("--output", type=Path)
        cmd.add_argument("--limit", type=int)
        cmd.add_argument("--max-steps", type=int, default=12)
        cmd.add_argument("--question-timeout", type=float, default=240)
        cmd.add_argument("--sql-timeout", type=float, default=5)
        cmd.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            report = inspect_hardware()
            write_report(args.output, report)
            print_json(report)
            return 2 if args.require_cuda and not report["cuda_available"] else 0
        if args.command == "make-suite":
            if args.size<1: raise ValueError("size must be positive")
            print_json(make_suite(args.train, args.output, args.seed,args.size,args.exclude_suite,args.exclude_businesses))
        elif args.command == "profile":
            _, questions = load_suite(args.train, args.suite)
            entries = {}
            for q in questions:
                if q["business_id"] in entries:
                    continue
                business = safe_business(args.train, q["business_id"])
                _, schema, timing = build_database(business, args.train / "calendar.csv", args.cache)
                entries[q["business_id"]] = {**timing, "fingerprint": schema["fingerprint"]}
                print(q["business_id"], json.dumps(timing), flush=True)
            write_report(args.output, {"businesses": entries,
                                      "total_ingest_seconds": sum(v["ingest_seconds"] for v in entries.values()),
                                      "model_was_run": False})
        elif args.command == "speed":
            model = LocalModel(args.endpoint, args.model, args.max_tokens, args.seed)
            model.connect()
            messages = [{"role": "user", "content": "Write a short Python function that sums order line amounts, ignoring cancelled orders. Explain briefly."}]
            runs = []
            for warmup in [True, False, False, False]:
                _, timing = model.chat(messages, timeout=240, json_mode=False, require_complete=False)
                tokens = timing["usage"].get("completion_tokens")
                runs.append({"warmup": warmup, **timing,
                             "end_to_end_output_tokens_per_second": tokens / timing["request_seconds"] if tokens else None})
            report = {"model": model.model, "runs": runs,
                      "note": "Request time includes prefill/network/decode. Server timings, when present, separate decode. Warmup excluded from interpretation. Not a QA accuracy run."}
            write_report(args.output, report)
            print_json(report)
        elif args.command == "run":
            if (args.limit is not None and args.limit < 1) or args.max_steps < 1 or args.question_timeout <= 0:
                raise ValueError("limit/max_steps/time budget must be positive")
            run_benchmark(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
