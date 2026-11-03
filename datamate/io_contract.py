"""Official DataMate I/O contract (from the devkit + rulebook §9).

/data/
  questions.jsonl   {"id", "business_id", "question", "as_of_date"}
  rulebook.md       scoring + business rules (differs between train / public / private!)
  calendar.csv      date, weekday, lunar_day, lunar_month, lunar_year, lunar_leap, holiday
  data/<business_id>/  xlsx / csv / README.md

submission.jsonl lines: {"id", "type", "answer"}
  types: number, percent, date, boolean, string, list, table, unanswerable (+ "reason"), abstain
"""
from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

ANSWER_TYPES = ("number", "percent", "date", "boolean", "string", "list", "table")
ABSTAIN = "abstain"
UNANSWERABLE = "unanswerable"


def find_questions_file(data_dir: Path) -> Path:
    for p in (data_dir / "questions.jsonl", *sorted(data_dir.glob("*/questions.jsonl"))):
        if p.exists():
            return p
    raise FileNotFoundError(f"No questions.jsonl under {data_dir}")


def load_questions(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append({"id": str(r["id"]), "question": r["question"], "business_id": r.get("business_id"),
                        "as_of_date": r.get("as_of_date"), "raw": r})
    return out


def load_rulebook(data_dir: Path) -> str:
    p = data_dir / "rulebook.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def find_dataset_dir(data_dir: Path, business_id: str | None) -> Path:
    if business_id:
        for base in (data_dir / "data", data_dir, data_dir / "datasets"):
            if (base / business_id).is_dir():
                return base / business_id
    raise FileNotFoundError(f"business folder for {business_id!r} not found under {data_dir}")


def _num(v):
    if isinstance(v, bool):
        raise ValueError("bool is not a number")
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        x = float(str(v).replace("%", "").replace(" ", "").replace(",", ""))
    if not math.isfinite(x):
        raise ValueError("non-finite")
    return int(x) if x == int(x) and abs(x) < 1e15 else x


def normalize_answer(typ: str, answer):
    """Coerce the model's answer into the exact JSON shape of the type (rulebook §9).
    Raises ValueError when impossible -> caller abstains."""
    if typ == "number":
        return _num(answer)
    if typ == "percent":
        return round(float(_num(answer)), 2)
    if typ == "date":
        s = str(answer).strip()
        if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", s):
            raise ValueError(f"bad date {s!r}")
        return s
    if typ == "boolean":
        if isinstance(answer, bool):
            return answer
        if str(answer).strip().lower() in ("true", "false"):
            return str(answer).strip().lower() == "true"
        raise ValueError("bad boolean")
    if typ == "string":
        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("bad string")
        return answer.strip()
    if typ == "list":
        if not isinstance(answer, list):
            answer = [answer]
        return [_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for v in answer]
    if typ == "table":
        if not (isinstance(answer, dict) and isinstance(answer.get("columns"), list)
                and isinstance(answer.get("rows"), list)):
            raise ValueError("bad table")
        rows = [[_num(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else c for c in r]
                for r in answer["rows"]]
        return {"columns": answer["columns"], "rows": rows}
    raise ValueError(f"unknown type {typ}")


def format_record(qid: str, typ: str, answer=None, reason: str | None = None) -> dict:
    if typ == ABSTAIN:
        return {"id": qid, "type": ABSTAIN}
    if typ == UNANSWERABLE:
        return {"id": qid, "type": UNANSWERABLE, "reason": str(reason or "")}
    return {"id": qid, "type": typ, "answer": answer}


class SubmissionWriter:
    """Appends one line per finished question immediately (so a timeout keeps
    what is done), supports resume, and rewrites in input order at the end."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.done: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    # abstains are retried on resume (backend errors / time-outs are written as abstain)
                    if rec.get("type") != ABSTAIN:
                        self.done[str(rec["id"])] = rec

    def write(self, record: dict) -> None:
        with self._lock:
            self.done[record["id"]] = record
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def finalize(self, order: list[str]) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for qid in order:
                    rec = self.done.get(qid) or format_record(qid, ABSTAIN)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.replace(self.path)
