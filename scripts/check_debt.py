"""Check the debt recipe against train answers (gold only tests the code; never shown to the agent).
Covers 'total debt as of X' (number) and 'top N debtors as of X' (table)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.evaluate import match  # noqa: E402
from datamate.recipes import build_debt_ledger  # noqa: E402
from datamate.rules import active_rules  # noqa: E402
from datamate.semantics import strip_accents  # noqa: E402
from datamate.workspace import load_workspace  # noqa: E402

DATA = Path("devkit/train")
rules = active_rules((DATA / "rulebook.md").read_text(encoding="utf-8"))
qs = [json.loads(l) for l in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
gold = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "answers.jsonl").read_text(encoding="utf-8").splitlines()}
cache, res, why = {}, Counter(), Counter()
only = sys.argv[1] if len(sys.argv) > 1 else None
for q in qs:
    s = strip_accents(q["question"])
    g = gold[q["id"]]
    if not re.search(r"\bno\b|cong no", s) or re.search(r"no them|tra no|thu no|tra duoc|tuoi", s):
        continue
    if g["type"] not in ("number", "table", "unanswerable") or (g["type"] == "number" and not re.search(r"tong", s)):
        continue
    if only and q["business_id"] != only:
        continue
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    as_of = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else q["as_of_date"]
    b = q["business_id"]
    if b not in cache:
        ws = load_workspace(DATA / "data" / b, {"calendar": pd.read_csv(DATA / "calendar.csv")}, rules)
        cache[b] = build_debt_ledger(ws)
    led = cache[b]
    if led is None:
        pred = ("unanswerable", None)
    elif g["type"] == "table" or re.search(r"\bbang\b|\btop\b|khach nao|\bai\b", s):
        n = re.search(r"(?:top\s?|\b)(\d+)\s?khach", s)
        pred = ("table", led.debt_table(as_of, int(n.group(1)) if n else None))
    else:
        age = re.search(r"(?:qua|hon|tren)\s?(\d+)\s?ngay", s)
        pred = ("number", led.debt_total(as_of, older_than_days=int(age.group(1)) if age else None))
    ok = pred[0] == g["type"] and (g["type"] == "unanswerable" or match(g["type"], pred[1], g["answer"], g.get("alias")))
    res[(g["type"], ok)] += 1
    if not ok:
        why[b] += 1
        print(f"BAD {b} {q['id']} {g['type']}: pred={json.dumps(pred, ensure_ascii=False)[:160]} gold={json.dumps(g['answer'], ensure_ascii=False)[:160]}")
print(dict(res))
print("misses by shop:", dict(why))
for b, led in cache.items():
    if led is not None and (only or why.get(b)):
        print(b, led.info)
