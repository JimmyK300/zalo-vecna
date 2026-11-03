"""Check debt_movements ([new debt, repaid]) against train answers (gold only tests code)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.evaluate import match  # noqa: E402
from datamate.primitives import period  # noqa: E402
from datamate.recipes import build_debt_ledger, _key  # noqa: E402
from datamate.rules import active_rules  # noqa: E402
from datamate.semantics import strip_accents  # noqa: E402
from datamate.workspace import load_workspace  # noqa: E402

DATA = Path("devkit/train")
rules = active_rules((DATA / "rulebook.md").read_text(encoding="utf-8"))
qs = [json.loads(l) for l in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
gold = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "answers.jsonl").read_text(encoding="utf-8").splitlines()}
cache, res = {}, Counter()
for q in qs:
    s = strip_accents(q["question"])
    g = gold[q["id"]]
    if not re.search(r"no them|tra no|thu no|tra duoc", s):
        continue
    b = q["business_id"]
    if b not in cache:
        cache[b] = build_debt_ledger(load_workspace(DATA / "data" / b, {"calendar": pd.read_csv(DATA / "calendar.csv")}, rules))
    led = cache[b]
    m = re.search(r"thang\w?\s?(\d{1,2})(?:\s?/\s?(\d{4}))?", s)
    if not m or led is None:
        res[("skip", g["type"])] += 1
        print("SKIP", q["question"][:90], g["answer"])
        continue
    st, en = period("month", q["as_of_date"], month=int(m.group(1)), year=int(m.group(2)) if m.group(2) else None)
    names = led.orders["customer"].dropna().astype(str).unique()
    who = [n for n in names if strip_accents(n) and strip_accents(n) in s]
    who = max(who, key=len) if who else None
    if who:
        new, back = led.debt_movements(who, st, en)
    else:  # whole shop
        new, back = led.debt_movements(None, st, en)
    if g["type"] == "list":
        pred = [new, back]
    elif re.search(r"no them", s) and not re.search(r"tra no|thu no|tra duoc", s):
        pred = new
    else:
        pred = back
    ok = g["type"] != "unanswerable" and match(g["type"], pred, g["answer"], None)
    res[(g["type"], ok)] += 1
    if not ok:
        print(f"BAD {b} who={who} {q['question'][:80]} pred={pred} gold={g['answer']}")
print(dict(res))
