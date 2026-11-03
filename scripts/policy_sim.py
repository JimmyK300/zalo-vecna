"""Offline policy simulation on a finished run: what if a recipe candidate replaces / vetoes the
model's answer? Uses the CURRENT recipe code; gold only scores the alternatives."""
import json
import glob
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.evaluate import _load, score_one  # noqa: E402
from datamate.items import item_candidate, item_intent  # noqa: E402
from datamate.metrics import metric_candidate  # noqa: E402
from datamate.recipes import debt_candidate, debt_mentioned  # noqa: E402
from datamate.rules import active_rules  # noqa: E402
from scripts.wscache import load_cached  # noqa: E402

run = sys.argv[1]
D = Path("devkit/train")
g = _load(D / "answers.jsonl")
rules = active_rules((D / "rulebook.md").read_text(encoding="utf-8"))
cal = {"calendar": pd.read_csv(D / "calendar.csv")}
final = _load(Path(f"out/{run}/submission.jsonl"))
ws = {}
rows = []
for f in glob.glob(f"out/{run}/logs/q_*.json"):
    l = json.load(open(f, encoding="utf-8"))
    q = l["id"]
    dis = l.get("disagreement") or {}
    first = {"type": dis["first"][0], "answer": dis["first"][1]} if dis.get("first") else final[q]
    b = l["business_id"]
    if b not in ws:
        ws[b] = load_cached(D / "data" / b, cal, rules)
    w = ws[b]
    c = debt_candidate(l["question"], l["as_of_date"], w.debt, debt_mentioned(w.docs)) or metric_candidate(l["question"], l["as_of_date"], w)
    if c is None:
        it = item_intent(l["question"], l["as_of_date"], w)
        if it:
            c = item_candidate(l["question"], l["as_of_date"], w)
    cand = {"type": c[0], "answer": c[1]} if c else None
    rows.append((score_one(first, g[q]), score_one(final[q], g[q]), score_one(cand, g[q]) if cand else None))
n = len(rows)
pa = lambda ks: (ks.count("correct") - ks.count("wrong")) / n  # noqa: E731
print(f"{run}: n={n}")
print(f"  model first attempt        PA {pa([r[0] for r in rows]):.3f}")
print(f"  as run (checks)            PA {pa([r[1] for r in rows]):.3f}")
print(f"  candidate overrides first  PA {pa([r[2] if r[2] else r[0] for r in rows]):.3f}")
print(f"  candidate overrides final  PA {pa([r[2] if r[2] else r[1] for r in rows]):.3f}")
cov = [r for r in rows if r[2]]
print(f"  candidates: {len(cov)} questions, candidate {dict(Counter(r[2] for r in cov))}, model-first on them {dict(Counter(r[0] for r in cov))}")
