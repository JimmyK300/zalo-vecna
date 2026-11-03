"""Accuracy and coverage of the deterministic debt candidate over ALL train questions."""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.evaluate import match  # noqa: E402
from datamate.recipes import debt_candidate, debt_mentioned  # noqa: E402
from datamate.rules import active_rules  # noqa: E402
from datamate.workspace import load_workspace  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else "devkit/train")
rules = active_rules((D / "rulebook.md").read_text(encoding="utf-8"))
qs = [json.loads(l) for l in open(D / "questions.jsonl", encoding="utf-8")]
gold_path = D / "answers.jsonl"
gold = {json.loads(l)["id"]: json.loads(l) for l in open(gold_path, encoding="utf-8")} if gold_path.exists() else {}
cache, res = {}, Counter()
for q in qs:
    b = q["business_id"]
    if b not in cache:
        cache[b] = None
    from datamate.recipes import debt_intent
    it = debt_intent(q["question"], q["as_of_date"])
    if not it or it["kind"] == "movements":
        continue
    if cache[b] is None:
        cache[b] = load_workspace(D / "data" / b, {"calendar": pd.read_csv(D / "calendar.csv")}, rules)
    c = debt_candidate(q["question"], q["as_of_date"], cache[b].debt, debt_mentioned(cache[b].docs))
    if c is None:
        res["no candidate"] += 1
        continue
    if not gold:
        res[c[0]] += 1
        continue
    g = gold[q["id"]]
    ok = c[0] == g["type"] and (g["type"] == "unanswerable" or match(g["type"], c[1], g["answer"], g.get("alias")))
    res[ok] += 1
    if not ok:
        print("WRONG", b, q["question"][:90], "| cand", json.dumps(c, ensure_ascii=False)[:80], "| gold", g["type"],
              json.dumps(g.get("answer"), ensure_ascii=False)[:60])
print(dict(res))
