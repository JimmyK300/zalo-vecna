"""Accuracy of the filtered-metric recipe over a whole split (gold only tests the code).
python scripts/check_filters.py [devkit/train] [--quiet]"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.evaluate import score_one  # noqa: E402
from datamate.filters import filter_candidate, filter_intent  # noqa: E402
from datamate.io_contract import normalize_answer  # noqa: E402
from datamate.rules import active_rules  # noqa: E402
from datamate.run import recipe_candidate  # noqa: E402
from scripts.wscache import load_cached  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = Path(args[0] if args else "devkit/train")
rules = active_rules((D / "rulebook.md").read_text(encoding="utf-8"))
cal = pd.read_csv(D / "calendar.csv")
qs = [json.loads(l) for l in open(D / "questions.jsonl", encoding="utf-8")]
gp = D / "answers.jsonl"
gold = {json.loads(l)["id"]: json.loads(l) for l in open(gp, encoding="utf-8")} if gp.exists() else {}
wss, res, misses = {}, defaultdict(Counter), []
for q in qs:
    b = q["business_id"]
    if b not in wss:
        wss[b] = load_cached(D / "data" / b, {"calendar": cal}, rules)
    ws = wss[b]
    it = filter_intent(q["question"], q["as_of_date"], ws)
    if not it:
        continue
    c = filter_candidate(q["question"], q["as_of_date"], ws)
    key = f"{it['dim']}/{it['kind']}/{it.get('metric', '')}"
    if c is None:
        res[key]["none"] += 1
        continue
    try:
        ans = normalize_answer(c[0], c[1]) if c[0] != "unanswerable" else None
    except (ValueError, TypeError):
        res[key]["bad_format"] += 1
        continue
    s = score_one({"type": c[0], "answer": ans}, gold[q["id"]]) if gold else "?"
    res[key]["ok" if s == "correct" else "BAD"] += 1
    if s != "correct":
        misses.append((b, q["question"][:100], c, gold[q["id"]]["type"], str(gold[q["id"]].get("answer"))[:60]))
tot = Counter()
for k in sorted(res):
    print(f"{k:38s} {dict(res[k])}")
    tot.update(res[k])
print("TOTAL", dict(tot))
if "--quiet" not in sys.argv:
    for m in misses:
        print("  MISS", m)
