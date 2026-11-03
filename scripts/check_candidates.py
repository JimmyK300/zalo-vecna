"""Accuracy of every deterministic candidate (metric engine + debt) over a whole split.
Gold is used only to test the code. Prints per-kind accuracy and the misses.
python scripts/check_candidates.py [devkit/train] [--kinds revenue,order_count] [--quiet]"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.evaluate import match  # noqa: E402
from datamate.metrics import metric_candidate, metric_intent  # noqa: E402
from datamate.recipes import debt_candidate, debt_intent, debt_mentioned  # noqa: E402
from datamate.rules import active_rules  # noqa: E402
from scripts.wscache import load_cached as load_workspace  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = Path(args[0] if args else "devkit/train")
kinds = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--kinds=")), None)
quiet = "--quiet" in sys.argv
rules = active_rules((D / "rulebook.md").read_text(encoding="utf-8"))
qs = [json.loads(l) for l in open(D / "questions.jsonl", encoding="utf-8")]
gp = D / "answers.jsonl"
gold = {json.loads(l)["id"]: json.loads(l) for l in open(gp, encoding="utf-8")} if gp.exists() else {}
cal = pd.read_csv(D / "calendar.csv")
ws_cache, res, misses = {}, defaultdict(Counter), []
for q in qs:
    mi = metric_intent(q["question"], q["as_of_date"])
    di = debt_intent(q["question"], q["as_of_date"])
    kind = ("debt:" + di["kind"]) if di else (mi["kind"] if mi else ("item?" if not di else None))
    if not kind or (kinds and kind not in kinds):
        continue
    b = q["business_id"]
    if b not in ws_cache:
        ws_cache[b] = load_workspace(D / "data" / b, {"calendar": cal}, rules)
    ws = ws_cache[b]
    c = debt_candidate(q["question"], q["as_of_date"], ws.debt, debt_mentioned(ws.docs)) if di else \
        metric_candidate(q["question"], q["as_of_date"], ws)
    if c is None and not di:
        from datamate.items import item_candidate, item_intent
        it = item_intent(q["question"], q["as_of_date"], ws)
        if it:
            kind = "item:" + it["kind"]
            c = item_candidate(q["question"], q["as_of_date"], ws)
        elif kind == "item?":
            continue
    if c is None:
        res[kind]["none"] += 1
        continue
    if not gold:
        res[kind][c[0]] += 1
        continue
    g = gold[q["id"]]
    try:
        ok = c[0] == g["type"] and (g["type"] == "unanswerable" or match(g["type"], c[1], g["answer"], g.get("alias")))
    except Exception:  # noqa: BLE001
        ok = False
    res[kind]["ok" if ok else "BAD"] += 1
    if not ok:
        misses.append(f"BAD {kind} {b} {q['as_of_date']} | {q['question'][:100]} | cand={json.dumps(c, ensure_ascii=False)[:70]} "
                      f"| gold={g['type']}:{json.dumps(g.get('answer'), ensure_ascii=False)[:60]}")
if not quiet:
    print("\n".join(misses))
tot = Counter()
for k in sorted(res):
    tot.update(res[k])
    print(f"{k:24s} {dict(res[k])}")
print("TOTAL", dict(tot))
