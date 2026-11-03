"""Altered-rulebook test: when the rulebook drops or changes one rule, do the recipes stop answering
the questions that depend on it (instead of silently applying yesterday's rule)?

For each rule-encoding helper R, the active rule set loses R and every train question is asked again.
'stale' = a recipe still answers a question whose answer depends on R (by the question's wording).
Expected: 0 stale for every rule. 'kept' shows how much coverage survives an unrelated change.

python scripts/check_rule_changes.py [devkit/train]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.metrics import metric_intent, rules_needed  # noqa: E402
from datamate.recipes import debt_intent  # noqa: E402
from datamate.rules import RULES, active_rules  # noqa: E402
from datamate.run import recipe_candidate  # noqa: E402
from datamate.semantics import strip_accents  # noqa: E402
from scripts.wscache import load_cached  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else "devkit/train")
base_rules = active_rules((D / "rulebook.md").read_text(encoding="utf-8"))
cal = pd.read_csv(D / "calendar.csv")
qs = [json.loads(l) for l in open(D / "questions.jsonl", encoding="utf-8")]
wss = {}


def depends(q: dict, rule: str, ws) -> bool:
    """Does the answer to q depend on `rule`, judged from the question alone?"""
    s = strip_accents(q["question"])
    mi = metric_intent(q["question"], q["as_of_date"], ws)
    kind = mi["kind"] if mi else None
    if rule in rules_needed(q["question"], kind):
        return True
    if rule in ("debt_ledger",):
        return bool(debt_intent(q["question"], q["as_of_date"]))
    if rule == "in_revenue":
        # every revenue / count / item figure follows the revenue rule (debt has its own §4 definition)
        return not debt_intent(q["question"], q["as_of_date"])
    if rule == "best_seller_base_units":
        return bool(re.search(r"ban chay", s))
    if rule == "shift_hours":
        return bool(re.search(r"\bca (sang|chieu)", s))
    if rule == "size_suffix":
        return bool(re.search(r"\bsize [ml]\b", s))
    if rule == "group_exact":
        return bool(re.search(r"\bnhom\b", s)) and not re.search(r"khach", s)
    if rule == "staff_by_revenue":
        return bool(re.search(r"nhan vien .{0,25}(gioi nhat|cao nhat|nhieu nhat)|nhan vien nao|nguoi ban nao", s))
    if rule == "gross_margin_line_cost":
        return bool(re.search(r"\blai (gop|nhieu)|loi nhuan|\blai nhat", s))
    return False


def answers(rules: set) -> dict:
    out = {}
    for q in qs:
        b = q["business_id"]
        if b not in wss:
            wss[b] = load_cached(D / "data" / b, {"calendar": cal}, base_rules)
        ws = wss[b]
        saved = ws.rules
        ws.rules = rules
        try:
            c = recipe_candidate(q, ws)
        finally:
            ws.rules = saved
        if c is not None:
            out[q["id"]] = c
    return out


base = answers(base_rules)
print(f"baseline: {len(base)} recipe answers")
byid = {q["id"]: q for q in qs}
for rule in sorted(base_rules):
    ans = answers(base_rules - {rule})
    stale = Counter()
    for qid in ans:
        q = byid[qid]
        if ans[qid][0] == "unanswerable":
            continue  # "no such data in this shop" holds under any rulebook
        if depends(q, rule, wss[q["business_id"]]):
            stale[rule] += 1
            print(f"    stale: {qid} {q['question'][:90]} -> {str(ans[qid])[:60]}")
    print(f"- {rule:26s} answers={len(ans):5d} kept={len(ans) / max(1, len(base)):.0%}  stale={stale[rule]}")
