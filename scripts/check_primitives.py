"""Check layer-2 primitives against known train answers (gold used only to test the code)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from datamate.primitives import period  # noqa: E402
from datamate.rules import active_rules  # noqa: E402
from datamate.workspace import load_workspace  # noqa: E402

DATA = Path("devkit/train")
rules = active_rules((DATA / "rulebook.md").read_text(encoding="utf-8"))
cal = {"calendar": pd.read_csv(DATA / "calendar.csv")}
_ws = {}


def orders(biz):
    if biz not in _ws:
        _ws[biz] = load_workspace(DATA / "data" / biz, cal, rules)
    return _ws[biz].tables["orders"].df


def between(o, rng):
    return o[(o.order_date >= pd.Timestamp(rng[0])) & (o.order_date <= pd.Timestamp(rng[1]))]


def revenue(biz, rng):
    o = between(orders(biz), rng)
    return int(round(o.loc[o.in_revenue, "revenue_lines"].sum()))


def cancel_rate(biz, rng):
    o = between(orders(biz), rng)
    return round((o.status == "huy").sum() / len(o) * 100, 2)


cases = [
    ("revenue yesterday", lambda: revenue("biz_108", period("yesterday", "2025-06-28")), 7241300),
    ("revenue yesterday", lambda: revenue("biz_119", period("yesterday", "2025-07-07")), 3462700),
    ("revenue Q3/2025", lambda: revenue("biz_123", period("quarter", "2026-06-28", quarter=3, year=2025)), 616402300),
    ("revenue Q3/2025", lambda: revenue("biz_110", period("quarter", "2026-07-03", quarter=3, year=2025)), 456544000),
    ("revenue this week", lambda: revenue("biz_112", period("this_week", "2025-07-05")), 62109000),
    ("cancel rate 2/2026", lambda: cancel_rate("biz_121", period("month", "2026-09-20", month=2, year=2026)), 1.1),
    ("cancel rate 4/2026", lambda: cancel_rate("biz_103", period("month", "2026-07-09", month=4, year=2026)), 1.89),
    ("cancel rate 7/2026", lambda: cancel_rate("biz_118", period("month", "2026-09-10", month=7, year=2026)), 1.92),
    ("orders >=130k 6/2025", lambda: int(((lambda o: o[o.in_revenue & (o.revenue_lines >= 130000)])(
        between(orders("biz_105"), period("month", "2026-06-24", month=6, year=2025)))).shape[0]), 459),
]
ok = 0
for name, f, gold in cases:
    try:
        got = f()
    except Exception as e:  # noqa: BLE001
        got = f"ERR {type(e).__name__}: {e}"
    hit = got == gold
    ok += hit
    print(f"{'OK ' if hit else 'BAD'} {name:22s} got={got} gold={gold}")
print(f"{ok}/{len(cases)}")
