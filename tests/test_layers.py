"""Checks for the deterministic layers. Run: python tests/test_layers.py [--all-shops]

1. period(): rulebook §2 definitions on fixed dates
2. as_of_balance(): toy ledger
3. rule-change test: an edited rulebook must switch the matching helpers off
4. (--all-shops) every train + public-test shop loads, builds `orders`, the graph and a plan card
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.graph import parse_rulebook, plan_card  # noqa: E402
from datamate.primitives import as_of_balance, pct_change, period, resolve_periods  # noqa: E402
from datamate.rules import RULES, active_rules  # noqa: E402
from datamate.workspace import load_workspace  # noqa: E402

D = dt.date
DATA_ROOT = Path(os.environ.get("DATAMATE_TEST_DATA", "devkit"))
fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got}, want {want}")


# 1. periods (2026-04-23 is a Thursday)
a = "2026-04-23"
check("today", period("today", a), (D(2026, 4, 23), D(2026, 4, 23)))
check("yesterday", period("yesterday", a), (D(2026, 4, 22), D(2026, 4, 22)))
check("this_week", period("this_week", a), (D(2026, 4, 20), D(2026, 4, 23)))
check("last_week", period("last_week", a), (D(2026, 4, 13), D(2026, 4, 19)))
check("this_month", period("this_month", a), (D(2026, 4, 1), D(2026, 4, 23)))
check("last_month_same_days", period("last_month_same_days", a), (D(2026, 3, 1), D(2026, 3, 23)))
check("same days, short month", period("last_month_same_days", "2026-03-31"), (D(2026, 2, 1), D(2026, 2, 28)))
check("same days, January", period("last_month_same_days", "2026-01-15"), (D(2025, 12, 1), D(2025, 12, 15)))
check("month no year, past", period("month", a, month=3), (D(2026, 3, 1), D(2026, 3, 31)))
check("month no year, future", period("month", a, month=7), (D(2025, 7, 1), D(2025, 7, 31)))
check("quarter 3/2025", period("quarter", a, quarter=3, year=2025), (D(2025, 7, 1), D(2025, 9, 30)))
check("quarter no year, future", period("quarter", a, quarter=3), (D(2025, 7, 1), D(2025, 9, 30)))
check("last 7 days", period("last_n_days", a, n=7), (D(2026, 4, 17), D(2026, 4, 23)))
check("ytd", period("year_to_date", a), (D(2026, 1, 1), D(2026, 4, 23)))
cal = pd.read_csv(DATA_ROOT / "train/calendar.csv")
s, e = period("tet", a, year=2026, cal=cal)
check("tet 2026", (s, e), (D(2026, 2, 10), D(2026, 2, 26)))  # 23/12 âm 2025 .. 10/1 âm 2026 (calendar.csv)
check("pct_change", pct_change(90, 100), -10.0)
r = resolve_periods("tổng dthu hqua là nhiêu", "2025-07-07")
check("slang 'hqua'", r[:1], ["yesterday: 2025-07-06"])
r = resolve_periods("tuần nayf dthu đươc bao nhiêu", "2025-07-05")
check("typo 'nayf'", r[:1], ["this week: 2025-06-30 .. 2025-07-05"])

# 2. as_of_balance: two orders, payments before and after the cut-off
orders = pd.DataFrame({"oid": ["A", "B"], "total": [100, 50], "d": ["2026-01-01", "2026-01-05"]})
pays = pd.DataFrame({"oid": ["A", "A", "B"], "amt": [30, 70, 50], "d": ["2026-01-02", "2026-01-10", "2026-01-05"]})
bal = as_of_balance(orders, "oid", "total", "d", "2026-01-06", pays, minus_amount="amt")
check("balance rows", bal[["oid", "balance"]].values.tolist(), [["A", 70]])
bal = as_of_balance(orders, "oid", "total", "d", "2026-01-31", pays, minus_amount="amt")
check("balance all paid", len(bal), 0)

# 3. rule-change test
rb = (DATA_ROOT / "train/rulebook.md").read_text(encoding="utf-8")
check("train rulebook: all helpers on", active_rules(rb), set(RULES) - {"no_status_note"})  # public-only rule
changed = rb.replace("bắt đầu **thứ Hai**", "bắt đầu **Chủ nhật**").replace("**kể cả chỉ trả một phần**", "")
on = active_rules(changed)
check("week rule switched off", "week_monday" in on, False)
check("revenue rule switched off", "in_revenue" in on, False)
ws = load_workspace(DATA_ROOT / "train/data/biz_105", {"calendar": cal}, on)
check("no in_revenue column when rule changed", any("in_revenue" in t.df for t in ws.tables.values()), False)
check("profile tells the model", any("disabled" in e for e in ws.errors), True)
card = plan_card("doanh thu tuần trước bao nhiêu", "2026-04-23", ws, parse_rulebook(changed))
check("plan card defers week definition", "week definition changed" in card, True)

# 4. every shop loads (slow)
if "--all-shops" in sys.argv:
    for split in ("train", "public_test"):
        root = DATA_ROOT / split
        rules = active_rules((root / "rulebook.md").read_text(encoding="utf-8"))
        rows = parse_rulebook((root / "rulebook.md").read_text(encoding="utf-8"))
        qs = {}
        for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines():
            q = json.loads(line)
            qs.setdefault(q["business_id"], q)
        for d in sorted(p for p in (root / "data").iterdir() if p.is_dir()):
            try:
                w = load_workspace(d, {"calendar": cal}, rules)
                o = w.tables.get("orders")
                cols = set(o.df.columns) if o else set()
                q = qs.get(d.name)
                plan_card(q["question"], q["as_of_date"], w, rows) if q else None
                miss = [c for c in ("order_date", "status", "revenue_lines") if c not in cols]
                print(f"{split:11s} {d.name}: orders={len(o.df) if o else 0:6d} missing={miss} "
                      f"joins={len(w.graph.joins) if w.graph else 0} errors={w.errors}")
            except Exception:  # noqa: BLE001
                fails.append(f"{d}: crash\n{traceback.format_exc(limit=3)}")

print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
