from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest

import duckdb
import pandas as pd

from scripts.compose_queries import LEVELS, compose, load_primitives, prepared_tables


class ComposeQueriesTests(unittest.TestCase):
    def test_composes_all_levels_with_correct_filters_and_baselines(self):
        orders = pd.DataFrame([
            (1, "2026-01-01", True, "hoan_thanh", 1, "a", False),
            (2, "2026-02-01", True, "hoan_thanh", 100, "a", False),
            (3, "2026-02-02", True, "hoan_thanh", 100, "b", False),
            (4, "2026-03-01", True, "hoan_thanh", 50, "a", False),
            (5, "2026-03-01", True, "hoan_thanh", 50, "a", False),
            (6, "2026-03-02", True, "dang_giao", 50, "a", False),
            (7, "2026-03-03", True, "hoan_thanh", 50, "a", False),
            (8, "2026-03-03", True, "hoan_thanh", 100, "b", False),
            (9, "2026-03-03", True, "hoan_thanh", 50, None, True),
            (10, "2026-03-03", False, "huy", 999, "c", False),
            (11, "2026-03-03", False, "tra_hang", 999, "c", False),
        ], columns=["order_id", "order_date", "in_revenue", "status", "revenue_lines",
                    "customer_code", "is_walkin"])
        orders["order_date"] = pd.to_datetime(orders.order_date)
        lines = pd.DataFrame([
            ("2026-02-02", "A", "01", 2, True, False),
            ("2026-02-02", "B", "02", 100, True, False),
            ("2026-03-02", "A", "01", 10, True, False),
            ("2026-03-02", "B", "02", 10, True, False),
            ("2026-03-02", "C", "03", 4, True, False),
            ("2026-03-02", "D", "04", 3, True, False),
            ("2026-03-02", "E", "05", 1, True, False),
            ("2026-03-02", "Excluded", "00", 1000, False, False),
            ("2026-03-02", "Fee", "06", 1000, True, True),
        ], columns=["order_date", "item", "item_code", "qty_base", "in_revenue", "is_fee"])
        lines["order_date"] = pd.to_datetime(lines.order_date)
        ws = SimpleNamespace(tables={"orders": SimpleNamespace(df=orders)}, lines=lines)
        names, examples, rules = load_primitives(Path(__file__).resolve().parents[1] / "primitives")
        with duckdb.connect(":memory:") as db:
            flags = prepared_tables(db, ws)
            answers = {}
            levels = set()
            for level, recipe, _, sql, params, text, rule_names, answer in compose(
                db, flags, [date(2026, 3, 3)], examples
            ):
                levels.add(level)
                self.assertLessEqual(set(text[1:]), names)
                self.assertLessEqual(set(rule_names), rules.keys())
                answers[recipe] = db.execute(sql, params).fetchone()[0] if sql else answer
        self.assertEqual(levels, set(LEVELS))
        self.assertEqual(answers["daily_orders"], 3)
        self.assertEqual(answers["recent_revenue"], 350)
        self.assertEqual(answers["month_to_date_growth"], 75)
        self.assertEqual(answers["top_products"][:2], ["A", "B"])
        self.assertEqual(answers["leader_quantity_delta"], 8)
        self.assertEqual(answers["repeat_buyer_share_delta"], 50)
