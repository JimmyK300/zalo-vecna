"""CPU regression checks: python -m unittest discover -s tests -p test_cleaner.py"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from datamate.cleaning import clean_question, validate_rewrite, validate_shop_hints
from datamate.cleaner_llm import CleanerLLM
from datamate.workspace import Workspace, load_workspace


class CleanerChecks(unittest.TestCase):
    def test_rules_and_invariants(self):
        r = clean_question("dthu hqua ko tính Otrivin 0,1% chai 10ml", ["Otrivin 0,1%"])
        self.assertEqual(r.normalized, "doanh thu hôm qua không tính Otrivin 0,1% chai 10ml")
        self.assertTrue(validate_rewrite(r, r.normalized)[0])
        self.assertFalse(validate_rewrite(r, r.normalized.replace("không ", ""))[0])
        self.assertFalse(validate_rewrite(r, r.normalized.replace("10ml", "20ml"))[0])
        self.assertFalse(validate_rewrite(r, r.normalized.replace("doanh thu", "số đơn"))[0])
        self.assertFalse(validate_rewrite(clean_question("giá trị -500"), "giá trị 500")[0])
        self.assertEqual(clean_question("mã KO001 00123, 500k").normalized, "mã KO001 00123, 500k")
        self.assertTrue(clean_question("một đơn là nhiu").ambiguous)
        self.assertNotIn("nhiều", clean_question("một đơn là nhiu").normalized)
        self.assertEqual(clean_question("dthu của ko", ["ko"]).normalized, "doanh thu của ko")

    def test_raw_and_failed_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            pd.DataFrame({"Ngày": ["01/01/2025", "sai"], "Tiền": ["2tr300", "38.000đ"],
                          "Mã": ["001", "002"]}).to_csv(p / "sample.csv", index=False)
            a = load_workspace(p, cleaner=False)
            b = load_workspace(p, cleaner=True)
            self.assertEqual(set(a.tables), set(b.tables))
            for key in a.tables:
                pd.testing.assert_frame_equal(a.tables[key].df, b.tables[key].df)
            self.assertTrue(b.raw_tables)
            self.assertEqual(b.raw_tables["sample"]["Tiền"].iloc[0], "2tr300")
            report = b.cleaning_reports["sample"]
            self.assertTrue(any(r["failed"] == 1 for r in report["columns"]))

    def test_monthly_raw(self):
        ws = Workspace(Path("."), cleaner=True)
        ws._add_sheets("sales", "sales.xlsx", {"T1_2025": pd.DataFrame({"Tien": ["38.000đ"]}),
                                               "T2_2025": pd.DataFrame({"Tien": ["45.000đ"]})})
        self.assertEqual(ws.raw_tables["sales"]["Tien"].tolist(), ["38.000đ", "45.000đ"])
        self.assertEqual(len(ws.raw_tables["sales"]), 2)

    def test_local_only(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                CleanerLLM("qwen", "https://api.example.com/v1", d)

    def test_strict_money(self):
        from datamate.evaluate import match
        self.assertFalse(match("number", 1000000500, 1000000000, []))
        self.assertTrue(match("number", 1.0000000001, 1, []))

    def test_cache_timeout_budget_and_guards(self):
        with tempfile.TemporaryDirectory() as d:
            fake = unittest.mock.MagicMock()
            response = {"normalized": "một đơn là bao nhiêu", "uncertain_spans": []}
            fake.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(response)))])
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **kw: fake)}):
                service = CleanerLLM("same-qwen", "http://127.0.0.1:8000/v1", d)
            r = service.question(clean_question("một đơn là nhiu"), "rules-a", time.time()+60)
            self.assertEqual(r.normalized, response["normalized"])
            r2 = service.question(clean_question("một đơn là nhiu"), "rules-a", time.time()+60)
            self.assertEqual(r2.llm["status"], "cache")
            self.assertEqual(service.calls, 1)
            fake.with_options.return_value.chat.completions.create.side_effect = TimeoutError("slow")
            r3 = service.question(clean_question("bn tiền 500k"), "rules-b", time.time()+60)
            self.assertEqual(r3.llm["status"], "fallback")
            self.assertEqual(r3.normalized, "bn tiền 500k")
            service.max_calls = 0
            self.assertEqual(service.request("question", {}, "new", time.time()+60)[1]["status"], "budget_exhausted")

    def test_shop_hints_cannot_invent_columns(self):
        p = {"columns": [{"key": "sales.x", "dtype": "object", "samples": ["đã trả"]}]}
        hints = {"hints": [{"column": "sales.x", "meaning": "payment status", "evidence": "đã trả"}]}
        self.assertEqual(len(validate_shop_hints(p, hints)), 1)
        hints["hints"][0]["column"] = "sales.invented"
        with self.assertRaises(ValueError):
            validate_shop_hints(p, hints)


if __name__ == "__main__":
    unittest.main()
