import contextlib
import json
import sqlite3
import tempfile
import threading
import unittest
from argparse import Namespace
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openpyxl import Workbook

from datamate_bench.agent import solve
from datamate_bench.client import LocalModel
from datamate_bench.data import ReadOnlySQL, build_database, iso_date, number
from datamate_bench.scoring import matches, normalize, score, validate
from datamate_bench.suite import make_suite, load_suite
from datamate_bench.suite import digest


class ScoringTests(unittest.TestCase):
    def test_strict_types(self):
        for value in ["123", True, float("nan"), float("inf")]:
            self.assertFalse(validate({"type": "number", "answer": value}))
        self.assertFalse(validate({"type": "boolean", "answer": "false"}))
        self.assertFalse(validate({"type": "list", "answer": [False]}))
        self.assertFalse(validate({"type": "date", "answer": "2026-02-30"}))

    def test_normalization(self):
        self.assertEqual(normalize("  TÃ_Má!! "), "tã má")
        self.assertFalse(matches({"type": "string", "answer": "Ca phe"},
                                 {"type": "string", "answer": "Cà phê"}))

    def test_number_not_rounded_and_percent_is(self):
        self.assertFalse(matches({"type": "number", "answer": 12.341}, {"type": "number", "answer": 12.34}))
        self.assertTrue(matches({"type": "percent", "answer": 12.344}, {"type": "percent", "answer": 12.34}))
        self.assertFalse(matches({"type": "percent", "answer": -12.34}, {"type": "percent", "answer": 12.34}))

    def test_lists_order_and_alias(self):
        ref = {"type": "list", "answer": ["Cà phê", "Trà"], "alias": [["SP01"], ["SP02"]]}
        self.assertTrue(matches({"type": "list", "answer": ["SP01", "SP02"]}, ref))
        self.assertFalse(matches({"type": "list", "answer": ["SP02", "SP01"]}, ref))

    def test_table_column_reordering_alias_and_row_order(self):
        ref = {"type": "table", "answer": {"columns": ["khách", "tiền"], "rows": [["Lan", 20], ["Tú", 30]]},
               "alias": [["KH01"], ["KH02"]]}
        pred = {"type": "table", "answer": {"columns": ["tiền", "khách"], "rows": [[30, "KH02"], [20, "KH01"]]}}
        self.assertTrue(matches(pred, ref))
        pred["answer"]["rows"][0][0] = "30"
        self.assertFalse(matches(pred, ref))

    def test_table_positional_fallback_and_duplicates(self):
        ref = {"type": "table", "answer": {"columns": ["name", "amount"], "rows": [["A", 1], ["A", 1]]}}
        pred = {"type": "table", "answer": {"columns": ["x", "y"], "rows": [["A", 1], ["A", 2]]}}
        self.assertFalse(matches(pred, ref))
        pred["answer"]["rows"][1][1] = 1
        self.assertTrue(matches(pred, ref))

    def test_unanswerable_and_abstain(self):
        self.assertEqual(score({"type": "abstain"}, {"type": "number", "answer": 0}), 0)
        self.assertEqual(score({"type": "unanswerable"}, {"type": "number", "answer": 0}), -1)
        self.assertEqual(score({"type": "unanswerable"}, {"type": "unanswerable", "answer": None}), 1)


class DataTests(unittest.TestCase):
    def test_numeric_and_date_parsing(self):
        self.assertEqual(number("4,38tr"), 4380000)
        self.assertEqual(number("-10.000đ"), -10000)
        self.assertEqual(number("270k"), 270000)
        self.assertEqual(number("703,000"),703000)
        self.assertEqual(number("210,100đ"),210100)
        self.assertEqual(number("1,234.50"),1234.5)
        self.assertEqual(number("1.234,50"),1234.5)
        self.assertIsNone(number("SP002"))
        self.assertEqual(iso_date("1/6", 2026), "2026-06-01")
        self.assertEqual(iso_date("1 thg 5 2025"), "2025-05-01")
        self.assertEqual(iso_date(45870), "2025-08-01")

    def test_real_excel_provenance_cache_and_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shop = root / "shop"
            shop.mkdir()
            (shop / "README.md").write_text("A raw workbook", encoding="utf-8")
            calendar = root / "calendar.csv"
            calendar.write_text("date,lunar_day\n2026-01-01,13\n", encoding="utf-8")
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "T06-2026"
            sheet.append(["Tiền", None, "Hàng hoá"])
            sheet.append(["Ngày", "Hàng", "Thành tiền"])
            sheet.append(["1/6", "Cà phê", "20.000đ"])
            sheet.append(["1/6", "Cà phê", "20.000đ"])
            workbook.save(shop / "sales.xlsx")
            db, manifest, timing = build_database(shop, calendar, root / "cache")
            self.assertFalse(timing["cache_hit"])
            self.assertEqual(manifest["tables"]["t01"]["rows"], 2)  # Never silently dedup raw rows.
            sql = ReadOnlySQL(db)
            try:
                rows = sql.query('SELECT SUM(num(thanh_tien)), MIN(iso_date(ngay,_source_year)) FROM t01')["rows"]
                self.assertEqual(rows, [(40000.0, "2026-06-01")])
            finally:
                sql.close()
            _, _, second = build_database(shop, calendar, root / "cache")
            self.assertTrue(second["cache_hit"])
            (shop / "README.md").write_text("Changed meaning", encoding="utf-8")
            changed, _, third = build_database(shop, calendar, root / "cache")
            self.assertNotEqual(changed, db)
            self.assertFalse(third["cache_hit"])

    def test_sql_rejects_mutation_attach_file_read_and_interrupts(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "test.sqlite"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE sales(amount)")
            con.execute("INSERT INTO sales VALUES(20)")
            con.commit()
            con.close()
            tool = ReadOnlySQL(db, seconds=.01, row_limit=2)
            try:
                for statement in ["DELETE FROM sales", "DROP TABLE sales", "ATTACH DATABASE ':memory:' AS other",
                                  "PRAGMA database_list", "SELECT load_extension('x')", "SELECT readfile('x')"]:
                    with self.assertRaises(sqlite3.DatabaseError, msg=statement):
                        tool.query(statement)
                with self.assertRaises(sqlite3.DatabaseError):
                    tool.query("WITH RECURSIVE x(a) AS (VALUES(1) UNION ALL SELECT a+1 FROM x) SELECT sum(a) FROM x")
                self.assertEqual(tool.query("SELECT amount FROM sales")["rows"], [(20,)])
                self.assertTrue(tool.query("SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3")["truncated"])
            finally:
                tool.close()


class AgentHTTPTests(unittest.TestCase):
    def test_non_loopback_rejected(self):
        with self.assertRaises(ValueError):
            LocalModel("https://example.com/v1")

    def test_agent_uses_rules_then_sql_without_labels(self):
        received = []
        actions = iter([{"action": "rules", "section": "1"},
                        {"action": "sql", "sql": "SELECT SUM(amount) FROM sales"},
                        {"action": "finish", "type": "number", "answer": 30}])

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"test-local"}]}')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(body)
                payload = {"choices": [{"message": {"content": json.dumps(next(actions))}, "finish_reason": "stop"}],
                           "usage": {"completion_tokens": 20}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                db = Path(temp) / "data.sqlite"
                con = sqlite3.connect(db)
                con.execute("CREATE TABLE sales(amount)")
                con.executemany("INSERT INTO sales VALUES (?)", [(10,), (20,)])
                con.commit()
                con.close()
                model = LocalModel(f"http://127.0.0.1:{server.server_port}/v1")
                model.connect()
                sql = ReadOnlySQL(db)
                try:
                    result = solve({"id": "not-visible", "question": "Doanh thu?", "as_of_date": "2026-01-01",
                                    "answer": "SECRET_REFERENCE"}, "shop", "intro\n## 1. Doanh thu\nUse valid orders",
                                   {"warnings": [], "tables": {"sales": {"columns": ["amount"], "rows": 2}}},
                                   sql, model)
                finally:
                    sql.close()
                self.assertEqual(result["prediction"], {"type": "number", "answer": 30})
                self.assertEqual(result["successful_sql"], 1)
                self.assertNotIn("SECRET_REFERENCE", json.dumps(received))
                self.assertNotIn("not-visible", json.dumps(received))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_step_limit_abstains(self):
        class BrokenModel:
            def chat(self, *args, **kwargs):
                return "not JSON", {}
        result = solve({"question": "?", "as_of_date": "2026-01-01"}, "", "intro", {"warnings": [], "tables": {}},
                       None, BrokenModel(), max_steps=2)
        self.assertEqual(result["prediction"]["type"], "abstain")
        self.assertEqual(len(result["trace"]), 2)


class RunnerTests(unittest.TestCase):
    def test_end_to_end_runner_writes_and_scores_after_inference(self):
        from datamate_bench.__main__ import run_benchmark

        class FixtureModel:
            def __init__(self, *args):
                self.model = "fixture-only-not-a-real-model"
                self.records = []
                self.actions = iter([{"action": "rules", "section": "1"},
                                     {"action": "sql", "sql": "SELECT SUM(num(amount)) FROM t01"},
                                     {"action": "finish", "type": "number", "answer": 30}])

            def connect(self):
                return [self.model]

            def chat(self, messages, **kwargs):
                self.records.append({"request_seconds": .001, "usage": {}})
                return json.dumps(next(self.actions)), self.records[-1]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "train"
            shop = split / "data" / "shop"
            shop.mkdir(parents=True)
            (shop / "README.md").write_text("One row per sale", encoding="utf-8")
            (shop / "sales.csv").write_text("amount\n10\n20\n", encoding="utf-8")
            (split / "questions.jsonl").write_text(json.dumps({"id": "fixture", "business_id": "shop",
                                                               "question": "Total?", "as_of_date": "2026-01-01"}) + "\n")
            (split / "answers.jsonl").write_text('{"id":"fixture","type":"number","answer":30}\n')
            (split / "rulebook.md").write_text("intro\n## 1. Revenue\nSum amounts", encoding="utf-8")
            (split / "calendar.csv").write_text("date,lunar_day\n2026-01-01,13\n")
            suite = root / "suite.json"
            suite.write_text(json.dumps({"questions_sha256": digest(split / "questions.jsonl"),
                                         "answers_sha256": digest(split / "answers.jsonl"),
                                         "rulebook_sha256": digest(split / "rulebook.md"), "items": [{"id": "fixture"}]}))
            output = root / "output"
            args = Namespace(train=split, suite=suite, output=output, limit=None, endpoint="http://127.0.0.1:1/v1",
                             model=None, max_tokens=128, seed=42, require_cuda=False, max_steps=5,
                             question_timeout=10, sql_timeout=1, cache=root / "cache")
            with patch("datamate_bench.__main__.LocalModel", FixtureModel), patch(
                    "datamate_bench.__main__.inspect_hardware", return_value={"cuda_available": False}):
                with contextlib.redirect_stdout(__import__("io").StringIO()):
                    run_benchmark(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["score"]["correct"], 1)
            self.assertFalse(summary["gpu_execution_verified"])
            self.assertTrue((output / "report.md").exists())
            log = json.loads((output / "logs" / "fixture.json").read_text())
            self.assertNotIn("reference", log)
            self.assertEqual(read_jsonl_for_test(output / "submission.jsonl")[0]["answer"], 30)


def read_jsonl_for_test(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    unittest.main()
