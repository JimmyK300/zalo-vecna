"""CPU checks; no model download, inference or credentials needed."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.evaluate import match, score
from datamate.question import auxiliary_question
from datamate.hf_backend import parse_calls


class UpgradeChecks(unittest.TestCase):
    def test_money_and_missing_predictions(self):
        self.assertFalse(match("number", 1000000500, 1000000000, []))
        self.assertFalse(match("number", True, 1, []))
        self.assertFalse(match("number", float("nan"), 1, []))
        self.assertTrue(match("number", 1.0000000001, 1, []))
        gold = {"a": {"type": "number", "answer": 1}, "b": {"type": "number", "answer": 2}}
        result = score({"a": {"type": "number", "answer": 1}}, gold)
        self.assertEqual((result["n"], result["PA"], result["abstain"]), (2, .5, 1))

    def test_conservative_question(self):
        self.assertEqual(auxiliary_question("dthu hqua ko tính KH001 500k"),
                         "doanh thu hôm qua không tính KH001 500k")
        self.assertEqual(auxiliary_question("mã HNAY_01, DT 2k, chị Tống"), "mã HNAY_01, DT 2k, chị Tống")

    def test_native_tools(self):
        text = '<tool_call>\n<function=run_sql>\n<parameter=query>\nSELECT 1 AS x\n</parameter>\n</function>\n</tool_call>'
        self.assertEqual(parse_calls(text), [("run_sql", {"query": "SELECT 1 AS x"})])
        with self.assertRaises(ValueError):
            parse_calls(text[:-12])
        with self.assertRaises(ValueError):
            parse_calls(text.replace('run_sql', 'delete_files'))
        self.assertEqual(parse_calls("Please continue"), [])


if __name__ == "__main__":
    unittest.main()
