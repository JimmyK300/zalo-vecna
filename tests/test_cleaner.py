"""Comprehensive CPU regression checks for data-cleaner:
python -m unittest discover -s tests -p test_cleaner.py
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from datamate.cleaning import clean_question, validate_rewrite, validate_shop_hints, column_reports, QuestionCleanResult
from datamate.cleaner_llm import CleanerLLM
from datamate.workspace import Workspace, load_workspace
from datamate.semantics import TOTAL_ROW_RE


class CleanerChecks(unittest.TestCase):

    def test_three_smoke_regressions_and_variants(self):
        """Regression tests for the 3 meaning-distortion cases in the smoke test,
        plus accented/unaccented variations. All must be REJECTED by guard."""
        # Case 1: lost "hôm qua" / "hqua"
        r1 = clean_question("check giùm hqua có tổng cộng bao nhiêu đơn")
        prop1 = "check giúp mình có tổng cộng bao nhiêu đơn"
        ok1, why1 = validate_rewrite(r1, prop1)
        self.assertFalse(ok1, "Guard must reject drop of 'hqua'")
        self.assertTrue(any(k in why1.lower() for k in ["yesterday", "hôm qua", "hqua"]), f"Expected yesterday/hôm qua in {why1}")

        # Case 1 unaccented variant
        r1_unacc = clean_question("check gium hqua co tong cong bao nhiu don")
        prop1_unacc = "check giup minh co tong cong bao nhieu don"
        ok1_u, why1_u = validate_rewrite(r1_unacc, prop1_unacc)
        self.assertFalse(ok1_u, "Guard must reject drop of 'hqua' in unaccented question")

        # Case 2: question inquiry "bao nhiêu tiền" turned into statement "nhiều tiền"
        r2 = clean_question("xem giup 7 ngay gan nhat cua hang ban dc bnhiu tien")
        prop2 = "Xem giúp 7 ngày gần nhất của hàng bán được nhiều tiền"
        ok2, why2 = validate_rewrite(r2, prop2)
        self.assertFalse(ok2, "Guard must reject converting inquiry to assertion 'nhiều tiền'")
        self.assertIn("intent", why2.lower())

        # Case 2 accented variant
        r2_acc = clean_question("xem giúp 7 ngày gần nhất cửa hàng bán được bao nhiêu tiền")
        prop2_acc = "xem giúp 7 ngày gần nhất cửa hàng bán được nhiều tiền"
        ok2_a, why2_a = validate_rewrite(r2_acc, prop2_acc)
        self.assertFalse(ok2_a, "Guard must reject converting accented inquiry to assertion")

        # Case 3: lost "hôm nay" / "hnay"
        r3 = clean_question("hnay co bao nhiu don")
        prop3 = "hãy có bao nhiêu đơn"
        ok3, why3 = validate_rewrite(r3, prop3)
        self.assertFalse(ok3, "Guard must reject drop of 'hnay'")
        self.assertTrue(any(k in why3.lower() for k in ["today", "hôm nay", "hnay"]), f"Expected today/hôm nay in {why3}")

        # Case 3 unaccented variant
        r3_unacc = clean_question("hnay co bn don")
        prop3_unacc = "hay co bao nhieu don"
        ok3_u, why3_u = validate_rewrite(r3_unacc, prop3_unacc)
        self.assertFalse(ok3_u, "Guard must reject drop of 'hnay' in unaccented variant")

    def test_valid_rewrites_allowed(self):
        """Legitimate safe expansions and diacritic additions must be ALLOWED."""
        c1 = clean_question("check giùm hqua có tổng cộng bao nhiêu đơn")
        ok, _ = validate_rewrite(c1, "check giùm hôm qua có tổng cộng bao nhiêu đơn")
        self.assertTrue(ok, "Safe expansion of hqua -> hôm qua must be accepted")

        c2 = clean_question("xem giup 7 ngay gan nhat cua hang ban dc bnhiu tien")
        ok, _ = validate_rewrite(c2, "xem giúp 7 ngày gần nhất cửa hàng bán được bao nhiêu tiền")
        self.assertTrue(ok, "Safe expansion of bnhiu tien -> bao nhiêu tiền must be accepted")

        c3 = clean_question("hnay co bao nhiu don")
        ok, _ = validate_rewrite(c3, "hôm nay có bao nhiêu đơn")
        self.assertTrue(ok, "Safe expansion of hnay -> hôm nay must be accepted")

        c4 = clean_question("dthu hqua ko tính Otrivin 0,1% chai 10ml", ["Otrivin 0,1%"])
        ok, _ = validate_rewrite(c4, "doanh thu hôm qua không tính Otrivin 0,1% chai 10ml")
        self.assertTrue(ok, "Safe expansion with protected entity must be accepted")

    def test_guard_semantic_invariants(self):
        """Guard must catch dropped negations, filters, ordering, dates, units, and signs."""
        # 1. Dropped negation
        r = clean_question("dthu hqua ko tính hàng khuyến mãi")
        self.assertFalse(validate_rewrite(r, "doanh thu hôm qua tính hàng khuyến mãi")[0])

        # 2. Filter condition altered (trở lên -> trở xuống)
        r_filt = clean_question("đơn hàng từ 50k trở lên")
        self.assertFalse(validate_rewrite(r_filt, "đơn hàng từ 50k trở xuống")[0])

        # 3. Ordering direction flipped (cũ đến mới -> mới đến cũ)
        r_ord = clean_question("doanh thu các tháng từ cũ đến mới")
        self.assertFalse(validate_rewrite(r_ord, "doanh thu các tháng từ mới đến cũ")[0])

        # 4. Date changed (15/08/2025 -> 20/08/2025)
        r_date = clean_question("doanh thu ngày 15/08/2025 là bao nhiêu")
        self.assertFalse(validate_rewrite(r_date, "doanh thu ngày 20/08/2025 là bao nhiêu")[0])

        # 5. Negative sign lost (-500 -> 500)
        r_neg = clean_question("khoản điều chỉnh -500k")
        self.assertFalse(validate_rewrite(r_neg, "khoản điều chỉnh 500k")[0])

        # 6. Leading zero lost in code (00123 -> 123)
        r_code = clean_question("mã khách hàng 00123")
        self.assertFalse(validate_rewrite(r_code, "mã khách hàng 123")[0])

        # 7. Unit altered (10ml -> 20ml)
        r_unit = clean_question("thuốc chai 10ml bán được bao nhiêu")
        self.assertFalse(validate_rewrite(r_unit, "thuốc chai 20ml bán được bao nhiêu")[0])

        # 8. Protected entity missing
        r_ent = clean_question("doanh thu của Sting dâu 330ml", ["Sting dâu 330ml"])
        self.assertFalse(validate_rewrite(r_ent, "doanh thu của nước ngọt")[0])

    def test_ambiguous_chat_words_not_forced(self):
        """Word 'nhiu' alone must NOT be blindly turned into 'nhiều'."""
        r = clean_question("một đơn là nhiu")
        self.assertTrue(r.ambiguous)
        self.assertIn("nhiu", r.ambiguous_spans)
        self.assertNotIn("nhiều", r.normalized)

        # Standalone short chat words must raise ambiguity warning
        for word in ["dt", "kh", "th", "bn"]:
            res = clean_question(f"kiểm tra {word} hôm nay")
            self.assertTrue(res.ambiguous)

    def test_llm_cleaner_mock_and_fallbacks(self):
        """Test LLM cleaner responses, schema validation, ambiguity rejection, and timeouts."""
        with tempfile.TemporaryDirectory() as d:
            fake_client = MagicMock()

            # Scenario A: Valid LLM response passes guard
            valid_resp = {"normalized": "một đơn là bao nhiêu", "uncertain_spans": []}
            fake_client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(valid_resp)))])
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **kw: fake_client)}):
                cleaner = CleanerLLM("test-qwen", "http://127.0.0.1:8000/v1", d)

            r_amb = clean_question("một đơn là nhiu")
            r_out = cleaner.question(r_amb, "ctx_1", time.time() + 60)
            self.assertEqual(r_out.normalized, "một đơn là bao nhiêu")
            self.assertEqual(r_out.llm["status"], "generated")

            # Scenario B: Cache hit
            r_cached = cleaner.question(clean_question("một đơn là nhiu"), "ctx_1", time.time() + 60)
            self.assertEqual(r_cached.llm["status"], "cache")

            # Scenario C: LLM returns uncertain_spans -> Must reject and keep rules normalization
            uncertain_resp = {"normalized": "một đơn là bao nhiêu", "uncertain_spans": ["nhiu"]}
            fake_client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(uncertain_resp)))])
            r_unc = clean_question("một đơn là nhiu")
            r_unc_out = cleaner.question(r_unc, "ctx_2", time.time() + 60)
            self.assertEqual(r_unc_out.llm["status"], "rejected")
            self.assertEqual(r_unc_out.normalized, "một đơn là nhiu")

            # Scenario D: Truncated output (finish_reason != 'stop') -> Fallback
            fake_client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="{}"))])
            r_trunc = clean_question("hnay co bao nhiu don")
            r_trunc_out = cleaner.question(r_trunc, "ctx_3", time.time() + 60)
            self.assertEqual(r_trunc_out.llm["status"], "fallback")

            # Scenario E: Timeout error -> Fallback
            fake_client.with_options.return_value.chat.completions.create.side_effect = TimeoutError("client timeout")
            r_timeout = clean_question("dthu ngay mai")
            r_timeout_out = cleaner.question(r_timeout, "ctx_4", time.time() + 60)
            self.assertEqual(r_timeout_out.llm["status"], "fallback")

            # Scenario F: Budget exhausted
            cleaner.max_calls = 0
            _, rep = cleaner.request("question", {}, "ctx_5", time.time() + 60)
            self.assertEqual(rep["status"], "budget_exhausted")

    def test_cache_key_isolation(self):
        """Cache must not mix results when context_hash or code_hash differs."""
        with tempfile.TemporaryDirectory() as d:
            fake_client = MagicMock()
            resp = {"normalized": "test", "uncertain_spans": []}
            fake_client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(resp)))])
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **kw: fake_client)}):
                c = CleanerLLM("test-qwen", "http://127.0.0.1:8000/v1", d)
            _, rep1 = c.request("question", {"q": 1}, "ctx_A", time.time() + 60)
            _, rep2 = c.request("question", {"q": 1}, "ctx_B", time.time() + 60)
            self.assertNotEqual(rep1["key"], rep2["key"])

    def test_shop_data_provenance_and_failed_parse(self):
        """Verify raw_tables preservation, failed parse tracking, and 'Tống' surname protection."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            # Table with customer "Tống Nam Loan", subtotal rows, and broken date
            df = pd.DataFrame({
                "Khách hàng": ["Tống Nam Loan", "TỔNG CỘNG", "Cộng ngày 01/05", "Trần Văn B"],
                "Ngày": ["01/01/2025", "02/01/2025", "03/01/2025", "sai_ngay"],
                "Tiền": ["2tr300", "5tr000", "5tr000", "45.000đ"],
                "Mã": ["00123", "TOTAL", "SUB", "00456"]
            })
            df.to_csv(p / "donhang.csv", index=False)

            ws = load_workspace(p, cleaner=True)
            self.assertIn("donhang", ws.raw_tables)
            raw = ws.raw_tables["donhang"]
            # Raw data is preserved as ingested
            self.assertEqual(raw["Tiền"].iloc[0], "2tr300")
            self.assertEqual(raw["Mã"].iloc[0], "00123")

            report = ws.cleaning_reports["donhang"]
            self.assertEqual(report["input_rows"], 4)
            # Failed parse is captured in provenance
            failed_col = next((c for c in report["columns"] if c["failed"] > 0), None)
            self.assertIsNotNone(failed_col)
            self.assertIn("sai_ngay", [e["raw"] for e in failed_col["examples"]])

            # Verify surname "Tống" is NOT removed by subtotal regex
            v = raw["Khách hàng"].dropna().astype(str).str.strip().str.lower()
            subtotals = v.str.match(TOTAL_ROW_RE)
            self.assertFalse(subtotals.iloc[0], "'Tống Nam Loan' must NOT be matched as subtotal")
            self.assertTrue(subtotals.iloc[1], "'TỔNG CỘNG' must be matched as subtotal")
            self.assertTrue(subtotals.iloc[2], "'Cộng ngày 01/05' must be matched as subtotal")

    def test_recipe_candidate_receives_original_question(self):
        """Deterministic recipes must always evaluate the original question,
        not the rewritten/normalized text."""
        from datamate.run import recipe_candidate
        ws = Workspace(Path("."))
        q = {
            "id": "q_test",
            "business_id": "biz_test",
            "question": "tổng dthu tuần rồi là bao nhiêu",
            "as_of_date": "2026-05-15",
            "cleaner": {"normalized": "câu hỏi đã bị thay đổi hoàn toàn"}
        }
        # recipe_candidate must read q["question"], not q["cleaner"]["normalized"]
        with patch("datamate.run.metric_candidate") as mock_metric:
            mock_metric.return_value = None
            recipe_candidate(q, ws)
            self.assertEqual(mock_metric.call_args[0][0], q["question"])

    def test_no_gold_leakage_in_cleaner(self):
        """Cleaner payload and results must never reference gold answers."""
        q = clean_question("doanh thu tháng 10")
        d = q.to_dict()
        self.assertNotIn("answer", d)
        self.assertNotIn("gold", d)
        self.assertNotIn("reference", d)

    def test_regression_offset_corruption_multiple_replacements(self):
        """Regression test for Bug 1: Multiple phrase replacements in the same sentence
        must not corrupt offsets or adjacent words."""
        r1 = clean_question("bao nhiu don va bao nhiu khach")
        self.assertEqual(r1.normalized, "bao nhiêu don va bao nhiêu khach")

        r2 = clean_question("Bao nhiu đơn và bao nhiu khách")
        self.assertEqual(r2.normalized, "Bao nhiêu đơn và bao nhiêu khách")

        r3 = clean_question("bao nhiu? bao nhiu!")
        self.assertEqual(r3.normalized, "bao nhiêu? bao nhiêu!")

    def test_regression_guard_semantic_relationships(self):
        """Regression test for Bug 2: Guard must protect temporal directions, shifts,
        intervals, and negation targets."""
        # 1. Temporal direction: sau -> trước
        r_sau = clean_question("sau ngày 01/05/2025")
        ok_sau, why_sau = validate_rewrite(r_sau, "trước ngày 01/05/2025")
        self.assertFalse(ok_sau, "Guard must reject reversing 'sau' to 'trước'")

        # 2. Shift / time-of-day: sáng -> chiều
        r_shift = clean_question("doanh thu ca sáng là bao nhiêu")
        ok_shift, why_shift = validate_rewrite(r_shift, "doanh thu ca chiều là bao nhiêu")
        self.assertFalse(ok_shift, "Guard must reject changing shift/time-of-day from sáng to chiều")

        # 3. Interval inversion: từ A đến B -> từ B đến A
        r_int = clean_question("từ ngày 11/05/2025 đến ngày 20/05/2025")
        ok_int, why_int = validate_rewrite(r_int, "từ ngày 20/05/2025 đến ngày 11/05/2025")
        self.assertFalse(ok_int, "Guard must reject reversing interval endpoints")

        # 4. Negation target swap: không tính tiền ship -> không tính tiền hàng
        r_neg = clean_question("không tính tiền ship")
        ok_neg, why_neg = validate_rewrite(r_neg, "không tính tiền hàng")
        self.assertFalse(ok_neg, "Guard must reject swapping target of negation")

        # 5. Valid rewrite must pass
        ok_valid, why_valid = validate_rewrite(r_sau, "Sau ngày 01/05/2025")
        self.assertTrue(ok_valid, f"Valid case capitalization should pass: {why_valid}")

    def test_regression_llm_missing_uncertain_spans(self):
        """Regression test for Bug 3: LLM response missing required uncertain_spans must be rejected."""
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            cl = CleanerLLM("test-model", "http://127.0.0.1:8000/v1", tempfile.mkdtemp())
            r = clean_question("check giùm hqua có tổng cộng bao nhiêu đơn")
            cl.request = MagicMock(return_value=({"normalized": "check giùm hôm qua có tổng cộng bao nhiêu đơn"}, {"status": "generated"}))
            res = cl.question(r, "ctx", 9999999999)
            self.assertEqual(res.llm.get("status"), "rejected", "Missing uncertain_spans must be rejected")
            self.assertIn("uncertain_spans", res.llm.get("reason", "").lower())

    def test_regression_benchmark_split_allows_sequential_modes(self):
        """Regression test for Bug 4: Benchmark split manifest must separate frozen split
        from run configuration so sequential modes can run on the same split."""
        from scripts.benchmark_cleaner import check_or_save_manifest
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            split_data = dict(seed=42, fold="dev", ids=["q1", "q2"], dataset_fingerprint="abc")
            # First run: modes off, rules
            ok1, err1 = check_or_save_manifest(out_dir, split_data, run_config=dict(modes=["off", "rules"], backend="scripted"))
            self.assertTrue(ok1, f"First run should succeed: {err1}")

            # Second sequential run: mode rules+llm with same split
            ok2, err2 = check_or_save_manifest(out_dir, split_data, run_config=dict(modes=["rules+llm"], backend="scripted"))
            self.assertTrue(ok2, f"Sequential run with same split must succeed: {err2}")

            # Run with altered split (different seed or fingerprint) must fail
            altered_split = dict(seed=99, fold="dev", ids=["q1", "q3"], dataset_fingerprint="xyz")
            ok3, err3 = check_or_save_manifest(out_dir, altered_split, run_config=dict(modes=["rules+llm"], backend="scripted"))
            self.assertFalse(ok3, "Altered split data must be rejected")

    def test_regression_benchmark_review_records_proposals_correctly(self):
        """Regression test for Bug 5: cleaner_review row must record the raw LLM proposal
        when rejected, and record the accepted LLM proposal when cached."""
        from scripts.benchmark_cleaner import format_review_row
        # Rejected proposal
        row_rej = format_review_row(
            qid="q1",
            original="check giùm hqua",
            rules_norm="check giùm hôm qua",
            llm_result=dict(
                proposal="check giúp mình",  # rejected because lost hqua
                decision="rejected",
                source="live",
                reason="time condition changed: lost hôm qua"
            )
        )
        self.assertEqual(row_rej["llm_proposed"], "check giúp mình")
        self.assertEqual(row_rej["final_normalized"], "check giùm hôm qua")
        self.assertEqual(row_rej["guard_status"], "rejected")

        # Accepted cached proposal
        row_cached = format_review_row(
            qid="q2",
            original="bao nhiu don",
            rules_norm="bao nhiêu đơn",
            llm_result=dict(
                proposal="Bao nhiêu đơn",
                decision="accepted",
                source="cache",
                reason="ok"
            )
        )
        self.assertEqual(row_cached["llm_proposed"], "Bao nhiêu đơn")
        self.assertEqual(row_cached["final_normalized"], "Bao nhiêu đơn")
        self.assertEqual(row_cached["decision"], "accepted")
        self.assertEqual(row_cached["source"], "cache")

    def test_regression_guard_attribute_and_relation_protection(self):
        """Regression test for Codex confirmed issues:
        1. Swapping customer attributes (khách mới -> khách cũ).
        2. Swapping metrics between objects (doanh thu cửa hàng A và lợi nhuận cửa hàng B -> lợi nhuận A và doanh thu B).
        3. Swapping order status (đơn hoàn -> đơn huỷ; đơn thành công -> đơn huỷ).
        4. Valid paraphrases that preserve attributes and relations must pass.
        """
        # Case 1: Swapping customer attributes
        r_moi = clean_question("số đơn hôm qua của khách mới")
        ok_moi, why_moi = validate_rewrite(r_moi, "số đơn hôm qua của khách cũ")
        self.assertFalse(ok_moi, "Guard must reject changing 'khách mới' to 'khách cũ'")

        # Case 2: Swapping metrics between objects
        r_rel = clean_question("doanh thu từ cửa hàng A và lợi nhuận từ cửa hàng B")
        ok_rel, why_rel = validate_rewrite(r_rel, "lợi nhuận từ cửa hàng A và doanh thu từ cửa hàng B")
        self.assertFalse(ok_rel, "Guard must reject swapping metrics between stores")

        # Case 3: Order status swaps
        r_hoan = clean_question("tổng số đơn hoàn trong tháng")
        ok_hoan, why_hoan = validate_rewrite(r_hoan, "tổng số đơn huỷ trong tháng")
        self.assertFalse(ok_hoan, "Guard must reject changing 'đơn hoàn' to 'đơn huỷ'")

        r_tc = clean_question("tổng đơn thành công")
        ok_tc, why_tc = validate_rewrite(r_tc, "tổng đơn huỷ")
        self.assertFalse(ok_tc, "Guard must reject changing 'đơn thành công' to 'đơn huỷ'")

        # Case 4: Valid paraphrases preserving attributes and relations
        ok_v1, why_v1 = validate_rewrite(r_moi, "Số lượng đơn hôm qua của khách hàng mới")
        self.assertTrue(ok_v1, f"Valid paraphrase of 'khách hàng mới' must pass: {why_v1}")

        ok_v2, why_v2 = validate_rewrite(r_rel, "Doanh thu từ cửa hàng A, cùng với lợi nhuận từ cửa hàng B")
        self.assertTrue(ok_v2, f"Valid phrasing preserving metric-object binding must pass: {why_v2}")

    def test_regression_benchmark_solver_compatibility(self):
        """Regression test for Codex confirmed issue:
        Benchmark paired comparison must check that solver backend and model match.
        Mismatched solver configs must not be presented as a valid cleaner PA delta."""
        from scripts.benchmark_cleaner import validate_comparison_compatibility
        # Compatible runs: same backend and model
        run_a = {"mode": "off", "backend": "ollama", "model": "qwen3.5-9b-40k", "dataset_fingerprint": "fp1"}
        run_b = {"mode": "rules+llm", "backend": "ollama", "model": "qwen3.5-9b-40k", "dataset_fingerprint": "fp1"}
        compat, reason = validate_comparison_compatibility(run_a, run_b)
        self.assertTrue(compat, f"Compatible runs should pass: {reason}")

        # Incompatible runs: different backend
        run_c = {"mode": "off", "backend": "scripted", "model": None, "dataset_fingerprint": "fp1"}
        compat_diff_backend, reason_diff = validate_comparison_compatibility(run_c, run_b)
        self.assertFalse(compat_diff_backend, "Different solver backends must be flagged as incompatible")

        # Incompatible runs: different model
        run_d = {"mode": "rules", "backend": "ollama", "model": "other-model-7b", "dataset_fingerprint": "fp1"}
        compat_diff_model, reason_model = validate_comparison_compatibility(run_b, run_d)
        self.assertFalse(compat_diff_model, "Different solver models must be flagged as incompatible")

        # Incompatible runs: different dataset fingerprint
        run_e = {"mode": "rules", "backend": "ollama", "model": "qwen3.5-9b-40k", "dataset_fingerprint": "fp2"}
        compat_diff_fp, reason_fp = validate_comparison_compatibility(run_b, run_e)
        self.assertFalse(compat_diff_fp, "Different dataset fingerprints must be flagged as incompatible")

    def test_smoke_cleaner_offline_and_resume(self):
        """Test scripts/smoke_cleaner.py offline execution, checkpoint generation, and resume."""
        from scripts.smoke_cleaner import run_cleaner_smoke
        import csv

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            # Create synthetic questions.jsonl with 35 businesses to cover train/dev/audit splits
            questions = [
                {"id": f"q_{i}", "business_id": f"biz_{i%30:03d}", "question": f"dthu hqua la bnhiu don {i}"}
                for i in range(35)
            ]
            q_file = data_dir / "questions.jsonl"
            q_file.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")

            out_dir = tmp_path / "smoke_out"

            # 1. First run
            res1 = run_cleaner_smoke(
                data_dir=data_dir,
                out_dir=out_dir,
                modes=["rules"],
                limit=3,
                seed=3407,
                fold="dev",
                skip_entities=True,
                resume=True
            )
            self.assertIn("summary", res1)
            self.assertEqual(res1["summary"]["total_questions"], 3)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "cleaner_review.csv").exists())
            self.assertTrue((out_dir / "cleaner_review.jsonl").exists())
            self.assertTrue((out_dir / "smoke_manifest.json").exists())
            checkpoints = list((out_dir / "checkpoints").glob("*.json"))
            self.assertEqual(len(checkpoints), 3)

            # Check review CSV headers and blank auditor columns
            with (out_dir / "cleaner_review.csv").open(encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 3)
                for r in rows:
                    self.assertIn("review_label", r)
                    self.assertIn("error_type", r)
                    self.assertIn("reviewer_notes", r)
                    self.assertEqual(r["review_label"], "")
                    self.assertEqual(r["error_type"], "")
                    self.assertEqual(r["reviewer_notes"], "")
                    self.assertIn("doanh thu", r["rules_normalized"])
                    self.assertIn("hôm qua", r["rules_normalized"])
                    self.assertIn("bao nhiêu", r["rules_normalized"])

            # 2. Second run: resume should load from checkpoints
            res2 = run_cleaner_smoke(
                data_dir=data_dir,
                out_dir=out_dir,
                modes=["rules"],
                limit=3,
                seed=3407,
                fold="dev",
                skip_entities=True,
                resume=True
            )
            self.assertEqual(res2["summary"]["total_questions"], 3)


if __name__ == "__main__":
    unittest.main()
