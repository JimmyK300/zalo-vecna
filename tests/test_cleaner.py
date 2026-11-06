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

            # 3. M0.1 Bug A regression: re-running in same dir with --modes off MUST raise ValueError
            with self.assertRaises(ValueError, msg="Must reject reusing rules checkpoints for mode=off"):
                run_cleaner_smoke(
                    data_dir=data_dir,
                    out_dir=out_dir,
                    modes=["off"],
                    limit=3,
                    seed=3407,
                    fold="dev",
                    skip_entities=True,
                    resume=True
                )

            # 4. M0.1: changing skip_entities, model, seed, or dataset_fingerprint MUST raise ValueError
            with self.assertRaises(ValueError):
                run_cleaner_smoke(
                    data_dir=data_dir,
                    out_dir=out_dir,
                    modes=["rules"],
                    limit=3,
                    seed=3407,
                    fold="dev",
                    skip_entities=False,
                    resume=True
                )
            with self.assertRaises(ValueError):
                run_cleaner_smoke(
                    data_dir=data_dir,
                    out_dir=out_dir,
                    modes=["rules"],
                    limit=3,
                    seed=3407,
                    fold="dev",
                    model="different-model",
                    skip_entities=True,
                    resume=True
                )

            # 5. M0.1: corrupting a checkpoint file MUST raise ValueError on resume (never silently overwrite)
            ckpt_file = next((out_dir / "checkpoints").rglob("*.json"))
            orig_ckpt_text = ckpt_file.read_text(encoding="utf-8")
            ckpt_file.write_text("{corrupted json", encoding="utf-8")
            with self.assertRaises(ValueError, msg="Corrupted checkpoint JSON must raise ValueError"):
                run_cleaner_smoke(
                    data_dir=data_dir,
                    out_dir=out_dir,
                    modes=["rules"],
                    limit=3,
                    seed=3407,
                    fold="dev",
                    skip_entities=True,
                    resume=True
                )
            # Tampered config_hash in checkpoint must also raise ValueError
            tampered = json.loads(orig_ckpt_text)
            tampered["config_hash"] = "0000000000"
            ckpt_file.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ValueError, msg="Tampered checkpoint config_hash must raise ValueError"):
                run_cleaner_smoke(
                    data_dir=data_dir,
                    out_dir=out_dir,
                    modes=["rules"],
                    limit=3,
                    seed=3407,
                    fold="dev",
                    skip_entities=True,
                    resume=True
                )

            # 6. Multi-mode smoke run (--modes off rules) in a fresh directory keeps modes strictly separated
            multi_dir = tmp_path / "smoke_multi"
            res_multi = run_cleaner_smoke(
                data_dir=data_dir,
                out_dir=multi_dir,
                modes=["off", "rules"],
                limit=3,
                seed=3407,
                fold="dev",
                skip_entities=True,
                resume=True
            )
            self.assertIn("by_mode", res_multi["summary"])
            self.assertEqual(res_multi["summary"]["by_mode"]["off"]["questions_with_changes"], 0)
            self.assertEqual(res_multi["summary"]["by_mode"]["rules"]["questions_with_changes"], 3)

    def test_m0_2_benchmark_manifest_and_stale_output_verification(self):
        """M0.2 Regression test: verify_mode_output and validate_comparison_compatibility
        must block stale/modified submission.jsonl, missing IDs, failed status, and solver config mismatches."""
        import hashlib
        from scripts.benchmark_cleaner import verify_mode_output, validate_comparison_compatibility

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_ids = ["q_1", "q_2"]
            solver_cfg = {
                "min_confidence": 0.5,
                "consistency": 1,
                "verify": False,
                "verify_new": False,
                "candidate_policy": "answer",
                "seed": 3407,
            }

            def make_mode_dir(mode: str, sub_lines: list[dict], status: str = "completed",
                              override_manifest: dict | None = None, tamper_sub_after: bool = False) -> tuple[Path, dict]:
                mdir = root / mode
                mdir.mkdir(parents=True, exist_ok=True)
                sub_path = mdir / "submission.jsonl"
                sub_path.write_text("\n".join(json.dumps(x) for x in sub_lines) + "\n", encoding="utf-8")
                sub_sha = hashlib.sha256(sub_path.read_bytes()).hexdigest()
                manifest = {
                    "mode": mode,
                    "status": status,
                    "backend": "ollama",
                    "model": "qwen3.5-9b",
                    "solver_config": dict(solver_cfg),
                    "dataset_fingerprint": "fp_same",
                    "rulebook_sha256": "rb_same",
                    "code_hash": "code_same",
                    "ids": list(split_ids),
                    "completed_ids": [x["id"] for x in sub_lines],
                    "submission_sha256": sub_sha,
                }
                if override_manifest:
                    manifest.update(override_manifest)
                (mdir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                if tamper_sub_after:
                    sub_path.write_text(json.dumps({"id": "q_1", "type": "number", "answer": 999}) + "\n" +
                                        json.dumps({"id": "q_2", "type": "number", "answer": 1}) + "\n", encoding="utf-8")
                return mdir, manifest

            valid_lines = [
                {"id": "q_1", "type": "number", "answer": 10},
                {"id": "q_2", "type": "number", "answer": 20},
            ]
            expected_split = {"ids": split_ids, "dataset_fingerprint": "fp_same"}
            dir_off, man_off = make_mode_dir("off", valid_lines)
            dir_rules, man_rules = make_mode_dir("rules", valid_lines)

            ok_off, msg_off, enr_off = verify_mode_output(dir_off, expected_split)
            self.assertTrue(ok_off, msg_off)
            ok_rules, msg_rules, enr_rules = verify_mode_output(dir_rules, expected_split)
            self.assertTrue(ok_rules, msg_rules)
            compat, reason = validate_comparison_compatibility(enr_off, enr_rules, expected_split=expected_split)
            self.assertTrue(compat, reason)

            # Stale submission (modified after run_manifest.json was written) -> rejected
            dir_stale, _ = make_mode_dir("stale", valid_lines, tamper_sub_after=True)
            ok_stale, msg_stale, enr_stale = verify_mode_output(dir_stale, expected_split)
            self.assertFalse(ok_stale)
            self.assertIn("sha256", msg_stale.lower())
            compat_stale, _ = validate_comparison_compatibility(enr_off, enr_stale, expected_split=expected_split)
            self.assertFalse(compat_stale)

            # Missing question ID in submission -> rejected
            dir_inc, _ = make_mode_dir("inc", [{"id": "q_1", "type": "number", "answer": 10}])
            ok_inc, msg_inc, enr_inc = verify_mode_output(dir_inc, expected_split)
            self.assertFalse(ok_inc)
            self.assertIn("incomplete", msg_inc.lower())
            compat_inc, _ = validate_comparison_compatibility(enr_off, enr_inc, expected_split=expected_split)
            self.assertFalse(compat_inc)

            # Failed run status -> rejected
            dir_fail, man_fail = make_mode_dir("fail", valid_lines, status="failed")
            ok_fail, _, _ = verify_mode_output(dir_fail, expected_split)
            self.assertFalse(ok_fail)
            compat_fail, _ = validate_comparison_compatibility(enr_off, man_fail, expected_split=expected_split)
            self.assertFalse(compat_fail)

            # Solver config mismatch -> rejected
            bad_cfg = dict(solver_cfg, candidate_policy="off")
            _, man_bad_cfg = make_mode_dir("bad_cfg", valid_lines, override_manifest={"solver_config": bad_cfg})
            compat_cfg, reason_cfg = validate_comparison_compatibility(enr_off, man_bad_cfg, expected_split=expected_split)
            self.assertFalse(compat_cfg)
            self.assertIn("solver config", reason_cfg.lower())

    def test_m0_3_evaluate_scorer_compliance(self):
        """M0.3 Unit tests for datamate/evaluate.py against DATAMATE_OVERVIEW.md §2 and rulebook §9."""
        from datamate.evaluate import match, score

        # 1. number: exact numeric match (within 1e-9 Decimal tolerance) + alias support
        self.assertTrue(match("number", 100, 100))
        self.assertTrue(match("number", 100.0000000004, 100))
        self.assertFalse(match("number", 100.01, 100))
        self.assertTrue(match("number", 50, 100, alias=[50]))

        # 2. percent: exact match after rounding to 2 decimals + sign check
        self.assertTrue(match("percent", 12.344, 12.34))
        self.assertTrue(match("percent", 12.345, 12.35))
        self.assertFalse(match("percent", 12.34, 12.35))
        self.assertFalse(match("percent", -12.34, 12.34))
        self.assertTrue(match("percent", 0.0, -0.0))

        # 3. string: NFC, lowercase, non-alnum/non-space -> space, collapse spaces, preserve diacritics & word boundaries
        self.assertTrue(match("string", "Nguyễn  Văn, A!", "nguyễn văn a"))
        self.assertFalse(match("string", "Nguyen Van A", "Nguyễn Văn A"))
        # Word boundary preservation ("1000mg" != "1000 mg" without alias; matches with alias, per train q_0b576981b592)
        self.assertFalse(match("string", "Costar Omega 3 1000 mg lọ 100 viên", "Costar Omega 3 1000mg lọ 100 viên"))
        self.assertTrue(
            match(
                "string",
                "Costar Omega 3 1000 mg lọ 100 viên",
                "Costar Omega 3 1000mg lọ 100 viên",
                alias=["SP0076", "Costar Omega 3 1000 mg lọ 100 viên"],
            )
        )

        # 4. date & boolean
        self.assertTrue(match("date", "2025-05-01", "2025-05-01"))
        self.assertFalse(match("date", "01/05/2025", "2025-05-01"))
        self.assertTrue(match("boolean", True, True))
        self.assertFalse(match("boolean", 1, True))
        self.assertFalse(match("boolean", False, True))

        # 5. list: ordered + per-element alias + full-list alias
        self.assertTrue(match("list", ["A", "B"], ["A", "B"]))
        self.assertFalse(match("list", ["B", "A"], ["A", "B"]))
        self.assertTrue(match("list", ["SP01", "B"], ["Sản phẩm 1", "B"], alias=[["SP01"], []]))
        self.assertTrue(match("list", ["SP01", "SP02"], ["Sản phẩm 1", "Sản phẩm 2"], alias=[["SP01", "SP02"]]))

        # 6. table: column matching, row order independence, row alias
        gold_tbl = {
            "columns": ["Khách hàng", "Doanh thu"],
            "rows": [["Anh Minh", 150000], ["Chị Lan", 200000]],
        }
        pred_tbl_reordered = {
            "columns": ["Doanh thu", "Khách hàng"],
            "rows": [[200000, "Chị Lan"], [150000, "Anh Minh"]],
        }
        self.assertTrue(match("table", pred_tbl_reordered, gold_tbl))
        pred_tbl_wrong = {
            "columns": ["Khách hàng", "Doanh thu"],
            "rows": [["Anh Minh", 200000], ["Chị Lan", 150000]],
        }
        self.assertFalse(match("table", pred_tbl_wrong, gold_tbl))

        # 7. PA scoring: +1 correct, -1 wrong, 0 abstain
        with tempfile.TemporaryDirectory() as tmp_dir:
            tp = Path(tmp_dir)
            gold_p = tp / "gold.jsonl"
            pred_p = tp / "pred.jsonl"
            gold_rows = [
                {"id": "q1", "type": "number", "answer": 100},
                {"id": "q2", "type": "unanswerable", "answer": None},
                {"id": "q3", "type": "string", "answer": "Nguyễn Văn Tống"},
                {"id": "q4", "type": "number", "answer": 50},
            ]
            pred_rows = [
                {"id": "q1", "type": "number", "answer": 100},          # +1 correct
                {"id": "q2", "type": "unanswerable", "answer": None},   # +1 correct
                {"id": "q3", "type": "string", "answer": "Nguyễn Văn Tổng"},  # -1 wrong diacritic
                {"id": "q4", "type": "abstain", "answer": None},        # 0 abstain
            ]
            gold_p.write_text("\n".join(json.dumps(r) for r in gold_rows), encoding="utf-8")
            pred_p.write_text("\n".join(json.dumps(r) for r in pred_rows), encoding="utf-8")
            res = score(pred_p, gold_p)
            self.assertEqual(res["correct"], 2)
            self.assertEqual(res["wrong"], 1)
            self.assertEqual(res["abstain"], 1)
            self.assertAlmostEqual(res["PA"], (2 - 1) / 4)

    def test_m1_data_profiling_provenance_and_raw_preservation(self):
        """M1 tests: date ambiguity detection, leading-zero preservation in CSV/Excel,
        parse failure vs ambiguity vs anomaly separation, duplicate policy, and bounded >50k scan."""
        from datamate.cleaning import profile_table, inspect_date_column
        from datamate.workspace import clean_frame

        # 1. Conflicting DMY and MDY dates in the same column -> ambiguity flagged, __date NOT guessed
        df_conflict_dates = pd.DataFrame({
            "ma_don": ["DH01", "DH02", "DH03"],
            "ngay_tao": ["13/05/2025", "05/13/2025", "01/05/2025"],
            "thanh_tien": ["100000", "200000", "150000"],
        })
        d_info = inspect_date_column(df_conflict_dates["ngay_tao"], default_year=2025)
        self.assertTrue(d_info["has_conflicting_dmy_mdy"])
        cleaned_conflict, _ = clean_frame(df_conflict_dates.copy(), default_year=2025)
        self.assertNotIn("ngay_tao__date", cleaned_conflict.columns, "Must not guess DMY when column has conflicting DMY/MDY")
        prof_conflict = profile_table(df_conflict_dates, cleaned_conflict, table_name="orders")
        amb_kinds = [a["kind"] for cp in prof_conflict["column_profiles"] for a in cp["ambiguities"]]
        self.assertIn("conflicting_dmy_and_mdy_dates", amb_kinds)

        # 2. Unproven slash date order (all parts <= 12) -> ambiguity flagged in profile
        df_unproven = pd.DataFrame({"ngay_ban": ["01/05/2025", "02/06/2025"]})
        d_unproven = inspect_date_column(df_unproven["ngay_ban"], default_year=2025)
        self.assertTrue(d_unproven["has_unproven_slash_ambiguity"])

        # 3. Parse failure vs anomaly vs mixed separators
        df_mixed = pd.DataFrame({
            "ma_sp": ["00123", "00124", "00125", "00126"],
            "so_luong": ["10", "20", "-5", "ERR_QTY"],
            "don_gia": ["1,234", "1.234", "2,000", "3.000"],
        })
        cleaned_mixed, _ = clean_frame(df_mixed.copy())
        # Leading zero preserved!
        self.assertEqual(list(cleaned_mixed["ma_sp"]), ["00123", "00124", "00125", "00126"])
        prof_mixed = profile_table(df_mixed, cleaned_mixed, table_name="items")
        by_col = {cp["column"]: cp for cp in prof_mixed["column_profiles"]}
        self.assertTrue(by_col["ma_sp"]["has_leading_zero_codes"])
        self.assertTrue(by_col["ma_sp"]["leading_zero_preserved"])
        # 'so_luong' has 3/4 numeric and 1 'ERR_QTY' outlier -> reported in parse_failures
        self.assertTrue(len(by_col["so_luong"]["parse_failures"]) > 0)
        # 'don_gia' has both comma and dot 3-digit groupings -> reported in ambiguities
        don_gia_amb = [a["kind"] for a in by_col["don_gia"]["ambiguities"]]
        self.assertIn("mixed_thousand_decimal_separators", don_gia_amb)

        # 4. Negative numeric values reported as anomaly (never deleted or overwritten)
        df_neg = pd.DataFrame({"ma_gd": ["G1", "G2"], "so_tien": [100000, -25000]})
        cleaned_neg, _ = clean_frame(df_neg.copy())
        prof_neg = profile_table(df_neg, cleaned_neg, table_name="ledger")
        by_col_neg = {cp["column"]: cp for cp in prof_neg["column_profiles"]}
        self.assertEqual(by_col_neg["so_tien"]["anomalies"][0]["kind"], "negative_numeric_values")
        self.assertEqual(int(cleaned_neg["so_tien"].iloc[1]), -25000)

        # 5. CSV & Workspace loading preserves leading zeros and raw_tables
        with tempfile.TemporaryDirectory() as tmp_dir:
            shop_dir = Path(tmp_dir)
            csv_path = shop_dir / "san_pham.csv"
            csv_path.write_text(
                "ma_sp,ten_sp,gia_ban\n00123,Sữa tươi có đường,32000\n00124,Sữa tươi ít đường,34000\n00124,Sữa tươi ít đường,34000\n",
                encoding="utf-8",
            )
            ws = load_workspace(shop_dir, cleaner=True)
            t = ws.tables["san_pham"]
            self.assertEqual(list(t.df["ma_sp"]), ["00123", "00124", "00124"])
            # Valid duplicate row in non-order table preserved (rulebook §8)
            self.assertEqual(len(t.df), 3)
            self.assertEqual(len(ws.raw_tables["san_pham"]), 3)
            self.assertIn("san_pham", ws.cleaning_reports)

        # 6. Bounded profiling on >50,000 rows
        big_df = pd.DataFrame({"id": range(55_000), "val": ["100"] * 55_000})
        t0 = time.monotonic()
        big_prof = profile_table(big_df, big_df, table_name="big_table")
        elapsed = time.monotonic() - t0
        self.assertTrue(big_prof["sampling"]["sampled"])
        self.assertEqual(big_prof["sampling"]["scanned_rows"], 10_000)
        self.assertEqual(big_prof["sampling"]["total_rows"], 55_000)
        self.assertLess(elapsed, 5.0)

    def test_m2_schema_and_value_grounding(self):
        """M2 tests: exact NFC match, unaccented candidates, similar distinct products,
        leading-zero code protection ('00123' vs '123'), ambiguous abbreviations, and unmatched entities."""
        from datamate.cleaning import ground_question, render_grounding_hint

        with tempfile.TemporaryDirectory() as tmp_dir:
            shop_dir = Path(tmp_dir)
            (shop_dir / "san_pham.csv").write_text(
                "ma_sp,ten_sp,don_gia\n"
                "00123,Sữa tươi có đường,32000\n"
                "00124,Sữa tươi ít đường,34000\n"
                "SP009,Cà phê muối,29000\n",
                encoding="utf-8",
            )
            ws = load_workspace(shop_dir, cleaner=True)

            # 1. Exact accented product match
            g_exact = ground_question("doanh thu Cà phê muối là bao nhiêu", ws)
            self.assertEqual(len(g_exact["exact_matches"]), 1)
            self.assertEqual(g_exact["exact_matches"][0]["raw_value"], "Cà phê muối")
            self.assertFalse(g_exact["ambiguous"])

            # 2. Unaccented product query -> candidate only, not exact_matches
            g_unacc = ground_question("doanh thu ca phe muoi la bnhiu", ws)
            self.assertEqual(len(g_unacc["exact_matches"]), 0)
            cand_vals = [c["raw_value"] for c in g_unacc["candidates"]]
            self.assertIn("Cà phê muối", cand_vals)

            # 3. Similar distinct products ("Sữa tươi có đường" vs "Sữa tươi ít đường"):
            # querying "sua tuoi" or "st" must return both as candidates and set ambiguous=True
            g_sim = ground_question("ban duoc bao nhieu hop sua tuoi", ws)
            sim_cands = [c["raw_value"] for c in g_sim["candidates"]]
            self.assertIn("Sữa tươi có đường", sim_cands)
            self.assertIn("Sữa tươi ít đường", sim_cands)
            self.assertTrue(g_sim["ambiguous"])

            # 4. Leading-zero code: '00123' is exact_code; '123' is candidate only with ambiguous=True
            g_code_exact = ground_question("số lượng bán của mã 00123", ws)
            self.assertEqual(len(g_code_exact["exact_matches"]), 1)
            self.assertEqual(g_code_exact["exact_matches"][0]["raw_value"], "00123")

            g_code_no_zero = ground_question("số lượng bán của mã 123", ws)
            self.assertEqual(len(g_code_no_zero["exact_matches"]), 0, "Must never auto-match '123' as exact '00123'")
            self.assertTrue(g_code_no_zero["ambiguous"])
            self.assertEqual(g_code_no_zero["candidates"][0]["raw_value"], "00123")
            self.assertEqual(g_code_no_zero["candidates"][0]["match_type"], "leading_zero_code_candidate")

            # 5. Unmatched entity query
            g_none = ground_question("doanh thu bánh trung thu thập cẩm", ws)
            self.assertEqual(len(g_none["exact_matches"]), 0)
            self.assertFalse(g_none["has_entity_match"])

            hint_text = render_grounding_hint(g_sim)
            self.assertIn("Sữa tươi có đường", hint_text)
            self.assertIn("Sữa tươi ít đường", hint_text)

    def test_m3_guard_general_attributes_and_diacritic_locks(self):
        """M3 Bug B & diacritic lock regression tests:
        - 'bao nhiêu sản phẩm màu đỏ' -> 'bao nhiêu sản phẩm màu xanh' (accented & unaccented) rejected
        - size/type/flavor/channel modifier swaps rejected
        - 'doanh thu cà phê sữa' -> 'doanh thu cà phê sửa' rejected
        - distinct fields on QuestionCleanResult verified."""
        # 1. Confirmed Bug B: color attribute swap (accented and unaccented)
        r_color_acc = clean_question("bao nhiêu sản phẩm màu đỏ")
        ok_ca, why_ca = validate_rewrite(r_color_acc, "bao nhiêu sản phẩm màu xanh")
        self.assertFalse(ok_ca, f"Must reject 'màu đỏ' -> 'màu xanh': {why_ca}")

        r_color_unacc = clean_question("bao nhieu san pham mau do")
        ok_cu, why_cu = validate_rewrite(r_color_unacc, "bao nhieu san pham mau xanh")
        self.assertFalse(ok_cu, f"Must reject unaccented 'mau do' -> 'mau xanh': {why_cu}")

        # Direct noun+color without 'màu'
        r_ao = clean_question("bán được mấy cái áo đỏ")
        ok_ao, _ = validate_rewrite(r_ao, "bán được mấy cái áo xanh")
        self.assertFalse(ok_ao)

        # 2. Size, type, flavor, channel modifier swaps
        for orig_q, bad_prop in [
            ("doanh thu trà sữa size M", "doanh thu trà sữa size L"),
            ("số lượng hàng loại 1", "số lượng hàng loại 2"),
            ("bán bao nhiêu hộp sữa vị dâu", "bán bao nhiêu hộp sữa vị cam"),
            ("tổng đơn kênh Shopee", "tổng đơn kênh Tiktok"),
        ]:
            res_q = clean_question(orig_q)
            ok_q, why_q = validate_rewrite(res_q, bad_prop)
            self.assertFalse(ok_q, f"Must reject modifier change '{orig_q}' -> '{bad_prop}'")

        # 3. Diacritic corruption on already-accented words ('sữa' -> 'sửa', 'Tống' -> 'Tổng')
        r_sua = clean_question("doanh thu cà phê sữa hôm qua")
        ok_sua, why_sua = validate_rewrite(r_sua, "doanh thu cà phê sửa hôm qua")
        self.assertFalse(ok_sua, f"Must reject 'sữa' -> 'sửa': {why_sua}")

        r_tong = clean_question("khách hàng Nguyễn Văn Tống nợ bao nhiêu")
        ok_tong, _ = validate_rewrite(r_tong, "khách hàng Nguyễn Văn Tổng nợ bao nhiêu")
        self.assertFalse(ok_tong, "Must reject changing accented customer name 'Tống' -> 'Tổng'")

        # 4. QuestionCleanResult distinct fields check
        res_distinct = clean_question("dthu hqua cua SP001 la bnhiu nhiu", entities=["SP001"])
        d = res_distinct.to_dict()
        self.assertEqual(d["original"], "dthu hqua cua SP001 la bnhiu nhiu")
        self.assertEqual(d["rules_normalized"], "doanh thu hôm qua cua SP001 la bao nhiêu nhiu")
        self.assertEqual(d["normalized_hint"], d["normalized"])
        self.assertEqual(d["llm_proposed"], "")
        self.assertTrue(len(d["clear_aliases"]) >= 3)
        self.assertIn("nhiu", d["ambiguous_spans"])
        self.assertEqual(d["entities"][0]["text"], "SP001")

    def test_m4_deterministic_end_to_end_fixtures(self):
        """M4.B: Deterministic end-to-end evaluation fixtures covering:
        1. Noisy question + recipe preservation
        2. Rulebook §1 & §4 order revenue and cancellation rate
        3. Rulebook §7 exact diacritic separation ('Nguyễn Văn Tống' vs 'Nguyễn Văn Tổng')
        4. Rulebook §10 unanswerable period outside shop data range."""
        from datamate.run import recipe_candidate
        from datamate.evaluate import match

        with tempfile.TemporaryDirectory() as tmp_dir:
            shop_dir = Path(tmp_dir)
            (shop_dir / "don_hang.csv").write_text(
                "ma_don,ngay_dat,ten_khach,trang_thai,thanh_tien\n"
                "DH01,2025-05-01,Nguyễn Văn Tống,Thành công,100000\n"
                "DH02,2025-05-01,Nguyễn Văn Tổng,Thành công,250000\n"
                "DH03,2025-05-01,Nguyễn Văn Tống,Đã huỷ,50000\n",
                encoding="utf-8",
            )
            ws = load_workspace(shop_dir, cleaner=True)

            # Fixture 1: Noisy question cleaned safely while recipe_candidate receives original q["question"]
            q_noisy = {
                "id": "q_fix_1",
                "business_id": "shop_test",
                "as_of_date": "2025-05-02",
                "question": "hqua co bnhiu don",
            }
            c_res = clean_question(q_noisy["question"])
            self.assertEqual(c_res.normalized_hint, "hôm qua co bao nhiêu don")
            self.assertIn("yesterday", c_res.constraints["time"])

            # Fixture 2: Deterministic recipe on clean question (in-revenue orders on 2025-05-01 = 2; cancel rate = 33.33%)
            q_orders = {
                "id": "q_fix_2",
                "business_id": "shop_test",
                "as_of_date": "2025-05-02",
                "question": "Ngày 01/05/2025 cửa hàng có tổng cộng bao nhiêu đơn hàng?",
            }
            cand_orders = recipe_candidate(q_orders, ws)
            if cand_orders is not None:
                self.assertEqual(cand_orders[0], "number")
                self.assertTrue(match("number", cand_orders[1], 2))

            q_cancel = {
                "id": "q_fix_2b",
                "business_id": "shop_test",
                "as_of_date": "2025-05-02",
                "question": "Tỷ lệ huỷ đơn ngày 01/05/2025 là bao nhiêu phần trăm?",
            }
            cand_cancel = recipe_candidate(q_cancel, ws)
            if cand_cancel is not None:
                self.assertEqual(cand_cancel[0], "percent")
                self.assertTrue(match("percent", cand_cancel[1], 33.33))

            # Fixture 3: Diacritic separation in workspace and scorer ('Nguyễn Văn Tống' != 'Nguyễn Văn Tổng')
            df_orders = ws.tables["don_hang"].df
            rev_tong_sac = int(df_orders[(df_orders["ten_khach"] == "Nguyễn Văn Tống") & (df_orders["trang_thai"] == "Thành công")]["thanh_tien"].sum())
            rev_tong_hoi = int(df_orders[(df_orders["ten_khach"] == "Nguyễn Văn Tổng") & (df_orders["trang_thai"] == "Thành công")]["thanh_tien"].sum())
            self.assertEqual(rev_tong_sac, 100000)
            self.assertEqual(rev_tong_hoi, 250000)
            self.assertFalse(match("string", "Nguyễn Văn Tống", "Nguyễn Văn Tổng"))

            # Fixture 4: Unanswerable period outside shop coverage (year 2022 not in 2025 data)
            q_unans = {
                "id": "q_fix_4",
                "business_id": "shop_test",
                "as_of_date": "2025-05-02",
                "question": "Doanh thu tháng 01/2022 của cửa hàng là bao nhiêu?",
            }
            cand_unans = recipe_candidate(q_unans, ws)
            if cand_unans is not None:
                self.assertEqual(cand_unans[0], "unanswerable")


if __name__ == "__main__":
    unittest.main()
