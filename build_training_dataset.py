"""
DataMate — Zalo AI Challenge 2026
Synthetic Dataset Generator via Rejection Sampling (RFT)

Generates "Golden Trajectories" for SFT and GRPO warm-up:
1. Loads questions and ground-truth answers from train/
2. Generates candidate Python/DuckDB solutions using LLM
3. Executes each candidate in the corresponding business DuckDB database
4. Evaluates result against ground-truth using DataMateEvaluator
5. Stores strictly correct (Prompt, Code, Answer) pairs to sft_dataset.jsonl
"""

import os
import sys
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ingestion import DataIngestionEngine
from src.temporal import TemporalEngine
from src.evaluator import DataMateEvaluator
from run_public_test import PROMPT_TEMPLATE, SYSTEM_PROMPT, PublicTestRunner, extract_target_month_year

def main():
    parser = argparse.ArgumentParser(description="Generate Golden SFT Trajectories via Rejection Sampling")
    parser.add_argument("--data_dir", type=str, default="train", help="Directory containing questions.jsonl, answers.jsonl, and data/")
    parser.add_argument("--output", type=str, default="train/sft_dataset.jsonl", help="Path to output jsonl dataset")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--samples_per_question", type=int, default=2, help="Number of rollouts per question to try")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for LLM inference")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")
    args = parser.parse_args()

    questions_file = os.path.join(args.data_dir, "questions.jsonl")
    answers_file = os.path.join(args.data_dir, "answers.jsonl")
    data_base_dir = os.path.join(args.data_dir, "data")

    if not os.path.exists(questions_file) or not os.path.exists(answers_file):
        print(f"[!] Error: Missing questions or answers file in {args.data_dir}")
        return

    # Load ground-truth answers
    ground_truth: Dict[str, Dict[str, Any]] = {}
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ground_truth[item["id"]] = item

    # Load questions
    questions: List[Dict[str, Any]] = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[:args.limit]

    print(f"[*] Total questions to process: {len(questions)}")
    print(f"[*] Initializing model: {args.model_id}")

    runner = PublicTestRunner(
        model_id=args.model_id,
        load_in_4bit=not args.fp16
    )

    calendar_path = os.path.join(args.data_dir, "calendar.csv")
    temporal_engine = TemporalEngine(calendar_path if os.path.exists(calendar_path) else None)
    ingestion_engine = DataIngestionEngine()

    biz_cache: Dict[str, Any] = {}
    golden_samples = 0
    total_processed = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_mode = "w" if args.overwrite else "a"

    with open(args.output, out_mode, encoding="utf-8") as out_f:
        for q_idx in tqdm(range(0, len(questions), args.batch_size), desc="Mining SFT Trajectories"):
            batch = questions[q_idx : q_idx + args.batch_size]
            batch_prompts = []
            batch_meta = []

            for q in batch:
                q_id = q["id"]
                biz_id = q["business_id"]
                as_of_date = q.get("as_of_date", "2026-08-31")
                question_text = q["question"]

                if q_id not in ground_truth:
                    continue

                if biz_id not in biz_cache:
                    biz_dir = os.path.join(data_base_dir, biz_id)
                    readme_file = os.path.join(biz_dir, "README.md")
                    readme_content = ""
                    if os.path.exists(readme_file):
                        try:
                            with open(readme_file, "r", encoding="utf-8") as rf:
                                readme_content = rf.read()
                        except Exception:
                            pass
                    con, schema_summary = ingestion_engine.load_business_data(biz_dir)
                    biz_cache[biz_id] = (con, schema_summary, readme_content)

                con, schema_summary, readme_content = biz_cache[biz_id]
                dates = temporal_engine.resolve_relative_dates(as_of_date)
                t_month, t_year = extract_target_month_year(question_text, as_of_date)
                month_hint = ""
                if t_month and t_year and con:
                    try:
                        all_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
                        m_keys = [f"t{t_month}_{t_year}", f"t0{t_month}_{t_year}", f"t{t_month:02d}_{t_year}"]
                        matching = [t for t in all_tables if any(k in t for k in m_keys)]
                        if matching:
                            month_hint = f"- GỢI Ý BẢNG THEO THỜI GIAN: Dữ liệu câu hỏi thuộc Tháng {t_month}/{t_year}. Bảng có sẵn: {matching[:3]}."
                    except Exception:
                        pass

                prompt = PROMPT_TEMPLATE.format(
                    biz_id=biz_id,
                    readme_content=readme_content,
                    schema_summary=schema_summary,
                    as_of_date=as_of_date,
                    hom_qua=dates["hom_qua"],
                    tuan_nay_start=dates["tuan_nay"][0],
                    tuan_nay_end=dates["tuan_nay"][1],
                    tuan_roi_start=dates["tuan_roi"][0],
                    tuan_roi_end=dates["tuan_roi"][1],
                    thang_nay_start=dates["thang_nay"][0],
                    thang_nay_end=dates["thang_nay"][1],
                    thang_truoc_start=dates["thang_truoc"][0],
                    thang_truoc_end=dates["thang_truoc"][1],
                    tet_season=f"Từ {dates['tet_season'][0]} đến {dates['tet_season'][1]}" if dates["tet_season"] else "Không xác định",
                    month_hint=month_hint,
                    question=question_text
                )

                batch_prompts.append(prompt)
                batch_meta.append((q_id, biz_id, question_text, prompt, con))

            if not batch_prompts:
                continue

            # Run generation
            raw_responses = runner.generate_code_batch(batch_prompts)

            for (q_id, biz_id, question_text, prompt, con), raw_resp in zip(batch_meta, raw_responses):
                total_processed += 1
                code = runner.extract_python_code(raw_resp)
                if not code:
                    continue

                log_item: Dict[str, Any] = {}
                exec_result = runner.execute_code(code, con, question_text, log_item)
                if not exec_result or exec_result.get("type") == "abstain":
                    continue

                gt = ground_truth[q_id]
                ref_type = gt.get("type")
                ref_val = gt.get("answer")
                ref_aliases = gt.get("alias", [])

                pred_type = exec_result.get("type")
                pred_val = exec_result.get("answer")

                try:
                    is_correct = DataMateEvaluator.match_answer(
                        pred_type=pred_type,
                        pred_val=pred_val,
                        ref_type=ref_type,
                        ref_val=ref_val,
                        ref_aliases=ref_aliases
                    )
                except Exception:
                    is_correct = False

                if is_correct:
                    golden_samples += 1
                    sft_entry = {
                        "id": q_id,
                        "business_id": biz_id,
                        "question": question_text,
                        "prompt": prompt,
                        "completion": f"```python\n{code}\n```",
                        "answer_type": ref_type,
                        "answer": ref_val
                    }
                    out_f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
                    out_f.flush()

    print(f"\n[+] Mining Completed! Mined {golden_samples}/{total_processed} ({golden_samples*100.0/max(1, total_processed):.1f}%) Golden SFT Trajectories.")
    print(f"[+] Output saved to: {args.output}")

if __name__ == "__main__":
    main()
