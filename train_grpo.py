"""
DataMate — Zalo AI Challenge 2026
Group Relative Policy Optimization (GRPO) Reinforcement Learning

Trains Qwen2.5-Coder-7B with GRPO using Rule-Based Verifiable Rewards:
1. Generates G candidate completions per prompt
2. Executes each completion dynamically in DuckDB
3. Evaluates execution result against ground-truth answers (answers.jsonl)
4. Computes group-normalized advantage and updates policy via PPO-style objective
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, Any, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ingestion import DataIngestionEngine
from src.temporal import TemporalEngine
from src.evaluator import DataMateEvaluator
from run_public_test import PROMPT_TEMPLATE, SYSTEM_PROMPT, CodeExecutor, extract_target_month_year

# Global engine and DB cache for fast in-loop rewards
INGESTION_ENGINE = DataIngestionEngine()
BIZ_CON_CACHE: Dict[str, Any] = {}
DATA_BASE_DIR = ""

def get_or_load_con(biz_id: str):
    global BIZ_CON_CACHE, DATA_BASE_DIR, INGESTION_ENGINE
    if biz_id not in BIZ_CON_CACHE:
        biz_dir = os.path.join(DATA_BASE_DIR, biz_id)
        con, _ = INGESTION_ENGINE.load_business_data(biz_dir)
        BIZ_CON_CACHE[biz_id] = con
    return BIZ_CON_CACHE[biz_id]

def extract_python_code(response: str) -> Optional[str]:
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match_generic = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
    return None

def duckdb_outcome_reward_func(prompts, completions, business_id, question_text, ref_type, ref_val, ref_aliases, **kwargs) -> List[float]:
    """
    Primary Verifiable Reward:
    +1.0 : Code runs and output EXACTLY matches ground truth
    +0.2 : Code runs without error but output does not match
    -0.5 : Code crashes (DuckDB Binder/Syntax/Runtime error)
    -0.5 : No valid python code block found
    """
    rewards = []
    for comp, biz_id, q_text, r_type, r_val, r_alias in zip(completions, business_id, question_text, ref_type, ref_val, ref_aliases):
        # Extract completion text if dict/messages
        if isinstance(comp, list):
            comp_text = comp[-1]["content"] if comp else ""
        elif isinstance(comp, dict):
            comp_text = comp.get("content", "")
        else:
            comp_text = str(comp)

        code = extract_python_code(comp_text)
        if not code:
            rewards.append(-0.5)
            continue

        try:
            con = get_or_load_con(biz_id)
            log_item: Dict[str, Any] = {}
            exec_res = CodeExecutor.execute_code(code, con, q_text, log_item)

            if not exec_res or exec_res.get("type") == "abstain":
                if log_item.get("exec_error"):
                    rewards.append(-0.5)  # Code execution crashed
                else:
                    rewards.append(-0.1)  # Abstain / Empty result
                continue

            pred_type = exec_res.get("type")
            pred_val = exec_res.get("answer")

            is_correct = DataMateEvaluator.match_answer(
                pred_type=pred_type,
                pred_val=pred_val,
                ref_type=r_type,
                ref_val=r_val,
                ref_aliases=r_alias if r_alias else []
            )

            if is_correct:
                rewards.append(1.0)
            else:
                rewards.append(0.2)  # Partial credit for syntactically executable code

        except Exception:
            rewards.append(-0.5)

    return rewards

def format_compliance_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    Format Reward:
    +0.2 : Properly formats code with ```python ... ``` and assigns `result = ...`
     0.0 : Otherwise
    """
    rewards = []
    for comp in completions:
        comp_text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
        has_code_block = bool(re.search(r'```python.*```', comp_text, re.DOTALL))
        has_result_assign = bool(re.search(r'\bresult\s*=', comp_text))
        if has_code_block and has_result_assign:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards

def main():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-Coder with GRPO")
    parser.add_argument("--data_dir", type=str, default="train", help="Path to train directory")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional SFT LoRA checkpoint to warm-start GRPO")
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo_adapter", help="Output directory for RL adapter")
    parser.add_argument("--num_generations", type=int, default=4, help="Group size G for relative reward normalization")
    parser.add_argument("--epochs", type=int, default=2, help="Number of RL epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="GRPO policy learning rate")
    parser.add_argument("--max_prompt_length", type=int, default=1536, help="Max prompt length")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of train questions")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    args = parser.parse_args()

    global DATA_BASE_DIR
    DATA_BASE_DIR = os.path.join(args.data_dir, "data")

    questions_file = os.path.join(args.data_dir, "questions.jsonl")
    answers_file = os.path.join(args.data_dir, "answers.jsonl")

    if not os.path.exists(questions_file) or not os.path.exists(answers_file):
        print(f"[!] Error: Missing questions or answers in {args.data_dir}")
        return

    # Load ground-truth answers
    ground_truth = {}
    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ground_truth[item["id"]] = item

    # Load questions
    questions = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if args.limit:
        questions = questions[:args.limit]

    print(f"[*] Preparing GRPO dataset for {len(questions)} questions...")

    calendar_path = os.path.join(args.data_dir, "calendar.csv")
    temporal_engine = TemporalEngine(calendar_path if os.path.exists(calendar_path) else None)

    dataset_rows = []
    for q in questions:
        q_id = q["id"]
        if q_id not in ground_truth:
            continue
        gt = ground_truth[q_id]
        biz_id = q["business_id"]
        as_of_date = q.get("as_of_date", "2026-08-31")
        question_text = q["question"]

        biz_dir = os.path.join(DATA_BASE_DIR, biz_id)
        readme_file = os.path.join(biz_dir, "README.md")
        readme_content = ""
        if os.path.exists(readme_file):
            try:
                with open(readme_file, "r", encoding="utf-8") as rf:
                    readme_content = rf.read()
            except Exception:
                pass

        con, schema_summary = INGESTION_ENGINE.load_business_data(biz_dir)
        BIZ_CON_CACHE[biz_id] = con

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

        user_content = PROMPT_TEMPLATE.format(
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

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        dataset_rows.append({
            "prompt": prompt_messages,
            "business_id": biz_id,
            "question_text": question_text,
            "ref_type": gt.get("type"),
            "ref_val": gt.get("answer"),
            "ref_aliases": gt.get("alias", [])
        })

    train_dataset = Dataset.from_list(dataset_rows)
    print(f"[+] Dataset created with {len(train_dataset)} prompts.")

    # Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Initializing Policy Model: {args.model_id}")
    model_kwargs = {"trust_remote_code": True}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", **model_kwargs)

    # Configure LoRA for GRPO policy
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        print("[!] TRL version >= 0.14 required for GRPO. Please install via: pip install 'trl>=0.14.0'")
        return

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=0.8,
        logging_steps=5,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[duckdb_outcome_reward_func, format_compliance_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config
    )

    print("[*] Starting GRPO Reinforcement Learning...")
    trainer.train()

    print(f"[+] Saving final GRPO Policy checkpoint to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[+] GRPO Training Completed Successfully!")

if __name__ == "__main__":
    main()
