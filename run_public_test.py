"""
DataMate — Zalo AI Challenge 2026
Public Test Inference Pipeline (Compatible with Google Colab & Local GPU)

Usage:
    python run_public_test.py --data_dir public_test --output submission.jsonl --model_id Qwen/Qwen2.5-Coder-7B-Instruct --load_in_4bit
"""

import os
import sys

# Ensure the directory containing this script is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import json
import re
import time
import io
import argparse
import traceback
import unicodedata
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# Import baseline modules
from src.temporal import TemporalEngine
from src.ingestion import DataIngestionEngine
from src.evaluator import DataMateEvaluator


# =====================================================================
# 1. Prompt Construction for Code Generation
# =====================================================================

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia phân tích dữ liệu bảng và lập trình Python/DuckDB phục vụ bài toán DataMate (Zalo AI Challenge 2026).
Nhiệm vụ của bạn:
1. Đọc kỹ câu hỏi, ngày tham chiếu (as_of_date), quy tắc nghiệp vụ mặc định, và cấu trúc các bảng dữ liệu (schema) trong DuckDB.
2. Viết một đoạn mã Python để truy vấn và tính toán đáp án chính xác từ kết nối DuckDB `con`.
3. Gán đáp án cuối cùng vào biến toàn cục `result`.
   Bạn có thể gán dictionary: {"type": "<type>", "answer": <answer>}
   Hoặc gán trực tiếp giá trị vào `result` (ví dụ: `result = 48250000`, `result = 'Sting dâu'`, `result = 12.5`).

QUY TẮC BẮT BUỘC KHI VIẾT CODE:
- Chỉ trả về duy nhất khối mã Python trong cặp dấu ```python ... ```. Không giải thích dông dài.
- Kết nối DuckDB đã được mở sẵn và lưu trong biến `con`. KHÔNG gọi `duckdb.connect()` mới.
- Lưu kết quả vào biến `result`.
- DÙNG `ILIKE '%...%'` khi so khớp tên hàng hoá / sản phẩm để không phân biệt hoa thường.
- Cột ngày tháng (thoi_gian, ngay, ngay_dat_hang) đã được chuẩn hóa về định dạng chuỗi 'YYYY-MM-DD'. So khớp trực tiếp bằng `'YYYY-MM-DD'` hoặc `BETWEEN '...' AND '...'`.
- Khi câu hỏi hỏi về 1 tháng cụ thể (ví dụ: tháng 5/2026, tháng 11/2025), hãy chọn đúng bảng tương ứng trong Cấu Trúc Dữ Liệu (ví dụ: `sobanhang_t05_2026`, `banhang_t11_2025_...`).
"""

PROMPT_TEMPLATE = """### THÔNG TIN CỬA HÀNG ({biz_id})
{readme_content}

### CẤU TRÚC DỮ LIỆU ĐANG CÓ TRONG DUCKDB (kết nối qua biến `con`):
{schema_summary}

### THÔNG TIN THỜI GIAN ĐÃ PHÂN GIẢI THEO LỊCH & RULEBOOK:
- Ngày tham chiếu (as_of_date / Hôm nay): {as_of_date}
- Hôm qua: {hom_qua}
- Tuần này: Từ {tuan_nay_start} đến {tuan_nay_end}
- Tuần rồi: Từ {tuan_roi_start} đến {tuan_roi_end}
- Tháng này: Từ {thang_nay_start} đến {thang_nay_end}
- Tháng trước: Từ {thang_truoc_start} đến {thang_truoc_end}
- Khoảng thời gian Tết năm nay: {tet_season}

### QUY TẮC MẶC ĐỊNH QUAN TRỌNG:
- Doanh thu: Chỉ tính đơn không huỷ và không trả hàng. Đơn có trả hàng (kể cả trả 1 phần) loại bỏ CẢ ĐƠN.
- Doanh thu = tổng thành tiền từng dòng hàng sau giảm giá dòng. Không trừ voucher cả đơn, không cộng VAT, không gồm phí ship.
- Bán chạy mặc định = theo SỐ LƯỢNG (không phải theo doanh thu).
- Đơn thuộc kỳ nào = theo ngày đặt đơn.

### CÂU HỎI CỦA NGƯỜI DÙNG:
"{question}"

Hãy viết đoạn mã Python (trong ```python ... ```) để tính toán và gán vào biến `result`:"""


# =====================================================================
# 2. Pipeline Manager
# =====================================================================

class PublicTestRunner:
    def __init__(self, model_id: str, load_in_4bit: bool = True, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing model: {model_id} on {self.device} (4-bit: {load_in_4bit})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        if load_in_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        self.model.eval()
        print("[+] Model loaded successfully!")

    def generate_code(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False
            )
        
        output_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response

    @staticmethod
    def extract_python_code(response: str) -> str:
        """Extracts code from ```python ... ``` blocks."""
        match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match_generic = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match_generic:
            return match_generic.group(1).strip()
        return response.strip()

    @classmethod
    def infer_type_from_raw_value(cls, val: Any, question_text: str) -> Dict[str, Any]:
        """Automatically infers answer type if model assigned a raw value instead of dict."""
        import pandas as pd
        import numpy as np

        if val is None or pd.isna(val):
            return {"type": "abstain"}

        if isinstance(val, (bool, np.bool_)):
            return {"type": "boolean", "answer": bool(val)}

        if isinstance(val, (int, float, np.integer, np.floating)):
            if np.isnan(val):
                return {"type": "abstain"}
            # Check if question asks about percentage
            if any(w in question_text.lower() for w in ["%", "phần trăm", "tỉ lệ", "tỷ lệ"]):
                return {"type": "percent", "answer": round(float(val), 2)}
            num = float(val)
            if num.is_integer():
                return {"type": "number", "answer": int(num)}
            return {"type": "number", "answer": round(num, 2)}

        if isinstance(val, (list, tuple)):
            return {"type": "list", "answer": list(val)}

        if isinstance(val, pd.DataFrame):
            if val.empty:
                return {"type": "abstain"}
            if val.shape == (1, 1):
                raw_elem = val.iloc[0, 0]
                return cls.infer_type_from_raw_value(raw_elem, question_text)
            elif val.shape[0] == 1 or val.shape[1] == 1:
                return {"type": "list", "answer": val.values.flatten().tolist()}
            else:
                return {"type": "table", "answer": {"columns": list(val.columns), "rows": val.values.tolist()}}

        val_str = str(val).strip()
        # Check if date format YYYY-MM-DD or YYYY-MM
        if re.match(r"^\d{4}-\d{2}(-\d{2})?$", val_str):
            return {"type": "date", "answer": val_str}

        return {"type": "string", "answer": val_str}

    @classmethod
    def execute_code(cls, code: str, con: Any, question_text: str, log_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Safely executes the generated python code with the duckdb connection."""
        import pandas as pd
        import numpy as np

        stdout_capture = io.StringIO()
        exec_globals = {
            "con": con,
            "pd": pd,
            "np": np,
            "datetime": datetime,
            "result": None
        }

        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        try:
            exec(code, exec_globals)
        except Exception as e:
            sys.stdout = old_stdout
            log_item["exec_error"] = str(e)
            log_item["exec_traceback"] = traceback.format_exc()
            return None
        finally:
            sys.stdout = old_stdout

        captured_output = stdout_capture.getvalue().strip()
        if captured_output:
            log_item["captured_stdout"] = captured_output

        res = exec_globals.get("result")
        if isinstance(res, dict) and "type" in res:
            return res
        elif res is not None:
            return cls.infer_type_from_raw_value(res, question_text)
        elif captured_output:
            # If model printed result instead of assigning
            try:
                # Try parsing as JSON or float
                return cls.infer_type_from_raw_value(captured_output, question_text)
            except Exception:
                pass

        log_item["exec_error"] = "Variable 'result' was not assigned"
        return None


# =====================================================================
# 3. Output Post-Processing & Normalization
# =====================================================================

def format_final_submission(q_id: str, raw_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensures exact matching compliance with Zalo AI submission format."""
    if not raw_result or not isinstance(raw_result, dict):
        return {"id": q_id, "type": "abstain"}
    
    ans_type = raw_result.get("type", "abstain").lower().strip()
    val = raw_result.get("answer")
    
    if ans_type == "abstain":
        return {"id": q_id, "type": "abstain"}
    
    if ans_type == "unanswerable":
        res = {"id": q_id, "type": "unanswerable"}
        if "reason" in raw_result:
            res["reason"] = str(raw_result["reason"])
        return res
    
    if ans_type == "number":
        try:
            num = float(val)
            if num.is_integer():
                return {"id": q_id, "type": "number", "answer": int(num)}
            return {"id": q_id, "type": "number", "answer": round(num, 2)}
        except (ValueError, TypeError):
            return {"id": q_id, "type": "abstain"}
            
    elif ans_type == "percent":
        try:
            pct = round(float(val), 2)
            return {"id": q_id, "type": "percent", "answer": pct}
        except (ValueError, TypeError):
            return {"id": q_id, "type": "abstain"}
            
    elif ans_type == "string":
        if val is None:
            return {"id": q_id, "type": "abstain"}
        return {"id": q_id, "type": "string", "answer": str(val).strip()}
        
    elif ans_type == "date":
        if not val:
            return {"id": q_id, "type": "abstain"}
        return {"id": q_id, "type": "date", "answer": str(val).strip()}
        
    elif ans_type == "boolean":
        return {"id": q_id, "type": "boolean", "answer": bool(val)}
        
    elif ans_type == "list":
        if isinstance(val, list):
            return {"id": q_id, "type": "list", "answer": val}
        return {"id": q_id, "type": "abstain"}
        
    elif ans_type == "table":
        if isinstance(val, dict) and "columns" in val and "rows" in val:
            return {"id": q_id, "type": "table", "answer": val}
        return {"id": q_id, "type": "abstain"}
        
    return {"id": q_id, "type": "abstain"}


# =====================================================================
# 4. Main Inference Runner
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Public Test Inference for DataMate")
    parser.add_argument("--data_dir", type=str, default="public_test", help="Path to split directory (public_test or train)")
    parser.add_argument("--output", type=str, default="submission.jsonl", help="Output submission JSONL file path")
    parser.add_argument("--log_dir", type=str, default="result/logs", help="Directory to save question logs")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process (for quick testing)")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"=== Starting DataMate Inference ===")
    # Auto-resolve data_dir if running from subfolder like zalo-vecna
    if not os.path.exists(args.data_dir):
        candidates = [
            os.path.join("..", args.data_dir),
            f"/content/drive/MyDrive/ZALO_AI_Challenge/{args.data_dir}",
            f"/content/{args.data_dir}"
        ]
        for c in candidates:
            if os.path.exists(c):
                args.data_dir = c
                break
    print(f"[*] Using data directory: {os.path.abspath(args.data_dir)}")

    calendar_path = os.path.join(args.data_dir, "calendar.csv")
    questions_path = os.path.join(args.data_dir, "questions.jsonl")
    data_biz_dir = os.path.join(args.data_dir, "data")

    # 1. Initialize Engines
    temporal_engine = TemporalEngine(calendar_path)
    ingestion_engine = DataIngestionEngine()
    runner = PublicTestRunner(model_id=args.model_id, load_in_4bit=args.load_in_4bit)

    # 2. Load questions
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        questions = questions[:args.limit]
        print(f"[*] Processing first {args.limit} questions only.")
    else:
        print(f"[*] Processing all {len(questions)} questions.")

    # 3. Cache for loaded DuckDB business instances to avoid reloading data repeatedly
    biz_cache = {}

    # Open output file for streaming writes
    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, q in enumerate(tqdm(questions, desc="Inferring")):
            q_id = q["id"]
            biz_id = q["business_id"]
            question_text = q["question"]
            as_of_date = q["as_of_date"]

            start_time = time.time()
            log_item = {"id": q_id, "question": question_text, "business_id": biz_id}

            try:
                # Load business data and schema
                if biz_id not in biz_cache:
                    biz_path = os.path.join(data_biz_dir, biz_id)
                    con, schema_summary = ingestion_engine.load_business_data(biz_path)
                    
                    readme_path = os.path.join(biz_path, "README.md")
                    readme_content = ""
                    if os.path.exists(readme_path):
                        with open(readme_path, "r", encoding="utf-8", errors="ignore") as rf:
                            readme_content = rf.read()
                            
                    biz_cache[biz_id] = (con, schema_summary, readme_content)

                con, schema_summary, readme_content = biz_cache[biz_id]

                # Resolve temporal dates
                dates = temporal_engine.resolve_relative_dates(as_of_date)

                # Format prompt
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
                    question=question_text
                )

                # Generate code
                raw_response = runner.generate_code(prompt)
                extracted_code = runner.extract_python_code(raw_response)
                log_item["generated_code"] = extracted_code

                # Execute code with enhanced error logging and auto-inference
                exec_result = runner.execute_code(extracted_code, con, question_text, log_item)

                # Self-correction: if execution failed, retry once with error feedback
                if (exec_result is None or exec_result.get("type") == "abstain") and log_item.get("exec_error"):
                    error_msg = log_item.get("exec_error")
                    retry_prompt = prompt + f"\n\nLƯU Ý: Đoạn code bạn đã sinh trước đó gặp lỗi thực thi sau:\n```python\n{extracted_code}\n```\nLỗi thực thi: {error_msg}\nHãy sửa lại code Python để sửa lỗi này và gán kết quả vào `result`:"
                    retry_response = runner.generate_code(retry_prompt)
                    retry_code = runner.extract_python_code(retry_response)
                    log_item["retry_code"] = retry_code
                    log_item.pop("exec_error", None)
                    exec_result = runner.execute_code(retry_code, con, question_text, log_item)

                # Format final answer
                final_submission = format_final_submission(q_id, exec_result)

            except Exception as e:
                # In case of any unhandled error, fallback safely to abstain
                final_submission = {"id": q_id, "type": "abstain"}
                log_item["error"] = str(e)
                log_item["traceback"] = traceback.format_exc()

            elapsed = round(time.time() - start_time, 3)
            log_item["elapsed_time"] = elapsed
            log_item["final_submission"] = final_submission

            # Write submission
            out_f.write(json.dumps(final_submission, ensure_ascii=False) + "\n")
            out_f.flush()

            # Write question log (required by Rule 8)
            with open(os.path.join(args.log_dir, f"{q_id}.json"), "w", encoding="utf-8") as lf:
                json.dump(log_item, lf, ensure_ascii=False, indent=2)

    print(f"\n[+] Inference completed successfully!")
    print(f"[+] Output saved to: {args.output}")
    print(f"[+] Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
