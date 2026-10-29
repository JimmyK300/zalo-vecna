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
- TUYỆT ĐỐI CHỈ SỬ DỤNG CÁC CỘT CÓ TRONG DANH SÁCH COLUMNS CỦA BẢNG ĐƯỢC LIỆT KÊ TRONG PHẦN CẤU TRÚC DỮ LIỆU. Không tự đoán tên cột (ví dụ: nếu bảng không có cột `trang_thai_xu_ly`, tuyệt đối không viết `WHERE trang_thai_xu_ly...` kẻo lỗi Binder Error).
- DÙNG `ILIKE '%...%'` khi so khớp tên hàng hoá / sản phẩm để không phân biệt hoa thường.
- Cột ngày tháng (thoi_gian, ngay, ngay_dat_hang) đã được chuẩn hóa về định dạng chuỗi 'YYYY-MM-DD'. So khớp trực tiếp bằng `'YYYY-MM-DD'` hoặc `BETWEEN '...' AND '...'`.
- Chọn đúng bảng tháng/năm: BẮT BUỘC chọn đúng bảng tháng/năm tương ứng với câu hỏi hoặc as_of_date (xem phần Gợi ý bảng bên dưới).
- Xử lý đơn huỷ / trạng thái đơn: CHỈ lọc trạng thái đơn huỷ NẾU bảng thực sự có cột trạng thái/ghi chú trong danh sách Columns (như `trang_thai`, `tt_don`, hoặc `trang_thai_giao`). Khi lọc trên cột có thể NULL, BẮT BUỘC dùng COALESCE: `COALESCE(col, '') NOT ILIKE '%huỷ%'`. Nếu bảng KHÔNG có cột trạng thái/ghi chú nào, TUYỆT ĐỐI KHÔNG tự thêm điều kiện lọc vào WHERE kẻo lỗi Binder Error!
- Phân biệt nhóm hàng vs mặt hàng cụ thể:
  + Khi câu hỏi hỏi MẶT HÀNG CỤ THỂ (ví dụ: 'Ống PVC Bình Minh D50 cây 4m' hay 'Trà đào'): Lọc trực tiếp trên bảng bán hàng theo tên hàng (ví dụ: `hang_ban ILIKE '%D50%'` hoặc `ten_san_pham ILIKE '%trà đào%'`). KHÔNG JOIN bảng danh mục nếu mặt hàng không có trong danh mục.
  + Khi câu hỏi hỏi theo NHÓM HÀNG (ví dụ: 'nhóm đồ dùng', 'nhóm trà'):
    * Nếu trong bảng bán hàng hoặc tên mặt hàng đã thể hiện nhóm (ví dụ: quán nước có tên sản phẩm như 'Trà đào...', 'Hồng trà...'), có thể lọc trực tiếp: `ten_san_pham ILIKE '%trà%'`.
    * Nếu bảng bán hàng không có tên nhóm nhưng có bảng danh mục hàng hoá (xem đúng tên bảng danh mục trong phần CẤU TRÚC DỮ LIỆU), hãy JOIN bảng danh mục qua tên hàng để lọc cột `nhom`. LƯU Ý: Bỏ chữ "nhóm" khi so khớp (ví dụ câu hỏi hỏi "nhóm đồ dùng" thì lọc `nhom ILIKE '%đồ dùng%'`, KHÔNG lọc '%nhóm đồ dùng%').
- Đơn vị tính và Quy đổi: Đọc kỹ phần Quy ước trong THÔNG TIN CỬA HÀNG (README.md).
  * Ví dụ Ống PVC bán theo bó (1 bó = 10 cây): Khi tính số cây, nếu đơn vị `dvt` là 'bó' thì nhân 10, nếu là 'cây' thì giữ nguyên:
    `CASE WHEN dvt ILIKE '%bó%' THEN sl_ban * 10 ELSE sl_ban END` (chú ý: nhân 10 với 'bó', KHÔNG nhân với 'cây').
- Tên thợ / nhân viên và Huê hồng:
  * Trong các bảng ghi chép công việc (ví dụ `so_sua_xe`), thợ thường được ghi theo tên gọi / biệt danh (như 'A.Phúc', 'a Phúc', 'Phúc'). Khi tính huê hồng cho thợ (ví dụ 'Đoàn Bảo Phúc'), hãy so khớp cột thợ với tên gọi: `tho ILIKE '%Phúc%'`.
  * Khi tính tiền huê hồng: BẮT BUỘC dùng hàm gộp cho hoa_hong khi có SUM: `SUM(s.tien) * ANY_VALUE(d.hoa_hong) / 100.0` (hoặc `MAX(d.hoa_hong)`) để không bị lỗi DuckDB GROUP BY. Cột ngày tháng nằm ở bảng sổ sửa xe, bảng thợ chỉ có họ tên và tỷ lệ huê hồng.
- Doanh thu trung bình mỗi đơn:
  * Công thức tính: `SUM(thanh_tien) * 1.0 / COUNT(DISTINCT don_so)`. TUYỆT ĐỐI KHÔNG gán số giả định / ví dụ mẫu (như 1234567.89).
- Tính tỷ lệ phần trăm (%):
  * BẮT BUỘC nhân 100.0 trước khi chia: `COUNT(...) * 100.0 / COUNT(...)` để tránh lỗi integer division ra 0.0.
- Ưu tiên dùng VIEW hợp nhất toàn bộ:
  * Nếu phần CẤU TRÚC DỮ LIỆU có 'View hợp nhất toàn bộ' (như `chitiethoadon`, `sobanhang`, `banhang_tonghop_banhang`, `sosuaxe`), hãy ƯU TIÊN truy vấn từ View này để bao quát toàn bộ thời gian 2025-2026, không bị thiếu dữ liệu hay chọn nhầm file/sheet/năm.
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
{month_hint}

### QUY TẮC MẶC ĐỊNH QUAN TRỌNG:
- Doanh thu: Chỉ tính đơn không huỷ và không trả hàng. Đơn có trả hàng (kể cả trả 1 phần) loại bỏ CẢ ĐƠN.
- Doanh thu = tổng thành tiền từng dòng hàng sau giảm giá dòng. Không trừ voucher cả đơn, không cộng VAT, không gồm phí ship.
- Bán chạy mặc định = theo SỐ LƯỢNG (không phải theo doanh thu).
- Đơn thuộc kỳ nào = theo ngày đặt đơn.
- Không lọc cứng trang_thai_xu_ly = 'Đã giao' nếu bảng không có.

### CÂU HỎI CỦA NGƯỜI DÙNG:
"{question}"

Hãy viết đoạn mã Python (trong ```python ... ```) để tính toán và gán vào biến `result`:"""


def extract_target_month_year(question_text: str, as_of_date: str):
    """
    Detects target month and year from question text or as_of_date.
    Returns (month, year) as ints, or None.
    """
    # Check question text first for explicit month/year
    # Matches: 'tháng 5/2026', 'tháng 05/2026', 'tháng 5 năm 2026', 't05-2026', 't5/2026'
    m = re.search(r'(?:tháng|t)\s*(\d{1,2})[\s/_\-nen]*(?:năm\s*)?(\d{4})', question_text, re.IGNORECASE)
    if m:
        try:
            month = int(m.group(1))
            year = int(m.group(2))
            if 1 <= month <= 12 and 2020 <= year <= 2030:
                return month, year
        except ValueError:
            pass

    # Next check question text for just month e.g. "tháng 11"
    m_month_only = re.search(r'(?:tháng|t)\s*(\d{1,2})\b', question_text, re.IGNORECASE)
    if m_month_only and as_of_date:
        try:
            month = int(m_month_only.group(1))
            year = int(as_of_date.split("-")[0])
            if 1 <= month <= 12:
                return month, year
        except ValueError:
            pass

    # Fallback to as_of_date if question asks about today / yesterday / this week / this month
    if as_of_date:
        parts = as_of_date.split("-")
        if len(parts) >= 2:
            try:
                return int(parts[1]), int(parts[0])
            except ValueError:
                pass

    return None, None


# =====================================================================
# 2. Pipeline Manager
# =====================================================================

class PublicTestRunner:
    def __init__(self, model_id: str, load_in_4bit: bool = True, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing model: {model_id} on {self.device} (4-bit: {load_in_4bit})")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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

    def generate_code_batch(self, prompts: list) -> list:
        """Batch generation for ultra-fast parallel inference on high-end GPUs like A100."""
        if not prompts:
            return []
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        input_len = inputs.input_ids.shape[1]
        responses = []
        for i in range(len(prompts)):
            out_ids = outputs[i][input_len:]
            responses.append(self.tokenizer.decode(out_ids, skip_special_tokens=True))
        return responses

    def generate_code(self, prompt: str) -> str:
        return self.generate_code_batch([prompt])[0]

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

        if val is None:
            return {"type": "abstain"}

        if isinstance(val, np.ndarray):
            val = val.tolist()

        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return {"type": "abstain"}
            # Flatten 1-element tuples / lists: [(x,), (y,)] -> [x, y]
            flattened = []
            for item in val:
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    flattened.append(item[0])
                else:
                    flattened.append(item)
            
            # Check if this was a scalar query returning [(scalar,)]
            is_list_q = any(w in question_text.lower() for w in ["nhung mon", "những món", "danh sách", "top ", "cac ", "các "])
            if len(flattened) == 1 and not is_list_q:
                return cls.infer_type_from_raw_value(flattened[0], question_text)
            
            # Clean elements (strip strings)
            clean_list = [x.strip() if isinstance(x, str) else x for x in flattened]
            return {"type": "list", "answer": clean_list}

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

        # For scalar values: check if NaN
        try:
            if pd.isna(val):
                return {"type": "abstain"}
        except Exception:
            pass

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

        # Safeguard: prevent model code from calling duckdb.connect() and wiping out preloaded tables
        import duckdb
        old_duckdb_connect = duckdb.connect
        duckdb.connect = lambda *args, **kwargs: con

        code_sanitized = re.sub(r'con\s*=\s*duckdb\.connect\s*\([^)]*\)', '# [AUTO] con already connected', code)
        code_sanitized = re.sub(r'conn\s*=\s*duckdb\.connect\s*\([^)]*\)', 'conn = con', code_sanitized)

        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        try:
            exec(code_sanitized, exec_globals)
        except Exception as e:
            err_msg = str(e)
            healed = False
            # Self-healing 1: Catalog Error Did you mean "yyy"?
            m_cat = re.search(r'Table with name ([a-zA-Z0-9_]+) does not exist!.*?Did you mean "([a-zA-Z0-9_]+)"\?', err_msg, re.DOTALL)
            if m_cat:
                bad_tbl, good_tbl = m_cat.group(1), m_cat.group(2)
                code_sanitized = re.sub(rf'\b{bad_tbl}\b', good_tbl, code_sanitized)
                healed = True

            # Self-healing 2: Binder Error non-existent column in WHERE filter
            m_ref = re.search(r'Referenced column "([a-zA-Z0-9_]+)" not found in FROM clause', err_msg)
            if m_ref:
                missing_col = m_ref.group(1)
                pattern = rf'(?:AND|OR)?\s*(?:COALESCE\s*\(\s*{missing_col}\b[^)]*\)|{missing_col}\b)\s*(?:NOT\s+)?(?:I?LIKE|=|!=)\s*\'[^\']*\'\s*'
                new_code = re.sub(pattern, ' ', code_sanitized, flags=re.IGNORECASE)
                if new_code != code_sanitized:
                    code_sanitized = new_code
                    healed = True

            # Self-healing 3: column must appear in GROUP BY
            m_grp = re.search(r'column "([a-zA-Z0-9_.]+)" must appear in the GROUP BY clause', err_msg)
            if m_grp:
                col = m_grp.group(1).split('.')[-1]
                pattern = rf'(?<!AS\s)(?<!ANY_VALUE\()(?:\b[a-zA-Z0-9_]+\.)?\b({col})\b'
                new_code = re.sub(pattern, r'ANY_VALUE(\g<0>)', code_sanitized, flags=re.IGNORECASE)
                if new_code != code_sanitized:
                    code_sanitized = new_code
                    healed = True

            # Retry healed code if fixes were applied
            if healed:
                try:
                    exec(code_sanitized, exec_globals)
                    err_msg = None
                except Exception as e2:
                    err_msg = str(e2)
                    # Attempt second healing pass if another column is missing
                    m_ref2 = re.search(r'Referenced column "([a-zA-Z0-9_]+)" not found in FROM clause', err_msg)
                    if m_ref2:
                        missing_col2 = m_ref2.group(1)
                        pattern2 = rf'(?:AND|OR)?\s*(?:COALESCE\s*\(\s*{missing_col2}\b[^)]*\)|{missing_col2}\b)\s*(?:NOT\s+)?(?:I?LIKE|=|!=)\s*\'[^\']*\'\s*'
                        code_sanitized = re.sub(pattern2, ' ', code_sanitized, flags=re.IGNORECASE)
                        try:
                            exec(code_sanitized, exec_globals)
                            err_msg = None
                        except Exception as e3:
                            err_msg = str(e3)

            if err_msg:
                sys.stdout = old_stdout
                log_item["exec_error"] = err_msg
                log_item["exec_traceback"] = traceback.format_exc()
                return None
        finally:
            sys.stdout = old_stdout
            duckdb.connect = old_duckdb_connect

        captured_output = stdout_capture.getvalue().strip()
        if captured_output:
            log_item["captured_stdout"] = captured_output

        res = exec_globals.get("result")
        if isinstance(res, (tuple, list)) and len(res) == 1:
            res = res[0]

        if isinstance(res, dict) and "type" in res:
            return res
        elif res is not None and not pd.isna(res):
            return cls.infer_type_from_raw_value(res, question_text)
        elif captured_output:
            # If model printed result instead of assigning
            try:
                # Try parsing as JSON or float
                parsed = cls.infer_type_from_raw_value(captured_output, question_text)
                if parsed.get("type") != "abstain":
                    return parsed
            except Exception:
                pass

        log_item["exec_error"] = (
            "Kết quả 'result' là None hoặc chưa được gán (truy vấn DuckDB trả về NULL/không có dòng nào khớp). "
            "GỢI Ý SỬA LỖI: "
            "1. Nếu câu hỏi hỏi theo 'nhóm hàng', hãy JOIN bảng danh mục (ví dụ: quanly_danhmuc) qua ten_hang để lọc cột nhom ILIKE '%...%'. "
            "2. Nếu câu hỏi hỏi mặt hàng cụ thể (ví dụ Ống PVC), hãy lọc trực tiếp trên bảng bán hàng và KHÔNG JOIN bảng danh mục nếu mặt hàng không có trong danh mục. "
            "3. Không lọc cứng 'Đã giao' nếu bảng không có giá trị này (chỉ loại bỏ đơn huỷ/không giao). "
            "4. Đọc kỹ phần Quy ước trong README để áp dụng quy đổi đơn vị (ví dụ 1 bó = 10 cây)."
        )
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
    
    if ans_type in ["number", "integer", "int", "float", "double", "numeric"]:
        try:
            num = float(val)
            if num.is_integer():
                return {"id": q_id, "type": "number", "answer": int(num)}
            return {"id": q_id, "type": "number", "answer": round(num, 2)}
        except (ValueError, TypeError):
            return {"id": q_id, "type": "abstain"}
            
    elif ans_type in ["percent", "percentage"]:
        try:
            pct = round(float(val), 2)
            return {"id": q_id, "type": "percent", "answer": pct}
        except (ValueError, TypeError):
            return {"id": q_id, "type": "abstain"}
            
    elif ans_type in ["string", "str", "text"]:
        if val is None:
            return {"id": q_id, "type": "abstain"}
        return {"id": q_id, "type": "string", "answer": str(val).strip()}
        
    elif ans_type == "date":
        if not val:
            return {"id": q_id, "type": "abstain"}
        return {"id": q_id, "type": "date", "answer": str(val).strip()}
        
    elif ans_type in ["boolean", "bool"]:
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
    parser.add_argument("--fp16", action="store_true", default=False, help="Load model in native FP16 (recommended for A100 / L4 with >=24GB VRAM)")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size (e.g. 4, 8, 16 for high-end GPUs like A100)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process (for quick testing)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output file instead of resuming")
    args = parser.parse_args()

    # Determine 4-bit vs FP16
    use_4bit = False if args.fp16 else args.load_in_4bit

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
    runner = PublicTestRunner(model_id=args.model_id, load_in_4bit=use_4bit)

    # 2. Load questions
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        questions = questions[:args.limit]
        print(f"[*] Processing first {args.limit} questions only.")
    else:
        print(f"[*] Processing all {len(questions)} questions.")

    # 3. Check existing completed question IDs for auto-resume
    processed_ids = set()
    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        d = json.loads(line_str)
                        if "id" in d:
                            processed_ids.add(d["id"])
                    except Exception:
                        pass
        if processed_ids:
            print(f"[*] Found {len(processed_ids)} already processed questions. Resuming...")

    batch_size = max(1, args.batch_size)
    questions_to_process = [q for q in questions if q["id"] not in processed_ids]
    if processed_ids:
        print(f"[*] Resuming: {len(questions_to_process)} / {len(questions)} questions remaining.")

    # 4. Cache for loaded DuckDB business instances to avoid reloading data repeatedly
    biz_cache = {}

    # Open output file for streaming writes (append mode if resuming, else write mode)
    write_mode = "w" if (args.overwrite or not processed_ids) else "a"
    with open(args.output, write_mode, encoding="utf-8") as out_f, \
         tqdm(total=len(questions), initial=len(processed_ids), desc="Inferring") as pbar:

        for b_start in range(0, len(questions_to_process), batch_size):
            batch_q = questions_to_process[b_start : b_start + batch_size]
            batch_prompts = []
            batch_meta = []

            for q in batch_q:
                q_id = q["id"]
                biz_id = q["business_id"]
                question_text = q["question"]
                as_of_date = q["as_of_date"]
                log_item = {"id": q_id, "question": question_text, "business_id": biz_id, "_start_time": time.time()}

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
                    dates = temporal_engine.resolve_relative_dates(as_of_date)
                    t_month, t_year = extract_target_month_year(question_text, as_of_date)
                    month_hint = ""
                    if t_month and t_year and con:
                        try:
                            all_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
                            m_keys = [f"t{t_month}_{t_year}", f"t0{t_month}_{t_year}", f"t{t_month:02d}_{t_year}"]
                            matching = [t for t in all_tables if any(k in t for k in m_keys)]
                            if matching:
                                month_hint = f"- GỢI Ý BẢNG THEO THỜI GIAN: Dữ liệu câu hỏi thuộc Tháng {t_month}/{t_year}. Hãy chọn từ các bảng tháng có sẵn trong DuckDB: {matching[:4]}."
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
                    batch_meta.append((q, con, question_text, prompt, log_item))
                except Exception as e:
                    batch_prompts.append("")
                    batch_meta.append((q, None, question_text, "", log_item))

            # Batch LLM Generation
            valid_indices = [i for i, p in enumerate(batch_prompts) if p]
            valid_prompts = [batch_prompts[i] for i in valid_indices]
            if valid_prompts:
                raw_responses = runner.generate_code_batch(valid_prompts)
            else:
                raw_responses = []

            resp_map = {valid_indices[i]: raw_responses[i] for i in range(len(valid_indices))}

            for item_idx, (q, con, question_text, prompt, log_item) in enumerate(batch_meta):
                q_id = q["id"]
                start_time = log_item.pop("_start_time", time.time())
                try:
                    if item_idx not in resp_map or con is None:
                        final_submission = {"id": q_id, "type": "abstain"}
                    else:
                        raw_response = resp_map[item_idx]
                        extracted_code = runner.extract_python_code(raw_response)
                        log_item["generated_code"] = extracted_code

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

                        final_submission = format_final_submission(q_id, exec_result)

                except Exception as e:
                    final_submission = {"id": q_id, "type": "abstain"}
                    log_item["error"] = str(e)
                    log_item["traceback"] = traceback.format_exc()

                log_item["elapsed_time"] = round(time.time() - start_time, 3)
                log_item["final_submission"] = final_submission

                # Write submission
                out_f.write(json.dumps(final_submission, ensure_ascii=False) + "\n")

                # Write question log
                with open(os.path.join(args.log_dir, f"{q_id}.json"), "w", encoding="utf-8") as lf:
                    json.dump(log_item, lf, ensure_ascii=False, indent=2)

            out_f.flush()
            pbar.update(len(batch_q))

    print(f"\n[+] Inference completed successfully!")
    print(f"[+] Output saved to: {args.output}")
    print(f"[+] Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    main()
