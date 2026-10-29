import unicodedata
import re
from typing import Any, Dict, List, Tuple

class DataMateEvaluator:
    """
    Implements official Zalo AI Challenge 2026 DataMate evaluation metric:
    - Normalization rules for all types (number, percent, string, date, boolean, list, table)
    - Per-question scoring (correct = 1, abstain = 0, wrong = -1)
    - Penalized Accuracy (PA) calculation
    """
    @staticmethod
    def normalize_string(s: str) -> str:
        """
        Rule for type 'string':
        (1) Unicode NFC
        (2) lowercase
        (3) every character not a letter, digit or whitespace replaced by space
        (4) leading/trailing whitespace removed, whitespace runs collapsed
        (Diacritics are NOT removed)
        """
        if s is None:
            return ""
        s = unicodedata.normalize('NFC', str(s)).lower()
        # Replace characters that are not unicode letters, digits, or whitespace
        s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    @classmethod
    def match_answer(cls, pred_type: str, pred_val: Any, ref_type: str, ref_val: Any, ref_aliases: List[Any] = None) -> bool:
        """
        Checks whether predicted answer matches reference answer according to type matching rules.
        Safe against arbitrary malformed outputs.
        """
        try:
            if pred_type != ref_type:
                return False

            if pred_type == "unanswerable":
                return True

            if pred_type == "abstain":
                return False

            if pred_type == "number":
                try:
                    # Match integers or rounded to 2 decimals
                    p_num = round(float(pred_val), 2)
                    r_num = round(float(ref_val), 2)
                    return p_num == r_num
                except (ValueError, TypeError):
                    return False

            elif pred_type == "percent":
                try:
                    # Exact match after rounding to 2 decimals, correct sign
                    p_pct = round(float(pred_val), 2)
                    r_pct = round(float(ref_val), 2)
                    return p_pct == r_pct
                except (ValueError, TypeError):
                    return False

            elif pred_type == "string":
                p_norm = cls.normalize_string(pred_val)
                r_norm = cls.normalize_string(ref_val)
                if p_norm == r_norm:
                    return True
                # Check against aliases
                if ref_aliases:
                    for alias in ref_aliases:
                        if p_norm == cls.normalize_string(alias):
                            return True
                return False

            elif pred_type == "date":
                # Exact match: yyyy-mm-dd or yyyy-mm
                return str(pred_val).strip() == str(ref_val).strip()

            elif pred_type == "boolean":
                # Exact boolean match
                return bool(pred_val) == bool(ref_val)

            elif pred_type == "list":
                # Always ordered, matched position by position
                if not isinstance(pred_val, (list, tuple)) or not isinstance(ref_val, (list, tuple)):
                    return False
                if len(pred_val) != len(ref_val):
                    return False
                for idx, (p_elem, r_elem) in enumerate(zip(pred_val, ref_val)):
                    if isinstance(p_elem, (list, tuple)) and len(p_elem) == 1:
                        p_elem = p_elem[0]
                    if isinstance(r_elem, (int, float)):
                        try:
                            if round(float(p_elem), 2) != round(float(r_elem), 2):
                                return False
                        except (ValueError, TypeError):
                            return False
                    else:
                        p_norm = cls.normalize_string(str(p_elem))
                        r_norm = cls.normalize_string(str(r_elem))
                        if p_norm != r_norm:
                            matched_alias = False
                            if ref_aliases and idx < len(ref_aliases) and isinstance(ref_aliases[idx], (list, tuple)):
                                for al in ref_aliases[idx]:
                                    if p_norm == cls.normalize_string(str(al)):
                                        matched_alias = True
                                        break
                            if not matched_alias:
                                return False
                return True

            elif pred_type == "table":
                if not isinstance(pred_val, dict) or not isinstance(ref_val, dict):
                    return False
                pred_rows = pred_val.get("rows", [])
                ref_rows = ref_val.get("rows", [])
                if not isinstance(pred_rows, (list, tuple)) or not isinstance(ref_rows, (list, tuple)):
                    return False
                if len(pred_rows) != len(ref_rows):
                    return False

                def norm_row(row):
                    if not isinstance(row, (list, tuple)):
                        return (cls.normalize_string(str(row)),)
                    out = []
                    for cell in row:
                        if isinstance(cell, (int, float)):
                            out.append(str(round(float(cell), 2)))
                        else:
                            try:
                                out.append(str(round(float(cell), 2)))
                            except (ValueError, TypeError):
                                out.append(cls.normalize_string(str(cell)))
                    return tuple(out)

                return sorted(norm_row(r) for r in pred_rows) == sorted(norm_row(r) for r in ref_rows)

            return False
        except Exception:
            return False

    @classmethod
    def score_question(cls, pred_obj: Dict[str, Any], ref_obj: Dict[str, Any]) -> int:
        """
        Scores a single question:
        - 1 if correct
        - 0 if abstain (or missing)
        - -1 if wrong
        """
        if not pred_obj or pred_obj.get("type") == "abstain":
            return 0
        
        pred_type = pred_obj.get("type")
        pred_val = pred_obj.get("answer")
        ref_type = ref_obj.get("type")
        ref_val = ref_obj.get("answer")
        ref_aliases = ref_obj.get("alias", [])

        is_correct = cls.match_answer(pred_type, pred_val, ref_type, ref_val, ref_aliases)
        return 1 if is_correct else -1

    @classmethod
    def compute_penalized_accuracy(cls, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes the Penalized Accuracy (PA) across all questions.
        """
        ref_map = {r["id"]: r for r in references}
        pred_map = {p["id"]: p for p in predictions}

        n = len(references)
        scores = []
        correct_cnt = 0
        abstain_cnt = 0
        wrong_cnt = 0

        for q_id, ref in ref_map.items():
            pred = pred_map.get(q_id)
            score = cls.score_question(pred, ref)
            scores.append(score)
            if score == 1:
                correct_cnt += 1
            elif score == 0:
                abstain_cnt += 1
            else:
                wrong_cnt += 1

        pa = sum(scores) / n if n > 0 else 0.0
        return {
            "penalized_accuracy": round(pa, 4),
            "total": n,
            "correct": correct_cnt,
            "abstain": abstain_cnt,
            "wrong": wrong_cnt,
            "accuracy_when_answering": round(correct_cnt / (correct_cnt + wrong_cnt), 4) if (correct_cnt + wrong_cnt) > 0 else 0.0
        }
