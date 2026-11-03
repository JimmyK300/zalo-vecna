"""Rough question-type map (keyword buckets) for a split, to find where recipes pay off."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from datamate.semantics import strip_accents  # noqa: E402

BUCKETS = [  # first match wins
    ("debt", r"\bno\b|cong no|mua chiu"),
    ("stock", r"\bton\b|het hang|con hang|ton kho"),
    ("import", r"\bnhap\b|nha cung cap|\bncc\b"),
    ("expiry", r"han dung|het han|\blo\b"),
    ("profit", r"\bloi\b|lai gop|lai rong|\blai\b|gia von"),
    ("price", r"\bgia\b|tang gia|doi gia"),
    ("cancel", r"\bhuy\b|tra hang|tra lai"),
    ("best_seller", r"ban chay|ban nhieu nhat|hot nhat|ban duoc nhieu|mat hang nao|mon nao|hang nao|\bsp\b|san pham"),
    ("customer", r"\bkhach\b"),
    ("staff", r"nhan vien"),
    ("delivery", r"giao|dot giao"),
    ("orders_count", r"bao nhieu don|so don|may don|\bdon\b"),
    ("revenue", r"doanh thu|dthu|\bdt\b|ban duoc|thu ve|tien hang|tien ban"),
]


def bucket(q: str) -> str:
    s = strip_accents(q)
    return next((b for b, p in BUCKETS if re.search(p, s)), "other")


if __name__ == "__main__":
    D = Path(sys.argv[1] if len(sys.argv) > 1 else "devkit/train")
    qs = [json.loads(l) for l in open(D / "questions.jsonl", encoding="utf-8")]
    c = Counter(bucket(q["question"]) for q in qs)
    print(D.name, len(qs), dict(c.most_common()))
