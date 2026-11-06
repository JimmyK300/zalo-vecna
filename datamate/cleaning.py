"""Conservative cleaner, data profiler with provenance, and schema/value grounding.
Provides hints for interpretation; never overwrites raw business facts or forces ambiguous meanings.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

VERSION = "cleaner-v4"

# Safe unambiguous word-level aliases for rules pass
ALIASES = {
    "hqua": "hôm qua",
    "hnay": "hôm nay",
    "dthu": "doanh thu",
    "bnhiu": "bao nhiêu",
    "bnhieu": "bao nhiêu",
    "ko": "không",
    "khum": "không",
    "trc": "trước",
}

# Known safe abbreviations and their allowed folded token expansions for guard verification
SAFE_EXPANSION_MAP: dict[str, set[str]] = {
    "hqua": {"hqua", "hom", "qua"},
    "hnay": {"hnay", "hom", "nay"},
    "dthu": {"dthu", "doanh", "thu"},
    "dt": {"dt", "doanh", "thu", "dien", "thoai"},
    "bnhiu": {"bnhiu", "bao", "nhieu"},
    "bnhieu": {"bnhieu", "bao", "nhieu"},
    "bn": {"bn", "bao", "nhieu"},
    "nhiu": {"nhiu", "bao", "nhieu"},
    "ko": {"ko", "khong"},
    "k": {"k", "khong", "nghin", "ngan"},
    "khum": {"khum", "khong"},
    "trc": {"trc", "truoc"},
    "dc": {"dc", "duoc"},
    "sp": {"sp", "san", "pham"},
    "kh": {"kh", "khach", "hang", "khong"},
    "sl": {"sl", "so", "luong"},
    "th": {"th", "thang", "thung"},
    "tg": {"tg", "thoi", "gian"},
    "vs": {"vs", "voi"},
    "nv": {"nv", "nhan", "vien"},
    "ncc": {"ncc", "nha", "cung", "cap"},
    "hd": {"hd", "hoa", "don"},
    "dh": {"dh", "don", "hang"},
    "tk": {"tk", "ton", "kho", "tai", "khoan"},
    "ln": {"ln", "loi", "nhuan"},
    "cn": {"cn", "cong", "no", "chi", "nhanh", "chu", "nhat"},
    "tb": {"tb", "trung", "binh"},
    "gium": {"gium", "giup"},
    "giup": {"gium", "giup"},
    "so": {"so", "luong"},
    "khach": {"khach", "hang"},
    "don": {"don", "hang"},
    "mat": {"mat", "hang"},
    "cua": {"cua", "hang"},
    "va": {"va", "cung", "voi"},
}

# Benign conversational / structural words that may be normalized without changing business query semantics
BENIGN_STRUCTURAL_TOKENS = {
    "la", "co", "cua", "cho", "toi", "minh", "giup", "gium", "xem", "coi",
    "hoi", "nhi", "ha", "a", "vay", "nhe", "trong", "cac", "nhung", "mot",
    "ve", "tinh", "duoc", "dc", "cung", "voi", "va", "check", "thu", "em",
}

MAX_FULL_SCAN_ROWS = 50_000
SAMPLE_SCAN_ROWS = 10_000
MAX_INDEX_VALUES_PER_COL = 2_000
MAX_GROUNDING_CANDIDATES = 8
MAX_GROUNDING_CHARS = 1_200


def fold(text: str) -> str:
    """Normalize for case-insensitive matching while stripping diacritics."""
    text = unicodedata.normalize("NFD", str(text)).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def norm_nfc(text: str) -> str:
    """Normalize to Unicode NFC, lowercase, collapsed whitespace, keeping diacritics."""
    s = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"\s+", " ", s).strip()


def match_case(orig: str, repl: str) -> str:
    """Preserve case pattern of orig when applying repl."""
    if orig.isupper():
        return repl.upper()
    if orig and orig[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def extract_numeric_tokens(s: str) -> list[str]:
    """Extract numeric tokens, percentages, decimal numbers and date strings in textual order."""
    tokens = []
    for m in re.finditer(r"(?<!\w)[+-]?\d+(?:[.,/:-]\d+)*(?:\s*%)?", s):
        val = re.sub(r"\s+", "", m.group().strip())
        tokens.append(val)
    return tokens


@dataclass
class QuestionCleanResult:
    original: str
    normalized: str
    normalized_hint: str = ""
    rules_normalized: str = ""
    llm_proposed: str = ""
    ambiguous: bool = False
    ambiguous_spans: list[str] = field(default_factory=list)
    clear_aliases: list[dict] = field(default_factory=list)
    unaccented_or_typo: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    protected_spans: list[dict] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.normalized_hint:
            self.normalized_hint = self.normalized
        if not self.rules_normalized:
            self.rules_normalized = self.normalized

    def to_dict(self):
        return asdict(self)


def extract_protected_spans(text: str, entities=()) -> tuple[list[str], list[dict]]:
    """Identify spans that must never be altered: catalog entities, codes, numbers, units, dates."""
    spans = []

    # 1. Exact catalog entities
    for e in entities:
        if isinstance(e, str) and e and len(e.strip()) >= 2:
            for m in re.finditer(re.escape(e), text):
                spans.append(dict(text=m.group(), start=m.start(), end=m.end(), kind="entity"))

    # 2. Date patterns (dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd)
    for m in re.finditer(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text):
        spans.append(dict(text=m.group(), start=m.start(), end=m.end(), kind="date"))
    for m in re.finditer(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text):
        spans.append(dict(text=m.group(), start=m.start(), end=m.end(), kind="date"))

    # 3. Numeric tokens with units, percentages, negative numbers, floats
    for m in re.finditer(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?\s*(?:ml|mg|kg|g|l|cm|mm|k|tr|triệu|trieu|nghìn|nghin|ngàn|ngan|đ|vnd|vnđ|%)\b", text, re.I):
        spans.append(dict(text=m.group(), start=m.start(), end=m.end(), kind="unit_number"))

    # 4. Standalone numbers, codes with leading zeros, or alphanumeric identifiers
    for m in re.finditer(r"(?<!\w)[+-]?\d+(?:[.,/:%-]\d+)*%?", text):
        spans.append(dict(text=m.group(), start=m.start(), end=m.end(), kind="number"))
    for m in re.finditer(r"\b0\d+\b", text):
        spans.append(dict(text=m.group(), start=m.start(), end=m.end(), kind="leading_zero_code"))
    for m in re.finditer(r"\b[A-Za-z]+\d+[A-Za-z0-9_-]*\b", text):
        spans.append(dict(text=m.group(), start=m.start(), end=m.end(), kind="sku_code"))

    # Deduplicate while preserving longest matching spans
    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    unique_spans = []
    seen_ranges = []
    for s in spans:
        if not any(r[0] <= s["start"] and s["end"] <= r[1] for r in seen_ranges):
            unique_spans.append(s)
            seen_ranges.append((s["start"], s["end"]))

    protected_strings = sorted(set(s["text"] for s in unique_spans), key=len, reverse=True)
    return protected_strings, unique_spans


def _extract_attribute_modifiers(f: str) -> list[str]:
    """Extract general (head, modifier) attribute bindings on folded text, such as:
    'mau do', 'mau xanh', 'size m', 'loai 1', 'vi dau', 'nhom ca phe', 'kenh shopee', etc."""
    bindings = []
    head_pat = (
        r"\b(mau|size|kich\s+co|kich\s+thuoc|loai|dang|vi|huong|mui|nhom\s+hang|nhom|"
        r"thuong\s+hieu|nhan\s+hieu|hoat\s+chat|kenh|chi\s+nhanh|cua\s+hang|shop|"
        r"kho|quay|phong|ban\s+so|ca|buoi|nhan\s+vien|nguon)\s+([^\s,;.?]+)"
    )
    stop_mods = {
        "nao", "gi", "may", "bao", "nhieu", "bnhiu", "bnhieu", "bn", "la", "co",
        "cua", "trong", "nay", "do", "day", "ay", "khac", "nhau", "va", "hoac",
        "tu", "den", "tren", "duoi", "theo", "cho", "tai", "o", "vua", "qua",
        "roi", "truoc", "trc", "sau", "gan", "nhat", "cao", "thap", "nhieu", "it",
    }
    # Note: 'do' after 'mau' means 'đỏ' (red), so do not treat 'do' as stopword when head is 'mau'!
    for m in re.finditer(head_pat, f):
        head = re.sub(r"\s+", " ", m.group(1).strip())
        mod = m.group(2).strip()
        if head == "mau" or mod not in stop_mods:
            if mod not in ("nao", "gi", "bao", "nhieu", "bnhiu", "bnhieu", "bn"):
                bindings.append(f"{head}::{mod}")

    # Color words attached directly to product nouns (e.g. 'ao do', 'gach xam', 'tra xanh')
    color_pat = r"\b(san\s+pham|sp|mat\s+hang|mon|hang|ao|quan|giay|tui|chai|hop|lo|vien|gach|son|tra)\s+(?:mau\s+)?(do|xanh|vang|trang|den|hong|tim|nau|xam|cam|bac)\b"
    for m in re.finditer(color_pat, f):
        bindings.append(f"color_attr::{m.group(2)}")

    return sorted(set(bindings))


def _extract_accented_words(text: str) -> list[str]:
    """Return sorted unique lowercased NFC content words from text that already carry Vietnamese diacritics."""
    nfc = unicodedata.normalize("NFC", text).lower()
    words = re.findall(r"[^\W\d_]+", nfc, flags=re.U)
    ignore = {"giùm", "giúp", "đc", "dc", "ko", "khum", "trc", "bữa"}
    accented = []
    for w in words:
        if w not in ignore and fold(w) not in BENIGN_STRUCTURAL_TOKENS and fold(w) != w:
            accented.append(w)
    return sorted(set(accented))


def extract_constraints(text: str) -> dict:
    """Extract semantic invariant anchors: relative time, question intent, negations, filters,
    ordering, metrics, customer/order/prescription attributes, general attribute modifiers,
    metric-object bindings, and accented word anchors."""
    f = fold(text)

    # 1. Time expressions
    times = []
    if re.search(r"\b(?:hqua|hom qua)\b", f):
        times.append("yesterday")
    if re.search(r"\b(?:hnay|hom nay|bua nay)\b", f):
        times.append("today")
    if re.search(r"\b(?:ngay mai)\b", f):
        times.append("tomorrow")
    if re.search(r"\b(?:hom kia)\b", f):
        times.append("day_before_yesterday")
    if re.search(r"\b(?:tuan roi|tuan truoc|tuan trc)\b", f):
        times.append("last_week")
    if re.search(r"\b(?:tuan nay)\b", f):
        times.append("this_week")
    if re.search(r"\b(?:thang nay)\b", f):
        times.append("this_month")
    if re.search(r"\b(?:thang truoc|thang trc|thang roi)\b", f):
        times.append("last_month")
    if re.search(r"\b(?:nam ngoai|nam truoc|nam trc)\b", f):
        times.append("last_year")
    if re.search(r"\b(?:nam nay)\b", f):
        times.append("this_year")
    if re.search(r"\b(?:cung ky)\b", f):
        times.append("same_period")
    if re.search(r"\b(?:gan nhat)\b", f):
        times.append("most_recent")
    if re.search(r"\b(?:dau thang)\b", f):
        times.append("month_start")
    if re.search(r"\b(?:cuoi thang)\b", f):
        times.append("month_end")
    if re.search(r"\b(?:het|den het)\b", f):
        times.append("inclusive_end")

    recent_days = re.findall(r"\b(\d+)\s+ngay\s+gan\s+nhat\b", f)
    for rd in recent_days:
        times.append(f"recent_days:{rd}")

    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    months = re.findall(r"\bthang\s+\d{1,2}(?:[/-]\d{2,4})?\b", f)
    times.extend([f"date:{d}" for d in dates])
    times.extend([f"month:{m}" for m in months])

    # Temporal direction: sau vs trước
    temporal_after = []
    for m in re.finditer(r"\bsau\s+(?:ngay\s+|thang\s+|nam\s+|gio\s+|ca\s+)?([^\s,;.?]+)", f):
        target = m.group(1)
        if target not in ("do", "khi"):
            temporal_after.append(target)
    temporal_before = []
    for m in re.finditer(r"(?<!tuan\s)(?<!thang\s)(?<!nam\s)(?<!hom\s)\b(?:truoc|trc)\s+(?:ngay\s+|thang\s+|nam\s+|gio\s+|ca\s+)?([^\s,;.?]+)", f):
        target = m.group(1)
        if target not in ("day", "khi", "mat"):
            temporal_before.append(target)

    # Shifts / Time of day
    shifts = []
    if re.search(r"\b(?:ca\s+sang|buoi\s+sang|ban\s+sang|vao\s+sang)\b", f) or (re.search(r"\bsang\b", f) and any(w in f for w in ("ca ", "buoi ", "gio ", "doanh thu", "don", "tien"))):
        shifts.append("morning")
    if re.search(r"\b(?:ca\s+chieu|buoi\s+chieu|ban\s+chieu|vao\s+chieu)\b", f) or (re.search(r"\bchieu\b", f) and any(w in f for w in ("ca ", "buoi ", "gio ", "doanh thu", "don", "tien"))):
        shifts.append("afternoon")
    if re.search(r"\b(?:ca\s+toi|buoi\s+toi|vao\s+toi)\b", f) or (re.search(r"\btoi\b", f) and any(w in f for w in ("ca ", "buoi ", "gio ", "doanh thu", "don", "tien"))):
        shifts.append("evening")
    if re.search(r"\b(?:ca\s+trua|buoi\s+trua)\b", f):
        shifts.append("noon")
    if re.search(r"\b(?:ca\s+dem|buoi\s+dem)\b", f):
        shifts.append("night")

    # Intervals: từ <A> đến <B>
    intervals = []
    for m in re.finditer(r"\btu\s+(?:ngay\s+|thang\s+|nam\s+)?([^\s,;.?]+)\s+den\s+(?:ngay\s+|thang\s+|nam\s+)?([^\s,;.?]+)", f):
        intervals.append(f"{m.group(1)}->{m.group(2)}")

    # 2. Intent markers
    inquiry_markers = []
    if re.search(r"\b(?:bao nhieu|bnhiu|bnhieu|bao nhiu|bn|may)\b", f):
        inquiry_markers.append("quantity")
    if re.search(r"\b(?:bao nhieu tien|bn tien|bnhieu tien|bao nhiu tien|nhiu tien|gia bao nhieu|dthu bao nhieu|doanh thu bao nhieu)\b", f):
        inquiry_markers.append("money")
    if re.search(r"\b(?:bao nhieu %|bn %|may %|ty le|ti le|chiem may|chiem bao nhieu)\b", f):
        inquiry_markers.append("percentage")
    if re.search(r"\b(?:la gi|chi|nhung gi|nao|mat hang nao|sp nao|san pham nao|ai)\b", f):
        inquiry_markers.append("identity")
    if re.search(r"\b(?:nhat|cao nhat|thap nhat|nhieu nhat|it nhat|chay nhat|e nhat|top)\b", f):
        inquiry_markers.append("ranking")
    if re.search(r"\b(?:hon hay kem|tang hay giam|chenh lech)\b", f):
        inquiry_markers.append("comparison")
    if re.search(r"\b(?:co\s+.*\s+(?:khong|ko|khum|chua)|phai khong|dung khong|ha|chua)\b|\?", f):
        inquiry_markers.append("check")

    # Aggregations
    aggregations = []
    if re.search(r"\b(?:tong|tong cong|tong so|tong dthu|tong doanh thu)\b", f):
        aggregations.append("sum")
    if re.search(r"\b(?:trung binh|binh quan|tb)\b", f):
        aggregations.append("avg")
    if re.search(r"\b(?:cao nhat|nhieu nhat|lon nhat|top|chay nhat)\b", f):
        aggregations.append("max")
    if re.search(r"\b(?:thap nhat|it nhat|nho nhat|e nhat)\b", f):
        aggregations.append("min")

    has_quantity_inquiry = bool(re.search(r"\b(?:bao nhieu|bnhiu|bnhieu|bao nhiu|bn|may)\b", f))
    is_statement_nhieu = bool(re.search(r"\b(?:nhieu tien|nhieu don)\b", f)) and not has_quantity_inquiry

    # 3. Negations & targets
    negations = re.findall(r"\b(?:khong|ko|khum|chua|khong tinh|ko tinh|ngoai tru|loai tru|tru)\b", f)
    neg_canonical = []
    for n in negations:
        if n in ("ko", "khum"):
            neg_canonical.append("khong")
        elif n == "ko tinh":
            neg_canonical.append("khong tinh")
        elif n in ("loai tru", "ngoai tru"):
            neg_canonical.append("tru")
        else:
            neg_canonical.append(n)

    negation_targets = []
    for m in re.finditer(r"\b(?:khong\s+tinh|ko\s+tinh|khum\s+tinh|ngoai\s+tru|loai\s+tru|khong\s+bao\s+gom|chua\s+tinh|tru)\s+([^\s,;.?]+(?:\s+[^\s,;.?]+){0,2})", f):
        negation_targets.append(m.group(1).strip())

    # 4. Filters & Comparisons
    filters = re.findall(r"\b(?:tro len|tro xuong|duoi|tren|tu|den|toi da|toi thieu|it nhat|nhieu nhat)\b", f)

    # 5. Ordering
    ordering = re.findall(r"\b(?:tu cu den moi|cu den moi|tu moi den cu|tang dan|giam dan|lan luot|theo thu tu)\b", f)

    # 6. Metrics
    f_metrics = re.sub(r"\b(?:so\s+luong\s+don|so\s+don)\b", "don", f)
    f_metrics = re.sub(r"\b(?:so\s+luong\s+khach|so\s+khach)\b", "khach", f_metrics)
    f_metrics = re.sub(r"\b(?:so\s+luong\s+san\s+pham|so\s+san\s+pham|so\s+sp)\b", "san pham", f_metrics)
    metrics_found = re.findall(r"\b(?:doanh thu|dthu|dt|gia ban|gia von|loi nhuan|cong no|ton kho|so luong|don|khach)\b", f_metrics)
    metrics_canonical = []
    for m in metrics_found:
        if m in ("dthu", "dt"):
            metrics_canonical.append("doanh thu")
        else:
            metrics_canonical.append(m)

    # 7. Customer attributes (moi / cu / quen / le / si / vang lai)
    customer_attributes = []
    if re.search(r"\b(?:khach(?:\s+hang)?\s+moi|nguoi\s+mua\s+moi)\b", f):
        customer_attributes.append("khach_moi")
    if re.search(r"\b(?:khach(?:\s+hang)?\s+cu|khach(?:\s+hang)?\s+quay\s+lai|nguoi\s+mua\s+cu)\b", f):
        customer_attributes.append("khach_cu")
    if re.search(r"\b(?:khach(?:\s+hang)?\s+quen)\b", f):
        customer_attributes.append("khach_quen")
    if re.search(r"\b(?:khach(?:\s+hang)?\s+le)\b", f):
        customer_attributes.append("khach_le")
    if re.search(r"\b(?:khach(?:\s+hang)?\s+si)\b", f):
        customer_attributes.append("khach_si")
    if re.search(r"\b(?:khach(?:\s+hang)?\s+vang\s+lai)\b", f):
        customer_attributes.append("khach_vang_lai")

    # 8. Order status attributes
    order_statuses = []
    if re.search(r"\b(?:don\s+(?:da\s+)?hoan|don\s+tra|tra\s+hang|doi\s+tra|hoan\s+tra)\b", f):
        order_statuses.append("don_hoan")
    if re.search(r"\b(?:don\s+(?:bi\s+|da\s+)?huy|huy\s+don|bi\s+huy)\b", f):
        order_statuses.append("don_huy")
    if re.search(r"\b(?:don\s+(?:da\s+)?thanh\s+cong|giao\s+thanh\s+cong|hoan\s+thanh|don\s+hoan\s+thanh)\b", f):
        order_statuses.append("don_thanh_cong")
    if re.search(r"\b(?:don\s+(?:chua\s+)?hoan\s+tat|chua\s+giao|dang\s+giao)\b", f):
        order_statuses.append("don_chua_hoan")

    # 9. Prescription status
    prescription_status = []
    if re.search(r"\b(?:khong\s+ke\s+don|ko\s+ke\s+don|otc)\b", f):
        prescription_status.append("khong_ke_don")
    elif re.search(r"\b(?:ke\s+don|thuoc\s+ke\s+don)\b", f):
        prescription_status.append("ke_don")

    # 10. Metric to object bindings
    metric_bindings = []
    metric_pat = r"\b(doanh thu|dthu|dt|loi nhuan|so don|so luong|cong no|gia von|gia ban|tien)\s+(?:tu|cua|o|tai|cho|theo)\s+([^\s,;.?]+(?:\s+[^\s,;.?]+){0,2})"
    for m in re.finditer(metric_pat, f):
        metric = m.group(1)
        if metric in ("dthu", "dt"):
            metric = "doanh thu"
        raw_obj = m.group(2).strip()
        obj_words = []
        for w in raw_obj.split():
            if w in ("va", "voi", "cung", "nhung", "hoac", "trong", "khi", "la", "co", "dat", "duoc"):
                break
            obj_words.append(w)
        if obj_words:
            metric_bindings.append(f"{metric}::{' '.join(obj_words)}")

    obj_metric_pat = r"\b(cua hang [^\s,;.?]+|chi nhanh [^\s,;.?]+|shop [^\s,;.?]+)\s+(?:co|dat)\s+(?:tong\s+)?(doanh thu|dthu|dt|loi nhuan|so don|so luong)"
    for m in re.finditer(obj_metric_pat, f):
        obj = m.group(1).strip()
        metric = m.group(2)
        if metric in ("dthu", "dt"):
            metric = "doanh thu"
        metric_bindings.append(f"{metric}::{obj}")

    # 11. General attribute/modifier bindings (colors, sizes, types, flavors, brands, etc.)
    attribute_modifiers = _extract_attribute_modifiers(f)

    # 12. Accented words already present in original
    accented_words = _extract_accented_words(text)

    return {
        "time": dict(Counter(times)),
        "temporal_after": sorted(temporal_after),
        "temporal_before": sorted(temporal_before),
        "shifts": dict(Counter(shifts)),
        "intervals": sorted(intervals),
        "intent_markers": sorted(set(inquiry_markers)),
        "aggregations": dict(Counter(aggregations)),
        "is_interrogative": bool(inquiry_markers) or "?" in text,
        "is_statement_nhieu": is_statement_nhieu,
        "negations": dict(Counter(neg_canonical)),
        "negation_targets": sorted(negation_targets),
        "filters": dict(Counter(filters)),
        "ordering": dict(Counter(ordering)),
        "metrics": dict(Counter(metrics_canonical)),
        "customer_attributes": sorted(customer_attributes),
        "order_statuses": sorted(order_statuses),
        "prescription_status": sorted(prescription_status),
        "metric_bindings": sorted(metric_bindings),
        "attribute_modifiers": attribute_modifiers,
        "accented_words": accented_words,
    }


def clean_question(text: str, entities=()) -> QuestionCleanResult:
    """Safe rule-based normalization:
    - Normalizes Unicode NFC & whitespace
    - Replaces unambiguous abbreviations (hqua -> hôm qua, hnay -> hôm nay, dthu -> doanh thu, etc.)
    - Replaces multi-word phrases safely from right to left without offset drift
    - Protects numbers, dates, negative signs, units, codes, and entities verbatim
    - Never turns 'nhiu' into 'nhiều' blindly; preserves ambiguity for LLM/routing
    - Classifies clear aliases, unaccented/typo spans, ambiguous tokens, entities, and constraints
    """
    original = text
    text = unicodedata.normalize("NFC", text)
    protected, protected_spans = extract_protected_spans(text, entities)

    def is_protected(s: int, e: int) -> bool:
        for p in protected_spans:
            if not (e <= p["start"] or s >= p["end"]):
                return True
        return False

    candidates = []

    # 1. Multi-word phrase replacements
    phrase_patterns = [
        (r"\bbao\s+nhiu\b", "bao nhiêu", "phrase_expansion"),
        (r"\blà\s+bn\b", "là bao nhiêu", "phrase_expansion"),
        (r"\bbn\s+(?=[đd]ơn|tiền|ly|món|hàng|khách|phiếu|%)\b", "bao nhiêu ", "phrase_expansion"),
    ]
    for pat, repl, kind in phrase_patterns:
        for m in re.finditer(pat, text, flags=re.I):
            s, e = m.start(), m.end()
            if not is_protected(s, e):
                repl_cased = match_case(m.group(), repl)
                candidates.append((s, e, m.group(), repl_cased, kind))

    # 2. Single word alias replacements
    alias_pat = r"\b(?:" + "|".join(re.escape(k) for k in ALIASES) + r")\b"
    for m in re.finditer(alias_pat, text, flags=re.I):
        s, e = m.start(), m.end()
        word = m.group().lower()
        if word in ALIASES and not is_protected(s, e):
            repl = ALIASES[word]
            repl_cased = match_case(m.group(), repl)
            candidates.append((s, e, m.group(), repl_cased, "alias"))

    # Resolve overlapping candidates: prioritize longer matches
    candidates.sort(key=lambda c: (c[1] - c[0]), reverse=True)
    selected = []
    for cand in candidates:
        s, e, before, after, kind = cand
        overlap = False
        for sel in selected:
            if not (e <= sel[0] or s >= sel[1]):
                overlap = True
                break
        if not overlap:
            selected.append(cand)

    # Sort selected candidates by start index DESCENDING (from right to left)
    selected.sort(key=lambda c: c[0], reverse=True)

    changes = []
    for s, e, before, after, kind in selected:
        changes.append(dict(start=s, end=e, before=before, after=after, kind=kind))
        text = text[:s] + after + text[e:]

    changes.reverse()
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Detect ambiguous words, unaccented words, and telex typos
    warnings = []
    ambiguous_spans = []
    unaccented_or_typo = []
    f = fold(text)

    if re.search(r"(?<!bao\s)\bnhiu\b", f):
        ambiguous_spans.append("nhiu")
        warnings.append("ambiguous chat word 'nhiu' (may mean 'bao nhiêu' or 'nhiều'); keep original")

    for token, desc in [
        (r"\bdt\b", "dt (may mean 'doanh thu' or 'điện thoại')"),
        (r"\bkh\b", "kh (may mean 'khách hàng' or 'không')"),
        (r"\bth\b", "th (may mean 'tháng' or 'thùng')"),
        (r"\bbn\b", "bn (may mean 'bao nhiêu' or identifier)"),
        (r"\bdc\b", "dc (may mean 'được' or identifier)"),
        (r"\bk\b", "k (may mean 'không', 'nghìn', or identifier)"),
    ]:
        if re.search(token, f):
            matched = re.findall(token, f)
            if not any(m in p.lower() for m in matched for p in protected):
                ambiguous_spans.append(token.strip(r"\b"))
                warnings.append(f"ambiguous token: {desc}")

    for m_typo in re.finditer(r"\b(?:[a-z]*[aeiouy]{2}[a-z]*|(?:ngay|nao|nhat|tuan|thang)[fsrxj])\b", f):
        unaccented_or_typo.append(m_typo.group(0))
    if unaccented_or_typo:
        warnings.append("possible repeated letter/Telex typo")

    # Detect unaccented Vietnamese words that commonly carry diacritics
    common_unaccented = re.findall(
        r"\b(?:ngay|thang|tuan|nam|don|hang|khach|tien|ban|mua|nhap|ton|no|thuoc|sua|ca|phe|cua|nhieu|nhat|bao|tong|loi|nhuan)\b",
        text.lower(),
    )
    if common_unaccented:
        unaccented_or_typo.extend(common_unaccented)

    entity_spans = [s for s in protected_spans if s.get("kind") in ("entity", "leading_zero_code", "sku_code")]
    is_ambiguous = bool(ambiguous_spans) or bool(warnings)
    constraints = extract_constraints(original)

    return QuestionCleanResult(
        original=original,
        normalized=text,
        normalized_hint=text,
        rules_normalized=text,
        llm_proposed="",
        ambiguous=is_ambiguous,
        ambiguous_spans=sorted(set(ambiguous_spans)),
        clear_aliases=list(changes),
        unaccented_or_typo=sorted(set(unaccented_or_typo)),
        entities=entity_spans,
        changes=changes,
        warnings=warnings,
        protected=protected,
        protected_spans=protected_spans,
        constraints=constraints,
    )


def _check_content_token_alignment(original: str, proposed: str, protected: list[str]) -> tuple[bool, str]:
    """Ensure rewrite does not drop content words from original or introduce ungrounded content words.
    Also prevents changing diacritics on words that were already accented in original."""
    orig_nfc = unicodedata.normalize("NFC", original).lower()
    prop_nfc = unicodedata.normalize("NFC", proposed).lower()

    # 1. Content words in original that already had Vietnamese diacritics must not be replaced by a different
    # word or stripped/altered to a different accented word (e.g. 'đỏ' -> 'xanh', 'sữa' -> 'sửa').
    prop_words_nfc = set(re.findall(r"[^\W\d_]+", prop_nfc, flags=re.U))
    for w in _extract_accented_words(original):
        if w not in prop_words_nfc:
            return False, f"accented word in original altered or dropped: '{w}'"

    # 2. Folded content-token alignment
    orig_folded_words = re.findall(r"[a-z]+", fold(original))
    prop_folded_words = re.findall(r"[a-z]+", fold(proposed))

    allowed_in_prop: set[str] = set(BENIGN_STRUCTURAL_TOKENS)
    for w in orig_folded_words:
        allowed_in_prop.add(w)
        if w in SAFE_EXPANSION_MAP:
            allowed_in_prop.update(SAFE_EXPANSION_MAP[w])
        # Handle Telex trailing typo character stripping (e.g. 'nayf' -> 'nay')
        if len(w) > 3 and w[-1] in "fsrxj":
            allowed_in_prop.add(w[:-1])

    for p in protected:
        allowed_in_prop.update(re.findall(r"[a-z]+", fold(p)))

    # Every folded word in proposed must come from original, safe alias expansions, or benign connectors
    for pw in prop_folded_words:
        if pw not in allowed_in_prop:
            return False, f"ungrounded content word introduced in rewrite: '{pw}'"

    # Every content word in original must either remain in proposed or have its safe expansion in proposed
    prop_folded_set = set(prop_folded_words)
    for ow in orig_folded_words:
        if ow in BENIGN_STRUCTURAL_TOKENS:
            continue
        if ow in prop_folded_set:
            continue
        if len(ow) > 3 and ow[-1] in "fsrxj" and ow[:-1] in prop_folded_set:
            continue
        expansions = SAFE_EXPANSION_MAP.get(ow, set()) - {ow}
        if expansions and (expansions & prop_folded_set):
            continue
        return False, f"content word dropped or replaced in rewrite: '{ow}'"

    return True, ""


def validate_rewrite(result: QuestionCleanResult, proposed: str) -> tuple[bool, str]:
    """Strict defensive guard against semantic distortion.
    NOTE: Passing lexical/structural guard does NOT mathematically prove semantic equivalence;
    the cleaned string is used strictly as an auxiliary hint while original question controls."""
    if not isinstance(proposed, str) or not proposed.strip() or len(proposed) > max(512, len(result.original) * 3):
        return False, "empty/oversize rewrite"
    if len(proposed) < max(3, len(result.original) // 4):
        return False, "severely truncated rewrite"

    # 1. Protected entity/code/unit spans
    for p in result.protected:
        if p not in proposed:
            return False, f"protected entity/code/unit changed: missing '{p}'"

    # 2. Numeric tokens and dates check (including sequence and counts)
    if extract_numeric_tokens(result.original) != extract_numeric_tokens(proposed):
        return False, "numbers/dates/order changed"

    # 3. Negative numbers invariant
    if re.findall(r"-\d+", result.original) != re.findall(r"-\d+", proposed):
        return False, "negative number sign changed or dropped"

    # 4. Codes with leading zero invariant
    if re.findall(r"\b0\d+\b", result.original) != re.findall(r"\b0\d+\b", proposed):
        return False, "code with leading zero changed or lost"

    c_orig = result.constraints if result.constraints else extract_constraints(result.original)
    c_prop = extract_constraints(proposed)

    # 5. Time Expressions Invariant (Smoke Test 1 & 3 Protection)
    time_keys = [
        ("yesterday", "hôm qua / hqua"),
        ("today", "hôm nay / hnay"),
        ("tomorrow", "ngày mai"),
        ("day_before_yesterday", "hôm kia"),
        ("last_week", "tuần rồi / tuần trước"),
        ("this_week", "tuần này"),
        ("this_month", "tháng này"),
        ("last_month", "tháng trước / tháng rồi"),
        ("last_year", "năm ngoái / năm trước"),
        ("this_year", "năm nay"),
        ("same_period", "cùng kỳ"),
        ("most_recent", "gần nhất"),
        ("month_start", "đầu tháng"),
        ("month_end", "cuối tháng"),
        ("inclusive_end", "hết / đến hết"),
    ]
    for t_key, t_label in time_keys:
        in_orig = t_key in c_orig["time"]
        in_prop = t_key in c_prop["time"]
        if in_orig and not in_prop:
            return False, f"time condition changed: lost '{t_label}'"
        if not in_orig and in_prop:
            return False, f"time condition altered: introduced ungrounded '{t_label}'"

    for d in [k for k in c_orig["time"] if k.startswith("date:") or k.startswith("month:") or k.startswith("recent_days:")]:
        if d not in c_prop["time"]:
            return False, f"time anchor changed: missing {d}"

    # 6. Temporal Direction (sau vs trước)
    if c_orig.get("temporal_after") != c_prop.get("temporal_after"):
        return False, "temporal direction changed: 'sau' condition altered"
    if c_orig.get("temporal_before") != c_prop.get("temporal_before"):
        return False, "temporal direction changed: 'trước' condition altered"

    # 7. Shifts / Time of day (ca / buổi)
    if c_orig.get("shifts") != c_prop.get("shifts"):
        return False, "shift / time of day condition changed (sáng/chiều/tối/trưa)"

    # 8. Interval Bounds (từ A đến B)
    if c_orig.get("intervals") != c_prop.get("intervals"):
        return False, "interval direction or bounds reversed: từ ... đến ..."

    # 9. Negation & Negation Targets
    if c_orig["negations"] != c_prop["negations"]:
        return False, "negation condition changed"
    if c_orig.get("negation_targets") != c_prop.get("negation_targets"):
        return False, "negation target changed or dropped"

    # 10. Question Intent Invariant (Smoke Test 2 Protection)
    if c_orig["is_interrogative"] and not c_prop["is_interrogative"]:
        return False, "question intent changed: interrogative inquiry turned into assertion or statement"

    if c_orig["is_interrogative"] and c_prop["is_statement_nhieu"]:
        return False, "question intent changed: inquiry turned into assertion ('nhiều tiền' / 'nhiều đơn')"

    orig_intents = set(c_orig["intent_markers"])
    prop_intents = set(c_prop["intent_markers"])
    if ("money" in orig_intents) and not ({"money", "quantity"} & prop_intents):
        return False, "question intent changed: money inquiry dropped"

    if ("percentage" in orig_intents) and ("percentage" not in prop_intents):
        return False, "question intent changed: percentage inquiry dropped"

    # 11. Aggregations (sum, avg, max, min)
    if c_orig.get("aggregations") != c_prop.get("aggregations"):
        return False, "aggregation intent changed (tổng/trung bình/cao nhất/thấp nhất)"

    # 12. Filters & Comparisons Invariant
    if c_orig["filters"] != c_prop["filters"]:
        return False, "filter/comparison condition changed"

    # 13. Ordering Invariant
    if c_orig["ordering"] != c_prop["ordering"]:
        return False, "ordering direction changed"

    # 14. Metric Concepts Invariant
    if c_orig["metrics"] != c_prop["metrics"]:
        return False, "metric/entity concept changed"

    # 15. Customer Attributes Invariant (mới / cũ / quen / lẻ / sỉ)
    if c_orig.get("customer_attributes") != c_prop.get("customer_attributes"):
        return False, "customer attribute changed (mới/cũ/quen/lẻ/sỉ)"

    # 16. Order Status Invariant (hoàn / huỷ / thành công / chưa hoàn)
    if c_orig.get("order_statuses") != c_prop.get("order_statuses"):
        return False, "order status attribute changed (hoàn/huỷ/thành công/chưa hoàn)"

    # 17. Prescription Status Invariant (kê đơn / không kê đơn)
    if c_orig.get("prescription_status") != c_prop.get("prescription_status"):
        return False, "prescription status changed (kê đơn / không kê đơn)"

    # 18. Metric-to-Object Bindings Invariant
    if c_orig.get("metric_bindings") != c_prop.get("metric_bindings"):
        return False, "metric-to-object binding changed or swapped"

    # 19. General Object Attribute & Modifier Bindings (e.g. màu đỏ vs màu xanh, size M vs size L)
    if c_orig.get("attribute_modifiers") != c_prop.get("attribute_modifiers"):
        return False, "object attribute or modifier changed (color/size/type/category)"

    # 20. Content-token alignment & Accented word lock
    ok_tok, reason_tok = _check_content_token_alignment(result.original, proposed, result.protected)
    if not ok_tok:
        return False, reason_tok

    return True, "lexical checks passed; hint only"


# ==============================================================================
# MILESTONE M1: DATA PROFILING WITH PROVENANCE
# ==============================================================================

def inspect_date_column(series: pd.Series, default_year: int | None = None) -> dict:
    """Inspect a series for date format consistency and DMY vs MDY ambiguity.
    Does not modify the series."""
    nonempty = series.dropna().astype(str).str.strip()
    nonempty = nonempty[nonempty != ""]
    if len(nonempty) > MAX_FULL_SCAN_ROWS:
        sampled = True
        scan = nonempty.iloc[:SAMPLE_SCAN_ROWS]
    else:
        sampled = False
        scan = nonempty

    dmy_proof_count = 0
    mdy_proof_count = 0
    ambiguous_dmy_mdy = []
    missing_year_no_hint = []
    iso_count = 0
    vn_text_count = 0

    slash_re = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?(?:\s+.*)?$")
    iso_re = re.compile(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}")
    vn_re = re.compile(r"^\d{1,2}\s*thg\s*\d{1,2}", re.I)

    for idx, val in scan.items():
        if iso_re.match(val):
            iso_count += 1
            continue
        if vn_re.match(val):
            vn_text_count += 1
            continue
        m = slash_re.match(val)
        if m:
            p1, p2, yr = int(m.group(1)), int(m.group(2)), m.group(3)
            if yr is None and default_year is None:
                if len(missing_year_no_hint) < 3:
                    missing_year_no_hint.append({"row": str(idx), "raw": val})
            if p1 > 12 and 1 <= p2 <= 12:
                dmy_proof_count += 1
            elif p2 > 12 and 1 <= p1 <= 12:
                mdy_proof_count += 1
            elif 1 <= p1 <= 12 and 1 <= p2 <= 12 and p1 != p2:
                if len(ambiguous_dmy_mdy) < 3:
                    ambiguous_dmy_mdy.append({"row": str(idx), "raw": val})

    has_conflicting_dmy_mdy = (dmy_proof_count > 0 and mdy_proof_count > 0)
    has_mdy_only = (mdy_proof_count > 0 and dmy_proof_count == 0)
    has_unproven_slash_ambiguity = (len(ambiguous_dmy_mdy) > 0 and dmy_proof_count == 0 and mdy_proof_count == 0)
    syntax_families = sum(x > 0 for x in (dmy_proof_count + mdy_proof_count + len(ambiguous_dmy_mdy), iso_count, vn_text_count))

    return {
        "sampled": sampled,
        "scanned_rows": int(len(scan)),
        "dmy_proof_count": dmy_proof_count,
        "mdy_proof_count": mdy_proof_count,
        "has_conflicting_dmy_mdy": has_conflicting_dmy_mdy,
        "has_mdy_only": has_mdy_only,
        "has_unproven_slash_ambiguity": has_unproven_slash_ambiguity,
        "mixed_syntax_families": syntax_families > 1,
        "ambiguous_dmy_mdy_examples": ambiguous_dmy_mdy,
        "missing_year_examples": missing_year_no_hint,
    }


def column_reports(raw: pd.DataFrame, cleaned: pd.DataFrame, notes: list[str], table_name: str = "", source: str = "") -> list[dict]:
    """Detailed provenance tracking for transformed table columns."""
    reports = []
    for col in raw.columns:
        targets = [col] if col in cleaned and raw[col].dtype != cleaned[col].dtype else []
        if f"{col}__date" in cleaned:
            targets.append(f"{col}__date")
        for target in targets:
            s_raw = raw[col]
            null_cnt = int(s_raw.isna().sum())
            str_s = s_raw.dropna().astype(str).str.strip()
            empty_cnt = int((str_s == "").sum())
            nonempty = s_raw.notna() & s_raw.astype(str).str.strip().ne("")
            failed = nonempty & cleaned[target].isna()
            parsed_cnt = int((nonempty & ~cleaned[target].isna()).sum())
            failed_cnt = int(failed.sum())
            nonempty_cnt = int(nonempty.sum())
            op = "date_parse" if target != col else "numeric_parse_compatibility"
            rule_ev = (
                "derived datetime column (__date) from date-named column"
                if target != col
                else "in-place numeric conversion on uniform thousands/currency column (leading-zero codes excluded)"
            )
            reports.append(dict(
                source_table=str(table_name),
                source=str(source),
                source_column=str(col),
                derived_column=str(target),
                target_column=str(target),
                operation=op,
                rule_or_evidence=rule_ev,
                raw_dtype=str(s_raw.dtype),
                inferred_dtype=str(cleaned[target].dtype),
                null_count=null_cnt,
                empty_string_count=empty_cnt,
                input_nonempty=nonempty_cnt,
                parsed=parsed_cnt,
                succeeded=parsed_cnt,
                failed=failed_cnt,
                parse_success_rate=round(parsed_cnt / max(1, nonempty_cnt), 4),
                examples=[dict(row=str(i), raw=str(raw.at[i, col])) for i in raw.index[failed][:3]],
                notes=list(notes),
            ))
    return reports


def profile_table(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame | None = None,
    notes: list[str] | None = None,
    table_name: str = "",
    source: str = "",
    roles: dict[str, str] | None = None,
    default_year: int | None = None,
) -> dict:
    """Build a comprehensive M1 data profile with provenance, separating:
    - parse_failure: value could not be parsed into target type
    - ambiguity: multiple valid interpretations exist (e.g. 03/04/2025 DMY vs MDY, missing year, mixed separators)
    - anomaly: unusual value that is not necessarily an error (e.g. negative return qty, subtotal row)
    Bounded cost for large tables (>50k rows use deterministic 10k sample for regex checks).
    """
    if cleaned is None:
        cleaned = raw
    notes = notes or []
    roles = roles or {}
    n_raw = len(raw)
    n_clean = len(cleaned)
    is_sampled = n_raw > MAX_FULL_SCAN_ROWS
    scan_df = raw.iloc[:SAMPLE_SCAN_ROWS] if is_sampled else raw

    col_transforms = column_reports(raw, cleaned, notes, table_name=table_name, source=source)
    col_profiles = []
    candidate_keys = []

    date_name_re = re.compile(r"ngày|ngay|date|hạn|han_|thời gian|thoi gian|time|giờ|gio", re.I)

    for col in raw.columns:
        s_raw = raw[col]
        s_scan = scan_df[col]
        null_cnt = int(s_raw.isna().sum())
        is_str_col = s_raw.dtype == object or pd.api.types.is_string_dtype(s_raw)
        if is_str_col:
            non_na = s_scan.dropna().astype(str).str.strip()
            empty_cnt = int((non_na == "").sum())
            valid_str = non_na[non_na != ""]
            if is_sampled:
                nonempty_total = max(0, n_raw - null_cnt - empty_cnt)
            else:
                nonempty_total = int(len(valid_str))
        else:
            empty_cnt = 0
            valid_str = pd.Series([], dtype=str)
            nonempty_total = n_raw - null_cnt
        distinct_cnt = int(s_raw.dropna().nunique())

        # Check candidate key (unique, non-empty, not float)
        is_cand_key = (
            nonempty_total == n_raw
            and n_raw > 0
            and distinct_cnt == n_raw
            and not pd.api.types.is_float_dtype(s_raw)
        )
        if is_cand_key:
            candidate_keys.append(str(col))

        # Check leading zero code preservation
        has_leading_zero = bool(not valid_str.empty and valid_str.str.match(r"^0\d+$").any())
        inferred_col = cleaned[col] if col in cleaned else s_raw
        leading_zero_preserved = bool(
            not has_leading_zero
            or (inferred_col.dtype == object or pd.api.types.is_string_dtype(inferred_col))
        )

        parse_failures = []
        ambiguities = []
        anomalies = []

        # 1. Parse failures (from derived date or numeric conversion)
        target_col = f"{col}__date" if f"{col}__date" in cleaned else (col if col in cleaned and cleaned[col].dtype != s_raw.dtype else None)
        failed_cnt = 0
        parsed_cnt = nonempty_total
        if target_col is not None:
            mask_nonempty = s_raw.notna() & s_raw.astype(str).str.strip().ne("")
            mask_failed = mask_nonempty & cleaned[target_col].isna()
            failed_cnt = int(mask_failed.sum())
            parsed_cnt = int((mask_nonempty & ~cleaned[target_col].isna()).sum())
            if failed_cnt > 0:
                parse_failures.append({
                    "kind": "type_parse_failure",
                    "target_column": target_col,
                    "count": failed_cnt,
                    "examples": [{"row": str(i), "raw": str(raw.at[i, col])} for i in raw.index[mask_failed][:3]],
                })
        elif not valid_str.empty and not date_name_re.search(str(col)) and not has_leading_zero:
            # Check if column is mostly numeric (>=70%) but has some unparseable strings (causing in-place conversion to be skipped)
            num_probe = pd.to_numeric(valid_str.str.replace(r"[.,đd\s]", "", regex=True), errors="coerce")
            num_rate = float(num_probe.notna().mean())
            if 0.65 <= num_rate < 1.0:
                bad_idx = valid_str.index[num_probe.isna()]
                failed_cnt = int(len(bad_idx))
                parsed_cnt = int(num_probe.notna().sum())
                parse_failures.append({
                    "kind": "unconverted_numeric_outliers",
                    "target_column": str(col),
                    "count": failed_cnt,
                    "examples": [{"row": str(i), "raw": str(scan_df.at[i, col])} for i in bad_idx[:3]],
                })

        # 2. Ambiguities (date order DMY vs MDY, missing year, mixed thousand/decimal separators)
        if date_name_re.search(str(col)) and not valid_str.empty:
            d_info = inspect_date_column(s_raw, default_year=default_year)
            if d_info["has_conflicting_dmy_mdy"]:
                ambiguities.append({
                    "kind": "conflicting_dmy_and_mdy_dates",
                    "description": "Column contains both DD/MM (>12 first) and MM/DD (>12 second) dates; rows with both <=12 are ambiguous.",
                    "examples": d_info["ambiguous_dmy_mdy_examples"],
                })
            elif d_info["has_mdy_only"]:
                ambiguities.append({
                    "kind": "mdy_date_format_detected",
                    "description": "Column has MM/DD dates (second part >12); standard DMY assumption would misparse.",
                    "examples": d_info["ambiguous_dmy_mdy_examples"],
                })
            elif d_info["has_unproven_slash_ambiguity"]:
                ambiguities.append({
                    "kind": "ambiguous_date_order_dmy_vs_mdy",
                    "description": "All slash dates have both day and month <=12 without a >12 anchor in the column.",
                    "examples": d_info["ambiguous_dmy_mdy_examples"],
                })
            if d_info["missing_year_examples"]:
                ambiguities.append({
                    "kind": "date_missing_year_without_context",
                    "description": "Date values omit year and no sheet/file year hint was found.",
                    "examples": d_info["missing_year_examples"],
                })
            if d_info["mixed_syntax_families"]:
                ambiguities.append({
                    "kind": "inconsistent_date_syntax_formats",
                    "description": "Multiple date syntax styles (e.g. ISO, slash, Vietnamese text) mixed in one column.",
                    "examples": [{"row": str(i), "raw": str(v)} for i, v in list(valid_str.head(3).items())],
                })

        if not valid_str.empty and not date_name_re.search(str(col)):
            has_comma_sep = bool(valid_str.str.match(r"^-?\d{1,3}(?:,\d{3})+$").any())
            has_dot_sep = bool(valid_str.str.match(r"^-?\d{1,3}(?:\.\d{3})+$").any())
            if has_comma_sep and has_dot_sep:
                ambiguities.append({
                    "kind": "mixed_thousand_decimal_separators",
                    "description": "Both comma-grouped ('1,234') and dot-grouped ('1.234') numbers appear in the same column.",
                    "examples": [{"row": str(i), "raw": str(v)} for i, v in list(valid_str.head(3).items())],
                })

        # 3. Anomalies (negative numbers in qty/amount, subtotal rows)
        num_series = cleaned[col] if (col in cleaned and pd.api.types.is_numeric_dtype(cleaned[col])) else (
            s_raw if pd.api.types.is_numeric_dtype(s_raw) else None
        )
        if num_series is not None:
            neg_mask = num_series.notna() & (num_series < 0)
            neg_cnt = int(neg_mask.sum())
            if neg_cnt > 0:
                anomalies.append({
                    "kind": "negative_numeric_values",
                    "count": neg_cnt,
                    "note": "Anomaly only (may represent valid returns, discounts, or adjustments per rulebook §1).",
                    "examples": [{"row": str(i), "raw": str(raw.at[i, col])} for i in raw.index[neg_mask][:3]],
                })

        col_profiles.append({
            "column": str(col),
            "raw_dtype": str(s_raw.dtype),
            "inferred_dtype": str(inferred_col.dtype),
            "total_rows": n_raw,
            "null_count": null_cnt,
            "empty_string_count": empty_cnt,
            "nonempty_count": nonempty_total,
            "distinct_count": distinct_cnt,
            "parsed_count": parsed_cnt,
            "parse_failure_count": failed_cnt,
            "parse_success_rate": round(parsed_cnt / max(1, nonempty_total), 4) if nonempty_total else 1.0,
            "is_candidate_key": is_cand_key,
            "has_leading_zero_codes": has_leading_zero,
            "leading_zero_preserved": leading_zero_preserved,
            "parse_failures": parse_failures,
            "ambiguities": ambiguities,
            "anomalies": anomalies,
        })

    # Duplicate row analysis on scan_df
    exact_dup_mask = scan_df.duplicated(keep="first")
    exact_dup_cnt = int(exact_dup_mask.sum())
    dup_examples = [str(i) for i in scan_df.index[exact_dup_mask][:3]]

    return {
        "table_name": table_name,
        "source": source,
        "input_rows": n_raw,
        "parsed_rows": n_clean,
        "sampling": {
            "sampled": is_sampled,
            "scanned_rows": len(scan_df),
            "total_rows": n_raw,
            "full_scan_operations": ["input_rows", "parsed_rows", "null_count", "distinct_count"],
            "sampled_operations": ["regex_format_checks", "duplicate_scan"] if is_sampled else [],
            "notice": "Sampled checks do not guarantee detecting every row-level anomaly in large tables.",
        },
        "candidate_keys": candidate_keys,
        "duplicates": {
            "exact_duplicate_rows": exact_dup_cnt,
            "example_rows": dup_examples,
            "policy": "Raw rows preserved; non-order table duplicates are valid separate records (rulebook §8), same-order duplicate lines flagged via is_dup_line.",
        },
        "columns": col_transforms,
        "column_profiles": col_profiles,
        "join_diagnostics": [],
    }


# ==============================================================================
# MILESTONE M2: SCHEMA & VALUE GROUNDING
# ==============================================================================

def build_grounding_index(ws, max_values_per_col: int = MAX_INDEX_VALUES_PER_COL) -> dict:
    """Build a bounded, cached schema and catalog entity index for a Workspace.
    Uses exact Unicode NFC and accent-folded keys without altering raw data.
    Zero neural parameters (strictly respects <=15B competition budget)."""
    cached = getattr(ws, "_grounding_index_cache", None)
    if cached is not None:
        return cached

    schema_entries = []
    entity_entries = []
    nfc_to_entities: dict[str, list[dict]] = {}
    folded_to_entities: dict[str, list[dict]] = {}
    code_to_entities: dict[str, list[dict]] = {}

    entity_roles = {"item_code", "item", "item_group", "customer_code", "customer", "channel"}
    code_roles = {"item_code", "customer_code", "order_id"}

    for t in ws.tables.values():
        if t.shared or t.source.startswith("derived:"):
            continue
        role_by_col = {col: r for r, col in t.roles.items()}
        for col in t.df.columns:
            if str(col).startswith("_") or str(col).endswith("__date"):
                continue
            meta = t.meta.get(col) or {}
            orig_name = str(meta.get("orig") or col)
            desc = str(meta.get("desc") or "")
            role = meta.get("role") or role_by_col.get(col)

            schema_entry = {
                "table": t.name,
                "column": str(col),
                "orig_name": orig_name,
                "role": role,
                "desc": desc[:160],
                "nfc_key": norm_nfc(orig_name),
                "folded_key": fold(orig_name),
                "desc_folded": fold(desc),
            }
            schema_entries.append(schema_entry)

            # Index entity columns and code/name columns
            col_slug = fold(orig_name)
            is_entity_col = (
                role in entity_roles
                or any(
                    k in col_slug
                    for k in (
                        "ten_", "ten ", "ma_", "ma ", "mat hang", "mat_hang",
                        "san pham", "san_pham", "khach", "nhom", "hoat chat",
                        "hoat_chat", "chi nhanh", "chi_nhanh", "cua hang", "cua_hang",
                        "kenh", "loai", "danh muc", "danh_muc",
                    )
                )
            )
            if not is_entity_col:
                continue

            s = t.df[col].dropna()
            if s.empty:
                continue
            unique_vals = s.astype(str).str.strip().unique()[:max_values_per_col]
            for raw_v in unique_vals:
                if not raw_v or len(raw_v) < 2 or raw_v.lower() in ("nan", "none"):
                    continue
                nfc_k = norm_nfc(raw_v)
                fold_k = fold(raw_v)
                is_code = bool(
                    role in code_roles
                    or re.fullmatch(r"0\d+", raw_v)
                    or re.fullmatch(r"[A-Za-z]{1,6}\d+[A-Za-z0-9_-]*", raw_v)
                )
                ent = {
                    "raw_value": raw_v,
                    "table": t.name,
                    "column": str(col),
                    "role": role or "entity",
                    "nfc_key": nfc_k,
                    "folded_key": fold_k,
                    "is_code": is_code,
                    "has_leading_zero": bool(re.fullmatch(r"0\d+", raw_v)),
                }
                entity_entries.append(ent)
                nfc_to_entities.setdefault(nfc_k, []).append(ent)
                folded_to_entities.setdefault(fold_k, []).append(ent)
                if is_code:
                    code_to_entities.setdefault(raw_v, []).append(ent)

    index = {
        "schema": schema_entries,
        "entities": entity_entries,
        "nfc_map": nfc_to_entities,
        "folded_map": folded_to_entities,
        "code_map": code_to_entities,
    }
    try:
        ws._grounding_index_cache = index
    except Exception:
        pass
    return index


def ground_question(
    question: str,
    ws_or_index,
    max_candidates: int = MAX_GROUNDING_CANDIDATES,
) -> dict:
    """Ground a question against workspace schema and entity catalog:
    1. Exact NFC / exact code matches first (leading zeros strictly respected: '00123' != '123').
    2. Accent-folded and partial-token matches are returned ONLY as candidates, never auto-merged.
    3. If multiple distinct entities match an unaccented/partial/abbreviated query span, marks
       ambiguous=True and keeps the candidate list with provenance.
    """
    index = ws_or_index if isinstance(ws_or_index, dict) and "schema" in ws_or_index else build_grounding_index(ws_or_index)
    q_nfc = norm_nfc(question)
    q_fold = fold(question)
    q_tokens_fold = set(re.findall(r"[a-z0-9]+", q_fold))

    exact_matches = []
    candidates = []
    seen_exact_vals = set()
    seen_cand_vals = set()
    ambiguous = False
    ambiguity_reasons = []

    # 1. Check code tokens in question (strictly respecting leading zeros)
    q_num_codes = re.findall(r"\b(?:0\d+|[A-Za-z]{1,6}\d+[A-Za-z0-9_-]*|\d{3,})\b", question)
    for q_code in q_num_codes:
        # Exact code match
        if q_code in index["code_map"]:
            for ent in index["code_map"][q_code]:
                key = (ent["table"], ent["column"], ent["raw_value"])
                if key not in seen_exact_vals:
                    seen_exact_vals.add(key)
                    exact_matches.append({**ent, "match_type": "exact_code", "matched_span": q_code})
        else:
            # Check if user wrote '123' without leading zero when catalog has '00123' -> candidate only, never exact!
            for raw_code, ents in index["code_map"].items():
                if raw_code.lstrip("0") == q_code.lstrip("0") and raw_code != q_code:
                    for ent in ents:
                        key = (ent["table"], ent["column"], ent["raw_value"])
                        if key not in seen_cand_vals:
                            seen_cand_vals.add(key)
                            candidates.append({
                                **ent,
                                "match_type": "leading_zero_code_candidate",
                                "matched_span": q_code,
                                "note": f"Query '{q_code}' differs in leading zeros from catalog code '{raw_code}'; do not auto-merge.",
                            })
                    ambiguous = True
                    ambiguity_reasons.append(f"code '{q_code}' has leading-zero variant '{raw_code}'")

    # 2. Exact NFC phrase match for catalog entities (longest first)
    sorted_nfc_keys = sorted(index["nfc_map"].keys(), key=len, reverse=True)
    for nfc_k in sorted_nfc_keys:
        if len(nfc_k) < 2:
            continue
        pat = r"(?<!\w)" + re.escape(nfc_k) + r"(?!\w)"
        if re.search(pat, q_nfc):
            ents = index["nfc_map"][nfc_k]
            for ent in ents:
                key = (ent["table"], ent["column"], ent["raw_value"])
                if key not in seen_exact_vals:
                    seen_exact_vals.add(key)
                    exact_matches.append({**ent, "match_type": "exact_nfc", "matched_span": nfc_k})

    # Check if an exact NFC match is also a prefix/substring of other distinct catalog items
    # (e.g. query says "Cà phê sữa" and catalog has both "Cà phê sữa (M)" and "Cà phê sữa (L)", or "Sữa tươi có đường" vs "Sữa tươi ít đường")
    for nfc_k, ents in index["nfc_map"].items():
        for ex in list(exact_matches):
            ex_k = ex["nfc_key"]
            if nfc_k != ex_k and re.search(r"(?<!\w)" + re.escape(ex_k) + r"(?!\w)", nfc_k):
                # Only trigger if the longer item is NOT itself fully in q_nfc
                if not re.search(r"(?<!\w)" + re.escape(nfc_k) + r"(?!\w)", q_nfc):
                    for ent in ents:
                        key = (ent["table"], ent["column"], ent["raw_value"])
                        if key not in seen_exact_vals and key not in seen_cand_vals:
                            seen_cand_vals.add(key)
                            candidates.append({**ent, "match_type": "more_specific_variant_candidate", "matched_span": ex_k})
                    ambiguous = True
                    ambiguity_reasons.append(f"'{ex['raw_value']}' has more specific catalog variants (e.g. '{ents[0]['raw_value']}')")

    # 3. Accent-folded match (when query is unaccented or differs in diacritics)
    for fold_k, ents in index["folded_map"].items():
        if len(fold_k) < 3:
            continue
        pat = r"(?<!\w)" + re.escape(fold_k) + r"(?!\w)"
        if re.search(pat, q_fold):
            distinct_raw = {e["raw_value"] for e in ents}
            if len(distinct_raw) > 1:
                ambiguous = True
                ambiguity_reasons.append(f"folded phrase '{fold_k}' matches {len(distinct_raw)} distinct accented entities: {sorted(distinct_raw)[:3]}")
            for ent in ents:
                key = (ent["table"], ent["column"], ent["raw_value"])
                if key not in seen_exact_vals and key not in seen_cand_vals:
                    seen_cand_vals.add(key)
                    candidates.append({
                        **ent,
                        "match_type": "accent_folded_candidate",
                        "matched_span": fold_k,
                    })

    # 4. Partial token / abbreviation candidate search for multi-token entities
    stop_entity_tokens = BENIGN_STRUCTURAL_TOKENS | {
        "bao", "nhieu", "bnhiu", "tien", "doanh", "thu", "dthu", "ngay", "thang", "nam",
        "tuan", "hom", "qua", "nay", "don", "hang", "khach", "san", "pham", "mat", "mon",
        "ban", "mua", "ton", "kho", "nhap", "gia", "von", "loi", "nhuan", "tong", "cong",
    }
    query_content_tokens = [t for t in re.findall(r"[a-z0-9]+", q_fold) if t not in stop_entity_tokens and len(t) >= 2]
    if query_content_tokens:
        q_content_set = set(query_content_tokens)
        scored_partials = []
        for ent in index["entities"]:
            if ent["is_code"]:
                continue
            key = (ent["table"], ent["column"], ent["raw_value"])
            if key in seen_exact_vals or key in seen_cand_vals:
                continue
            ent_tokens = [t for t in re.findall(r"[a-z0-9]+", ent["folded_key"]) if len(t) >= 2]
            if not ent_tokens:
                continue
            ent_set = set(ent_tokens)
            shared = q_content_set & ent_set
            # Also check abbreviation initials (e.g. 'cf' for 'ca phe', 'st' for 'sua tuoi')
            initials = "".join(w[0] for w in ent_tokens if w)
            abbr_hit = len(initials) >= 2 and initials in q_content_set
            if len(shared) >= max(1, min(2, len(ent_set))) or abbr_hit:
                score_val = (len(shared) / max(1, len(ent_set))) + (0.5 if abbr_hit else 0.0)
                if score_val >= 0.5:
                    scored_partials.append((score_val, ent, sorted(shared) or [initials]))

        scored_partials.sort(key=lambda x: (-x[0], x[1]["raw_value"]))
        if len(scored_partials) > 1:
            top_score = scored_partials[0][0]
            tied_top = [x[1]["raw_value"] for x in scored_partials if abs(x[0] - top_score) < 1e-6]
            if len(tied_top) > 1 and not exact_matches:
                ambiguous = True
                ambiguity_reasons.append(f"multiple catalog candidates match query tokens: {tied_top[:4]}")

        for score_val, ent, matched_toks in scored_partials[:max_candidates]:
            key = (ent["table"], ent["column"], ent["raw_value"])
            if key not in seen_cand_vals:
                seen_cand_vals.add(key)
                candidates.append({
                    **ent,
                    "match_type": "partial_or_abbr_candidate",
                    "matched_span": " ".join(matched_toks),
                    "score": round(score_val, 3),
                })

    # 5. Schema (table/column/README) grounding
    schema_matches = []
    for sc in index["schema"]:
        if sc["nfc_key"] and len(sc["nfc_key"]) >= 3 and sc["nfc_key"] in q_nfc:
            schema_matches.append({**sc, "match_type": "exact_column_name"})
        elif sc["folded_key"] and len(sc["folded_key"]) >= 3 and sc["folded_key"] in q_fold:
            schema_matches.append({**sc, "match_type": "folded_column_name"})
        else:
            sc_toks = set(re.findall(r"[a-z0-9]+", sc["folded_key"] + " " + sc["desc_folded"])) - BENIGN_STRUCTURAL_TOKENS
            overlap = q_tokens_fold & sc_toks
            if len(overlap) >= 2:
                schema_matches.append({**sc, "match_type": "schema_keyword_overlap", "overlap": sorted(overlap)})

    candidates = candidates[:max_candidates]
    has_entity_match = bool(exact_matches or candidates)

    return {
        "exact_matches": exact_matches[:max_candidates],
        "candidates": candidates,
        "schema_matches": schema_matches[:max_candidates],
        "ambiguous": ambiguous,
        "ambiguity_reasons": ambiguity_reasons,
        "has_entity_match": has_entity_match,
    }


def render_grounding_hint(grounding: dict, max_chars: int = MAX_GROUNDING_CHARS) -> str:
    """Render a compact, bounded grounding summary for the solver prompt."""
    if not grounding:
        return ""
    lines = []
    if grounding.get("exact_matches"):
        items = [f"{e['raw_value']!r} ({e['table']}.{e['column']})" for e in grounding["exact_matches"][:5]]
        lines.append("Exact catalog matches: " + ", ".join(items))
    if grounding.get("candidates"):
        cands = [f"{e['raw_value']!r} ({e['table']}.{e['column']}, {e['match_type']})" for e in grounding["candidates"][:6]]
        lines.append("Candidate entities (verify in data; do not assume identical): " + ", ".join(cands))
    if grounding.get("ambiguous") and grounding.get("ambiguity_reasons"):
        lines.append("Entity ambiguity notes: " + "; ".join(grounding["ambiguity_reasons"][:3]))
    if not lines:
        return ""
    text = "# Grounded catalog candidates (advisory)\n" + "\n".join(f"- {line}" for line in lines)
    return text[:max_chars]


def shop_payload(ws):
    """Only unresolved source columns and bounded document excerpts; never gold answers."""
    columns = []
    for table in ws.tables.values():
        if table.shared or table.source.startswith("derived:"):
            continue
        for col in table.df.columns:
            if col in table.roles.values() or str(col).startswith("_") or "__" in str(col):
                continue
            key = f"{table.name}.{col}"
            columns.append(dict(
                key=key,
                dtype=str(table.df[col].dtype),
                samples=[str(v)[:80] for v in table.df[col].dropna().head(3)]
            ))
    return dict(columns=columns[:24], docs={k: v[:2500] for k, v in list(ws.docs.items())[:2]})


def validate_shop_hints(payload, response):
    if not isinstance(response, dict) or set(response) != {"hints"} or not isinstance(response["hints"], list):
        raise ValueError("invalid shop schema")
    allowed = {c["key"]: c for c in payload["columns"]}
    haystack = json.dumps(payload, ensure_ascii=False)
    output = []
    for hint in response["hints"]:
        if not isinstance(hint, dict) or set(hint) != {"column", "meaning", "evidence"}:
            raise ValueError("invalid hint schema")
        if hint["column"] not in allowed or not all(isinstance(v, str) and 0 < len(v) <= 300 for v in hint.values()):
            raise ValueError("unknown column/invalid value")
        if hint["evidence"] not in haystack:
            raise ValueError("unsupported evidence")
        output.append(hint)
    return output[:12]
