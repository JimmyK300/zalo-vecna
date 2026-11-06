"""Conservative cleaner: hints for interpretation, never rewritten business facts."""
from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field

VERSION = "cleaner-v3"

# Safe unambiguous word-level aliases
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


def fold(text: str) -> str:
    """Normalize for case-insensitive matching while stripping diacritics."""
    text = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


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
    ambiguous: bool = False
    ambiguous_spans: list[str] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    protected_spans: list[dict] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.normalized_hint:
            self.normalized_hint = self.normalized

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


def extract_constraints(text: str) -> dict:
    """Extract semantic invariant anchors: relative time, question intent, negations, filters, ordering, metrics."""
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

    # Recent N days (e.g. 7 ngay gan nhat, 30 ngay gan nhat)
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

    # Shifts / Time of day (ca / buổi / sáng / chiều / tối / trưa / đêm)
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

    # Aggregations:
    aggregations = []
    if re.search(r"\b(?:tong|tong cong|tong so|tong dthu|tong doanh thu)\b", f):
        aggregations.append("sum")
    if re.search(r"\b(?:trung binh|binh quan|tb)\b", f):
        aggregations.append("avg")
    if re.search(r"\b(?:cao nhat|nhieu nhat|lon nhat|top|chay nhat)\b", f):
        aggregations.append("max")
    if re.search(r"\b(?:thap nhat|it nhat|nho nhat|e nhat)\b", f):
        aggregations.append("min")

    # Flag statements that erroneously turned question into statement ('nhiều tiền' / 'nhiều đơn')
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

    # 8. Order status attributes (hoan / huy / thanh cong / chua hoan)
    order_statuses = []
    if re.search(r"\b(?:don\s+(?:da\s+)?hoan|don\s+tra|tra\s+hang|doi\s+tra|hoan\s+tra)\b", f):
        order_statuses.append("don_hoan")
    if re.search(r"\b(?:don\s+(?:bi\s+|da\s+)?huy|huy\s+don|bi\s+huy)\b", f):
        order_statuses.append("don_huy")
    if re.search(r"\b(?:don\s+(?:da\s+)?thanh\s+cong|giao\s+thanh\s+cong|hoan\s+thanh|don\s+hoan\s+thanh)\b", f):
        order_statuses.append("don_thanh_cong")
    if re.search(r"\b(?:don\s+(?:chua\s+)?hoan\s+tat|chua\s+giao|dang\s+giao)\b", f):
        order_statuses.append("don_chua_hoan")

    # 9. Prescription status (ke don / khong ke don)
    prescription_status = []
    if re.search(r"\b(?:khong\s+ke\s+don|ko\s+ke\s+don|otc)\b", f):
        prescription_status.append("khong_ke_don")
    elif re.search(r"\b(?:ke\s+don|thuoc\s+ke\s+don)\b", f):
        prescription_status.append("ke_don")

    # 10. Metric to object bindings (e.g. doanh thu cua hang A vs loi nhuan cua hang B)
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
        "metric_bindings": sorted(metric_bindings)
    }


def clean_question(text: str, entities=()) -> QuestionCleanResult:
    """Safe rule-based normalization:
    - Normalizes Unicode NFC & whitespace
    - Replaces unambiguous abbreviations (hqua -> hôm qua, hnay -> hôm nay, dthu -> doanh thu, etc.)
    - Replaces multi-word phrases safely from right to left without offset drift
    - Protects numbers, dates, negative signs, units, codes, and entities verbatim
    - Never turns 'nhiu' into 'nhiều' blindly; preserves ambiguity for LLM/routing
    """
    original = text
    text = unicodedata.normalize("NFC", text)
    protected, protected_spans = extract_protected_spans(text, entities)

    def is_protected(s: int, e: int) -> bool:
        for p in protected_spans:
            if not (e <= p["start"] or s >= p["end"]):
                return True
        return False

    # Collect candidate replacements on current text
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

    # 3. Detect ambiguous words and telex typos
    warnings = []
    ambiguous_spans = []
    f = fold(text)

    # Check for standalone 'nhiu' (NOT preceded by 'bao')
    if re.search(r"(?<!bao\s)\bnhiu\b", f):
        ambiguous_spans.append("nhiu")
        warnings.append("ambiguous chat word 'nhiu' (may mean 'bao nhiêu' or 'nhiều'); keep original")

    # Check other ambiguous short tokens
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

    if re.search(r"(?:[aeiouy]{2}|(?:ngay|nao|nhat)[fsrxj]\b)", f):
        warnings.append("possible repeated letter/Telex typo")

    is_ambiguous = bool(ambiguous_spans) or bool(warnings)
    constraints = extract_constraints(original)

    return QuestionCleanResult(
        original=original,
        normalized=text,
        normalized_hint=text,
        ambiguous=is_ambiguous,
        ambiguous_spans=sorted(set(ambiguous_spans)),
        changes=changes,
        warnings=warnings,
        protected=protected,
        protected_spans=protected_spans,
        constraints=constraints
    )


def validate_rewrite(result: QuestionCleanResult, proposed: str) -> tuple[bool, str]:
    """Strict guard against semantic distortion:
    1. Validates non-empty, bounded length rewrite
    2. Protects numbers, negative signs, codes with leading zeros, units, entities verbatim
    3. Guarantees relative time conditions (hôm qua, hôm nay, tuần rồi, etc.) are never dropped
    4. Guarantees temporal direction (sau vs trước), shifts, and intervals are strictly preserved
    5. Guarantees negation and its targets are preserved
    6. Guarantees question intent and aggregations are preserved
    """
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

    # Check explicit date & month anchors
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

    return True, "lexical checks passed; hint only"


def column_reports(raw, cleaned, notes):
    """Detailed provenance tracking for transformed table columns."""
    reports = []
    for col in raw.columns:
        targets = [col] if col in cleaned and raw[col].dtype != cleaned[col].dtype else []
        if f"{col}__date" in cleaned:
            targets.append(f"{col}__date")
        for target in targets:
            nonempty = raw[col].notna() & raw[col].astype(str).str.strip().ne("")
            failed = nonempty & cleaned[target].isna()
            reports.append(dict(
                source_column=str(col),
                derived_column=str(target),
                operation="date_parse" if target != col else "numeric_parse_compatibility",
                input_nonempty=int(nonempty.sum()),
                parsed=int((nonempty & ~cleaned[target].isna()).sum()),
                failed=int(failed.sum()),
                examples=[dict(row=str(i), raw=str(raw.at[i, col])) for i in raw.index[failed][:3]],
                notes=list(notes)
            ))
    return reports


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
        # Evidence must quote supplied material; it still does not prove the claimed meaning.
        if hint["evidence"] not in haystack:
            raise ValueError("unsupported evidence")
        output.append(hint)
    return output[:12]
