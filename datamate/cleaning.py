"""Conservative cleaner: hints for interpretation, never rewritten business facts."""
from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field

VERSION = "cleaner-v1"
ALIASES = {"hqua": "hôm qua", "hnay": "hôm nay", "bnhiu": "bao nhiêu",
           "ko": "không", "khum": "không", "trc": "trước", "dthu": "doanh thu"}


def fold(text):
    text = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


@dataclass
class QuestionCleanResult:
    original: str
    normalized: str
    ambiguous: bool = False
    changes: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    llm: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def clean_question(text, entities=()):
    original = text
    text = unicodedata.normalize("NFC", text)
    protected = [e for e in entities if isinstance(e, str) and e and e in text]
    # Protect numbers with their unit, mixed letter/number identifiers, and known entities.
    protected += re.findall(r"\b\w*\d\w*(?:[.,/:%-]\w+)*%?", text)
    protected += re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:ml|mg|kg|cm|mm|k|tr|%)\b", text, re.I)
    matches = list(re.finditer(r"\b(?:" + "|".join(ALIASES) + r")\b", text))
    changes = []
    for m in reversed(matches):
        if any(m[0] in p for p in protected):
            continue
        changes.append(dict(start=m.start(), end=m.end(), before=m[0], after=ALIASES[m[0]], kind="alias"))
        text = text[:m.start()] + ALIASES[m[0]] + text[m.end():]
    text = re.sub(r"\s+", " ", text).strip()
    warnings = []
    if re.search(r"\b(?:bn|nhiu|dt|kh|th|ni|mik|dc)\b", fold(text)):
        warnings.append("ambiguous chat word: retain original; do not assume one expansion")
    if re.search(r"(?:[aeiouy]{2}|(?:ngay|nao|nhat)[fsrxj]\b)", fold(text)):
        warnings.append("possible repeated letter/Telex typo")
    return QuestionCleanResult(original, text, bool(warnings), list(reversed(changes)), warnings,
                               sorted(set(protected), key=len, reverse=True))


def validate_rewrite(result, proposed):
    if not isinstance(proposed, str) or not proposed.strip() or len(proposed) > max(512, len(result.original) * 3):
        return False, "empty/oversize rewrite"
    numbers = lambda s: re.findall(r"(?<!\w)[+-]?\d+(?:[.,/:%-]\d+)*%?", s)
    if numbers(proposed) != numbers(result.original):
        return False, "numbers/dates/order changed"
    if any(span not in proposed for span in result.protected):
        return False, "protected entity/code/unit changed"
    # Conservative lexical checks, not a proof of semantic equivalence.
    def conditions(s):
        s = fold(s)
        for a, b in {"ko": "khong", "khum": "khong", "trc": "truoc"}.items():
            s = re.sub(r"\b" + a + r"\b", b, s)
        return Counter(re.findall(r"\b(?:khong|chua|tru|ngoai|truoc|sau|hon|kem|it nhat|nhieu nhat|cuoi|dau|tro len|tro xuong)\b", s))
    if conditions(proposed) != conditions(result.original):
        return False, "negation/comparison/time condition changed"
    def concepts(s):
        s = fold(s)
        s = re.sub(r"\bdthu\b", "doanh thu", s)
        s = s.replace("binh quan", "trung binh")
        return Counter(re.findall(r"\b(?:doanh thu|gia ban|gia von|cong no|ton kho|so luong|trung binh|don|khach|nhom)\b", s))
    if concepts(proposed) != concepts(result.original):
        return False, "metric/entity concept changed"
    return True, "lexical checks passed; hint only"


def column_reports(raw, cleaned, notes):
    reports = []
    for col in raw.columns:
        targets = [col] if col in cleaned and raw[col].dtype != cleaned[col].dtype else []
        if f"{col}__date" in cleaned:
            targets.append(f"{col}__date")
        for target in targets:
            nonempty = raw[col].notna() & raw[col].astype(str).str.strip().ne("")
            failed = nonempty & cleaned[target].isna()
            reports.append(dict(source_column=str(col), derived_column=str(target),
                operation="date_parse" if target != col else "numeric_parse_compatibility",
                input_nonempty=int(nonempty.sum()), parsed=int((nonempty & ~cleaned[target].isna()).sum()),
                failed=int(failed.sum()), examples=[dict(row=str(i), raw=str(raw.at[i, col]))
                    for i in raw.index[failed][:3]], notes=list(notes)))
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
            columns.append(dict(key=key, dtype=str(table.df[col].dtype),
                samples=[str(v)[:80] for v in table.df[col].dropna().head(3)]))
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
