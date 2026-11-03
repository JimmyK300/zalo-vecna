"""Candidate relationship graph (layer 3): a small per-shop graph of tables, columns, joins,
README caveats and rulebook rules, used to write a short "plan card" for each question.

Nodes: tables, columns (with README meaning + role), rulebook rows (section > rule: text).
Edges: table-table joins with an overlap score; column caveats (e.g. "balance at the END of the
data" — a trap for "as of day X" questions); question -> rules / tables by lexical retrieval.

Nothing here decides an answer; it narrows where to look and flags traps. Deterministic and
rule-free: the rules it shows are quoted from the current rulebook.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .items import item_card
from .metrics import metric_card
from .primitives import resolve_periods
from .recipes import debt_card, debt_mentioned
from .semantic_model import semantic_on
from .semantics import strip_accents

JOIN_ROLES = ("order_id", "customer_code", "item_code", "item", "customer")
CAVEAT_RE = re.compile(r"cuối dữ liệu|hiện tại|cột trống|không có dữ liệu|chỉ đơn|dòng trả hàng|"
                       r"trước giảm giá|chưa gồm|đã tính|gồm phí|không gồm|không phải|lặp lại|ghi tay", re.I)
SLANG = {  # common chat abbreviations in the questions -> words used in the rulebook / READMEs
    "dthu": "doanh thu", "dt": "doanh thu", "hqua": "hom qua", "hnay": "hom nay", "bnhiu": "bao nhieu",
    "bn": "bao nhieu", "nhiu": "nhieu", "ko": "khong", "k": "khong", "sp": "san pham", "kh": "khach",
    "th": "thang", "tg": "thoi gian", "sl": "so luong", "dc": "duoc", "đc": "duoc", "vs": "voi",
}
STOP = set("la co cua cho toi minh giup xem coi hoi nhi ha nao bao nhieu va cac nhung mot trong tu den "
           "het ngay thang nam the nay do duoc khong voi so voi".split())


def _tokens(text: str) -> list[str]:
    t = re.sub(r"[^a-z0-9 ]+", " ", strip_accents(text))
    words = []
    for w in t.split():
        words.extend(SLANG.get(w, w).split())
    return [w for w in words if len(w) > 1 and w not in STOP]


def _grams(text: str) -> Counter:
    w = _tokens(text)
    return Counter(w + [f"{a}_{b}" for a, b in zip(w, w[1:])])


@dataclass
class RuleRow:
    section: str
    key: str
    text: str

    def render(self) -> str:
        return f"[{self.section}] {self.key}: {self.text}"


def parse_rulebook(text: str) -> list[RuleRow]:
    rows, sec = [], ""
    for line in text.splitlines():
        h = re.match(r"^#{2,3}\s+(.*)", line)
        if h:
            sec = h.group(1).strip()
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if len(cells) >= 2 and not set(cells[0]) <= set("-: ") and cells[0].lower() not in ("luật", "cách nói", "từ", "cách ghi", "type"):
            rows.append(RuleRow(sec, re.sub(r"\*", "", cells[0]), re.sub(r"\*", "", " | ".join(cells[1:]))))
        elif line.startswith("- ") and sec:
            rows.append(RuleRow(sec, "", re.sub(r"\*", "", line[2:])))
    return rows


@dataclass
class ShopGraph:
    joins: list[tuple[str, str, str, str, float]] = field(default_factory=list)   # t1, c1, t2, c2, overlap
    caveats: dict[str, list[str]] = field(default_factory=dict)                     # table -> notes
    table_text: dict[str, Counter] = field(default_factory=dict)


def build_graph(ws) -> ShopGraph:
    g = ShopGraph()
    tabs = [t for t in ws.tables.values() if not t.shared]
    cols: list[tuple[str, str, set]] = []
    for t in tabs:
        words = [t.name, t.source] + [f"{c} {m.get('desc', '')}" for c, m in t.meta.items()]
        g.table_text[t.name] = _grams(" ".join(words))
        for c, m in t.meta.items():
            if m.get("desc") and CAVEAT_RE.search(m["desc"]):
                g.caveats.setdefault(t.name, []).append(f"{c}: {m['desc'][:140]}")
            if m.get("role") == "balance_end":
                g.caveats.setdefault(t.name, []).append(
                    f"{c} is the balance at the END of the data, not as of the asked day: for 'đến hết ngày X' "
                    f"rebuild it from order totals and payments dated <= X (as_of_balance)")
        if t.name.startswith("orders"):
            continue
        for role in JOIN_ROLES:
            c = t.roles.get(role)
            if c and c in t.df:
                vals = set(t.df[c].dropna().astype(str).unique()[:20000])
                if vals:
                    cols.append((t.name, c, vals))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            (t1, c1, v1), (t2, c2, v2) = cols[i], cols[j]
            if t1 == t2:
                continue
            ov = len(v1 & v2) / max(1, min(len(v1), len(v2)))
            if ov >= 0.3:
                g.joins.append((t1, c1, t2, c2, round(ov, 2)))
    return g


def _score(q: Counter, doc: Counter, idf: dict[str, float]) -> float:
    return sum(min(q[w], doc[w]) * idf.get(w, 1.0) * (2 if "_" in w else 1) for w in q if w in doc)


def plan_card(question: str, as_of: str, ws, rules: list[RuleRow], k_rules: int = 4, k_tables: int = 3) -> str:
    q = _grams(question)
    parts = ["# Plan card (auto-generated from the data and rulebook; verify before relying on it)"]
    cal = ws.tables["calendar"].df if "calendar" in ws.tables else None
    week = ws.rules is None or "week_monday" in ws.rules
    dates = resolve_periods(question, as_of, cal, week_monday=week)
    if dates:
        parts.append("Dates (as_of_date = " + str(as_of) + "):\n" + "\n".join(f"- {d}" for d in dates))
    if rules:
        docs = [_grams(r.key + " " + r.key + " " + r.text) for r in rules]
        df_ = Counter(w for d in docs for w in d)
        idf = {w: math.log(1 + len(docs) / n) for w, n in df_.items()}
        ranked = sorted(range(len(rules)), key=lambda i: -_score(q, docs[i], idf))
        top = [rules[i] for i in ranked[:k_rules] if _score(q, docs[i], idf) > 0]
        if top:
            parts.append("Rules most related to the question (quoted):\n" + "\n".join(f"- {r.render()[:300]}" for r in top))
    dc = debt_card(question, as_of, getattr(ws, "debt", None), debt_mentioned(ws.docs))
    if dc:
        parts.append(dc)
    else:
        try:
            mc = metric_card(question, as_of, ws)
        except Exception:  # noqa: BLE001 - advisory only
            mc = None
        if mc:
            parts.append(mc)
        else:
            try:
                ic = item_card(question, as_of, ws)
            except Exception:  # noqa: BLE001 - advisory only
                ic = None
            if ic:
                parts.append(ic)
    g = getattr(ws, "graph", None)
    if g:
        n = len(g.table_text)
        df_t = Counter(w for d in g.table_text.values() for w in d)
        idf_t = {w: math.log(1 + n / c) for w, c in df_t.items()}
        ranked_t = sorted(g.table_text, key=lambda t: -_score(q, g.table_text[t], idf_t))
        pick = [t for t in ranked_t[:k_tables] if _score(q, g.table_text[t], idf_t) > 0]
        if "orders" in ws.tables and "orders" not in pick:
            pick.append("orders")
        # tables keyed by the same order id as a picked table (payments, returns, line tables)
        oid_cols = {(t, ws.tables[t].roles.get("order_id")) for t in pick if t in ws.tables}
        for a, ca, b, cb, ov in g.joins:
            for x, cx, y in ((a, ca, b), (b, cb, a)):
                if (x, cx) in oid_cols and y not in pick and ov >= 0.5 and len(pick) < 7:
                    pick.append(y)
        if pick:
            parts.append("Likely tables: " + ", ".join(pick))
            sem = getattr(ws, "semantic", None)
            if sem is not None and semantic_on():
                # evidence-scored joins (uniqueness, coverage, fan-out) instead of raw value overlap
                base = lambda name: name.split(" (+")[0]  # noqa: E731 - table family -> first table
                js = [f"- {r.render()}" for r in sem.relations if r.verdict != "weak"
                      and base(r.left[0]) in pick and base(r.right[0]) in pick]
            else:
                js = [f"- {a}.{ca} = {b}.{cb} (value overlap {ov:.0%})" for a, ca, b, cb, ov in g.joins
                      if a in pick and b in pick]
            if js:
                parts.append("Join keys:\n" + "\n".join(js[:6]))
            cv = [f"- {t}.{c}" for t in pick for c in g.caveats.get(t, [])]
            if cv:
                parts.append("Caveats from the README:\n" + "\n".join(cv[:6]))
    return "\n".join(parts)
