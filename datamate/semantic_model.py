"""Inspectable semantic model (layer 3): what each shop's files mean, with evidence and unknowns.

Wren-style: business concepts (orders, payments, customers, items, sale lines, debt) are mapped to
the source table/column that holds them, each with the evidence for the mapping (README wording,
column name, derived table) and a confidence. Valentine-style: candidate joins are scored by
evidence, not just value overlap: key uniqueness, coverage in both directions, cardinality,
row fan-out of the join, and whether the shared values are too few to mean anything.

Two sections are kept apart on purpose:
- facts:  what the data contains (columns, joins, gaps). These do not change with the rulebook.
- rules:  which rulebook definitions the helpers applied (switched off when the wording is gone).

Nothing here decides an answer. The model reads it so it does not have to rediscover the shop,
and the "unknowns" list says what could not be determined instead of inventing a definition.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd


def semantic_on() -> bool:
    """DATAMATE_SEMANTIC=1 shows the semantic model to the model (and replaces raw-overlap joins)."""
    return os.environ.get("DATAMATE_SEMANTIC", "0") not in ("", "0", "off")

JOIN_ROLES = ("order_id", "customer_code", "item_code", "item", "customer")
DATE_NAME_RE = re.compile(r"ngay|date|thoi_gian|time", re.I)

CONCEPTS = {  # concept -> (roles that identify it, fields to show)
    "payments": (("pay_amount",), ("order_id", "pay_date", "pay_amount", "pay_method", "customer_code", "customer")),
    "customers": (("customer_code",), ("customer_code", "customer")),
    "items": (("item", "item_code"), ("item_code", "item", "item_group", "unit", "cost", "big_unit", "factor")),
}
RULE_TEXT = {
    "in_revenue": "orders.in_revenue: not cancelled and not returned (even partly); in-delivery counts",
    "walkin": "orders.is_walkin: walk-in sales are not customers",
    "dup_line": "identical lines within one order counted once",
    "order_fill": "blank cells on continuation lines repeat the order's first line",
    "debt_ledger": "debt = order total - payments up to day X; not cancelled; COD/marketplace excluded",
    "debt_movements": "new debt / repayments only for credit orders; same-day payment = paid at purchase",
    "service_fee_not_item": "service-fee line is not an item but stays in revenue",
    "best_seller_base_units": "best seller ranked by base-unit quantity",
    "gross_margin_line_cost": "gross margin = line revenue - qty x cost on the line",
    "no_status_note": "no status column: status read from the note, no mention = completed",
    "week_monday": "weeks start on Monday",
    "month_to_date_same_days": "this month vs last month uses the same number of days",
    "month_no_year": "month without a year = this year unless in the future",
}


@dataclass
class Mapping:
    field: str
    table: str
    column: str
    orig: str
    evidence: str
    confidence: str  # high (README says so) | medium (column name / derived) | low

    def render(self) -> str:
        o = f' "{self.orig}"' if self.orig and self.orig != self.column else ""
        return f"{self.field}: {self.table}.{self.column}{o} [{self.confidence}; {self.evidence}]"


@dataclass
class Relation:
    left: tuple[str, str]
    right: tuple[str, str]
    cover_l: float          # share of left values found on the right
    cover_r: float
    unique_l: bool
    unique_r: bool
    fanout: float           # rows produced per matched left row
    shared: int             # number of shared distinct values
    verdict: str            # key | probable | weak

    @property
    def cardinality(self) -> str:
        return {(True, True): "1:1", (False, True): "N:1", (True, False): "1:N"}.get((self.unique_l, self.unique_r), "N:M")

    def render(self) -> str:
        (a, ca), (b, cb) = self.left, self.right
        bits = [self.cardinality, f"{self.cover_l:.0%} of {a}.{ca} found in {b}",
                f"{self.cover_r:.0%} of {b}.{cb} found in {a}"]
        if self.fanout > 1.05:
            bits.append(f"join repeats rows ×{self.fanout:.1f}")
        return f"{a}.{ca} = {b}.{cb} [{self.verdict}: {', '.join(bits)}]"


@dataclass
class SemanticModel:
    concepts: dict[str, list[Mapping]] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    rules_on: list[str] = field(default_factory=list)
    rules_off: list[str] = field(default_factory=list)

    def render(self, max_rel: int = 8) -> str:
        out = ["## Semantic model (facts found in the files; evidence in brackets)"]
        for name, maps in self.concepts.items():
            if maps:
                out.append(f"{name}:")
                out += [f"  {m.render()}" for m in maps]
        rel = [r for r in self.relations if r.verdict != "weak"][:max_rel]
        if rel:
            out.append("relationships:")
            out += [f"  {r.render()}" for r in rel]
        if self.unknowns:
            out.append("unknowns (could not be determined; check before relying on them):")
            out += [f"  - {u}" for u in self.unknowns[:10]]
        out.append("## Rule decisions applied by helpers (from the current rulebook)")
        out += [f"  - {RULE_TEXT.get(r, r)}" for r in self.rules_on]
        if self.rules_off:
            out.append("  helpers OFF (rulebook wording changed; apply the rulebook text yourself): "
                       + ", ".join(self.rules_off))
        return "\n".join(out)


def _evidence(t, col: str) -> tuple[str, str, str]:
    m = t.meta.get(col) or {}
    desc = (m.get("desc") or "").strip()
    if desc:
        return m.get("orig", col), f'README: "{desc[:70]}"', "high"
    return m.get("orig", col), "column name only", "medium"


def _relation(t1, c1, t2, c2) -> Relation | None:
    s1, s2 = t1.df[c1].dropna().astype(str), t2.df[c2].dropna().astype(str)
    if s1.empty or s2.empty:
        return None
    v1, v2 = set(s1.unique()), set(s2.unique())
    shared = v1 & v2
    if not shared:
        return None
    cover_l, cover_r = len(shared) / len(v1), len(shared) / len(v2)
    u1, u2 = s1.is_unique, s2.is_unique
    counts_r = s2.value_counts()
    matched = s1[s1.isin(shared)]
    fanout = float(counts_r.reindex(matched).mean()) if len(matched) else 1.0
    # verdict: a key join needs one unique side and the other side mostly covered by it
    if len(shared) < 5 and max(len(v1), len(v2)) > 10:
        verdict = "weak"            # a handful of shared labels (e.g. "Khách lẻ") is coincidence
    elif (u2 and cover_l >= 0.9) or (u1 and cover_r >= 0.9):
        verdict = "key"
    elif (u1 or u2) and max(cover_l, cover_r) >= 0.5:
        verdict = "probable"        # N:M overlap means two fact tables share a dimension, not a join key
    else:
        verdict = "weak"
    return Relation((t1.name, c1), (t2.name, c2), round(cover_l, 3), round(cover_r, 3), u1, u2,
                    round(fanout, 2), len(shared), verdict)


def families(tables) -> dict[str, list]:
    """Group tables with the same column layout (monthly / yearly files of one book)."""
    fam: dict[tuple, list] = {}
    for t in tables:
        fam.setdefault(tuple(sorted(map(str, t.df.columns))), []).append(t)
    return {g[0].name: g for g in fam.values()}


def fam_name(group: list) -> str:
    return group[0].name + (f" (+{len(group) - 1} same-layout tables)" if len(group) > 1 else "")


def score_relations(tables) -> list[Relation]:
    """Compare one representative per table family (all members pooled), same key role only."""
    reps = []
    for group in families(tables).values():
        rep = group[0]
        if len(group) > 1:
            rep = type(rep)(**{**rep.__dict__, "df": pd.concat([g.df for g in group], ignore_index=True)})
            rep.name = fam_name(group)
        reps.append(rep)
    cols = [(t, role, t.roles[role]) for t in reps for role in JOIN_ROLES
            if t.roles.get(role) in getattr(t.df, "columns", [])]
    rels = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            (t1, r1, c1), (t2, r2, c2) = cols[i], cols[j]
            # only compare the same kind of key (order id with order id ...); names with codes never
            if t1.name == t2.name or r1 != r2:
                continue
            r = _relation(t1, c1, t2, c2)
            if r:
                rels.append(r)
    order = {"key": 0, "probable": 1, "weak": 2}
    return sorted(rels, key=lambda r: (order[r.verdict], -max(r.cover_l, r.cover_r)))


def build_semantic_model(ws) -> SemanticModel:
    sm = SemanticModel()
    own = [t for t in ws.tables.values() if not t.shared and not t.source.startswith("derived")]

    # orders: the derived table, with the source columns it was built from
    if "orders" in ws.tables:
        o = ws.tables["orders"].df
        srcs = [s for s in o.attrs.get("sources", []) if s in ws.tables]
        maps = []
        fam = families([ws.tables[s] for s in srcs])
        for t in (g[0] for g in list(fam.values())[:2]):
            for fld, roles in (("order_id", ("order_id",)), ("order_date", ("sale_date",)),
                               ("status", ("status", "note_status")), ("customer", ("customer", "customer_code")),
                               ("amount", ("line_amount", "order_total", "goods_amount"))):
                col = next((t.roles[r] for r in roles if t.roles.get(r) in t.df.columns), None)
                if col:
                    orig, ev, conf = _evidence(t, col)
                    maps.append(Mapping(fld, t.name, col, orig, ev, conf))
        maps.append(Mapping("one row per order", "orders", "*", "",
                            "derived from " + ", ".join(fam_name(g) for g in fam.values()) if fam else "derived", "medium"))
        sm.concepts["orders"] = maps
        if "status" in o and o["status"].nunique() <= 1:
            sm.unknowns.append("orders.status has a single value: no cancel/return information found "
                               f"(all '{o['status'].dropna().iloc[0] if o['status'].notna().any() else None}')")
    else:
        sm.unknowns.append("no order table could be built: no table has an order id + date + amount")

    for concept, (ident, fields) in CONCEPTS.items():
        maps = []
        for t in own:
            if not any(t.roles.get(r) for r in ident):
                continue
            if concept == "customers" and (t.roles.get("sale_date") or t.roles.get("pay_amount")):
                continue
            if concept == "items" and (t.roles.get("sale_date") or t.roles.get("pay_amount")):
                continue
            for fld in fields:
                col = t.roles.get(fld)
                if col in t.df.columns:
                    orig, ev, conf = _evidence(t, col)
                    maps.append(Mapping(fld, t.name, col, orig, ev, conf))
        sm.concepts[concept] = maps[:10]
    if getattr(ws, "lines", None) is not None:
        sm.concepts["sale lines"] = [Mapping("one row per sale line", "lines", "*", "",
                                             "derived; qty_base = qty x unit factor", "medium")]
    if getattr(ws, "debt", None) is not None:
        info = ws.debt.info or {}
        sm.concepts["debt"] = [Mapping(k, "debt", "-", "", str(v)[:80], "medium") for k, v in list(info.items())[:5]]

    sm.relations = score_relations(own)
    for r in sm.relations:
        if r.verdict == "key" and min(r.cover_l, r.cover_r) < 0.9 and r.left[1] and "order" in r.left[1] + r.right[1]:
            many = r.left if not r.unique_l else r.right
            cov = r.cover_l if many == r.left else r.cover_r
            if cov < 0.97:
                sm.unknowns.append(f"{1 - cov:.0%} of {many[0]}.{many[1]} ids have no match in the joined table")

    # unknowns: tables with no recognised role, sales tables with unexplained date columns
    for t in own:
        if not t.roles and len(t.df) > 0:
            sm.unknowns.append(f"table {t.name} ({t.source}): purpose not identified from README or column names")
        used = set(t.roles.values())
        dates = [c for c in t.df.columns if c not in used and (pd.api.types.is_datetime64_any_dtype(t.df[c])
                 or DATE_NAME_RE.search(c)) and not (t.meta.get(c) or {}).get("desc")]
        if dates and (t.roles.get("sale_date") or t.roles.get("pay_date")):
            sm.unknowns.append(f"{t.name}: extra date column(s) {', '.join(dates[:3])} with no README meaning")
    for e in ws.errors:
        if "not built" in e or "failed" in e.lower():
            sm.unknowns.append(e[:160])

    rules = ws.rules
    if rules is None:
        sm.rules_on = sorted(RULE_TEXT)
    else:
        sm.rules_on = sorted(r for r in rules if r in RULE_TEXT)
        sm.rules_off = sorted(set(RULE_TEXT) - set(rules) - {"no_status_note"})
    return sm
