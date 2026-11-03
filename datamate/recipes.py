"""Ready-to-run recipes (layer 3): the relationship graph fills a generic computation with this
shop's real tables and columns, so a small model does not have to assemble multi-table logic.

Debt (rulebook §4) is one ledger: per order, total owed minus payments dated up to day X.
  debt_orders(as_of)            orders still owing at the end of as_of
  debt_total(as_of)             total still owed (walk-ins included)
  debt_table(as_of, n)          top customers by debt: {"columns": ["khach_hang", "so_tien_no"], ...}
  debt_movements(customer, start, end) -> [new debt, repaid] for "ghi nợ" orders
Every rulebook filter is an argument (exclude_methods / exclude_channels), defaulting to the
current rulebook wording only when the rules guard confirms it (rules.py: debt_ledger).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .semantics import strip_accents

COD_RE = r"\bcod\b|thu ho|giao hang thu tien"
MARKET_RE = r"\bsan\b|shopee|lazada|tiktok|tiki|tmdt|marketplace|sendo"
DEBT_METHOD_RE = r"\bno\b|ghi no|cong no|debt|tra sau|credit"


@dataclass
class DebtLedger:
    orders: pd.DataFrame                      # order_id, order_date, customer, customer_key, is_walkin, status, total, method, channel
    payments: pd.DataFrame                    # order_id, date, amount
    info: dict = field(default_factory=dict)  # where each piece came from (shown in the plan card)
    rules_ok: bool = True

    def _filtered(self, as_of, exclude_methods, exclude_channels) -> pd.DataFrame:
        o = self.orders
        o = o[o["order_date"] <= pd.Timestamp(as_of)]
        o = o[o["status"].fillna("") != "huy"]
        if exclude_methods is None:
            exclude_methods = COD_RE if self.rules_ok else None
        if exclude_channels is None:
            exclude_channels = MARKET_RE if self.rules_ok else None
        if exclude_methods and "method" in o:
            o = o[~o["method"].fillna("").map(strip_accents).str.contains(exclude_methods, regex=True)]
        if exclude_channels and "channel" in o:
            o = o[~o["channel"].fillna("").map(strip_accents).str.contains(exclude_channels, regex=True)]
        return o

    def debt_orders(self, as_of, exclude_methods=None, exclude_channels=None,
                    older_than_days: int | None = None) -> pd.DataFrame:
        """Orders still owing at the end of as_of, with age_days = as_of - order date.
        older_than_days=N keeps orders placed MORE than N days before as_of ("quá N ngày")."""
        o = self._filtered(as_of, exclude_methods, exclude_channels).copy()
        end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
        p = self.payments[self.payments["date"] < end]
        paid = p.groupby("order_id")["amount"].sum()
        o["paid"] = o["order_id"].map(paid).fillna(0)
        o["owed"] = o["total"] - o["paid"]
        o["age_days"] = (pd.Timestamp(as_of).normalize() - o["order_date"].dt.normalize()).dt.days
        o = o[o["owed"] > 0.5]
        return o[o["age_days"] > older_than_days] if older_than_days is not None else o

    def debt_total(self, as_of, **kw) -> int:
        return int(round(self.debt_orders(as_of, **kw)["owed"].sum()))

    def debt_customers(self, as_of, **kw) -> int:
        """Number of customers (walk-ins excluded, rulebook §3) still owing at the end of as_of."""
        d = self.debt_orders(as_of, **kw)
        return int(d.loc[~d["is_walkin"].fillna(False).astype(bool), "customer_key"].nunique())

    def debt_by_age(self, as_of, buckets: list[tuple[int, int | None]], **kw) -> list[int]:
        """Amount owed per debt-age bucket, e.g. [(0, 7), (8, 15), (16, None)] (days, inclusive)."""
        d = self.debt_orders(as_of, **kw)
        out = []
        for lo, hi in buckets:
            m = (d["age_days"] >= lo) & ((d["age_days"] <= hi) if hi is not None else True)
            out.append(int(round(d.loc[m, "owed"].sum())))
        return out

    def debt_table(self, as_of, n: int | None = None, **kw) -> dict:
        d = self.debt_orders(as_of, **kw)
        d = d[~d["is_walkin"].fillna(False).astype(bool)]
        g = d.groupby("customer_key").agg(khach_hang=("customer", "first"), so_tien_no=("owed", "sum")).reset_index()
        g["so_tien_no"] = g["so_tien_no"].round().astype(int)
        g = g.sort_values(["so_tien_no", "customer_key"], ascending=[False, True], kind="mergesort")
        if n:
            g = g.head(n)
        return {"columns": ["khach_hang", "so_tien_no"], "rows": g[["khach_hang", "so_tien_no"]].values.tolist()}

    def debt_movements(self, customer: str | None, start, end, **kw) -> list[int]:
        """[nợ thêm, trả nợ] for orders recorded as debt, of one customer (name or code) or of
        the whole shop (customer=None)."""
        o = self.orders[self.orders["status"].fillna("") != "huy"]
        if "method" in o:
            o = o[o["method"].fillna("").map(strip_accents).str.contains(DEBT_METHOD_RE, regex=True)]
        if customer is not None:
            key = _key(customer)
            # rulebook §3: with customer codes the code is the customer, whatever spelling the name has
            keys = set(self.orders.loc[self.orders["customer"].map(_key) == key, "customer_key"]) | {key, str(customer)}
            o = o[o["customer_key"].isin(keys)]
        s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        p = self.payments.merge(o[["order_id", "order_date", "total"]], on="order_id")
        same_day = p[p["date"].dt.normalize() == p["order_date"]].groupby("order_id")["amount"].sum()
        placed = o[(o["order_date"] >= s) & (o["order_date"] < e)]
        new = (placed["total"] - placed["order_id"].map(same_day).fillna(0)).clip(lower=0).sum()
        later = p[(p["date"].dt.normalize() > p["order_date"]) & (p["date"] >= s) & (p["date"] < e)]
        return [int(round(new)), int(round(later["amount"].sum()))]


def _key(v) -> str:
    return re.sub(r"\s+", " ", str(v).strip().casefold())


def build_debt_ledger(ws) -> DebtLedger | None:
    """Assemble the ledger from the graph's facts; None when the shop does not record debt."""
    if "orders" not in ws.tables:
        return None
    o = ws.tables["orders"].df
    src = [t for t in ws.tables.values() if not t.shared and not t.name.startswith("orders")
           and t.roles.get("order_id") and t.name in (ws.tables["orders"].source or "")]
    head = src[0] if src else None
    hr = head.roles if head else {}
    info = {}
    total = hr.get("order_total")
    if not total or total not in o:
        return None
    L = pd.DataFrame({"order_id": o["order_id"].astype(str), "order_date": pd.to_datetime(o.get("order_date")),
                      "status": o.get("status"), "total": pd.to_numeric(o[total], errors="coerce").fillna(0)})
    info["orders"] = f"{head.name}.{total} (order total) per {hr.get('order_id')}"
    # customer identity: code if the shop has codes (rulebook §3), else the name as written
    name = o["customer"] if "customer" in o else None
    code = o["customer_code"] if "customer_code" in o else None
    if code is not None and name is None:
        for t in ws.tables.values():
            if t is head or t.shared or not t.roles.get("customer_code"):
                continue
            nm = t.roles.get("customer") or next((c for c, m in t.meta.items() if re.match(r"^tên\b", m.get("desc", ""))
                                                  or re.fullmatch(r"ten_?goi|ho_?ten|ten|ten_?khach(_?hang)?", c)), None)
            if nm:
                mp = dict(zip(t.df[t.roles["customer_code"]].astype(str), t.df[nm]))
                name = code.astype(str).map(mp).fillna(code.astype(str))
                info["names"] = f"{t.name}.{nm} by {t.roles['customer_code']}"
                break
    L["customer"] = (name if name is not None else code).values if (name is not None or code is not None) else None
    L["customer_key"] = (code.astype(str) if code is not None else L["customer"].map(_key)).values
    L["is_walkin"] = o["is_walkin"].values if "is_walkin" in o else (L["customer"].isna()).values
    if code is not None:
        L["is_walkin"] = L["is_walkin"] | code.isna().values
    for role, std in (("pay_method", "method"), ("channel", "channel")):
        c = hr.get(role)
        if c and c in o:
            L[std] = o[c].astype(str).values
            info[std] = f"{head.name}.{c}: " + ", ".join(sorted(map(str, o[c].dropna().unique()))[:8])
    # payments: a table keyed by the same order ids with a payment amount (+ date)
    P = None
    ids = set(L["order_id"])
    for t in ws.tables.values():
        if t is head or t.shared or t.name.startswith("orders"):
            continue
        r = t.roles
        if not (r.get("order_id") and r.get("pay_amount")):
            continue
        if t.df[r["order_id"]].astype(str).isin(ids).mean() < 0.3:
            continue
        dcol = r.get("pay_date")
        dcol = f"{dcol}__date" if dcol and f"{dcol}__date" in t.df else dcol
        if not dcol:
            dcol = next((c for c in t.df.columns if c.endswith("__date")), None)
        if not dcol:
            continue
        P = pd.DataFrame({"order_id": t.df[r["order_id"]].astype(str), "date": pd.to_datetime(t.df[dcol], errors="coerce"),
                          "amount": pd.to_numeric(t.df[r["pay_amount"]], errors="coerce").fillna(0)})
        info["payments"] = f"{t.name}.{r['pay_amount']} dated {dcol}, by {r['order_id']}"
        pm = r.get("pay_method")
        if pm and "method" not in L:  # method recorded per payment: an order is COD if its payment says so
            first = pd.Series(t.df[pm].astype(str).values, index=t.df[r["order_id"]].astype(str)).groupby(level=0).first()
            L["method"] = L["order_id"].map(first).fillna("")
            info["method"] = f"{t.name}.{pm} (per payment): " + ", ".join(sorted(map(str, t.df[pm].dropna().unique()))[:8])
        break
    if P is None:
        paid = hr.get("paid")
        if not paid or paid not in o:
            return None
        P = pd.DataFrame({"order_id": L["order_id"], "date": L["order_date"],
                          "amount": pd.to_numeric(o[paid], errors="coerce").fillna(0).values})
        info["payments"] = f"{head.name}.{paid} (paid at purchase; no payment ledger found)"
    rules_ok = ws.rules is None or "debt_ledger" in ws.rules
    return DebtLedger(L, P, info, rules_ok)


def debt_intent(question: str, as_of: str) -> dict | None:
    """Recognise a debt question and its parameters (accent-insensitive, typo tolerant).
    kind: table (top debtors) / total / movements; None if not a debt question."""
    s = re.sub(r"\s+", " ", strip_accents(question))
    if not re.search(r"\bno\b|cong no|\bchiu\b", s):
        return None
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    day = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else str(as_of)
    if re.search(r"no them|tra no|thu no|tra duoc|tra bot|mua chiu", s):
        return {"kind": "movements", "both": bool(re.search(r"no them|chiu them", s) and re.search(r"tra", s)),
                "new_only": bool(re.search(r"no them|chiu them", s)) and not re.search(r"tra", s)}
    if re.search(r"tuoi no|nhom tuoi", s) and re.search(r"\d+\s?[-–]\s?\d+", s):
        b = [(int(a), int(c)) for a, c in re.findall(r"(\d+)\s?[-–]\s?(\d+)", s)]
        top = re.search(r"(?:tren|qua|hon)\s?(\d+)", s)
        if top:
            b.append((int(top.group(1)) + 1, None))
        return {"kind": "age_buckets", "as_of": day, "buckets": b}
    if re.search(r"(bao nhieu|bnhiu|\bbn\b|may|so) khach", s) and not re.search(r"\bbang\b|\btop\b|khach nao", s):
        return {"kind": "customers", "as_of": day}
    if re.search(r"\bbang\b|\btop\b|khach nao|\bai\b|nhung khach|liet ke", s):
        n = re.search(r"(?:top\s?|\b)(\d+)\s?khach", s)
        return {"kind": "table", "as_of": day, "n": int(n.group(1)) if n else None}
    if re.search(r"tong|bao nhieu|bnhiu|\bbn\b|nhieu", s) and re.search(r"con no|dang no|chua tra", s):
        age = re.search(r"(?:qua|hon|tren)\s?(\d+)\s?ngay", s)
        return {"kind": "total", "as_of": day, "older_than_days": int(age.group(1)) if age else None}
    return None


def debt_candidate(question: str, as_of: str, ledger: "DebtLedger | None",
                   debt_mentioned: bool = False) -> tuple[str, object] | None:
    """Deterministic answer for clearly recognised debt questions (used as a second candidate).
    No ledger: 'unanswerable' only if the shop's docs never mention debt; otherwise no candidate
    (the shop records debt in a layout the recipe does not know)."""
    it = debt_intent(question, as_of)
    if not it or it["kind"] == "movements":
        return None
    if ledger is None:
        return None if debt_mentioned else ("unanswerable", None)
    if not ledger.rules_ok:
        return None  # the rulebook's debt definition changed: let the model apply the new text
    if it["kind"] == "table":
        return ("table", ledger.debt_table(it["as_of"], it["n"]))
    if it["kind"] == "customers":
        return ("number", ledger.debt_customers(it["as_of"]))
    if it["kind"] == "age_buckets":
        return ("list", ledger.debt_by_age(it["as_of"], it["buckets"]))
    return ("number", ledger.debt_total(it["as_of"], older_than_days=it["older_than_days"]))


def debt_mentioned(docs: dict[str, str]) -> bool:
    return any(re.search(r"\bnợ\b|công nợ|ghi nợ|trả góp", t, re.I) for t in docs.values())


def debt_card(question: str, as_of: str, ledger: "DebtLedger | None", mentioned: bool = False) -> str | None:
    it = debt_intent(question, as_of)
    if not it:
        return None
    if ledger is None and mentioned:
        return ("Debt recipe: no standard order-total + payment ledger was recognised, but the README talks "
                "about debt: read how this shop records it (e.g. a separate debt book) and compute it by the "
                "rulebook §4 definitions yourself.")
    if ledger is None:
        return ("Debt recipe: this shop's files have no order totals with payments and the README never "
                "mentions debt: the question is most likely unanswerable (rulebook §7).")
    src = "; ".join(f"{k}: {v}" for k, v in ledger.info.items())
    lines = [f"Debt recipe (ledger built from {src}). Ready to run in run_python:"]
    if it["kind"] == "table":
        lines.append(f"  print(debt.debt_table('{it['as_of']}', n={it['n']}))   # rows: customer, amount owed")
    elif it["kind"] == "customers":
        lines.append(f"  print(debt.debt_customers('{it['as_of']}'))   # customers (not walk-ins) still owing")
    elif it["kind"] == "age_buckets":
        lines.append(f"  print(debt.debt_by_age('{it['as_of']}', {it['buckets']}))   # amount owed per age bucket (days)")
    elif it["kind"] == "total":
        lines.append(f"  print(debt.debt_total('{it['as_of']}', older_than_days={it['older_than_days']}))")
    else:
        lines.append("  print(debt.debt_movements('<customer name or code, or None for the whole shop>', '<start>', '<end>'))"
                     "   # -> [new debt, repaid] for 'ghi nợ' orders; dates from the plan card")
    lines.append("  Filters follow the rulebook (cancelled orders out, COD/marketplace out); override with "
                 "exclude_methods= / exclude_channels= regexes if the rulebook says otherwise. "
                 "debt.debt_orders(day) shows the per-order balances.")
    return "\n".join(lines)
