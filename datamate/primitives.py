"""Deterministic, rule-free building blocks (layer 2). The model chooses inputs and filters from
the rulebook; these compute. Exposed in run_python and (orders) as a DuckDB table.

  period(kind, as_of, ...)            -> (start, end) dates, inclusive, per rulebook §2
  resolve_periods(question, as_of)    -> date ranges for every time phrase in the question
  build_orders(ws)                    -> one row per order (standard column names) from line tables
  as_of_balance(plus, minus, ...)     -> per-key balance at the end of a day (debt, stock, payables...)
  pct_change(curr, prev)              -> percent change, 2 decimals
  rank(df, value, n, code, name)      -> sort with the tie-break by code, then name
"""
from __future__ import annotations

import calendar as _cal
import datetime as dt
import re

import pandas as pd

from .semantics import strip_accents

D = dt.date


def _d(x) -> D:
    return pd.Timestamp(x).date()


def _month_end(y: int, m: int) -> D:
    return D(y, m, _cal.monthrange(y, m)[1])


def infer_year(month: int, as_of: D, quarter: bool = False) -> int:
    """Rulebook §2: a month/quarter without a year is this year unless that is after today."""
    first = D(as_of.year, (month - 1) * 3 + 1 if quarter else month, 1)
    return as_of.year if first <= as_of else as_of.year - 1


def period(kind: str, as_of, n: int | None = None, month: int | None = None, quarter: int | None = None,
           year: int | None = None, cal: pd.DataFrame | None = None) -> tuple[D, D]:
    """Inclusive date range for a time phrase, relative to as_of (rulebook §2 definitions)."""
    a = _d(as_of)
    if kind == "today":
        return a, a
    if kind == "yesterday":
        return a - dt.timedelta(1), a - dt.timedelta(1)
    if kind == "this_week":            # Monday of this week .. today
        return a - dt.timedelta(a.weekday()), a
    if kind == "last_week":            # the whole previous Monday..Sunday
        mon = a - dt.timedelta(a.weekday() + 7)
        return mon, mon + dt.timedelta(6)
    if kind == "this_month":           # 1st .. today
        return a.replace(day=1), a
    if kind == "last_month_same_days":  # 1st .. same day number of last month (or its last day)
        y, m = (a.year, a.month - 1) if a.month > 1 else (a.year - 1, 12)
        return D(y, m, 1), D(y, m, min(a.day, _cal.monthrange(y, m)[1]))
    if kind == "last_month":
        y, m = (a.year, a.month - 1) if a.month > 1 else (a.year - 1, 12)
        return D(y, m, 1), _month_end(y, m)
    if kind == "year_to_date":
        return D(a.year, 1, 1), a
    if kind == "last_n_days":          # N consecutive days including today
        return a - dt.timedelta((n or 1) - 1), a
    if kind == "month":
        y = year or infer_year(month, a)
        return D(y, month, 1), _month_end(y, month)
    if kind == "quarter":
        y = year or infer_year(quarter, a, quarter=True)
        m0 = (quarter - 1) * 3 + 1
        return D(y, m0, 1), _month_end(y, m0 + 2)
    if kind == "this_quarter":
        q = (a.month - 1) // 3 + 1
        return D(a.year, (q - 1) * 3 + 1, 1), a
    if kind == "year":
        return D(year, 1, 1), D(year, 12, 31)
    if kind == "until":                # stock / debt "đến hết ngày X": from the start of the data
        return D(1900, 1, 1), a
    if kind == "tet":                  # 23 tháng Chạp (lunar Y-1) .. mùng 10 tháng Giêng (lunar Y)
        if cal is None:
            raise ValueError("tet needs the calendar table")
        c = cal.copy()
        c["date"] = pd.to_datetime(c["date"]).dt.date
        leap = c["lunar_leap"].astype(str).str.lower().isin(["1", "true", "yes"])
        start = c[(c.lunar_year == year - 1) & (c.lunar_month == 12) & (c.lunar_day == 23) & ~leap]["date"]
        end = c[(c.lunar_year == year) & (c.lunar_month == 1) & (c.lunar_day == 10) & ~leap]["date"]
        if start.empty or end.empty:
            raise ValueError(f"Tết {year} not in calendar")
        return start.iloc[0], end.iloc[0]
    raise ValueError(f"unknown period kind {kind}")


_PATTERNS = [  # (regex on accent-stripped question, kind) — tolerant to chat typos ("hqua", "thangs")
    (r"\bh(?:om)?\s?qua\b", "yesterday"),
    (r"\bhom\s?nay\b|\bhnay\b", "today"),
    (r"\btuan\s?(?:nay|nayf)\b", "this_week"),
    (r"\btuan\s?(?:truoc|roi)\b", "last_week"),
    (r"\bthang\w?\s?nay\b", "this_month"),
    (r"\bthang\w?\s?truoc\b", "last_month"),
    (r"\btu\s?dau\s?nam\b|\bdau\s?nam\s?(?:toi|den)\s?nay\b", "year_to_date"),
    (r"\bquy\s?nay\b", "this_quarter"),
]


def resolve_periods(question: str, as_of, cal: pd.DataFrame | None = None, week_monday: bool = True) -> list[str]:
    """Human-readable date ranges for the time phrases found in a question (for the plan card)."""
    q = re.sub(r"\s+", " ", strip_accents(question))
    a = _d(as_of)
    out = []

    def add(label: str, rng: tuple[D, D]) -> None:
        s = f"{label}: {rng[0]} .. {rng[1]}" if rng[0] != rng[1] else f"{label}: {rng[0]}"
        if s not in out:
            out.append(s)

    for pat, kind in _PATTERNS:
        if re.search(pat, q):
            if kind in ("this_week", "last_week") and not week_monday:
                out.append(f"'{kind}': the rulebook's week definition changed; derive it from §2")
                continue
            add(kind.replace("_", " "), period(kind, a))
            if kind == "this_month" and re.search(r"cung\s?ky|so voi|so vs|thang truoc|th truoc", q):
                add("same days of last month (for comparison)", period("last_month_same_days", a))
    m = re.search(r"(\d+)\s?ngay\s?gan\s?nhat", q)
    if m:
        add(f"last {m.group(1)} days", period("last_n_days", a, n=int(m.group(1))))
    for m in re.finditer(r"\bquy\s?(\d)(?:\s?(?:/|nam)\s?(\d{4}))?", q):
        qq, y = int(m.group(1)), int(m.group(2)) if m.group(2) else None
        if 1 <= qq <= 4:
            add(f"quý {qq}" + (f"/{y}" if y else f" (no year → {infer_year(qq, a, True)})"), period("quarter", a, quarter=qq, year=y))
    for m in re.finditer(r"\bthang\w?\s?(\d{1,2})(?:\s?(?:/|nam|-)\s?(\d{4}))?", q):
        mm, y = int(m.group(1)), int(m.group(2)) if m.group(2) else None
        if 1 <= mm <= 12:
            add(f"tháng {mm}" + (f"/{y}" if y else f" (no year → {infer_year(mm, a)})"), period("month", a, month=mm, year=y))
            if re.search(r"cung\s?ky\s?(?:nam\s?(?:truoc|ngoai)|nam\s?ngoai)", q):
                yy = (y or infer_year(mm, a)) - 1
                add(f"tháng {mm}/{yy} (same period last year)", period("month", a, month=mm, year=yy))
    for m in re.finditer(r"\bnam\s?(20\d\d)\b", q):
        add(f"năm {m.group(1)}", period("year", a, year=int(m.group(1))))
    if re.search(r"\btet\b", q) and cal is not None:
        m = re.search(r"\btet\s?(?:nam\s?)?(20\d\d)", q)
        y = int(m.group(1)) if m else a.year
        try:
            add(f"Tết {y}", period("tet", a, year=y, cal=cal))
        except ValueError:
            pass
    for m in re.finditer(r"(?:den|toi)\s?het\s?(?:ngay\s?)?(\d{1,2})/(\d{1,2})/(\d{4})", q):
        x = D(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        out.append(f"'đến hết ngày {x}': a balance (còn nợ/còn tồn) using every record from the start of the "
                   f"data up to and including {x}; NOT a monthly period")
    return out


def _first_nonnull(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else None


def build_orders(ws) -> dict[str, pd.DataFrame]:
    """One row per order with standard columns, from the shop's sales tables:
    order_id, order_date, customer, customer_code, is_walkin, status (order_status), in_revenue,
    revenue_lines (sum of line amounts, duplicate lines counted once), n_lines, plus every column
    that is constant within an order (payment method, channel, order total, discount...)."""
    tabs = [t for t in ws.tables.values() if not t.shared and t.roles.get("order_id")]
    heads = [t for t in tabs if {"sale_date", "status", "note_status"} & set(t.roles) and not t.roles.get("pay_amount")]
    if not heads:  # rulebook §8: a sales book without order ids -> every row is its own record
        heads = [t for t in ws.tables.values() if not t.shared and t.roles.get("sale_date")
                 and (t.roles.get("line_amount") or t.roles.get("order_total")) and "order_status" in t.df]
    groups: dict[tuple, list] = {}
    for t in heads:
        groups.setdefault(tuple(t.df.columns), []).append(t)
    out = {}
    for i, grp in enumerate(groups.values()):
        r = dict(grp[0].roles)
        df = pd.concat([t.df for t in grp], ignore_index=True)
        if not r.get("order_id"):
            df["row_id"] = [f"row{k + 1}" for k in range(len(df))]
            r["order_id"] = "row_id"
        oid = r["order_id"]
        if "is_dup_line" in df:
            df = df[~df["is_dup_line"]]
        df = df[df[oid].notna()]
        g = df.groupby(oid, sort=False)
        nun = g.nunique(dropna=True)
        const = [c for c in df.columns if c != oid and (nun[c] <= 1).mean() >= 0.99]
        o = g[const].first().reset_index() if const else g.size().reset_index()[[oid]]
        o["n_lines"] = g.size().values
        # line amounts: this table, or EVERY detail table sharing the order ids (detail books are
        # often split by period: one file per year)
        la = r.get("line_amount")
        rev = None
        if la and pd.api.types.is_numeric_dtype(df[la]):
            rev = df.groupby(df[oid].astype(str))[la].sum()
        elif not la:
            ids = set(df[oid].astype(str))
            parts = []
            for t in tabs:
                if t in grp or not t.roles.get("line_amount"):
                    continue
                to, tl = t.roles["order_id"], t.roles["line_amount"]
                if t.df[to].astype(str).isin(ids).mean() >= 0.5 and pd.api.types.is_numeric_dtype(t.df[tl]):
                    d = t.df[~t.df["is_dup_line"]] if "is_dup_line" in t.df else t.df
                    parts.append(d.groupby(d[to].astype(str))[tl].sum())
            if parts:
                rev = pd.concat(parts).groupby(level=0).sum()
        if rev is not None:
            o["revenue_lines"] = o[oid].astype(str).map(rev)
        elif r.get("goods_amount") and r["goods_amount"] in o:  # order-level "tiền hàng" = sum of line amounts
            o["revenue_lines"] = pd.to_numeric(o[r["goods_amount"]], errors="coerce")
        elif r.get("order_total") and r["order_total"] in o and not r.get("item") and not any(
                t.roles.get("line_amount") and (t.roles.get("item") or t.roles.get("item_code"))
                for t in ws.tables.values() if not t.shared):
            # one row per ticket and only a ticket total (e.g. a repair book): the total is the revenue
            o["revenue_lines"] = pd.to_numeric(o[r["order_total"]], errors="coerce")
        std = {"order_id": oid, "order_date": r.get("sale_date_parsed") or r.get("sale_date"),
               "customer": r.get("customer"), "customer_code": r.get("customer_code"),
               "status": "order_status" if "order_status" in o else None}
        for name, col in std.items():
            if col and col in o and name not in o:
                o[name] = o[col]
        if "customer" not in o and "customer_code" in o:  # names from a customer list, by code
            for t in ws.tables.values():
                if t.shared or t in grp or not t.roles.get("customer_code"):
                    continue
                nm = t.roles.get("customer") or next(
                    (c for c, m in t.meta.items() if re.match(r"^tên\b", m.get("desc", ""))
                     or re.fullmatch(r"ten_?goi|ho_?ten|ten|ten_?khach(_?hang)?", c)), None)
                if nm:
                    mp = dict(zip(t.df[t.roles["customer_code"]].astype(str), t.df[nm]))
                    code = o["customer_code"].astype(str)
                    o["customer"] = code.map(mp).where(o["customer_code"].notna())
                    break
        walk = [c for c in o.columns if c.endswith("_is_walkin")]
        if walk and "is_walkin" not in o:
            o["is_walkin"] = o[walk[0]]
        if "order_date" in o:
            o["order_date"] = pd.to_datetime(o["order_date"], errors="coerce").dt.normalize()
        front = [c for c in ("order_id", "order_date", "customer", "customer_code", "is_walkin", "status",
                             "in_revenue", "revenue_lines", "n_lines") if c in o]
        o = o[front + [c for c in o.columns if c not in front]]
        out["orders" if i == 0 else f"orders_{i + 1}"] = o
        o.attrs["sources"] = [t.name for t in grp] + ([f"line amounts from detail table"] if not r.get("line_amount") and "revenue_lines" in o else [])
    return out


def as_of_balance(plus: pd.DataFrame, key: str, amount: str, date: str, as_of,
                  minus: pd.DataFrame | None = None, minus_key: str | None = None, minus_amount: str | None = None,
                  minus_date: str | None = None, positive_only: bool = True) -> pd.DataFrame:
    """Balance per key at the END of day as_of: sum(plus.amount with date <= as_of)
    - sum(minus.amount with date <= as_of). Debt: plus = qualifying orders (order total),
    minus = payments, key = order id (then group the positive rows by customer). Stock:
    plus = receipts, minus = sales (base units), key = item. Filter plus/minus BEFORE calling."""
    end = pd.Timestamp(_d(as_of)) + pd.Timedelta(days=1)
    p = plus[pd.to_datetime(plus[date], errors="coerce") < end]
    res = p.groupby(p[key].astype(str).rename(key))[amount].sum().rename("plus").to_frame()
    if minus is not None:
        mk, ma, md = minus_key or key, minus_amount or amount, minus_date or date
        m = minus[pd.to_datetime(minus[md], errors="coerce") < end]
        res = res.join(m.groupby(m[mk].astype(str).rename(key))[ma].sum().rename("minus"), how="left")
    else:
        res["minus"] = 0.0
    res["minus"] = res["minus"].fillna(0)
    res["balance"] = res["plus"] - res["minus"]
    res = res.reset_index()
    return res[res["balance"] > 0.5] if positive_only else res


def pct_change(curr: float, prev: float) -> float:
    """Rulebook §1: change versus the previous/compared period, negative when it fell, 2 decimals."""
    return round((curr - prev) / prev * 100, 2)


def rank(df: pd.DataFrame, value: str, n: int | None = None, code: str | None = None,
         name: str | None = None, ascending: bool = False) -> pd.DataFrame:
    """Rulebook ordering: by value, ties by code ascending (else by name)."""
    keys, asc = [value], [ascending]
    for k in (code, name):
        if k:
            keys.append(k)
            asc.append(True)
    out = df.sort_values(keys, ascending=asc, kind="mergesort")
    return out.head(n) if n else out
