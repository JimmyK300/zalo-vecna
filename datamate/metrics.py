"""Order-level metric engine (layer 3 recipe): question = metric x period(s) [x ranking].

Covers the largest question groups (revenue, order counts, average order value, cancellation
rate, change vs another period, peak day/month, revenue trend, top customer, number of buying
customers) on the derived `orders` table. Every definition is the rulebook's (§1-§3, §7, §9);
the rule-encoding parts sit behind the rules guard. `metric_intent` only fires when the question
has no extra filter it does not understand (items, groups, channels, payment methods, shifts...),
so a candidate is either computed on exactly the question's population or not offered at all.
"""
from __future__ import annotations

import calendar as _cal
import datetime as dt
import math
import re

import pandas as pd

from .primitives import infer_year, period
from .semantics import strip_accents

FILTER_WORDS = (r"nhom|kenh|cod|tien mat|chuyen khoan|vi dien tu|ghi no|size|\(m\)|\(l\)|ca sang|ca chieu|"
                r"mat hang|san pham|\bsp\b|\bmon\b|hang nao|loai|nhan vien|chi nhanh|zalo|shopee|lazada|tiktok|"
                r"online|tai quay|tai tiem|phone|dien thoai|giao hang|ship|thuoc|ke don|hoat chat|voucher|giam gia|"
                r"khuyen mai|tet|phong lanh|phi|vat|thue|gio|buoi|cuoi tuan|thu (hai|ba|tu|nam|sau|bay)|chu nhat|"
                r"tu \d|tren \d|duoi \d|it nhat|tro len|lan dau|moi|quay lai|nhap|von|lai|loi|tra hang|doi tra|"
                r"huy|lien tuc|tang dan|giam dan|trung binh moi ngay|moi ngay|ngay nao|thang nao")
Rng = tuple[dt.date, dt.date]


def _year(s: str | None) -> int | None:
    return None if not s else int(s) if len(s) == 4 else 2000 + int(s)


def find_periods(q: str, as_of) -> list[tuple[str, Rng]]:
    """Explicit and relative periods in order of appearance (accent-stripped question)."""
    a = pd.Timestamp(as_of).date()
    found: list[tuple[int, str, Rng]] = []
    rel = [(r"\bh(?:om)?\s?qua\b", "yesterday"), (r"\bhom\s?nay\b|\bhnay\b", "today"),
           (r"\btuan\s?(?:nay|nayf)\b", "this_week"), (r"\btuan\s?(?:truoc|roi)\b", "last_week"),
           (r"\bthang\w?\s?nay\b", "this_month"), (r"\bthang\w?\s?truoc\b", "last_month"),
           (r"\btu\s?dau\s?nam\b", "year_to_date"), (r"\bquy\s?nay\b", "this_quarter")]
    for pat, kind in rel:
        for m in re.finditer(pat, q):
            found.append((m.start(), kind, period(kind, a)))
    for m in re.finditer(r"(\d+)\s?ngay\s?gan\s?nhat", q):
        found.append((m.start(), "last_n_days", period("last_n_days", a, n=int(m.group(1)))))
    # a year is 4 digits, or 2 digits above 12 ("2/25" = 2/2025; "12/05" stays day/month)
    Y = r"(\d{4}|1[3-9]|[2-9]\d)(?!\d)"
    for m in re.finditer(rf"\bquy\w?\s?(\d)(?:\s?(?:/|nam)\s?{Y})?", q):
        if 1 <= int(m.group(1)) <= 4:
            found.append((m.start(), "quarter", period("quarter", a, quarter=int(m.group(1)),
                                                       year=_year(m.group(2)))))
    for m in re.finditer(rf"\b(?:thang\w?|th|t)\s?(\d{{1,2}})(?:\s?(?:/|nam|-)\s?{Y})?", q):
        if 1 <= int(m.group(1)) <= 12 and (m.group(2) or q[m.start():m.start() + 5].startswith("thang")):
            found.append((m.start(), "month", period("month", a, month=int(m.group(1)), year=_year(m.group(2)))))
    for m in re.finditer(rf"\bngay\s?(\d{{1,2}})/(\d{{1,2}})/{Y}", q):
        if not re.search(r"(den|toi)\s?het\s?$", q[:m.start()]):
            d = dt.date(_year(m.group(3)), int(m.group(2)), int(m.group(1)))
            found.append((m.start(), "day", (d, d)))
    for m in re.finditer(rf"(?<![/\d])(\d{{1,2}})/{Y}", q):  # bare "4/2026" or "4/26" after "tháng 3/2026, "
        if 1 <= int(m.group(1)) <= 12 and not any(abs(m.start() - p) < 8 for p, _, _ in found):
            found.append((m.start(), "month", period("month", a, month=int(m.group(1)), year=_year(m.group(2)))))
    found.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, k, r in found:
        if r not in seen:
            seen.add(r)
            out.append((k, r))
    return out


def previous_period(kind: str, r: Rng, q: str, as_of) -> Rng | None:
    """The comparison period (rulebook §1/§2) when only one period is named."""
    s, e = r
    if re.search(r"cung\s?(ky|thang|quy)\s?(cua\s?)?(nam\s?(truoc|ngoai))|nam\s?ngoai|so\s?(voi\s?|vs\s?)?nam\s?truoc", q):
        try:
            return s.replace(year=s.year - 1), _clamp(e.year - 1, e.month, e.day)
        except ValueError:
            return None
    if kind == "this_month":
        return period("last_month_same_days", as_of)
    if kind in ("month",):
        y, m = (s.year, s.month - 1) if s.month > 1 else (s.year - 1, 12)
        return dt.date(y, m, 1), dt.date(y, m, _cal.monthrange(y, m)[1])
    if kind in ("quarter",):
        ps = (pd.Timestamp(s) - pd.DateOffset(months=3)).date()
        pe = (pd.Timestamp(s) - pd.Timedelta(days=1)).date()
        return ps, pe
    if kind in ("this_week", "last_week", "day", "yesterday", "today", "last_n_days"):
        n = (e - s).days + 1
        return s - dt.timedelta(n if kind != "this_week" else 7), e - dt.timedelta(n if kind != "this_week" else 7)
    return None


def _clamp(y, m, d) -> dt.date:
    return dt.date(y, m, min(d, _cal.monthrange(y, m)[1]))


def shop_vocab(ws) -> list[str]:
    """Accent-stripped names that, if present in a question, mean it is filtered (items, item
    groups, customers, staff, suppliers, channels...). Cached on the workspace."""
    if getattr(ws, "_vocab", None) is not None:
        return ws._vocab
    names = set()
    for t in ws.tables.values():
        if t.shared:
            continue
        for c, m in t.meta.items():
            d = (m.get("desc") or "").lower()
            if m.get("role") in ("item", "customer") or re.match(r"^(tên|nhóm|kênh|nhà cung cấp|nhân viên|người bán)", d) \
                    or re.fullmatch(r"(ten_?(hang|sp|san_?pham|mon)|nhom.*|kenh.*|nhan_?vien.*|ncc|nha_?cung_?cap.*)", c):
                if t.df[c].dtype == object or pd.api.types.is_string_dtype(t.df[c]):
                    for v in t.df[c].dropna().astype(str).unique()[:5000]:
                        k = re.sub(r"\s+", " ", strip_accents(v)).strip()
                        if len(k) >= 5 and not re.fullmatch(r"khach le|kl|[\d\W]+", k):
                            names.add(k)
    ws._vocab = sorted(names, key=len, reverse=True)
    return ws._vocab


def normalize_question(question: str) -> str:
    """Accent-stripped, lower-cased question with common chat typos fixed."""
    q = re.sub(r"\s+", " ", strip_accents(question))
    # common chat typos: a doubled vowel or a trailing tone key ("nhaats", "ngayf", "quaan", "trc")
    q = re.sub(r"\b(\w*?)([aeiouy])\2(\w*)\b", r"\1\2\3", q)
    q = re.sub(r"\b(ngay|thang|nam|nhat|nao|sang|chieu|nhieu)[fsrxj]\b", r"\1", q)
    return re.sub(r"\btrc\b", "truoc", q).replace("nhiu", "nhieu").replace("binh quan", "trung binh")


def metric_intent(question: str, as_of, ws=None) -> dict | None:
    q = normalize_question(question)
    if re.search(r"\bno\b|cong no|so ngay|[≥≤<>]|nguoi ban|nhan vien|thu ngan|khach (nam|nu|duong|o |tai |tinh)|"
                 r"ben nao|nhom|loai khach|tru .{0,30}ra|ngoai tru|khong tinh|lai gop|lai rong|\bloi\b", q):
        return None
    if ws is not None and any(n in q for n in shop_vocab(ws)):
        return None  # the question names an item / group / customer...: not a whole-shop metric
    if re.search(r"chiem|ti trong|ty trong|(%|phan tram) (cua )?tong|bang (bao nhieu|bn|bnhiu|may) (%|phan tram)", q):
        return None  # a share of a sub-population, not a whole-shop metric
    periods = find_periods(q, as_of)
    ytd = re.search(r"\btu (ngay )?(1/1|01/01|mung 1 thang 1)\b|\bnam nay\b|\btu dau nam\b", q)
    if ytd:
        periods = [("year_to_date", period("year_to_date", as_of))] + [p for p in periods if p[0] not in ("today", "year_to_date")]
    if not periods:
        return None
    rev = re.search(r"doanh thu|dthu|\bdt\b|ban duoc (bao nhieu|bn|nhieu) tien|thu ve|tien hang|tien ban|"
                    r"ban dc (bao nhieu|bn) tien|ban duoc nhieu tien|ban dat", q)
    k1, r1 = periods[0]
    explicit = [p for p in periods if p[0] in ("month", "quarter", "day", "year_to_date", "this_month", "this_week",
                                                 "last_n_days", "yesterday", "today")]
    # trend across consecutive months
    if rev and re.search(r"tang lien tuc|luon cao hon|thang sau.{0,20}cao hon|lien tuc tang|giam lien tuc|luon thap hon", q):
        months = [r for k, r in periods if k == "month"]
        if len(months) >= 2:
            return {"kind": "trend", "periods": _fill_months(months),
                    "down": bool(re.search(r"giam lien tuc|thap hon", q))}
        return None
    if rev and re.search(r"thang nao", q) and sum(k == "month" for k, _ in periods) >= 2:
        return {"kind": "peak_month", "periods": _fill_months([r for k, r in periods if k == "month"])}
    if rev and re.search(r"ngay nao", q) and k1 in ("month", "this_month", "last_month", "this_week", "last_week"):
        return {"kind": "peak_day", "period": r1}
    top_n = re.search(r"\btop\s?(\d{1,2})\b|(?<![/\d])\b(\d{1,2}) khach", q)
    n = int(top_n.group(1) or top_n.group(2)) if top_n else None
    if re.search(r"(\bkhach\b|\bai\b).{0,40}(nhieu tien|chi tieu nhieu|chi nhieu)|mua nhieu tien nhat|chi tieu nhieu nhat", q):
        return {"kind": "top_customer_revenue", "period": r1, "n": n}
    if re.search(r"(\bkhach\b|\bai\b).{0,40}(nhieu don|nhieu lan|so don.{0,20}nhieu nhat)|mua nhieu (don|lan) nhat", q):
        return {"kind": "top_customer_orders", "period": r1, "n": n}
    rest = re.sub(r"doanh thu|tang truong|tang|giam|thay doi|bao nhieu|phan tram|so voi|cung ky|thang truoc|tuan truoc|"
                  r"nam truoc|truoc do|hon hay kem|hon kem", " ", q)
    if re.search(FILTER_WORDS + r"|tien tuoi|\btm\b|\bck\b|quet the|momo", rest):
        return None
    if re.search(r"(ti|ty) le.{0,20}huy|% don huy", q):
        return {"kind": "cancel_rate", "period": r1}
    if rev and re.search(r"%|phan tram|tang truong|tang hay giam|tang giam|thay doi|hon hay kem|tawng|giamr", q):
        asked, comp = _asked_and_compare(periods, explicit, q, as_of)
        return {"kind": "pct_change", "period": asked, "compare": comp} if comp else None
    if rev and re.search(r"chenh lech|hon (bao nhieu|bn)|kem (bao nhieu|bn)|tang (bao nhieu|bn) tien|giam (bao nhieu|bn) tien", q):
        asked, comp = _asked_and_compare(periods, explicit, q, as_of)
        return {"kind": "diff", "period": asked, "compare": comp} if comp else None
    if re.search(r"trung binh", q):
        if re.search(r"trung binh.{0,30}\b(mot|1|moi|moix|mooix)\s?don|gia tri trung binh.{0,10}don|tren (mot|1) don", q):
            return {"kind": "avg_order", "period": r1}
        return None
    if re.search(r"(bao nhieu|bn|may|so) khach.{0,20}(da )?mua|khach da mua", q):
        return {"kind": "buying_customers", "period": r1}
    if re.search(r"(bao nhieu|bn|may|so|tong so) don", q):
        return {"kind": "order_count", "period": r1}
    if rev and len(periods) == 1:
        return {"kind": "revenue", "period": r1}
    return None


def _asked_and_compare(periods, explicit, q, as_of):
    """Which period is asked and which is the comparison (rulebook §1-§2)."""
    kinds = [k for k, _ in periods]
    if "this_month" in kinds:  # "từ đầu tháng này ... so với (cùng kỳ) tháng trước" -> same days of last month
        r = dict(periods)["this_month"]
        if re.search(r"nam (truoc|ngoai)", q):
            return r, (r[0].replace(year=r[0].year - 1), _clamp(r[1].year - 1, r[1].month, r[1].day))
        return r, period("last_month_same_days", as_of)
    named = [(k, r) for k, r in periods if k in ("month", "quarter", "day", "year_to_date", "this_week", "last_n_days")]
    if len(named) >= 2:
        return named[0][1], named[1][1]
    if named:  # "so với tháng trước (đó)" next to an explicit period = the period before IT
        k, r = named[0]
        return r, previous_period(k, r, q, as_of)
    k, r = periods[0]
    if len(periods) >= 2:
        return r, periods[1][1]
    return r, previous_period(k, r, q, as_of)


def _fill_months(months: list[Rng]) -> list[Rng]:
    """'từ tháng 2/2026 đến tháng 4/2026' -> every month in between."""
    if len(months) == 2 and (months[1][0] - months[0][0]).days > 40:
        out, s = [], months[0][0]
        while s <= months[1][0]:
            out.append((s, _clamp(s.year, s.month, 31)))
            s = (pd.Timestamp(s) + pd.DateOffset(months=1)).date()
        return out
    return months


class Metrics:
    def __init__(self, orders: pd.DataFrame, as_of=None):
        self.o = orders
        self.lo, self.hi = orders["order_date"].min(), orders["order_date"].max()

    def _in(self, r: Rng) -> pd.DataFrame:
        o = self.o
        return o[(o["order_date"] >= pd.Timestamp(r[0])) & (o["order_date"] <= pd.Timestamp(r[1]))]

    def outside(self, r: Rng) -> bool:
        return pd.Timestamp(r[1]) < self.lo or pd.Timestamp(r[0]) > self.hi

    def revenue(self, r: Rng) -> int:
        o = self._in(r)
        return int(round(o.loc[o["in_revenue"], "revenue_lines"].sum()))

    def order_count(self, r: Rng) -> int:
        return int(self._in(r)["in_revenue"].sum())

    def avg_order(self, r: Rng) -> int:
        o = self._in(r)
        n = int(o["in_revenue"].sum())
        return int(math.floor(o.loc[o["in_revenue"], "revenue_lines"].sum() / n + 0.5)) if n else 0

    def cancel_rate(self, r: Rng) -> float:
        o = self._in(r)
        return round((o["status"] == "huy").sum() / len(o) * 100, 2) if len(o) else 0.0

    def pct_change(self, r: Rng, r2: Rng) -> float:
        a, b = self.revenue(r), self.revenue(r2)
        return round((a - b) / b * 100, 2) if b else float("nan")

    def daily(self, r: Rng) -> pd.Series:
        o = self._in(r)
        o = o[o["in_revenue"]]
        return o.groupby(o["order_date"].dt.normalize())["revenue_lines"].sum()

    def peak_day(self, r: Rng) -> str:
        d = self.daily(r)
        d = d[d == d.max()]
        return str(d.index.min().date())

    def peak_month(self, months: list[Rng]) -> str:
        vals = [(self.revenue(m), m[0]) for m in months]
        best = max(v for v, _ in vals)
        return next(s for v, s in vals if v == best).strftime("%Y-%m")

    def trend(self, months: list[Rng], down: bool = False) -> bool:
        v = [self.revenue(m) for m in months]
        return all((b < a) if down else (b > a) for a, b in zip(v, v[1:]))

    def _customers(self, r: Rng) -> pd.DataFrame:
        o = self._in(r)
        o = o[o["in_revenue"] & ~o.get("is_walkin", pd.Series(False, index=o.index)).fillna(False).astype(bool)]
        key = "customer_code" if "customer_code" in o and o["customer_code"].notna().any() else "customer"
        return o.assign(_key=o[key].astype(str))

    def top_customers(self, r: Rng, by: str = "revenue", n: int = 1) -> list[str]:
        """Rulebook §3/§9: rank by revenue (or order count), ties by code ascending (else name)."""
        o = self._customers(r)
        if o.empty:
            return []
        g = o.groupby("_key").agg(v=("revenue_lines", "sum") if by == "revenue" else ("order_id", "nunique"),
                                  name=("customer", "first") if "customer" in o else ("_key", "first"))
        g = g.reset_index().sort_values(["v", "_key"], ascending=[False, True], kind="mergesort")
        return [str(x) for x in g["name"].head(n)]

    def buying_customers(self, r: Rng) -> int:
        return int(self._customers(r)["_key"].nunique())


def rules_needed(question: str, kind: str | None = None) -> set[str]:
    """Rulebook rules a recipe answer to this question relies on (period conventions, walk-in).
    A recipe must stay silent when any of them is not in the current rulebook."""
    q = re.sub(r"\s+", " ", strip_accents(question))
    need = set()
    kinds = {k for k, _ in find_periods(q, "2000-01-01")}
    if kinds & {"this_week", "last_week"}:
        need.add("week_monday")
    if "this_month" in kinds and (kind in ("pct_change", "diff") or re.search(r"\bso\s?(voi|vs)?\b|\bvs\b", q)):
        need.add("month_to_date_same_days")
    if re.search(r"\b(thang\w?|quy\w?)\s?\d{1,2}\b(?!\s?(/|nam|-)\s?\d{4})", q):
        need.add("month_no_year")
    if kind and (kind.startswith("top_customer") or kind == "buying_customers"):
        need.add("walkin")
    return need


def metric_candidate(question: str, as_of, ws) -> tuple[str, object] | None:
    it = metric_intent(question, as_of, ws)
    o = ws.tables.get("orders")
    if not it or o is None or ws.rules is not None and "in_revenue" not in ws.rules:
        return None
    if ws.rules is not None and rules_needed(question, it["kind"]) - set(ws.rules):
        return None
    df = o.df
    need = {"order_date", "status", "in_revenue", "revenue_lines"}
    if not need <= set(df.columns) or df["order_date"].isna().all():
        return None
    m = Metrics(df)
    k = it["kind"]
    rngs = [it[x] for x in ("period", "compare") if x in it] + it.get("periods", [])
    if rngs and all(m.outside(r) for r in rngs[:1]):
        return ("unanswerable", None)
    if k in ("pct_change", "diff") and m.outside(it["compare"]):
        return ("unanswerable", None)
    if k == "revenue":
        return ("number", m.revenue(it["period"]))
    if k == "order_count":
        return ("number", m.order_count(it["period"]))
    if k == "avg_order":
        return ("number", m.avg_order(it["period"]))
    if k == "cancel_rate":
        return ("percent", m.cancel_rate(it["period"]))
    if k == "pct_change":
        return ("percent", m.pct_change(it["period"], it["compare"]))
    if k == "diff":
        return ("number", m.revenue(it["period"]) - m.revenue(it["compare"]))
    if k == "peak_day":
        return ("date", m.peak_day(it["period"]))
    if k == "peak_month":
        return ("date", m.peak_month(it["periods"]))
    if k == "trend":
        return ("boolean", m.trend(it["periods"], it["down"]))
    if k in ("top_customer_revenue", "top_customer_orders"):
        if "customer" not in df and "customer_code" not in df:
            return None
        names = m.top_customers(it["period"], "revenue" if k.endswith("revenue") else "orders", it.get("n") or 1)
        if not names:
            return None
        return ("list", names) if it.get("n") else ("string", names[0])
    if k == "buying_customers":
        if "customer" not in df and "customer_code" not in df:
            return None
        return ("number", m.buying_customers(it["period"]))
    return None


def metric_card(question: str, as_of, ws) -> str | None:
    """Ready-to-run call for the plan card (same computation as the candidate)."""
    it = metric_intent(question, as_of, ws)
    if not it or "orders" not in ws.tables:
        return None
    fmt = lambda r: f"('{r[0]}', '{r[1]}')"  # noqa: E731
    k = it["kind"]
    calls = {
        "revenue": f"M.revenue({fmt(it.get('period', ('', '')))})",
        "order_count": f"M.order_count({fmt(it.get('period', ('', '')))})",
        "avg_order": f"M.avg_order({fmt(it.get('period', ('', '')))})",
        "cancel_rate": f"M.cancel_rate({fmt(it.get('period', ('', '')))})",
        "peak_day": f"M.peak_day({fmt(it.get('period', ('', '')))})",
        "buying_customers": f"M.buying_customers({fmt(it.get('period', ('', '')))})",
    }
    if k in calls:
        call = calls[k]
    elif k in ("pct_change", "diff"):
        call = (f"M.pct_change({fmt(it['period'])}, {fmt(it['compare'])})" if k == "pct_change" else
                f"M.revenue({fmt(it['period'])}) - M.revenue({fmt(it['compare'])})")
    elif k in ("peak_month", "trend"):
        ms = "[" + ", ".join(fmt(r) for r in it["periods"]) + "]"
        call = f"M.peak_month({ms})" if k == "peak_month" else f"M.trend({ms}, down={it['down']})"
    elif k.startswith("top_customer"):
        call = f"M.top_customers({fmt(it['period'])}, by='{'revenue' if k.endswith('revenue') else 'orders'}', n={it.get('n') or 1})"
    else:
        return None
    return (f"Metric recipe ({k}, on the `orders` table with the rulebook definitions: revenue = in_revenue "
            f"orders' line amounts, period by order date, ties by code). Ready to run in run_python:\n  print({call})\n"
            "  Use it unless the question adds a condition the recipe does not cover.")
