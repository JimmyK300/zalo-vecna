"""Filtered metrics (layer 3 recipe): revenue / order count / units / average order restricted to one
value of a shop dimension, e.g. "doanh thu các đơn trả bằng chuyển khoản tháng 5", "ca chiều so với
ca sáng", "nhóm mì gói chiếm bao nhiêu %", "bao nhiêu ly size L".

Dimensions are discovered per shop from the data, not hard-coded per shop:
- order level: payment method, sales channel, staff, customer group, shift (from the order time);
- line level: item group, size M/L (item name suffix).
A question is matched to a dimension value only by that value appearing in it (plus a few generic
aliases such as "tm" = tiền mặt). Rule-encoding parts (shift hours, size suffix, exact group name,
best salesperson by revenue) sit behind the rules guard (rules.py). Anything the parser does not
understand returns None, so the model handles it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .metrics import Metrics, find_periods, normalize_question, period, rules_needed, shop_vocab
from .semantics import strip_accents

# (no bare "vi" -> "vi dien tu": accent-stripped "vỉ" (blister pack) reads the same)
ALIASES = [(r"\btm\b", "tien mat"), (r"\bck\b", "chuyen khoan"), (r"\bddown\b", "don"), (r"\bbill\b", "don")]
# coded values (English / short codes in POS exports) -> the Vietnamese words questions use
SYNONYMS = [(r"^(cash|tien mat|tm)$", ["tien mat"]), (r"^(transfer|bank|bank_?transfer|chuyen khoan|ck)$", ["chuyen khoan"]),
            (r"^(e-?wallet|ewallet|wallet|momo|vi dien tu)$", ["vi dien tu"]), (r"^(cod|thu ho)$", ["cod", "thu ho"]),
            (r"^(debt|credit|ghi no)$", ["ghi no"]), (r"^(card|the)$", ["the", "quet the"]),
            (r"^(offline|store|in_?store|pos|counter|tai quay)$", ["tai quay"]), (r"^(phone|dien thoai)$", ["dien thoai"])]
# conditions this recipe does not handle: leave the question to the model
UNHANDLED = re.compile(r"\b(no|cong no|tra hang|huy|tru|khong tinh|ngoai tru|tren \d|duoi \d|it nhat|tro len|"
                       r"tu \d+ (don|lan)|ngay nao|thang nao|moi ngay|lien tuc|lai|loi nhuan|von|nhap|ton|gia|"
                       r"khuyen mai|lan dau|quay lai|khach moi|top \d|\d+ (mat hang|mon)|mat hang nao|mon nao|"
                       r"san pham nao|khach nao|ai mua|tet|cuoi tuan|thu (hai|ba|tu|nam|sau|bay)|chu nhat|"
                       r"gio cao diem|khung gio|gio|buoi)\b")   # whole words: "ton" must not match "tong"
SHARE = re.compile(r"chiem|ti trong|ty trong|bao nhieu (%|phan tram)|bn (%|phan tram)|may (%|phan tram)")
COMPARE = re.compile(r"so voi|\bso\b|\bvs\b|chenh|hon|kem")


def _norm(v) -> str:
    return re.sub(r"[\s_]+", " ", strip_accents(str(v))).strip()   # "zalo_oa" reads as "zalo oa"


@dataclass
class Dim:
    name: str
    level: str                                   # "order" | "line"
    values: dict[str, str] = field(default_factory=dict)   # normalised text -> raw value
    series: pd.Series | None = None              # order_id -> value (order) / line index -> value (line)


def _heads(ws) -> list:
    o = ws.tables["orders"].df
    return [t for t in (ws.tables.get(s) for s in o.attrs.get("sources", [])) if t is not None and "order_id" in t.roles]


def _per_order(parts: list[pd.Series]) -> pd.Series | None:
    """Combine the per-order values of every source table (yearly / monthly files), first value wins."""
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return None
    s = pd.concat(parts)
    return s[~s.index.duplicated()]


def _order_col(ws, roles: tuple[str, ...], name_re: str | None = None) -> pd.Series | None:
    """One value per order from the head-table column with one of `roles` (or a name match)."""
    parts = []
    for t in _heads(ws):
        col = next((t.roles[r] for r in roles if t.roles.get(r) in t.df.columns), None)
        if col is None and name_re:
            col = next((c for c in t.df.columns if re.search(name_re, c)), None)
        if col is not None:
            s = t.df[[t.roles["order_id"], col]].dropna()
            parts.append(s.drop_duplicates(t.roles["order_id"]).set_index(t.roles["order_id"])[col].astype(str))
    return _per_order(parts)


def _shift_series(ws) -> pd.Series | None:
    """Order -> 'sang' / 'chieu' by the order's creation time (rulebook §2), when a time exists."""
    return _per_order([_shift_one(t) for t in _heads(ws)])


def _created(t, c: str) -> bool:
    d = (t.meta.get(c) or {}).get("desc", "") or ""
    return bool(re.search(r"tao|dat_don|created", c) or re.search(r"tạo|đặt đơn|đặt hàng", d))


def _shift_one(t) -> pd.Series | None:
    # creation time first ("thời gian tạo", "giờ đặt đơn"), then the sale time / date columns
    cands = [c for c in t.df.columns if _created(t, c)] + [t.roles.get("sale_time"), t.roles.get("sale_date")] \
        + list(t.df.columns)
    for c in dict.fromkeys(x for x in cands if x and x in t.df.columns):
        s = t.df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            h = s.dt.hour + s.dt.minute / 60
            if (h.dropna() > 0).mean() < 0.5:
                continue
        elif s.dtype == object:
            m = s.astype(str).str.extract(r"(?:^|\s)(\d{1,2})[:h](\d{2})")
            if m[0].notna().mean() < 0.8:
                continue
            h = m[0].astype(float) + m[1].astype(float) / 60
        else:
            continue
        lab = pd.Series(pd.NA, index=t.df.index, dtype=object)
        lab[(h >= 6) & (h < 14)] = "sang"
        lab[(h >= 14) & (h < 22)] = "chieu"
        df = pd.DataFrame({"oid": t.df[t.roles["order_id"]], "s": lab}).dropna()
        return df.drop_duplicates("oid").set_index("oid")["s"]
    return None


def _customer_group(ws) -> pd.Series | None:
    o = ws.tables["orders"].df
    for t in ws.tables.values():
        if t.shared or t.roles.get("order_id") or t.roles.get("sale_date"):
            continue
        key = t.roles.get("customer_code") or t.roles.get("customer")
        grp = next((c for c in t.df.columns if re.search(r"nhom|hang_khach|hang_thanh_vien|loai_khach", c)
                    and c != key and 1 < t.df[c].nunique() <= 20), None)
        if key and grp:
            okey = "customer_code" if t.roles.get("customer_code") and "customer_code" in o else "customer"
            m = dict(zip(t.df[key].astype(str), t.df[grp].astype(str)))
            s = o.set_index("order_id")[okey].astype(str).map(m).dropna()
            return s if len(s) else None
    return None


def dimensions(ws) -> dict[str, Dim]:
    if getattr(ws, "_dims", None) is not None:
        return ws._dims
    dims: dict[str, Dim] = {}
    rules = ws.rules
    if "orders" in ws.tables:
        found = {"pay_method": _order_col(ws, ("pay_method",)),
                 "channel": _order_col(ws, ("channel",)),
                 "staff": _order_col(ws, ("staff",), r"nhan_?vien|nguoi_ban|thu_ngan|^ma_?nv$"),
                 "customer_group": _customer_group(ws)}
        if rules is None or "shift_hours" in rules:
            found["shift"] = _shift_series(ws)
        for name, s in found.items():
            if s is not None and 1 < s.nunique() <= 60:
                d = Dim(name, "order", series=s)
                for v in s.unique():
                    n = _norm(v)
                    d.values[n] = v
                    inner = re.findall(r"\(([^)]+)\)", n)   # "thu ho (cod)" -> also "cod"
                    for x in inner + [re.sub(r"\s*\([^)]*\)", "", n)]:
                        if len(x) >= 2:
                            d.values.setdefault(x.strip(), v)
                    for pat, words in SYNONYMS:
                        if re.match(pat, n):
                            for w in words:
                                d.values.setdefault(w, v)
                dims[name] = d
    L = getattr(ws, "lines", None)
    if L is not None and not L.empty:
        if "group" in L and L["group"].notna().any() and (rules is None or "group_exact" in rules):
            g = L["group"].dropna().astype(str)
            dims["group"] = Dim("group", "line", {_norm(v): v for v in g.unique() if len(_norm(v)) >= 2}, L["group"])
        if rules is None or "size_suffix" in rules:
            size = L["item"].astype(str).str.extract(r"\(\s*([mlML])\s*\)\s*$")[0].str.lower()
            if size.notna().mean() > 0.2:
                dims["size"] = Dim("size", "line", {"m": "m", "l": "l"}, size)
    ws._dims = dims
    return dims


def _hits(q: str, ws, dims: dict[str, Dim]) -> tuple[str, list[str]] | None:
    """(dimension, values in order of mention) named by the question; one dimension only."""
    found: list[tuple[int, str, str]] = []
    if "shift" in dims:
        found += [(m.start(), "shift", m.group(1)) for m in re.finditer(r"\bca (sang|chieu)\b", q)]
    if "size" in dims:
        found += [(m.start(), "size", m.group(1)) for m in re.finditer(r"\b(?:size|co|ly) ([ml])\b", q)]
    if "group" in dims:
        for m in re.finditer(r"\bnhom (?:hang (?:hoa )?)?", q):
            rest = q[m.end():]
            v = next((k for k in sorted(dims["group"].values, key=len, reverse=True)
                      if rest.startswith(k) and (len(rest) == len(k) or not rest[len(k)].isalnum())), None)
            if v:
                found.append((m.start(), "group", dims["group"].values[v]))
    if "customer_group" in dims:
        for m in re.finditer(r"\bkhach (?:hang )?(?:nhom |hang |loai )?", q):
            rest = q[m.end():]
            v = next((k for k in sorted(dims["customer_group"].values, key=len, reverse=True)
                      if rest.startswith(k) and (len(rest) == len(k) or not rest[len(k)].isalnum())), None)
            if v:
                found.append((m.start(), "customer_group", dims["customer_group"].values[v]))
    for name in ("pay_method", "channel", "staff"):
        if name not in dims:
            continue
        for k, raw in sorted(dims[name].values.items(), key=lambda kv: -len(kv[0])):
            if len(k) < 3:   # two-letter codes collide with ordinary words once accents are stripped
                continue
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", q):
                if not any(abs(m.start() - p) < len(k) and n == name for p, n, _ in found):
                    found.append((m.start(), name, raw))
    if not found:
        return None
    names = {n for _, n, _ in found}
    if len(names) != 1:
        return None
    found.sort()
    vals = list(dict.fromkeys(v for _, _, v in found))
    return found[0][1], vals


def filter_intent(question: str, as_of, ws) -> dict | None:
    if "orders" not in ws.tables:
        return None
    q = normalize_question(question)
    for pat, rep in ALIASES:
        q = re.sub(pat, rep, q)
    if UNHANDLED.search(q):
        return None
    dims = dimensions(ws)
    top = "staff" in dims and not re.search(r"\bkhach\b", q) and re.search(
        r"(nhan vien|nguoi ban) .{0,25}(cao nhat|nhieu nhat|gioi nhat)|(nhan vien|nguoi ban) nao|ai la nguoi ban|ai ban", q)
    hit = ("staff", []) if top else _hits(q, ws, dims)
    if not hit:
        return None
    dim, vals = hit
    # another named item / customer in the question is a second filter this recipe does not handle
    names = set(dims[dim].values)
    if any(n in q and n not in names and not any(n in x for x in names) for n in shop_vocab(ws)):
        return None
    periods = find_periods(q, as_of)
    if re.search(r"\btu dau nam\b|\bnam nay\b", q):
        periods = [("year_to_date", period("year_to_date", as_of))] + [p for p in periods if p[0] != "year_to_date"]
    if len(periods) != 1:
        return None
    if top:
        return {"kind": "top", "dim": dim, "period": periods[0][1], "kind_period": periods[0][0]}
    many = r"(bao nhieu|bnhieu|nhieu|bn|may\w*|so\w*|total)"
    if re.search(r"trung binh", q):
        metric = "avg"
    elif re.search(rf"\b{many} (don|hoa don|luot)\b", q) and not re.search(r"doanh thu|dthu|tien", q):
        metric = "orders"
    elif re.search(rf"\b{many} (ly|coc|mon|cai|chai|lon|goi|hop)\b", q) or (dim == "size" and re.search(r"\bly\b", q)):
        metric = "units"
    else:
        metric = "revenue"
    if metric == "avg" and dims[dim].level != "order":
        return None
    if SHARE.search(q) and len(vals) == 1:
        shape = "share"
    elif len(vals) == 2 and re.search(r"(%|phan tram)", q) and COMPARE.search(q):
        shape = "compare"
    elif len(vals) == 2:
        shape = "list"
    elif len(vals) == 1:
        shape = "number"
    else:
        return None
    if shape == "compare":
        base = vals[1]
        m = re.search(r"(so voi|\bvs\b|chenh( lech)?( voi)?|hon|kem)\s+(.*)", q)
        if dim == "shift":
            base = "sang"                                 # rulebook §2: the morning is the base
        elif m:
            after = m.group(4)
            base = next((v for v in vals if _norm(v) in after or any(k in after[:40] for k, r in dims[dim].values.items()
                                                                       if r == v)), vals[1])
        vals = [v for v in vals if v != base] + [base]
    return {"kind": shape, "dim": dim, "values": vals, "metric": metric, "period": periods[0][1],
            "kind_period": periods[0][0]}


def _value(ws, dim: Dim, value, metric: str, r) -> float:
    o = ws.tables["orders"].df
    L = ws.lines
    if dim.level == "order":
        ids = set(dim.series[dim.series == value].index.astype(str))
        if metric != "units":
            m = Metrics(o[o["order_id"].astype(str).isin(ids)])
            return {"revenue": m.revenue, "orders": m.order_count, "avg": m.avg_order}[metric](r)
        if L is None:
            return float("nan")
        mask = L["order_id"].astype(str).isin(ids)
    elif dim.name == "group":
        mask = L["group"].fillna("").map(_norm).eq(_norm(value))
    else:
        mask = (dim.series == value).fillna(False)
    sel = L[mask & L["in_revenue"].fillna(False).astype(bool)
            & (L["order_date"] >= pd.Timestamp(r[0])) & (L["order_date"] <= pd.Timestamp(r[1]))]
    if metric == "orders":
        return int(sel["order_id"].nunique())
    if metric == "units":
        sel = sel[~sel.get("is_fee", pd.Series(False, index=sel.index)).fillna(False).astype(bool)]
        return float(sel["qty_base"].sum())
    return int(round(sel["amount"].sum()))


def filter_candidate(question: str, as_of, ws) -> tuple[str, object] | None:
    it = filter_intent(question, as_of, ws)
    if not it or (ws.rules is not None and ({"in_revenue"} | rules_needed(question)) - set(ws.rules)):
        return None
    o = ws.tables["orders"].df
    if not {"order_date", "in_revenue", "revenue_lines"} <= set(o.columns):
        return None
    m = Metrics(o)
    r = it["period"]
    if m.outside(r):
        return ("unanswerable", None)
    dim = dimensions(ws)[it["dim"]]
    if it["kind"] == "top":
        if ws.rules is not None and "staff_by_revenue" not in ws.rules:
            return None
        vals = sorted(dim.values.values(), key=str)
        scores = sorted(((_value(ws, dim, v, "revenue", r), str(v)) for v in set(vals)), key=lambda x: (-x[0], x[1]))
        return ("string", scores[0][1]) if scores and scores[0][0] > 0 else None
    vals = [_value(ws, dim, v, it["metric"], r) for v in it["values"]]
    if it["kind"] == "number":
        return ("number", vals[0])
    if it["kind"] == "list":
        return ("list", vals)
    if it["kind"] == "share":
        total = m.revenue(r) if it["metric"] == "revenue" else m.order_count(r)
        return ("percent", round(vals[0] / total * 100, 2)) if total else None
    a, b = vals
    return ("percent", round((a - b) / b * 100, 2)) if b else None
