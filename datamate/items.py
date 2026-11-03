"""Item-level recipes (layer 3): a derived `lines` table (one row per sale line, standard columns)
and the item questions built on it — best sellers (base units), revenue / units of one item,
an item group's share of revenue, most profitable item. Definitions are the rulebook's (§1, §3,
§5); the rule-encoding parts sit behind the rules guard (rules.py)."""
from __future__ import annotations

import re

import pandas as pd

from .metrics import find_periods, rules_needed, shop_vocab
from .semantics import strip_accents

FEE_RE = r"^phi phuc vu"


def _catalog(ws):
    """Item catalog: a table with item names (and/or codes) that is not a sales table."""
    best = None
    for t in ws.tables.values():
        if t.shared or t.name.startswith(("orders", "lines")) or t.roles.get("order_id"):
            continue
        if t.roles.get("item") or t.roles.get("item_code"):
            score = len(t.df) + 1000 * bool(t.roles.get("item_group"))
            if best is None or score > best[0]:
                best = (score, t)
    return best[1] if best else None


def build_lines(ws) -> pd.DataFrame | None:
    o = ws.tables.get("orders")
    if o is None or "in_revenue" not in o.df:
        return None
    od = o.df.set_index(o.df["order_id"].astype(str))
    parts = []
    for t in ws.tables.values():
        r = t.roles
        if t.shared or t.name.startswith(("orders", "lines")) or not r.get("line_amount") or not (r.get("item") or r.get("item_code")):
            continue
        df = t.df[~t.df["is_dup_line"]] if "is_dup_line" in t.df else t.df
        oid = r.get("order_id") or ("row_id" if "row_id" in od.columns else None)
        if not r.get("order_id"):
            continue
        p = pd.DataFrame({"order_id": df[oid].astype(str).values})
        p["item"] = df[r["item"]].astype(str).values if r.get("item") else None
        p["item_code"] = df[r["item_code"]].astype(str).values if r.get("item_code") else None
        p["unit"] = df[r["unit"]].astype(str).values if r.get("unit") else None
        p["qty"] = _lead_number(df[r["qty"]]).values if r.get("qty") else None
        p["qty_base"] = _lead_number(df[r["qty_base"]]).values if r.get("qty_base") else None
        p["source"] = t.name
        p["amount"] = pd.to_numeric(df[r["line_amount"]], errors="coerce").values
        p["cost"] = pd.to_numeric(df[r["cost"]], errors="coerce").values if r.get("cost") else None
        parts.append(p)
    if not parts:
        return None
    L = pd.concat(parts, ignore_index=True)
    L = L[L["order_id"].isin(od.index)]
    for c in ("order_date", "in_revenue", "status"):
        L[c] = L["order_id"].map(od[c]).values
    cat = _catalog(ws)
    if cat is not None:
        cr = dict(cat.roles)
        if not cr.get("item"):  # catalog name column without a role: "tên ..." description or name
            cr["item"] = next((c for c, m in cat.meta.items() if re.match(r"^tên", m.get("desc", ""))
                               or re.fullmatch(r"ten_?(hang|sp|san_?pham|mon|hang_?hoa|thuoc|vat_?tu)?|tengoi|mon", c)), None)
        if cr.get("item_code") and cr.get("item"):
            names = dict(zip(cat.df[cr["item_code"]].astype(str), cat.df[cr["item"]].astype(str)))
            codes = {v: k for k, v in names.items()}
            if L["item"].isna().all():
                L["item"] = L["item_code"].map(names)
            if L["item_code"].isna().all():
                L["item_code"] = L["item"].map(codes)
        if cr.get("item_group"):
            key = cr.get("item_code") if cr.get("item_code") and L["item_code"].notna().any() else cr.get("item")
            lk = "item_code" if key == cr.get("item_code") else "item"
            # names written with other case / accents / spacing are the same item (rulebook §3)
            grp = {_k(k): g for k, g in zip(cat.df[key].astype(str), cat.df[cr["item_group"]].astype(str))}
            L["group"] = L[lk].map(lambda v: grp.get(_k(v)) if pd.notna(v) else None)
            if lk == "item_code" and cr.get("item"):  # lines without a code: fall back to the name
                by_name = {_k(k): g for k, g in zip(cat.df[cr["item"]].astype(str), cat.df[cr["item_group"]].astype(str))}
                miss = L["group"].isna() & L["item"].notna()
                L.loc[miss, "group"] = L.loc[miss, "item"].map(lambda v: by_name.get(_k(v)))
    # base-unit quantity where the book does not give one: qty x factor(item, line unit)
    fac = unit_factors(ws)
    need = L["qty_base"].isna() if "qty_base" in L else pd.Series(True, index=L.index)
    if need.any() and L["qty"].notna().any():
        f = [fac.get((_k(i), _k(u)), fac.get((_k(c), _k(u)), 1.0)) if u is not None else 1.0
             for i, c, u in zip(L["item"], L["item_code"], L["unit"])]
        L.loc[need, "qty_base"] = (L["qty"] * pd.Series(f, index=L.index))[need]
    L["item"] = L["item"].where(L["item"].notna() & (L["item"].astype(str) != "None"), L["item_code"])
    L["is_fee"] = L["item"].fillna("").map(strip_accents).str.match(FEE_RE)
    return L


class Items:
    def __init__(self, lines: pd.DataFrame, rules: set | None = None):
        self.L = lines
        self.fee_ok = rules is None or "service_fee_not_item" in rules

    def _in(self, r) -> pd.DataFrame:
        L = self.L
        m = (L["order_date"] >= pd.Timestamp(r[0])) & (L["order_date"] <= pd.Timestamp(r[1])) & L["in_revenue"].fillna(False).astype(bool)
        return L[m]

    def _key(self, L):
        return "item_code" if L["item_code"].notna().any() else "item"

    def best_sellers(self, r, n: int = 1) -> list[str]:
        """Rulebook §3: by base-unit quantity, cancelled/returned orders out, service fee not an item,
        ties by code ascending."""
        L = self._in(r)
        if self.fee_ok:
            L = L[~L["is_fee"]]
        k = self._key(L)
        g = L.groupby(k).agg(q=("qty_base", "sum"), name=("item", "first")).reset_index()
        g = g.sort_values(["q", k], ascending=[False, True], kind="mergesort")
        return [str(x) for x in g["name"].head(n)]

    def _match(self, L, item: str):
        key = re.sub(r"\s+", " ", strip_accents(item)).strip()
        return L[L["item"].fillna("").map(lambda v: re.sub(r"\s+", " ", strip_accents(v)).strip()) == key]

    def item_revenue(self, item: str, r) -> int:
        return int(round(self._match(self._in(r), item)["amount"].sum()))

    def item_units(self, item: str, r) -> float:
        v = self._match(self._in(r), item)["qty_base"].sum()
        return int(v) if float(v).is_integer() else round(float(v), 2)

    def group_share(self, group: str, r, orders: pd.DataFrame | None = None) -> float:
        """Rulebook §1/§3: group revenue / revenue of the whole shop in the period (fee lines and
        every order included in the denominator)."""
        L = self._in(r)
        tot = L["amount"].sum()
        if orders is not None and "revenue_lines" in orders:
            o = orders[(orders["order_date"] >= pd.Timestamp(r[0])) & (orders["order_date"] <= pd.Timestamp(r[1]))]
            tot = o.loc[o["in_revenue"].fillna(False).astype(bool), "revenue_lines"].sum() or tot
        g = L[L["group"].fillna("").map(strip_accents).str.strip() == strip_accents(group).strip()]
        return round(g["amount"].sum() / tot * 100, 2) if tot else float("nan")

    def most_profitable(self, r, n: int = 1) -> list[str] | None:
        L = self._in(r)
        if L["cost"].isna().all():
            return None
        if self.fee_ok:
            L = L[~L["is_fee"]]
        L = L.assign(margin=L["amount"] - L["qty"].fillna(0) * L["cost"].fillna(0))
        k = self._key(L)
        g = L.groupby(k).agg(m=("margin", "sum"), name=("item", "first")).reset_index()
        g = g.sort_values(["m", k], ascending=[False, True], kind="mergesort")
        return [str(x) for x in g["name"].head(n)]


def item_intent(question: str, as_of, ws) -> dict | None:
    q = re.sub(r"\s+", " ", strip_accents(question))
    q = re.sub(r"\b(\w*?)([aeiouy])\2(\w*)\b", r"\1\2\3", q).replace("nhiu", "nhieu")
    if re.search(r"\bno\b|cong no|tru .{0,30}ra|ngoai tru|khong tinh|size|\(m\)|\(l\)|kenh|khach|nhan vien|ke don|hoat chat|"
                 r"\bton\b|trong kho|con (lai|bao nhieu|bn)|het hang|nhap|chan|\ble\b", q):
        return None
    periods = find_periods(q, as_of)
    if not periods:
        return None
    r1 = periods[0][1]
    top = re.search(r"\btop\s?(\d{1,2})\b|(?<![/\d])\b(\d{1,2}) (mat hang|san pham|mon|hang|sp)\b", q)
    n = int(top.group(1) or top.group(2)) if top else None
    if re.search(r"ban chay|ban nhieu nhat|ban duoc nhieu nhat|hot nhat", q) and not re.search(r"\bit nhat|e nhat|cham nhat", q):
        return {"kind": "best_sellers", "period": r1, "n": n}
    if re.search(r"(loi|lai) nhat|lai gop (cao|nhieu) nhat", q):
        return {"kind": "most_profitable", "period": r1, "n": n}
    names = [v for v in shop_vocab(ws) if v in q]
    if not names:
        return None
    name = names[0]  # longest match (vocab is sorted by length)
    L = getattr(ws, "lines", None)
    if L is None:
        return None
    if re.search(r"nhom", q) and re.search(r"%|phan tram|ti trong|ty trong|chiem", q):
        groups = {strip_accents(g).strip(): g for g in L["group"].dropna().unique()} if "group" in L else {}
        g = next((groups[k] for k in sorted(groups, key=len, reverse=True) if k and k in q), None)
        return {"kind": "group_share", "period": r1, "group": g} if g else None
    items = {strip_accents(i).strip(): i for i in L["item"].dropna().unique()}
    if name not in items:
        return None
    if re.search(r"doanh thu|dthu|tien ban|thu ve|ban duoc (bao nhieu|bn) tien|tien hang", q):
        return {"kind": "item_revenue", "period": r1, "item": items[name]}
    if re.search(r"(ban duoc|ban|tong cong).{0,30}(bao nhieu|bn|may)|so luong", q) and not re.search(r"\bgia\b|tien", q):
        return {"kind": "item_units", "period": r1, "item": items[name]}
    return None


def item_candidate(question: str, as_of, ws) -> tuple[str, object] | None:
    L = getattr(ws, "lines", None)
    if L is None or L.empty:
        return None
    it = item_intent(question, as_of, ws)
    if not it:
        return None
    rules = ws.rules
    if rules is not None and ({"in_revenue"} | rules_needed(question)) - set(rules):
        return None  # item revenue follows the order revenue rule and the period conventions
    I = Items(L, rules)
    k = it["kind"]
    if k == "best_sellers":
        if rules is not None and "best_seller_base_units" not in rules:
            return None
        v = I.best_sellers(it["period"], it["n"] or 1)
        return (("list", v) if it["n"] else ("string", v[0])) if v else None
    if k == "most_profitable":
        if rules is not None and "gross_margin_line_cost" not in rules:
            return None
        v = I.most_profitable(it["period"], it["n"] or 1)
        return (("list", v) if it["n"] else ("string", v[0])) if v else None
    if k == "item_revenue":
        return ("number", I.item_revenue(it["item"], it["period"]))
    if k == "item_units":
        return ("number", I.item_units(it["item"], it["period"]))
    if k == "group_share":
        return ("percent", I.group_share(it["group"], it["period"], ws.tables["orders"].df if "orders" in ws.tables else None))
    return None


def item_card(question: str, as_of, ws) -> str | None:
    L = getattr(ws, "lines", None)
    it = item_intent(question, as_of, ws) if L is not None else None
    if not it:
        return None
    r = lambda p: f"('{p[0]}', '{p[1]}')"  # noqa: E731
    k = it["kind"]
    call = {"best_sellers": f"I.best_sellers({r(it['period'])}, n={it.get('n') or 1})",
            "most_profitable": f"I.most_profitable({r(it['period'])}, n={it.get('n') or 1})",
            "item_revenue": f"I.item_revenue({it.get('item')!r}, {r(it['period'])})",
            "item_units": f"I.item_units({it.get('item')!r}, {r(it['period'])})",
            "group_share": f"I.group_share({it.get('group')!r}, {r(it['period'])})"}[k]
    return (f"Item recipe ({k}, on the derived `lines` table: in-revenue orders only, base units, service fee "
            f"not an item, ties by code). Ready to run in run_python:\n  print({call})")


def _k(v) -> str:
    return re.sub(r"\s+", " ", strip_accents(str(v))).strip()


def unit_factors(ws) -> dict[tuple[str, str], float]:
    """(item, unit) -> base units in 1 unit, from README lines '- Bia Tiger lon: 1 thùng = 24 lon'
    and conversion tables (đơn vị lớn / hệ số / tên hàng). Unknown pairs default to 1."""
    f: dict[tuple[str, str], float] = {}
    for text in ws.docs.values():
        for m in re.finditer(r"^\s*[-*]\s*(.+?):\s*1\s+([^\s=]+(?:\s[^\s=]+)?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(\S+)", text, re.M):
            f[(_k(m.group(1)), _k(m.group(2)))] = float(m.group(3).replace(",", "."))
    for t in ws.tables.values():
        r = t.roles
        if t.shared or not (r.get("big_unit") and r.get("factor")):
            continue
        item = r.get("item") or r.get("item_code")
        if not item:
            continue
        for it, u, fa in zip(t.df[item], t.df[r["big_unit"]], pd.to_numeric(t.df[r["factor"]], errors="coerce")):
            if pd.notna(fa):
                f[(_k(it), _k(u))] = float(fa)
    return f


def _lead_number(s: pd.Series) -> pd.Series:
    """'2 ly' -> 2, '1,5 kg' -> 1.5, numbers unchanged."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    x = s.astype(str).str.extract(r"^\s*(-?\d+(?:[.,]\d+)?)")[0].str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")
