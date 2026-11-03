"""Make the shop data easy for a small model, without per-shop code.

Every shop README describes its columns with the same vocabulary
("- **Mã hóa đơn**: mã đơn", "- **Status**: trạng thái đơn"...). From that we derive column
roles and add helper columns that encode rulebook definitions once, in code:

  * ASCII snake_case column names (no SQL quoting of "Ngày bán__date" any more)
  * order-level cells left blank on later lines of an order are filled from the order (rulebook)
  * status_norm   per line: hoan_thanh / huy / tra_hang / dang_giao (None = unknown)
  * order_status  per order: tra_hang if any line is a return or has negative quantity,
                  else huy / dang_giao / hoan_thanh
  * in_revenue    order_status in (hoan_thanh, dang_giao)
  * <customer>_is_walkin   khách lẻ / KL / blank customer
  * is_dup_line   exact duplicate of an earlier line of the same order (count once)

Raw columns are kept; the profile lists original name, README meaning and what was derived.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

ROLE_PATTERNS = [  # (role, regex on the README description); first match wins
    ("order_id", r"^mã đơn\b|^số (hoá|hóa) đơn|^số thứ tự đơn"),
    ("status", r"^trạng thái đơn"),
    ("note_status", r"^ghi chú.*trạng thái|^ghi chú của đơn;.*(huỷ|hủy)"),
    ("customer_code", r"^mã khách"),
    ("customer", r"^tên khách"),
    ("sale_date", r"^(ngày bán|thời điểm đặt đơn|ngày đặt|thời gian đặt|ngày (sửa|mua|lập|xuất|giao dịch))"),
    ("sale_time", r"^giờ đặt"),
    ("qty_base", r"^số lượng đã đổi ra đơn vị cơ bản"),
    ("qty_returned", r"^số lượng khách trả lại"),
    ("qty", r"^số lượng(\s|$|,|:)"),
    ("line_amount", r"^thành tiền sau giảm|^thành tiền của dòng"),
    ("goods_amount", r"^tiền hàng"),
    ("order_total", r"^tổng tiền đơn|^số khách phải trả cho cả đơn|^tổng tiền cả (phiếu|đơn)"),
    ("paid", r"^đã thanh toán|^khách đã trả"),
    ("pay_id", r"^mã phiếu thu"),
    ("pay_date", r"^(thời điểm|ngày|thời gian|ngày giờ) (thu|trả)\b"),
    ("pay_amount", r"^số tiền( thu| trả| khách trả)?( lần đó)?$|^tiền thu$|^số tiền trả lần"),
    ("pay_method", r"^hình thức thanh toán|^phương thức thanh toán"),
    ("channel", r"^kênh bán|^nguồn đơn"),
    ("balance_end", r"^số khách còn nợ"),
    ("item_group", r"^nhóm hàng"),
    ("big_unit", r"^đơn vị lớn|^đơn vị quy đổi"),
    ("factor", r"^hệ số"),
    ("base_unit", r"^đơn vị cơ bản"),
    ("unit", r"^đơn vị tính"),
    ("cost", r"^giá vốn"),
    ("item_code", r"^mã hàng"),
    ("item", r"^tên hàng"),
    ("empty", r"^cột trống"),
]
# matched on the ORIGINAL lower-cased text: "tổng"/"tong" is a total, "tống" is a surname
# (accent stripping would turn customer "Tống Nam Loan" into a fake "tổng năm" total row)
# a whole-cell label: "Cộng", "Tổng tháng", "Tổng tháng 2/2025", "TỔNG NGÀY 01/10", "T01-2025 Tổng cộng"
TOTAL_ROW_RE = (r"^((tổng|tong|cộng|cong|lũy kế|luỹ kế|luy ke)(\s+(ngày|ngay|tháng|thang|tuần|tuan|năm|nam|quý|quy|kỳ|ky|ca))?"
                r"\s*[:\-]?\s*([\d/.\-]+\s*)*$|(tổng|tong|cộng|cong)\s+(ngày|ngay|tháng|thang|tuần|tuan|quý|quy)\s*[:\-]?\s*\d|"
                r"(tổng|tong) (cộng|cong)\b|(tổng|tong|cộng|cong)\s*:|total\b|subtotal\b|grand total\b)")
ORDER_LEVEL_ROLES = ("status", "note_status", "customer", "customer_code", "sale_date")
WALKIN = {"khachle", "kl", "khachvanglai", "vanglai", "walkin", "khach", "le", "khle", "nan", "none", ""}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def parse_readme(text: str) -> dict[str, dict[str, str]]:
    """'## Cột trong `File.xlsx`' sections -> {file: {column: description}}; a section naming a
    sheet ('`QuanLy.xlsx`, sheet `NhapHang`') is keyed 'file#sheet'; names may hold <placeholders>."""
    out: dict[str, dict[str, str]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^#+\s*Cột trong\s+`([^`]+)`(?:\s*,\s*sheet\s+`([^`]+)`)?", line.strip(), re.I)
        if m:
            key = f"{m.group(1)}#{m.group(2)}" if m.group(2) else m.group(1)
            cur = out.setdefault(key, {})
            continue
        if line.startswith("#"):
            cur = None
        m = re.match(r"^\s*-\s*\*\*(.+?)\*\*\s*:\s*(.+)$", line)
        if m and cur is not None:
            cur[m.group(1).strip()] = m.group(2).strip()
    return out


NAME_ROLES = [  # fallback when the README gives no per-column description (e.g. SQLite POS dumps)
    ("order_id", r"^(ma_?)?(hoa_?don|don_?hang|bill|order_?id|ma_?don)$|^mahoadon$|^madonhang$|"
                 r"^(don_?so|so_?don|so_?hd|ma_?hd|so_?hoa_?don|ma_?giao_?dich)$"),
    ("status", r"^(trang_?thai|tinh_?trang|status)(_?(don(_?hang)?|giao|xu_?ly|hd))?$"),
    ("pay_date", r"^(ngay_?gio_?thu|thoi_?diem_?thu|ngay_?thu|thoi_?gian_?thu|ngaygiothu|ngay_?tt)$"),
    ("sale_date", r"^(ngay_?lap|ngay_?ban|ngay_?dat(_?hang)?|ngay_?giao_?dich|thoi_?gian_?lap|order_?date|thoi_?gian|"
                  r"ngay_?(hd|hoa_?don|ct|chung_?tu)|ngay)$"),
    ("customer_code", r"^ma_?(khach|khach_?hang|kh)(_?mua)?$|^makhachhang$|^makh$"),
    ("customer", r"^(ten_?(khach|kh)(_?hang)?|khach_?hang(_?ten)?|ben_?mua|khach)$"),
    ("pay_method", r"^(phuong_?thuc|hinh_?thuc)(_?tt|_?thanh_?toan)?$|^pttt$"),
    ("channel", r"^(kenh_?ban(_?hang)?|nguon_?don|nguon)$"),
    ("qty_base", r"^(sl|so_?luong)_?(quy_?doi|dv_?co_?ban|co_?ban)$"),
    ("qty", r"^(sl|so_?luong|s_?l)$"),
    ("line_amount", r"^(tri_?gia|thanh_?tien(_?d)?)$"),
    ("item", r"^(hang_?hoa|ten_?hang(_?hoa)?|mat_?hang|ten_?sp)$"),
    ("order_total", r"^tong_?thanh_?toan$"),
    ("item_code", r"^ma_?hang(_?hoa)?$|^mahanghoa$"),
]


def role_from_name(slug: str) -> str | None:
    for role, pat in NAME_ROLES:
        if re.search(pat, slug):
            return role
    return None


def role_of(desc: str) -> str | None:
    d = desc.strip().casefold()
    for role, pat in ROLE_PATTERNS:
        if re.search(pat, d):
            return role
    return None


def norm_status(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = re.sub(r"\s+", " ", strip_accents(v)).strip()
    if not s:
        return None
    if re.search(r"\bdang\b|\bcho\b|chua giao|processing|pending|shipping|delivering|in transit", s):
        return "dang_giao"
    if re.search(r"\btra\b|\bdoi\b|hoan hang|return|refund", s):
        return "tra_hang"
    if re.search(r"huy|cancel|khong giao duoc|giao that bai|that bai|fail|\bbom\b", s):
        return "huy"
    if re.search(r"xong|hoan thanh|thanh cong|\bda giao\b|\bda lay\b|\bda nhan\b|complete|done|success|delivered|\bok\b", s):
        return "hoan_thanh"
    return None


def is_walkin(v) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return True
    return re.sub(r"[\W_]+", "", strip_accents(v)) in WALKIN


def _slug(text: str) -> str:
    t = re.sub(r"[^0-9a-z]+", "_", strip_accents(text)).strip("_")
    return t if t and not t[0].isdigit() else f"c_{t}"


def _source_file(source: str) -> str:
    return re.split(r" \(|#", source, maxsplit=1)[0].split("/")[-1]


def _sheets_of(source: str) -> list[str]:
    m = re.search(r"#(.+)$", source)
    if m:
        return [m.group(1)]
    m = re.search(r"\(sheets: (.+)\)", source)
    return [s.strip() for s in m.group(1).split(",")] if m else []


def _name_match(pattern: str, name: str) -> bool:
    """Exact (slug) match, or a README pattern with <placeholders> ('BanHang_T<tháng>_<năm>.xlsx')."""
    if _slug(pattern) == _slug(name):
        return True
    if "<" in pattern:
        rx = re.escape(pattern)
        rx = re.sub(r"<[^>]*>", ".+?", rx.replace(r"\<", "<").replace(r"\>", ">"))
        return re.fullmatch(rx, name, re.I) is not None
    return False


def _match_desc(readme: dict[str, dict[str, str]], source: str) -> dict[str, str]:
    f = _source_file(source)
    sheets = _sheets_of(source)
    for key, cols in readme.items():  # file + sheet first
        if "#" in key:
            kf, ks = key.split("#", 1)
            if _name_match(kf, f) and any(_slug(ks) == _slug(s) for s in sheets):
                return cols
    for key, cols in readme.items():
        if "#" not in key and _name_match(key, f):
            return cols
    for key, cols in readme.items():  # a sheet-specific section when the file has that single kind of sheet
        if "#" in key and _name_match(key.split("#", 1)[0], f) and not sheets:
            return cols
    return {}


ALL_RULES = {"in_revenue", "walkin", "dup_line", "order_fill", "no_status_note"}


def enrich_table(t, readme: dict[str, dict[str, str]], rules: set[str] | None = None) -> None:
    """Rename to ASCII, attach roles/descriptions and add helper columns (in place on TableInfo).
    `rules`: rule-encoding helpers allowed by the current rulebook (see rules.py)."""
    rules = ALL_RULES if rules is None else rules
    df = t.df
    descs = _match_desc(readme, t.source)
    new_cols, seen, meta = [], set(), {}
    for c in df.columns:
        base = c[:-6] if c.endswith("__date") else c
        n = "_sheet" if c == "_sheet" else _slug(base) + ("__date" if c.endswith("__date") else "")
        while n in seen:
            n += "_2"
        seen.add(n)
        new_cols.append(n)
        desc = descs.get(c) or descs.get(base) or ""
        role = None
        if not c.endswith("__date"):
            role = (role_of(desc) if desc else None) or role_from_name(n)
        meta[n] = {"orig": c, "desc": desc if not c.endswith("__date") else f"parsed date of {_slug(base)}",
                   "role": role}
    df.columns = new_cols
    notes: list[str] = []
    # subtotal / total rows exported into the data ("Cộng ngày 01/05/2025", "Tổng cộng") are not records
    text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if text_cols:
        tot = pd.Series(False, index=df.index)
        for c in text_cols:
            v = df[c].dropna().astype(str).map(lambda x: unicodedata.normalize("NFC", x).casefold().strip())
            tot.loc[v.index] |= v.str.match(TOTAL_ROW_RE)
        if tot.any():
            first = df.loc[tot, text_cols].iloc[0]
            sample = " ".join(str(v) for v in first if isinstance(v, str))[:60]
            df = df[~tot].reset_index(drop=True)
            notes.append(f"removed {int(tot.sum())} subtotal/total rows exported into the data (e.g. '{sample}')")
    roles: dict[str, str] = {}
    for n, m in meta.items():
        if m["role"] and m["role"] not in roles:
            roles[m["role"]] = n
    if "item" not in roles and "hang" in df.columns and (roles.get("line_amount") or roles.get("qty")):
        roles["item"] = "hang"  # bare "Hàng" is the item only in a sales table (elsewhere it can be a tier)
        meta["hang"]["role"] = "item"
    if "sale_date" in roles and f"{roles['sale_date']}__date" in df.columns:
        roles["sale_date_parsed"] = f"{roles['sale_date']}__date"
    oid = roles.get("order_id")
    if oid and df[oid].duplicated().any() and "order_fill" in rules:
        if df[oid].isna().any():
            k = int(df[oid].isna().sum())
            df[oid] = df[oid].ffill()
            notes.append(f"{oid}: {k} blank cells on continuation lines filled from the line above")
        g = df.groupby(oid, sort=False)
        for r in ORDER_LEVEL_ROLES + ("sale_date_parsed",):
            c = roles.get(r)
            if c and df[c].isna().any():
                before = int(df[c].isna().sum())
                df[c] = g[c].transform("first")
                filled = before - int(df[c].isna().sum())
                if filled:
                    notes.append(f"{c}: {filled} blank cells filled from the first line of the same order (rulebook)")
    if oid and df[oid].duplicated().any() and "dup_line" in rules:
        raw_cols = [n for n in df.columns if n != "_sheet" and not n.endswith("__date")]
        df["is_dup_line"] = df.duplicated(subset=raw_cols, keep="first")
        if df["is_dup_line"].any():
            notes.append(f"is_dup_line: {int(df['is_dup_line'].sum())} exact duplicate lines within an order (count once)")
    if not roles.get("status") and not roles.get("note_status") and "no_status_note" in rules and roles.get("sale_date")             and (roles.get("line_amount") or roles.get("order_total") or roles.get("goods_amount")):
        note = next((c for c, m in meta.items() if re.match(r"^ghi chú", m.get("desc", "").lower())
                     or re.fullmatch(r"ghi_?chu|note|notes", c)), None)
        if note:
            roles["note_status"] = note
        else:
            df["status_norm"] = "hoan_thanh"
            notes.append("no status column and no note column: every row completed (rulebook)")
    st = roles.get("status") or roles.get("note_status")
    if st:
        if roles.get("status"):
            df["status_norm"] = df[st].map(norm_status)
        else:  # rulebook: no status column -> status in the note; a note without status = completed
            df["status_norm"] = df[st].map(lambda v: norm_status(v) or "hoan_thanh")
        vc = df.groupby(df[st].astype(str), dropna=False)["status_norm"].first()
        mapping = ", ".join(f"'{k}'→{v}" for k, v in vc.items() if k != "nan")
        notes.append(f"status_norm from {st}: {mapping}" + (" (others → hoan_thanh)" if not roles.get("status") else ""))
        if oid and "in_revenue" in rules:
            neg = pd.Series(False, index=df.index)
            for r in ("qty", "qty_base"):
                c = roles.get(r)
                if c and pd.api.types.is_numeric_dtype(df[c]):
                    neg |= df[c] < 0
            ret = roles.get("qty_returned")
            if ret and pd.api.types.is_numeric_dtype(df[ret]):
                neg |= df[ret].fillna(0) > 0
            line = df["status_norm"].mask(neg, "tra_hang")
            flags = pd.DataFrame({"o": df[oid], "s": line})
            agg = flags.groupby("o")["s"].agg(lambda s: "tra_hang" if (s == "tra_hang").any() else
                                              "huy" if (s == "huy").any() else
                                              "dang_giao" if (s == "dang_giao").any() else
                                              "hoan_thanh" if (s == "hoan_thanh").any() else None)
            df["order_status"] = df[oid].map(agg)
            df["in_revenue"] = df["order_status"].isin(["hoan_thanh", "dang_giao"])
            notes.append("order_status: one status per order (tra_hang if any line returned or negative qty, else "
                         "huy / dang_giao / hoan_thanh); in_revenue = order_status in (hoan_thanh, dang_giao)")
        elif not oid and "in_revenue" in rules and (roles.get("line_amount") or roles.get("order_total")):
            # rulebook §8: a book not recorded by order -> every row is its own record
            df["order_status"] = df["status_norm"]
            df["in_revenue"] = df["order_status"].isin(["hoan_thanh", "dang_giao"])
            notes.append("no order id: each row is one record (rulebook §8); order_status = status_norm, "
                         "in_revenue = order_status in (hoan_thanh, dang_giao)")
    for r in ("customer", "customer_code"):
        c = roles.get(r)
        if c and "walkin" in rules:
            df[f"{c}_is_walkin"] = df[c].map(is_walkin)
            notes.append(f"{c}_is_walkin: khách lẻ / KL / blank → not a customer (rulebook §3)")
    t.df = df
    t.meta = meta
    t.roles = roles
    t.notes = t.notes + notes


def _returned_ids(t) -> set:
    """Order ids with a return recorded on lines (negative quantity or returned quantity > 0)."""
    df, r = t.df, t.roles
    oid = r.get("order_id")
    if not oid:
        return set()
    neg = pd.Series(False, index=df.index)
    for k in ("qty", "qty_base"):
        c = r.get(k)
        if c and pd.api.types.is_numeric_dtype(df[c]):
            neg |= df[c] < 0
    c = r.get("qty_returned")
    if c and pd.api.types.is_numeric_dtype(df[c]):
        neg |= df[c].fillna(0) > 0
    return set(df.loc[neg, oid].astype(str))


def link_orders(tables: list, rules: set[str] | None = None) -> None:
    """Header tables carry order_status; line tables (same order ids) get it too, and returns
    recorded only on line tables mark the whole order as tra_hang everywhere."""
    heads = [t for t in tables if "order_status" in t.df.columns and t.roles.get("order_id")]
    if not heads or (rules is not None and "in_revenue" not in rules):
        return
    returned = set().union(*(_returned_ids(t) for t in tables))
    status: dict[str, str] = {}
    for h in heads:
        ids = h.df[h.roles["order_id"]].astype(str)
        status.update(dict(zip(ids, h.df["order_status"])))
    for k in returned & set(status):
        status[k] = "tra_hang"
    for t in tables:
        oid = t.roles.get("order_id")
        if not oid:
            continue
        ids = t.df[oid].astype(str)
        if ids.isin(status).mean() < 0.5:
            continue  # different kind of id (e.g. purchase orders)
        new = ids.map(status)
        changed = int((t.df.get("order_status", pd.Series(index=t.df.index, dtype=object)) != new).sum())
        t.df["order_status"] = new
        t.df["in_revenue"] = new.isin(["hoan_thanh", "dang_giao"])
        if t in heads:
            if changed:
                t.notes.append(f"order_status: {changed} lines set to tra_hang from returns recorded in a line table")
        else:
            t.notes.append(f"order_status / in_revenue: copied from {heads[0].name} by {oid} "
                           "(returns on any table make the whole order tra_hang)")


CONVENTIONS = [  # README "Quy ước" sentences that name a column's meaning -> role
    (r"là số tiền trả lần đó", "pay_amount"),
    (r"là số khách phải trả cho cả đơn", "order_total"),
    (r"là giá vốn một đơn vị bán", "cost"),
]


def apply_conventions(tables: list, docs: dict[str, str]) -> None:
    """'**Tiền thu** là số tiền trả lần đó' or '`thanh_toan.SoTienThu` là ...' -> set that column's role."""
    text = "\n".join(v for k, v in docs.items() if k.lower().split("/")[-1].startswith("readme"))
    for phrase, role in CONVENTIONS:
        for m in re.finditer(r"(?:\*\*([^*]+)\*\*|`(\w+)\.(\w+)`)\s*" + phrase, text):
            col, tab = (m.group(1), None) if m.group(1) else (m.group(3), m.group(2))
            for t in tables:
                if tab and _slug(tab) != t.name:
                    continue
                for c, meta in t.meta.items():
                    if meta.get("orig", "").strip().lower() == col.strip().lower() or c == _slug(col):
                        meta["role"] = role
                        t.roles.setdefault(role, c)
    # "các dòng cùng **STT** là một đơn": that column groups the lines of an order; numbers restart
    # in every monthly sheet, so the order key is sheet + number
    for m in re.finditer(r"các dòng cùng \*\*([^*]+)\*\* là một đơn", text):
        col = m.group(1).strip().lower()
        for t in tables:
            c = next((c for c, meta in t.meta.items() if meta.get("orig", "").strip().lower() == col or c == _slug(col)), None)
            if c and not t.roles.get("order_id"):
                key = (t.df["_sheet"].astype(str) + "#" if "_sheet" in t.df else "") + t.df[c].astype(str)
                t.df["order_key"] = key.where(t.df[c].notna())
                t.meta["order_key"] = {"orig": "order_key", "desc": f"order key = sheet + {c} (lines of one order)", "role": "order_id"}
                t.roles["order_id"] = "order_key"
    for t in tables:  # in a payment ledger the time column is the payment date, not a sale date
        if t.roles.get("pay_amount") and t.roles.get("sale_date") and not t.roles.get("pay_date"):
            t.roles["pay_date"] = t.roles.pop("sale_date")
            t.roles.pop("sale_date_parsed", None)
