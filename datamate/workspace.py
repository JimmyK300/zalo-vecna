"""Load a business folder (CSV / Excel / SQLite / JSON / docs) into DataFrames
and build a compact schema profile for the LLM."""
from __future__ import annotations

import datetime
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .graph import build_graph
from .items import build_lines
from .primitives import build_orders
from .recipes import build_debt_ledger
from .semantic_model import build_semantic_model
from .semantics import apply_conventions, enrich_table, link_orders, parse_readme

TABLE_EXT = {".csv", ".tsv", ".txt_table", ".xlsx", ".xlsm", ".xls", ".sqlite", ".sqlite3", ".db", ".json", ".parquet"}
DOC_EXT = {".md", ".txt", ".rst"}
MAX_DOC_CHARS = 12000
MAX_PROFILE_CHARS = 40000


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).replace("đ", "d").replace("Đ", "D")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return text or "t"


def _demojibake(name: str) -> str:
    """UTF-8 file names unzipped as cp1252/latin-1 ("BÃ¡o cÃ¡o") -> original; else unchanged."""
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = name.encode(enc).decode("utf-8")
        except UnicodeError:
            continue
        if fixed != name:
            return fixed
    return name


def _read_csv(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:  # noqa: BLE001 - try the next encoding
            last = e
    raise last  # type: ignore[misc]


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    """Excel sheets often have title rows above the real header. Pick the first
    row that is mostly filled with non-numeric strings."""
    width = raw.shape[1]
    for i in range(min(max_scan, len(raw))):
        row = raw.iloc[i]
        filled = row.notna().sum()
        strings = sum(isinstance(v, str) and not re.fullmatch(r"[\d.,\-\s]+", v) for v in row)
        if width and filled >= max(2, 0.6 * width) and strings >= 0.6 * filled:
            return i
    return 0


THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^-?\d{1,3}(?:\.\d{3})+$|^-?\d+(?:\.\d+)?$")
SHORT_MONEY_RE = re.compile(r"^(-?\d+(?:[.,]\d+)?)\s*(k|tr|triệu)$", re.I)
DATE_NAME_RE = re.compile(r"ngày|ngay|date|hạn|han_|thời gian|thoi gian|time|giờ|gio", re.I)
VN_MONTH_RE = re.compile(r"^(\d{1,2})\s*thg\s*(\d{1,2})(?:,?\s*(\d{4}))?", re.I)
DMY_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$")
YMD_RE = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?")


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def parse_money(v) -> float:
    """Vietnamese hand-written amounts: '38.000đ', '2tr300' (2.3M), '71,114tr', '1,92tr', '3696k',
    '2k5' (2,500), '1.250.000', '84,000'. NaN when it is not an amount."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    t = re.sub(r"\s+", "", str(v).strip().lower())
    t = re.sub(r"(đ|vnđ|vnd|dong|đồng)$", "", t)
    m = re.fullmatch(r"(-?\d+)(tr|k)(\d{1,3})", t)
    if m:
        return float(f"{m.group(1)}.{m.group(3)}") * (1e6 if m.group(2) == "tr" else 1e3)
    m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)(tr|triệu|trieu|k|nghìn|nghin|ngàn)", t)
    if m:
        return float(m.group(1).replace(",", ".")) * (1e6 if m.group(2) in ("tr", "triệu", "trieu") else 1e3)
    if re.fullmatch(r"-?\d{1,3}([.,])\d{3}(\1\d{3})*", t):
        return float(re.sub(r"[.,]", "", t))
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return float(t)
    return np.nan


def _mixed_money(s: pd.Series) -> pd.Series | None:
    """A column of mixed amount styles ('2tr300', '38.000đ', '3696k') -> numbers, only if ~all
    values parse and some carry a unit (so code columns are never touched)."""
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""]
    if v.empty or v.str.match(r"^0\d").any():
        return None
    parsed = v.map(parse_money)
    has_unit = v.str.contains(r"\d\s*(?:tr|k|đ)", case=False, regex=True).mean()
    if parsed.notna().mean() >= 0.97 and has_unit >= 0.2:
        return s.map(parse_money)
    return None


def _text_numbers(s: pd.Series) -> pd.Series | None:
    """'84,000' / '1.200.000' / '-15,000' text -> numbers, only when (nearly) the whole
    column follows one thousands style. Codes with leading zeros are left alone."""
    out = _text_numbers_uniform(s)
    if out is None:
        return _mixed_money(s)
    lost = out.isna() & s.notna() & (s.astype(str).str.strip() != "")
    if lost.any():  # the odd '1tr245' among '45.000đ' values must not silently become NaN
        out = out.copy()
        out[lost] = s[lost].map(parse_money)
    return out


def _text_numbers_uniform(s: pd.Series) -> pd.Series | None:
    vals = s.dropna().astype(str).str.strip().str.replace(r"\s*(đ|vnd|vnđ|d)$", "", regex=True, flags=re.I)
    vals = vals[vals != ""]
    if vals.empty or vals.str.match(r"^0\d").any():
        return None
    short = vals.str.match(SHORT_MONEY_RE)  # '41k', '1.2tr'
    plain = vals[~short]
    ok = plain.str.match(THOUSANDS_RE)
    if (ok.sum() + short.sum()) / len(vals) < 0.98 or short.all() and len(vals) < 3:
        return None
    has_comma = plain.str.contains(",").any()
    has_dot_groups = plain.str.match(r"^-?\d{1,3}(?:\.\d{3})+$").any()
    if has_comma and has_dot_groups:
        return None
    def conv(v):
        if pd.isna(v):
            return np.nan
        t = re.sub(r"\s*(đ|vnd|vnđ|d)$", "", str(v).strip(), flags=re.I)
        if t == "":
            return np.nan
        m = SHORT_MONEY_RE.match(t)
        if m:
            return float(m.group(1).replace(",", ".")) * (1000 if m.group(2).lower() == "k" else 1_000_000)
        t = t.replace(",", "") if has_comma else (t.replace(".", "") if has_dot_groups else t)
        try:
            return float(t)
        except ValueError:
            return np.nan
    return s.map(conv)


def _parse_date(v, default_year: int | None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
        return pd.Timestamp(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v)) if 20000 < v < 80000 else pd.NaT
    t = str(v).strip()
    if re.fullmatch(r"\d{5}(\.\d+)?", t):
        return _parse_date(float(t), default_year)
    m = YMD_RE.match(t)
    if m:
        y, mo, d, hh, mm, ss = m.groups()
        return _ts(int(y), int(mo), int(d), hh, mm, ss)
    m = DMY_RE.match(t)
    if m:
        d, mo, y, hh, mm, ss = m.groups()
        y = int(y) + (2000 if len(y) == 2 else 0) if y else default_year
        return _ts(y, int(mo), int(d), hh, mm, ss) if y else pd.NaT
    m = VN_MONTH_RE.match(t)
    if m:
        d, mo, y = m.groups()
        y = int(y) if y else default_year
        return _ts(y, int(mo), int(d)) if y else pd.NaT
    return pd.NaT


def _ts(y, mo, d, hh=None, mm=None, ss=None):
    try:
        return pd.Timestamp(year=y, month=mo, day=d, hour=int(hh or 0), minute=int(mm or 0), second=int(ss or 0))
    except ValueError:
        return pd.NaT


def clean_frame(df: pd.DataFrame, default_year: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Light, rule-free cleaning. Raw columns are kept; derived columns are added:
      <col>  text numbers converted in place (noted in the profile)
      <col>__date  parsed datetime for messy date columns (dd/mm/yyyy, '1 thg 3 2025', Excel serials)"""
    notes = []
    for c in list(df.columns):
        s = df[c]
        if _is_text(s):
            conv = _text_numbers(s)
            if conv is not None and not DATE_NAME_RE.search(str(c)):
                df[c] = conv
                notes.append(f"{c}: text numbers like '{s.dropna().astype(str).iloc[0]}' converted to numbers")
                continue
        if (_is_text(s) or pd.api.types.is_numeric_dtype(s)) and DATE_NAME_RE.search(str(c)) \
                and not pd.api.types.is_datetime64_any_dtype(s):
            parsed = s.map(lambda v: _parse_date(v, default_year))
            rate = parsed.notna().sum() / max(1, s.notna().sum())
            if rate >= 0.5:
                df[f"{c}__date"] = pd.to_datetime(parsed)
                notes.append(f"{c}: mixed date formats parsed into column '{c}__date' ({rate:.0%} parsed)")
    return df, notes


def _read_excel(path: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for sheet, raw in pd.read_excel(path, sheet_name=None, header=None).items():
        raw = raw.dropna(how="all").dropna(axis=1, how="all")
        if raw.empty:
            continue
        h = _detect_header_row(raw)
        df = raw.iloc[h + 1:].copy()
        cols, seen = [], {}
        for j, c in enumerate(raw.iloc[h]):
            name = str(c).strip() if pd.notna(c) else f"col_{j}"
            seen[name] = seen.get(name, 0) + 1
            cols.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
        df.columns = cols
        df = df.reset_index(drop=True).infer_objects()
        for c in df.columns:  # numeric-looking object columns -> numbers
            if _is_text(df[c]):
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() >= 0.9 * df[c].notna().sum() > 0:
                    df[c] = conv
        out[sheet] = df
    return out


def _read_sqlite(path: Path) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
        return {n: pd.read_sql_query(f'SELECT * FROM "{n}"', con) for n in names}
    finally:
        con.close()


@dataclass
class TableInfo:
    name: str
    source: str
    df: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    shared: bool = False
    meta: dict[str, dict] = field(default_factory=dict)   # column -> {orig, desc, role}
    roles: dict[str, str] = field(default_factory=dict)   # role -> column


def _year_hint(*texts: str) -> int | None:
    for t in texts:
        m = re.search(r"(20\d\d)", t or "")
        if m:
            return int(m.group(1))
    return None


@dataclass
class Workspace:
    root: Path
    tables: dict[str, TableInfo] = field(default_factory=dict)
    docs: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    profile: str = ""
    rules: set[str] | None = None
    graph: object = None
    debt: object = None
    lines: object = None
    semantic: object = None

    def _add(self, name: str, source: str, df: pd.DataFrame, clean: bool = True, shared: bool = False) -> None:
        base, i = _slug(name), 2
        key = base
        while key in self.tables:
            key, i = f"{base}_{i}", i + 1
        df.columns = [str(c) for c in df.columns]
        notes = []
        if clean:
            df, notes = clean_frame(df, _year_hint(source))
        self.tables[key] = TableInfo(key, source, df, notes, shared)

    def _add_sheets(self, stem: str, rel: str, sheets: dict[str, pd.DataFrame]) -> None:
        """Monthly sheets with the same columns become one table with a `_sheet` column."""
        groups: dict[tuple, list[str]] = {}
        for s, df in sheets.items():
            groups.setdefault(tuple(map(str, df.columns)), []).append(s)
        for cols, names in groups.items():
            if len(names) > 1:
                parts = []
                for s in names:
                    df, _ = clean_frame(sheets[s].copy(), _year_hint(s, stem))
                    df.insert(0, "_sheet", s)
                    parts.append(df)
                merged = pd.concat(parts, ignore_index=True)
                _, notes = clean_frame(sheets[names[0]].head(200).copy(), _year_hint(names[0], stem))
                key = _slug(stem if len(groups) == 1 else f"{stem}_{names[0]}")
                self._add(key, f"{rel} (sheets: {', '.join(names)})", merged, clean=False)
                self.tables[list(self.tables)[-1]].notes = notes + [
                    "_sheet: source sheet name; dates without a year take the sheet's year"]
            else:
                s = names[0]
                self._add(stem if len(sheets) == 1 else f"{stem}_{s}", f"{rel}#{s}", sheets[s])

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Fresh in-memory DuckDB with every table registered (one per question)."""
        con = duckdb.connect(":memory:")
        for t in self.tables.values():
            con.register(t.name, t.df)
        return con


def load_workspace(root: Path, shared: dict[str, pd.DataFrame] | None = None,
                   rules: set[str] | None = None) -> Workspace:
    ws = Workspace(root=root)
    ws.rules = rules
    for name, df in (shared or {}).items():
        ws._add(name, f"{name} (shared, all businesses)", df.copy(), clean=False, shared=True)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for p in files:
        ext = p.suffix.lower()
        rel = _demojibake(str(p.relative_to(root.parent if root.is_file() else root)))
        stem = _demojibake(p.stem)
        try:
            if ext in (".csv", ".tsv"):
                ws._add(stem, rel, _read_csv(p))
            elif ext in (".xlsx", ".xlsm", ".xls"):
                ws._add_sheets(stem, rel, _read_excel(p))
            elif ext in (".sqlite", ".sqlite3", ".db"):
                for t, df in _read_sqlite(p).items():
                    ws._add(t, f"{rel}#{t}", df)
            elif ext == ".parquet":
                ws._add(stem, rel, pd.read_parquet(p))
            elif ext == ".json":
                try:
                    ws._add(stem, rel, pd.read_json(p))
                except ValueError:
                    ws.docs[rel] = p.read_text(encoding="utf-8", errors="replace")[:MAX_DOC_CHARS]
            elif ext in DOC_EXT:
                ws.docs[rel] = p.read_text(encoding="utf-8", errors="replace")[:MAX_DOC_CHARS]
        except Exception as e:  # noqa: BLE001 - keep going, tell the LLM
            ws.errors.append(f"{rel}: {type(e).__name__}: {e}")
    readme = {}
    for name, text in ws.docs.items():
        if Path(name).name.lower().startswith("readme"):
            readme.update(parse_readme(text))
    own = [t for t in ws.tables.values() if not t.shared]
    for t in own:
        enrich_table(t, readme, rules)
    apply_conventions(own, ws.docs)
    link_orders(own, rules)
    try:
        for name, odf in build_orders(ws).items():
            desc = {"order_id": "order id", "order_date": "order date (period by this date)",
                    "customer": "customer name", "customer_code": "customer code", "is_walkin": "walk-in, not a customer",
                    "status": "one status per order: hoan_thanh / huy / tra_hang / dang_giao",
                    "in_revenue": "counts in revenue (rulebook §1)", "n_lines": "number of lines",
                    "revenue_lines": "order revenue = sum of line amounts (duplicate lines once); "
                                     "no order-level discount, VAT or delivery fee"}
            ws.tables[name] = TableInfo(
                name, "derived: one row per order from " + ", ".join(odf.attrs.get("sources", [])), odf,
                notes=["use this for counts, revenue, cancellation rate, top customers by orders"],
                meta={c: {"orig": c, "desc": desc.get(c, "constant within the order"), "role": None} for c in odf.columns})
    except Exception as e:  # noqa: BLE001 - derived table is a convenience; never block loading
        ws.errors.append(f"orders table not built: {type(e).__name__}: {e}")
    if rules is not None:
        off = sorted({"in_revenue", "walkin", "dup_line", "order_fill"} - set(rules))
        if off:
            ws.errors.append("rule helpers disabled because the rulebook wording changed: " + ", ".join(off)
                             + " -> apply the rulebook text yourself")
    try:
        lines = build_lines(ws)
        ws.lines = lines
        if lines is not None:
            ws.tables["lines"] = TableInfo("lines", "derived: one row per sale line (orders joined)", lines,
                                           notes=["use for best sellers, item revenue/units, item-group share, profit"],
                                           meta={c: {"orig": c, "desc": "", "role": None} for c in lines.columns})
    except Exception as e:  # noqa: BLE001
        ws.lines = None
        ws.errors.append(f"lines table not built: {type(e).__name__}: {e}")
    try:
        ws.debt = build_debt_ledger(ws)
    except Exception as e:  # noqa: BLE001
        ws.debt = None
        ws.errors.append(f"debt ledger not built: {type(e).__name__}: {e}")
    try:
        ws.graph = build_graph(ws)
    except Exception as e:  # noqa: BLE001
        ws.errors.append(f"relationship graph not built: {type(e).__name__}: {e}")
    try:
        ws.semantic = build_semantic_model(ws)
    except Exception as e:  # noqa: BLE001 - advisory only
        ws.semantic = None
        ws.errors.append(f"semantic model not built: {type(e).__name__}: {e}")
    ws.profile = build_profile(ws)
    return ws


def _fmt(v) -> str:
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def build_profile(ws: Workspace) -> str:
    parts = ["## Tables (DuckDB SQL; column names are ASCII snake_case, no quotes needed; the original "
             "name and README meaning follow each column; in Python use tables['name'] or con)"]
    for t in ws.tables.values():
        df = t.df
        parts.append(f"\n### {t.name}  (source: {t.source}, {len(df)} rows)")
        if t.shared:
            parts.append(f"- columns: {', '.join(map(str, df.columns))}")
            continue
        if t.roles:
            parts.append("  roles: " + ", ".join(f"{r}={c}" for r, c in t.roles.items()))
        for n in t.notes:
            parts.append(f"  * derived: {n}")
        for c in df.columns:
            s = df[c]
            nn = s.notna().sum()
            samples = ", ".join(_fmt(v) for v in s.dropna().unique()[:4])
            extra = ""
            if (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)) and nn:
                extra = f" min={_fmt(s.min())} max={_fmt(s.max())}"
            elif _is_text(s) and nn:
                # show values of different shapes, so messy formats are visible
                shapes = s.dropna().astype(str).map(lambda v: re.sub(r"\d", "9", re.sub(r"[^\W\d_]+", "a", v)))
                rare = s.dropna()[~shapes.duplicated()].astype(str).unique()[:6]
                if len(rare) > 1:
                    extra = f" formats: {' | '.join(_fmt(v) for v in rare)}"
            m = t.meta.get(c)
            label = c if not m else f'{c} ← "{m["orig"]}"' + (f" ({m['desc'][:80]})" if m["desc"] else "")
            if m is None and c not in ("_sheet",):
                label = f"{c} (helper)"
            parts.append(f"- {label} [{s.dtype}] nulls={len(s) - nn} distinct={s.nunique()}{extra} e.g. {samples}")
    if ws.docs:
        parts.append("\n## Documents")
        for name, text in ws.docs.items():
            parts.append(f"\n### {name}\n{text}")
    if ws.errors:
        parts.append("\n## Files that failed to load (inspect with Python if relevant)")
        parts += [f"- {e}" for e in ws.errors]
    text = "\n".join(parts)
    if len(text) > MAX_PROFILE_CHARS:
        text = text[:MAX_PROFILE_CHARS] + "\n...[profile truncated; explore with SQL]"
    return text


class WorkspaceCache:
    """Load each business folder once, shared across worker threads."""

    def __init__(self, shared: dict[str, pd.DataFrame] | None = None, rules: set[str] | None = None):
        self.rules = rules
        self._cache: dict[Path, Workspace] = {}
        self._locks: dict[Path, threading.Lock] = {}
        self._lock = threading.Lock()
        self.shared = shared or {}

    def get(self, root: Path) -> Workspace:
        root = root.resolve()
        with self._lock:  # one lock per business, so different businesses load in parallel
            lock = self._locks.setdefault(root, threading.Lock())
        with lock:
            if root not in self._cache:
                self._cache[root] = load_workspace(root, self.shared, self.rules)
            return self._cache[root]
