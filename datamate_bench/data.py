"""Raw-table ingestion, provenance and time-bounded read-only SQL tools.

This intentionally makes no claim to canonicalize business semantics. Source
headers/cells survive ingestion; the model must apply the current rulebook.
"""

import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

FORMAT_VERSION = "raw-v1"


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", str(text).lower().replace("đ", "d"))
                   if unicodedata.category(c) != "Mn")


def number(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().lower().replace("\u00a0", "").replace(" ", "")
    short_million=re.fullmatch(r'([+-]?\d+)tr(\d{1,3})',text)
    if short_million:
        sign=-1 if short_million[1].startswith('-') else 1
        return int(short_million[1])*1_000_000+sign*int(short_million[2].ljust(3,'0'))*1000
    # Do not strip arbitrary text: an identifier/product name is not a number.
    match = re.fullmatch(r"([+-]?[\d.,]+)(tr|triệu|k|đ|₫|vnd|%)?", text)
    if not match:
        return None
    text, unit = match.groups()
    multiplier = 1_000_000 if unit in {"tr", "triệu"} else 1000 if unit == "k" else 1
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        text = text.replace(thousands, "").replace(decimal, ".")
    elif re.fullmatch(r"[+-]?[1-9]\d{0,2}(?:[.,]\d{3})+", text) and unit not in {"tr","triệu","%"}:
        text = text.replace(",", "").replace(".", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        result = float(text) * multiplier
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def source_year(text):
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", str(text))
    return int(match.group(1)) if match else None


def iso_date(value, year=None):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and 20000 < value < 80000:
        return (dt.datetime(1899, 12, 30) + dt.timedelta(days=value)).date().isoformat()
    text = str(value).strip()
    # CSV and SQLite exports often serialize the Excel day number as text.
    if re.fullmatch(r"\d{5}(?:\.\d+)?", text) and 20000 < float(text) < 80000:
        return (dt.datetime(1899, 12, 30) + dt.timedelta(days=float(text))).date().isoformat()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        try:
            return dt.date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            return None
    text = re.sub(r"\s+(?:thg|tháng|thang)\s+", "/", text, flags=re.I)
    text = re.sub(r"\s+(\d{4})$", r"/\1", text)
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y %H:%M:%S"]:
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    if year and re.fullmatch(r"\d{1,2}/\d{1,2}", text):
        try:
            day, month = map(int, text.split("/"))
            return dt.date(int(year), month, day).isoformat()
        except (ValueError, TypeError):
            return None
    return None


def identifier(text):
    result = re.sub(r"[^a-z0-9]+", "_", fold(text)).strip("_") or "column"
    return "c_" + result if result[0].isdigit() else result


def headers_for(row):
    used, headers = set(), []
    for i, value in enumerate(row):
        base = identifier(value if value is not None else f"column_{i + 1}")
        if base.startswith("_source"):
            base = "data" + base
        key, suffix = base, 2
        while key in used or key in {"source_file", "source_sheet", "source_row", "source_year"}:
            key, suffix = f"{base}_{suffix}", suffix + 1
        used.add(key)
        headers.append(key)
    return headers


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def cell(value):
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bytes)) or value is None:
        return value
    return str(value)


def sheet_rows(path):
    """Yield source sheet, headers, (Excel/csv row number, original cells)."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            yield "csv", header, enumerate(reader, 2)
        return
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook:
            rows = iter(sheet.iter_rows(values_only=True))
            head = list(itertools.islice(rows, 10))
            if not head:
                continue

            def header_score(row):
                strings = [v for v in row if isinstance(v, str) and v.strip()]
                return sum(number(v) is None and iso_date(v) is None
                           and bool(re.search(r"[^\W\d_]", v)) for v in strings)

            index = max(range(len(head)), key=lambda i: (header_score(head[i]), -i))
            header = list(head[index])
            while header and header[-1] is None:
                header.pop()
            if not header:
                continue
            remaining = itertools.chain(head[index + 1:], rows)
            yield sheet.title, header, enumerate(remaining, index + 2)
    finally:
        workbook.close()


def cache_key(business, calendar):
    h = hashlib.sha256(FORMAT_VERSION.encode())
    files = sorted(p for p in business.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".md", ".csv", ".xlsx", ".sqlite", ".db"})
    for path in [*files, calendar]:
        h.update(str(path.relative_to(business) if path.is_relative_to(business) else "calendar.csv").encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest(), files


def build_database(business, calendar, cache_dir):
    start = time.perf_counter()
    business, calendar, cache_dir = Path(business).resolve(), Path(calendar).resolve(), Path(cache_dir)
    key, files = cache_key(business, calendar)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.sqlite"
    manifest_path = path.with_suffix(".json")
    hit = path.exists() and manifest_path.exists()
    if not hit:
        # An interrupted build is overwritten; only a completed manifest marks validity.
        if path.exists():
            path.unlink()
        con = sqlite3.connect(path)
        tables, signatures = {}, {}

        def ingest(relative, sheet, raw_headers, rows, fixed_name=None):
            headers = headers_for(raw_headers)
            if not headers:
                return
            signature = tuple(str(h) for h in raw_headers)
            if fixed_name:
                signature = (fixed_name, *signature)
            table = signatures.get(signature)
            if table is None:
                table = fixed_name or f"t{len(tables) + 1:02d}"
                signatures[signature] = table
                columns = [*headers, "_source_file", "_source_sheet", "_source_row", "_source_year"]
                con.execute(f"CREATE TABLE {quote(table)} ({','.join(quote(c) for c in columns)})")
                tables[table] = {"columns": headers, "original_headers": list(map(str, raw_headers)),
                                 "rows": 0, "sources": [], "sample": []}
            info = tables[table]
            info["sources"].append({"file": relative, "sheet": sheet})
            year = source_year(sheet) or source_year(relative)
            statement = f"INSERT INTO {quote(table)} VALUES ({','.join('?' for _ in range(len(headers) + 4))})"
            batch = []
            for row_number, values in rows:
                values = list(values)
                if not any(v is not None and v != "" for v in values):
                    continue
                # Never silently truncate populated cells beyond the detected header.
                if any(v is not None and v != "" for v in values[len(headers):]):
                    raise ValueError(f"Populated cells beyond header: {relative}/{sheet}/{row_number}")
                values = (values + [None] * len(headers))[:len(headers)]
                values = [cell(v) for v in values]
                batch.append([*values, relative, sheet, row_number, year])
                info["rows"] += 1
                if len(info["sample"]) < 2:
                    info["sample"].append(values)
                if len(batch) >= 2000:
                    con.executemany(statement, batch)
                    batch.clear()
            if batch:
                con.executemany(statement, batch)

        try:
            for file in files:
                relative = str(file.relative_to(business))
                if file.suffix.lower() in {".csv", ".xlsx"}:
                    for sheet, headers, rows in sheet_rows(file):
                        ingest(relative, sheet, headers, rows)
                elif file.suffix.lower() in {".sqlite", ".db"}:
                    source = sqlite3.connect(file.as_uri() + "?mode=ro", uri=True)
                    try:
                        names = [v[0] for v in source.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                        for name in names:
                            cursor = source.execute(f"SELECT * FROM {quote(name)}")
                            ingest(relative, name, [c[0] for c in cursor.description], enumerate(cursor, 1))
                    finally:
                        source.close()
            for sheet, headers, rows in sheet_rows(calendar):
                ingest("calendar.csv", sheet, headers, rows, "calendar")
            con.commit()
            manifest = {"version": FORMAT_VERSION, "fingerprint": key, "tables": tables,
                        "business_id": business.name,
                        "warnings": ["Raw rows, not business-cleaned facts. No forward-fill or deduplication is applied.",
                                     "Identical headers share a table. Check _source_file/_source_sheet before combining.",
                                     "Excel header detection is heuristic; inspect original_headers and source samples."]}
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        finally:
            con.close()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return path, manifest, {"cache_hit": hit, "ingest_seconds": time.perf_counter() - start,
                            "tables": len(manifest["tables"]),
                            "raw_rows": sum(t["rows"] for t in manifest["tables"].values())}


class ReadOnlySQL:
    def __init__(self, path, seconds=5, row_limit=100):
        self.con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.con.enable_load_extension(False)
        self.con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        self.con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 64_000)
        self.con.create_function("num", 1, number, deterministic=True)
        self.con.create_function("fold", 1, lambda x: fold(x) if x is not None else None, deterministic=True)
        self.con.create_function("iso_date", 1, iso_date, deterministic=True)
        self.con.create_function("iso_date", 2, iso_date, deterministic=True)
        self.con.create_function("source_year", 1, source_year, deterministic=True)
        self.con.execute("PRAGMA query_only=ON")
        self.con.set_authorizer(self.authorize)
        self.seconds, self.row_limit = seconds, row_limit

    @staticmethod
    def authorize(action, arg1, arg2, database, trigger):
        if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (arg2 or arg1 or "").lower() in {
                "load_extension", "readfile", "writefile", "eval", "fts3_tokenizer"} else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def query(self, sql, seconds=None):
        deadline = time.monotonic() + min(self.seconds, seconds or self.seconds)
        self.con.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        start = time.perf_counter()
        try:
            cursor = self.con.execute(sql)
            if cursor.description is None:
                raise ValueError("Only SELECT queries are supported")
            rows = cursor.fetchmany(self.row_limit + 1)
            return {"columns": [v[0] for v in cursor.description], "rows": rows[:self.row_limit],
                    "truncated": len(rows) > self.row_limit,
                    "seconds": time.perf_counter() - start}
        finally:
            self.con.set_progress_handler(None, 0)

    def close(self):
        self.con.close()
