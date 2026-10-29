import os
import re
import sqlite3
import datetime
import pandas as pd
import numpy as np
import duckdb
from unidecode import unidecode
from typing import Dict, List, Tuple, Any, Optional

# Strict pattern for numbers/currency values
CURRENCY_PATTERN = re.compile(
    r'^\s*[-+]?\s*(\d{1,3}([.,]\d{3})*|\d+)([.,]\d+)?\s*(k|tr|trieu|triệu|d|đ|vnd|đồng)?\s*$',
    re.IGNORECASE
)

# Text columns that must NEVER be converted to numeric
TEXT_KEYWORDS = [
    'ten', 'name', 'hang', 'san_pham', 'product', 'item', 'mon',
    'dien_giai', 'mo_ta', 'desc', 'description',
    'ma_', 'so_don', 'so_phieu', 'hd', 'code', 'id',
    'khach', 'customer', 'dien_thoai', 'sdt', 'phone', 'dia_chi',
    'ghi_chu', 'note', 'trang_thai', 'status', 'kenh', 'channel',
    'bien_so', 'xe', 'lai_xe', 'driver', 'dvt', 'don_vi', 'unit',
    'nhom', 'group', 'category', 'loai', 'brand', 'hang_sx', 'size'
]

def is_date_column(col: str) -> bool:
    """Checks if column is a date or time column."""
    if any(ex in col for ex in ['so_ngay', 'ky_han', 'han_no']):
        return False
    return any(dk in col for dk in ['ngay', 'thoi_gian', 'date', 'gio', 'time'])

def is_numeric_column(col: str) -> bool:
    """Checks if column should be numeric."""
    if is_date_column(col):
        return False
    tokens = col.split('_')
    numeric_tokens = {'tien', 'tong', 'gia', 'sl', 'amount', 'no', 'diem', 'ton', 'cost'}
    if any(t in numeric_tokens for t in tokens):
        return True
    numeric_substrings = [
        'so_luong', 'thanh_tien', 'chi_phi', 'don_gia', 'chiet_khau',
        'giam_gia', 'he_so', 'ti_le', 'ky_han', 'so_ngay'
    ]
    return any(sub in col for sub in numeric_substrings)

def generate_table_aliases(tbl_name: str) -> List[str]:
    """Generates comprehensive variations of table names (snake_case, compacted, month padding, without sheet)."""
    aliases = set()
    no_sheet = re.sub(r'_sheet\d+$', '', tbl_name)
    if no_sheet != tbl_name:
        aliases.add(no_sheet)
        aliases.add(no_sheet.replace('_', ''))
        
    compact_common = tbl_name.replace('so_ban_hang', 'sobanhang').replace('ban_hang', 'banhang') \
                             .replace('chi_tiet', 'chitiet').replace('hoa_don', 'hoadon') \
                             .replace('danh_muc', 'danhmuc').replace('quan_ly', 'quanly') \
                             .replace('so_tay', 'sotay').replace('so_sua_xe', 'sosuaxe') \
                             .replace('danh_sach', 'danhsach').replace('bao_cao', 'baocao')
    aliases.add(compact_common)
    aliases.add(re.sub(r'_sheet\d+$', '', compact_common))
    aliases.add(tbl_name.replace('_', ''))

    cur_aliases = list(aliases) + [tbl_name]
    for name in cur_aliases:
        # Month padding: _t01_ to _t1_, _t05_ to _t5_ and vice versa
        m_pad = re.search(r'_t0([1-9])_', name)
        if m_pad:
            aliases.add(re.sub(r'_t0([1-9])_', r'_t\1_', name))
        m_unpad = re.search(r'_t([1-9])_', name)
        if m_unpad:
            aliases.add(re.sub(r'_t([1-9])_', r'_t0\1_', name))

        # Year duplication e.g. banhang_2026_t06_2026 -> banhang_t06_2026
        m_dup = re.search(r'^(.*?)_\d{4}_(t\d{1,2}_\d{4})$', name)
        if m_dup:
            base, month_yr = m_dup.group(1), m_dup.group(2)
            aliases.add(f"{base}_{month_yr}")

        # Short sheet name e.g. t05_2026, t5_2026
        m_sheet = re.search(r'(t\d{1,2}_\d{4})$', name)
        if m_sheet:
            s_name = m_sheet.group(1)
            aliases.add(s_name)
            if 't0' in s_name:
                aliases.add(s_name.replace('t0', 't'))
            else:
                aliases.add(re.sub(r't([1-9])', r't0\1', s_name))

    aliases.discard(tbl_name)
    return [a for a in aliases if a]


class DataIngestionEngine:
    """
    Unified Ingestion Engine:
    Loads CSV, Excel (.xlsx, multi-sheet), and SQLite tables into an in-memory DuckDB database.
    - Smart Header Detection (handles banners & headers on row 0, 1, 2, 3...)
    - Currency & Numeric Shorthand Cleaning (converts '29k', '43.000đ', '5tr520', '2 cái' to float)
    - Safe Column Type Guard: Text/product/order columns are protected from numeric conversion
    - Date Normalization: Unifies mixed dates to ISO 'YYYY-MM-DD' strings
    - Column Name Sanitization (clean snake_case for robust SQL queries without space/diacritic errors)
    - Auto Table Aliasing: Creates views for common table/sheet variations (t5 vs t05)
    """

    def __init__(self):
        pass

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Sanitizes names to valid ASCII snake_case SQL identifiers, preserving CamelCase transitions."""
        s_raw = str(name)
        s_camel = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s_raw)
        s_camel = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s_camel)
        s = unidecode(s_camel).lower().strip()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')

    @classmethod
    def clean_currency_value(cls, val: Any) -> Any:
        """Converts Vietnamese currency/number strings to float (e.g. '29k', '43.000đ', '5tr520', '2 cái')."""
        if pd.isna(val) or val is None:
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        s = str(val).strip()
        if not s or s.lower() in ['-', 'none', 'nan', 'null']:
            return np.nan

        # Strip currency symbols first
        s_clean_sym = re.sub(r'\s*(?:đ|d|vnd|đồng)\s*$', '', s, flags=re.IGNORECASE).strip()
        s_lower = s_clean_sym.lower().replace(' ', '')
        is_neg = '-' in s_lower

        # Pattern 1: XtrY (e.g. 5tr520, 1tr5) or Xtr
        tr_match = re.search(r'(\d+)(?:tr|trieu|triệu)(\d*)', s_lower)
        if tr_match:
            millions = float(tr_match.group(1))
            rem_str = tr_match.group(2)
            if rem_str:
                rem = float(rem_str.ljust(6, '0')[:6])
                res = millions * 1_000_000 + rem
            else:
                res = millions * 1_000_000
            return -abs(res) if is_neg else res

        # Pattern 2: XkY (e.g. 2k5, 20k5) or Xk
        k_match = re.search(r'(\d+)(?:k)(\d*)', s_lower)
        if k_match:
            thousands = float(k_match.group(1))
            rem_str = k_match.group(2)
            if rem_str:
                rem = float(rem_str.ljust(3, '0')[:3])
                res = thousands * 1_000 + rem
            else:
                res = thousands * 1_000
            return -abs(res) if is_neg else res

        # Pattern 3: Number with unit (e.g. "2 cái", "5 viên", "1.5 kg", "1 thùng", "2 cuộn")
        qty_match = re.match(r'^[-+]?\s*(\d+([.,]\d+)?)\s*[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]+$', s_clean_sym)
        if qty_match:
            try:
                num = float(qty_match.group(1).replace(',', '.'))
                return -abs(num) if is_neg else num
            except ValueError:
                pass

        # Pattern 4: Standard currency/number string e.g. "43.000", "1.048.000", "25000", "12.5"
        s_digits = re.sub(r'[^\d,\.]', '', s_lower)
        if not s_digits:
            return np.nan

        if '.' in s_digits and ',' not in s_digits:
            parts = s_digits.split('.')
            if len(parts[-1]) == 3 or len(parts) > 2:
                s_digits = s_digits.replace('.', '')
        elif ',' in s_digits and '.' not in s_digits:
            parts = s_digits.split(',')
            if len(parts[-1]) == 3 or len(parts) > 2:
                s_digits = s_digits.replace(',', '')
            else:
                s_digits = s_digits.replace(',', '.')
        elif '.' in s_digits and ',' in s_digits:
            s_digits = s_digits.replace('.', '').replace(',', '.')

        try:
            res = float(s_digits)
            return -abs(res) if is_neg else res
        except ValueError:
            return np.nan

    @classmethod
    def normalize_date_value(cls, val: Any, default_year: int = 2026) -> Any:
        """Normalizes various date inputs into standard 'YYYY-MM-DD' string."""
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, (datetime.datetime, datetime.date, pd.Timestamp)):
            return val.strftime("%Y-%m-%d")
        
        # Check Excel serial date (e.g. 45658 or "45658")
        if isinstance(val, (int, float, np.integer, np.floating)):
            if 30000 <= val <= 60000:
                try:
                    dt = pd.to_datetime(val, unit='D', origin='1899-12-30')
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
        elif isinstance(val, str) and val.strip().isdigit():
            val_int = int(val.strip())
            if 30000 <= val_int <= 60000:
                try:
                    dt = pd.to_datetime(val_int, unit='D', origin='1899-12-30')
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        s = str(val).strip()
        if not s or s.lower() in ['none', 'nan', 'null', '-']:
            return None

        # Check ISO YYYY-MM-DD
        if len(s) >= 10 and s[4] == '-' and s[7] == '-':
            return s[:10]

        parts = s.split('/')
        if len(parts) == 3:
            try:
                d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                return f"{y:04d}-{m:02d}-{d:02d}"
            except ValueError:
                pass
        elif len(parts) == 2:
            try:
                d, m = int(parts[0]), int(parts[1])
                return f"{default_year:04d}-{m:02d}-{d:02d}"
            except ValueError:
                pass

        # Vietnamese month notation e.g. "1 thg 1 2025" or "1 tháng 1 2025"
        s_clean = re.sub(r'thg|tháng', '', s, flags=re.IGNORECASE).strip()
        try:
            dt = pd.to_datetime(s_clean, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return s

    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans columns, normalizes dates, protects text columns, and auto-converts currency/numbers."""
        clean_cols = []
        seen_cols = {}
        for idx, c in enumerate(df.columns):
            c_clean = cls.sanitize_name(c)
            if not c_clean or c_clean.startswith("unnamed"):
                c_clean = f"col_{idx}"
            if c_clean in seen_cols:
                seen_cols[c_clean] += 1
                c_clean = f"{c_clean}_{seen_cols[c_clean]}"
            else:
                seen_cols[c_clean] = 0
            clean_cols.append(c_clean)
        df.columns = clean_cols

        # Auto-fill missing date from timestamp column if available
        date_candidates = ['thoi_gian', 'thoi_gian_lap', 'gio', 'gio_ban', 'gio_tao_don']
        for d_col in ['ngay', 'ngay_lap', 'ngay_xuat', 'ngay_hd']:
            if d_col in df.columns:
                for t_col in date_candidates:
                    if t_col in df.columns and t_col != d_col:
                        d_norm = df[d_col].apply(cls.normalize_date_value)
                        t_norm = df[t_col].apply(cls.normalize_date_value)
                        df[d_col] = d_norm.fillna(t_norm)
                        break

        for col in df.columns:
            # 1. Date column
            if is_date_column(col):
                df[col] = df[col].apply(cls.normalize_date_value)
                continue

            # 2. Text column (never numeric)
            is_text = any(kw in col for kw in TEXT_KEYWORDS)
            is_numeric = is_numeric_column(col)

            if is_text and not is_numeric:
                df[col] = df[col].astype(str).replace({'nan': None, 'None': None, '<NA>': None})
                continue

            # 3. Numeric or currency column
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample = non_null.head(30)
                parsed_sample = [cls.clean_currency_value(v) for v in sample]
                valid_num_count = sum(1 for v in parsed_sample if not np.isnan(v))
                if is_numeric or (valid_num_count / len(sample) >= 0.8):
                    df[col] = df[col].apply(cls.clean_currency_value)

        return df

    @staticmethod
    def find_header_row(df_raw: pd.DataFrame, max_search: int = 8) -> int:
        """Detects the real column header row in messy Excel sheets with title banners."""
        best_row = 0
        max_score = -1
        for r in range(min(max_search, len(df_raw))):
            row = df_raw.iloc[r]
            non_null_count = row.count()
            str_count = sum(1 for v in row if isinstance(v, str) and v.strip())
            score = str_count * 2 + non_null_count
            if score > max_score and str_count >= 2:
                max_score = score
                best_row = r
        return best_row

    def load_business_data(self, biz_dir: str, cache_dir: Optional[str] = None, overwrite: bool = False) -> Tuple[duckdb.DuckDBPyConnection, str]:
        """
        Loads all data files (.csv, .xlsx, .sqlite) of a business into DuckDB.
        Returns connection and clean schema summary.
        Supports high-speed on-disk caching (.duckdb file) for near-instantaneous (0.01s) loading.
        """
        biz_id = os.path.basename(os.path.normpath(biz_dir))
        cached_db_path = None
        cached_schema_path = None
        use_cache = False

        try:
            if cache_dir is None:
                parent_dir = os.path.dirname(os.path.normpath(biz_dir))
                cache_dir = os.path.join(parent_dir, ".duckdb_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cached_db_path = os.path.join(cache_dir, f"{biz_id}.duckdb")
            cached_schema_path = os.path.join(cache_dir, f"{biz_id}_schema.txt")
            use_cache = True
        except Exception:
            use_cache = False

        # Fast path: Load from pre-built cache in ~0.01s!
        if not overwrite and use_cache and os.path.exists(cached_db_path) and os.path.exists(cached_schema_path):
            try:
                con = duckdb.connect(database=cached_db_path, read_only=False)
                con.execute("PRAGMA threads=8;")
                con.execute("PRAGMA preserve_insertion_order=false;")
                with open(cached_schema_path, "r", encoding="utf-8") as f:
                    schema_summary = f.read()
                return con, schema_summary
            except Exception:
                pass

        if use_cache and cached_db_path:
            # Delete stale cache file if incomplete
            if os.path.exists(cached_db_path):
                try:
                    os.remove(cached_db_path)
                except Exception:
                    pass
            try:
                con = duckdb.connect(database=cached_db_path)
            except Exception:
                con = duckdb.connect(database=':memory:')
                use_cache = False
        else:
            con = duckdb.connect(database=':memory:')

        try:
            con.execute("PRAGMA threads=8;")
            con.execute("PRAGMA preserve_insertion_order=false;")
        except Exception:
            pass

        schema_lines = []
        table_meta = []

        files = sorted(os.listdir(biz_dir))
        for filename in files:
            if filename == "README.md":
                continue
            file_path = os.path.join(biz_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            base_name = os.path.splitext(filename)[0]

            if ext == '.csv':
                tbl_name = self.sanitize_name(base_name)
                df = pd.read_csv(file_path, low_memory=False)
                df = self.clean_dataframe(df)
                con.register(f"temp_{tbl_name}", df)
                con.execute(f"CREATE TABLE \"{tbl_name}\" AS SELECT * FROM temp_{tbl_name}")
                con.unregister(f"temp_{tbl_name}")
                
                # Describe column types
                col_types = [f"{desc[0]} ({desc[1]})" for desc in con.execute(f"DESCRIBE \"{tbl_name}\"").fetchall()]
                table_meta.append((tbl_name, ', '.join(col_types), f"{filename}"))

                # Create table aliases
                for alias in generate_table_aliases(tbl_name):
                    try:
                        con.execute(f'CREATE OR REPLACE VIEW "{alias}" AS SELECT * FROM "{tbl_name}"')
                    except Exception:
                        pass

            elif ext == '.xlsx':
                engine = None
                try:
                    import python_calamine
                    engine = 'calamine'
                except ImportError:
                    pass

                xls = pd.ExcelFile(file_path, engine=engine)
                for sheet_name in xls.sheet_names:
                    df_sample = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=10)
                    h_idx = self.find_header_row(df_sample)
                    
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h_idx)
                    df = self.clean_dataframe(df)
                    
                    tbl_name = self.sanitize_name(f"{base_name}_{sheet_name}")
                    con.register(f"temp_{tbl_name}", df)
                    con.execute(f"CREATE TABLE \"{tbl_name}\" AS SELECT * FROM temp_{tbl_name}")
                    con.unregister(f"temp_{tbl_name}")
                    
                    col_types = [f"{desc[0]} ({desc[1]})" for desc in con.execute(f"DESCRIBE \"{tbl_name}\"").fetchall()]
                    table_meta.append((tbl_name, ', '.join(col_types), f"{filename} -> sheet '{sheet_name}'"))

                    # Create table aliases (e.g. t5 vs t05, without duplicated year)
                    for alias in generate_table_aliases(tbl_name):
                        try:
                            con.execute(f'CREATE OR REPLACE VIEW "{alias}" AS SELECT * FROM "{tbl_name}"')
                        except Exception:
                            pass

            elif ext == '.sqlite':
                sql_conn = sqlite3.connect(file_path)
                cursor = sql_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for (t_name,) in tables:
                    if t_name.startswith("sqlite_"):
                        continue
                    df = pd.read_sql_query(f"SELECT * FROM \"{t_name}\"", sql_conn)
                    df = self.clean_dataframe(df)
                    tbl_name = self.sanitize_name(f"sqlite_{t_name}")
                    con.register(f"temp_{tbl_name}", df)
                    con.execute(f"CREATE TABLE \"{tbl_name}\" AS SELECT * FROM temp_{tbl_name}")
                    con.unregister(f"temp_{tbl_name}")
                    
                    col_types = [f"{desc[0]} ({desc[1]})" for desc in con.execute(f"DESCRIBE \"{tbl_name}\"").fetchall()]
                    table_meta.append((tbl_name, ', '.join(col_types), f"{filename} -> table '{t_name}'"))
                sql_conn.close()

        # Group tables with identical columns to drastically reduce prompt token footprint
        from collections import defaultdict
        schema_groups = defaultdict(list)
        for tbl_name, col_str, src in table_meta:
            schema_groups[col_str].append((tbl_name, src))

        def derive_unified_name(table_names: List[str]) -> Optional[str]:
            clean_names = []
            for t in table_names:
                c = re.sub(r'_(sheet\d+|t\d{1,2}_\d{4}|\d{4}|\d+)', '', t)
                c = re.sub(r'_+', '_', c).strip('_')
                clean_names.append(c)
            if len(set(clean_names)) == 1 and clean_names[0]:
                return clean_names[0]
            s1, s2 = min(table_names), max(table_names)
            for i, c in enumerate(s1):
                if i >= len(s2) or c != s2[i]:
                    prefix = s1[:i].rstrip('_')
                    return prefix if len(prefix) >= 3 else None
            return s1.rstrip('_') if len(s1.rstrip('_')) >= 3 else None

        schema_lines = []
        for col_str, items in schema_groups.items():
            if len(items) == 1:
                tbl_name, src = items[0]
                schema_lines.append(f"Table '{tbl_name}' ({src}):\n  Columns: {col_str}")
            else:
                names = [t[0] for t in items]
                uname = derive_unified_name(names)
                if uname and uname not in names:
                    try:
                        union_sql = " UNION ALL ".join([f'SELECT * FROM "{tbl}"' for tbl in names])
                        con.execute(f'CREATE OR REPLACE VIEW "{uname}" AS {union_sql}')
                        # Generate all common alias variations
                        aliases = {uname, uname.replace('_', '')}
                        m = re.sub(r'_(ban_hang|banhang)$', '', uname)
                        if m != uname:
                            aliases.add(m)
                            aliases.add(m.replace('_', ''))
                        sub = uname.replace('ban_hang', 'banhang').replace('tong_hop', 'tonghop').replace('hoa_don', 'hoadon').replace('chi_tiet', 'chitiet')
                        aliases.add(sub)
                        if sub.endswith('_banhang'):
                            aliases.add(sub[:-8])
                        for a in aliases:
                            if a and a != uname:
                                try:
                                    con.execute(f'CREATE OR REPLACE VIEW "{a}" AS SELECT * FROM "{uname}"')
                                except Exception:
                                    pass
                        schema_lines.append(
                            f"View hợp nhất toàn bộ '{uname}' (hợp nhất từ {len(names)} bảng: {names}):\n"
                            f"  Columns: {col_str}\n"
                            f"  * LƯU Ý: Khuyên dùng view '{uname}' này để truy vấn toàn bộ thời gian không lo thiếu bảng!"
                        )
                    except Exception:
                        schema_lines.append(f"Tables ({len(names)} bảng cùng cấu trúc cột): {names}\n  Columns: {col_str}")
                else:
                    schema_lines.append(f"Tables ({len(names)} bảng cùng cấu trúc cột): {names}\n  Columns: {col_str}")

        schema_summary = "\n".join(schema_lines)
        if use_cache and cached_schema_path:
            try:
                with open(cached_schema_path, "w", encoding="utf-8") as f:
                    f.write(schema_summary)
            except Exception:
                pass
        return con, schema_summary
