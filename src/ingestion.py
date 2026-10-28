import os
import re
import sqlite3
import pandas as pd
import numpy as np
import duckdb
from unidecode import unidecode
from typing import Dict, List, Tuple, Any

class DataIngestionEngine:
    """
    Unified Ingestion Engine:
    Loads CSV, Excel (.xlsx, multi-sheet), and SQLite tables into an in-memory DuckDB database.
    - Smart Header Detection (handles banners & headers on row 0, 1, 2, 3...)
    - Currency & Numeric Shorthand Cleaning (converts '29k', '43.000đ', '1,01tr' to float)
    - Column Name Sanitization (clean snake_case for robust SQL queries without space/diacritic errors)
    """
    def __init__(self):
        pass

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Sanitizes names to valid ASCII snake_case SQL identifiers."""
        s = unidecode(str(name)).lower().strip()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')

    @staticmethod
    def clean_currency_value(val: Any) -> Any:
        """Converts Vietnamese currency/number strings to float (e.g. '29k', '43.000đ', '1,01tr')."""
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        s = str(val).strip().lower()
        if not s or s in ['-', 'none', 'nan', 'null']:
            return np.nan

        is_neg = '-' in s
        if 'tr' in s:
            num_str = re.sub(r'[^\d,\.]', '', s).replace(',', '.')
            try:
                res = float(num_str) * 1_000_000
                return -abs(res) if is_neg else res
            except ValueError:
                pass

        if 'k' in s:
            num_str = re.sub(r'[^\d,\.]', '', s).replace(',', '.')
            try:
                res = float(num_str) * 1_000
                return -abs(res) if is_neg else res
            except ValueError:
                pass

        s_clean = re.sub(r'[^\d,\.]', '', s)
        if not s_clean:
            return np.nan

        if '.' in s_clean and ',' not in s_clean:
            parts = s_clean.split('.')
            if len(parts[-1]) == 3 or len(parts) > 2:
                s_clean = s_clean.replace('.', '')
        elif ',' in s_clean and '.' not in s_clean:
            parts = s_clean.split(',')
            if len(parts[-1]) == 3 or len(parts) > 2:
                s_clean = s_clean.replace(',', '')
            else:
                s_clean = s_clean.replace(',', '.')
        elif '.' in s_clean and ',' in s_clean:
            s_clean = s_clean.replace('.', '').replace(',', '.')

        try:
            res = float(s_clean)
            return -abs(res) if is_neg else res
        except ValueError:
            return np.nan

    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans columns and auto-converts currency/number strings to float."""
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

        # Auto-convert numeric/currency columns
        numeric_keywords = ['tien', 'tong', 'gia', 'so_luong', 'sl', 'thanh_tien', 'chi_phi', 'quy_doi', 'don_gia', 'no']
        for col in df.columns:
            # Check if column name strongly indicates numeric
            is_num_name = any(kw in col for kw in numeric_keywords)
            # Sample column to check if values are parseable as currency/number
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample = non_null.head(50)
                parsed_sample = [cls.clean_currency_value(v) for v in sample]
                valid_num_count = sum(1 for v in parsed_sample if not np.isnan(v))
                if is_num_name or (valid_num_count / len(sample) >= 0.7):
                    # Convert whole column
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

    def load_business_data(self, biz_dir: str) -> Tuple[duckdb.DuckDBPyConnection, str]:
        """
        Loads all data files (.csv, .xlsx, .sqlite) of a business into DuckDB.
        Returns connection and clean schema summary.
        """
        con = duckdb.connect(database=':memory:')
        schema_lines = []

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
                schema_lines.append(f"Table '{tbl_name}' (from {filename}):\n  Columns: {', '.join(col_types)}")

            elif ext == '.xlsx':
                xls = pd.ExcelFile(file_path)
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
                    schema_lines.append(f"Table '{tbl_name}' (from {filename} -> sheet '{sheet_name}'):\n  Columns: {', '.join(col_types)}")

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
                    schema_lines.append(f"Table '{tbl_name}' (from {filename} -> table '{t_name}'):\n  Columns: {', '.join(col_types)}")
                sql_conn.close()

        schema_summary = "\n".join(schema_lines)
        return con, schema_summary
