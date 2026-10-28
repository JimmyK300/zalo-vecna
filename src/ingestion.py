import os
import re
import sqlite3
import pandas as pd
import duckdb
from typing import Dict, List, Tuple

class DataIngestionEngine:
    """
    Unified Ingestion Engine:
    Loads CSV, Excel (.xlsx, multi-sheet), and SQLite tables into an in-memory DuckDB database.
    Provides smart header detection for messy Excel banners and extracts schema profiles.
    """
    def __init__(self):
        pass

    @staticmethod
    def sanitize_table_name(name: str) -> str:
        """Sanitizes file / sheet names to valid SQL table names."""
        clean = re.sub(r'[^\w]+', '_', name.strip().lower())
        return clean.strip('_')

    @staticmethod
    def find_header_row(df_raw: pd.DataFrame, max_search: int = 8) -> int:
        """
        Detects the real column header row in messy Excel sheets that have title banners at the top.
        """
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
        Loads all data files (.csv, .xlsx, .sqlite) of a business into an in-memory DuckDB connection.
        Returns the DuckDB connection and a textual schema summary.
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
                tbl_name = self.sanitize_table_name(base_name)
                # Load CSV via DuckDB's native fast scanner
                con.execute(f"CREATE TABLE \"{tbl_name}\" AS SELECT * FROM read_csv_auto('{file_path.replace(os.sep, '/')}', header=True)")
                cols = [desc[0] for desc in con.execute(f"DESCRIBE \"{tbl_name}\"").fetchall()]
                schema_lines.append(f"Table '{tbl_name}' (from {filename}): columns = {cols}")

            elif ext == '.xlsx':
                # Read all sheets from Excel file with smart header detection
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    # Sample top rows without header to detect real header position
                    df_sample = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=10)
                    h_idx = self.find_header_row(df_sample)
                    
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=h_idx)
                    # Clean column names (strip whitespace and handle duplicates)
                    clean_cols = []
                    seen_cols = {}
                    for idx, c in enumerate(df.columns):
                        c_str = str(c).strip()
                        if not c_str or c_str.startswith("Unnamed:"):
                            c_str = f"col_{idx}"
                        if c_str in seen_cols:
                            seen_cols[c_str] += 1
                            c_str = f"{c_str}_{seen_cols[c_str]}"
                        else:
                            seen_cols[c_str] = 0
                        clean_cols.append(c_str)
                    df.columns = clean_cols

                    tbl_name = self.sanitize_table_name(f"{base_name}_{sheet_name}")
                    con.register(f"temp_{tbl_name}", df)
                    con.execute(f"CREATE TABLE \"{tbl_name}\" AS SELECT * FROM temp_{tbl_name}")
                    con.unregister(f"temp_{tbl_name}")
                    cols = list(df.columns)
                    schema_lines.append(f"Table '{tbl_name}' (from {filename} -> sheet '{sheet_name}'): columns = {cols}")

            elif ext == '.sqlite':
                # Attach or copy tables from SQLite
                sql_conn = sqlite3.connect(file_path)
                cursor = sql_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for (t_name,) in tables:
                    if t_name.startswith("sqlite_"):
                        continue
                    df = pd.read_sql_query(f"SELECT * FROM \"{t_name}\"", sql_conn)
                    tbl_name = self.sanitize_table_name(f"sqlite_{t_name}")
                    con.register(f"temp_{tbl_name}", df)
                    con.execute(f"CREATE TABLE \"{tbl_name}\" AS SELECT * FROM temp_{tbl_name}")
                    con.unregister(f"temp_{tbl_name}")
                    cols = list(df.columns)
                    schema_lines.append(f"Table '{tbl_name}' (from {filename} -> table '{t_name}'): columns = {cols}")
                sql_conn.close()

        schema_summary = "\n".join(schema_lines)
        return con, schema_summary
