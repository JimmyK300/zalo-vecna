import os
import sys
import time
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ingestion import DataIngestionEngine

def main():
    parser = argparse.ArgumentParser(description="Pre-build DuckDB cache for all businesses")
    parser.add_argument("--data_dir", type=str, default="public_test", help="Path to public_test or test directory")
    args = parser.parse_args()

    biz_base = os.path.join(args.data_dir, "data")
    if not os.path.exists(biz_base):
        print(f"[!] Directory not found: {biz_base}")
        return

    businesses = sorted([d for d in os.listdir(biz_base) if os.path.isdir(os.path.join(biz_base, d))])
    print(f"[*] Found {len(businesses)} businesses in {biz_base}.")
    print("[*] Pre-building high-speed DuckDB disk cache (.duckdb)...")

    engine = DataIngestionEngine()
    total_start = time.time()

    for biz_id in tqdm(businesses, desc="Caching businesses"):
        biz_dir = os.path.join(biz_base, biz_id)
        t0 = time.time()
        con, schema = engine.load_business_data(biz_dir)
        # Verify and close
        table_count = len(con.execute("SHOW TABLES").fetchall())
        con.close()
        t1 = time.time()
        # print(f" [+] {biz_id}: {table_count} tables cached in {t1-t0:.2f}s")

    print(f"\n[+] All {len(businesses)} businesses cached in {time.time()-total_start:.2f} seconds!")
    print("[+] Subsequent pipeline runs will load in ~0.01s per business!")

if __name__ == "__main__":
    main()
