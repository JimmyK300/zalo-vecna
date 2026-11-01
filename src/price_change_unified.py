import sys, os, json, re, calendar
from datetime import datetime, timedelta
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import test_enhanced_engine as eng
import autonomous_solver_v2 as solver

def solve_price_change_unified(con, biz_id, qtext):
    qn = solver.remove_accents(qtext).lower()
    cfg = solver.BIZ_CONFIG.get(biz_id, {})
    table = cfg.get("table", "chi_tiet_hoa_don")
    date_col = cfg.get("date_col", "ngay_xuat")
    item_col = cfg.get("item_col", "ten_hang")
    qty_col = cfg.get("qty_col", "s_lg")
    not_canceled = solver.get_not_canceled_filter(cfg)

    if biz_id == "dev_006":
        table = "chi_tiet_don c JOIN don_hang d ON c.ma_hoa_don = d.ma_hoa_don JOIN san_pham s ON c.ma_hang_hoa = s.ma_hang_hoa"
        date_col = "d.ngay_lap"
        item_col = "s.ten_goi"
        qty_col = "c.sl"
        not_canceled = "(d.tinh_trang IS NULL OR d.tinh_trang NOT ILIKE '%cancel%')"
    elif biz_id == "dev_008":
        table = "chi_tiet_hoa_don"
        date_col = "ngay_xuat"
        item_col = "ten_hang"
        qty_col = "s_lg"
        not_canceled = "(tt_don IS NULL OR tt_don NOT ILIKE '%hủy%')"
    elif biz_id == "dev_007":
        table = "chi_tiet_hoa_don"
        date_col = "SUBSTRING(ngay_giao_dich, 1, 10)"
        item_col = "mat_hang"
        qty_col = "sluong"
        not_canceled = "1=1"

    if not item_col or not qty_col:
        return None

    m_d = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', qtext)
    m_n = re.search(r'(\d+)\s*ngay', qn)
    n_days = int(m_n.group(1)) if m_n else 30

    try:
        all_items = [r[0] for r in con.execute(f"SELECT DISTINCT {item_col} FROM {table}").fetchall() if r[0]]
    except:
        return None

    best_item = None
    best_score = 0
    for item in all_items:
        item_norm = solver.remove_accents(item).lower()
        tokens = [t for t in item_norm.split() if len(t) >= 3]
        if not tokens: continue
        score = sum(1 for t in tokens if t in qn)
        if score > best_score and score >= min(2, len(tokens)):
            best_score = score
            best_item = item

    if not best_item:
        return None

    if m_d:
        day, month, year = int(m_d.group(1)), int(m_d.group(2)), int(m_d.group(3))
        d_dt = datetime(year, month, day)
    else:
        m_pr = re.search(r'(?:len|tang gia len|len gia)\s*([\d.,]+)\s*(?:d|dong|k)', qn)
        if not m_pr: return None
        target_p = eng.parse_val(m_pr.group(1))
        # Find price column
        p_candidates = ["gia_ban_le", "don_gia_d", "don_gia", "gia_don_vi", "gia_1_dv", "gia"]
        row = None
        for pc in p_candidates:
            try:
                row = con.execute(f"SELECT MIN({date_col}) FROM {table} WHERE {item_col} = '{best_item}' AND {pc} = {target_p}").fetchone()
                if row and row[0]: break
            except: pass
        if not row or not row[0]: return None
        d_dt = datetime.strptime(str(row[0])[:10], "%Y-%m-%d")

    after_s = d_dt.strftime("%Y-%m-%d")
    after_e = (d_dt + timedelta(days=n_days - 1)).strftime("%Y-%m-%d")
    before_e = (d_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    before_s = (d_dt - timedelta(days=n_days)).strftime("%Y-%m-%d")

    try:
        q_b = f"SELECT SUM({qty_col}) FROM {table} WHERE {date_col} >= '{before_s}' AND {date_col} <= '{before_e}' AND {item_col} = '{best_item}' AND {not_canceled}"
        q_a = f"SELECT SUM({qty_col}) FROM {table} WHERE {date_col} >= '{after_s}' AND {date_col} <= '{after_e}' AND {item_col} = '{best_item}' AND {not_canceled}"
        vb = con.execute(q_b).fetchone()[0] or 0.0
        va = con.execute(q_a).fetchone()[0] or 0.0
        pct = ((va - vb) / vb) * 100.0 if vb > 0 else 0.0
        return {"type": "number", "answer": round(float(pct), 2)}
    except:
        return None
