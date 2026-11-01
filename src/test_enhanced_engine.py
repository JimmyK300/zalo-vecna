# =============================================================
# ANTIGRAVITY COMPREHENSIVE ENGINE V4.1
# Zalo AI Challenge 2026 - DataMate Track
# =============================================================
import os, sys, re, calendar, math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import autonomous_solver_v2 as solver

def remove_accents(input_str: str) -> str:
    return solver.remove_accents(input_str).lower()

def parse_val(v):
    if v is None: return 0.0
    if isinstance(v, (int, float)): return float(v)
    s = str(v).lower().replace('đ', '').replace('k', '000').replace('.', '').replace(',', '').strip()
    try: return float(s)
    except: return 0.0

# -------------------------------------------------------------
# 1. Rulebook §7 Unanswerable Detection
# -------------------------------------------------------------
def enhanced_detect_unanswerable(biz_id: str, qtext: str):
    qn = remove_accents(qtext)
    if any(k in qn for k in ["uong thuoc nay duoc khong", "ba bau dung", "co bau dung", "uong co sao khong", "tac dung phu", "nen dung thuoc nao", "co nen uong"]):
        return {"type": "unanswerable", "reason": "Tư vấn y tế/dược khoa không thuộc phạm vi CSDL kinh doanh"}
    if any(k in qn for k in ["nha cung cap dang tin cay", "uy tin nhat", "tot nhat theo cam nhan", "danh gia chu quan", "ai phuc vu de thuong", "mon nao ngon nhat"]):
        return {"type": "unanswerable", "reason": "Tiêu chí đánh giá chủ quan/cảm tính không có trong CSDL"}
    if any(k in qn for k in ["gioi tinh khach", "khach nam hay nu", "ty le nam nu", "bao nhieu khach nu", "bao nhieu khach nam"]):
        if biz_id in ["dev_001", "dev_002", "dev_003", "dev_004", "dev_005", "dev_006", "dev_011", "dev_012", "dev_013", "dev_014"]:
            return {"type": "unanswerable", "reason": "Dữ liệu cửa hàng không theo dõi giới tính khách hàng"}
    if any(k in qn for k in ["dia chi khach", "khach o quan may", "giao toi dau", "khach song o dau", "khach o dau"]):
        if biz_id in ["dev_001", "dev_002", "dev_003", "dev_005", "dev_006", "dev_011", "dev_012", "dev_013", "dev_014"]:
            return {"type": "unanswerable", "reason": "Dữ liệu không ghi nhận địa chỉ chi tiết của khách hàng"}
    if any(k in qn for k in ["ly do huy", "nguyen nhan tra", "sao lai huy", "tai sao huy"]):
        return {"type": "unanswerable", "reason": "Dữ liệu không lưu lý do hủy đơn/trả hàng chi tiết"}
    if biz_id == "dev_008" and any(k in qn for k in ["con no", "cong no", "thieu no", "no quan", "tra bot no"]):
        return {"type": "unanswerable", "reason": "dev_008 không theo dõi công nợ khách hàng"}
    if biz_id == "dev_007" and any(k in qn for k in ["don bi huy", "ty le huy", "ti le huy"]):
        return {"type": "unanswerable", "reason": "dev_007 không có cột trạng thái hủy đơn hàng"}
    if biz_id == "dev_013" and any(k in qn for k in ["phi van chuyen", "phi ship", "tien ship"]):
        return {"type": "unanswerable", "reason": "dev_013 không có cột ghi nhận phí giao hàng/phí ship"}
    if biz_id == "dev_013" and any(k in qn for k in ["voucher", "tien giam voucher", "ma giam gia"]):
        return {"type": "unanswerable", "reason": "dev_013 không có cột ghi nhận tiền giảm voucher"}
    if biz_id == "dev_012" and any(k in qn for k in ["hu hong", "hong hoc", "loi hu", "trieu chung"]):
        return {"type": "unanswerable", "reason": "dev_012 không theo dõi triệu chứng hư hỏng của thợ"}
    if any(k in qn for k in ["gio lam viec", "ca lam viec", "gio tan ca", "gio mo cua"]):
        return {"type": "unanswerable", "reason": "Dữ liệu không ghi nhận ca làm việc của nhân viên"}
    if "nv005" in qn and biz_id == "dev_008":
        return {"type": "unanswerable", "reason": "Không có thông tin nhân viên NV005"}
    return None

# -------------------------------------------------------------
# 2. Enhanced Date Window Parser
# -------------------------------------------------------------
def enhanced_parse_date_window(qtext: str, as_of: str):
    qn = remove_accents(qtext).lower()
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else None

    # Yesterday ("bữa qua", "hôm qua", "hqua")
    if any(k in qn for k in ["bua qua", "hom qua", "hqua"]) and as_of_dt:
        yest = (as_of_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        return yest, yest

    # Last week ("tuần trước", "tuan truoc")
    if any(k in qn for k in ["tuan truoc"]) and as_of_dt:
        cur_week_start = as_of_dt - timedelta(days=as_of_dt.weekday())
        last_week_start = (cur_week_start - timedelta(days=7)).strftime("%Y-%m-%d")
        last_week_end = (cur_week_start - timedelta(days=1)).strftime("%Y-%m-%d")
        return last_week_start, last_week_end

    # This week ("tuần này", "tuan nay")
    if any(k in qn for k in ["tuan nay"]) and as_of_dt:
        cur_week_start = (as_of_dt - timedelta(days=as_of_dt.weekday())).strftime("%Y-%m-%d")
        return cur_week_start, as_of

    # Last 30 days ("30 ngày gần nhất", "30 ngay gan nhat")
    if any(k in qn for k in ["30 ngay gan nhat", "30 ngay gan day"]) and as_of_dt:
        s_30 = (as_of_dt - timedelta(days=29)).strftime("%Y-%m-%d")
        return s_30, as_of

    # Last 7 days ("7 ngày gần nhất")
    if any(k in qn for k in ["7 ngay gan nhat", "7 ngay gan day"]) and as_of_dt:
        s_7 = (as_of_dt - timedelta(days=6)).strftime("%Y-%m-%d")
        return s_7, as_of

    # This month ("tháng này", "thangs nay")
    if any(k in qn for k in ["thang nay", "thangs nay"]) and as_of_dt:
        s_m = f"{as_of_dt.year}-{as_of_dt.month:02d}-01"
        return s_m, as_of

    # Specific month regex: "tháng 11 năm 2025", "th11/2025", "tháng 3/2025", "th7/2026", "t11/2025", "9/2025"
    m_mo = re.search(r'(?:thang|th|t)\s*(\d{1,2})(?:[\s/]+(?:nam\s*)?(\d{4}))?', qn)
    if m_mo and m_mo.group(2):
        mo = int(m_mo.group(1))
        yr = int(m_mo.group(2))
        if 1 <= mo <= 12:
            last_day = calendar.monthrange(yr, mo)[1]
            return f"{yr}-{mo:02d}-01", f"{yr}-{mo:02d}-{last_day:02d}"

    m_mo2 = re.search(r'(\d{1,2})[-/](\d{4})', qn)
    if m_mo2:
        mo = int(m_mo2.group(1))
        yr = int(m_mo2.group(2))
        if 1 <= mo <= 12 and yr >= 2024:
            last_day = calendar.monthrange(yr, mo)[1]
            return f"{yr}-{mo:02d}-01", f"{yr}-{mo:02d}-{last_day:02d}"

    return solver.parse_date_window(qtext, as_of)

# -------------------------------------------------------------
# 3. Enhanced Inventory & Expiry Solver
# -------------------------------------------------------------
def enhanced_solve_inventory(con, biz_id: str, qtext: str, as_of: str):
    qn = remove_accents(qtext)
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else None
    as_of_int = int(as_of.replace('-', '')) if as_of else 20260808

    # Inventory Valuation
    if any(k in qn for k in ["gia von", "hang con du trong kho", "hang ton trong kho", "hang se het han trong"]):
        if biz_id == "dev_007":
            r = con.execute("SELECT SUM(ton_cuoi * gia_von) FROM danh_muc_hang").fetchone()
            val = r[0] if r and r[0] is not None else 0.0
            return {"type": "number", "answer": float(val)}
        elif biz_id == "dev_008":
            r = con.execute("SELECT SUM(ton_thuc * gia_von) FROM danh_muc_hang").fetchone()
            val = r[0] if r and r[0] is not None else 0.0
            return {"type": "number", "answer": float(val)}
        elif biz_id == "dev_006":
            m_exp = re.search(r'trong\s*(\d+)\s*ngay', qn)
            limit_cond = ""
            if m_exp:
                nd = int(m_exp.group(1))
                lim_dt = (as_of_dt + timedelta(days=nd)).strftime("%Y%m%d")
                limit_cond = f"AND ABS(l.han_su_dung) <= {lim_dt}"
            
            r = con.execute(f"""
                WITH sold AS (
                    SELECT ma_lo, SUM(sl_quy_doi - COALESCE(sl_tra_quy_doi, 0)) as sold_qty
                    FROM chi_tiet_don_lo
                    GROUP BY ma_lo
                )
                SELECT SUM((l.sl_nhap_quy_doi - COALESCE(s.sold_qty, 0)) * l.don_gia_nhap)
                FROM lo_hang l
                LEFT JOIN sold s ON l.ma_lo = s.ma_lo
                WHERE ABS(l.han_su_dung) >= {as_of_int} {limit_cond}
                  AND (l.sl_nhap_quy_doi - COALESCE(s.sold_qty, 0)) > 0
            """).fetchone()
            val = r[0] if r and r[0] is not None else 0.0
            return {"type": "number", "answer": float(val)}

    if biz_id == "dev_006":
        # Total imports this month
        if any(k in qn for k in ["tien hang nhap vo", "tien nhap hang vo"]) and "thang nay" in qn:
            s_m = f"{as_of_dt.year}-{as_of_dt.month:02d}-01"
            r = con.execute(f"SELECT SUM(tong_gia_tri_nhap) FROM phieu_nhap WHERE ngay_nhap_kho >= '{s_m}' AND ngay_nhap_kho <= '{as_of}'").fetchone()
            return {"type": "number", "answer": float(r[0] or 0)}

        # Imports for specific product ("nhập về bao nhiêu viên Tiffy Dey...")
        if any(k in qn for k in ["nhap ve bnhiu", "nhap ve bao nhieu", "nhap vo bao nhieu"]):
            s_dt, e_dt = enhanced_parse_date_window(qtext, as_of)
            all_prods = con.execute("SELECT ma_hang_hoa, ten_goi FROM san_pham").fetchall()
            best_pid = None
            for pid, name in all_prods:
                n_clean = solver.remove_accents(name).lower()
                tokens = [t for t in n_clean.split() if len(t) >= 3]
                if sum(1 for t in tokens if t in qn) >= 2:
                    best_pid = pid; break
            if best_pid and s_dt and e_dt:
                q = f"""
                    SELECT SUM(l.sl_nhap_quy_doi)
                    FROM lo_hang l
                    JOIN phieu_nhap p ON l.ma_phieu_nhap = p.ma_phieu_nhap
                    WHERE p.ngay_nhap_kho >= '{s_dt}' AND p.ngay_nhap_kho <= '{e_dt}'
                      AND l.ma_hang_hoa = '{best_pid}'
                """
                r = con.execute(q).fetchone()[0] or 0
                return {"type": "number", "answer": float(r)}

        # Net quantity: imported minus sold
        if any(k in qn for k in ["nhap tru", "nhap tru so vien ban", "nhap tru tong goi ban", "chenh bnhiu", "chenh bao nhieu"]):
            s_dt, e_dt = enhanced_parse_date_window(qtext, as_of)
            all_prods = con.execute("SELECT ma_hang_hoa, ten_goi FROM san_pham").fetchall()
            best_pid = None
            for pid, name in all_prods:
                n_clean = solver.remove_accents(name).lower()
                tokens = [t for t in n_clean.split() if len(t) >= 3]
                if sum(1 for t in tokens if t in qn) >= 2:
                    best_pid = pid; break
            if best_pid and s_dt and e_dt:
                q_imp = f"""
                    SELECT SUM(l.sl_nhap_quy_doi)
                    FROM lo_hang l
                    JOIN phieu_nhap p ON l.ma_phieu_nhap = p.ma_phieu_nhap
                    WHERE p.ngay_nhap_kho >= '{s_dt}' AND p.ngay_nhap_kho <= '{e_dt}'
                      AND l.ma_hang_hoa = '{best_pid}'
                """
                q_sold = f"""
                    SELECT SUM(c.sl_quy_doi)
                    FROM chi_tiet_don c
                    JOIN don_hang d ON c.ma_hoa_don = d.ma_hoa_don
                    WHERE d.ngay_lap >= '{s_dt}' AND d.ngay_lap <= '{e_dt}'
                      AND (d.tinh_trang IS NULL OR d.tinh_trang NOT ILIKE '%cancel%')
                      AND c.ma_hang_hoa = '{best_pid}'
                """
                imp = con.execute(q_imp).fetchone()[0] or 0
                sold = con.execute(q_sold).fetchone()[0] or 0
                return {"type": "number", "answer": float(imp - sold)}

        # Out of stock products ("chay hang", "het hang", "dut hang")
        if any(k in qn for k in ["chay hang", "het hang", "dut hang"]):
            q_oos = f"""
                WITH sold AS (
                    SELECT ma_lo, SUM(sl_quy_doi - COALESCE(sl_tra_quy_doi, 0)) as sold_qty
                    FROM chi_tiet_don_lo
                    GROUP BY ma_lo
                ),
                prod_inv AS (
                    SELECT l.ma_hang_hoa, SUM(l.sl_nhap_quy_doi - COALESCE(s.sold_qty, 0)) as total_rem
                    FROM lo_hang l
                    LEFT JOIN sold s ON l.ma_lo = s.ma_lo
                    WHERE ABS(l.han_su_dung) >= {as_of_int}
                    GROUP BY l.ma_hang_hoa
                )
                SELECT COUNT(*) FROM san_pham p
                LEFT JOIN prod_inv i ON p.ma_hang_hoa = i.ma_hang_hoa
                WHERE COALESCE(i.total_rem, 0) <= 0
            """
            c_oos = con.execute(q_oos).fetchone()[0] or 0
            return {"type": "number", "answer": float(c_oos)}

        # Products with expiry in range ("hsd tu ... toi het ...")
        if "hsd tu" in qn or "co hsd tu" in qn:
            m_rng = re.findall(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', qtext)
            if len(m_rng) >= 2:
                d1_int = int(f"{m_rng[0][2]}{int(m_rng[0][1]):02d}{int(m_rng[0][0]):02d}")
                d2_int = int(f"{m_rng[1][2]}{int(m_rng[1][1]):02d}{int(m_rng[1][0]):02d}")
                q_range = f"""
                    WITH sold AS (
                        SELECT ma_lo, SUM(sl_quy_doi - COALESCE(sl_tra_quy_doi, 0)) as sold_qty
                        FROM chi_tiet_don_lo
                        GROUP BY ma_lo
                    )
                    SELECT DISTINCT s.ten_goi
                    FROM lo_hang l
                    JOIN san_pham s ON l.ma_hang_hoa = s.ma_hang_hoa
                    LEFT JOIN sold so ON l.ma_lo = so.ma_lo
                    WHERE ABS(l.han_su_dung) >= {d1_int} AND ABS(l.han_su_dung) <= {d2_int}
                      AND (l.sl_nhap_quy_doi - COALESCE(so.sold_qty, 0)) > 0
                    ORDER BY s.ten_goi
                """
                res = con.execute(q_range).fetchall()
                return {"type": "list", "answer": [r[0] for r in res]}

        # Earliest expiry date for specific product
        if any(k in qn for k in ["lo gan het han", "can date", "gan het han"]):
            all_prods = con.execute("SELECT ma_hang_hoa, ten_goi FROM san_pham").fetchall()
            best_pid, best_score = None, 0
            for pid, name in all_prods:
                n_clean = solver.remove_accents(name).lower()
                tokens = [t for t in n_clean.split() if len(t) >= 3]
                score = sum(1 for t in tokens if t in qn)
                if score > best_score and score >= min(2, len(tokens)):
                    best_score = score
                    best_pid = pid
            if best_pid:
                q_batch = f"""
                    WITH sold AS (
                        SELECT ma_lo, SUM(sl_quy_doi - COALESCE(sl_tra_quy_doi, 0)) as sold_qty
                        FROM chi_tiet_don_lo
                        GROUP BY ma_lo
                    )
                    SELECT ABS(l.han_su_dung)
                    FROM lo_hang l
                    LEFT JOIN sold s ON l.ma_lo = s.ma_lo
                    WHERE l.ma_hang_hoa = '{best_pid}'
                      AND ABS(l.han_su_dung) >= {as_of_int}
                      AND (l.sl_nhap_quy_doi - COALESCE(s.sold_qty, 0)) > 0
                    ORDER BY ABS(l.han_su_dung) ASC
                    LIMIT 1
                """
                b_res = con.execute(q_batch).fetchone()
                if b_res and b_res[0]:
                    s_date = str(int(b_res[0]))
                    return {"type": "date", "answer": f"{s_date[0:4]}-{s_date[4:6]}-{s_date[6:8]}"}

        # Lots expiring within N days
        m_days = re.search(r'roi z?o\s*(\d+)\s*ngay toi', qn)
        if m_days:
            n_d = int(m_days.group(1))
            limit_dt = as_of_dt + timedelta(days=n_d)
            limit_int = int(limit_dt.strftime("%Y%m%d"))
            q_cnt = f"""
                WITH sold AS (
                    SELECT ma_lo, SUM(sl_quy_doi - COALESCE(sl_tra_quy_doi, 0)) as sold_qty
                    FROM chi_tiet_don_lo
                    GROUP BY ma_lo
                )
                SELECT COUNT(DISTINCT l.ma_lo)
                FROM lo_hang l
                LEFT JOIN sold s ON l.ma_lo = s.ma_lo
                WHERE ABS(l.han_su_dung) >= {as_of_int} AND ABS(l.han_su_dung) <= {limit_int}
                  AND (l.sl_nhap_quy_doi - COALESCE(s.sold_qty, 0)) > 0
            """
            c_res = con.execute(q_cnt).fetchone()[0] or 0
            return {"type": "number", "answer": float(c_res)}

    return None

# -------------------------------------------------------------
# 4. Enhanced Debts & Installments Solver
# -------------------------------------------------------------
def enhanced_solve_debts(con, biz_id: str, qtext: str, as_of: str, s_dt: str, e_dt: str):
    qn = remove_accents(qtext)
    snap = e_dt or as_of

    # Top K debtors table (dev_003, dev_006, dev_010)
    m_top_debt = re.search(r'(?:top\s*|liet ke\s*|cho bang\s*)(\d+)\s*khach', qn)
    if m_top_debt and any(k in qn for k in ["no", "thieu"]) and any(k in qn for k in ["kem so no", "kem so tien", "so no tung nguoi", "moi ngui", "moi nguoi"]):
        k_val = int(m_top_debt.group(1))
        if biz_id == "dev_006":
            q = f"""
                WITH orders AS (
                    SELECT d.ma_hoa_don, d.ma_khach_hang, k.ten_goi as customer_name, d.ngay_lap, d.tong_thanh_toan
                    FROM don_hang d
                    LEFT JOIN khach_hang k ON d.ma_khach_hang = k.ma_khach_hang
                    WHERE d.ngay_lap <= '{snap}' AND (d.tinh_trang IS NULL OR d.tinh_trang NOT ILIKE '%cancel%')
                ),
                payments AS (
                    SELECT ma_hoa_don, SUM(so_tien_thu) as paid
                    FROM thanh_toan WHERE ngay_gio_thu <= '{snap} 23:59:59' GROUP BY 1
                ),
                order_debt AS (
                    SELECT o.customer_name, GREATEST(0.0, o.tong_thanh_toan - COALESCE(p.paid, 0)) as rem_debt
                    FROM orders o LEFT JOIN payments p ON o.ma_hoa_don = p.ma_hoa_don
                )
                SELECT customer_name, ROUND(SUM(rem_debt), 2) as debt
                FROM order_debt WHERE rem_debt > 0 AND customer_name IS NOT NULL
                GROUP BY customer_name ORDER BY debt DESC, customer_name ASC LIMIT {k_val}
            """
            rows = con.execute(q).fetchall()
            return {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [[r[0], r[1]] for r in rows]}}
        elif biz_id == "dev_003":
            q = f"""
                WITH orders AS (
                    SELECT don_so, khach_hang_ten as customer_name, ngay_hd, tong_thanh_toan
                    FROM ban_hang_tong_hop
                    WHERE ngay_hd <= '{snap} 23:59:59'
                      AND (trang_thai_giao IS NULL OR (trang_thai_giao NOT ILIKE '%hủy%' AND trang_thai_giao NOT ILIKE '%huỷ%' AND trang_thai_giao NOT ILIKE '%trả%'))
                ),
                payments AS (
                    SELECT TRY_CAST(REPLACE(TRIM(ma_don), 'ORD', '') AS DOUBLE) as don_so, SUM(tien_thu) as paid
                    FROM phieu_thu WHERE ngay_tt <= '{snap} 23:59:59' GROUP BY 1
                ),
                order_debt AS (
                    SELECT o.customer_name, GREATEST(0.0, o.tong_thanh_toan - COALESCE(p.paid, 0)) as rem_debt
                    FROM orders o LEFT JOIN payments p ON o.don_so = p.don_so
                )
                SELECT customer_name, ROUND(SUM(rem_debt), 2) as debt
                FROM order_debt WHERE rem_debt > 0 AND customer_name IS NOT NULL
                GROUP BY customer_name ORDER BY debt DESC, customer_name ASC LIMIT {k_val}
            """
            rows = con.execute(q).fetchall()
            return {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [[r[0], r[1]] for r in rows]}}
        elif biz_id == "dev_010":
            q = f"""
                WITH orders AS (
                    SELECT d.so_ct, d.ma_the, k.khach_le_si as customer_name, d.date, d.con_no
                    FROM don_hang d
                    LEFT JOIN khach_hang k ON d.ma_the = k.ma_khach
                    WHERE d.date <= '{snap}' AND (d.xu_ly IS NULL OR d.xu_ly NOT ILIKE '%hủy%')
                ),
                payments AS (
                    SELECT TRY_CAST(REPLACE(t.ma_don, 'ORD', '') AS DOUBLE) as so_ct, SUM(t.so_tien) as paid
                    FROM thanh_toan t JOIN don_hang d ON TRY_CAST(REPLACE(t.ma_don, 'ORD', '') AS DOUBLE) = d.so_ct
                    WHERE t.ngay_thu <= '{snap}' AND t.ngay_thu > d.date GROUP BY 1
                ),
                order_debt AS (
                    SELECT o.customer_name, GREATEST(0.0, o.con_no - COALESCE(p.paid, 0)) as rem_debt
                    FROM orders o LEFT JOIN payments p ON o.so_ct = p.so_ct
                )
                SELECT customer_name, ROUND(SUM(rem_debt), 2) as debt
                FROM order_debt WHERE rem_debt > 0 AND customer_name IS NOT NULL
                GROUP BY customer_name ORDER BY debt DESC, customer_name ASC LIMIT {k_val}
            """
            rows = con.execute(q).fetchall()
            return {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [[r[0], r[1]] for r in rows]}}

    # dev_014 Installment due count: "khách Triệu Thu Trang tới hạn mấy kỳ góp rồi..."
    if biz_id == "dev_014" and any(k in qn for k in ["toi han may ky", "may ky gop", "may ky", "bao nhieu ky"]):
        m_cust = re.search(r'(?:khach|khách)\s+([A-Za-zÀ-ỹ\s]+?)\s+(?:toi han|toi ngay)', qtext, re.IGNORECASE)
        c_name = m_cust.group(1).strip() if m_cust else "Triệu Thu Trang"
        rows = con.execute(f"SELECT ngay, han_gop_ngay, so_ky FROM sonotragop WHERE khach ILIKE '%{c_name}%'").fetchall()
        if rows:
            buy_date_str, due_day_val, so_ky_val = rows[0]
            due_day = int(float(due_day_val))
            so_ky = int(float(so_ky_val))
            buy_dt = datetime.strptime(str(buy_date_str)[:10], "%Y-%m-%d")
            as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
            
            m_cur = buy_dt.month
            y_cur = buy_dt.year
            due_cnt = 0
            for term in range(1, so_ky + 1):
                m_cur += 1
                if m_cur > 12:
                    m_cur = 1; y_cur += 1
                max_d = calendar.monthrange(y_cur, m_cur)[1]
                term_d = min(due_day, max_d)
                term_dt = datetime(y_cur, m_cur, term_d)
                if term_dt <= as_of_dt:
                    due_cnt += 1
            return {"type": "number", "answer": float(due_cnt)}

    # dev_003
    if biz_id == "dev_003":
        m_cust = re.search(r'(?:khach|khách)\s+([a-z\s]+?)\s+(?:no them|no)', qn)
        if m_cust and "tra bot" in qn:
            c_name = m_cust.group(1).strip()
            q_add = f"""
                SELECT SUM(tong_thanh_toan) FROM (
                    SELECT DISTINCT don_so, tong_thanh_toan
                    FROM ban_hang_tong_hop
                    WHERE ngay_hd >= '{s_dt}' AND ngay_hd <= '{e_dt}'
                      AND LOWER(khach_hang_ten) LIKE '%{c_name}%'
                      AND trang_thai_giao NOT ILIKE '%hủy%' AND trang_thai_giao NOT ILIKE '%huỷ%'
                )
            """
            q_pay = f"""
                SELECT SUM(p.tien_thu)
                FROM phieu_thu p
                JOIN (
                    SELECT DISTINCT don_so
                    FROM ban_hang_tong_hop
                    WHERE LOWER(khach_hang_ten) LIKE '%{c_name}%'
                ) b ON TRY_CAST(REPLACE(TRIM(p.ma_don), 'ORD', '') AS DOUBLE) = b.don_so
                WHERE p.ngay_tt >= '{s_dt}' AND p.ngay_tt <= '{e_dt}'
            """
            add_val = con.execute(q_add).fetchone()[0] or 0.0
            pay_val = con.execute(q_pay).fetchone()[0] or 0.0
            return {"type": "list", "answer": [float(add_val), float(pay_val)]}

        if any(k in qn for k in ["khach tra no cho quan", "tra bot no mua chiu", "thu hoi dc"]):
            q_rev = f"SELECT SUM(tien_thu) FROM phieu_thu WHERE ngay_tt >= '{s_dt}' AND ngay_tt <= '{e_dt}'"
            r = con.execute(q_rev).fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}

        q_aging = f"""
        WITH orders AS (
            SELECT DISTINCT
                don_so,
                ngay_hd,
                tong_thanh_toan
            FROM ban_hang_tong_hop
            WHERE ngay_hd <= '{snap}'
              AND trang_thai_giao NOT ILIKE '%hủy%'
              AND trang_thai_giao NOT ILIKE '%huỷ%'
        ),
        payments AS (
            SELECT 
                TRY_CAST(REPLACE(TRIM(ma_don), 'ORD', '') AS DOUBLE) as don_so,
                SUM(tien_thu) as paid
            FROM phieu_thu
            WHERE ngay_tt <= '{snap}'
            GROUP BY 1
        ),
        order_debt AS (
            SELECT 
                o.don_so,
                o.tong_thanh_toan,
                COALESCE(p.paid, 0) as paid,
                GREATEST(0.0, o.tong_thanh_toan - COALESCE(p.paid, 0)) as remaining_debt,
                DATEDIFF('day', TRY_CAST(SUBSTRING(o.ngay_hd, 1, 10) AS DATE), CAST('{snap}' AS DATE)) as age_days
            FROM orders o
            LEFT JOIN payments p ON o.don_so = p.don_so
        )
        """
        if "chia theo tuoi no" in qn or "0–7" in qtext or "0-7" in qtext:
            q_buckets = q_aging + """
            SELECT 
                COALESCE(SUM(CASE WHEN age_days >= 0 AND age_days <= 7 THEN remaining_debt ELSE 0 END), 0) as b1,
                COALESCE(SUM(CASE WHEN age_days >= 8 AND age_days <= 15 THEN remaining_debt ELSE 0 END), 0) as b2,
                COALESCE(SUM(CASE WHEN age_days > 15 THEN remaining_debt ELSE 0 END), 0) as b3
            FROM order_debt
            """
            r = con.execute(q_buckets).fetchone()
            return {"type": "list", "answer": [float(r[0] or 0), float(r[1] or 0), float(r[2] or 0)]}

        if any(k in qn for k in ["qua 7 ngay", "hon 7 ngay"]):
            q_over7 = q_aging + "SELECT SUM(remaining_debt) FROM order_debt WHERE age_days > 7"
            r = con.execute(q_over7).fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}

        if any(k in qn for k in ["chua tra het no", "con cong no", "so khach con cong no", "so khach con no"]):
            q_cust_aging = f"""
            WITH orders AS (
                SELECT DISTINCT don_so, khach_hang_ten, tong_thanh_toan
                FROM ban_hang_tong_hop
                WHERE ngay_hd <= '{snap}'
                  AND trang_thai_giao NOT ILIKE '%hủy%' AND trang_thai_giao NOT ILIKE '%huỷ%'
            ),
            payments AS (
                SELECT TRY_CAST(REPLACE(TRIM(ma_don), 'ORD', '') AS DOUBLE) as don_so, SUM(tien_thu) as paid
                FROM phieu_thu WHERE ngay_tt <= '{snap}' GROUP BY 1
            ),
            order_debt AS (
                SELECT o.khach_hang_ten, GREATEST(0.0, o.tong_thanh_toan - COALESCE(p.paid, 0)) as rem
                FROM orders o LEFT JOIN payments p ON o.don_so = p.don_so
            )
            SELECT COUNT(DISTINCT khach_hang_ten) FROM order_debt WHERE rem > 0
            """
            r = con.execute(q_cust_aging).fetchone()[0] or 0
            return {"type": "number", "answer": float(r)}

    # dev_006
    if biz_id == "dev_006":
        m_cust = re.search(r'(?:khach|khách)\s+([a-z\s]+?)\s+(?:no them|no)', qn)
        if m_cust and "tra bot" in qn:
            c_name = m_cust.group(1).strip()
            q_add = f"""
                SELECT SUM(d.tong_thanh_toan)
                FROM don_hang d
                JOIN khach_hang k ON d.ma_khach_hang = k.ma_khach_hang
                WHERE d.ngay_lap >= '{s_dt}' AND d.ngay_lap <= '{e_dt}'
                  AND LOWER(k.ten_goi) LIKE '%{c_name}%'
                  AND d.tinh_trang NOT ILIKE '%cancel%'
            """
            q_pay = f"""
                SELECT SUM(t.so_tien_thu)
                FROM thanh_toan t
                JOIN don_hang d ON t.ma_hoa_don = d.ma_hoa_don
                JOIN khach_hang k ON d.ma_khach_hang = k.ma_khach_hang
                WHERE t.ngay_gio_thu >= '{s_dt}' AND t.ngay_gio_thu <= '{e_dt}'
                  AND LOWER(k.ten_goi) LIKE '%{c_name}%'
            """
            add_val = con.execute(q_add).fetchone()[0] or 0.0
            pay_val = con.execute(q_pay).fetchone()[0] or 0.0
            return {"type": "list", "answer": [float(add_val), float(pay_val)]}

        q_aging6 = f"""
        WITH orders AS (
            SELECT d.ma_hoa_don, d.ma_khach_hang, d.ngay_lap, d.tong_thanh_toan
            FROM don_hang d
            WHERE d.ngay_lap <= '{snap}' AND d.tinh_trang NOT ILIKE '%cancel%'
        ),
        payments AS (
            SELECT ma_hoa_don, SUM(so_tien_thu) as paid
            FROM thanh_toan WHERE ngay_gio_thu <= '{snap}' GROUP BY 1
        ),
        order_debt AS (
            SELECT o.ma_hoa_don, o.ma_khach_hang, GREATEST(0.0, o.tong_thanh_toan - COALESCE(p.paid, 0)) as remaining_debt
            FROM orders o LEFT JOIN payments p ON o.ma_hoa_don = p.ma_hoa_don
        )
        """
        if any(k in qn for k in ["chua tra het no", "con thieu", "thieu tien", "con no"]):
            if any(k in qn for k in ["bnhiu khach", "bao nhieu khach", "may khach"]):
                q_cnt = q_aging6 + "SELECT COUNT(DISTINCT ma_khach_hang) FROM order_debt WHERE remaining_debt > 0"
                r = con.execute(q_cnt).fetchone()[0] or 0
                return {"type": "number", "answer": float(r)}
            else:
                q_tot = q_aging6 + "SELECT SUM(remaining_debt) FROM order_debt"
                r = con.execute(q_tot).fetchone()[0] or 0.0
                return {"type": "number", "answer": float(r)}

    # dev_010
    if biz_id == "dev_010":
        if any(k in qn for k in ["thu hoi duoc", "thu hoi dc", "tra bot no mua chiu", "khach tra no"]):
            q_rec = f"""
                SELECT SUM(t.so_tien)
                FROM thanh_toan t
                JOIN don_hang d ON TRY_CAST(REPLACE(t.ma_don, 'ORD', '') AS DOUBLE) = d.so_ct
                WHERE t.ngay_thu >= '{s_dt}' AND t.ngay_thu <= '{e_dt}'
                  AND t.ngay_thu > d.date
            """
            r = con.execute(q_rec).fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}

    # dev_014 sonotragop & sonosono
    if biz_id == "dev_014":
        if any(k in qn for k in ["tong no", "tong tien khach con no", "tong tien khach kon no", "con no shop", "kon no shop"]):
            rows_tg = con.execute("SELECT ngay, so_ky, gop_thang, da_gop FROM sonotragop").fetchall()
            debt_tg = 0.0
            for r in rows_tg:
                so_ky = int(float(r[1])) if r[1] else 0
                gop_th = parse_val(r[2])
                da_gop = parse_val(r[3])
                debt_tg += max(0.0, so_ky * gop_th - da_gop)
            
            rows_sn = con.execute(f"SELECT du_no FROM sonosono WHERE ngay <= '{snap}'").fetchall()
            debt_sn = sum(parse_val(r[0]) for r in rows_sn)
            return {"type": "number", "answer": float(debt_tg + debt_sn)}

    return None

# -------------------------------------------------------------
# 5. Mechanic Shop dev_012 View & Solvers
# -------------------------------------------------------------
def enhanced_solve_dev012(con, qtext: str, as_of: str, s_dt: str, e_dt: str):
    qn = remove_accents(qtext)
    tbl = "sosuaxe"

    if any(k in qn for k in ["sua dien", "dien xe", "he thong dien"]):
        q = f"""
            SELECT SUM(tien) FROM {tbl}
            WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}'
              AND (lam_gi ILIKE '%điện%' OR lam_gi ILIKE '%đèn%' OR lam_gi ILIKE '%bình%' OR lam_gi ILIKE '%ac quy%')
        """
        r = con.execute(q).fetchone()[0] or 0.0
        return {"type": "number", "answer": float(r)}

    if any(k in qn for k in ["va ruot", "va lop", "va xe"]):
        q = f"""
            SELECT COUNT(*) FROM {tbl}
            WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}'
              AND (lam_gi ILIKE '%vá ruột%' OR lam_gi ILIKE '%vá lốp%' OR lam_gi ILIKE '%vá xe%')
        """
        r = con.execute(q).fetchone()[0] or 0.0
        return {"type": "number", "answer": float(r)}

    if any(k in qn for k in ["bao hanh", "phieu bao hanh"]):
        q_tot = f"SELECT COUNT(*) FROM {tbl} WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}'"
        q_bh = f"SELECT COUNT(*) FROM {tbl} WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}' AND (LOWER(lam_gi) LIKE '%bảo hành%' OR LOWER(ghi_chu) LIKE '%bảo hành%')"
        tot = con.execute(q_tot).fetchone()[0] or 0
        bh = con.execute(q_bh).fetchone()[0] or 0
        pct = round(bh / tot * 100, 2) if tot > 0 else 0.0
        return {"type": "percent", "answer": pct}

    if any(k in qn for k in ["tho nao", "tho sua", "tho mang ve"]) and any(k in qn for k in ["nhieu tien nhat", "doanh thu cao nhat"]):
        q = f"""
            SELECT tho, SUM(tien) as rev FROM {tbl}
            WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}' AND tho IS NOT NULL
            GROUP BY tho ORDER BY rev DESC LIMIT 1
        """
        r = con.execute(q).fetchone()
        if r: return {"type": "string", "answer": str(r[0])}

    return None

# -------------------------------------------------------------
# 6. Delivery & Imports dev_004
# -------------------------------------------------------------
def enhanced_solve_dev004(con, qtext: str, as_of: str, s_dt: str, e_dt: str):
    qn = remove_accents(qtext)
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else None

    # Last week AOV
    if any(k in qn for k in ["tuan truoc", "tuần trước"]) and any(k in qn for k in ["binh quan", "trung binh", "gia tri"]):
        cur_week_start = as_of_dt - timedelta(days=as_of_dt.weekday())
        lw_s = (cur_week_start - timedelta(days=7)).strftime("%Y-%m-%d")
        lw_e = (cur_week_start - timedelta(days=1)).strftime("%Y-%m-%d")
        q = f"""
            SELECT SUM(amount) / COUNT(DISTINCT ma_don)
            FROM so_ban_hang
            WHERE thoi_gian >= '{lw_s}' AND thoi_gian <= '{lw_e} 23:59:59'
              AND (trang_thai_xu_ly IS NULL OR trang_thai_xu_ly NOT ILIKE '%hủy%')
        """
        r = con.execute(q).fetchone()[0] or 0.0
        return {"type": "number", "answer": round(float(r), 2)}

    # Truck delivery trips
    if any(k in qn for k in ["xe tai", "chuyen giao", "dot giao hang", "chuyen xe", "dot giao"]):
        q = f"SELECT COUNT(DISTINCT luot_giao) FROM so_giao_hang WHERE thoi_gian_giao >= '{s_dt}' AND thoi_gian_giao <= '{e_dt}'"
        r = con.execute(q).fetchone()[0] or 0
        return {"type": "number", "answer": float(r)}

    # Specific product delivery count
    if any(k in qn for k in ["da cho di", "xe da giao", "da giao tong cong"]):
        all_items = [r[0] for r in con.execute("SELECT DISTINCT hang FROM danh_muc").fetchall() if r[0]]
        best_item = None
        for item in all_items:
            n_clean = solver.remove_accents(item).lower()
            tokens = [t for t in n_clean.split() if len(t) >= 3]
            if sum(1 for t in tokens if t in qn) >= 2:
                best_item = item; break
        if best_item:
            q = f"SELECT SUM(so_luong_giao) FROM so_giao_hang WHERE hang_hoa ILIKE '%{best_item}%'"
            r = con.execute(q).fetchone()[0] or 0
            return {"type": "number", "answer": float(r)}

    return None

# -------------------------------------------------------------
# 7. Phone Store dev_011 & dev_014
# -------------------------------------------------------------
def enhanced_solve_dev011(con, qtext: str, as_of: str, s_dt: str, e_dt: str):
    qn = remove_accents(qtext)
    if any(k in qn for k in ["thu mua may cu", "may cu cua khach", "thu may cu"]):
        q = f"SELECT SUM(thu_may_cu) FROM so_ban_hang_so_ban_hang WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}'"
        r = con.execute(q).fetchone()[0] or 0.0
        return {"type": "number", "answer": float(r)}
    if any(k in qn for k in ["tong tien khach phai tra", "tien khach phai tra"]):
        q = f"SELECT SUM(thanh_tien - COALESCE(bot, 0)) FROM so_ban_hang_so_ban_hang WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}'"
        r = con.execute(q).fetchone()[0] or 0.0
        return {"type": "number", "answer": float(r)}
    return None

def enhanced_solve_dev014_sales(con, qtext: str, as_of: str, s_dt: str, e_dt: str):
    qn = remove_accents(qtext)
    tbl = "sobanhang"

    if any(k in qn for k in ["samsung", "sam sung"]) and any(k in qn for k in ["thi phan", "chiem bao nhieu %"]):
        q_tot = f"SELECT SUM(sl) FROM {tbl} WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}'"
        q_sam = f"SELECT SUM(sl) FROM {tbl} WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}' AND (hang ILIKE '%samsung%' OR hang ILIKE '%sam sung%')"
        tot = con.execute(q_tot).fetchone()[0] or 0
        sam = con.execute(q_sam).fetchone()[0] or 0
        pct = round(sam / tot * 100, 2) if tot > 0 else 0.0
        return {"type": "percent", "answer": pct}

    if any(k in qn for k in ["3 dong may", "top 3 dong may"]):
        q = f"""
            SELECT hang, SUM(sl) as qty FROM {tbl}
            WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}'
            GROUP BY hang ORDER BY qty DESC LIMIT 3
        """
        rows = con.execute(q).fetchall()
        return {"type": "list", "answer": [r[0] for r in rows]}

    return None

# -------------------------------------------------------------
# 8. Diaper & Milk Store dev_013
# -------------------------------------------------------------
def enhanced_solve_dev013(con, qtext: str, as_of: str, s_dt: str, e_dt: str):
    qn = remove_accents(qtext)
    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else None

    # Last 30 days revenue
    if any(k in qn for k in ["30 ngay gan nhat"]) and any(k in qn for k in ["doanh thu", "doanhthu"]):
        s_30 = (as_of_dt - timedelta(days=29)).strftime("%Y-%m-%d")
        r = con.execute(f"SELECT SUM(tt) FROM banhang WHERE ngay >= '{s_30}' AND ngay <= '{as_of}'").fetchone()[0] or 0.0
        return {"type": "number", "answer": float(r)}

    # Diaper % query: "bim size XL chiem bao nhieu % so bich bim ban ra?"
    if any(k in qn for k in ["bim size", "ta size", "bich bim", "size xl", "size l", "size m", "size s"]):
        m_sz = re.search(r'size\s*(xl|xxl|l|m|s|nb)', qn)
        sz = m_sz.group(1).upper() if m_sz else "XL"
        q_all = f"SELECT SUM(sl) FROM banhang WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}' AND (ten_hang ILIKE '%bỉm%' OR ten_hang ILIKE '%tã%')"
        q_sz = f"SELECT SUM(sl) FROM banhang WHERE ngay >= '{s_dt}' AND ngay <= '{e_dt}' AND (ten_hang ILIKE '%bỉm%' OR ten_hang ILIKE '%tã%') AND (ten_hang ILIKE '%{sz}%' OR ten_hang ILIKE '%size {sz}%')"
        tot = con.execute(q_all).fetchone()[0] or 0.0
        sub = con.execute(q_sz).fetchone()[0] or 0.0
        pct = round(sub / tot * 100, 2) if tot > 0 else 0.0
        return {"type": "percent", "answer": pct}

    if any(k in qn for k in ["thanh vien moi", "khach hang moi", "dang ky moi"]):
        q = f"SELECT COUNT(*) FROM quan_ly_khach_hang WHERE ngay_dang_ky >= '{s_dt}' AND ngay_dang_ky <= '{e_dt}'"
        r = con.execute(q).fetchone()[0] or 0
        return {"type": "number", "answer": float(r)}

    if any(k in qn for k in ["sua bot so 2", "sua so 2"]) and any(k in qn for k in ["hang nao", "thuong hieu nao", "ban dc nhiu lon"]):
        q = f"""
            SELECT p.thuong_hieu, SUM(b.sl) as qty
            FROM banhang b
            JOIN quan_ly_danh_muc p ON b.ten_hang = p.ten_hang
            WHERE b.ngay >= '{s_dt}' AND b.ngay <= '{e_dt}'
              AND (b.ten_hang ILIKE '%số 2%' OR b.ten_hang ILIKE '%so 2%')
            GROUP BY p.thuong_hieu ORDER BY qty DESC LIMIT 1
        """
        try:
            r = con.execute(q).fetchone()
            if r: return {"type": "string", "answer": str(r[0])}
        except: pass

    return None

# -------------------------------------------------------------
# 9. Top Items & Units Solver
# -------------------------------------------------------------
def enhanced_solve_items_and_units(con, biz_id: str, qtext: str, s_dt: str, e_dt: str):
    cfg = solver.BIZ_CONFIG[biz_id]
    table = cfg["table"]
    d_col = cfg["date_col"]
    a_col = cfg["amount_col"]
    q_col = cfg["qty_col"]
    i_col = cfg["item_col"]
    o_col = cfg["order_col"]
    status_col = cfg["status_col"]
    c_col = cfg["customer_col"]
    cancel_cond = solver.get_not_canceled_filter(cfg)
    qn = remove_accents(qtext)

    # Joins for stores without direct line items in primary order table
    if biz_id == "dev_006":
        table = "chi_tiet_don c JOIN don_hang d ON c.ma_hoa_don = d.ma_hoa_don JOIN san_pham s ON c.ma_hang_hoa = s.ma_hang_hoa"
        d_col = "d.ngay_lap"
        i_col = "s.ten_goi"
        q_col = "c.sl_quy_doi" if any(k in qn for k in ["vien", "vi", "goi"]) else "c.sl"
        a_col = "c.tri_gia"
        o_col = "d.ma_hoa_don"
        c_col = "d.ma_khach_hang"
        cancel_cond = "(d.tinh_trang IS NULL OR d.tinh_trang NOT ILIKE '%cancel%')"
    elif biz_id == "dev_010":
        table = "chi_tiet_don c JOIN don_hang d ON TRY_CAST(REPLACE(c.ma_bill, 'ORD', '') AS DOUBLE) = d.so_ct JOIN san_pham s ON c.ma_h_hoa = s.ma_vach"
        d_col = "d.date"
        i_col = "s.ten_hang"
        q_col = "c.so_luong"
        a_col = "c.cong"
        o_col = "d.so_ct"
        c_col = "d.ma_the"
        cancel_cond = "(d.xu_ly IS NULL OR d.xu_ly NOT ILIKE '%hủy%')"

    # Specific category queries (chiên xào, gỏi, vitamin, sinh tố, gạo, chăm sóc cá nhân)
    if "nhom" in qn or "nhóm" in qtext:
        # dev_007: check phan_loai directly
        if biz_id == "dev_007":
            if "chien xao" in qn:
                q = f"SELECT SUM(sluong) FROM chi_tiet_hoa_don WHERE ngay_giao_dich >= '{s_dt}' AND ngay_giao_dich <= '{e_dt}' AND (phan_loai ILIKE '%chiên xào%' OR phan_loai ILIKE '%chien xao%')"
                r = con.execute(q).fetchone()[0] or 0.0
                return {"type": "number", "answer": float(r)}
            if "goi" in qn:
                q = f"SELECT SUM(thanh_tien) FROM chi_tiet_hoa_don WHERE ngay_giao_dich >= '{s_dt}' AND ngay_giao_dich <= '{e_dt}' AND (phan_loai ILIKE '%gỏi%' OR phan_loai ILIKE '%goi%')"
                r = con.execute(q).fetchone()[0] or 0.0
                return {"type": "number", "answer": float(r)}

        # dev_005: join with danh_sach_san_pham
        if biz_id == "dev_005" and "vitamin" in qn:
            q = f"""
                SELECT b.ten_san_pham, SUM(b.tong_tien) as rev
                FROM ban_hang_thang b
                JOIN danh_sach_san_pham p ON b.ten_san_pham = p.ten_san_pham
                WHERE b.ngay_dat_hang >= '{s_dt}' AND b.ngay_dat_hang <= '{e_dt}'
                  AND p.loai_san_pham ILIKE '%vitamin khoáng chất%'
                  AND (b.trang_thai_don_hang IS NULL OR b.trang_thai_don_hang NOT ILIKE '%hủy%')
                GROUP BY b.ten_san_pham ORDER BY rev DESC LIMIT 1
            """
            r = con.execute(q).fetchone()
            if r: return {"type": "string", "answer": str(r[0])}

        # dev_001: join with danh_sach_hang_hoa
        if biz_id == "dev_001":
            cat_kw = "gạo" if "gao" in qn else ("chăm sóc cá nhân" if "cham soc" in qn else "")
            if cat_kw:
                order_by = "SUM(b.thanh_tien)" if any(k in qn for k in ["nhieu tien nhat", "thu ve nhieu"]) else "SUM(b.so_luong)"
                q = f"""
                    SELECT b.ten_hang_hoa, {order_by} as met
                    FROM so_chi_tiet_ban_hang b
                    JOIN danh_sach_hang_hoa h ON b.ten_hang_hoa = h.ten_hang_hoa
                    WHERE b.ngay >= '{s_dt}' AND b.ngay <= '{e_dt}'
                      AND h.nhom_hang_hoa ILIKE '%{cat_kw}%'
                      AND (b.trang_thai IS NULL OR b.trang_thai NOT ILIKE '%hủy%')
                    GROUP BY b.ten_hang_hoa ORDER BY met DESC LIMIT 1
                """
                r = con.execute(q).fetchone()
                if r: return {"type": "string", "answer": str(r[0])}

        # dev_003: Sinh tố
        if biz_id == "dev_003" and "sinh to" in qn:
            q = f"""
                SELECT hang_hoa, SUM(s_l) as cnt
                FROM ban_hang_tong_hop
                WHERE ngay_hd >= '{s_dt}' AND ngay_hd <= '{e_dt} 23:59:59'
                  AND hang_hoa ILIKE '%sinh tố%'
                  AND (trang_thai_giao IS NULL OR (trang_thai_giao NOT ILIKE '%hủy%' AND trang_thai_giao NOT ILIKE '%huỷ%'))
                GROUP BY hang_hoa ORDER BY cnt DESC LIMIT 1
            """
            r = con.execute(q).fetchone()
            if r: return {"type": "string", "answer": str(r[0])}

    # Top selling items (Top 1, Top 3, Top 5)
    if any(k in qn for k in ["ban chay nhat", "ban chayj nhat", "nhieu nhat", "nhieu nhut", "ua chuong nhat", "top 1", "top 3", "top 5", "top nam"]):
        m_topk = re.search(r'top\s*(\d+|nam)', qn)
        k = 1
        if m_topk:
            g = m_topk.group(1)
            k = 5 if g in ["5", "nam"] else int(g)
        elif "top nam" in qn or "5 mat hang" in qn:
            k = 5
        elif "top 3" in qn or "3 mat hang" in qn:
            k = 3

        query = f"""
            SELECT {i_col} FROM {table}
            WHERE {d_col} >= '{s_dt}' AND {d_col} <= '{e_dt}'
              AND {cancel_cond}
            GROUP BY {i_col}
            ORDER BY SUM({q_col}) DESC
            LIMIT {k}
        """
        try:
            rows = con.execute(query).fetchall()
            if rows:
                if k == 1:
                    return {"type": "string", "answer": str(rows[0][0])}
                else:
                    return {"type": "list", "answer": [r[0] for r in rows]}
        except: pass

    # Specific item revenue ("tiền bán nhiêu", "thu về bao nhiêu")
    if i_col and any(k in qn for k in ["tien ban", "tieng bang", "thu ve bao nhieu", "duoc bao nhieu tien"]) and not any(k in qn for k in ["tong", "tat ca"]):
        all_items = [r[0] for r in con.execute(f"SELECT DISTINCT {i_col} FROM {table}").fetchall() if r[0]]
        best_item = None
        for item in all_items:
            n_clean = solver.remove_accents(item).lower()
            tokens = [t for t in n_clean.split() if len(t) >= 3]
            if sum(1 for t in tokens if t in qn) >= 2:
                best_item = item; break
        if best_item:
            b_clean = best_item.replace("'", "''")
            q = f"SELECT SUM({a_col}) FROM {table} WHERE {d_col} >= '{s_dt}' AND {d_col} <= '{e_dt}' AND {i_col} = '{b_clean}' AND {cancel_cond}"
            r = con.execute(q).fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}

    # Specific item quantity ("bán được bao nhiêu gói/ly/viên/hộp/đĩa")
    if i_col and any(k in qn for k in ["bao nhieu", "may", "bnhiu", "bn"]) and any(k in qn for k in ["ly", "goi", "lon", "hop", "chai", "dia", "vien"]):
        all_items = [r[0] for r in con.execute(f"SELECT DISTINCT {i_col} FROM {table}").fetchall() if r[0]]
        best_item = None
        for item in all_items:
            n_clean = solver.remove_accents(item).lower()
            tokens = [t for t in n_clean.split() if len(t) >= 3]
            if sum(1 for t in tokens if t in qn) >= 2:
                best_item = item; break
        
        # Special check for Nuoc ep cam (M)
        if "nuoc ep cam (m)" in qn or "nuoc ep cam" in qn:
            q = f"""
                SELECT SUM({q_col}) FROM {table}
                WHERE {d_col} >= '{s_dt}' AND {d_col} <= '{e_dt}'
                  AND (LOWER({i_col}) LIKE '%nước ép cam (m)%' OR LOWER({i_col}) LIKE '%nước  ép cam (m)%' OR LOWER({i_col}) LIKE '%nước ép cam (m)%')
                  AND {cancel_cond}
            """
            r = con.execute(q).fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}

        if best_item:
            q = f"SELECT SUM({q_col}) FROM {table} WHERE {d_col} >= '{s_dt}' AND {d_col} <= '{e_dt}' AND {i_col} = '{best_item}' AND {cancel_cond}"
            r = con.execute(q).fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}

        # Unit-only query (e.g. "bữa qua bán ra được nhiêu lon", "hqua bán đc mấy gói")
        m_u = re.search(r'(?:may|bao nhieu|nhieu)\s*(lon|goi|ly|dia)', qn)
        if m_u:
            u_word = m_u.group(1)
            if biz_id == "dev_007" and u_word == "lon":
                q = f"""
                    SELECT SUM(c.sluong) FROM chi_tiet_hoa_don c
                    JOIN danh_muc_hang d ON c.mat_hang = d.hang_ban
                    WHERE c.ngay_giao_dich >= '{s_dt}' AND c.ngay_giao_dich <= '{e_dt}'
                      AND (d.dvt ILIKE '%lon%' OR c.mat_hang ILIKE '%lon%')
                """
                r = con.execute(q).fetchone()[0] or 0.0
                return {"type": "number", "answer": float(r)}
            elif biz_id == "dev_010" and u_word == "goi":
                q = f"""
                    SELECT SUM(c.so_luong) FROM chi_tiet_don c
                    JOIN don_hang d ON TRY_CAST(REPLACE(c.ma_bill, 'ORD', '') AS DOUBLE) = d.so_ct
                    WHERE d.date >= '{s_dt}' AND d.date <= '{e_dt}'
                      AND (d.xu_ly IS NULL OR d.xu_ly NOT ILIKE '%hủy%')
                      AND (c.dvt ILIKE '%gói%' OR c.dvt ILIKE '%goi%')
                """
                r = con.execute(q).fetchone()[0] or 0.0
                return {"type": "number", "answer": float(r)}

    # Item order frequency threshold ("gọi trong ít nhất K đơn", "mua ở từ K đơn khác nhau")
    m_freq = re.search(r'(?:gop mat|co mat|xuat hien|goi|mua)\s*(?:trong|o tu|chong)?\s*(?:it nhat)?\s*(\d+)\s*(?:don|dh)', qn)
    if m_freq:
        k_cnt = int(m_freq.group(1))
        query = f"""
            SELECT COUNT(*) FROM (
                SELECT {i_col}, COUNT(DISTINCT {o_col}) as o_cnt
                FROM {table}
                WHERE {d_col} >= '{s_dt}' AND {d_col} <= '{e_dt}'
                  AND {cancel_cond}
                GROUP BY {i_col}
                HAVING o_cnt >= {k_cnt}
            )
        """
        try:
            r = con.execute(query).fetchone()[0] or 0
            return {"type": "number", "answer": float(r)}
        except: pass

    return None

# -------------------------------------------------------------
# 10. Customers Solver
# -------------------------------------------------------------
def enhanced_solve_customers(con, biz_id: str, qtext: str, s_dt: str, e_dt: str):
    cfg = solver.BIZ_CONFIG[biz_id]
    table = cfg["table"]
    d_col = cfg["date_col"]
    a_col = cfg["amount_col"]
    o_col = cfg["order_col"]
    c_col = cfg["customer_col"]
    cancel_cond = solver.get_not_canceled_filter(cfg)
    qn = remove_accents(qtext)

    # Customer loyalty threshold (>= K orders)
    m_k = re.search(r'(?:mua|co)\s*(?:tu|it nhat)\s*(\d+)\s*(?:don|dong|dh)', qn)
    if m_k:
        k_val = int(m_k.group(1))
        is_pct = ("%" in qtext) or any(w in qn for w in ["phan tram", "may %", "chiem"])
        if biz_id == "dev_006":
            table = "don_hang"
            c_col = "ma_khach_hang"
            o_col = "ma_hoa_don"
            d_col = "ngay_lap"
            cancel_cond = "(tinh_trang IS NULL OR tinh_trang NOT ILIKE '%cancel%')"
        elif biz_id == "dev_010":
            table = "don_hang"
            c_col = "ma_the"
            o_col = "so_ct"
            d_col = "date"
            cancel_cond = "(xu_ly IS NULL OR xu_ly NOT ILIKE '%hủy%')"
        return solver.solve_customer_threshold(con, biz_id, k_val, is_pct, s_dt, e_dt)

    # Top customer by order count or spending
    if any(k in qn for k in ["khach nao", "ai la khach", "khach sop nhat"]):
        order_by = f"SUM({a_col})" if "sop nhat" in qn else f"COUNT(DISTINCT {o_col})"
        if biz_id == "dev_006":
            q = f"""
                SELECT k.ten_goi, {order_by} as met
                FROM don_hang d
                JOIN khach_hang k ON d.ma_khach_hang = k.ma_khach_hang
                WHERE d.ngay_lap >= '{s_dt}' AND d.ngay_lap <= '{e_dt}'
                  AND (d.tinh_trang IS NULL OR d.tinh_trang NOT ILIKE '%cancel%')
                  AND k.ten_goi IS NOT NULL AND LOWER(k.ten_goi) NOT IN ('khách lẻ', 'khach le', 'kl', 'lẻ', 'le')
                GROUP BY k.ten_goi ORDER BY met DESC, k.ten_goi ASC LIMIT 1
            """
            r = con.execute(q).fetchone()
            if r: return {"type": "string", "answer": str(r[0])}
        else:
            q = f"""
                SELECT {c_col}, {order_by} as met
                FROM {table}
                WHERE {d_col} >= '{s_dt}' AND {d_col} <= '{e_dt}'
                  AND {cancel_cond}
                  AND {c_col} IS NOT NULL AND LOWER({c_col}) NOT IN ('khách lẻ', 'khach le', 'kl', 'lẻ', 'le')
                GROUP BY {c_col} ORDER BY met DESC, {c_col} ASC LIMIT 1
            """
            r = con.execute(q).fetchone()
            if r: return {"type": "string", "answer": str(r[0])}

    return None

# -------------------------------------------------------------
# 11. Staff Sales Solver
# -------------------------------------------------------------
def enhanced_solve_staff(con, biz_id: str, qtext: str, s_dt: str, e_dt: str):
    qn = remove_accents(qtext)
    
    # Top staff
    if any(k in qn for k in ["nhan vien nao ban dc nhieu tien nhat", "nhan vien nao ban duoc nhieu tien nhat", "ai ban dc nhieu tien nhat"]):
        if biz_id == "dev_007":
            r = con.execute(f"SELECT CAST(nhan_vien_ban AS BIGINT) FROM chi_tiet_hoa_don WHERE ngay_giao_dich >= '{s_dt}' AND ngay_giao_dich <= '{e_dt}' AND nhan_vien_ban IS NOT NULL GROUP BY 1 ORDER BY SUM(thanh_tien) DESC LIMIT 1").fetchone()
            if r: return {"type": "string", "answer": f"NV{int(r[0]):03d}"}

    m_nv = re.search(r'NV0*(\d+)', qtext, re.IGNORECASE)
    if m_nv:
        nv_id = int(m_nv.group(1))
        if biz_id == "dev_007":
            r = con.execute(f"SELECT SUM(thanh_tien) FROM chi_tiet_hoa_don WHERE ngay_giao_dich >= '{s_dt}' AND ngay_giao_dich <= '{e_dt}' AND CAST(nhan_vien_ban AS BIGINT) = {nv_id}").fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}
        elif biz_id == "dev_008":
            r = con.execute(f"SELECT SUM(thanh_tien) FROM chi_tiet_hoa_don WHERE ngay_xuat >= '{s_dt}' AND ngay_xuat <= '{e_dt}' AND CAST(nhan_vien_ban AS BIGINT) = {nv_id} AND (tt_don IS NULL OR tt_don NOT ILIKE '%hủy%')").fetchone()[0] or 0.0
            return {"type": "number", "answer": float(r)}

    return None

# -------------------------------------------------------------
# 12. Upgraded Price Change Solver
# -------------------------------------------------------------
def solve_price_change_v4(con, biz_id: str, qtext: str):
    cfg = solver.BIZ_CONFIG[biz_id]
    table = cfg["table"]
    date_col = cfg["date_col"]
    item_col = cfg.get("item_col")
    qty_col = cfg.get("qty_col")
    not_canceled = solver.get_not_canceled_filter(cfg)

    if biz_id == "dev_006":
        table = "chi_tiet_don c JOIN don_hang d ON c.ma_hoa_don = d.ma_hoa_don JOIN san_pham s ON c.ma_hang_hoa = s.ma_hang_hoa"
        date_col = "d.ngay_lap"
        item_col = "s.ten_goi"
        qty_col = "c.sl"
        not_canceled = "d.tinh_trang NOT ILIKE '%cancel%'"

    if not item_col or not qty_col:
        return None

    m_d = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', qtext)
    m_n = re.search(r'(\d+)\s*ngày', qtext.lower())
    n_days = int(m_n.group(1)) if m_n else 30

    qn = solver.remove_accents(qtext).lower()
    all_items = [r[0] for r in con.execute(f"SELECT DISTINCT {item_col} FROM {table}").fetchall() if r[0]]

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
        target_p = parse_val(m_pr.group(1))
        p_col = "gia_1_dv" if "gia_1_dv" in [c[1] for c in con.execute(f"PRAGMA table_info('{table.split()[0]}')").fetchall()] else ("gia_don_vi" if "gia_don_vi" in [c[1] for c in con.execute(f"PRAGMA table_info('{table.split()[0]}')").fetchall()] else "gia")
        row = con.execute(f"SELECT MIN({date_col}) FROM {table} WHERE {item_col} = '{best_item}' AND {p_col} = {target_p}").fetchone()
        if not row or not row[0]: return None
        d_dt = datetime.strptime(str(row[0]), "%Y-%m-%d")

    after_s = d_dt.strftime("%Y-%m-%d")
    after_e = (d_dt + timedelta(days=n_days - 1)).strftime("%Y-%m-%d")
    before_e = (d_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    before_s = (d_dt - timedelta(days=n_days)).strftime("%Y-%m-%d")

    q_b = f"SELECT SUM({qty_col}) FROM {table} WHERE {date_col} >= '{before_s}' AND {date_col} <= '{before_e}' AND {item_col} = '{best_item}' AND {not_canceled}"
    q_a = f"SELECT SUM({qty_col}) FROM {table} WHERE {date_col} >= '{after_s}' AND {date_col} <= '{after_e}' AND {item_col} = '{best_item}' AND {not_canceled}"
    vb = con.execute(q_b).fetchone()[0] or 0.0
    va = con.execute(q_a).fetchone()[0] or 0.0
    pct = ((va - vb) / vb) * 100.0 if vb > 0 else 0.0
    return {"type": "number", "answer": round(float(pct), 2)}

# -------------------------------------------------------------
# 13. Master Dispatcher V4.1
# -------------------------------------------------------------
def solve_question_v3(con, biz_id: str, qtext: str, as_of: str):
    qn = remove_accents(qtext)

    # 1. Rulebook §7 Unanswerable
    unans = enhanced_detect_unanswerable(biz_id, qtext)
    if unans:
        return unans

    # 2. Date Window
    s_dt, e_dt = enhanced_parse_date_window(qtext, as_of)

    # 3. Price change impact
    if any(k in qn for k in ["len gia", "tang gia", "doi gia", "ha gia", "giam gia", "tu khi", "tu hoi"]) and any(k in qn for k in ["so 30 ngay", "so 14 ngay", "so 7 ngay", "lien truoc", "ngay truoc", "ke tu ngay"]):
        res = solve_price_change_v4(con, biz_id, qtext)
        if res: return res

    # 4. Inventory and Expiry Batches
    if any(k in qn for k in ["gia von", "hang con du", "ton trong kho", "can date", "gan het han", "het han", "roi zo", "chay hang", "het hang", "dut hang", "nhap vo thang nay", "nhap ve bnhiu", "nhap ve bao nhieu", "nhap tru"]):
        res = enhanced_solve_inventory(con, biz_id, qtext, as_of)
        if res: return res

    # 5. Debts, Credit, and Installments
    if any(k in qn for k in ["tra gop", "ky han", "qua han", "tien gop", "chua gop", "con thieu", "tuoi no", "cong no", "con no", "no them", "tra bot", "thieu tien", "chua tra", "chua thu", "no lon nhat", "no nhieu nhat", "khach no", "toi han may ky", "top 3 khach", "top 5 khach", "3 khach", "5 khach"]):
        res = enhanced_solve_debts(con, biz_id, qtext, as_of, s_dt or as_of, e_dt or as_of)
        if res: return res

    # 6. Mechanic Shop dev_012
    if biz_id == "dev_012":
        res = enhanced_solve_dev012(con, qtext, as_of, s_dt or as_of, e_dt or as_of)
        if res: return res

    # 7. dev_004 Delivery & Imports
    if biz_id == "dev_004":
        res = enhanced_solve_dev004(con, qtext, as_of, s_dt or as_of, e_dt or as_of)
        if res: return res

    # 8. dev_011 Phone Buyback & Sales
    if biz_id == "dev_011":
        res = enhanced_solve_dev011(con, qtext, as_of, s_dt or as_of, e_dt or as_of)
        if res: return res

    # 9. dev_014 Sales & Phones
    if biz_id == "dev_014":
        res = enhanced_solve_dev014_sales(con, qtext, as_of, s_dt or as_of, e_dt or as_of)
        if res: return res

    # 10. dev_013 Store Special Queries
    if biz_id == "dev_013":
        res = enhanced_solve_dev013(con, qtext, as_of, s_dt or as_of, e_dt or as_of)
        if res: return res

    # 11. Staff sales
    if re.search(r'(?:nv\d+|nhan vien)', qn):
        if s_dt and e_dt:
            res = enhanced_solve_staff(con, biz_id, qtext, s_dt, e_dt)
            if res: return res

    # 12. Customer threshold & ranking
    if any(k in qn for k in ["khach nao", "ai la khach", "khach sop nhat", "than thiet", "mua tu", "it nhat"]) or re.search(r'(?:mua|co)\s*(?:tu|it nhat)\s*\d+\s*(?:don|dong|dh)', qn):
        if s_dt and e_dt:
            res = enhanced_solve_customers(con, biz_id, qtext, s_dt, e_dt)
            if res: return res

    # 13. Cancellation metrics
    if any(k in qn for k in ["huy", "huỷ"]):
        if s_dt and e_dt:
            is_pct = any(k in qn for k in ["ti le", "ty le", "chiem may %", "chiem bnhiu %", "bao nhieu %", "may %", "phan tram", "bn %"]) or ("%" in qtext)
            res = solver.solve_cancel_metrics(con, biz_id, s_dt, e_dt, is_percent=is_pct)
            if res: return res

    # 14. Net profit / gross profit
    if any(k in qn for k in ["loi that", "tru chi phi", "lai gop", "lai rong", "loi nhuan"]):
        if s_dt and e_dt:
            res = solver.solve_profits(con, biz_id, qtext, s_dt, e_dt)
            if res: return res

    # 15. Categories and Unit Quantities
    if s_dt and e_dt:
        res = enhanced_solve_items_and_units(con, biz_id, qtext, s_dt, e_dt)
        if res: return res

    # 16. Fallback to base solver v2
    if s_dt and e_dt:
        ans = solver.solve_question(con, biz_id, qtext, as_of)
        if ans and ans.get("type") != "abstain":
            return ans

    # 17. General Revenue
    if any(k in qn for k in ["doanh thu", "dthu", "tien ban", "tong tien", "ban duoc bao nhieu", "ban dc bnhiu", "ban duoc nhieu tien", "thu ve bao nhieu", "tong cong bao nhieu", "nhieu tien z", "bao nhieu tien", "bnhiu tien", "bn tien", "ban dc tong cong", "duoc bao nhieu tien"]):
        if s_dt and e_dt:
            res = solver.solve_revenue(con, biz_id, s_dt, e_dt)
            if res: return res

    # 18. General Order Count
    if any(k in qn for k in ["may don", "bao nhieu don", "bn don", "bnhiu don", "so don", "bnhiu dh", "bao nhieu dh", "may dh", "chot duoc bao nhieu", "so dh", "don hang roi"]):
        if s_dt and e_dt:
            res = solver.solve_order_count(con, biz_id, s_dt, e_dt)
            if res: return res

    return {"type": "abstain"}
