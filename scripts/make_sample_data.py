"""Build a small synthetic DataMate-style test set with known answers.

Output:
  sample_data/data/questions.jsonl          (what the agent sees, like /data)
  sample_data/data/datasets/<business>/...  (CSV, messy Excel, SQLite, README)
  sample_data/gold.jsonl                    (answers, kept outside /data)

Real devkit data will be different; this exists to test the pipeline end to end.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1] / "sample_data"
DATA = ROOT / "data"
rng = random.Random(2026)


def cafe() -> list[tuple[str, str, object]]:
    d = DATA / "datasets" / "cafe_suongmai"
    d.mkdir(parents=True, exist_ok=True)
    products = pd.DataFrame([
        ("SP01", "Cà phê sữa đá", "Cà phê", 12000, 29000),
        ("SP02", "Bạc xỉu", "Cà phê", 13000, 32000),
        ("SP03", "Cà phê muối", "Cà phê", 15000, 39000),
        ("SP04", "Trà đào cam sả", "Trà", 14000, 45000),
        ("SP05", "Trà sen vàng", "Trà", 16000, 49000),
        ("SP06", "Bánh mì que", "Đồ ăn", 6000, 15000),
        ("SP07", "Croissant bơ", "Đồ ăn", 18000, 35000),
        ("SP08", "Sinh tố bơ", "Sinh tố", 20000, 55000),
    ], columns=["ma_sp", "ten_sp", "danh_muc", "gia_von", "gia_niem_yet"])

    # Messy Excel: title + blank row above the header
    wb = Workbook()
    ws = wb.active
    ws.title = "SanPham"
    ws.append(["DANH MỤC SẢN PHẨM - CÀ PHÊ SƯƠNG MAI"])
    ws.append(["Cập nhật: 01/09/2026"])
    ws.append([])
    ws.append(["Mã SP", "Tên sản phẩm", "Danh mục", "Giá vốn (VND)", "Giá niêm yết (VND)"])
    for r in products.itertuples(index=False):
        ws.append(list(r))
    wb.save(d / "san_pham.xlsx")

    branches = ["Quận 1", "Quận 3", "Thủ Đức"]
    rows = []
    for i, day in enumerate(pd.date_range("2025-01-01", "2026-09-30", freq="D")):
        for _ in range(rng.randint(3, 7)):
            p = products.iloc[rng.randrange(len(products))]
            rows.append({
                "ma_don": f"DH{len(rows) + 1:06d}",
                "ngay_ban": day.strftime("%d/%m/%Y"),
                "chi_nhanh": rng.choice(branches),
                "ma_sp": p.ma_sp,
                "so_luong": rng.randint(1, 5),
                "don_gia": int(p.gia_niem_yet),
                "giam_gia": rng.choice([0, 0, 0, 0.05, 0.1, 0.2]),
                "trang_thai": "Đã hủy" if rng.random() < 0.06 else "Hoàn thành",
            })
    sales = pd.DataFrame(rows)
    sales.to_csv(d / "ban_hang.csv", index=False, encoding="utf-8-sig")

    staff = pd.DataFrame([
        (1, "Nguyễn Văn An", "Quận 1", "Quản lý", 15000000, "2024-03-01"),
        (2, "Trần Thị Bình", "Quận 1", "Pha chế", 8000000, "2024-06-15"),
        (3, "Lê Hoàng Cường", "Quận 3", "Quản lý", 14500000, "2023-11-20"),
        (4, "Phạm Thu Dung", "Quận 3", "Thu ngân", 7500000, "2025-01-10"),
        (5, "Võ Minh Em", "Thủ Đức", "Quản lý", 14000000, "2024-09-01"),
        (6, "Đặng Gia Hân", "Thủ Đức", "Pha chế", 8200000, "2025-04-05"),
        (7, "Bùi Quốc Khánh", "Thủ Đức", "Phục vụ", 6500000, "2025-07-22"),
    ], columns=["id", "ho_ten", "chi_nhanh", "chuc_vu", "luong_thang", "ngay_vao_lam"])
    con = sqlite3.connect(d / "nhan_su.sqlite")
    staff.to_sql("nhan_vien", con, index=False, if_exists="replace")
    con.close()

    (d / "README.md").write_text(
        "# Cà phê Sương Mai - dữ liệu kinh doanh\n\n"
        "- Ngày hiện tại (ngày tham chiếu cho các câu hỏi): 30/09/2026.\n"
        "- Dữ liệu bán hàng từ 01/01/2025 đến 30/09/2026, định dạng ngày dd/mm/yyyy.\n"
        "- Doanh thu = so_luong * don_gia * (1 - giam_gia), chỉ tính đơn có trạng thái 'Hoàn thành'.\n"
        "- Lợi nhuận = Doanh thu - so_luong * giá vốn.\n"
        "- Đơn 'Đã hủy' không tính vào doanh thu hay số lượng bán.\n", encoding="utf-8")

    s = sales.merge(products, on="ma_sp")
    s["ngay"] = pd.to_datetime(s["ngay_ban"], format="%d/%m/%Y")
    ok = s[s.trang_thai == "Hoàn thành"].copy()
    ok["dt"] = ok.so_luong * ok.don_gia * (1 - ok.giam_gia)
    ok["ln"] = ok.dt - ok.so_luong * ok.gia_von
    aug = ok[(ok.ngay.dt.year == 2026) & (ok.ngay.dt.month == 8)]
    sep = ok[(ok.ngay.dt.year == 2026) & (ok.ngay.dt.month == 9)]
    y2026 = ok[ok.ngay.dt.year == 2026]
    y2025 = ok[ok.ngay.dt.year == 2025]
    q3 = s[(s.ngay >= "2026-07-01") & (s.ngay <= "2026-09-30")]

    return [
        ("Tháng trước doanh thu của chi nhánh Quận 1 là bao nhiêu?",
         "answered", round(float(aug[aug.chi_nhanh == "Quận 1"].dt.sum()), 2)),
        ("Tháng trước sản phẩm nào có lợi nhuận cao nhất?",
         "answered", aug.groupby("ten_sp").ln.sum().idxmax()),
        ("Có bao nhiêu đơn hàng bị hủy trong quý 3 năm 2026?",
         "answered", int((q3.trang_thai == "Đã hủy").sum())),
        ("Danh mục nào có tổng số lượng bán ra nhiều nhất trong năm nay?",
         "answered", y2026.groupby("danh_muc").so_luong.sum().idxmax()),
        ("Tổng lương tháng của nhân viên chi nhánh Thủ Đức là bao nhiêu?",
         "answered", int(staff[staff.chi_nhanh == "Thủ Đức"].luong_thang.sum())),
        ("Nhân viên nào vào làm sớm nhất?",
         "answered", staff.sort_values("ngay_vao_lam").iloc[0].ho_ten),
        ("Mức giảm giá trung bình (tính theo %) của các đơn hoàn thành trong tháng 9/2026 là bao nhiêu?",
         "answered", round(float(sep.giam_gia.mean() * 100), 2)),
        ("Top 3 sản phẩm có doanh thu cao nhất năm 2025 là gì? (theo thứ tự giảm dần)",
         "answered", list(y2025.groupby("ten_sp").dt.sum().sort_values(ascending=False).index[:3])),
        ("Chi phí marketing tháng trước là bao nhiêu?", "unanswerable", None),
        ("Doanh thu năm 2024 của quán là bao nhiêu?", "unanswerable", None),
    ]


def bookstore() -> list[tuple[str, str, object]]:
    d = DATA / "datasets" / "nhasach_anhduong"
    d.mkdir(parents=True, exist_ok=True)
    books = pd.DataFrame([
        (1, "Dế Mèn phiêu lưu ký", "Tô Hoài", 65000),
        (2, "Tắt đèn", "Ngô Tất Tố", 52000),
        (3, "Số đỏ", "Vũ Trọng Phụng", 70000),
        (4, "Mắt biếc", "Nguyễn Nhật Ánh", 110000),
        (5, "Cho tôi xin một vé đi tuổi thơ", "Nguyễn Nhật Ánh", 85000),
        (6, "Vợ chồng A Phủ", "Tô Hoài", 45000),
    ], columns=["book_id", "title", "author", "price"])
    customers = [f"KH{i:03d}" for i in range(1, 41)]
    orders, items = [], []
    for oid, day in enumerate(sorted(rng.choice(pd.date_range("2026-01-01", "2026-08-15", freq="D")) for _ in range(400)), 1):
        orders.append((oid, day.strftime("%Y-%m-%d"), rng.choice(customers)))
        for b in rng.sample(range(1, 7), rng.randint(1, 3)):
            items.append((oid, b, rng.randint(1, 3)))
    o = pd.DataFrame(orders, columns=["order_id", "order_date", "customer_code"])
    it = pd.DataFrame(items, columns=["order_id", "book_id", "qty"])
    con = sqlite3.connect(d / "store.db")
    books.to_sql("books", con, index=False, if_exists="replace")
    o.to_sql("orders", con, index=False, if_exists="replace")
    it.to_sql("order_items", con, index=False, if_exists="replace")
    con.close()
    (d / "README.md").write_text(
        "Nhà sách Ánh Dương. Hôm nay là 15/08/2026. Doanh thu = qty * price (giá bìa, không giảm giá).\n",
        encoding="utf-8")

    m = it.merge(o, on="order_id").merge(books, on="book_id")
    m["order_date"] = pd.to_datetime(m.order_date)
    jul = m[(m.order_date.dt.year == 2026) & (m.order_date.dt.month == 7)]
    return [
        ("Tổng doanh thu tháng 7/2026 là bao nhiêu?", "answered", int((jul.qty * jul.price).sum())),
        ("Tác giả nào bán được nhiều cuốn sách nhất?", "answered", m.groupby("author").qty.sum().idxmax()),
        ("Tháng trước có bao nhiêu khách hàng khác nhau mua sách?", "answered", int(jul.customer_code.nunique())),
        ("Mức độ hài lòng trung bình của khách hàng là bao nhiêu?", "unanswerable", None),
    ]


def main() -> None:
    qs, gold = [], []
    for business, fn in (("cafe_suongmai", cafe), ("nhasach_anhduong", bookstore)):
        for i, (q, status, ans) in enumerate(fn(), 1):
            qid = f"{business[:4]}_{i:02d}"
            qs.append({"id": qid, "question": q, "dataset": business})
            if isinstance(ans, (np.integer,)):
                ans = int(ans)
            gold.append({"id": qid, "status": status, "answer": ans})
    (DATA / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n", encoding="utf-8")
    (ROOT / "gold.jsonl").write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gold) + "\n", encoding="utf-8")
    print(f"{len(qs)} questions -> {DATA}")


if __name__ == "__main__":
    main()
