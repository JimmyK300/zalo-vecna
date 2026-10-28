# DataMate — Zalo AI Challenge 2026

Dự án phát triển AI Agent trả lời câu hỏi tự nhiên bằng tiếng Việt trên dữ liệu bảng biểu thực tế (CSV, Excel, SQLite) phục vụ bài toán **DataMate** trong khuôn khổ cuộc thi **Zalo AI Challenge 2026**.

---

## 1. Yêu cầu & Cài đặt Môi trường (Setup)

### 1.1. Yêu cầu hệ thống
* **Python:** Khuyến nghị **Python 3.10** hoặc **Python 3.11** (trên Windows hoặc Linux/WSL).
* **Git:** Quản lý mã nguồn.
* **Bộ nhớ RAM:** Tối thiểu 16GB (để nạp và xử lý các file Excel lớn).

### 1.2. Khởi tạo môi trường ảo (Virtual Environment)

Mở terminal tại thư mục gốc của dự án:

```bash
# 1. Tạo môi trường ảo venv với Python 3.11
py -3.11 -m venv venv
```

**Kích hoạt môi trường ảo:**

* **Trên Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi script execution policy, chạy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

* **Trên Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

* **Trên Linux / macOS / WSL:**
  ```bash
  source venv/bin/activate
  ```

### 1.3. Cài đặt các thư viện cần thiết

```bash
# Nâng cấp pip
pip install --upgrade pip

# Cài đặt toàn bộ dependencies
pip install -r requirements.txt
```

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
ZALO_AI_Challenge/
├── Overview.md              # Tổng quan cuộc thi, luật thi, format đáp án & giải thưởng
├── Plan.md                  # Kế hoạch chi tiết, phân tích dữ liệu & kiến trúc kỹ thuật
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── .gitignore               # Cấu hình bỏ qua file rác, dữ liệu nặng và weights
├── README.md                # Tài liệu hướng dẫn sử dụng và thiết lập
│
├── train/                   # Tập dữ liệu huấn luyện (3,000 câu hỏi, 30 doanh nghiệp)
│   ├── data/<business_id>/  # Các file CSV, Excel, SQLite + README của từng shop
│   ├── calendar.csv         # Lịch âm - dương (2024–2027) phục vụ tra cứu Tết
│   ├── rulebook.md          # Quy tắc nghiệp vụ mặc định tập train
│   ├── questions.jsonl      # Danh sách câu hỏi tập train
│   └── answers.jsonl        # Nhãn chuẩn (ground truth) có kèm alias
│
├── public_test/             # Tập kiểm thử công khai (1,000 câu hỏi, 14 doanh nghiệp)
│   ├── data/<business_id>/  # Dữ liệu của 14 doanh nghiệp test (dev_001 – dev_014)
│   ├── calendar.csv         # Lịch âm - dương
│   ├── rulebook.md          # Quy tắc nghiệp vụ cập nhật của tập public test
│   └── questions.jsonl      # Danh sách câu hỏi tập public test
│
└── src/                     # Mã nguồn baseline hệ thống Agent
    ├── __init__.py
    ├── temporal.py          # Xử lý ngày tương đối (as_of_date, tuần rồi, tháng này, Tết)
    ├── ingestion.py         # Nạp đồng nhất file Excel đa sheet, CSV, SQLite vào DuckDB
    └── evaluator.py         # Bộ chấm điểm Penalized Accuracy (PA) chuẩn theo BTC
├── run_public_test.py       # Script chạy inference tự động trên Public Test (hoặc Train)
└── DataMate_Public_Test_Colab.ipynb # Notebook sẵn sàng chạy trên Google Colab GPU (T4/L4/A100)
```

---

## 3. Các Module Baseline Đã Xây Dựng

1. **`src/temporal.py` (Temporal Engine):**
   * Tự động phân giải `as_of_date` thành các khoảng ngày: hôm nay, hôm qua, tuần này (Thứ 2 đến `as_of_date`), tuần rồi (Thứ 2 đến Chủ Nhật), tháng này (ngày đầu tháng đến `as_of_date`), tháng trước.
   * Tra cứu `calendar.csv` để tự động xác định khoảng thời gian mùa Tết (từ 23 tháng Chạp đến hết mùng 10 tháng Giêng).

2. **`src/ingestion.py` (Unified Data Ingestion):**
   * Nạp trực tiếp dữ liệu từ file Excel nhiều sheet (`.xlsx`), file phẳng (`.csv`) và database SQLite (`.sqlite`) vào engine DuckDB in-memory.
   * Tự động trích xuất metadata schema của từng shop để đưa vào ngữ cảnh prompt cho LLM.

3. **`src/evaluator.py` (Evaluation & Normalization Engine):**
   * Triển khai toàn bộ quy tắc chuẩn hóa văn bản Unicode NFC, làm tròn số thực 2 chữ số, chuẩn hóa bảng và danh sách.
   * Chấm điểm từng câu hỏi theo chuẩn: Đúng (+1), Bỏ cuộc (0), Sai (-1).
   * Tính điểm **Penalized Accuracy (PA)** toàn tập dữ liệu.

---

## 4. Kiểm Thử Nhanh Baseline

Bạn có thể chạy thử nghiệm các module baseline bằng lệnh:

```bash
python -c "from src.temporal import TemporalEngine; from src.evaluator import DataMateEvaluator; t = TemporalEngine('train/calendar.csv'); print(t.resolve_relative_dates('2026-03-04'))"
```

---

## 5. Hướng Dẫn Đóng Gói Docker & Nộp Bài

* **Định dạng file nộp bảng xếp hạng:** `submission.jsonl` (tối đa 5 lượt nộp / ngày).
* **Định dạng nộp giải pháp (Hạn chót 16/11):** Docker image hoàn chỉnh chứa mã nguồn, weights offline và file kịch bản `predict.sh` đọc từ `/data` và ghi kết quả vào `/result/submission.jsonl` kèm log tại `/result/logs/`.
* Xem chi tiết hướng dẫn tại [Overview.md](Overview.md) và chiến lược từng bước tại [Plan.md](Plan.md).
