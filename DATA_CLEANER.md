# Data Cleaner & Evaluation Architecture (`cleaner-v4`)

Nhánh `data-cleaner` bắt đầu trực tiếp từ `datamate-agent` (commit `7a2a5e6`).
Cleaner là lớp bảo vệ, chuẩn hóa thận trọng và hồ sơ hóa dữ liệu (data profiling + schema/value grounding) bổ sung cho pipeline Jimmy:
- **Không ghi đè dữ liệu nguồn**: luôn giữ nguyên `ws.raw_tables` sau bước đọc file/header.
- **Không học hoặc nhìn đáp án gold**: mọi bước chuẩn hóa và gợi ý chỉ dựa trên dữ liệu shop và `rulebook.md`.
- **Cô lập hoàn toàn với deterministic recipes**: các recipe trong `datamate/run.py` luôn nhận câu hỏi gốc `q["question"]`. Bản đã chuẩn hóa (`normalized_hint`) và gợi ý grounding (`grounding_hint`) chỉ là thông tin phụ trợ (advisory hint) cho plan card và solver.
- **Tuân thủ giới hạn $\le 15\text{B}$ tham số**: toàn bộ bước profiling, grounding và guard chạy bằng thuật toán xác định (0 tham số neural tăng thêm). Khi bật `rules+llm`, hệ thống tái sử dụng cùng một local endpoint của solver.

---

## 1. Kiến trúc Đã Triển Khai vs. Phần Cần Kiểm Chứng Thêm Trên GPU

### Các lớp đã triển khai và kiểm chứng bằng test tự động (`cleaner-v4`)

| Lớp kiến trúc | File chính | Chức năng đã hoàn thành |
|---|---|---|
| **M0.1 — Smoke Checkpoint & Manifest Lock** | `scripts/smoke_cleaner.py` | Khóa cấu hình trong `smoke_manifest.json` (`dataset_fingerprint`, `ids`, `seed`, `fold`, `limit`, `modes`, `model`, `cleaner_url`, `skip_entities`, budgets, `version`, `code_hash`). Mỗi file checkpoint lưu kèm `config_hash` và `mode` (`checkpoints/<mode>/<qid>.json`), chặn tuyệt đối việc dùng nhầm checkpoint của `rules` cho `off` hoặc ghi đè khi cấu hình thay đổi. |
| **M0.2 — Paired Benchmark Verification** | `scripts/benchmark_cleaner.py` | Tách biệt `split.json` (frozen split) và `run_manifest.json` riêng cho từng mode (`off`, `rules`, `rules+llm`, `rules+profile`). Hàm `verify_mode_output` và `validate_comparison_compatibility` xác minh `submission_sha256`, `completed_ids`, `status == "completed"`, `backend`, `model`, `solver_config`, `dataset_fingerprint` và `rulebook_sha256` trước khi ghép cặp tính $\Delta\text{PA}$. |
| **M0.3 — Local Scorer Alignment** | `datamate/evaluate.py` | Căn chỉnh theo `description/DATAMATE_OVERVIEW.md` §2 và `dataset/train/rulebook.md` §9: chuẩn hóa chuỗi 4 bước (Unicode NFC, lowercase, thay ký tự không phải chữ/số/khoảng trắng bằng 1 khoảng trắng, gộp khoảng trắng liên tiếp, giữ nguyên dấu tiếng Việt và ranh giới từ như `"1000mg"` vs `"1000 mg"`), so khớp `percent` làm tròn 2 chữ số thập phân (`ROUND_HALF_UP`) kèm kiểm tra dấu, hỗ trợ `alias` cho `number`, `percent`, `string`, `list`, `table`. |
| **M1 — Data Profiling with Provenance** | `datamate/cleaning.py`, `datamate/workspace.py` | Hàm `profile_table` và `inspect_date_column` phân tách rõ 3 nhóm: `parse_failures` (lỗi ép kiểu kèm row index + raw value), `ambiguities` (xung đột ngày `DD/MM` vs `MM/DD`, ngày thiếu năm không có context, trộn dấu phân cách nghìn/thập phân `,` và `.`), và `anomalies` (số âm hợp lệ trong trả hàng/chiết khấu, dòng tổng cộng). Bảo vệ mã có số `0` đứng đầu (`"00123"`) trong cả CSV và Excel. Không xóa dòng trùng ở bảng ngoài đơn hàng (rulebook §8). Giới hạn chi phí quét regex/duplicate ở mức 10.000 dòng cho bảng $>50.000$ dòng. |
| **M2 — Schema & Value Grounding** | `datamate/cleaning.py`, `datamate/run.py`, `datamate/agent.py` | Hàm `build_grounding_index`, `ground_question`, `render_grounding_hint`: ưu tiên khớp chính xác Unicode NFC và mã code (`"00123"` $\neq$ `"123"`). Khớp bỏ dấu (`accent_folded_candidate`) hoặc khớp một phần/viết tắt chỉ trả về dưới dạng ứng viên (`candidates`), đánh dấu `ambiguous=True` khi có nhiều mặt hàng tương tự (ví dụ `"Sữa tươi có đường"` vs `"Sữa tươi ít đường"`). |
| **M3 — Conservative Question Cleaning & Guard** | `datamate/cleaning.py`, `datamate/cleaner_llm.py` | Lưu tách biệt `original`, `rules_normalized`, `llm_proposed`, `normalized_hint`, `clear_aliases`, `unaccented_or_typo`, `ambiguous_spans`, `entities`, `constraints`. Guard (`validate_rewrite`) chặn mất mốc thời gian (`hqua`, `hnay`), chặn biến câu hỏi thành nhận định (`bnhiu tien` $\rightarrow$ `nhiều tiền`), chặn đảo chiều thời gian (`sau` $\leftrightarrow$ `trước`), chặn đổi thuộc tính chung (`màu đỏ` $\rightarrow$ `màu xanh`, `size M` $\rightarrow$ `size L`, `loại 1` $\rightarrow$ `loại 2`, `khách mới` $\rightarrow$ `khách cũ`, `đơn hoàn` $\rightarrow$ `đơn huỷ`), chặn đảo chỉ số giữa các cửa hàng, và khóa dấu tiếng Việt của các từ đã có dấu trong câu gốc (`sữa` $\neq$ `sửa`, `Tống` $\neq$ `Tổng`). |
| **M4 — Smoke Tool, Deterministic Fixtures & 4-Mode Benchmark** | `scripts/smoke_cleaner.py`, `scripts/benchmark_cleaner.py`, `tests/test_cleaner.py` | Bộ 23 bài kiểm tra tự động bao phủ toàn bộ regression bugs (M0–M4) và các fixture đánh giá tất định. |

### Phần chưa kiểm chứng trên GPU mục tiêu (cần chạy khi có máy 2×RTX 5090 / local LLM server)

- Chưa đo $\Delta\text{PA}$ thực tế của solver LLM giữa 4 chế độ `off`, `rules`, `rules+llm`, `rules+profile` trên toàn bộ tập `dev` và `audit`.
- Chưa đo tổng thời gian chạy (wall-clock time) của toàn bộ public/private test để khẳng định giới hạn $\le 6$ giờ trên phần cứng thi đấu của BTC.

---

## 2. Hướng Dẫn Chạy `scripts/smoke_cleaner.py`

Công cụ kiểm tra độc lập tầng question cleaner (không gọi solver, không sinh submission):

```bash
# 1. Chạy nhanh chế độ off và rules trên 20 câu fold dev (bỏ qua nạp catalog shop, <1 giây):
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-dev-fast \
    --modes off rules --limit 20 --fold dev --skip-entities

# 2. Chạy chế độ rules có nạp entity thực tế từ các bảng của shop:
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-dev-entities \
    --modes rules --limit 20 --fold dev

# 3. Chạy kèm local LLM cleaner (chỉ gọi endpoint local localhost/127.0.0.1):
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-dev-llm \
    --modes off rules rules+llm \
    --model qwen2.5-7b-instruct \
    --cleaner-url http://127.0.0.1:11434/v1 \
    --cleaner-budget-sec 120 --cleaner-timeout-sec 8 \
    --limit 20 --fold dev
```

### Xử lý khi gặp lỗi Checkpoint / Manifest Mismatch

- Mỗi thư mục `--out` của `smoke_cleaner.py` được khóa chặt với `smoke_manifest.json` và `config_hash` bên trong từng file `checkpoints/<mode>/<qid>.json`.
- Nếu bạn đổi `--modes`, `--model`, `--skip-entities`, `--limit`, `--seed`, `--fold`, hoặc khi code cleaner thay đổi (`code_hash` khác), lệnh chạy lại vào cùng thư mục `--out` sẽ dừng ngay với `ValueError` để bảo vệ các checkpoint cũ.
- **Cách xử lý**:
  1. Chỉ định một thư mục `--out` mới cho cấu hình mới (khuyến nghị), **hoặc**
  2. Nếu muốn chạy lại từ đầu với đúng cấu hình trong cùng thư mục sau khi sửa lỗi tạm thời, dùng `--no-resume` (nhưng lưu ý: nếu `smoke_manifest.json` khác cấu hình, luôn phải dùng thư mục `--out` mới).

---

## 3. Hướng Dẫn Chạy `scripts/benchmark_cleaner.py` (4 Chế Độ Paired)

`scripts/benchmark_cleaner.py` chạy toàn bộ pipeline giải bài và chấm điểm cục bộ trên cùng một split cố định (`split.json`), hỗ trợ 4 chế độ:
- `off`: Tắt hoàn toàn cleaner.
- `rules`: Bật chuẩn hóa bảng thận trọng + chuẩn hóa câu hỏi bằng rules.
- `rules+llm`: Bật thêm local LLM cleaner cho các câu hỏi/cột mơ hồ, đi qua Guard.
- `rules+profile`: Bật `rules` kèm theo **Schema & Value Grounding Hint** (`ground_question`) và cảnh báo chất lượng dữ liệu từ **Data Profiling** (`profile_table`).

```bash
# Bước 1: Chạy paired benchmark cho các chế độ không cần LLM cleaner riêng:
python scripts/benchmark_cleaner.py --data dataset/train --out out/bench-ab \
    --backend ollama --model qwen3.5-9b-40k \
    --modes off rules rules+profile --limit 20 --seed 3407 --fold dev

# Bước 2: Chạy tiếp chế độ rules+llm trên cùng thư mục benchmark (cùng split & cùng solver):
python scripts/benchmark_cleaner.py --data dataset/train --out out/bench-ab \
    --backend ollama --model qwen3.5-9b-40k \
    --modes rules+llm --cleaner-url http://127.0.0.1:11434/v1 \
    --limit 20 --seed 3407 --fold dev
```

### Quy tắc xác minh tính hợp lệ của Paired Benchmark

Trước khi ghép kết quả giữa các mode trong cùng thư mục `--out`, `benchmark_cleaner.py` kiểm tra `run_manifest.json` của từng mode:
1. File `submission.jsonl` phải tồn tại, khớp chính xác `submission_sha256` đã ghi trong `run_manifest.json` (chặn sử dụng nhầm file cũ/stale).
2. Trạng thái `status` phải là `"completed"` và chứa đầy đủ mọi question ID của split (`completed_ids` bao phủ toàn bộ `split["ids"]`, không có câu `"error"` hay `"skipped"`).
3. Hai mode đem so sánh phải có cùng `dataset_fingerprint`, `rulebook_sha256`, `backend`, `model` và `solver_config` (`min_confidence`, `consistency`, `verify`, `verify_new`, `candidate_policy`, `seed`). Nếu bất kỳ thông số nào lệch, `compatible` sẽ bằng `False` và hệ thống không tính $\Delta\text{PA}$.

---

## 4. Hướng Dẫn Review Thủ Công (`cleaner_review.csv`)

Trong `cleaner_review.csv` (sinh bởi `smoke_cleaner.py` hoặc `benchmark_cleaner.py`), mỗi dòng ghi rõ:
- `original_question`: Câu hỏi gốc.
- `rules_normalized`: Kết quả sau tầng rules.
- `llm_proposed`: Đề xuất nguyên gốc của LLM (ngay cả khi bị Guard từ chối).
- `final_normalized`: Câu chuẩn hóa cuối cùng được dùng làm hint.
- `decision`: `off` | `rules_unchanged` | `rules_applied` | `accepted` | `rejected` | `uncertain` | `fallback`.
- `reason`: Lý do chi tiết nếu bị Guard từ chối hoặc fallback.

Ba cột cuối được để trống dành cho người review:
1. `review_label`:
   - `giu_nghia`: Bản chuẩn hóa giữ đúng nghĩa, mốc thời gian, điều kiện lọc, thuộc tính đối tượng và ý hỏi.
   - `doi_nghia`: Bản chuẩn hóa làm lệch hoặc mất thông tin so với câu gốc.
   - `khong_ket_luan`: Câu hỏi gốc vốn đa nghĩa/mơ hồ mà dữ liệu không đủ phân giải.
2. `error_type` (khi `review_label = doi_nghia`):
   - `time_drop` | `intent_flip` | `attr_swap` | `entity_loss` | `relation_inversion` | `diacritic_corruption` | `other`.
3. `reviewer_notes`: Ghi chú cụ thể của người kiểm tra.

> [!IMPORTANT]
> Không tự điền sẵn nhãn `review_label` bằng code hay bịa đặt đáp án gold. Tập `audit` (5 shop cuối) chỉ dùng để kiểm tra độc lập cuối cùng, không dùng để tinh chỉnh rule hay prompt.

---

## 5. Giới Hạn Kỹ Thuật Cần Lưu Ý

1. **Giới hạn của Guard (`validate_rewrite`)**:
   - Guard kiểm tra bất biến từ vựng, số, ngày, dấu âm, đơn vị, mốc thời gian, quan hệ trước/sau, ca bán, phủ định, thuộc tính `(head, modifier)`, liên kết `metric::object`, căn chỉnh từ nội dung (content-token alignment) và khóa dấu tiếng Việt trên các từ đã có dấu.
   - Tuy nhiên, **vượt qua Guard không phải là chứng minh toán học cho sự tương đương ngữ nghĩa 100%**. Vì vậy, câu gốc `q["question"]` luôn được giữ làm nguồn điều khiển chính trong prompt của solver và là đầu vào duy nhất của các deterministic recipes.
2. **Giới hạn của Local Scorer (`datamate/evaluate.py`)**:
   - Repository không đi kèm bộ chấm điểm chính thức của BTC. `datamate/evaluate.py` là bản hiện thực lại cục bộ bám sát đặc tả trong `description/DATAMATE_OVERVIEW.md` §2 và `dataset/train/rulebook.md` §9 để phục vụ phát triển và so sánh nội bộ.
3. **Giới hạn của Data Profiling (`profile_table`)**:
   - Để đảm bảo thời gian phản hồi nhanh và không gây nghẽn bộ nhớ, với các bảng lớn hơn `50.000` dòng, các phép đếm `input_rows`, `parsed_rows`, `null_count`, `distinct_count` vẫn chạy trên toàn bộ bảng, nhưng các phép quét regex định dạng chuỗi và kiểm tra dòng trùng chạy trên mẫu tất định `10.000` dòng đầu (`sampling.sampled = true`). Do đó, kiểm tra mẫu không đảm bảo phát hiện mọi dòng bất thường đơn lẻ ở cuối bảng cực lớn.
4. **Giới hạn của Tập Đánh Giá (Split)**:
   - Kết quả trên tập mẫu hữu hạn (`dev` / `audit`) hoặc trên các fixture tất định chỉ phản ánh chất lượng trên phạm vi đã đo; không suy rộng thành tuyên bố "100% accuracy" trên dữ liệu ẩn của cuộc thi.

---

## 6. Lệnh Kiểm Tra Nhanh (CPU Verification)

```bash
# 1. Kiểm tra biên dịch toàn bộ mã nguồn:
python -m compileall -q datamate scripts tests

# 2. Chạy toàn bộ 23 bài kiểm tra đơn vị, hồi quy và tích hợp (M0 -> M4):
python -m unittest discover -s tests -p test_cleaner.py -v

# 3. Chạy smoke test trên fold dev (20 câu, chế độ off và rules):
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-verify --modes off rules --limit 20 --fold dev --skip-entities
```
