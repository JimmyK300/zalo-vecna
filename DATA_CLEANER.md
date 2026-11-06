# Data cleaner

Nhánh `data-cleaner` bắt đầu trực tiếp từ `datamate-agent`, commit `7a2a5e6`.
Không bao gồm backend Transformers/notebook của main. Cleaner là lớp bổ sung cho pipeline Jimmy,
không thay dữ liệu nguồn, không học đáp án và không thay câu hỏi gốc của recipe.

## Milestones

| Mốc | Code / output | Trạng thái |
|---|---|---|
| M0 | Frozen split theo shop 20/5/5, manifests, decoupled `split.json`, regression test suite | Hoàn thành |
| M1 | Rules cleaner bảo vệ offset thay thế, numeric token, mốc thời gian, ý hỏi, phủ định, lọc, thứ tự, quan hệ đối tượng | Hoàn thành & pass 17 regression tests |
| M2 | Bản raw sau ingestion + báo cáo parse tiền/ngày, dòng lỗi có dấu vết, bảo vệ họ 'Tống' | Hoàn thành |
| M3 | LLM local cho câu/cột mơ hồ, schema bắt buộc `uncertain_spans`, guard quan hệ/thuộc tính, cache cách ly theo context/hash | Hoàn thành |
| M4 | Công cụ smoke test độc lập `scripts/smoke_cleaner.py` (checkpoints, review CSV/JSONL, latency stats) | Hoàn thành |
| M5 | Audit nhãn người trên tập dev (phân loại `giu_nghia / doi_nghia / khong_ket_luan`, giữ nguyên split `audit`) | Sẵn sàng thực hiện theo template |
| M6 | Benchmark solver hoàn chỉnh `scripts/benchmark_cleaner.py` (kiểm tra tương thích backend/model, đo $\Delta$PA, kiểm chứng $\le 6$h/15B) | Sẵn sàng đo trên GPU thật |

---

## Phân Biệt Hai Công Cụ: Smoke Test vs. Benchmark

| Đặc điểm | `scripts/smoke_cleaner.py` (Smoke Cleaner) | `scripts/benchmark_cleaner.py` (Benchmark) |
|---|---|---|
| **Mục đích** | Kiểm tra độc lập chất lượng và độ an toàn của cleaner | Đo lường tác động của cleaner lên độ chính xác của solver ($\Delta\text{PA}$) |
| **Thực thi pipeline** | **Chỉ chạy cleaner** (rules và/hoặc LLM cleaner) | Chạy toàn bộ pipeline giải bài (Agent, DuckDB, Solver, Scorer) |
| **Sinh submission** | **Không** sinh file submission | Sinh submission cho từng mode và chấm điểm với gold dev |
| **Thời gian chạy** | Cực nhanh: <1 giây (rules) hoặc ~1-2 phút (rules+llm) | Vài phút đến hàng chục phút tùy backend/model solver |
| **Checkpoints & Resume** | Lưu checkpoint theo từng câu (`checkpoints/{qid}.json`), resume tức thì | Lưu run history vào `split.json` |
| **Output chính** | `summary.json`, `cleaner_review.csv`, `cleaner_review.jsonl` | `report.json`, `split.json`, `cleaner_review.csv` |

---

## Hướng Dẫn Sử Dụng `scripts/smoke_cleaner.py`

Công cụ kiểm tra nhanh cleaner trên mẫu nhỏ (mặc định 20 câu, khuyến nghị 20–30 câu từ fold `dev` hoặc `train`):

### 1. Chạy CPU Rules-only siêu tốc (không cần GPU/LLM)

```bash
# Chạy với kiểm tra entity từ thư mục shop (dev fold, 20 câu):
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-dev --modes rules --limit 20 --fold dev

# Chạy tức thì chỉ chuẩn hóa câu hỏi, bỏ qua nạp catalog shop:
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-fast --modes rules --limit 20 --skip-entities
```

### 2. Chạy với Local LLM Endpoint (`rules+llm`)

```bash
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-llm --modes rules+llm \
    --model qwen2.5-7b-instruct \
    --cleaner-url http://127.0.0.1:11434/v1 \
    --cleaner-budget-sec 120 \
    --cleaner-timeout-sec 8 \
    --limit 20 --fold dev
```

### 3. Cơ chế Checkpoint và Resume

Mỗi câu hỏi sau khi xử lý được ghi ngay vào `checkpoints/{qid}.json`. Nếu tiến trình bị ngắt quãng giữa chừng (ví dụ timeout, mất kết nối endpoint), chỉ cần chạy lại lệnh cũ: công cụ sẽ tự động nạp các câu đã hoàn thành từ checkpoint và tiếp tục những câu còn lại mà không chạy lại từ đầu. Thêm cờ `--no-resume` nếu muốn ép chạy lại từ đầu.

---

## Hướng Dẫn Gán Nhãn Con Người (M5 Human Audit)

Để đảm bảo an toàn tuyệt đối trước khi áp dụng vào thi đấu, các đề xuất viết lại của cleaner cần được đánh giá thủ công bởi con người dựa trên file `cleaner_review.csv` được sinh ra.

### Quy ước cột gán nhãn:

Trong `cleaner_review.csv`, ba cột sau được để trống dành riêng cho auditor:
1. `review_label`:
   - `giu_nghia`: Bản chuẩn hóa giữ đúng 100% ngữ nghĩa, các ràng buộc thời gian, điều kiện lọc, đối tượng và ý hỏi của câu gốc.
   - `doi_nghia`: Bản chuẩn hóa làm lệch hoặc mất thông tin so với câu gốc (ví dụ đổi ngày, đổi thuộc tính khách hàng, đảo quan hệ doanh thu/lợi nhuận).
   - `khong_ket_luan`: Câu hỏi gốc vốn mơ hồ hoặc ngữ cảnh không đủ để xác định tính đúng đắn (cleaner cần fallback về câu gốc).
2. `error_type`: Nếu `review_label` là `doi_nghia`, ghi rõ mã lỗi:
   - `time_drop`: Mất hoặc đổi mốc thời gian / khoảng thời gian.
   - `intent_flip`: Lật ngược ý hỏi thành nhận định (hoặc ngược lại).
   - `attr_swap`: Đổi thuộc tính đối tượng (ví dụ khách mới $\leftrightarrow$ khách cũ, đơn hoàn $\leftrightarrow$ đơn huỷ).
   - `entity_loss`: Mất, biến dạng tên mặt hàng, mã hàng, tên shop.
   - `relation_inversion`: Đảo quan hệ hoặc ghép sai chỉ số với thực thể (ví dụ doanh thu A và lợi nhuận B $\rightarrow$ lợi nhuận A và doanh thu B).
   - `other`: Lỗi ngữ pháp, cắt cụt câu hoặc biến đổi khác.
3. `reviewer_notes`: Ghi chú ngắn gọn của người kiểm tra.

> [!IMPORTANT]
> **Nguyên tắc đạo đức dữ liệu**: Tuyệt đối không tự ý bịa đặt nhãn vàng (gold answers/labels) dựa trên phán đoán chủ quan. Tập split `audit` (5 shop cuối) phải được giữ nguyên hoàn toàn, không được mở ra xem hoặc điều chỉnh rules/prompt dựa trên tập `audit` để tránh hiện tượng data leakage / overfitting.

---

## Chạy Benchmark từng bước và Chạy tuần tự trên cùng Split

`scripts/benchmark_cleaner.py` chạy **toàn bộ pipeline giải bài** và sinh submission để đo độ chính xác cuối cùng (PA).
Hệ thống hỗ trợ chạy tuần tự các chế độ trên cùng một split cố định (tách biệt `split.json` giữa manifest chia dữ liệu và lịch sử các lần chạy `runs`).

> [!WARNING]
> `benchmark_cleaner.py` chạy pipeline solver. Nếu chỉ muốn kiểm tra nhanh cleaner câu hỏi mà không chạy solver hay sinh submission, hãy dùng `scripts/smoke_cleaner.py`. Khi chạy thử benchmark trên tập dev, luôn truyền `--limit 20` để tránh chạy toàn bộ 150+ câu ngoài ý muốn.

### Bước 1: Chạy baseline `off` và `rules`

```bash
python scripts/benchmark_cleaner.py --data dataset/train --out out/cleaner-ab --backend ollama --model qwen3.5-9b-40k --modes off rules --limit 20 --seed 3407
```

Lệnh này tạo `split.json` khóa cố định seed, business split và question IDs. Thư mục `out/cleaner-ab` sẽ chứa kết quả của `off` và `rules`.

### Bước 2: Chạy tiếp chế độ `rules+llm` trên cùng thư mục benchmark

```bash
python scripts/benchmark_cleaner.py --data dataset/train --out out/cleaner-ab --backend ollama --model qwen3.5-9b-40k --modes rules+llm --cleaner-url http://127.0.0.1:11434/v1 --limit 20 --seed 3407
```

Hệ thống tự động phát hiện `split.json` đã tồn tại, kiểm tra tính toàn vẹn của split, thực thi tiếp `rules+llm`, sau đó nạp toàn bộ kết quả của `off`, `rules` và `rules+llm` để:
1. **Kiểm tra tương thích Solver (`validate_comparison_compatibility`)**: Xác minh hai lần chạy có cùng solver `backend`, cùng `model` giải bài, và cùng fingerprint dữ liệu. Nếu cấu hình solver khác nhau, hệ thống cảnh báo và từ chối tính $\Delta\text{PA}$ để tránh quy kết sai lệch của solver cho cleaner.
2. **Sinh bảng so sánh cặp (flips)**: `off_vs_rules`, `off_vs_rules+llm`, `rules_vs_rules+llm` trong `report.json`.
3. **Tạo file review**: `cleaner_review.csv` và `cleaner_review.jsonl` có đầy đủ các cột cho tất cả các mode đã chạy.

---

## Ý nghĩa các Trạng thái, Quyết định và Cache

Trong `cleaner_review.csv` và logs `q_*.json`:
- `decision`:
  - `accepted`: Đề xuất rewrite của LLM vượt qua toàn bộ các lớp Guard và được sử dụng làm hint.
  - `rejected`: Đề xuất rewrite vi phạm ít nhất một ràng buộc ngữ nghĩa (thời gian, ý hỏi, phủ định, quan hệ, entity); câu cuối cùng fallback về `rules_normalized`.
  - `uncertain`: LLM trả về danh sách `uncertain_spans` không rỗng; fallback về `rules_normalized`.
  - `fallback`: Lỗi kết nối, JSON sai schema, output bị cắt ngắn (truncation), hoặc cạn budget thời gian/số cuộc gọi.
  - `none`: Không kích hoạt LLM (mode `off` hoặc `rules`).
- `source`:
  - `live`: Gọi model trực tiếp sinh ra kết quả.
  - `cache`: Lấy kết quả từ disk cache (`cache_dir`). Cache key bao gồm hash của input, context shop, rulebook hash, model ID và phiên bản code (`VERSION = cleaner-v3`).
  - `none`: Không gọi LLM.
- `llm_proposed`: Ghi lại chính xác chuỗi do LLM đề xuất, kể cả khi bị Guard từ chối (giúp auditor đối chiếu nguyên nhân reject).
- `final_normalized`: Chuỗi chuẩn hóa cuối cùng được chuyển tiếp vào pipeline (nếu rejected thì là `rules_normalized`).
- `human_review`: Cột để trống dành riêng cho đánh giá độc lập của con người.

---

## Giới hạn của Guard và Những gì Chưa được Kiểm chứng

1. **Kiểm tra từ vựng và cấu trúc (lexical & structural check)**: Guard so khớp bất biến về entity, số, ngày, dấu âm, đơn vị, neo thời gian tương đối, quan hệ trước/sau, ca sáng/chiều, khoảng từ/đến và từ phủ định. Đây là lớp bảo vệ phòng thủ (defensive guardrail), **không phải bằng chứng toán học chứng minh tương đương ngữ nghĩa**.
2. **Không ép nghĩa từ mơ hồ**: Các từ chat đa nghĩa (`nhiu`, `k`, `kh`, `dt`) được giữ nguyên trạng thái mơ hồ; cleaner không võ đoán ép nghĩa.
3. **Chưa đo inference GPU thực tế**: Kết quả chạy trên môi trường CPU/scripted chỉ nhằm kiểm chứng tính toàn vẹn kỹ thuật và hợp đồng dữ liệu; **chưa phải là thước đo chất lượng đáp án (PA) hay khẳng định đạt mốc 6 giờ** trên cụm 2×RTX 5090 của BTC.

---

## Hướng dẫn Smoke Test trên Google Colab (L4 / A100)

Khi chạy thử nghiệm trên Google Colab:
- **Chọn GPU**:
  - Model $\le 9$B (ví dụ Qwen2.5-7B/9B quantized hoặc 16-bit): GPU **L4** (24GB VRAM) hoàn toàn đủ tài nguyên, tiết kiệm compute units, không bắt buộc dùng A100.
  - Model 14B: Có thể dùng GPU **L4** (với 4-bit/8-bit quantization qua vLLM/Ollama) hoặc **A100 (40GB/80GB)** nếu chạy fp16/bf16.
- **Lệnh smoke test nhỏ (3-5 câu)**:
  ```bash
  # Cài đặt và khởi chạy endpoint local (vLLM hoặc Ollama)
  # Chạy smoke test 3 câu:
  python scripts/benchmark_cleaner.py --data dataset/train --out out/colab-smoke --backend ollama --model qwen3.5-9b-40k --modes off rules rules+llm --cleaner-url http://127.0.0.1:11434/v1 --limit 3 --seed 3407
  ```
- **Lưu kết quả và Giải phóng Runtime**:
  - **Lưu kết quả**: Nén và tải thư mục kết quả về Google Drive trước khi dừng runtime:
    ```bash
    zip -r /content/cleaner_colab_smoke.zip /content/out/colab-smoke
    cp /content/cleaner_colab_smoke.zip /content/drive/MyDrive/
    ```
  - **Tắt Runtime**: Sau khi lưu kết quả, ngắt kết nối runtime trong giao diện Colab (`Runtime -> Disconnect and delete runtime`) hoặc chạy lệnh:
    ```python
    from google.colab import runtime
    runtime.unassign()
    ```
    *(Lưu ý: Không đặt lệnh tự tắt runtime vào unit test hoặc test script tự động).*

## Hành vi và giới hạn

- `cleaning.py`:
  - **Representation**: `QuestionCleanResult` lưu đầy đủ `original`, `normalized_hint`, `protected_spans` (mã SKU, mã đơn, số âm, số có số 0 đầu, tiền tệ, ngày, thực thể shop kèm vị trí `start/end`), `constraints` (mốc thời gian tương đối `hqua/hnay/...`, ý hỏi `quantity/money/...` vs nhận định `statement`, phủ định `khong/chua/...`, điều kiện lọc `tu/den/tren/...`, thứ tự `tang/giam/top`), `ambiguous_spans`, và danh sách `changes` (old, new, reason).
  - **Rules thận trọng**: Chuẩn hóa Unicode NFC và khoảng trắng; mở rộng an toàn các từ viết tắt ít mơ hồ (`hqua` -> `hôm qua`, `hnay` -> `hôm nay`, `dthu` -> `doanh thu`, `bnhiu/bnhieu` -> `bao nhiêu`, `ko/khum` -> `không`). Không tự động thay `nhiu` thành `nhiều` khi không rõ ngữ cảnh. Không tự gán nghĩa cho các token mơ hồ `dt`, `k`, `kh`, `th`, `bn`. Không tự restore dấu cho tên riêng chưa chắc chắn.
  - **Guard đa tầng**: Kiểm tra nghiêm ngặt đề xuất của LLM: chặn triệt để việc làm mất mốc thời gian (như `hqua`, `hnay`), chặn lật ngược ý hỏi thành nhận định (như biến `bnhiu tien` thành `nhiều tiền`), chặn mất từ phủ định, chặn biến đổi điều kiện lọc / thứ tự, và chặn thay đổi số/mã/đơn vị/thực thể. Khi guard từ chối hoặc phát hiện span không chắc chắn, hệ thống tự động fallback về rules hoặc câu gốc và ghi log lý do chi tiết.
- **Cô lập Recipe**: Mọi candidate recipe trong pipeline Jimmy chỉ nhận **câu hỏi gốc** `q["question"]`. Câu chuẩn hóa chỉ đóng vai trò gợi ý (hint) cho plan card và model fallback; tuyệt đối không để câu viết lại kích hoạt recipe trả lời trực tiếp.
- `cleaner_llm.py`:
  - Tối đa một phân tích cột mơ hồ mỗi workspace khi cần model fallback; câu chỉ gọi LLM khi rules đánh dấu có sự mơ hồ (`ambiguous_spans` hoặc cảnh báo ngữ cảnh).
  - Schema JSON ngắn gọn, kiểm tra chặt chẽ độ dài và uncertain spans. Mỗi request ≤384 output tokens, timeout mặc định 8s, budget 900s / tối đa 100 requests mỗi process.
  - Fallback an toàn khi JSON lỗi, server timeout, output bị cắt hoặc cạn budget (không crash pipeline).
  - Cache độc lập: cache key kết hợp `prompt_hash`, `model_name`, `code_version`, `rulebook_hash`, và ngữ cảnh workspace; tránh nhiễm bẩn cache khi cấu hình thay đổi.
- `Workspace.raw_tables` & Provenance:
  - Giữ nguyên DataFrame sau ingestion (đọc/header), trước khi ép kiểu parse. Không ghi đè hoặc làm hỏng file nguồn.
  - Báo cáo parse tiền/ngày (`column_reports`) theo dõi tỷ lệ thành công/thất bại, cột nguồn, và mẫu lỗi giới hạn.
  - Các dòng không parse được giữ nguyên dấu vết (không âm thầm biến thành 0 hay rớt khỏi raw data).
  - Họ "Tống" được phân biệt chính xác với dòng subtotal "tổng" qua collation chuẩn NFC và regex tách biệt.
  - LLM chỉ gợi ý role cột dựa trên evidence trích từ dữ liệu/header; không tự sửa cell hay join.
- **Benchmark & Human Review**:
  - So sánh paired giữa `off`, `rules`, `rules+llm` trên cùng tập câu.
  - Báo cáo chi tiết: PA, số câu thay đổi, accepted/rejected/fallback, lý do từ chối của guard, latencies (p50/p95), và các chuyển đổi đúng $\leftrightarrow$ sai (`correct_to_wrong`, `wrong_to_correct`).
  - Xuất file đánh giá con người `cleaner_review.jsonl` và `cleaner_review.csv` với cột `human_review` để trống phục vụ audit độc lập.
- **Giới hạn 15B / 6 giờ**:
  - Tuân thủ quy định BTC: toàn bộ model inference $\le 15$B tham số, chạy offline hoàn toàn trên máy 2×RTX 5090 trong $\le 6$ giờ. Kết quả chạy CPU smoke chỉ dùng để xác minh tính toàn vẹn kỹ thuật, chưa coi là đạt mốc thời gian BTC nếu chưa đo trên môi trường GPU mục tiêu.

## Kiểm tra

```bash
# 1. Chạy toàn bộ unit test & regression test (17 tests, <1s):
python -m unittest discover -s tests -p test_cleaner.py

# 2. Kiểm tra biên dịch cú pháp toàn bộ module:
python -m compileall -q datamate scripts tests

# 3. Smoke test nhanh cleaner (CPU rules-only, <1s):
python scripts/smoke_cleaner.py --data dataset/train --out out/smoke-test --modes rules --limit 20 --skip-entities
```

Test không cần GPU/API: invariant số/đơn vị/phủ định, entity/code, tiền/ngày lỗi, raw sheet gộp,
local-only endpoint, cache, timeout/fallback, budget và reject cột bịa. Cần kiểm tra thêm PA per-type,
các câu đúng→sai và bị đổi nghĩa trên dev, rồi full offline wall-clock và tái lập trên phần cứng mục tiêu.
