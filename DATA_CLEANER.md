# Data cleaner

Nhánh `data-cleaner` bắt đầu trực tiếp từ `datamate-agent`, commit `7a2a5e6`.
Không bao gồm backend Transformers/notebook của main. Cleaner là lớp bổ sung cho pipeline Jimmy,
không thay dữ liệu nguồn, không học đáp án và không thay câu hỏi gốc của recipe.

## Milestones

| Mốc | Code / output | Trạng thái |
|---|---|---|
| M0 | Benchmark theo shop 20/5/5, manifests, scorer số tiền chặt hơn | Đã có harness; chưa có A/B chất lượng bằng model thật |
| M1 | Rules cleaner, protected spans, log từng thay đổi | Đã triển khai, test CPU |
| M2 | Bản raw sau đọc file/header + báo cáo parse tiền/ngày, dòng thay đổi | Đã triển khai, test tương thích schema/giá trị và sheet gộp |
| M3 | LLM local cho câu/cột mơ hồ, guard, cache, timeout/budget | Đã triển khai; test client giả lập, chưa benchmark model/GPU thật |
| M4 | `off / rules / rules+llm`, log PA/latency/reject | Đã có script; cần chạy đủ dev với model đã chốt |
| M5 | Offline trên máy gần 2×5090, tổng ≤6h, lặp lại kết quả | Chưa xác nhận; không coi smoke CPU là đạt điều kiện BTC |

## Chạy

Từ thư mục repo, cài `requirements.txt`. Bắt đầu với rules-only (không thêm model):

```bash
python -m datamate.run --data dataset/train --out out/cleaner-rules/submission.jsonl --backend ollama --model qwen3.5-9b-40k --cleaner rules --workers 1
```

Baseline trong cùng code: dùng `--cleaner off` và một thư mục `--out` khác.
`off` giữ cách xử lý/prompt của Jimmy; scorer số tiền đã được siết cho mọi mode nên PA cũ trong README không so trực tiếp.
Mặc định `rules`; recipe vẫn chỉ nhận câu gốc. Bản chuẩn hóa được dùng cho plan card và là gợi ý cho model.

Nếu solver đang phục vụ model ở endpoint local OpenAI-compatible:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local
python -m datamate.run --data dataset/train --out out/cleaner-llm/submission.jsonl --backend openai --model YOUR_LOCAL_MODEL_ID --cleaner rules+llm --cleaner-url http://127.0.0.1:8000/v1 --cleaner-budget-sec 900 --cleaner-timeout-sec 8 --cleaner-max-calls 100 --workers 1
```

Ollama thường dùng `--cleaner-url http://127.0.0.1:11434/v1` và cùng tên model với solver.
Cleaner không tải weights riêng: nó gọi cùng model đã phục vụ. Phải cấu hình solver local tương ứng;
URL cleaner từ xa bị từ chối. Không mặc định dùng một model thương mại để clean.
Server cần hỗ trợ JSON object responses. Server không hỗ trợ, output bị cắt hay timeout đều quay về rules.
Timeout HTTP không chứng minh GPU server hủy ngay phép sinh; cần đo tài nguyên server thật.

Benchmark paired trên fold dev theo business; chỉ dùng train BTC có gold:

```bash
python scripts/benchmark_cleaner.py --data dataset/train --out out/cleaner-ab --backend ollama --model qwen3.5-9b-40k --modes off rules --cleaner-url http://127.0.0.1:11434/v1 --seed 3407
python scripts/benchmark_cleaner.py --data dataset/train --out out/cleaner-ab --backend ollama --model qwen3.5-9b-40k --modes rules+llm --cleaner-url http://127.0.0.1:11434/v1 --seed 3407
```

Giữ cùng cleaner-url giữa các lệnh của một benchmark vì config đã khóa; ví dụ với Ollama nên truyền
`--cleaner-url http://127.0.0.1:11434/v1` ngay từ lệnh off/rules đầu tiên.
`--limit 3 --backend scripted` chỉ để kiểm tra kỹ thuật (scripted cố ý abstain), không đo chất lượng LLM.
`split.json` cố định shop và IDs; audit cần `--fold audit --open-audit` sau khi chốt config.
Holdout không hoàn toàn sạch: recipe cũ đã phát triển trên toàn bộ train. Không dùng test để train cleaner.
Gold chỉ được đọc để chấm sau subprocess, không đưa vào prompt. Thiếu dự đoán tính abstain trên toàn fold.
Muốn so thời gian từ đầu: dùng output mới, tránh trộn cold run với resume/cache.

## Hành vi và giới hạn

- `cleaning.py`: NFC/khoảng trắng, từ viết tắt ít mơ hồ. Giữ `nhiu` chưa chắc nghĩa; bảo vệ mã/số/đơn vị và
  các entity có trong shop. Không tự restore dấu cho tên riêng. Guard so thứ tự số, protected spans,
  phủ định, so sánh và thời gian. Đây là kiểm tra lexical, không phải bằng chứng tương đương ngữ nghĩa.
- `cleaner_llm.py`: tối đa một phân tích cột mơ hồ mỗi workspace, chỉ khi cần model fallback;
  câu cần LLM chỉ khi rules đánh dấu mơ hồ. Mỗi request ≤384 output tokens, mặc định timeout 8s,
  tổng budget 900s / tối đa 100 request mỗi process. Cache gắn nội dung, model, rulebook/data hash và code hash.
  Câu đã có recipe không gọi LLM cleaner. Đến cuối ngân sách inference dành 60s dự phòng, cleaner dừng gọi.
- LLM chỉ đề xuất ý nghĩa cột với đoạn trích từ payload, không sửa cell, chọn join hoặc ghi đè semantic roles.
  Gợi ý vẫn là **unverified** trong prompt; không được dùng để recipe tự trả đáp án.
- `Workspace.raw_tables` giữ DataFrame sau ingestion (đọc/header), trước parse; không phải byte-for-byte file gốc.
  CSV/Excel nguồn không bị ghi. Sheet gộp giữ `_sheet`. Raw không đăng ký thêm vào DuckDB để tránh recipe đếm hai lần.
  Python tool có `raw_tables`, và `docs['cleaning_report']` để đối chiếu (bản copy).
- Bảng chuẩn hóa vẫn giữ tên/schema mà recipe đang dùng; tiền text vẫn được parser Jimmy chuyển trong bảng chính.
  Thay đổi này thêm bản raw và provenance, chưa rewrite toàn bộ loader thành kiến trúc cột `__num`.
  Báo cáo có parse failure, sample lỗi và chênh số dòng qua conventions cũ; không tự xóa/gộp tên bằng bỏ dấu.
- `logs/q_*.json`: câu gốc, cleaner changes/protected/warnings, LLM status, gợi ý shop và total_seconds.
  `logs/shop_*_cleaning.json`: provenance cột. `cleaner_manifest.json`: data/config để chặn resume khác cấu hình.
  `cleaner_usage.json`: request count và thời gian LLM cleaner. Không có khẳng định tăng PA hay đạt 6h trước benchmark GPU.

## Kiểm tra

```bash
python -m unittest discover -s tests -p test_cleaner.py
python -m compileall -q datamate scripts/benchmark_cleaner.py
```

Test không cần GPU/API: invariant số/đơn vị/phủ định, entity/code, tiền/ngày lỗi, raw sheet gộp,
local-only endpoint, cache, timeout/fallback, budget và reject cột bịa. Cần kiểm tra thêm PA per-type,
các câu đúng→sai và bị đổi nghĩa trên dev, rồi full offline wall-clock và tái lập trên phần cứng mục tiêu.
