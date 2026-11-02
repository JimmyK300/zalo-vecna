# DataMate — benchmark local

Pipeline mới chạy Qwen3.5-9B Q4_K_M trên RTX 5060 Laptop: đọc bảng, chuẩn hóa dữ liệu, gọi SQL và chấm câu hỏi train. Code cũ được giữ trong lịch sử nhánh `dev`; nhánh làm việc là `benchmark/local-clean`.

**Kết quả đáng dùng để đánh giá hiện tại:** bộ 200 câu ngẫu nhiên mới đạt **135 đúng, 47 sai, 18 bỏ trống**, accuracy **67,5%**, PA **0,44**, thời gian **17 phút 50 giây**. Solver giữ nguyên trong toàn bộ lượt. Bộ 90 câu phát triển trước đó đạt 89/90, không đại diện cho điểm câu mới. Xem [báo cáo 200 câu](reports/validation200.md).

Chưa fine-tune, chưa GRPO, **0 Colab CU**. Hiện tập trung hoàn thiện công cụ/nghiệp vụ trước khi train. Đây là benchmark train, chưa phải bộ nộp 1.000 câu public hay Docker thi chính thức.

## 1. Chuẩn bị

Cần Windows, Python 3.11+ (đã kiểm tra với 3.11.7), mạng khi tải model/runtime, driver NVIDIA tương thích CUDA 13.4. Cắm sạc và bật RTX; chế độ Eco có thể làm CUDA không khả dụng.

Dữ liệu gốc nằm **ngoài repo** như sau:

```text
E:\ZALO_AI_Challenge\
  train\
    questions.jsonl
    answers.jsonl
    rulebook.md
    calendar.csv
    data\biz_...\...
  public_test\...
  zalo-vecna\                  # repo này
```

Benchmark mặc định đọc `../train`; không đọc bản `train/` cũ bên trong repo. Không cần copy dữ liệu vào repo. Dữ liệu, model, môi trường Python, runtime và trace lớn đều được Git bỏ qua.

Mở PowerShell tại repo:

```powershell
cd E:\ZALO_AI_Challenge\zalo-vecna
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Lệnh tạo `.venv`, cài `openpyxl`, tái tạo các manifest benchmark local và chạy kiểm thử. Các file `benchmarks/*.json` chứa ID từ train nên không được đưa lên GitHub. Nếu Python không có trong PATH, truyền `-Python "đường-dẫn-đầy-đủ-tới-python.exe"` vào script setup.

## 2. Tải model và runtime một lần

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare-model.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-runtime.ps1
```

Model khoảng 5,68 GB; script có tải tiếp và kiểm SHA-256. Revision model được pin, runtime là llama.cpp b11960/CUDA 13.4. Manifest danh tính được lưu trong `models/` và `tools/llama.cpp/`. Không cài PyTorch/vLLM cho benchmark này.

## 3. Khởi động model — terminal thứ nhất

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-server.ps1
```

Giữ terminal này mở đến khi thấy `listening on http://127.0.0.1:8080`. Server dùng context 16K, một slot, thinking tắt. Script đã kiểm tra chạy được trên Windows PowerShell 5 và PowerShell 7. Nhấn Ctrl+C khi chạy xong để giải phóng VRAM.

Nếu thiếu VRAM, thử `-Context 8192` hoặc giảm `-GpuLayers`; ghi lại thay đổi khi so điểm/tốc độ. Server chỉ nghe localhost; mọi câu hỏi được xử lý local.

## 4. Chạy benchmark — terminal thứ hai

Không cần activate môi trường; dùng trực tiếp Python trong `.venv`.

### Kiểm tra GPU và thử 10 câu

```powershell
.\.venv\Scripts\python.exe -m datamate_bench preflight --require-cuda
.\.venv\Scripts\python.exe -m datamate_bench run --suite benchmarks/train10.json --require-cuda
```

Thêm `--limit 1` nếu chỉ muốn thử một câu. Hoặc chạy `scripts/benchmark.ps1`: script này kiểm GPU, đo tốc độ rồi chạy tối đa 10 câu; nó không tự chạy 200 câu.

### Chạy lại hai bộ đã lưu

```powershell
.\.venv\Scripts\python.exe -m datamate_bench run --suite benchmarks/development90.json --require-cuda
.\.venv\Scripts\python.exe -m datamate_bench run --suite benchmarks/validation200_random.json --require-cuda
```

Lệnh tự tạo thư mục mới `artifacts/run_<thời gian>/`. Có thể truyền `--output artifacts/ten_luot_moi`; thư mục output phải trống. Không ghi đè kết quả cũ. Giới hạn mặc định: 12 bước/câu, 240 giây/câu, 5 giây/truy vấn.

### Chọn bộ câu ngẫu nhiên mới

```powershell
.\.venv\Scripts\python.exe scripts/select_random_suite.py --exclude-suite benchmarks/seen290.json --size 200 --seed 20261108 --output benchmarks/validation_new200.json
.\.venv\Scripts\python.exe -m datamate_bench run --suite benchmarks/validation_new200.json --require-cuda
```

`seen290.json` loại cả 90 câu phát triển lẫn 200 câu đã đánh giá. Lấy mẫu không dùng type/đáp án tham chiếu. Doanh nghiệp vẫn có thể trùng; đây là tách theo câu hỏi. Script từ chối ghi đè manifest đã có.

### Tổng hợp sau khi chạy xong

```powershell
.\.venv\Scripts\python.exe scripts/audit_validation.py artifacts/ten_luot_moi
```

Script chỉ đọc lượt đã có `scores.json` và `summary.json`. Nó xuất điểm theo type, thời gian, kiểm hash solver, danh sách câu sai/bỏ trống và trace để rà soát. Nó không tự sửa hay chạy lại câu sai.

## 5. Các tệp kết quả

| Tệp trong thư mục lượt chạy | Nội dung |
|---|---|
| `report.md`, `summary.json` | Điểm và thời gian tổng |
| `submission.jsonl` | Dự đoán, ghi ngay sau mỗi câu |
| `logs/<id>.json` | Công cụ, SQL, lỗi, timing từng câu |
| `scores.json` | Dự đoán và nhãn, chỉ tạo sau toàn bộ inference |
| `config.json` | Hash code/input, model/runtime, tham số |
| `preflight.json` | Phần cứng và CUDA |
| `audit_*.json` | Tổng hợp bổ sung từ script audit |

Điểm PA = `(đúng − sai) / tổng số câu`; bỏ trống nhận 0. Accuracy khác PA. Bộ chấm local chưa phải scorer chính thức của BTC.

Thời gian `run` không gồm tải weights hoặc khởi động server. Lượt đầu còn tốn thời gian đọc Excel và dựng cache; các lượt sau nhanh hơn. Trạng thái GPU qua HTTP không tự chứng minh GPU offload; đối chiếu preflight, server log và nvidia-smi.

## 6. Chức năng từng script/module

| Tệp | Cách dùng / vai trò |
|---|---|
| `scripts/setup.ps1` | Chuẩn bị Python và kiểm thử |
| `scripts/prepare-model.ps1` | Tải/kiểm model GGUF |
| `scripts/install-runtime.ps1` | Tải llama.cpp và DLL CUDA |
| `scripts/start-server.ps1` | Chạy model local |
| `scripts/benchmark.ps1` | Kiểm GPU + đo tốc độ + bộ 10 câu |
| `scripts/rebuild_suites.py` | Tái tạo chính xác các bộ đã đánh giá từ dữ liệu gốc và seed |
| `scripts/select_random_suite.py` | Chọn bộ ngẫu nhiên, loại ID đã dùng |
| `scripts/audit_validation.py` | Tổng hợp một lượt đã hoàn tất |
| `datamate_bench/__main__.py` | CLI chung: `python -m datamate_bench ...` |
| `data.py`, `normalized.py` | Đọc/chuẩn hóa bảng, giữ nguồn và ô gốc |
| `agent.py`, `client.py` | Điều phối model và công cụ |
| `metrics.py`, `temporal.py` | Phép tính SQL và mốc thời gian |
| `scoring.py`, `suite.py` | Chấm điểm và quản lý bộ câu |

Các module trong `datamate_bench/` là thư viện; dùng CLI, không chạy riêng từng file bằng double-click.

Lệnh phụ:

```powershell
# Chỉ đo đọc/cache dữ liệu thô; không gọi model
.\.venv\Scripts\python.exe -m datamate_bench profile --suite benchmarks/development90.json
# Đo tốc độ model; cần server đang bật
.\.venv\Scripts\python.exe -m datamate_bench speed
# 24 bài kiểm tra; không cần tải weights hoặc bật server thật
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## 7. Trạng thái và giới hạn

- Bản solver của lượt 200 câu được giữ nguyên; hash khớp trước/sau chạy. Chưa sửa solver theo lỗi của bộ này.
- Rà soát 65 câu chưa đúng: 31 trường hợp ưu tiên mở rộng công cụ, 14 trường hợp ở hợp đồng/ánh xạ/định tuyến pipeline, 20 trường hợp chọn công cụ/SQL/đầu ra của model. Các nhóm có thể có nguyên nhân phụ chồng lấn.
- Ưu tiên sửa parser/nghiệp vụ và công cụ trước; chưa chi CU cho SFT hoặc GRPO. Sau đó đánh giá lại bằng bộ mới, rồi mới quyết định SFT nhỏ.
- Các câu đã xem lỗi không còn là test mù cho bản sửa kế tiếp. Không suy ra 1.000/1.000 từ điểm bộ phát triển.
- Không có nhánh theo ID câu/cửa hàng trong solver. Câu rõ nghĩa có thể được định tuyến vào SQL trực tiếp; đây là pipeline kết hợp luật và model.

Chi tiết: [báo cáo 200 câu](reports/validation200.md), [trạng thái tiếp tục](BENCHMARK_STATUS.md). Các trace đầy đủ nằm local trong `artifacts/`; GitHub chỉ chứa code, công thức/seed tái tạo manifest và báo cáo tổng hợp không kèm ID dữ liệu.
