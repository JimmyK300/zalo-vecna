# Đánh giá đóng băng trên 200 câu ngẫu nhiên

Ngày chạy: 02/11/2026. **135 đúng, 47 sai, 18 bỏ trống**, accuracy **67,5%**, PA **0,44**, coverage **91%**. Thời gian **1.070,16 giây = 17 phút 50 giây**. Không fine-tune, không thay weights, **0 Colab CU**.

## Cách đánh giá

- Seed 20261107, lấy mẫu ngẫu nhiên đều 200 câu từ 2.910 câu train còn lại sau khi loại 90 ID phát triển. Không cân bằng theo nhãn, không dùng giá trị đáp án/type để chọn mẫu.
- 30 doanh nghiệp đã xuất hiện trong quá trình phát triển; đây là tách theo câu hỏi, không phải doanh nghiệp mới. Không đại diện trực tiếp điểm public/private.
- Hash toàn bộ module solver khớp bản đã chốt trước và sau chạy; `source_files_changed_since_start=[]`. Không sửa solver hoặc chạy lại câu sai trong lượt này.
- Nhãn chỉ được đưa vào chấm sau khi hoàn tất cả 200 dự đoán. Full trace và nhãn giữ local trong `artifacts/frozen_validation200/`.
- Danh sách ID cố định được tạo local tại `benchmarks/validation200_random.json` bằng script `rebuild_suites.py`; ID không đưa lên GitHub. Provenance và thống kê tổng hợp nằm cùng thư mục reports này.

## Theo loại đáp án tham chiếu

| Type | Số câu | Đúng | Sai | Bỏ trống | Accuracy |
|---|---:|---:|---:|---:|---:|
| number | 125 | 82 | 31 | 12 | 65.6% |
| percent | 43 | 33 | 6 | 4 | 76.7% |
| list | 12 | 5 | 6 | 1 | 41.7% |
| date | 10 | 9 | 1 | 0 | 90.0% |
| string | 5 | 3 | 2 | 0 | 60.0% |
| table | 1 | 1 | 0 | 0 | 100.0% |
| unanswerable | 4 | 2 | 1 | 1 | 50.0% |
| boolean | 0 | 0 | 0 | 0 | — |

Mẫu ngẫu nhiên này không có câu nhãn boolean; không suy kết quả của type đó từ lượt này. Table chỉ có một câu nên độ bao phủ rất nhỏ.

## Rà soát 65 câu chưa đúng

Đã xem câu hỏi, dự đoán, nhãn và trace của cả 47 câu sai lẫn 18 câu bỏ trống. Mỗi câu được gán một **nhóm công việc ưu tiên**; có thể có nguyên nhân phụ chồng lấn. Đây không phải thí nghiệm chứng minh sửa một nhóm sẽ tự động cộng từng ấy câu đúng.

| Nhóm ưu tiên | Số câu | Ví dụ / việc cần làm |
|---|---:|---|
| Mở rộng công cụ và phép tính | 31 | Giá nhập bình quân theo lượng; tổng nhập; ca sáng/chiều; dòng tiền nợ thêm/trả nợ; ngưỡng doanh thu/ngày/đơn; số khách mua ít nhất N đơn; tỷ trọng kê đơn/kênh; hoạt chất, size và hàng không bán |
| Hợp đồng, định tuyến, ánh xạ pipeline | 14 | Regex `huy` khớp nhầm trong `chuyen`; mốc 1/1 bị override thành hôm nay; kỳ so sánh lệch năm; quy đổi che mất thông tin catalog; unit sentinel; nồng độ thuốc có % bị hiểu là type percent; alias thanh toán |
| Model chọn công cụ/SQL/đầu ra khi đã có khả năng tương ứng | 20 | Lấy tiền thay lượng; trung bình dòng thay trung bình đơn; bỏ valid_sale; nhầm danh sách tháng với boolean; số khách nợ với số tiền nợ; sai thực thể, kiểu hoặc thứ tự đầu ra |

Tổng hợp nhóm/issue: `validation200_error_categories.json`. Chi tiết theo ID được giữ local trong `artifacts/frozen_validation200/error_workstreams.json`. Phân loại không nằm trong solver và không được dùng để tạo đáp án.

Một lỗi dữ liệu đã xác minh: ở một shop, bảng quy đổi đứng trước catalog đầy đủ trong chỉ mục tên sản phẩm. Entry không đủ mã/nhóm che mất entry catalog, khiến sales chỉ nối được 22 mã cho 75 tên; phép tính nhóm hàng/kê đơn có thể sai dù model chọn đúng công cụ. Lỗi này cần sửa lớp dữ liệu, không giải quyết bằng train model đoán số khác.

## Quyết định về huấn luyện

**Chưa chạy SFT hoặc GRPO ở bước này.** 45/65 trường hợp ưu tiên nằm ở công cụ hoặc pipeline. SFT không thể làm đúng dữ liệu đã nối sai hoặc vượt một hard gate chọn nhầm ý định.

Thứ tự tiếp theo:

1. Sửa ranh giới từ của router, ý định/kiểu từ câu hỏi, mốc thời gian, ưu tiên catalog đầy đủ, chuẩn hóa enum và unit. Giữ các trường hợp lỗi làm regression với dữ liệu tổng hợp.
2. Hoàn thiện các phép tính còn thiếu, ưu tiên giá nhập bình quân, ca bán, dòng tiền công nợ, ngưỡng/group aggregate và phép tỷ trọng có thể kết hợp. Giữ timestamp và thuộc tính chuyên ngành cần thiết.
3. Chạy một bộ ngẫu nhiên mới, loại toàn bộ 290 ID đã xem. Bộ 200 hiện tại đã được phân tích lỗi nên là validation/regression, không còn là test mù cho bản sửa kế tiếp.
4. Khi công cụ đúng, nếu lỗi chọn công cụ/tham số/SQL vẫn chiếm phần lớn, thử SFT nhỏ trên các trajectory có kết quả được kiểm chứng, giữ ngân sách Colab. Chỉ xét GRPO sau khi reward và bộ chấm có thể kiểm tra đúng nghiệp vụ.

Chưa chạy 1.000 câu public và chưa có bằng chứng đạt 1.000/1.000.

## Hiệu năng và tái lập

- Qwen3.5-9B Q4_K_M; llama.cpp b11960/CUDA 13.4; RTX 5060 Laptop; context 16K; một slot; thinking tắt; seed model 42.
- 416 lượt model, 40 câu không cần gọi model nhờ định tuyến SQL. Trung vị 3,86 giây/câu; p95 khoảng 17,03 giây.
- Dữ liệu thô và typed cache đều đã có. Thời gian trên không gồm tải weights hoặc khởi động server.
- Năm mẫu nvidia-smi lưu local cho thấy GPU được sử dụng; mẫu đầu khoảng 6.163 MiB VRAM, 95% utilization. Đây là mẫu quan sát, không phải giá trị trung bình toàn lượt hay chứng nhận mọi layer offload.
- Bộ chấm là bản local, cần đối chiếu scorer chính thức nếu BTC cung cấp.

Chạy lại theo README. Không ghi đè output cũ. Các báo cáo JSON tổng hợp không chứa bản sao dữ liệu train, prompts hay đáp án gốc; các dữ liệu đó ở thư mục local được Git bỏ qua.
