# Trạng thái tiếp tục — 02/11/2026

## Kết quả hiện tại

- **Đánh giá 200 câu ngẫu nhiên mới:** 135 đúng, 47 sai, 18 bỏ trống; accuracy 67,5%, PA 0,44; 17 phút 50 giây, 416 lượt model.
- **Bộ phát triển 90 câu trước đó:** 89 đúng, 0 sai, 1 bỏ trống. Không dùng điểm này làm độ đúng câu mới.
- Solver giữ nguyên trong lượt 200 câu; hash trước/sau khớp. Chưa sửa solver theo 65 lỗi đã rà soát.
- Không huấn luyện, 0 Colab CU. Đã tắt server sau đánh giá và sau kiểm tra script.

## Quyết định và bước tiếp theo

Chưa train SFT/GRPO: 31 trường hợp ưu tiên mở rộng công cụ, 14 ở hợp đồng/định tuyến/ánh xạ, 20 ở lựa chọn công cụ/SQL/đầu ra của model. Các nhóm có thể có nguyên nhân phụ chồng lấn.

Ưu tiên sửa lớp dữ liệu và router trước, rồi thêm phép tính về nhập hàng, ca bán, dòng tiền nợ, ngưỡng và tỷ trọng. Sau đó chọn bộ mới loại 290 ID đã xem (`benchmarks/seen290.json`). Không train trên nhãn của validation200 hoặc dùng điểm sau chỉnh theo lỗi để gọi là test độc lập.

[Báo cáo đầy đủ](reports/validation200.md) · [Hướng dẫn chạy](README.md).

## Những gì được giữ

- `benchmarks/train10.json`: smoke test phát triển.
- `benchmarks/development90.json`: regression 90 câu.
- `benchmarks/validation200_random.json`: mẫu cố định 200 câu.
- `benchmarks/seen290.json`: chỉ để loại các ID đã xem khi lấy mẫu tiếp theo.
- `reports/`: điểm, phân loại lỗi tổng hợp và provenance để đưa lên GitHub; không chứa ID câu/doanh nghiệp.
- `artifacts/frozen_validation200/`: trace/nhãn đầy đủ ở local, không push.
- Manifest các vòng trung gian chuyển vào `artifacts/history_suites/`; code cũ ở nhánh dev. Các manifest JSON active cũng chỉ ở local, được tái tạo bằng `scripts/rebuild_suites.py`, không push lên GitHub.

Dữ liệu gốc dùng tại `E:\ZALO_AI_Challenge\train` và `public_test`, ngoài repo. Hai bản sao trong repo đã xác minh 259 tệp trùng SHA-256 và được giữ lại theo lựa chọn của người dùng. `train/`, `public_test/` và tên thay thế `public-test/` đều được Git bỏ qua.

Script start-server đã sửa tương thích Windows PowerShell 5 và kiểm tra khởi động/endpoint thành công. Công cụ lấy mẫu ngẫu nhiên tái tạo đúng 200/200 ID. Xem README cho lệnh setup, tải model/runtime, chạy server, benchmark và audit.

Nhánh làm việc: `benchmark/local-clean`. Không force-push hoặc xoá lịch sử dev/main.
