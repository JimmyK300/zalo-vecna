# Thử chuẩn hoá câu hỏi bằng Qwen — 03/11/2026

Thử nghiệm nhỏ không cho thấy lợi ích: câu gốc đúng 9/12, câu chuẩn hoá đúng 7/12.

| Chế độ | Đúng | Sai | Bỏ trống | Accuracy | PA | Giây giải | Lượt model giải |
|---|---:|---:|---:|---:|---:|---:|---:|
| Câu gốc | 9 | 2 | 1 | 75% | 0,5833 | 52,55 | 21 |
| Qwen chuẩn hoá, model vẫn được xem câu gốc | 7 | 4 | 1 | 58,33% | 0,25 | 65,20 | 36 |

Chuẩn hoá tốn thêm 9,64 giây / 12 lượt model, sửa 10/12 câu. Không train trọng số, không dùng Colab CU. Theo lựa chọn cuối của người dùng, lượt này chỉ thử agent, không GRPO.

## Thiết kế

- Lấy ngẫu nhiên 12 câu từ 2.710 câu còn lại sau khi loại 290 câu đã xem, không dùng đáp án/type để chọn. Seed `2026110301`, bằng `scripts/select_random_suite.py`.
- Câu hỏi mới, nhưng định dạng dữ liệu/doanh nghiệp có thể đã gặp; đây không phải kiểm tra doanh nghiệp mới.
- Qwen3.5-9B Q4_K_M local trên RTX 5060 Laptop, cùng model/runtime với benchmark trước. Prompt chuẩn hoá không nhận đáp án, README hay công cụ; chỉ yêu cầu sửa cách viết và giữ ý nghĩa.
- Chốt cả 12 đầu ra chuẩn hoá trước khi giải. Cùng solver, công cụ, dữ liệu, ngân sách 12 bước/240 giây, temperature 0, seed 42. Luân phiên thứ tự hai chế độ theo câu; không sửa solver giữa lượt, hash khớp trước/sau.
- Nhánh chuẩn hoá đưa câu đã sửa vào router/parser, đồng thời thêm câu gốc vào ngữ cảnh model và yêu cầu theo câu gốc nếu khác nghĩa. Đây là phép so sánh pipeline đó với baseline; không cô lập tác động của việc thêm ngữ cảnh.
- Đáp án tham chiếu chỉ được nạp để chấm sau khi hoàn thành cả hai nhánh. Chấm local, không phải scorer chính thức. Chỉ một lượt cho mỗi nhánh, chưa đo biến động qua nhiều seed.

## Phân tích cặp

- 7 câu đúng ở cả hai nhánh.
- 2 câu đúng thành sai.
- 1 câu bỏ trống thành sai.
- 1 câu sai thành bỏ trống: PA cải thiện, nhưng chưa giải đúng.
- 1 câu sai ở cả hai nhánh.
- **Không có câu chưa đúng nào được sửa thành đúng.**

Qwen vẫn đổi nghĩa dù prompt đã cấm: nhầm từ chỉ thời gian thành loại chứng từ, thêm từ vào tên nhóm hàng, đổi kiểu đóng gói thành tính từ kích thước, làm mất mốc so sánh tháng trước. Có trường hợp model vẫn trả đúng nhờ câu gốc/công cụ, nên đúng đáp án không tự chứng minh câu biên tập tương đương nghĩa.

Cả 12 câu đều vượt kiểm tra giữ nguyên số. Điều đó cho thấy kiểm tra số không đủ để bảo vệ thời gian, thực thể hay ý định. Chưa dùng đầu ra Qwen làm câu hỏi chính trong production. Ưu tiên mở rộng viết tắt bằng quy tắc có phạm vi rõ, giữ câu gốc cho router/parser; nếu thử prompt mới phải đánh giá trên câu mới, không gọi điểm của 12 câu này là test mù nữa.

## Tái chạy

Khởi động server bằng `scripts/start-server.ps1`, rồi ở terminal khác tại thư mục repo:

```powershell
.\.venv\Scripts\python.exe scripts/select_random_suite.py --exclude-suite benchmarks/seen290.json --output benchmarks/normalization12_random.json --size 12 --seed 2026110301
.\.venv\Scripts\python.exe scripts/test_question_normalization.py --output artifacts/normalization12_pilot
```

Lệnh chọn suite từ chối ghi đè: bỏ qua nếu tệp đã tồn tại. Thư mục kết quả hoàn tất cũng từ chối ghi đè; dùng thư mục khác cho lượt mới. `--normalize-only` chỉ tạo câu chuẩn hoá. Lượt bị ngắt có thể tiếp tục từ các tệp đã lưu khi suite/prompt/solver không đổi.

Chi tiết từng câu, đáp án và trace nằm local tại `artifacts/normalization12_pilot/`; không đưa dữ liệu đó lên Git. Báo cáo này chỉ chứa kết quả tổng hợp. Bộ loại trừ local `benchmarks/seen302.json` gồm 290 câu cũ và 12 câu lần này để lấy mẫu tiếp theo.
