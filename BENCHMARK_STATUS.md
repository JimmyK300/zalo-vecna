# Trạng thái tiếp tục — 03/11/2026

## Kết quả mới nhất

- Cùng 100 câu mới, bản trước sửa: 71 đúng/22 sai/7 bỏ trống, PA 0,49, wall 575,19 giây. Bản sửa đóng băng: 71 đúng/18 sai/11 bỏ trống, PA 0,53, wall 444,56 giây. Có 8 câu được sửa thành đúng và mất 8 câu đang đúng; không tăng accuracy.
- Trên 20 câu chọn sẵn từ mẫu 100: hybrid 11 đúng/6 sai/3 bỏ trống; model tự viết SQL 9 đúng/10 sai/1 bỏ trống, cần 88 lượt model so với 45. Mẫu nhỏ, chưa kết luận cho mọi model/nhiệm vụ.
- Đối chiếu 966 phép tính SQL/Python cùng dữ liệu: 966 khớp, 0 lệch. Không chứng nhận ánh xạ hoặc ý định đúng.
- Model Qwen3.5-9B Q4_K_M local, không train, 0 Colab CU. Thời gian QA có cache/model đã nạp, chưa kiểm chứng giới hạn 6 giờ trên máy BTC.
- Chi tiết: [báo cáo SQL](reports/sql_validation.md), [JSON tổng hợp](reports/sql_validation_summary.json).

## Code hiện tại và sửa sau đánh giá

Đã sửa ranh giới từ huỷ/chuyển khoản, nhận type phần trăm, quý và khoảng từ đầu năm, kỳ so tháng trước, catalog bị bảng quy đổi che, đơn vị giá bán và alias thanh toán. Bổ sung purchase_receipts/purchase_lines, tổng tiền nhập, giá nhập gia quyền và tỷ trọng doanh thu/kê đơn. `--tool-policy generated_sql` dùng để đối chứng; mặc định vẫn hybrid.

Các lượt 100/20 dùng bản cache v14 đóng băng. Sau xem kết quả, phát hiện và sửa lỗi mất nhà cung cấp khi tổng hợp sổ nhập dạng dòng; cache tăng v15, 28 unit tests qua, ca nhà cung cấp kiểm tra lại đúng. Điểm kiểm tra lại được ghi riêng, không cộng vào điểm mù. Chưa chạy lại toàn bộ 100 câu bằng model trên bản v15.

## Bước tiếp theo

Giữ SQL làm bộ tính toán; chưa chuyển sang model tự viết mọi truy vấn. Ưu tiên:

1. Giữ đầy đủ thuộc tính khi ánh xạ, đối chiếu bảng chuẩn hoá với nguồn.
2. Kiểm soát thực thể, làm tròn tiền, dạng list và xử lý thiếu dữ liệu; các lỗi model chọn công cụ/SQL vẫn còn.
3. Mở rộng công cụ ca bán, dòng tiền nợ, ngưỡng và tần suất khách khi có kiểm tra nguồn độc lập.
4. Chọn câu mới loại `benchmarks/seen402.json`; không gọi bộ 100 vừa rà soát là test mù nữa.
5. Chỉ thử SFT nhỏ khi lỗi còn lại chủ yếu thuộc model. Chưa có căn cứ chọn GRPO ở bước này.

## Thử chuẩn hoá trước đó

12 câu mới, seed 2026110301: câu gốc 9/12 đúng, Qwen chuẩn hoá có kèm câu gốc 7/12. Không bật viết lại câu hỏi mặc định. [Báo cáo](reports/normalization12.md). Lựa chọn cuối của người dùng là chỉ thử agent, chưa train GRPO.

## Tệp và lịch sử

- Local ignored: `artifacts/sql100_before/`, `sql100_after/`, `sql20_generated/`, `supplier_regression_after_fix/`, các trace và nhãn. Snapshot solver đóng băng: `artifacts/sql_fixes_source/solver.zip`; bản đối chứng: `artifacts/baseline_before_sql_fixes/`.
- Suite local: train10, development90, validation200_random, normalization12_random, sql100_random, sql20_paired; bộ loại seen290/302/402. Tái tạo ID bằng `scripts/rebuild_suites.py`.
- Lịch sử: bộ phát triển 90 câu đạt 89 đúng, 0 sai, 1 bỏ trống; bộ 200 câu mới trước đó đạt 135 đúng, 47 sai, 18 bỏ trống. Không so trực tiếp các tập khác nhau.
- Dữ liệu gốc ở `E:\ZALO_AI_Challenge\train` và `public_test`. Bản sao trong repo giữ nguyên theo yêu cầu, gitignore `train/`, `public_test/`, `public-test/`.
- Nhánh `benchmark/local-clean`; main/dev giữ nguyên. README có hướng dẫn chạy. Git chỉ chứa code và kết quả tổng hợp, không chứa dữ liệu/ID/trace.
