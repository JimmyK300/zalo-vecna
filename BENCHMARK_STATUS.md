# Trạng thái tiếp tục — 03/11/2026

## Mới nhất: verify sau đáp án

- Đã hoàn tất 40 câu mới trên agent Qwen 9B hybrid, seed 2026110305, loại 442 ID cũ. Không train thêm và không dùng Colab. Đây không phải pilot 1.5B đã train.
- Baseline 26 đúng/6 sai/8 abstain, PA 0,50, QA 216,13 giây. Code gate 26/5/9, PA 0,525, thêm khoảng 0,006 giây. Code + model review mọi câu 25/6/9, PA 0,475, QA 723,92 giây; sửa đúng thêm 2 nhưng mất 3 câu đúng.
- 40 review call, tối đa 1 phép sửa; 21 phép sửa hoàn tất, 14 giữ nguyên đáp án. 16 review/sửa không dùng được: 8 hết 1024 token, 4 schema SQL sai, 3 shape/rỗng, 1 tham số không hỗ trợ.
- Chẩn đoán chỉ áp dụng review cho 10 câu có cảnh báo: 27/6/7, PA 0,525, chi phí tăng ước tính 89,52 giây. Đây là tái dùng đầu ra, không phải test mù mới; PA không hơn code-only.
- Chưa bật model verify mọi câu trong agent mặc định. Cần cải thiện fallback khi phép sửa vi phạm hợp đồng và tránh đổi sai ý định/type trước thử trên câu mới. Code check chỉ kiểm một số hợp đồng, không chứng nhận mọi phép nối/nghiệp vụ.
- [Báo cáo/cách chạy](reports/answer_verification.md), [JSON tổng hợp](reports/answer_verification_summary.json). Artifact local: `artifacts/verification40_base/`, `verification40_review_v2/`. Lần đóng gói schema đầu bị ngắt giữ riêng; không đổi policy theo nhãn trong lượt hoàn tất.
- 38 unit tests qua; source hash khớp; GPU server đã tắt. Bộ loại tiếp theo: `benchmarks/seen482.json`.

## Mới nhất: SQL + SFT + Dr. GRPO local

- Người dùng yêu cầu kết hợp SQL/SFT/GRPO và chỉ dùng local vì Colab đã hết CU. Đã hoàn tất pilot thật bằng Qwen2.5-1.5B-Instruct cached, LoRA, trên RTX 5060; không dùng adapter cũ, không thay agent 9B.
- 128 ví dụ SFT được SQL xác minh, 16 ví dụ development để riêng chưa dùng tuning, 40 câu ngẫu nhiên mới loại 402 ID cũ. Seed evaluation 2026110304; các tập không giao ID. Nhãn evaluation chỉ dùng chấm sau cả ba lượt suy luận.
- Protocol nghiêm: gốc 0 đúng/0 sai/40 bỏ trống (40 action lỗi); SFT 19/7/14, PA 0,30; SFT + GRPO 20/7/13, PA 0,325. Sau chuẩn hoá alias cùng đầu ra để chẩn đoán: gốc 9/9/22; hai bản train giữ nguyên. Không gán toàn bộ cải thiện cho suy luận.
- SFT 64 steps/114,94 giây; Dr. GRPO 24 steps/115,12 giây, group 2, 48 rollout, 4 bước gradient khác 0, 20 nhóm cùng reward. Adapter thật thay đổi; GRPO chỉ thêm 1 câu nhờ chọn đúng tên tool, chưa chứng minh hiệu quả ổn định.
- Adapter và trace local ở `artifacts/local_sft_grpo/run1/`; [báo cáo/cách chạy](reports/local_sft_grpo.md), [JSON tổng hợp](reports/local_sft_grpo_summary.json). 31 unit tests qua; script setup local đã chạy thành công. GPU đã được giải phóng khi tiến trình kết thúc.
- Lượt train đã loại 402 ID cũ; các lần sau tiếp tục theo bộ loại mới nhất ở đầu tài liệu. Chưa train 9B; adapter 1.5B không dùng cho 9B. Không chi Colab CU, không kiểm chứng Docker/6 giờ BTC.

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
4. Chọn câu mới loại `benchmarks/seen482.json`; các bộ đã rà soát không còn là test mù.
5. Pilot 1.5B đã xác nhận có thể train local. Trước train lớn hơn, cải thiện dữ liệu công cụ và tín hiệu reward; chưa tăng mạnh GRPO dựa trên chỉ 1 câu cải thiện.

## Thử chuẩn hoá trước đó

12 câu mới, seed 2026110301: câu gốc 9/12 đúng, Qwen chuẩn hoá có kèm câu gốc 7/12. Không bật viết lại câu hỏi mặc định. [Báo cáo](reports/normalization12.md). Ở lượt đó chỉ thử agent; yêu cầu sau đã chuyển sang pilot train local như phần mới nhất.

## Tệp và lịch sử

- Local ignored: `artifacts/sql100_before/`, `sql100_after/`, `sql20_generated/`, `supplier_regression_after_fix/`, các trace và nhãn. Snapshot solver đóng băng: `artifacts/sql_fixes_source/solver.zip`; bản đối chứng: `artifacts/baseline_before_sql_fixes/`.
- Suite local: train10, development90, validation200_random, normalization12_random, sql100_random, sql20_paired, local_training40_random, verification40_random; bộ loại seen290/302/402/442/482. Tái tạo ID bằng `scripts/rebuild_suites.py`.
- Lịch sử: bộ phát triển 90 câu đạt 89 đúng, 0 sai, 1 bỏ trống; bộ 200 câu mới trước đó đạt 135 đúng, 47 sai, 18 bỏ trống. Không so trực tiếp các tập khác nhau.
- Dữ liệu gốc ở `E:\ZALO_AI_Challenge\train` và `public_test`. Bản sao trong repo giữ nguyên theo yêu cầu, gitignore `train/`, `public_test/`, `public-test/`.
- Nhánh `benchmark/local-clean`; main/dev giữ nguyên. README có hướng dẫn chạy. Git chỉ chứa code và kết quả tổng hợp, không chứa dữ liệu/ID/trace.
