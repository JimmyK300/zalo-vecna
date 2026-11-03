# Verify sau mỗi đáp án — 03/11/2026

**Không nên bật model review cho mọi câu với cấu hình này.** Trên cùng 40 câu mới, model review sửa đúng thêm 2 câu nhưng làm mất 3 câu đang đúng, khiến accuracy và PA giảm; tổng thời gian tăng khoảng 3,35 lần. Code check rẻ hơn nhiều, giữ nguyên số đúng và chặn được một đáp án sai.

## Kết quả chính

| Chế độ | Đúng | Sai | Bỏ trống | Accuracy | PA | Thời gian QA |
|---|---:|---:|---:|---:|---:|---:|
| Agent gốc, không verify thêm | 26 | 6 | 8 | 65% | 0,50 | 216,13 giây |
| Thêm code check | 26 | 5 | 9 | 65% | 0,525 | 216,13 giây |
| Code + model review mọi câu, tối đa một phép sửa | 25 | 6 | 9 | 62,5% | 0,475 | 723,92 giây |

Code check tốn thêm khoảng **0,006 giây/40 câu** cho phần kiểm tra thuần trên dữ liệu/context đã có. Nó chuyển một lời khẳng định thiếu dữ liệu sai thành abstain; không tạo thêm đáp án đúng. Kiểm tra bằng code trong pilot này là **gate**, không tự đoán phép sửa hoặc làm tròn rồi khẳng định đúng.

Model review tốn thêm **507,79 giây**, 40 lượt gọi model. Trong các đáp án đang đúng: 2 bị đổi thành abstain, 1 bị đổi thành sai. Có 1 câu sai và 1 câu abstain được sửa thành đúng; đồng thời có 1 abstain bị đổi thành sai.

Không so trực tiếp 26/40 ở đây với 71/100 hay với 20/40 của pilot SFT/GRPO: các bộ câu và/hoặc model, task khác nhau.

## Thiết kế

- Agent và reviewer đều là Qwen3.5-9B Q4_K_M local, không train thêm; đây **không** phải model 1.5B đã train SFT/GRPO.
- Chọn ngẫu nhiên 40 câu từ 2.558 câu còn lại sau khi loại 442 ID đã xem; seed `2026110305`, 22 doanh nghiệp. Không dùng nhãn/type để chọn.
- Chạy agent gốc một lần. Cả hai nhánh verify dùng lại chính các đáp án/trace đó, không sinh lại baseline riêng. Agent gốc dùng 87 lượt model.
- Code checker không đọc nhãn. Reviewer nhận câu gốc, as_of_date, luật liên quan, README, schema, khả năng dữ liệu, trace/kết quả SQL và cảnh báo code. Không nhận đáp án chuẩn/type chuẩn.
- Reviewer trả keep, abstain hoặc một metric/SELECT query để máy thực thi. Không chấp nhận số đáp án do model tự ghi; tối đa một thao tác sửa, không có vòng retry. Metric có thể chạy nhiều SELECT nội bộ; đó vẫn là một công cụ tính toán.
- Nếu review lỗi hoặc phép sửa không tạo được kết quả typed hợp lệ, fallback về đáp án cũ đã qua code gate. Nếu phép sửa thực thi được nhưng không qua kiểm tra code sau sửa, chính sách bảo thủ của lượt này trả abstain; đây là nguyên nhân có thể làm mất đáp án cũ đúng.
- Nhãn chỉ được script review dùng để chấm sau khi lưu xong toàn bộ đầu ra review. Baseline logs được đọc từ tệp không có reference/score, không dùng scores.json làm ngữ cảnh.
- Mã nguồn/prompt được đóng băng trong lượt hoàn tất; hash khớp. Có một lần đóng gói schema bị ngắt ở đầu và được sửa trước lượt hoàn tất; artifact cũ giữ riêng, một lượt review sơ bộ đó không nằm trong thời gian bảng chính. Không thay tiêu chí theo nhãn.

## Code checker kiểm tra gì?

Kiểm tra kiểu đáp án, yêu cầu phần trăm so với nồng độ thuốc, tiền lẻ chưa làm tròn, dạng nhiều giá trị, tỷ trọng ngoài 0–100, một số nhầm phép tính rõ ràng (giá nhập/giá bán, ca bán thiếu lọc giờ, xếp hạng theo số đơn so với tiền), đầu ra metric bị đổi và bằng chứng cho unanswerable. Query rỗng không tự chứng minh thiếu dữ liệu.

SQL scalar khác số cuối hoặc kết quả SQL bị cắt chỉ tạo cảnh báo, vì có thể còn bước biến đổi hợp lệ. Code gate chặn hard issues; không bảo đảm đã hiểu đúng mọi ý định, nối bảng đúng hay parse dữ liệu đúng.

Lượt này có 10 câu bị đánh dấu: 8 abstain, 1 SQL evidence bị cắt, 1 unanswerable không có chứng cứ cấu trúc đủ chắc. Chỉ câu cuối là hard gate. Các guard có thể loại nhầm một unanswerable thực sự trong dữ liệu khác, nên chưa coi pilot 40 câu là chứng nhận triển khai rộng.

## Reviewer đã làm gì?

- 29 lần đề xuất replace; 21 phép sửa hoàn tất, trong đó **14 vẫn cho cùng đáp án cũ**.
- 3 lần keep. 8 lượt dùng hết ngân sách 1.024 token nên không nhận văn bản hoàn chỉnh.
- Tổng 16 lượt review/sửa không dùng được: 8 hết token, 4 dùng tên bảng/cột không tồn tại, 3 có kết quả rỗng hoặc sai hình dạng scalar, 1 dùng tham số công cụ không được hỗ trợ.
- Context ước tính 4.619–7.134 token; không cần cắt context. Không có bằng chứng rằng lỗi do tràn giới hạn context.

Ví dụ lỗi có ích để rút kinh nghiệm:

- Có câu so lượng bán 30 ngày trước/sau tăng giá: reviewer sửa đúng biên ngày và SQL ra đáp án đúng.
- Có câu hỏi **ngày nào doanh thu cao nhất**, đáp án cũ đúng là date nhưng reviewer đổi sang operation **tổng doanh thu**, thành sai.
- Có câu tỷ trọng kê đơn đã đúng nhưng reviewer tưởng mới tính tử số, đề xuất SQL có type number thay vì percent; code check chặn thành abstain.
- Có câu trung bình một đơn: reviewer lấy `COUNT(*)` trên dòng hàng thay vì số đơn, tạo đáp án sai. SQLite vẫn chạy đúng truy vấn được yêu cầu.

Các lời giải thích dài và tự tin không phải bằng chứng độc lập. Cùng model và cùng lớp dữ liệu vẫn có thể lặp lại hoặc tạo thêm giả định sai.

## Chẩn đoán review chọn lọc

Tái sử dụng cùng các review đã sinh, chỉ áp dụng cho 10 câu code đánh dấu, giữ baseline cho 30 câu còn lại: **27 đúng, 6 sai, 7 abstain, PA 0,525**. Không mất câu baseline đang đúng, thêm được một câu đúng nhưng cũng thêm một câu sai so với code-only, nên **PA không hơn code-only**.

Đây là phân tích chẩn đoán trên cùng đầu ra, không phải lượt test mù mới. Chi phí tăng ước tính 89,52 giây nếu chỉ gọi review cho 10 câu; thứ tự/cache của một lượt chạy thật sẽ khác. Chưa dùng con số 27/40 để tuyên bố review chọn lọc cải thiện ổn định.

## Quyết định và giới hạn

Giữ các kiểm tra hợp đồng bằng code. **Chưa bật model verify mọi câu trong agent mặc định.** Trước lượt mới, cải thiện fallback khi phép sửa vi phạm hợp đồng và kiểm soát việc đổi ý định/type; chỉ thử review chọn lọc ở các câu có bằng chứng bất thường. Không tối ưu tiếp trên chính 40 câu này rồi gọi là test mù.

Mẫu nhỏ, một seed, doanh nghiệp/format đã gặp, cùng model solver/reviewer; chưa thử một verifier được train riêng. Không có kiểm chứng tổng quát về ngữ nghĩa SQL hoặc ánh xạ dữ liệu. Caches/model đã nạp; bảng thời gian không gồm nạp model hoặc ingest lạnh. Chưa kiểm tra Docker/6 giờ BTC; Colab CU dùng **0**.

## Chạy lại

Khởi động server bằng `scripts/start-server.ps1`, rồi trong terminal khác ở repo:

```powershell
.\.venv\Scripts\python.exe scripts/rebuild_suites.py
.\.venv\Scripts\python.exe -X utf8 -m datamate_bench run --suite benchmarks/verification40_random.json --output artifacts/verify_base_new --require-cuda
.\.venv\Scripts\python.exe -X utf8 scripts/test_answer_verification.py --base artifacts/verify_base_new --output artifacts/verify_review_new
.\.venv\Scripts\python.exe -X utf8 scripts/audit_answer_verification.py --base artifacts/verify_base_new --run artifacts/verify_review_new --output reports/verify_new_summary.json
```

Output phải mới; script không ghi đè lượt hoàn tất. Có thể tiếp tục từ các câu đã lưu khi code, prompt, suite và baseline không đổi. Bộ kiểm thử có tổng 38 ca, gồm từ chối model tự ghi answer, chặn SQL mutation và fallback khi review lỗi.

Kết quả thật local: `artifacts/verification40_base/`, `artifacts/verification40_review_v2/`; snapshot source trong thư mục review. Trace, câu hỏi, nhãn và suite JSON không push. [JSON tổng hợp](answer_verification_summary.json) chỉ chứa thống kê/hash, không có ID dữ liệu. Khi lấy mẫu tiếp theo, loại `benchmarks/seen482.json`.
