# Chạy DataMate từ GitHub + Google Drive

Mở `DataMate_Submission_Colab.ipynb` trong Colab, chọn **A100 + High RAM**.
Notebook lấy code từ `feat/query-scoring`, dùng Drive của nhóm và chạy full pipeline Jimmy:
recipe trả lời trực tiếp trước; câu còn lại qua model local và SQL/Python. Không fine-tune.

1. **M0:** Clone GitHub và cài thư viện. Không cần GitHub token vì repo public.
2. **M1:** Đăng nhập Google có quyền đọc Drive. Mặc định tải folder `public_test` bằng ID,
   gồm cả rulebook riêng của public test. Mount My Drive để lưu output.
3. **M2:** Giữ cấu hình mặc định cho lần đầu. Model là checkpoint text-only
   `VikramPal/Qwen3.5-9B-Agentic-GRPO`, pin revision, quantize NF4, một worker.
   Consistency=1, verify tắt để hạn chế vấn đề lặp từng thấy với checkpoint này.
4. **M3:** Đặt `RUN_SMOKE=True`, chạy cell. Bước này ép 3 câu đi qua model để kiểm tra tool calling.
   Xem `smoke/console.log` và trace. Qua smoke chỉ chứng minh kỹ thuật, không chứng minh đáp án đúng.
5. **M4:** Đặt `RUN_FULL=True`, chạy cell. Log cập nhật trên màn hình; output được copy sang Drive
   khoảng mỗi 60 giây. Không cần chạy các notebook fine-tune trước đó.
6. **M5:** Đặt `EXPORT_SUBMISSION=True`. Cell kiểm tra ID/format, đếm abstain và tải file.
   File để nộp là **submission.jsonl**, không cần ZIP.

Kết quả: `My Drive/DataMate_Submissions/public_agentic_nf4_v1/full/submission.jsonl`.
Log ở `full/logs/`, thống kê ở `M5.json`, provenance ở `manifest.json`.
Không tự upload leaderboard.

**Resume:** giữ `RUN_NAME`, dán `commit_sha` trong manifest cũ vào `CODE_REVISION`, chạy M0–M2,
rồi M4–M5. Các câu đã trả lời được bỏ qua; abstain được retry. Dữ liệu từ lần sync gần nhất được khôi phục.
Đổi model, revision, data, quantization hay cấu hình thì dùng `RUN_NAME` mới.
Nếu ngắt giữa lúc ghi file, kiểm tra log/JSON trước khi resume; giữ bản backup trên Drive.

**Phạm vi bản nâng cấp:** scorer tiền không dùng tolerance tỷ lệ; mặc định chấm đủ ID gold;
thêm bản đọc câu hỏi viết tắt cho retrieval/model nhưng giữ câu gốc;
backend Transformers dùng tool XML native và từ chối cắt context khi quá dài;
notebook khóa config, sync kết quả và có milestone.
Chưa chứng minh tăng PA so với baseline; chưa benchmark GPU/giới hạn 6h của BTC.
Checkpoint Agentic là ứng viên inference, không tự động được xem tốt hơn stock Qwen.
Recipe/filter mới và Docker tối ưu cho RTX 5090 cần đánh giá riêng theo kế hoạch.

**Đánh giá train:** scorer mới mặc định tính trên toàn bộ ID gold, thiếu dự đoán = abstain.
Nếu chỉ đánh giá một fold, chuẩn bị file gold chỉ chứa ID fold đó, gồm cả câu thiếu dự đoán.
`--only-pred` chỉ dành cho debug vì có thể khiến điểm trông cao hơn khi thiếu dự đoán.
Không so trực tiếp điểm cũ trong README với scorer mới.

Tham khảo cách load checkpoint: https://huggingface.co/VikramPal/Qwen3.5-9B-Agentic-GRPO
và API Qwen3.5: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5.
