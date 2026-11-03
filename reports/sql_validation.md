# Kiểm tra SQL và pipeline — 03/11/2026

**SQL phù hợp làm bộ tính toán. Chưa có bằng chứng rằng để model tự viết toàn bộ SQL sẽ tốt hơn công cụ SQL theo mẫu.** Bản cải tiến giảm thời gian và số câu sai, nhưng chưa tăng số câu đúng trên 100 câu mới.

## 1. Đối chứng 100 câu mới

| Phiên bản | Đúng | Sai | Bỏ trống | Accuracy | PA | Thời gian có cache | Lượt model |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trước sửa | 71 | 22 | 7 | 71% | 0,49 | 575,19 giây | 220 |
| Bản sửa được đóng băng để đánh giá | 71 | 18 | 11 | 71% | 0,53 | 444,56 giây | 183 |

- 8 câu chưa đúng được sửa thành đúng, đồng thời mất 8 câu đang đúng. Không gọi đây là tăng accuracy.
- Thời gian giảm khoảng 22,7%; số lượt model giảm 16,8%. PA tăng một phần do bỏ trống thay cho đáp án sai; coverage giảm 93% xuống 89%.
- P95 thời gian giải: 16,45 → 11,08 giây; trung vị: 3,78 → 3,84 giây. Mức giảm tổng thời gian chủ yếu ở những câu trước đây cần nhiều lượt.

Lấy mẫu đều không dùng nhãn từ 2.698 câu còn lại sau khi loại 302 câu đã xem; 100 câu, 27 doanh nghiệp, seed `2026110302`. Cùng câu, model Qwen3.5-9B Q4_K_M, llama.cpp b11960, nhiệt độ 0, seed 42, 12 bước và 240 giây/câu. Bản trước sửa được tách vào thư mục riêng; bản mới đóng băng trước khi xem điểm đối chứng. Hash không đổi trong từng lượt.

Đây là tách theo câu hỏi, không phải doanh nghiệp mới. Mỗi cấu hình chạy một lượt, chưa đo biến động nhiều seed. Không so trực tiếp 71% lần này với 67,5% của bộ 200 câu khác để kết luận tiến bộ.

## 2. Model tự sinh SQL so với công cụ theo mẫu

Chọn trước 20 câu đầu của cùng mẫu ngẫu nhiên 100 câu. Hai chế độ dùng cùng bản dữ liệu/solver, chỉ khác chính sách công cụ và prompt tương ứng.

| Chế độ | Đúng | Sai | Bỏ trống | PA | Thời gian giải | Lượt model |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid: SQL theo mẫu + SQL tự viết khi cần | 11/20 | 6 | 3 | 0,25 | 111,81 giây | 45 |
| Tắt metric, model tự sinh SQL | 9/20 | 10 | 1 | −0,05 | 182,49 giây | 88 |

SQL tự sinh giải thêm được 1 câu hybrid chưa đúng nhưng mất 3 câu hybrid đang đúng. Có truy vấn chạy thành công nhưng chọn sai thời gian, phép tính, quy tắc nghiệp vụ hoặc định dạng trả lời.

Đây là **hai cách dùng SQL**, không phải SQL đối chứng với phương pháp không dùng SQL. Mẫu 20 câu chỉ là pilot, chưa đủ kết luận cho mọi loại câu hay model khác.

## 3. Kiểm tra lớp tính toán

Đối chiếu 966 phép tính SQL với vòng lặp Python độc lập trên cùng bảng chuẩn hoá của 27 doanh nghiệp: **966 khớp, 0 lệch**. Bao gồm doanh thu, số đơn, số huỷ, lượng cơ bản, tổng tiền nhập và bình quân giá nhập theo lượng. Lượt đóng băng đo trung vị 2,48 ms/truy vấn, tối đa 77,38 ms; 966 truy vấn cộng khoảng 4,92 giây.

Kiểm tra này không gọi model, không đọc đáp án tham chiếu và không chứng nhận ánh xạ bảng đúng: SQL và Python có thể cùng tính trên dữ liệu đã ánh xạ sai. Vì vậy cần cả test dữ liệu và đánh giá câu hỏi thực.

Các ca dữ liệu tổng hợp kiểm tra riêng việc không đếm dòng thành đơn, không cộng lại tổng phiếu theo từng dòng, bỏ phiếu huỷ, bình quân theo số lượng thay vì AVG đơn giá, ưu tiên danh mục đầy đủ hơn bảng quy đổi, alias thanh toán và tỷ trọng trên doanh thu cả shop.

## 4. Các thay đổi

- Ranh giới từ khi nhận đơn huỷ; không coi chữ trong “chuyển khoản” là huỷ.
- Không suy type percent chỉ vì nồng độ thuốc có ký hiệu `%`.
- Quý theo lịch, khoảng từ 1/1 đến hôm nay, năm của cả kỳ hiện tại và kỳ so tháng trước; thêm dạng ngày sự kiện được nhắc lại bằng “ngày đó”.
- Ưu tiên danh mục sản phẩm đầy đủ khi nối tên/mã; chấp nhận `base_unit` trong công cụ giá bán.
- Bảng `purchase_receipts` và `purchase_lines`, giữ nguồn và đơn vị; công cụ tổng tiền nhập và bình quân gia quyền.
- Alias thanh toán ưu tiên giá trị thật đang có; thêm tỷ trọng doanh thu theo kênh/thanh toán và tỷ trọng kê đơn.
- Thêm chế độ `--tool-policy generated_sql` để thực nghiệm, mặc định vẫn hybrid. Không thêm bước Qwen viết lại câu hỏi.

## 5. Sửa hồi quy sau đánh giá

Rà soát bộ 100 phát hiện một lỗi mới: nguồn nhập dạng dòng có nhà cung cấp nhưng phép tổng hợp thành phiếu làm mất trường đó. Đã sửa để giữ nhà cung cấp, tăng phiên bản cache từ v14 lên v15 và thêm test hồi quy. Đây là lỗi ánh xạ, không phải SQLite cộng sai.

**Bảng điểm 100/20 phía trên thuộc bản v14 đóng băng trước sửa cuối.** Kiểm tra lại câu nhà cung cấp và kiểm tra dữ liệu sau sửa được ghi riêng, không cộng vào điểm test mù. Chưa chạy lại toàn bộ 100 câu bằng model sau sửa cuối; khi chạy lại, đó là regression, không còn là đánh giá mù. Kết quả kiểm tra lại nằm trong `reports/sql_postfix_summary.json` và chi tiết local `artifacts/supplier_regression_after_fix/`.

Bản v15 cuối: **28/28 unit tests qua, câu nhà cung cấp kiểm tra lại đúng**, đối chiếu lại toàn bộ **966/966 phép tính SQL/Python khớp** trên 27 doanh nghiệp. Trung vị truy vấn của lượt này 2,20 ms; tổng 4,23 giây. Đây vẫn không phải một lượt QA mù mới.

## 6. Giới hạn thời gian và bước tiếp theo

Các lượt QA đã nạp model và có cache dữ liệu. Số giây trong bảng không gồm tải weights, khởi động model hoặc dựng database lần đầu; **chưa chứng minh đạt giới hạn 6 giờ trên máy BTC**. Chưa chạy Docker trên 2×RTX 5090. Không dùng Colab CU, không SFT/GRPO.

Giữ hybrid làm nền. Tiếp tục kiểm tra ánh xạ và kiểm soát tham số/kết quả công cụ, đặc biệt thực thể, làm tròn tiền, dạng list và phân biệt thiếu dữ liệu với lỗi truy vấn. Chưa dùng pilot này để khẳng định đã sẵn sàng train hay đạt 1.000/1.000.

Trace/nhãn/ID chỉ lưu local: `artifacts/sql100_before/`, `sql100_after/`, `sql20_generated/`, `sql100_changes.json`. Các bộ câu được tái tạo bằng `scripts/rebuild_suites.py`; loại `benchmarks/seen402.json` khi lấy mẫu mới. Báo cáo tổng hợp có thể chia sẻ: [JSON kết quả](sql_validation_summary.json).
