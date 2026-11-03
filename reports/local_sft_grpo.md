# SQL + SFT + Dr. GRPO trên RTX 5060 — 03/11/2026

**Đã train thật hoàn toàn local.** Pilot dùng Qwen2.5-1.5B-Instruct, một bước chọn công cụ/tham số rồi thực thi SQL. SFT đạt 19/40 câu đúng; thêm Dr. GRPO đạt 20/40, không mất câu SFT đang đúng. Chỉ tăng một câu nên chưa chứng minh GRPO cải thiện ổn định.

Đây không phải fine-tune model 9B hay thay agent nhiều lượt đang có. Adapter 1.5B không dùng được cho model 9B; không so trực tiếp điểm 40 câu này với 71/100 của agent 9B.

## Kết quả

Protocol chính yêu cầu `{"action":"metric","operation":"...",...}` hoặc `{"action":"abstain"}`. Công cụ chỉ nhận tham số, tự chạy SQLite để tính đáp án; model không được ghi số đáp án.

| Model, cùng protocol và executor | Đúng | Sai | Bỏ trống | PA | Action không hợp lệ | Giây suy luận 40 câu |
|---|---:|---:|---:|---:|---:|---:|
| Gốc 1.5B | 0 | 0 | 40 | 0 | 40 | 114,78 |
| LoRA SFT + SQL | 19 | 7 | 14 | 0,30 | 14 | 115,49 |
| SFT → Dr. GRPO + SQL | 20 | 7 | 13 | 0,325 | 13 | 97,74 |

Action lỗi được pipeline đổi thành abstain khi suy luận; trong reward train, action lỗi nhận −1 để model không có lợi khi phá định dạng. Không có trường hợp model tự đoán số rồi được chấp nhận làm kết quả tool.

**Không diễn giải 0 → 19 là toàn bộ năng lực suy luận tăng lên.** Bản gốc thường dùng tên operation trực tiếp ở trường `action`; một phần thất bại là lỗi giao thức. Sau khi đã quan sát lỗi này, chấm chẩn đoán lại **cùng đầu ra**, cho cả ba bản qua cùng bộ chuẩn hoá alias operation-as-action:

| Chẩn đoán sau chuẩn hoá alias, không sinh lại | Đúng | Sai | Bỏ trống | PA |
|---|---:|---:|---:|---:|
| Gốc 1.5B | 9 | 9 | 22 | 0 |
| SFT | 19 | 7 | 14 | 0,30 |
| SFT + Dr. GRPO | 20 | 7 | 13 | 0,325 |

Đây là phân tích bổ sung sau quan sát, không phải test mù độc lập. Không dùng kết quả này để chọn checkpoint hoặc chỉnh lịch train. Trước và sau GRPO có 1 câu từ chưa đúng thành đúng, 0 câu đúng thành chưa đúng. Câu được cải thiện chuyển từ operation không tồn tại `growth` sang `revenue_growth`; SQL sau đó tính khớp đáp án. Lợi ích quan sát này vẫn thuộc lựa chọn công cụ, không phải model tự học tính toán số học.

## Dữ liệu và tránh rò nhãn

- 128 ví dụ SFT được lấy từ trace các câu đã xem, chỉ nhận thao tác có kết quả được chạy lại trên SQL hiện tại và khớp nhãn train. SFT chỉ học JSON công cụ/tham số, không học đáp án số.
- 16 ví dụ phát triển để riêng, chưa dùng để chỉnh siêu tham số hoặc chọn checkpoint trong pilot này.
- 40 câu mới lấy mẫu đều từ 2.598 câu còn lại sau khi loại 402 câu đã xem, seed `2026110304`. Không lọc theo nhãn/độ dễ/operation. Tập train, development và evaluation không giao ID.
- Câu hỏi giữ nguyên. Context gồm ngày tham chiếu, kỳ do parser xác định, khả năng dữ liệu, nhóm/kênh/thanh toán và tên hàng được tìm từ dữ liệu. Không đưa nhãn, type đáp án hay ID câu/doanh nghiệp vào prompt.
- GRPO chỉ tra nhãn của câu train. Nhãn 40 câu evaluation chỉ dùng để chấm sau khi cả ba bản đã suy luận xong.
- Cùng dữ liệu, executor, prompt và greedy generation cho ba bản. Model/task nhỏ, một seed, câu mới nhưng định dạng doanh nghiệp đã gặp. Chấm local, chưa phải scorer BTC.

## Huấn luyện thực tế

- Weights gốc cached: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Không dùng adapter cũ.
- CUDA RTX 5060 Laptop 8 GB; PyTorch `2.11.0+cu128`, Transformers `5.21.0`, TRL `1.17.1`, PEFT `0.21.2`. Môi trường `.train-venv` riêng, tái dùng runtime torch có sẵn, không sửa `.venv` benchmark.
- BF16 LoRA rank 8, alpha 16, q_proj/v_proj, 1.089.536 tham số train được. Batch train 1, gradient accumulation 2, gradient checkpointing; không vLLM, không Colab, không tracking ra ngoài.
- Prompt thực tế 835–1.134 token, completion tối đa 128, không cắt câu hỏi.
- SFT: 64 optimizer steps, một epoch/128 ví dụ, learning rate 1e−4; train **114,94 giây**, gồm khởi tạo trainer/lưu khoảng **118,80 giây**.
- Dr. GRPO tiếp từ SFT: 24 steps, 2 rollout/câu, 48 rollout, learning rate 5e−6, `loss_type=dr_grpo`, `scale_rewards=none`, `beta=0`; train **115,12 giây**, gồm lưu khoảng **115,68 giây**.
- Reward: đáp án SQL khớp +1; sai/action lỗi −1; abstain hợp lệ 0. Có 24 rollout +1 và 24 rollout −1; không thưởng riêng vì JSON đúng.
- Chỉ **4/24 bước có gradient khác 0**; 20/24 nhóm có cùng reward nên không có tín hiệu tương đối. So tensor adapter SFT/GRPO: 112 tensor thay đổi, L2 delta khoảng 0,0180. GRPO thực sự cập nhật trọng số, không chỉ chạy suy luận.
- Peak allocated do PyTorch ghi nhận trong tiến trình 4,17 GiB; đây không phải toàn bộ VRAM sử dụng/reserved. Có mẫu nvidia-smi 7.867 MiB khi SFT, không phải đo peak toàn phiên. Không suy ra từ pilot này rằng model 9B train tương tự sẽ vừa 8 GB.

Tổng thời gian SFT và GRPO chỉ khoảng 4 phút vì model, dữ liệu và số bước rất nhỏ; chưa gồm cài môi trường, chuẩn bị dữ liệu, nạp model và ba lượt đánh giá. Không phải ước lượng train model 9B cho toàn bộ cuộc thi. Colab CU: **0**.

## Tệp và chạy lại

Chạy trong thư mục repo, PowerShell:

```powershell
# Tái dùng Python có CUDA torch; có thể truyền -TorchPython đường dẫn khác
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-local-training.ps1
# Nếu các trace phát triển local trước đây còn tồn tại
.\.venv\Scripts\python.exe scripts/prepare_local_training.py --output artifacts/local_training_dataset_new
# Chạy ba bản; output mới, dữ liệu/model đều local
.\.train-venv\Scripts\python.exe -u -X utf8 scripts/train_local_sql_pilot.py --data artifacts/local_training_dataset_new/dataset.json --output artifacts/local_training_run_new
.\.train-venv\Scripts\python.exe scripts/audit_local_training.py --data artifacts/local_training_dataset_new/dataset.json --run artifacts/local_training_run_new --output reports/local_training_new_summary.json
```

Script train dùng weights pinned đã cache trong Hugging Face và `HF_HUB_OFFLINE=1`; nếu chưa có cache cần tải weights gốc trước. Không tải/upload dữ liệu trong lúc train. `--probe-only` thử CUDA trước. Không ghi đè run đã hoàn tất. Có thể chạy lại lệnh khi bị ngắt sau một giai đoạn đã lưu xong, với nguyên code/dataset/siêu tham số; không tự resume giữa optimizer step.

Trace huấn luyện mẫu không được đưa lên Git. Trên clone mới chưa có trace, tạo teacher run bằng Qwen 9B/server local và CLI benchmark với `benchmarks/seen402.json`, rồi truyền `--teacher-run <thư mục có scores.json>` cho script prepare; chỉ các câu có thao tác SQL khớp nhãn được nhận. Dữ liệu tái tạo có thể khác dataset pinned trong báo cáo; SHA của dataset thật nằm trong JSON tổng hợp.

Các tệp thật ở local:

- `artifacts/local_sft_grpo/dataset.json`: dữ liệu và SQL context đã chốt.
- `artifacts/local_sft_grpo/run1/sft_adapter/`: adapter sau SFT.
- `artifacts/local_sft_grpo/run1/grpo_adapter/`: adapter sau SFT + Dr. GRPO.
- Cùng thư mục run: config/hash, trace, reward, train history và điểm từng câu; không push lên GitHub.
- [JSON tổng hợp](local_sft_grpo_summary.json): số liệu, phiên bản, hash và kiểm tra tensor; không chứa ID hay dữ liệu từng câu.

`scripts/rebuild_suites.py` tái tạo đúng 40 ID đánh giá và bộ loại `benchmarks/seen442.json`. Khi tiếp tục lấy mẫu, loại cả 442 câu đã xem. Không dùng lại 40 câu này làm test mù sau chỉnh sửa.

## Quyết định

Giữ agent 9B hiện tại làm nền. Pilot chứng minh SQL + SFT + GRPO chạy được trên laptop và có thể học giao thức/công cụ. SFT có tín hiệu hữu ích; GRPO chỉ thêm một câu với rất ít nhóm có tín hiệu, chưa đáng tăng mạnh thời gian train dựa trên kết quả này. Trước lượt lớn hơn, cải thiện dữ liệu thao tác, xử lý alias/operation lỗi và đánh giá trên tập mới. Chưa có bằng chứng đạt 1.000/1.000 hoặc giới hạn 6 giờ trên máy BTC.

Nguồn phương pháp: [model card Qwen chính thức](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [TRL GRPO và Dr. GRPO](https://huggingface.co/docs/trl/grpo_trainer), [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora).
