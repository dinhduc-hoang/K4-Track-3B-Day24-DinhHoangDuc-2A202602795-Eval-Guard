# Đánh giá LLM Judge Phase B

Nhóm: DinhHoangDuc (nhóm một người)

Người phụ trách: Đinh Hoàng Đức — 2A202602795

Nguồn: `artifacts/judge_results.json`; run timestamp: 2026-10-12T10:20:32.614556+00:00.
Models generation/judge: generation = `openai/gpt-oss-20b`; judge = `openai/gpt-oss-20b`
(cả hai gọi qua Groq API tương thích OpenAI; pairwise ở `src/phase_b_judge.py`, nhãn 0/1 qua
helper `judge_binary`, cùng `config.JUDGE_MODEL`). Judge và generation dùng chung một model ID —
đây là giới hạn cần lưu ý (judge có thể ưu ái văn phong của chính model đã sinh answer A).

## Năm cặp A/B và hai lượt

Nguồn cố định: A từ artifacts/answers_50q.json:answer; B từ
fixtures/human_labels_10q.json:model_answer; reference từ fixtures/test_set_50q.json:ground_truth.
Ids: 1, 5, 12, 21, 41. Không đổi nguồn sau khi xem kết quả judge.

Lượt 1 đưa A trước, lượt 2 đưa B trước; winner/scores lượt 2 đã quy về danh tính A/B gốc
(`swap_and_average`). Reasoning lượt 2 giữ nguyên văn của judge nên chữ “Answer A” trong đó chỉ
candidate đứng đầu sau swap, tức B gốc.

| id | Candidate A (pipeline) | Candidate B (model_answer) | pass1 | pass2 (quy về gốc) | final | consistent | scores pass1 A/B | scores pass2 A/B |
|---|---|---|---|---|---|---|---|---|
| 1 | “3 ngày làm việc.” | “Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn.” | A | B | tie | False | 0.9 / 0.8 | 0.5 / 0.8 |
| 5 | “Kế toán trưởng.” | “Cần Giám đốc phòng ban phê duyệt.” | tie | tie | tie | True | 0.0 / 0.0 | 0.0 / 0.0 |
| 12 | “1 tháng lương” | “Nhân viên được thưởng Tết tối thiểu 1 tháng lương.” | tie | tie | tie | True | 1.0 / 1.0 | 1.0 / 1.0 |
| 21 | 18 ngày phép; nói tài liệu không có khung lương | “18 ngày phép (15 + 3). Lương Senior từ 20 đến 35 triệu.” | B | B | B | True | 0.3 / 0.9 | 0.0 / 1.0 |
| 41 | “15 ngày phép năm (tăng so với 12 ngày năm 2023)” | “Nhân viên được nghỉ 12 ngày phép năm.” | A | A | A | True | 0.95 / 0.05 | 1.0 / 0.0 |

Reasoning tóm tắt theo JSON:

- id 1 — lượt 1: A đúng 3 ngày và súc tích, cho rằng B “thêm chi tiết về lương không cần thiết”.
  Lượt 2 (B đứng đầu): candidate đầu “đúng 3 ngày có lương”, candidate sau “thiếu trạng thái có lương”.
- id 5 — cả hai lượt: cả hai sai vì reference yêu cầu Tổng Giám đốc phê duyệt đơn trên 50 triệu.
- id 12 — cả hai lượt: cả hai đúng “tối thiểu 1 tháng lương”, không có khác biệt.
- id 21 — cả hai lượt: B khớp 18 ngày và khung 20–35 triệu; A nói không xác định được khung lương.
- id 41 — cả hai lượt: A dùng v2024 hiện hành (15 ngày); B dùng 12 ngày của v2023 đã thay thế.

Position bias count/rate: 1/5 = 0.2 (`position_bias_count` = 1, `position_bias_rate` = 0.2).
Verbosity win rate: 0.5; decisive cases: 2 (`a_wins_a_longer` = 1, `b_wins_b_longer` = 0).

Diễn giải theo nội dung, không chỉ theo tỷ lệ:

- Case không nhất quán duy nhất là id 1. Ánh xạ A/B đã được kiểm tra: lượt 1 chọn A (đứng đầu),
  lượt 2 chọn candidate đứng đầu là B gốc — tức cả hai lượt đều chọn **vị trí đầu**. Hai answer
  gần tương đương (cùng “3 ngày”, B thêm “có lương”), nên quyết định nghiêng theo vị trí khi chất
  lượng sát nhau. Reasoning lượt 1 còn đọc nhầm “có lương” thành “chi tiết về lương”, cho thấy lỗi
  hiểu tiếng Việt của judge góp phần, không chỉ vị trí. Swap-and-average đã chuyển case này về tie.
- Hai case decisive (21, 41) đều được quyết định bởi **độ chính xác so với reference**: ở id 41
  answer dài hơn thắng vì đúng phiên bản v2024; ở id 21 answer ngắn hơn (B) thắng vì A thiếu khung
  lương. Verbosity 0.5 trên 2 case vì vậy không cho thấy judge ưu tiên độ dài.
- Id 21 cũng là một lỗi retrieval của pipeline: contexts của A không chứa bảng lương nên A trả lời
  “không xác định được” — judge phạt đúng theo reference.

## Calibration đúng model_answer

Helper đánh giá đúng question/model_answer đang có trong fixtures/human_labels_10q.json;
không lấy answer mới của pipeline. Prompt chỉ nhận question, answer, reference.
Không gửi human_label, human_note hoặc phân bố nhãn human vào prompt.

Runner (`src/experiment_runners.py::run_phase_b`) gọi `judge_binary` cho đủ 10 bản ghi trước,
sau đó mới đọc `human_label` để ghép theo `question_id`.

| question_id | model_answer gốc | judge | reasoning (tóm tắt) | human | agree |
|---|---|---|---|---|---|
| 1 | Nghỉ 3 ngày làm việc có lương khi kết hôn. | 1 | Đúng 3 ngày có lương như reference | 1 | yes |
| 5 | Cần Giám đốc phòng ban phê duyệt. | 0 | Sai cấp phê duyệt, reference là Tổng Giám đốc | 0 | yes |
| 12 | Thưởng Tết tối thiểu 1 tháng lương. | 1 | Khớp mức tối thiểu 1 tháng lương | 1 | yes |
| 21 | 18 ngày phép (15 + 3); lương Senior 20–35 triệu. | 1 | Khớp số ngày và khung lương | 1 | yes |
| 23 | Hoàn trả 25 triệu vì chưa đủ 1 năm cam kết. | 1 | Đúng nghĩa vụ hoàn trả 25 triệu | 1 | yes |
| 29 | Trưởng phòng phê duyệt. Phạt 2% tháng trên 8 triệu. | 0 | Thiếu Kế toán trưởng, không nêu đúng mức phạt 80.000 VNĐ | 0 | yes |
| 33 | Phép 19 ngày; phụ cấp 1.500.000 VNĐ/tháng. | 1 | Khớp 19 ngày và phụ cấp | 1 | yes |
| 41 | Nhân viên được nghỉ 12 ngày phép năm. | 0 | Dùng v2023 đã thay thế; hiện hành là 15 ngày | 0 | yes |
| 46 | Thử việc không được nghỉ phép năm, xin nghỉ không lương. | 1 | Giữ đúng phủ định và điều kiện | 1 | yes |
| 50 | Được, miễn là đảm bảo kết nối an toàn. | 0 | Đảo phủ định: chính sách cấm VPN cá nhân | 0 | yes |

Cohen’s κ: 1.0.
Observed/chance agreement và diễn giải: p_o = 10/10 = 1.0. Judge và human cùng có 6 nhãn 1 và
4 nhãn 0, nên p_e = 0.6 × 0.6 + 0.4 × 0.4 = 0.52. κ = (1.0 − 0.52) / (1 − 0.52) = 1.0
(“almost perfect” theo Landis–Koch), vượt ngưỡng bonus κ > 0.6. Kết quả trùng với giá trị
`binary_kappa` mà checker tính lại từ hai list nhãn.

## Giới hạn và cách dùng judge

- Mẫu rất nhỏ: 5 cặp pairwise và 10 nhãn nhị phân. κ = 1.0 trên 10 câu không chứng minh judge
  luôn đồng thuận với người; một bất đồng duy nhất đã làm κ giảm mạnh. Các lỗi trong bộ 10 câu
  (sai cấp phê duyệt, dùng bản cũ, đảo phủ định) khá rõ ràng nên dễ chấm hơn các lỗi tinh vi.
- Judge cùng model với generation (`openai/gpt-oss-20b`) và chạy với `reasoning_effort` mặc định;
  nên kiểm tra lại bằng một judge khác họ model trước khi dùng cho CI gate.
- Pairwise cho thấy judge hiểu sai tiếng Việt ở id 1 (“có lương”); mỗi case pass1/pass2 bất đồng
  cần được người đọc lại reasoning trước khi kết luận.
- Winner A/B chỉ là so sánh tương đối giữa hai candidates, không được đổi thành nhãn đúng/sai;
  id 5 cho thấy cả hai candidates có thể cùng sai (tie với scores 0.0).
- Cách dùng đề xuất: giữ swap-and-average; coi judge là tín hiệu sàng lọc, lấy mẫu định kỳ để
  người gán nhãn lại và tính lại κ khi đổi model/prompt.
