# Phân tích lỗi Phase A

Nhóm: DinhHoangDuc (nhóm một người)

Người phụ trách: Đinh Hoàng Đức — 2A202602795

> **Trạng thái: Phase A CHƯA CHẠY XONG — chưa có `artifacts/ragas_50q.json`.**
> Báo cáo này chưa có số đo RAGAS. Không có điểm nào trong file này được ước lượng hay
> lấy từ một lần chạy dở dang. Các bảng sẽ được điền từ JSON sau lần chạy hoàn chỉnh.

Nguồn: `artifacts/ragas_50q.json` — chưa tồn tại; run timestamp: chưa có.
Generation mode/fallback và điều kiện chạy: 50/50 answers `generation_mode = llm`,
`generation_model = openai/gpt-oss-20b` (Groq); retrieval BM25 + BGE-M3/Qdrant, rerank
bge-reranker-v2-m3, M5 enrichment ở chế độ fallback cục bộ; log setup không có dòng fallback nào.
Models generation/evaluator: generation `openai/gpt-oss-20b`; evaluator dự kiến `qwen/qwen3.8-27b`
+ embedding `BAAI/bge-m3` (sẽ lấy từ `metadata.evaluator_models` do m4 trả về).

## Vì sao chưa có kết quả

- RAGAS 0.1.22 với 4 metric tốn khoảng 10K token/câu (prompt few-shot, context_precision gọi
  LLM một lần cho mỗi context). 50 câu cần khoảng 500K token.
- Key Groq free giới hạn **200K token/ngày cho mỗi model** (thông báo 429: “tokens per day (TPD):
  Limit 200000, Used 199397”). Quota của `gpt-oss-20b` đã dùng cho setup, Phase B, Phase C; quota của
  `qwen/qwen3.8-27b` hết sau 18/50 câu. Key thứ hai thuộc cùng tổ chức Groq nên dùng chung quota.
  OpenRouter có cùng model nhưng tài khoản 0 credit.
- 18 câu (id 1–5, 7–11, 13–20) đã được chấm đủ 4 metric và lưu ở checkpoint `.cache/ragas_rows.json`
  (không commit); id 6 và 12 thiếu faithfulness. Lần chạy tiếp theo dùng lại các ô này, chỉ chấm phần
  còn thiếu, rồi runner mới xuất `ragas_50q.json`. 18 câu này **không** được dùng làm số đại diện vì
  chưa đủ 20/20/10 và chỉ thuộc nhóm factual.

## Điểm theo distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
|---|---|---|---|
| faithfulness | chưa đo | chưa đo | chưa đo |
| answer_relevancy | chưa đo | chưa đo | chưa đo |
| context_precision | chưa đo | chưa đo | chưa đo |
| context_recall | chưa đo | chưa đo | chưa đo |
| avg_score | chưa đo | chưa đo | chưa đo |

## Bottom 10 và nguyên nhân

Chưa có — bottom 10 do Task 3 (`bottom_10` + `diagnose`) tạo từ 50 dòng RAGAS và được exporter giữ
nguyên. Sẽ điền đúng 10 rows sau lần chạy hoàn chỉnh.

Quan sát định tính từ `artifacts/answers_50q.json` (đọc trực tiếp answer/contexts/reference,
**không phải điểm RAGAS**), dùng làm giả thuyết cần kiểm tra khi có số:

- id 23 (multi_hop): cả 3 contexts đều từ `hoan_chi_dao_tao.md` nhưng answer là “Không tìm thấy.” —
  nghi lỗi generation hoặc chunk lấy được không chứa điều khoản cam kết 1 năm.
- id 33 (multi_hop): contexts chỉ từ `nghi_phep_nam_v2024/v2023.md`, không có `phu_cap.md` →
  retrieval thiếu nguồn thứ hai; answer “Không tìm thấy.”
- id 21 (multi_hop): đúng 18 ngày phép nhưng không có `bang_luong_2024.md` trong contexts nên nói
  không xác định được khung lương — thiếu đoạn thứ hai của câu nhiều bước.
- id 6 và 12: answer cụt (“**Bí mật**”, “1 tháng lương”); evaluator không trích được câu khẳng định
  nên faithfulness ra NaN — m4 sẽ chấm lại ô này bằng evaluator dự phòng đã khai báo, không điền điểm.

## Matrix và nhóm yếu nhất

Chưa có. Matrix worst_metric × distribution và nhóm yếu theo mean thấp nhất sẽ lấy từ
`failure_clusters` trong JSON (Task 4 chọn theo `avg_score` trung bình, không theo tổng cột 20/20/10).

Nhóm yếu theo mean thấp nhất: chưa đo; mean: chưa đo.
Worst metric xuất hiện nhiều nhất: chưa đo; số câu: chưa đo.

## Bẫy adversarial và giới hạn

Quan sát định tính trên 10 câu adversarial (chưa có điểm RAGAS):

- Bẫy phiên bản (41, 42, 43, 44, 49): contexts lấy cả bản cũ lẫn mới (ví dụ id 44 có 2/3 contexts từ
  `mat_khau_v1.md`), nhưng cả 5 answers đều nêu giá trị hiện hành (15 ngày, 3 năm, 12 ký tự,
  120 ngày) và id 49 phân biệt đúng v2023/v2024. Context precision của nhóm này dự kiến bị kéo xuống
  bởi đoạn bản cũ dù answer đúng — cần số đo để xác nhận.
- Bẫy phủ định (46, 47, 48, 50): cả 4 answers giữ phủ định “Không”. id 48 đúng nhưng thiếu ý “chỉ
  tham gia BHXH bắt buộc”.
- Giới hạn: chưa có số; khi có, adversarial mean thấp hơn factual chỉ cho thấy nhóm này khó hơn với
  metric, không tự chứng minh pipeline nhận ra version conflict. Evaluator khác model với generation.
