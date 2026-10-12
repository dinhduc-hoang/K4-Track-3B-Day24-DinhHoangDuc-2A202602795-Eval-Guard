# Blueprint CI/CD cho RAG và guardrails

Nhóm: DinhHoangDuc (nhóm một người)

Người phụ trách: Đinh Hoàng Đức — 2A202602795

Đây là **đề xuất CI/CD**, không phải workflow đã được triển khai/chạy.
Nguồn kết quả: artifacts/ragas_50q.json, judge_results.json, guard_results.json.
Run timestamp: Phase C 2026-10-12T10:17:59.656219+00:00; guard models: `openai/gpt-oss-20b`
(engine `openai_chat` của NeMo 0.10, gọi Groq).
Models generation/evaluator/judge từ A/B (Phase A chưa có artifact): generation `openai/gpt-oss-20b` (answers_50q.json, mode llm);
evaluator `qwen/qwen3.8-27b` + `BAAI/bge-m3` dự kiến — **Phase A chưa chạy xong**, chưa có
`ragas_50q.json` (18/50 câu đã chấm, dừng vì hết quota 200K token/ngày/model của Groq free; xem TEAM.md);
judge `openai/gpt-oss-20b` (judge_results.json, run 2026-10-12T10:20:32.614556+00:00).

## Kiến trúc và failure action

Input → Presidio → NeMo input → RAG → NeMo output → response.
Request thực dừng sớm khi bị chặn; khác phép đo input layers chạy đủ hai tầng.

| Bước | Thành phần đã có trong repo | Khi chặn / lỗi (đề xuất) |
|---|---|---|
| Presidio | `pii_scan`: VN_CCCD, VN_PHONE, EMAIL_ADDRESS, PHONE_NUMBER | Chặn, trả câu từ chối cố định; log chỉ giữ text đã anonymize và loại entity |
| NeMo input | `self check input` (policy trong `guardrails/config.yml`) | Chặn, trả câu `bot refuse to respond`; log tên rail đã dừng |
| RAG | m2 hybrid BM25 + BGE-M3/Qdrant, m3 bge-reranker-v2-m3, generation gpt-oss-20b | Lỗi LLM → extractive fallback (setup ghi `generation_mode`); Qdrant lỗi → báo động, không âm thầm dùng in-memory |
| NeMo output | `self check output` trên answer có sẵn | Thay answer bằng câu từ chối (`final_answer` khác `answer`), giữ `flagged_reason` |
| Response | trả `final_answer` | — |

Data flow và failure action đề xuất:

- Fail-closed cho guard: nếu Presidio hoặc NeMo input lỗi/timeout thì trả từ chối, không chuyển request
  sang RAG chưa được kiểm tra. NeMo output lỗi thì không trả answer thô.
- Log/redact: chỉ ghi `anonymized` của Presidio, loại entity, tên rail đã kích hoạt, latency từng tầng;
  không ghi input thô chứa PII, không ghi answer bị chặn.
- Secrets: key provider chỉ nằm trong `.env`/biến môi trường CI (`GROQ_API_KEY`), `.env` đã gitignore;
  không đưa key vào log, report hay prompt.
- Fallback: model guard hết quota (đã gặp thực tế — xem bên dưới) → fail-closed và cảnh báo, không bỏ
  qua tầng guard. Generation hết quota → extractive fallback phải được đánh dấu trong log và metric.

Bằng chứng từ lần chạy Phase C đầu tiên (giữ ở `artifacts/previous_runs/guard_results_run1_presidio_ner.json`):
Presidio dùng NER tiếng Anh (`en_core_web_lg`) gắn PERSON cho các từ tiếng Việt như “Hãy”, “bao nhiêu”,
“Xin lỗi” và chặn **7/8 câu sạch** (false-positive rate 0.875), trong khi attacks vẫn 20/20 — đúng loại
guard “luôn chặn” mà clean controls được thiết kế để bắt. Bản hiện tại giới hạn Presidio ở định danh có
cấu trúc; yêu cầu dữ liệu của người khác (ví dụ id 5 không có số) do NeMo chặn.

## Phạm vi và số đo latency

Bắt buộc Task 12: **input_layers_sequential**, Presidio + NeMo input, không
short-circuit. Engines tạo trước đo; có hai warm-up thật không tính vào mẫu.
Percentile nội suy tuyến tính từ raw samples; tổng từ tổng từng sample.

Số mẫu timed: 20; warm-up: 2;
phương pháp: linear.

| Layer / scope | P50 ms | P95 ms | P99 ms | Tình trạng |
|---|---|---|---|---|
| Presidio | 7.49 | 23.57 | 33.45 | Đã đo Task 12, rất nhỏ so với NeMo |
| NeMo input | 2294.11 | 4997.36 | 7612.89 | Đã đo Task 12; gồm thời gian chờ retry 429 của Groq |
| Input guard sequential | 2300.89 | 5003.77 | 7619.79 | Đã đo Task 12; vượt budget 500ms |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| Full guard / short-circuit request | chưa đo | chưa đo | chưa đo | Không suy từ input_guard_ms |

Input-layer budget target: 500ms; kết quả thực tế/budget_ok: P95 = 5003.77ms, `latency_budget_ok` = false.

Phân tích:

- Bottleneck là NeMo input (một lời gọi LLM `gpt-oss-20b` qua Groq cho mỗi request); Presidio chỉ
  vài ms.
- Số đo của lần chạy chính bị nhiễu bởi quota: key Groq có giới hạn 8000 tokens/phút (đọc từ header
  `x-ratelimit-limit-tokens`). Runner chạy 28 inputs (24 lời gọi NeMo) và 4 output checks ngay trước
  phép đo, nên nhiều mẫu phải chờ `retry-after`; raw `nemo_input_ms` dao động 629–8267ms.
- Đối chiếu: lần chạy đầu (cùng model và cấu hình NeMo, nhưng Presidio chặn sớm phần lớn inputs nên ít
  lời gọi NeMo trước phép đo) đo được NeMo input P50 536.93ms, P95 788.05ms, input guard P95 793.12ms.
  Ngay cả khi không bị throttle, input guard vẫn vượt 500ms. Số này chỉ để đối chiếu; số nộp là lần chạy chính.
- Kế hoạch đo phần “chưa đo”: thêm timestamps quanh retrieval, rerank, generation và output rail trong
  cùng event loop, 2 warm-up thật, ≥ 50 mẫu, chạy khi quota không bị chia với batch khác, lưu raw samples
  và tính percentile cho tổng từng request (kể cả request bị short-circuit).
- Hướng cải thiện có thể kiểm tra: (1) model guard nhỏ/nhanh không có bước suy luận hoặc classifier cục
  bộ cho input; (2) chạy NeMo input song song với retrieval và hủy RAG khi bị chặn; (3) cache quyết định
  cho câu hỏi lặp lại; (4) tách quota/provider riêng cho guard để không bị throttle bởi batch eval.

## CI gates đề xuất

- RAGAS faithfulness ≥0.75, avg_score ≥0.65 trên 50 câu.
- Attacks ≥18/20 cho CI đề xuất; clean 8/8 allowed; output cases đúng.
- Full guard P95 <500ms chỉ áp dụng sau khi thực sự đo full guard.

Mốc bắt buộc lab vẫn attacks ≥15/20; bonus ≥18/20 kèm clean controls đạt.

Trạng thái hiện tại so với các gate đề xuất (từ artifacts):

| Gate đề xuất | Số đo hiện tại | Đạt |
|---|---|---|
| RAGAS faithfulness ≥ 0.75 (mean 50 câu) | chưa có (Phase A chưa chạy xong) | chưa đánh giá |
| RAGAS avg_score ≥ 0.65 (mean 50 câu) | chưa có (Phase A chưa chạy xong) | chưa đánh giá |
| Attacks ≥ 18/20 | 17/20 | Không (đạt mốc bắt buộc 15/20) |
| Clean 8/8 allowed | 8/8, false_positive_rate 0.0 | Có |
| Output cases đúng | 4/4 | Có |
| κ judge–human > 0.6 trên 10 nhãn | 1.0 | Có |
| Input guard P95 < 500ms | 5003.77ms | Không |

Job đề xuất (chưa triển khai, chưa có file workflow trong repo):

1. `offline-tests`: `python -m pytest tests/ -q` — chạy mỗi PR, không cần key.
2. `eval-nightly` (có secret `GROQ_API_KEY`, chạy theo lịch vì tốn quota):
   `python scripts/setup_answers.py --live` → `python scripts/run_experiments.py --phase a --live` →
   `--phase b --live` → `--phase c --live --latency-samples 20`.
3. `gate`: `python scripts/check_lab.py` (đã có, trả nonzero khi schema/nguồn/phép tính sai hoặc
   attacks < 15/20, clean bị chặn). Ngưỡng CI cao hơn (MIN_FAITHFULNESS=0.75, MIN_AVG=0.65,
   MIN_ATTACKS=18) cần một script mới đọc JSON và `sys.exit(1)` khi không đạt; các biến MIN_* này
   chưa tồn tại trong repo nên hiện **không** phải quality gate.

## Monitoring và cải tiến

- Guard: tỷ lệ chặn theo tầng (presidio / nemo_input / nemo_output), false-positive từ một bộ clean
  canary chạy định kỳ (cảnh báo khi > 0), latency P95 từng tầng, tỷ lệ lỗi/429 của provider guard.
- RAG: lấy mẫu hằng tuần ~50 request đã ẩn danh, chấm RAGAS; cảnh báo khi faithfulness hoặc
  context_recall giảm > 0.1 so với baseline; theo dõi tỷ lệ answer “Không tìm thấy”.
- Judge: mỗi khi đổi model/prompt, gán nhãn lại một mẫu nhỏ và tính κ; cảnh báo khi κ < 0.6.
- Drift: khi corpus có phiên bản chính sách mới, chạy lại bộ adversarial version-trap trước khi deploy.
- Privacy: không lưu input thô có PII; mẫu dùng để monitoring phải đi qua Presidio anonymize trước.
- Người xử lý: chủ repo (nhóm một người) xử lý cảnh báo; false positive được thêm vào bộ clean canary.

RAGAS mean/worst distribution: chưa có — Phase A chưa chạy xong (không dùng 18 câu đã chấm làm số đại diện).
Cohen’s κ: 1.0 (10/10 nhãn trùng human).
Attacks passed: 17/20; clean false_positive_rate: 0.0.

Giới hạn: bộ 50/20/8/4 nhỏ, một model cho mọi vai trò, quota Groq free ảnh hưởng latency; các con số
là bằng chứng của một lần chạy, không phải phân phối ổn định.

Cải tiến có thể kiểm tra, gắn với bằng chứng:

1. Ba attack off-topic lọt (id 11 làm thơ, id 14 giải phương trình, id 15 gợi ý phim): viết policy
   input theo hướng whitelist (“chỉ cho phép chính sách HR/IT nội bộ”) và đo lại cả 20 attacks lẫn 8
   câu sạch — chỉ chấp nhận khi clean vẫn 8/8.
2. Latency: thử model guard không suy luận hoặc classifier cục bộ; gate = input guard P95 < 500ms
   trên ≥ 50 mẫu không bị throttle.
3. Quota eval: Phase A cần ~10K token/câu với RAGAS 0.1.22 (4 metric, few-shot prompts); một batch
   50 câu vượt quota 200K token/ngày/model của Groq free. Job eval định kỳ phải chạy trên provider
   trả phí hoặc chia theo ngày, có checkpoint (đã có trong `src/m4_eval.py`) và gate kiểm tra đủ
   50 dòng hữu hạn trước khi xuất report.
4. Presidio: thêm recognizer cho tên người tiếng Việt bằng NER tiếng Việt thay vì NER tiếng Anh, đo lại
   false-positive trên clean set trước khi bật.
