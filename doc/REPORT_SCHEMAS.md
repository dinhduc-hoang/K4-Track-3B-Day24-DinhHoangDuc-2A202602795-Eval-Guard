# Runner và schema kết quả

## Lệnh chạy thực nghiệm thật

Chạy từ gốc repo, sau khi hoàn thành TODO, chuẩn bị pipeline/answers và key.
Các lệnh dưới đây dùng API/mạng, có chi phí. CLI yêu cầu `--live`; không có cờ
này thì không chạy experiment. Không dùng kết quả unit/mock làm bài nộp.

```bash
python scripts/run_experiments.py --phase a --live
python scripts/run_experiments.py --phase b --live
python scripts/run_experiments.py --phase c --live --latency-samples 20
```

Các entry point `src/phase_*.py --live` chuyển sang runner tương ứng, không
còn demo rồi bỏ quên việc lưu file. Runner cung cấp orchestration/validation,
không implement các hàm TODO của học viên.

## Schema JSON và đối chiếu số liệu

Schema máy đọc: [Answers](schemas/answers.schema.json), [Phase A](schemas/phase_a.schema.json),
[Phase B](schemas/phase_b.schema.json), [Phase C](schemas/phase_c.schema.json).
Checker và runner dùng `jsonschema` Draft 2020-12 để validate thực sự từng
JSON trước khi đối chiếu id, nguồn, phép tính, số lượng, passed flags, false
positives và placeholder trong `src/report_contracts.py`. Mọi refs là nội bộ;
kiểm tra schema không cần mạng. `requirements/requirements.txt` có dependency này.

Mỗi report có `metadata`: format_version=2, run_kind=live, `models`, created_at
(ISO thời gian UTC có timezone) và SHA-256 các file đầu vào. Không còn trường
`model` chung. `models` có bốn keys `generation`, `evaluator`, `judge`, `guard`;
mỗi key là list model IDs đã sắp xếp, không trùng, `[]` khi vai trò đó không
tham gia phase. Checker so hash để phát hiện
report cũ khi dữ liệu đầu vào đã đổi; hash không tự chứng minh đã gọi API.
`.gitattributes` cố định JSON artifact ở LF và fixtures ở CRLF đúng với bytes
đã dùng trong các lần chạy lưu bằng chứng, để checkout Windows/Linux giữ SHA-256.

- Generation: lấy từ `generation_model` của các answers có mode `llm`. Mỗi answer
  bắt buộc có `generation_mode` thuộc `llm`/`extractive_fallback` và
  `generation_model` là ID không trống với `llm`, hoặc `null` với fallback.
- Phase A evaluator: lấy từ `RagasEvaluation.evaluator_models`, chuyển trực tiếp
  từ `metadata.evaluator_models` của m4 trong chính lượt evaluation. List gồm
  model IDs LLM/embedding đã dùng; thiếu thì fail, không lấy JUDGE_MODEL thay.
- Phase B judge: ghi model IDs cấu hình cho pairwise judge và binary judge.
  Nếu đổi model trong hàm pairwise, phải cập nhật `phase_b_judge.JUDGE_MODEL`
  tương ứng; binary helper dùng model của chính helper. Generation vẫn là nguồn A.
- Phase C guard: lấy LLM IDs từ `config.models` của từng NeMo instance được
  truyền vào input, output và latency, không suy từ model judge. Đây là ID cấu
  hình dùng trong lượt chạy, không phải xác nhận alias provider đã resolve sang
  một snapshot cụ thể. Generation/evaluator/judge không chạy trong runner C.

Schema phiên bản 2 không chấp nhận reports phiên bản 1 với `model` chung.
Đọc model identifier là thông tin provenance do code ghi lại; nó không tự
chứng minh API đã chạy hoặc người làm đã khai báo trung thực.

| File | Khối nội dung |
|---|---|
| `artifacts/ragas_50q.json` | 50 per_question, per_distribution 20/20/10, failure_clusters, 10 bottom_10 từ đúng Task 3 |
| `artifacts/judge_results.json` | pair_sources; 5 pairwise_results; 10 binary_judgments; calibration và bias_report |
| `artifacts/guard_results.json` | adversarial 20; clean 8 với false_positive_rate; output_results 4; latency với raw samples |

Phase A exporter nhận trực tiếp `bottom_10(results)`. Diagnosis/suggested_fix
do học viên viết được giữ nguyên, không bị exporter tự tính lại thành mặc định.
Matrix vẫn thống kê worst_metric của mọi câu; nhóm yếu nhất được chọn bằng
avg_score trung bình thấp nhất, không bằng tổng 20/20/10.

Adapter m4 lưu checkpoint theo hash của question/answer/contexts/reference và
model evaluator. Khi đã xác nhận primary hết quota, `RAGAS_SKIP_PRIMARY=true`
giữ các ô primary đã đo và chỉ gọi evaluator dự phòng cho các ô thiếu;
`RAGAS_FALLBACK_EVALUATOR_MODEL` khai báo model đó. Không điền điểm thay API.
`RAGAS_BATCH_SIZE=1` ghi checkpoint sau từng câu để giảm mất kết quả khi ngắt.
Nếu nhiều evaluator hết quota, `RAGAS_ADDITIONAL_FALLBACK_MODELS` khai báo thêm
model theo thứ tự cố định; `RAGAS_CHECKPOINT_ONLY_MODELS` giữ điểm của các model
đã hết quota mà không gọi lại. Chỉ chấm ô chưa có điểm từ bất kỳ checkpoint nào;
không chọn model theo điểm cao/thấp. Metadata chỉ liệt kê model có đóng góp vào
điểm được chọn và embedding; từng ô dự phòng giữ model thực tế của nó.
Phase A có thêm `evaluator_details` từ metadata m4: provider, trạng thái
primary checkpoint-only và `fallback_cells` với index theo thứ tự `per_question`,
metric, model, score. Báo cáo phải ghi giới hạn của việc dùng evaluator khác nhau.
Các ô faithfulness có thêm `protocol`: mặc định RAGAS 0.1.22, hoặc
`ragas_0.1.22_retain_fragments_v1` cho adapter giữ nguyên các đoạn không có dấu chấm
khi bộ lọc mặc định làm danh sách trống. Score vẫn do RAGAS/LLM đo; các ô checkpoint
đã có điểm được giữ nguyên và không tự đổi nhãn protocol.
Đối chiếu lại artifacts offline bằng `python scripts/audit_evidence.py`;
`artifacts/evidence_audit.json` chứa summary và hash nguồn, không phải số đo API mới.

Phase B: mỗi binary_judgment giữ question_id, question, **model_answer gốc**,
label và reasoning; calibration giữ cùng thứ tự ids, judge_labels, human_labels,
cohen_kappa. Nhãn human được dùng sau judge cho đối chiếu, không đưa vào prompt.
`pair_sources` bắt buộc đúng ba nguồn A/B/reference trong đề bài. Bias report
cần cả `verbosity_details`; checker đối chiếu từng count với năm cặp gốc.

Phase C runner chạy 20 attack + 8 clean inputs một lần, tách nhóm theo id, chạy
4 output cases và đo latency. Mỗi stage reuse engine trong stage đó; các NeMo
instance riêng tránh reuse HTTP client qua event loop đã đóng. Kết quả tấn công
chưa đạt 15/20 vẫn được lưu để phân tích; checker không cho nộp khi chưa đạt.

## Phạm vi latency Task 12

`scope = input_layers_sequential`, không phải end-to-end guard stack.
Tạo engine trước đo; trong cùng event loop, gọi thật Presidio và NeMo input
**hai lượt warm-up** rồi loại chúng khỏi mẫu. Thu **ít nhất 20 mẫu timed**,
cycling 8 câu sạch, dùng perf_counter và milliseconds. Luôn đo cả hai tầng;
không dừng theo quyết định block trong phép đo này.

Mỗi raw sample có presidio_ms, nemo_input_ms, input_guard_ms=sum hai tầng.
Percentile dùng nội suy tuyến tính tại `(n-1)*q`, với q=.5/.95/.99; tổng P95
tính từ tổng từng mẫu, không cộng P95 riêng. Helper `summarize_latency()` cung
cấp thống kê; Task 12 vẫn phải thu mẫu và thực hiện warm-up thật.

JSON có n_samples, warmup_runs, percentile_method=linear, samples, ba khối
percentile và latency_budget_ok. Zero samples/giá trị toàn 0 không hợp lệ.
Mốc 500ms là target; đo vượt target phải báo cáo đúng, không sửa số cho đạt.

RAG latency, output rail latency và toàn bộ request có short-circuit nằm ngoài
phạm vi bắt buộc Task 12. Trong blueprint được ghi **“chưa đo”** cho các phần
này, nêu kế hoạch đo. Không dùng input_guard_ms để kết luận total guard đạt
500ms. Blueprint là đề xuất CI/CD, không có workflow được tạo tự động.

## Unit tests và integration

`python -m pytest` chọn tests offline: socket bị chặn; SDK, Presidio và NeMo
được mock. Marker `support` kiểm tra phần cung cấp; marker `student` kiểm tra
hàm TODO, nên phải fail trên starter chưa làm. Không cần key hoặc tải model.

Nếu muốn kiểm tra batch B/C thật qua pytest thay cho lệnh runner riêng:

```bash
python -m pytest -m integration --run-live
```

Fixture integration chạy mỗi phase một lần và dùng chung kết quả giữa các test.
Không chạy lệnh này sau runner B/C chỉ để xác nhận lại, vì sẽ phát sinh API calls
lặp lại. Khi đã có artifacts, `python scripts/check_lab.py` kiểm tra offline từ file;
`--with-tests` là tùy chọn gộp unit tests, không bật integration.
