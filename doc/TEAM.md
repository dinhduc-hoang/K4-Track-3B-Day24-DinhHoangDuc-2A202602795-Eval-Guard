# Nhóm thực hiện K4 Track 3B

Tên nhóm: DinhHoangDuc (nhóm một người)

Repo chung: https://github.com/dinhduc-hoang/K4-Track-3B-Day24-DinhHoangDuc-2A202602795-Eval-Guard

Trưởng nhóm/người nộp: Đinh Hoàng Đức — 2A202602795

Ngày lab/deadline áp dụng: lab 12/10/2026, deadline mặc định 11:59 ngày 13/10/2026 theo codelab — Asia/Ho_Chi_Minh (UTC+07:00)

Commit dùng để chấm: `911c719b137955c83c7dbf5c6e49ccd32cc899d5` — bản đầy đủ source, artifacts A/B/C, báo cáo và log kiểm chứng. Commit kế tiếp chỉ cập nhật dòng SHA này trong TEAM.

Mỗi nhóm nộp một repo. Nhóm có một thành viên nên bảng dưới và mục INDIVIDUAL chỉ có một người.

## Thành viên và phân công

| STT | Họ và tên | MSSV | Vai trò / công việc thật | Bằng chứng |
|---|---|---|---|---|
| 1 | Đinh Hoàng Đức | 2A202602795 | Toàn bộ: tích hợp m1–m5, Tasks 1–12 trong `src/phase_*.py`, chạy batch A/B/C, ba báo cáo và Task 13 | Commit nộp, `artifacts/*.json`, `doc/reports/*.md` |

Người tích hợp/chạy batch/nộp link: Đinh Hoàng Đức — đưa module Day 18 vào `src/`, viết adapter, chạy `setup_answers.py --live` và ba phase `--live`, chạy checker và nộp link.

## Môi trường và chuẩn bị trước lớp

OS/Python/compiler: Windows 11 + WSL2 Ubuntu 24.04.4 LTS (kernel 6.6.87.2-microsoft-standard-WSL2), CPU 8 nhân, RAM 7.6 GB, không GPU. Python **3.12.3** trong `.venv` của WSL, khác bản 3.11 mà ENVIRONMENT.md yêu cầu; NeMo 0.10.0, RAGAS 0.1.22, Presidio 2.2.364 và spaCy 3.8.16 vẫn cài và chạy thật trên 3.12 (bằng chứng: các lần chạy live bên dưới). Annoy 1.17.3 build bằng toolchain gcc của Ubuntu.
Virtualenv/dependencies/imports: `pip install -r requirements/requirements.txt -c requirements/constraints-core.txt` → `pip check`: “No broken requirements found”. `python -c "import ragas, nemoguardrails, presidio_analyzer, spacy"` → “imports OK 0.1.22 0.10.0 3.8.16”. `python -m spacy download en_core_web_lg` → en-core-web-lg 3.8.0. Lỗi thực tế gặp: pip kéo `pyarrow 26` và `pandas 3.0.6` (đòi NumPy ≥ 2) trong khi constraints pin `numpy==1.26.4`; `datasets`/`sentence-transformers` không import được nên lần setup đầu tiên âm thầm rơi về fallback (hashing embedding, lexical rerank). Lần chạy đó bị dừng và không dùng; đã pin `pyarrow>=15,<18`, `pandas>=2.2,<3` trong `requirements/constraints-core.txt` (pyarrow 17.0.0, pandas 2.3.3).
Models/Qdrant/50 answers đã chuẩn bị: Qdrant `qdrant/qdrant:v1.9.7` qua `docker compose -f infra/docker-compose.yml up -d` (Docker 29.1.3 trong WSL). BGE-M3 và `BAAI/bge-reranker-v2-m3` từ cache HuggingFace. `python scripts/setup_answers.py --live`: 26 tài liệu, 102 child chunks, index 253.3s, 50 câu trong 585s; log không có dòng fallback nào. Cả 50 answers đều `generation_mode = llm`, `generation_model = openai/gpt-oss-20b` (Groq). Giới hạn: `BCTC.pdf` và Nghị định 13/2023 là PDF ảnh scan, m1 bỏ qua vì không có text layer; M5 enrichment chạy chế độ fallback cục bộ (không gọi LLM, `GROQ_ENRICHMENT_ENABLED=false`), chỉ prepend một câu ngữ cảnh có tên file nguồn.

Provider LLM: Groq API tương thích OpenAI; `config.py` ánh xạ `GROQ_API_KEY` sang `OPENAI_API_KEY`/`OPENAI_BASE_URL`/`OPENAI_API_BASE`. Generation, judge và NeMo guard dùng `openai/gpt-oss-20b`; RAGAS evaluator thực tế: `BAAI/bge-m3`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. Key nằm trong `.env` (đã gitignore), không có trong source/report.

Sự cố quota và cách xử lý (theo thứ tự thực tế):
1. Key Groq đầu tiên hết quota giữa lúc chạy Phase A; lần chạy đó bị dừng, không ghi artifact. Answers, B/C và checkpoint Phase A ban đầu dùng key thứ hai (header đã quan sát: 1000 requests/ngày, 8000 tokens/phút); Phase A sau đó tiếp tục và dùng thêm key do người dùng cập nhật như mục 5.
2. Hai lần chạy Phase A tiếp theo với evaluator `gpt-oss-20b` chậm dần rồi mọi job timeout. Thông báo lỗi 429 (đọc trực tiếp từ response) cho thấy nguyên nhân: “tokens per day (TPD): Limit 200000, Used 199397” — key free giới hạn 200K token/ngày **cho mỗi model**, và setup + Phase B + Phase C đã dùng gần hết quota của gpt-oss-20b. Repo chưa lưu tổng token usage nên không báo cáo lượng token hoặc chi phí dự kiến cho Phase A. Header `x-ratelimit-*` không báo TPD, nên ban đầu tôi đã chẩn đoán nhầm là kết nối chết sau khi máy sleep. Hai lần chạy này không ghi artifact.
3. Đổi evaluator sang `qwen/qwen3.8-27b` (quota ngày riêng, không sinh token suy luận). Trước khi chạy đủ 50 câu, kiểm tra trên 1 câu: đủ 4 metric hữu hạn, khoảng 25s/câu. Đồng thời m4 chấm theo lô 5 câu có checkpoint (`.cache/ragas_rows.json`, gitignored, khóa theo hash input + model evaluator) để lần chạy bị ngắt có thể tiếp tục; HTTP client của evaluator có timeout 90s.
4. Qwen cũng hết 200K token/ngày sau 18/50 câu (id 1–20 trừ 6 và 12 thiếu faithfulness). Key Groq thứ ba thuộc cùng tổ chức nên dùng chung quota. m4 hỗ trợ chuyển riêng evaluator sang OpenRouter (cùng model ID, `OPENROUTER_API_KEY`, `max_tokens` 2048, tắt reasoning của Qwen), nhưng tài khoản OpenRouter có 0 credit.

5. Tiếp tục bằng Codex sau khi xác nhận gửi dữ liệu lab tới Groq. Qwen hết quota/ngày;
gpt-oss-120b chấm thêm đến đủ 42 câu rồi cũng hết quota (bằng chứng lỗi trong
`artifacts/verification/quota_probe_gpt120b_previous_key.json`). Groq báo 404 cho
Llama 70B trên key này nên không dùng model đó để tạo điểm. Hai câu 43/44 được chấm
bằng gpt-oss-20b; sau khi người dùng cập nhật `.env`, sáu câu cuối chấm bằng gpt-oss-120b.
Không lưu key trong artifact/log và không thay answer/reference/fixtures.
6. Đọc source RAGAS 0.1.22 thấy bộ lọc statement bỏ đoạn không kết thúc bằng dấu chấm.
Adapter giữ nguyên cụm từ khi bộ lọc làm danh sách trống; vẫn gọi LLM trích statement
và NLI, không gán NaN thành số. Protocol mới được ghi theo từng ô faithfulness.
Diagnosis Task 3 cũng được sửa để không mặc định quy câu abstain thành thiếu retrieval.

**Trạng thái Phase A: hoàn thành** — `artifacts/ragas_50q.json` có đủ 50 câu với bốn
metric hữu hạn, đúng phân bố 20/20/10; xuất artifact lúc 2026-10-12T16:27:12.813983+00:00 (UTC).
Models evaluator: `BAAI/bge-m3`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. Các ô dự phòng: openai/gpt-oss-120b: 114 ô, openai/gpt-oss-20b: 8 ô; các ô Qwen được giữ từ
checkpoint gốc. Mean tổng thể 0.7552; nhóm yếu nhất multi_hop.
Chi phí/tổng token batch chưa được lưu đầy đủ nên không báo cáo số ước tính.
Giới hạn: kết hợp evaluator và protocol, có evaluator cùng model với generation/judge;
không dùng chênh lệch distribution làm kết luận nguyên nhân riêng của pipeline.
Log chạy, checkpoint gốc và kiểm chứng được giữ trong `artifacts/verification/`,
`artifacts/previous_runs/`, `artifacts/evidence_audit.json`.

Lệnh resume đã dùng trong WSL (biến môi trường của tiến trình, không chứa secret):

```bash
RAGAS_EVALUATOR_BASE_URL=https://api.groq.com/openai/v1 RAGAS_SKIP_PRIMARY=true \
RAGAS_BATCH_SIZE=1 RAGAS_ADDITIONAL_FALLBACK_MODELS=openai/gpt-oss-20b \
RAGAS_CHECKPOINT_ONLY_MODELS=qwen/qwen3.8-27b \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/run_experiments.py --phase a --live
```


## INDIVIDUAL

### DinhHoangDuc-2A202602795

- Phần được giao và phần thực tế đã làm: toàn bộ bài.
  Tích hợp: chép m1–m5 từ repo Day 18 của chính mình (K4-Track3B-DAY18-DinhHoangDuc-2A202602795-ProductionRAG) và viết adapter: m2 dùng `client.search()` khi qdrant-client 1.9 chưa có `query_points()`, cảnh báo khi rơi về Qdrant in-memory; m4 đổi evaluator sang `qwen/qwen3.8-27b` + BGE-M3, trả `metadata.evaluator_models`, báo lỗi thay vì trả list rỗng, chấm theo lô có checkpoint và chỉ chấm lại các ô NaN; m5 đọc cấu hình Groq từ env.
  Phase A (Tasks 1–4): `group_by_distribution`, `run_ragas_50q` (đọc metric dạng object/mapping, từ chối NaN/ngoài [0,1], kiểm tra số dòng), `bottom_10` với `diagnose()` đọc nội dung (abstain, bẫy phiên bản, bẫy phủ định, multi-hop), `cluster_analysis` chọn nhóm yếu theo mean.
  Phase B (Tasks 5–8): prompt pairwise đưa candidates dạng JSON data, kiểm tra JSON trả về; swap-and-average quy lượt 2 về A/B gốc; κ tính tay; bias report có counts và diễn giải.
  Phase C (Tasks 9a–12): `pii_scan` giới hạn ở định danh có cấu trúc; input/output rail đọc `activated_rails[].stop`; batch 28 inputs trong một event loop; đo latency với 2 warm-up thật.
  Hạ tầng guard: `guardrails/config.py` đăng ký ChatOpenAI làm engine `openai_chat` (NeMo 0.10 gửi model không phải gpt-3.5/gpt-4 tới `/completions`, Groq không có endpoint này) và `max_tokens: 512` cho self-check (mặc định 3 token bị phần suy luận của gpt-oss dùng hết, trả rỗng và bị parse thành “unsafe”).
- Bằng chứng: commit nộp; `artifacts/answers_50q.json`, `judge_results.json`, `guard_results.json` (`ragas_50q.json` đã đủ 50 câu); lần chạy C đầu tiên giữ ở `artifacts/previous_runs/guard_results_run1_presidio_ner.json`; `doc/reports/failure_clusters.md`, `bias_report.md`, `blueprint.md`.
- Cách kiểm tra: `python scripts/check_lab.py --with-tests` → 98 passed, 3 deselected (integration), checker 0 failed checks, đối chiếu đủ artifacts A/B/C; log và phiên bản môi trường lưu tại `artifacts/verification/offline_check.json`. Live đã lưu: `setup_answers.py --live`, `run_experiments.py --phase c --live --latency-samples 20` (17/20 attacks, clean 8/8, output 4/4), `--phase b --live` (κ = 1.0), `--phase a --live` (tiếp tục checkpoint và hoàn tất 50/50 câu). `failure_clusters.md` có mean/matrix/bottom 10 và phân tích nội dung từ artifact đủ 50 câu.
- Điều đã học: matrix worst_metric chỉ mô tả, nhóm yếu phải chọn theo mean; winner A/B khác nhãn 0/1; position bias phải được kiểm tra bằng cách quy lượt swap về danh tính gốc; clean controls bắt được một guard “gần như luôn chặn” mà bộ attacks 20/20 không phát hiện; latency phải được đọc cùng điều kiện quota của provider.
- Giới hạn/cách xử lý: (1) Lần chạy C đầu: Presidio dùng NER tiếng Anh gắn PERSON cho từ tiếng Việt (“Hãy”, “bao nhiêu”, “Xin lỗi”) → chặn nhầm 7/8 câu sạch (FP 0.875) dù attacks 20/20; đã giới hạn Presidio ở VN_CCCD, VN_PHONE, EMAIL_ADDRESS, PHONE_NUMBER và chạy lại một lần, không đổi expected/fixtures. (2) Latency lần 2 bị cộng thời gian chờ 429 do quota 8000 tokens/phút; giữ nguyên số và giải thích trong blueprint. (2b) Quota/ngày làm Phase A phải kết hợp evaluator; tám ô dùng gpt-oss-20b cùng model với generation/judge/guard, ghi rõ trong metadata và báo cáo. (3) 3 attack off-topic lọt qua NeMo (chưa đạt bonus 18/20), ghi nhận trong blueprint, không chỉnh policy rồi chạy lại để lấy bonus. (4) Python 3.12 thay vì 3.11.
- Nguồn/AI hỗ trợ nếu có: Claude Code (Anthropic) hỗ trợ viết code Tasks 1–12, adapter m1–m5, cấu hình Groq/NeMo, chạy các lệnh live và soạn báo cáo. Codex hỗ trợ kiểm tra tiếp bài làm, viết `scripts/audit_evidence.py`, lưu log kiểm chứng và sửa diễn giải báo cáo; tiếp tục batch Phase A qua Groq sau xác nhận, sửa xử lý answer dạng cụm từ và diagnosis, lưu attribution từng ô dự phòng. Kiểm chứng: đọc từng diff, chạy unit tests, đối chiếu mọi số trong báo cáo với JSON trong `artifacts/`, đọc log setup để loại lần chạy dùng fallback, và chạy checker. Mã m1–m5 là bài Day 18 của chính tôi.

## Cập nhật sau khi nộp

Không có cập nhật sau khi nộp tại thời điểm viết; nếu có, sẽ ghi commit, thời điểm và lý do tại đây.
