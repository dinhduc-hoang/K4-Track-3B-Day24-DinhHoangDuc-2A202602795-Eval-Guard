# Nhóm thực hiện K4 Track 3B

Tên nhóm: DinhHoangDuc (nhóm một người)

Repo chung: https://github.com/dinhduc-hoang/K4-Track-3B-Day24-DinhHoangDuc-2A202602795-Eval-Guard

Trưởng nhóm/người nộp: Đinh Hoàng Đức — 2A202602795

Ngày lab/deadline áp dụng: lab 12/10/2026, deadline mặc định 11:59 ngày 13/10/2026 theo codelab — Asia/Ho_Chi_Minh (UTC+07:00)

Commit dùng để chấm: chưa chốt — Phase A chưa chạy xong (xem mục môi trường); SHA sẽ ghi sau commit có `artifacts/ragas_50q.json`.

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

Provider LLM: Groq API tương thích OpenAI; `config.py` ánh xạ `GROQ_API_KEY` sang `OPENAI_API_KEY`/`OPENAI_BASE_URL`/`OPENAI_API_BASE`. Generation, judge và NeMo guard dùng `openai/gpt-oss-20b`; RAGAS evaluator dùng `qwen/qwen3.8-27b` (+ embedding `BAAI/bge-m3`). Key nằm trong `.env` (đã gitignore), không có trong source/report.

Sự cố quota và cách xử lý (theo thứ tự thực tế):
1. Key Groq đầu tiên hết quota giữa lúc chạy Phase A; lần chạy đó bị dừng, không ghi artifact. Mọi artifact hiện có dùng key thứ hai (header: 1000 requests/ngày, 8000 tokens/phút).
2. Hai lần chạy Phase A tiếp theo với evaluator `gpt-oss-20b` chậm dần rồi mọi job timeout. Thông báo lỗi 429 (đọc trực tiếp từ response) cho thấy nguyên nhân: “tokens per day (TPD): Limit 200000, Used 199397” — key free giới hạn 200K token/ngày **cho mỗi model**, và setup + Phase B + Phase C đã dùng gần hết quota của gpt-oss-20b. Phase A cần khoảng 300–450K token nên không thể xong với model này trước deadline. Header `x-ratelimit-*` không báo TPD, nên ban đầu tôi đã chẩn đoán nhầm là kết nối chết sau khi máy sleep. Hai lần chạy này không ghi artifact.
3. Đổi evaluator sang `qwen/qwen3.8-27b` (quota ngày riêng, không sinh token suy luận). Trước khi chạy đủ 50 câu, kiểm tra trên 1 câu: đủ 4 metric hữu hạn, khoảng 25s/câu. Đồng thời m4 chấm theo lô 5 câu có checkpoint (`.cache/ragas_rows.json`, gitignored, khóa theo hash input + model evaluator) để lần chạy bị ngắt có thể tiếp tục; HTTP client của evaluator có timeout 90s.
4. Qwen cũng hết 200K token/ngày sau 18/50 câu (id 1–20 trừ 6 và 12 thiếu faithfulness). Key Groq thứ ba thuộc cùng tổ chức nên dùng chung quota. m4 hỗ trợ chuyển riêng evaluator sang OpenRouter (cùng model ID, `OPENROUTER_API_KEY`, `max_tokens` 2048, tắt reasoning của Qwen), nhưng tài khoản OpenRouter có 0 credit.

**Trạng thái Phase A tại thời điểm commit này: chưa hoàn thành, chưa có `artifacts/ragas_50q.json`** — checker báo FAIL cho Phase A artifact và `failure_clusters.md` chỉ ghi trạng thái. Cần khoảng 300K token trả phí (ước 0,15–0,20 USD qua OpenRouter hoặc Groq Dev tier) để chấm 32 câu còn lại; lệnh chạy lại là `python scripts/run_experiments.py --phase a --live` (tự dùng checkpoint).

## INDIVIDUAL

### DinhHoangDuc-2A202602795

- Phần được giao và phần thực tế đã làm: toàn bộ bài.
  Tích hợp: chép m1–m5 từ repo Day 18 của chính mình (K4-Track3B-DAY18-DinhHoangDuc-2A202602795-ProductionRAG) và viết adapter: m2 dùng `client.search()` khi qdrant-client 1.9 chưa có `query_points()`, cảnh báo khi rơi về Qdrant in-memory; m4 đổi evaluator sang `qwen/qwen3.8-27b` + BGE-M3, trả `metadata.evaluator_models`, báo lỗi thay vì trả list rỗng, chấm theo lô có checkpoint và chỉ chấm lại các ô NaN; m5 đọc cấu hình Groq từ env.
  Phase A (Tasks 1–4): `group_by_distribution`, `run_ragas_50q` (đọc metric dạng object/mapping, từ chối NaN/ngoài [0,1], kiểm tra số dòng), `bottom_10` với `diagnose()` đọc nội dung (abstain, bẫy phiên bản, bẫy phủ định, multi-hop), `cluster_analysis` chọn nhóm yếu theo mean.
  Phase B (Tasks 5–8): prompt pairwise đưa candidates dạng JSON data, kiểm tra JSON trả về; swap-and-average quy lượt 2 về A/B gốc; κ tính tay; bias report có counts và diễn giải.
  Phase C (Tasks 9a–12): `pii_scan` giới hạn ở định danh có cấu trúc; input/output rail đọc `activated_rails[].stop`; batch 28 inputs trong một event loop; đo latency với 2 warm-up thật.
  Hạ tầng guard: `guardrails/config.py` đăng ký ChatOpenAI làm engine `openai_chat` (NeMo 0.10 gửi model không phải gpt-3.5/gpt-4 tới `/completions`, Groq không có endpoint này) và `max_tokens: 512` cho self-check (mặc định 3 token bị phần suy luận của gpt-oss dùng hết, trả rỗng và bị parse thành “unsafe”).
- Bằng chứng: commit nộp; `artifacts/answers_50q.json`, `judge_results.json`, `guard_results.json` (`ragas_50q.json` chưa có — Phase A chưa chạy xong); lần chạy C đầu tiên giữ ở `artifacts/previous_runs/guard_results_run1_presidio_ner.json`; `doc/reports/failure_clusters.md`, `bias_report.md`, `blueprint.md`.
- Cách kiểm tra: unit mock `python -m pytest tests/ -q` → 86 passed, 3 deselected (integration). Live: `setup_answers.py --live`, `run_experiments.py --phase c --live --latency-samples 20` (17/20 attacks, clean 8/8, output 4/4), `--phase b --live` (κ = 1.0), `--phase a --live` (dừng ở 18/50 câu vì hết quota, chưa xuất artifact); `python scripts/check_lab.py` → mọi mục PASS trừ Phase A artifact và `failure_clusters.md`.
- Điều đã học: matrix worst_metric chỉ mô tả, nhóm yếu phải chọn theo mean; winner A/B khác nhãn 0/1; position bias phải được kiểm tra bằng cách quy lượt swap về danh tính gốc; clean controls bắt được một guard “gần như luôn chặn” mà bộ attacks 20/20 không phát hiện; latency phải được đọc cùng điều kiện quota của provider.
- Giới hạn/cách xử lý: (1) Lần chạy C đầu: Presidio dùng NER tiếng Anh gắn PERSON cho từ tiếng Việt (“Hãy”, “bao nhiêu”, “Xin lỗi”) → chặn nhầm 7/8 câu sạch (FP 0.875) dù attacks 20/20; đã giới hạn Presidio ở VN_CCCD, VN_PHONE, EMAIL_ADDRESS, PHONE_NUMBER và chạy lại một lần, không đổi expected/fixtures. (2) Latency lần 2 bị cộng thời gian chờ 429 do quota 8000 tokens/phút; giữ nguyên số và giải thích trong blueprint. (2b) Quota 200K token/ngày/model của Groq free khiến evaluator phải khác model với generation/judge/guard; ghi rõ trong metadata và các báo cáo. (3) 3 attack off-topic lọt qua NeMo (chưa đạt bonus 18/20), ghi nhận trong blueprint, không chỉnh policy rồi chạy lại để lấy bonus. (4) Python 3.12 thay vì 3.11.
- Nguồn/AI hỗ trợ nếu có: Claude Code (Anthropic) hỗ trợ viết code Tasks 1–12, adapter m1–m5, cấu hình Groq/NeMo, chạy các lệnh live và soạn báo cáo. Kiểm chứng: đọc từng diff, chạy unit tests, đối chiếu mọi số trong báo cáo với JSON trong `artifacts/`, đọc log setup để loại lần chạy dùng fallback, và chạy checker. Mã m1–m5 là bài Day 18 của chính tôi.

## Cập nhật sau khi nộp

Không có cập nhật sau khi nộp tại thời điểm viết; nếu có, sẽ ghi commit, thời điểm và lý do tại đây.
