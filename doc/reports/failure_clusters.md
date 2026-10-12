# Phân tích lỗi Phase A

Nhóm: DinhHoangDuc (nhóm một người)

Người phụ trách: Đinh Hoàng Đức — 2A202602795

Nguồn: `artifacts/ragas_50q.json`; thời điểm xuất artifact (UTC):
2026-10-12T16:27:12.813983+00:00. Đủ 50 câu và bốn metric hữu hạn trong [0,1].
Số đo và thứ tự bottom 10 lấy trực tiếp từ JSON; số trong bảng làm tròn bốn chữ số thập phân.
Đối chiếu offline bằng `scripts/audit_evidence.py`, kết quả và SHA-256 nguồn trong
`artifacts/evidence_audit.json`. Log chạy thực tế trong `artifacts/verification/`.

## Dữ liệu, evaluator và checkpoint

Giữ nguyên `artifacts/answers_50q.json`: generation `openai/gpt-oss-20b`,
mode LLM cho toàn bộ answers. Question, answer, contexts, ground_truth được checker
đối chiếu nguyên văn với nguồn. Không lấy reference thay answer và không xóa policy cũ.
Retrieval BM25 + BGE-M3/Qdrant, rerank `BAAI/bge-reranker-v2-m3`; M5 dùng enrichment cục bộ.

Evaluator thực tế: `BAAI/bge-m3`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`; provider của lần tiếp tục: `groq`.
RAGAS answer relevancy strictness = 1; embedding dùng BGE-M3 cục bộ.
Qwen đã đo một phần rồi hết quota token/ngày. Lần tiếp tục bật
`RAGAS_SKIP_PRIMARY=true`, giữ checkpoint và dùng evaluator dự phòng cho
122 ô metric. Mỗi ô dự phòng có index/metric/model/score
trong `evaluator_details.fallback_cells`; index bắt đầu từ 0 theo `per_question`.
Model dự phòng được chọn theo quota và thứ tự cấu hình, không theo điểm cao/thấp;
các ô đã có điểm không được thay bằng điểm của model khác.
Checkpoint ban đầu lưu tại `artifacts/previous_runs/ragas_checkpoint_before_resume.json`.
Các điểm checkpoint không được mô tả là lời gọi API mới trong lần tiếp tục.
Không có tổng token usage/chi phí được lưu cho cả batch, nên không báo cáo ước tính chi phí.

## Điểm theo distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
|---|---:|---:|---:|
| faithfulness | 0.9208 | 0.4477 | 0.8500 |
| answer_relevancy | 0.8958 | 0.6869 | 0.7358 |
| context_precision | 0.8917 | 0.7542 | 0.9500 |
| context_recall | 0.6833 | 0.6458 | 0.7167 |
| avg_score | 0.8479 | 0.6336 | 0.8131 |

Mean toàn bộ câu (trọng số theo số câu): faithfulness 0.7174,
answer_relevancy 0.7802, context_precision 0.8483,
context_recall 0.6750; avg_score 0.7552.
Không lấy trung bình không trọng số của ba distribution có kích thước khác nhau.

## Bottom 10

F = faithfulness, AR = answer_relevancy, CP = context_precision, CR = context_recall.
Diagnosis và suggested_fix dưới đây giữ nguyên nội dung Task 3 trong JSON.
Chúng là giả thuyết từ diagnostic tree, không tự chứng minh nguyên nhân.

| Rank | id | Distribution | F | AR | CP | CR | Avg | Worst metric |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 5 | factual | 0.0000 | 0.6893 | 0.0000 | 0.0000 | 0.1723 | faithfulness |
| 2 | 39 | multi_hop | 0.1667 | 0.8103 | 0.0000 | 0.0000 | 0.2442 | context_precision |
| 3 | 40 | multi_hop | 0.0000 | 0.6265 | 0.3333 | 0.3333 | 0.3233 | faithfulness |
| 4 | 33 | multi_hop | 0.0000 | 0.3199 | 0.5000 | 0.5000 | 0.3300 | faithfulness |
| 5 | 23 | multi_hop | 0.0000 | 0.5870 | 1.0000 | 0.5000 | 0.5217 | faithfulness |
| 6 | 21 | multi_hop | 0.6667 | 0.0000 | 1.0000 | 0.5000 | 0.5417 | answer_relevancy |
| 7 | 26 | multi_hop | 0.6667 | 0.8496 | 0.0000 | 0.6667 | 0.5457 | context_precision |
| 8 | 27 | multi_hop | 0.0000 | 0.7499 | 0.5833 | 1.0000 | 0.5833 | faithfulness |
| 9 | 35 | multi_hop | 0.5000 | 0.6467 | 1.0000 | 0.2500 | 0.5992 | context_recall |
| 10 | 31 | multi_hop | 0.0000 | 0.7874 | 1.0000 | 0.6667 | 0.6135 | faithfulness |

### Câu 5: Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?

Answer thực tế: Kế toán trưởng.

Diagnosis Task 3: LLM hallucinating (faithfulness=0.00)

Suggested fix Task 3: Tighten system prompt, lower temperature

### Câu 39: So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA.

Answer thực tế: **So sánh chính sách mật khẩu v1.0 và v2.0**  \| Yếu tố \| Chính sách v1.0 \| Chính sách v2.0 \| \|--------\|-----------------\|-----------------\| \| **Độ dài tối thiểu** \| 8 ký tự \| 12 ký tự (khuyến khích passphrase > 16 ký tự) \| \| **Thời hạn đổi mật khẩu** \| Không có thông tin trong đoạn trích \| Không có thông tin trong đoạn trích \| \| **Xác thực đa yếu tố (MFA)** \| Không đề cập đến MFA \| Bắt buộc cho tất cả nhân viên; các hệ thống: email, VPN, hệ thống nội bộ. Phương thức: Authenticator (ưu tiên), SMS OTP, YubiKey \|  > **Lưu ý:** Thông tin về thời hạn đổi mật khẩu không xuất hiện trong bất kỳ hai chính sách được trích dẫn.

Diagnosis Task 3: Too many irrelevant chunks (context_precision=0.00); bẫy phiên bản: reference phân biệt v1.0, v2.0; contexts: v1, v2, v2.0; câu nhiều bước: contexts có thể chỉ phủ một trong các nguồn cần ghép

Suggested fix Task 3: Add reranking or metadata filter; gắn metadata effective/superseded khi index, ưu tiên bản hiện hành và yêu cầu answer nêu rõ phiên bản; tách câu hỏi thành sub-queries hoặc tăng RERANK_TOP_K cho multi-hop

### Câu 40: Nhân viên cấp Manager có thời hạn thử việc 90 ngày, đang trong tháng thứ 3, phát hiện vi phạm bảo mật. Họ nên và không nên làm gì theo chính sách?

Answer thực tế: **Bạn nên:**  - **Báo cáo ngay lập tức**: Gửi email báo cáo vi phạm bảo mật đến `helpdesk@cty.vn` **trong vòng 1 giờ** kể từ khi phát hiện. - **Cung cấp đầy đủ thông tin**: Nêu rõ tính chất vi phạm, thời điểm, người/đối tượng liên quan và bất kỳ bằng chứng nào có sẵn. - **Theo dõi và hợp tác**: Khi đội ngũ CNTT & An ninh thông tin liên lạc để xử lý, bạn cần cung cấp hỗ trợ và thông tin thêm nếu được yêu cầu. - **Đăng ký theo quy trình nội bộ** (nếu có): Thông báo cho bộ phận HR (đặc biệt vì bạn đang trong thời hạn thử việc) để họ ghi nhận và theo dõi.  **Bạn không nên:**  - **Bỏ qua hoặc trì hoãn**: Không để ý hoặc chờ đợi lâu hơn 1 giờ trước khi báo cáo. - **Báo cáo sau khi đã được xử lý**: Nếu có thể, hãy báo ngay mà không chờ đợi kết quả xử lý. - **Chia sẻ thông tin vi phạm**: Không tiết lộ chi tiết cho đồng nghiệp hoặc bên ngoài mà chưa được sự đồng ý của bộ phận bảo mật. - **Bỏ qua quy định của chính sách thử việc**: Mặc dù bạn đang trong thời hạn thử việc, vẫn phải tuân thủ các quy định bảo mật nghiêm ngặt của công ty.

Diagnosis Task 3: LLM hallucinating (faithfulness=0.00)

Suggested fix Task 3: Tighten system prompt, lower temperature

### Câu 33: Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu?

Answer thực tế: Không tìm thấy.

Diagnosis Task 3: LLM hallucinating (faithfulness=0.00); pipeline trả 'Không tìm thấy' — cần đọc contexts để phân biệt thiếu retrieval và lỗi tổng hợp

Suggested fix Task 3: Tighten system prompt, lower temperature; đối chiếu điều khoản trong contexts; nếu đã đủ thì sửa generation/ghép chunk, nếu còn thiếu thì kiểm tra index và tăng top-k trước rerank

### Câu 23: Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?

Answer thực tế: Không tìm thấy.

Diagnosis Task 3: LLM hallucinating (faithfulness=0.00); pipeline trả 'Không tìm thấy' — cần đọc contexts để phân biệt thiếu retrieval và lỗi tổng hợp

Suggested fix Task 3: Tighten system prompt, lower temperature; đối chiếu điều khoản trong contexts; nếu đã đủ thì sửa generation/ghép chunk, nếu còn thiếu thì kiểm tra index và tăng top-k trước rerank

### Câu 21: Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

Answer thực tế: **Số ngày phép năm:**   - Cơ sở: 15 ngày (được quy định trong chính sách năm 2024).   - Bổ sung: 1 ngày cho mỗi 3 năm thâm niên → với 9 năm thâm niên: \(9 ÷ 3 = 3\) ngày.    **Tổng cộng:** \(15 + 3 = 18\) ngày phép năm.  **Lương:**   Chính sách nghỉ phép năm (Phiên bản 2024) không đề cập đến mức lương hay phạm vi lương cho nhân viên. Do đó, không thể xác định lương “trong khoảng nào” dựa trên tài liệu đã cung cấp.

Diagnosis Task 3: Answer doesn't match question (answer_relevancy=0.00)

Suggested fix Task 3: Improve prompt template

### Câu 26: Nhân viên vừa kết hôn và cùng tuần đó có con kết hôn. Tổng số ngày nghỉ đặc biệt có lương là bao nhiêu?

Answer thực tế: Tổng số ngày nghỉ đặc biệt có lương là **4 ngày** (3 ngày cho việc kết hôn + 1 ngày cho con kết hôn).

Diagnosis Task 3: Too many irrelevant chunks (context_precision=0.00); câu nhiều bước: contexts có thể chỉ phủ một trong các nguồn cần ghép

Suggested fix Task 3: Add reranking or metadata filter; tách câu hỏi thành sub-queries hoặc tăng RERANK_TOP_K cho multi-hop

### Câu 27: Nhân viên có thâm niên 7 năm theo v2024 được nghỉ bao nhiêu ngày phép? Nếu bị trừ thêm 4 ngày ốm không có giấy y tế thì còn lại bao nhiêu?

Answer thực tế: Nhân viên có thâm niên 7 năm theo chính sách v2024 được hưởng:  - **Ngày phép cơ bản:** 15 ngày - **Thêm theo thâm niên:** 1 ngày cho mỗi 3 năm → 7 năm ÷ 3 = 2 chặng (làm tròn xuống), tức 2 ngày  **Tổng ngày phép:** 15 + 2 = 17 ngày  Nếu bị trừ 4 ngày nghỉ do ốm mà không có giấy tờ y tế hợp lệ, ngày phép còn lại là:  17 – 4 = **13 ngày**.

Diagnosis Task 3: LLM hallucinating (faithfulness=0.00)

Suggested fix Task 3: Tighten system prompt, lower temperature

### Câu 35: Nhân viên Junior P1 có lương cơ bản 12 triệu vừa vào thử việc: nhận lương bao nhiêu và phụ cấp gì tháng đầu tiên?

Answer thực tế: Lương: 10 200 000 VNĐ (85 % của 12 triệu).   Phụ cấp: Phụ cấp ăn trưa được áp dụng đầy đủ từ ngày đầu tiên.

Diagnosis Task 3: Missing relevant chunks (context_recall=0.25); câu nhiều bước: contexts có thể chỉ phủ một trong các nguồn cần ghép

Suggested fix Task 3: Improve chunking or add BM25; tách câu hỏi thành sub-queries hoặc tăng RERANK_TOP_K cho multi-hop

### Câu 31: Nhân viên đi công tác trong nước, ở khách sạn 2 đêm với giá 1.500.000 VNĐ/đêm. Công ty thanh toán tối đa bao nhiêu cho tiền khách sạn?

Answer thực tế: Công ty sẽ trả tối đa **2 400 000 VND** cho tiền khách sạn.

Diagnosis Task 3: LLM hallucinating (faithfulness=0.00)

Suggested fix Task 3: Tighten system prompt, lower temperature

## Đọc nội dung để phân biệt nguyên nhân

- **Câu 5 — nhầm luồng phê duyệt và bảng bị cắt:** context đầu từ `tam_ung.md`
  nêu Kế toán trưởng, trong khi câu hỏi hỏi mua thiết bị; bảng trong `mua_sam.md`
  bị cắt trước ngưỡng đơn hàng tương ứng. Answer chọn Kế toán trưởng, reference
  là Tổng Giám đốc. Cần lấy đủ bảng thẩm quyền và phân biệt tạm ứng/mua sắm.
- **Câu 40 — thiếu điều khoản bảo mật và thêm suy diễn:** contexts thiên về thử việc;
  đoạn bảo mật mới có kênh/thời hạn báo cáo. Answer thêm bước thông báo HR chưa có
  trong contexts và thiếu các cấm tự xử lý malware/xóa dấu vết trong reference.
  Truy hồi thêm phần xử lý sự cố, bỏ các bước không được tài liệu hỗ trợ.
- **Câu 21 — thiếu nguồn thứ hai:** contexts chỉ có policy nghỉ phép, thiếu bảng lương.
  Answer tính đúng số ngày phép nhưng từ chối phần khung lương. Ưu tiên tách truy vấn
  nghỉ phép/lương, truy hồi từng nhánh rồi ghép kết quả trước generation.
- **Câu 33 — thiếu nguồn phụ cấp:** contexts chỉ có policy phép cũ/mới, thiếu `phu_cap.md`;
  answer “Không tìm thấy”. Retrieval cần phủ cả hai nguồn; generation vẫn nên trả phần
  đã có bằng chứng và ghi rõ phần còn thiếu.
- **Câu 23 — lỗi tổng hợp/generation:** contexts đã chứa cam kết làm việc ít nhất một năm
  và hoàn trả toàn bộ chi phí trước hạn, nhưng answer vẫn là “Không tìm thấy”.
  Vì vậy không thể kết luận abstain luôn do thiếu retrieval như diagnosis tự động;
  cần ghép hai chunk liền kề và yêu cầu generation áp dụng điều kiện từ chúng.
- **Câu 29/34 — cấp phê duyệt:** answer tự suy diễn cấp quản lý/Giám đốc thay vì đúng
  Kế toán trưởng. Kiểm tra chunk chứa ngưỡng phê duyệt và bắt answer trích điều khoản;
  không dựa vào tên chức vụ thường gặp để điền khoảng trống.
- **Câu 39 — thiếu điều khoản chu kỳ:** answer nói không có thông tin về thời hạn đổi
  mật khẩu. Truy vấn so sánh phiên bản cần lấy cả độ dài, chu kỳ và MFA của từng bản.
- **Câu 35 — thiếu định lượng phụ cấp:** answer tính lương thử việc và nêu phụ cấp ăn
  trưa từ ngày đầu nhưng không có số tiền, điều kiện điện thoại hoặc tổng nhận theo
  reference. Contexts chưa có `phu_cap.md`; cần truy hồi chính sách phụ cấp và tách
  câu trả lời theo từng khoản thay vì chỉ nêu tên khoản.
- **Câu 26 — cảnh báo về evaluator:** answer cộng đúng hai loại nghỉ, contexts đã
  có cả hai điều kiện, nhưng context_precision được LLM chấm bằng 0. Không thể từ
  điểm này kết luận mọi chunk đều không liên quan. Cần lưu phán đoán precision
  từng context và kiểm tra cách evaluator xử lý phép cộng/ý “không trừ phép năm”.
- **Câu 27/31 — suy luận số học và faithfulness:** answer khớp các kết quả chính
  trong reference; contexts có quy tắc thâm niên/trừ nghỉ ốm (27) và trần khách sạn
  mỗi đêm (31). Faithfulness vẫn bằng 0 trong lần đo này. Vì chưa lưu statement/NLI
  trace, chưa xác định được đây là lỗi trích statement, đánh giá suy luận từ quy tắc
  hay thiếu chi tiết hỗ trợ; không gọi chúng là hallucination chỉ từ điểm số.
  Cần thẩm định các trường hợp này bằng người và cùng một evaluator trước khi dùng
  faithfulness làm gate production.
- **Câu 6/12 — answer dạng cụm từ:** source RAGAS 0.1.22 lọc bỏ các đoạn không kết thúc
  bằng dấu chấm trước khi tạo prompt trích statement. Adapter sửa trường hợp danh sách
  bị rỗng bằng cách chuyển nguyên các đoạn không rỗng cho LLM; không thêm dấu chấm,
  sửa answer hoặc tự suy ra score. Điểm sửa lỗi vẫn phải qua lời gọi trích statement
  và NLI thật. Protocol của các ô faithfulness chấm theo adapter mới có nhãn
  `ragas_0.1.22_retain_fragments_v1`; ô checkpoint cũ giữ protocol mặc định.

## Matrix worst metric và nhóm yếu

Matrix đếm worst metric của mọi câu, kể cả câu có điểm cao. Khi hòa, code chọn
metric đầu tiên theo thứ tự F, AR, CP, CR; tổng cột mô tả kích thước nhóm.

| Worst metric | factual | multi_hop | adversarial | Tổng |
|---|---:|---:|---:|---:|
| faithfulness | 2 | 10 | 2 | 14 |
| answer_relevancy | 9 | 2 | 3 | 14 |
| context_precision | 2 | 3 | 1 | 6 |
| context_recall | 7 | 5 | 4 | 16 |

Nhóm yếu theo mean thấp nhất: **multi_hop**
(0.6336).
Worst metric xuất hiện nhiều nhất: **context_recall**
(16 câu).

Insight Task 4: Mean avg_score theo distribution (thấp→cao): multi_hop=0.634, adversarial=0.813, factual=0.848. 'multi_hop' yếu nhất; trong nhóm này metric trung bình thấp nhất là 'faithfulness' (0.448). Trên toàn bộ 50 câu, 'context_recall' là worst metric của 16 câu (matrix đếm mọi câu, không chỉ câu lỗi). Hướng kiểm tra đầu tiên: Improve chunking or add BM25; kết luận nguyên nhân cần đọc question/answer/contexts của bottom 10.

## Bẫy adversarial và giới hạn

Adversarial mean = 0.8131; factual mean = 0.8479.
Điều kiện bonus “adversarial thấp hơn factual”:
**đạt**.
Kết quả được giữ nguyên dù có đạt bonus hay không.

Đọc answer/contexts cho thấy bẫy phiên bản ở các câu 41/42/43/44/49: retrieval giữ
cả bản cũ lẫn mới. Câu 44 lấy cả chu kỳ cũ và mới, nhưng answer chọn chu kỳ hiện hành;
câu 49 phân biệt bản cũ và hiện hành. Bẫy phủ định ở các câu 46/47/48/50 được answer
trả theo hướng “Không”, nhưng câu 48 chỉ trả ngắn và thiếu chi tiết BHXH trong reference.
Answer đúng giá trị chính vẫn có thể thiếu các ý phụ mà RAGAS dùng để đánh giá recall.
Không kết luận pipeline hiểu mọi xung đột phiên bản chỉ từ mean của nhóm này.
Một số ground_truth adversarial còn có nhận xét về bẫy/pipeline ngoài nội dung policy.
Đó có thể là yếu tố ảnh hưởng context_recall; chưa có NLI trace từng statement để
kết luận nguyên nhân này từ điểm tổng. Không sửa reference của đề bài để tăng điểm.

Giới hạn quan trọng: factual có nhiều ô do Qwen chấm, các ô chưa đo của multi_hop/
adversarial dùng các evaluator dự phòng. Nếu có `openai/gpt-oss-20b` trong evaluator,
đó cũng là model generation/judge, nên có nguy cơ thiên lệch cùng model.
Chênh lệch distribution có thể bao gồm khác biệt
evaluator, nên chưa phải so sánh được kiểm soát bằng cùng một model. Muốn kiểm tra
nguyên nhân, cần chấm lại toàn bộ bằng một evaluator khi có quota; không dùng lần
chạy kết hợp này làm baseline ổn định cho production. Mẫu nhỏ, local enrichment,
child chunks bị cắt giữa điều khoản và answer relevancy strictness = 1 cũng là giới hạn.
