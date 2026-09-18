# HR CV Screener Agent — Spec (chốt ngày 2026-09-07)

Bản này là kết quả của phiên grilling. Mọi quyết định dưới đây đã được xác nhận.

## 1. Mục tiêu & khán giả
Demo chạy được, trình bày trước thầy cô viện AI và sinh viên nghiên cứu. Sản phẩm HR
độc lập, dựng mới hoàn toàn, KHÔNG tái dùng `LegalAgent/`. Một người thực hiện, 3-5 ngày.

Demo phải chứng minh hai điều:
- **E1** — mọi điểm số truy ngược được về câu chữ cụ thể trong CV (evidence-linked scoring).
- **E2** — agent nhiều bước hơn gì so với một lần gọi LLM, bằng số đo được.

Chốt kiểm tra: hết ngày 2 mà 5 tool tất định chưa chạy xanh test → bỏ phần parse PDF.

## 2. Ranh giới quyết định
Agent KHÔNG loại ứng viên. Nó phân hồ sơ vào 3 luồng khớp đúng nhãn dataset:
`Good Fit` / `Potential Fit` / `No Fit`.
Ngoại lệ duy nhất: thiếu tiêu chí `must_have` → `reject_fast`, nhưng VẪN ghi lại điểm
mọi tiêu chí khác để giao diện giải thích được vì sao.

## 3. Dữ liệu

| Vai trò | Nguồn | Số lượng |
|---|---|---|
| dev (chỉnh ngưỡng + prompt) | HF `cnamuangtoun/resume-job-description-fit` — `train.csv` | 300 mẫu phân tầng |
| test (chạy MỘT lần, cuối cùng) | cùng bộ — `test.csv` | 500 mẫu phân tầng |
| demo — CV có PDF thật | Kaggle `snehaanbhawal/resume-dataset` (CC0), lọc ngành IT | 20-30 CV |
| demo — JD thật | HF `jacob-hugging-face/job-descriptions` (`training_data.csv`, 853 JD) | 3-5 JD IT |

Sự thật đã kiểm chứng ngày 2026-09-07:
- `cnamuangtoun` là 2 file CSV phẳng: `train.csv`, `test.csv`. Cột:
  `resume_text`, `job_description_text`, `label`. `test.csv` = 1759 dòng
  (No Fit 857 / Good Fit 458 / Potential Fit 444). Độ dài resume trung bình ~5663 ký tự.
- **Văn bản bị mất dấu cách giữa các câu** (`"consulting projects.Proven ability"`).
  `search_evidence` BẮT BUỘC phải chịu được điều này.
- `jacob-hugging-face/job-descriptions` → `training_data.csv`, 853 dòng, cột
  `company_name, job_description, position_title, description_length, model_response`.
  Có sẵn các vị trí IT (Web Developer, Frontend Web Developer...). JD viết thường,
  đã lược bỏ dấu câu, ngăn cách bằng xuống dòng.
- Máy hiện KHÔNG có `~/.kaggle/kaggle.json` → phần CV PDF cần thao tác tay của người dùng.

Toàn bộ tiếng Anh. Parse PDF bằng `pdfplumber`, hỏng thì rơi về cột `Resume_str`.
Nhãn của dataset được dùng làm ground truth (quyết định của chủ dự án).

## 4. Kiến trúc graph (LangGraph)

```
ingest -> guard --(bẩn)--> quarantine -> END
           |
        extract --(thiếu trường, tối đa 2 lần)--> repair --+
           |<---------------------------------------------+
      load_rubric -> must_have_check --(thiếu)--> reject_fast -> END
           |
      score_criteria   [tools: search_evidence, calculate_experience, normalize_skill]
           |
       aggregate --(vùng xám)--> deep_review --+
           |<-----------------------------------+
        decide -> rank -> END
```

4 conditional edge: `guard`, `repair` (có giới hạn vòng lặp), `must_have`, `vùng xám`.
State là MỘT pydantic model chảy xuyên graph.
Mỗi nhánh phải có % lưu lượng thật đo trên dataset — không có số thì nhánh đó là trang trí.

## 5. Năm tool tất định

| Tool | Vì sao không để LLM tự làm |
|---|---|
| `calculate_experience` | LLM sai số học ngày tháng, gap và overlap |
| `normalize_skill` | Biến so chuỗi thành so khái niệm (React / ReactJS / React.js) |
| `search_evidence` | Truy ngược điểm về đúng đoạn văn bản — nền tảng của E1 |
| `scan_injection` | Phát hiện chỉ thị ẩn trong CV; nuôi nhánh `guard` |
| `aggregate_scorecard` | Cộng có trọng số phải tất định, không cho LLM tự cộng |

## 6. Rubric — JD thành dữ liệu, không thành prompt
Form có cấu trúc -> YAML: danh sách tiêu chí + trọng số + cờ `must_have` + hai ngưỡng cắt.
HR chỉnh được, hệ thống vẫn tái lập được.
Ngưỡng vùng xám chính là tham số điều khiển nhánh `deep_review`.

## 7. Đo đạc
- Chính: macro-F1 + confusion matrix trên 3 lớp, tập test 500 mẫu, chạy một lần.
- Ba baseline: (a) một lần gọi LLM prompt ngây thơ; (b) một lần gọi LLM CÓ đầy đủ rubric
  (baseline mạnh, có thể sát agent); (c) TF-IDF cosine, không LLM.
- Nếu (b) gần bằng agent: giá trị của agent nằm ở truy vết bằng chứng, xử lý CV lỗi,
  chặn injection, và chi phí thấp hơn nhờ đường tắt — phải có số token chứng minh.
- Bias: counterfactual đổi tên/giới/trường trên ~50 CV, xem điểm có đổi không.
- Injection: nút bật/tắt lớp guard, chạy lại CV độc, cho xem điểm sụp về đúng.

## 8. Tái lập
`temperature=0`, seed cố định, cache theo hash `(model, prompt, input)` ra JSONL,
log token + latency từng nhánh. Mọi số trên slide sinh từ một lệnh duy nhất.
Model: `gpt-4o-mini`, key OpenAI trả phí.

## 9. Giao diện & slide
Streamlit, toàn bộ tiếng Anh. Ba hình slide:
1. Graph tổng thể, conditional edge tô màu, kèm % lưu lượng thật.
2. Bảng 5 tool với cột "vì sao không để LLM làm".
3. Một hồ sơ chạy xuyên hệ thống, state biến đổi qua từng node.
Sơ đồ sinh tự động bằng Mermaid từ chính graph.

## 10. Cấu trúc repo
```
Agent/
  src/  contracts/  tools/  graph/  rubric/  data/  llm/
  eval/ (run.py, baselines.py, bias.py, cache.py)
  data/ (raw, samples, rubrics)
  app/  (streamlit)
  tests/
  scripts/
  docs/
```

## 11. Cố ý KHÔNG làm
CV scan/OCR · tiếng Việt trong luồng đo · vector DB / GraphRAG · tự động loại ứng viên ·
tự gán nhãn gold · tái dùng `LegalAgent/` · framework agent nào khác ngoài LangGraph.

## 12. Môi trường đã kiểm chứng (2026-09-07)
Python 3.13.12 (miniconda base). Đã cài sẵn: `pydantic` 2.12.4, `langgraph` 1.2.1,
`openai` 2.38.0, `pandas` 3.0.3, `streamlit` 1.61.1, `datasets` 4.8.5, `pytest` 9.0.3,
`huggingface_hub` 1.12.0, `scikit-learn` 1.8.0, `numpy` 2.4.4, `pypdf` 6.10.2,
`rank-bm25` 0.2.2, `PyYAML` 6.0.3, `python-dotenv` 1.2.1, `tenacity` 9.1.4,
`matplotlib` 3.10.9. Thiếu: `pdfplumber`, `kagglehub`. `uv` không có. `git` 2.53.0.
Thư mục CHƯA phải git repo. Chưa có `OPENAI_API_KEY` trong môi trường.
