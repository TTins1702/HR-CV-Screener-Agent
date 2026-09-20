# PROMPT — copy toàn bộ phần dưới đường kẻ, dán vào Claude CLI

---

/frontend-slides:frontend-slides

Hãy tạo cho tôi một bộ slide HTML để báo cáo nghiên cứu về **CV_SCREENER_AGENT** — một agent
sàng lọc CV có truy vết bằng chứng, dựng trên LangGraph.

Bài nói vừa phải nêu bật được hệ thống đã dựng, vừa phải để người nghe mang về được nguyên
tắc dùng lại được. Khán giả là thầy cô viện AI và sinh viên nghiên cứu, **sẽ phản biện khó**.
Vì vậy quy tắc số một của bộ slide này là: **không có câu nào trên slide mà không chỉ ra được
nguồn.**

## Nguồn nội dung — ĐỌC TRƯỚC KHI VIẾT SLIDE

Repo hiện tại là nguồn duy nhất cho mọi con số. Hãy đọc, theo đúng thứ tự này:

| File | Dùng cho |
|---|---|
| `docs/spec/2026-09-07-hr-cv-screener-spec.md` | Mục tiêu E1/E2, ranh giới quyết định, dữ liệu, 5 tool, những gì cố ý không làm |
| `docs/diagrams/graph.mmd` | Danh sách node và cạnh thật của graph |
| `docs/measurements/test_shipped.md` | Kết quả chính: test 500 mẫu, chạy MỘT lần |
| `docs/measurements/dev_shipped.md` | Kết quả dev 300 mẫu |
| `docs/measurements/baselines.md` | Agent so với 3 baseline |
| `docs/measurements/scoring_layer.md` | Ordering bị đảo, ceiling, bootstrap, power |
| `docs/measurements/must_have_gate.md` | Ablation cổng must-have |
| `docs/measurements/no_gray_zone.md` | Ablation vùng xám |
| `docs/measurements/no_guard.md` | Ablation guard, tách theo severity |
| `docs/measurements/bias_school.md`, `bias_identity.md` | Hai nhánh counterfactual |
| `docs/measurements/extraction_confidence.md` | Vì sao không route trên self-report |
| `src/tools/*.py` | Cơ chế thật của 5 tool — đọc docstring đầu file, chúng ghi rõ vì sao tool tồn tại |
| `src/graph/routes.py`, `src/graph/build.py` | 4 conditional edge |
| `data/eval/test500/agent__shipped.jsonl` | Chọn hồ sơ thật cho slide walkthrough |

### Luật về số liệu — tuyệt đối không phá

1. **Mọi con số phải copy nguyên từ file trong `docs/measurements/`.** Không tự tính lại,
   không làm tròn khác, không nội suy, không bịa. Nếu một con số bạn muốn đưa lên slide mà
   không có trong file nào, hãy **để trống và ghi rõ là còn thiếu** — đừng điền.
2. **Mỗi slide có số phải ghi nguồn ở chân slide**, dạng
   `nguồn: docs/measurements/test_shipped.md · n=500`.
3. **CẤM trích `docs/measurements/branch_traffic_25.md`** — chính file đó tự đánh dấu đã bị
   superseded (con số 32.0% trong đó không còn là hành vi của pipeline). Các file
   `branch_traffic*.md` khác cũng là log lịch sử, không phải số để trình bày.
4. **Không trộn hai tập vào một bảng.** `test_500` chỉ có số của agent (baseline chưa chạy
   trên tập đó). Bảng so 4 hệ thống là `dev_300`. Bảng nào cũng phải ghi n ngay trên tiêu đề
   cột hoặc trong caption.
5. **Không gộp `no_guard` HIGH và LOW thành một con số** — file đó nói rõ vì sao, và gộp lại
   là claim một lớp phòng thủ chưa từng tồn tại.
6. Repo đang có thay đổi **chưa commit** ở `src/tools/experience.py` và `src/graph/extract.py`
   (thêm đối chiếu `work_periods` và `excluded_years`). Mọi số đã đo đều lấy **trước** thay đổi
   này. Nếu slide nói về `calculate_experience`, hãy mô tả hành vi **đã được đo**; nếu có nhắc
   phần cải tiến thì ghi rõ là chưa vào số đo nào.

## Chín chỗ phải phát biểu cho đúng (đã đối chiếu code)

Đây là những chỗ dễ bị hỏi trúng nhất. Hãy dùng đúng cách diễn đạt ở cột phải.

| Đừng viết | Hãy viết, vì code là vậy |
|---|---|
| "12 node" | **13 node**: `ingest, guard, quarantine, extract, repair, load_rubric, must_have_check, reject_fast, score_criteria, aggregate, deep_review, decide, rank` |
| `search_evidence` dùng "Fuzzy/BM25" | Khớp **chính xác trên bản văn bản đã chuẩn hóa**; không có thì trượt cửa sổ theo từ và cho điểm bằng `difflib.SequenceMatcher`, ngưỡng `min_score=0.75`. **Không có BM25 ở bất kỳ đâu trong code** (`rank-bm25` nằm trong `requirements.txt` nhưng không được dùng). Nói BM25 trên slide là lỗi người nghe kiểm được trong 30 giây |
| `normalize_skill` dùng "ontology" | Một **bảng alias phẳng** trong `data/skills/aliases.yaml`, sinh ra hai bảng tra: khóa canonical để so sánh, và bảng surface form để làm truy vấn cho `search_evidence`. Gọi là ontology là hứa một cấu trúc phân cấp không tồn tại |
| `scan_injection` "chặn mã độc" | Phát hiện **chỉ thị ẩn nhắm vào chính model chấm điểm**, bằng regex trên văn bản đã chuẩn hóa. HIGH → `quarantine`, LOW → chỉ ghi chú và cho chạy tiếp |
| "guard chặn được tấn công" | **0 trong 300 CV thật kích hoạt bất kỳ rule nào** → nhánh guard không có lưu lượng thật. Số trong `no_guard.md` đến từ **fixture tổng hợp**, và file đó yêu cầu slide phải nói rõ điều này. Hơn nữa các hàng bị đầu độc đều chấm **đúng 0.550 — y như hàng sạch** kể cả khi tắt guard: giá trị của guard là **từ chối xử lý một tài liệu đang cố can thiệp**, KHÔNG phải ngăn được một can thiệp lẽ ra sẽ thành công |
| `must_have_check` "tiết kiệm token" | Đúng về token (`dev_300`: tiết kiệm 208.127 token) nhưng **phải trả giá bằng độ chính xác**: `shipped` 0.3721 so với `no_must_have_gate` 0.4052 macro-F1 — cổng này **làm giảm 0.0331**. Trên `test_500` cổng loại 121 hàng, **43 hàng không thật sự là `No Fit`, trong đó 28 hàng nhãn thật là `Good Fit`** → precision 0.645 |
| "Evidence Attribution: link trực tiếp đến văn bản gốc" | Coverage E1 = **0.486** trên `test_500` (0.439 trên `dev_300`), không phải 100%. Và **theo thiết kế**, tiêu chí chấm 0.0 trả về không kèm trích dẫn (`SCORE_SYSTEM` yêu cầu đúng như vậy), nên "CV im lặng về tiêu chí" và "CV rõ ràng không đạt" đến `aggregate_scorecard` như cùng một con số |
| "repair đã fix schema JSON" (trong 1 case study) | Lưu lượng `extract -> repair` là **1.4%** trên `test_500` (7 hàng), 3.0% trên `dev_300` (9 hàng). Gần như chắc chắn **không có hàng nào đi qua đồng thời** repair + injection + vùng xám. Dùng **một hàng thật** làm mạch chính, các nhánh khác đưa vào ô phụ nhỏ và **ghi rõ đó là hàng khác** |
| Agent "đạt chuẩn High-Risk AI" | Agent là **demo nghiên cứu**, không phải một conformity assessment. Claim đúng và giữ được: Annex III 4(a) xếp lớp hệ thống này vào High-Risk, nên truy vết / giải trình / giám sát con người là **yêu cầu bắt buộc chứ không phải tính năng tùy chọn**, và E1 là một **proxy đo được** cho yêu cầu đó |

## Yêu cầu chung

- **Ngôn ngữ: song ngữ.** Tiêu đề mỗi slide hiển thị tên tiếng Việt (dòng chính, cỡ lớn) kèm
  tên tiếng Anh (dòng phụ, nhỏ hơn, nhạt hơn). Nội dung viết tiếng Việt, giữ thuật ngữ tiếng
  Anh trong ngoặc khi cần — ví dụ: "truy vết bằng chứng (evidence-linked scoring)".
- **Giữ nguyên tiếng Anh, không dịch:** tên node (`must_have_check`), tên tool
  (`search_evidence`), tên nhãn (`Good Fit` / `Potential Fit` / `No Fit`), tên metric
  (`macro-F1`, `AUC`, `precision`, `recall`), tên file và tên dataset.
- **Style preset: Swiss Modern** (lưới rõ, hình học, tinh thần Bauhaus). Tông học thuật, hiện
  đại. Nền sáng, tương phản cao để chiếu máy chiếu trong phòng sáng.
- Định dạng 16:9, **một file HTML duy nhất**, CSS/JS inline, chạy được offline.
- Điều hướng bằng phím mũi tên / Space, hiển thị số slide.
- **Mỗi slide một ý.** Tối đa 5 gạch đầu dòng, mỗi dòng ≤ 14 từ. Phần diễn giải dài đưa xuống
  speaker notes.
- Mỗi slide kèm **speaker notes** 2–4 câu tiếng Việt, đặt trong khối ẩn, có phím tắt bật/tắt.
  Với slide có số, speaker notes phải chứa **câu trả lời cho câu phản biện dễ nhất** nhắm vào
  slide đó.
- Tổng cộng **13 slide**, tương ứng bài nói 20–25 phút cộng phần hỏi đáp.

## Hình ảnh — SVG trước, PNG thay sau

Repo không có ảnh chụp nào. Hãy **tự vẽ inline SVG** cho mọi hình, dùng dữ liệu thật đọc từ
`docs/measurements/`. Mỗi hình bọc trong một khối theo đúng khuôn này, để sau này tôi gửi PNG
thì chỉ cần thay phần bên trong:

```html
<figure class="asset-slot" data-asset="03-architecture-3-layers.png">
  <svg ...><!-- bản vẽ mẫu, dùng số thật --></svg>
  <figcaption>Chú thích ngắn tiếng Việt · nguồn: docs/diagrams/graph.mmd</figcaption>
</figure>
```

Các hình cần vẽ:

| `data-asset` | Nội dung | Slide |
|---|---|---|
| `01-bottleneck-map.png` | Sơ đồ 4 điểm nghẽn → thành phần nào của agent nhận trách nhiệm | 4 |
| `02-graph-13-nodes.png` | Graph đầy đủ 13 node, **4 conditional edge tô màu khác**, ghi % lưu lượng thật cạnh mỗi nhánh | 5 |
| `03-architecture-3-layers.png` | Cùng graph đó nhưng nhóm thành 3 khối logic, mỗi khối một mảng màu | 6 |
| `04-state-walkthrough.png` | Một hồ sơ thật đi qua từng node, hiển thị state biến đổi | 8 |
| `05-scorecard-evidence.png` | Scorecard kèm trích dẫn nguyên văn và offset ký tự trong CV | 9 |
| `06-baseline-bars.png` | Bar chart 4 hệ thống: macro-F1, token, evidence coverage | 10 |
| `07-confusion-test500.png` | Confusion matrix 3×3 dạng heatmap, `test_500` | 10 |
| `08-score-distribution.png` | Phân bố điểm theo nhãn thật (min/p25/median/p75/max), cho thấy ordering bị đảo | 11 |
| `09-branch-traffic.png` | Lưu lượng 4 nhánh + delta macro-F1 của từng ablation | 12 |

Cuối cùng, **in ra một bảng "slot ảnh → tên file PNG mong đợi"** kèm tỉ lệ khung hình gợi ý,
để tôi xuất PNG đúng cỡ.

## Ví dụ xuyên suốt

Thay cho một khối phụ trang trí, hãy chọn **một hồ sơ thật** từ
`data/eval/test500/agent__shipped.jsonl` và cho nó chạy xuyên bài. Trên slide 5–9, đặt một dải
phụ **cố định về vị trí và kiểu dáng**, nhãn cố định:

> **VÍ DỤ XUYÊN SUỐT: MỘT HỒ SƠ QUA PIPELINE / ONE CV THROUGH THE PIPELINE**

Dải đó cho biết hồ sơ đang ở node nào và state vừa đổi gì. Chọn hàng nào đi qua được
`score_criteria` và có trích dẫn bằng chứng thật — đừng chọn hàng `reject_fast` cho mạch chính.
Nếu không hàng nào vừa có `repair` vừa có vùng xám, đừng ghép hai hàng thành một: giữ mạch
chính một hàng, và gọi các hàng khác là hàng khác.

## Cấu trúc slide (đúng thứ tự này)

### 1. Slide mở đầu

**AGENT: CV_SCREENER_AGENT** làm dòng chính, cỡ rất lớn. Phụ đề: "Sàng lọc CV có truy vết bằng
chứng / Evidence-linked CV screening on a LangGraph state machine". Dưới cùng:
`{{TÊN NGƯỜI TRÌNH BÀY}}` · `{{NGÀY}}`.

Thêm một dòng số liệu duy nhất làm mỏ neo cho cả bài, lấy từ `test_shipped.md`:
13 node · 4 conditional edge · 5 tool tất định · 500 hồ sơ chạy một lần · evidence coverage 0.486.

### 2. Bối cảnh (1/3) — Luật đã xếp việc này vào nhóm nguy cơ cao
**Regulation already classes this as high-risk**

- Nguồn: **EU AI Act — Annex III, Point 4(a)**
  (`ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3`)
- Trích nguyên văn, để trong khối quote, giữ tiếng Anh:
  > "4. Employment, workers management and access to self-employment:
  > (a) AI systems intended to be used for the recruitment or selection of natural persons,
  > notably to place targeted job advertisements, to analyse and filter job applications, and
  > to evaluate candidates;"
- Hệ quả: rơi vào High-Risk thì bắt buộc **Article 12** (ghi nhật ký, truy vết),
  **Article 13** (minh bạch, giải trình được), **Article 14** (giám sát của con người).
- Luận điểm: một lần gọi LLM chấm điểm là **hộp đen** — không có cái nào trong ba điều trên.

Speaker notes phải chứa: đây là lập luận về **lớp hệ thống**, không phải tuyên bố đã tuân thủ.
Nếu bị hỏi "các bạn đã làm conformity assessment chưa" thì trả lời thẳng là chưa, đây là demo
nghiên cứu, và phần đóng góp là chỉ ra **một cách đo được** cho yêu cầu truy vết.

### 3. Bối cảnh (2/3) — Tự động hóa cũ loại oan người đủ năng lực
**Legacy automation rejects qualified people**

- Nguồn: **Fuller, J. B. & Raman, M. — "Hidden Workers: Untapped Talent"**, Harvard Business
  School & Accenture, 2021.
- Trích nguyên văn: "88% of employers acknowledge that qualified, high-skilled candidates are
  excluded from the hiring process simply because they do not match the exact criteria specified
  in a job description." — lên **94%** với nhóm kỹ năng bậc trung (middle-skilled).
- Cơ chế loại oan: lỗi format văn bản, thiếu đúng từ khóa nguyên văn (keyword mismatch), không
  đọc được ngữ cảnh kinh nghiệm thực tế.
- Luận điểm hai mặt: **không thể so khớp từ khóa ngô nghê, cũng không thể thả nổi cho LLM suy
  diễn.**

### 4. Bối cảnh (3/3) — Bốn điểm nghẽn, và ai trong agent nhận việc
**Four bottlenecks, and which component owns each**

Đây là slide bản lề. Dùng hình `01-bottleneck-map.png`, hai cột: điểm nghẽn → thành phần.
Hai điểm nghẽn đầu đến từ hai nguồn ngoài; **hai điểm nghẽn sau đến từ số đo trên chính dữ liệu
thật của dự án** — đó là phần mạnh nhất, hãy để nó cân bằng với hai nguồn kia:

| Điểm nghẽn | Bằng chứng | Thành phần nhận việc |
|---|---|---|
| Keyword mismatch loại oan người đủ năng lực | HBS 2021: 88% / 94% | `normalize_skill` (bảng alias) + `deep_review` cho vùng xám |
| Điểm không truy ngược được thì không dùng được ở nhóm High-Risk | EU AI Act Annex III 4(a) → Art. 12–14 | `search_evidence` + state graph có log từng node (E1) |
| Văn bản CV thật dính chữ, mất dấu cách giữa câu (`"projects.Proven ability"`) | spec §3, đã kiểm ngày 2026-09-07 | `text_norm`: mọi tool khớp trên bản chuẩn hóa, offset map về bản gốc |
| Cộng dồn thời gian làm việc kiểu naive sai trên hai phần ba CV thật | **198 trong 290** CV dev có khoảng thời gian **chồng lấn** (`src/tools/experience.py`) | `calculate_experience`: hợp các khoảng, đếm phần chồng lấn một lần |

**Không thêm số liệu nào không có trong hai nguồn trên hoặc trong repo.** Một con số thị trường
không nguồn trên slide này là món quà cho người phản biện.

### 5. Kiến trúc — cái graph thật
**The state graph, as built**

Hình `02-graph-13-nodes.png` chiếm phần lớn slide. Yêu cầu:

- Đủ 13 node, tên giữ nguyên tiếng Anh.
- **4 conditional edge** (`guard`, `repair` có giới hạn vòng lặp, `must_have`, vùng xám) tô màu
  khác hẳn các cạnh thường.
- Cạnh mỗi conditional edge ghi **% lưu lượng thật** từ `test_shipped.md`:
  `guard -> quarantine` 0.2% · `extract -> repair` 1.4% · `must_have_check -> reject_fast` 24.2% ·
  `aggregate -> deep_review` 15.4%.
- Một dòng dưới hình: **"Nhánh không có số đo là nhánh trang trí"** (spec §4) — đây là nguyên
  tắc người nghe mang về được.
- State là **một** pydantic model chảy xuyên graph.

### 6. Kiến trúc — ba khối logic
**Three logical layers**

Triết lý làm tiêu đề phụ, đóng khung nổi bật:

> **"Đừng bắt LLM làm những gì thuật toán cổ điển làm tốt hơn."**

Hình `03-architecture-3-layers.png`, ba mảng màu:

1. **Tầng phòng thủ & tiền xử lý (Safety & Ingestion)** — `ingest → guard`; HIGH severity cô lập
   sang `quarantine`, còn lại sang `extract`, thiếu trường thì `repair` (tối đa 2 vòng).
2. **Tầng lọc sớm & định lượng (Fast-Gating & Deterministic Scoring)** — `load_rubric →
   must_have_check`; thiếu tiêu chí tiên quyết thì `reject_fast`. Đây là nơi 5 tool tất định chạy.
3. **Tầng đánh giá sâu & vùng xám (Rubric & Grey-zone Resolution)** — `score_criteria →
   aggregate`; điểm rơi vùng tranh chấp thì `deep_review` trước `decide → rank`.

Một dòng về rubric, vì nó là quyết định thiết kế đáng nói: **JD trở thành dữ liệu, không trở
thành prompt** — YAML gồm tiêu chí + trọng số + cờ `must_have` + hai ngưỡng cắt (spec §6).

Với `reject_fast`, ghi rõ ngay trên slide: agent **không loại ứng viên**, nó phân vào ba luồng;
`reject_fast` vẫn **ghi lại điểm mọi tiêu chí khác** để giao diện giải thích được (spec §2) —
đó chính là chỗ Article 14 sống được.

### 7. Năm tool tất định
**Five deterministic tools**

Bảng 3 cột: tool · làm gì (cơ chế thật) · **vì sao không để LLM làm**. Dùng đúng cơ chế:

| Tool | Cơ chế | Vì sao không để LLM làm |
|---|---|---|
| `calculate_experience` | Parse khoảng ngày trên văn bản chuẩn hóa, hợp các khoảng chồng lấn, đếm một lần | 198/290 CV dev có khoảng chồng lấn — cộng dồn naive sai trên hai phần ba CV thật |
| `normalize_skill` | Bảng alias YAML → khóa canonical để so sánh + surface form để đi tìm trong CV | Biến so chuỗi thành so khái niệm (`React` = `ReactJS` = `React.js`), vẫn giữ `C#` khác `C` |
| `search_evidence` | Khớp chính xác trên bản chuẩn hóa; không có thì cửa sổ trượt cho điểm bằng `difflib.SequenceMatcher`, ngưỡng 0.75; quote trả về là **slice nguyên văn** của CV gốc | Nền tảng của E1: điểm chỉ bảo vệ được khi chỉ ra được đúng dãy ký tự sinh ra nó |
| `scan_injection` | Regex trên bản chuẩn hóa; HIGH → `quarantine`, LOW → ghi chú | CV là input không tin cậy được dán thẳng vào context của model |
| `aggregate_scorecard` | Tổng có trọng số, tất định | Không để model tự cộng điểm của chính nó |

Thêm một chi tiết nhỏ mà người phản biện sẽ thích: một rule ứng viên đã **bị loại bỏ** vì
false-positive — `act\s+as\s+(a|an|the)` khớp 3 CV thật ("act as a liaison"), và docstring ghi
"do not re-add it". Đây là bằng chứng rằng rule được **đo**, không phải được đoán.

### 8. Walkthrough — một hồ sơ, từng chặng
**One CV, node by node**

Hình `04-state-walkthrough.png` lớn. Với mỗi chặng, ghi state **vào** và state **ra**:

- `ingest → guard`: `scan_injection` không thấy gì → đi `extract`.
- `extract`: model lấy ra skills / work_periods / degrees. Nếu hàng này đi qua `repair` thì nói
  rõ trường nào thiếu và vòng repair lấp gì; nếu không, **đừng dựng ra**.
- `load_rubric`: JD đã thành YAML có trọng số và ngưỡng.
- `must_have_check`: đạt tiêu chí tiên quyết → `score_criteria`.
- `score_criteria`: `expand_skill` sinh surface form → `search_evidence` tìm trong CV →
  `calculate_experience` đặt điểm cho tiêu chí kinh nghiệm.
- `aggregate`: tổng có trọng số; nếu rơi vùng xám thì `deep_review`.
- `decide → rank`: nhãn cuối + scorecard.

Ô phụ nhỏ, nhãn rõ **"hàng khác"**: một hàng đi `repair` (1.4%) và một hàng đi `reject_fast`
(24.2%).

### 9. Đầu ra — bằng chứng gắn vào điểm
**Output: evidence attached to the score**

Hình `05-scorecard-evidence.png`: mỗi tiêu chí một dòng — tên tiêu chí, trọng số, điểm, và
**trích dẫn nguyên văn kèm offset ký tự** trong CV gốc.

Rồi nói thẳng con số và giới hạn của nó, trên cùng slide:

- Evidence coverage (E1) = **0.486** trên `test_500`. Không phải 100%.
- Vì sao: theo thiết kế, tiêu chí chấm **0.0 trả về không kèm trích dẫn**. Hệ quả thật là
  **"CV im lặng về tiêu chí" và "CV không đạt tiêu chí" đến `aggregate_scorecard` như cùng một
  con số 0.0** (`scoring_layer.md`).
- Đối chiếu: **cả ba baseline đều 0.000** — một lần gọi LLM không thể chỉ vào câu nào trong CV,
  vì nó chưa từng sinh ra câu nào.

Đưa luôn giới hạn lên slide chứ đừng để trong notes. Người phản biện đã đọc được 0.486 thì thà
mình nói trước.

### 10. Benchmark — agent so với ba baseline
**Benchmark against three baselines**

Hình `06-baseline-bars.png` và `07-confusion-test500.png`. Bảng lấy đúng từ `baselines.md`,
caption ghi rõ **`dev_300`, cùng 300 hàng, cùng hàm chấm trong `eval/metrics.py`**:

| System | macro-F1 | Good Fit recall | Tokens | Evidence (E1) |
|---|---:|---:|---:|---:|
| `agent` | 0.4155 | 0.176 | 1.142.849 | 0.439 |
| `baseline_naive` | 0.3742 | 0.041 | 521.102 | 0.000 |
| `baseline_rubric` | 0.3750 | 0.041 | 453.321 | 0.000 |
| `baseline_tfidf` | 0.3673 | 0.419 | 0 | 0.000 |

Rồi **tự nói ra hai điều bất lợi**, mỗi điều một dòng đậm:

- Khoảng cách với baseline mạnh nhất chỉ **+0.04 macro-F1**, đổi bằng **2,5 lần token**.
- **TF-IDF tìm `Good Fit` giỏi hơn agent**: recall 0.419 so với 0.176, không tốn token, kỹ thuật
  từ 1972. Toàn bộ phần agent hơn nằm ở `No Fit`.

Kèm ô riêng: `test_500`, chạy **một lần**, macro-F1 **0.3939**, evidence coverage **0.486** —
ghi rõ đây là tập khác và baseline chưa chạy trên đó.

Speaker notes: đây chính là tình huống spec §7 đã dự trù. Khi baseline mạnh sát agent, giá trị
của agent nằm ở truy vết bằng chứng, xử lý CV lỗi, chặn can thiệp và chi phí — và cột cuối bảng
là 0.439 so với 0.000.

### 11. Ordering — chỗ hệ thống thật sự hỏng
**Where the ordering breaks**

Hình `08-score-distribution.png`, số từ `scoring_layer.md`:

- Median điểm của `Potential Fit` thật (**0.490**) **cao hơn** median của `Good Fit` thật
  (**0.350**) → giữa hai lớp này thứ tự không yếu, mà **bị đảo**.
- AUC `Good Fit` trên `Potential Fit`: **0.482**, bootstrap 95% **[0.386, 0.575]**.
- Trần trên: một classifier có giám sát học trên văn bản thô, cross-validate chia fold theo JD,
  chỉ đạt **0.577** — dùng chính nhãn mà agent không bao giờ thấy.
- Chỉnh lại hai ngưỡng cắt được nhiều nhất **+0.0210 macro-F1**, và con số đó đã overfit.
- 87,7% điểm tiêu chí đúng bằng 0.0, 0.5 hoặc 1.0 — model đang trả lời không / có thể / có,
  chứ không dùng thang 0..1.
- **Power**: `dev_300` chỉ có power **0.37**; cần khoảng **436 hàng** contested mới đạt 0.80.
  Không split nào của dự án có đủ.

Kết luận của slide, và là bài học chuyển giao được: **một thay đổi không đo được thì không nên
làm chỉ vì có một câu chuyện hay về lý do nó nên hiệu quả.**

### 12. Lưu lượng nhánh & đóng góp từng module
**Branch traffic and per-module contribution**

Hình `09-branch-traffic.png`. Mỗi ablation một dòng, delta ghi kèm dấu:

| Ablation | Đo được | Đọc thế nào |
|---|---|---|
| `must_have_gate` | `shipped` 0.3721 vs tắt cổng 0.4052 (**−0.0331**), tiết kiệm 208.127 token; `test_500` loại 121 hàng, precision **0.645**, trong đó **28 hàng nhãn thật là `Good Fit`** | Cổng này **đổi độ chính xác lấy token**, không phải cải thiện miễn phí |
| `no_gray_zone` | `shipped` 0.4155 vs 0.4014 (**+0.0140**), 20/300 dự đoán đổi | `deep_review` có đóng góp, nhỏ và đo được |
| `no_guard` | HIGH: 6/6 bị `quarantine`; tắt guard thì cả 6 chấm **đúng 0.550 — y như hàng sạch**. LOW: 0/2 đổi | Guard **từ chối xử lý tài liệu đang cố can thiệp**; trên fixture này nó **không** ngăn một can thiệp lẽ ra thành công |

**Bắt buộc in trên slide:** các hàng `no_guard` là **tổng hợp (synthetic)**, vì 0/300 CV thật
kích hoạt rule nào. Đây là yêu cầu của chính `no_guard.md`.

Và slide này phải nối lại với slide 3: cổng `must_have` **tái tạo đúng cái lỗi HBS mô tả** —
loại người vì không khớp tiêu chí nguyên văn. Đừng để người phản biện ghép hai slide đó lại giúp
bạn. Cách nói đúng: hệ thống **đo được** cái giá đó (precision 0.645, 28 `Good Fit` bị loại), và
cổng là một **công tắc ablation** có thể tắt — biết giá là điều kiện để chọn.

### 13. Kết luận, bài học nghiên cứu & QA
**Conclusions, research lessons & QA**

Hai cột. Cột trái — đã chứng minh được gì:

- **E1 đạt và đo được**: 0.486 coverage so với 0.000 của mọi baseline một lần gọi.
- **E2 chỉ đạt một phần**: +0.04 macro-F1 với 2,5 lần token; phần giá trị thật nằm ở truy vết,
  chi phí đường tắt, và xử lý CV lỗi.
- Tái lập: `temperature=0`, cache theo hash `(model, prompt, input)` ra JSONL, log token +
  latency từng nhánh, **mọi số trên slide sinh từ một lệnh duy nhất** (spec §8).

Cột phải — bài học mang về được (đây là phần "xây dựng" của bài nói):

1. **Nhánh không có số đo là nhánh trang trí.** Bốn conditional edge, bốn con số lưu lượng thật.
2. **Đừng bắt LLM làm việc của thuật toán.** 198/290 CV có khoảng chồng lấn là lý do
   `calculate_experience` tồn tại — một con số đo được, không phải trực giác.
3. **Đo baseline mạnh trước khi mừng.** Prompt rubric đầu tiên sập về `No Fit` trên 297/300 hàng
   → macro-F1 0.2405; nếu báo cáo con số đó, agent nhận một chiến thắng không xứng đáng. Nay
   `eval/report.py` tự cảnh báo mọi run có 95% cùng một nhãn.
4. **Tách bất ổn khỏi thiên lệch.** Counterfactual: nhánh trường học động 72/261 cặp, đổi nhãn
   24, sign test **p = 0.024** → các dịch chuyển **có cùng hướng**; nhánh danh tính p = 0.635 →
   không có hướng ở cỡ mẫu này. Mean delta gần 0 **không** chứng minh công bằng — nó cũng có thể
   là các dịch chuyển lớn triệt tiêu nhau.
5. **Đo trước khi mô tả.** Docstring từng nói `extraction_confidence` "không bao giờ xuống dưới
   0.90" và "không mang thông tin". Đo ra: nó xuống 0.00 trên 23,8% lần trích xuất, chỉ nhận
   **4 giá trị**, và nó **có** thông tin — nhưng **dư thừa**, vì `total_experience_years is None`
   trả lời đúng câu hỏi đó, tất định và miễn phí. Nên: giữ trong contract, không route trên nó,
   và có một test giữ nó ngoài luồng điều khiển.
6. **Cố ý không làm** (spec §11): OCR, tiếng Việt trong luồng đo, vector DB / GraphRAG, tự động
   loại ứng viên, tự gán nhãn gold.
7. **Việc tiếp theo, đã biết cỡ**: cần ~436 hàng contested để trả lời câu hỏi ordering; lấy thêm
   dữ liệu là bước kế, không phải chỉnh thêm ngưỡng.

Kết bằng một câu: **"Chúng tôi không có một hệ thống chính xác. Chúng tôi có một hệ thống biết
mình sai ở đâu, sai bao nhiêu, và đo được điều đó."** Rồi slide QA.

## Sau khi xong

1. Lưu file vào `docs/slides/cv-screener-agent.html` (tạo thư mục nếu chưa có).
2. Cho tôi biết cách mở và cách xuất PDF.
3. In bảng "slot ảnh → tên PNG mong đợi + tỉ lệ khung hình".
4. **In danh sách mọi con số đã đưa lên slide kèm file nguồn**, để tôi đối chiếu lại một lượt.
5. Nếu có chỗ nào bạn muốn nói mà không tìm được số trong repo, **liệt kê riêng phần đó ra** thay
   vì điền số vào slide.
