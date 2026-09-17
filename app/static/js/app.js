/**
 * HR CV Screener Agent — Client Application Logic
 * Implements SSE real-time streaming, interactive evidence highlighting, and responsive UI.
 */

// Node descriptions for the live stepper
const NODE_EXPLANATIONS = {
  ingest: "Kiểm tra ranh giới văn bản và tính toàn vẹn ký tự, bảo toàn vị trí byte chính xác.",
  guard: "Quét CV tìm Prompt Injection và payload độc hại bằng các luật deterministic.",
  quarantine: "Phát hiện chỉ thị tấn công độc hại! Cách ly (Quarantine) tài liệu để bảo vệ LLM context.",
  extract: "Trích xuất hồ sơ ứng viên có cấu trúc (kỹ năng, lịch sử làm việc, bằng cấp).",
  repair: "Tự động sửa chữa các trường dữ liệu hồ sơ vi phạm schema ràng buộc.",
  load_rubric: "Nạp và gán trọng số các tiêu chí chấm điểm từ bản mô tả công việc (JD).",
  must_have_check: "Đánh giá các điều kiện tiên quyết (Must-Have) trước khi tính điểm chi tiết.",
  reject_fast: "Ứng viên thiếu yêu cầu Must-Have bắt buộc. Kích hoạt Fast Reject để tiết kiệm token.",
  score_criteria: "Thực thi các deterministic tools (tính năm kinh nghiệm, chuẩn hóa kỹ năng, trích xuất evidence) & chấm điểm.",
  aggregate: "Tổng hợp điểm trọng số vào Scorecard và kiểm tra ngưỡng vùng xám (Gray Zone).",
  deep_review: "Điểm ứng viên nằm trong Gray Zone. Thực hiện phản biện chuyên sâu đa góc nhìn.",
  decide: "Phân loại FitLabel cuối cùng dựa trên các ngưỡng điểm đã hiệu chỉnh.",
  rank: "Hợp nhất bảng xếp hạng ứng viên và tổng kết chỉ số telemetry.",
};

// Application State
let currentPresets = [];
let latestScreeningState = null;
let currentEvidences = [];
let currentInjectionSpans = [];
let selectedCriterionId = "all";
// The exact string the server received. Evidence offsets are relative to it, so
// highlighting `cvTextEl.value` instead would shift every mark by the leading
// whitespace `trim()` removed.
let currentCvText = "";
let streamFailed = false;

// DOM Elements
const presetSelect = document.getElementById("presetSelect");
const scenarioBox = document.getElementById("scenarioBox");
const scenarioDesc = document.getElementById("scenarioDesc");
const expectedBranch = document.getElementById("expectedBranch");
const cvTextEl = document.getElementById("cvText");
const jdTextEl = document.getElementById("jdText");
const rubricSelect = document.getElementById("rubricSelect");
const btnRunScreen = document.getElementById("btnRunScreen");
const pipelineStatusBadge = document.getElementById("pipelineStatusBadge");
const stepperContainer = document.getElementById("stepperContainer");

const valLatency = document.getElementById("valLatency");
const valTokens = document.getElementById("valTokens");
const valLlmCalls = document.getElementById("valLlmCalls");

const verdictBanner = document.getElementById("verdictBanner");
const verdictIcon = document.getElementById("verdictIcon");
const verdictLabel = document.getElementById("verdictLabel");
const verdictScore = document.getElementById("verdictScore");
const rejectedReasonAlert = document.getElementById("rejectedReasonAlert");
const rejectedReasonText = document.getElementById("rejectedReasonText");

const outputTabNav = document.getElementById("outputTabNav");
const scorecardList = document.getElementById("scorecardList");
const cvDocumentViewer = document.getElementById("cvDocumentViewer");
const evidenceQuotesList = document.getElementById("evidenceQuotesList");
const criterionFilterSelect = document.getElementById("criterionFilterSelect");
const outputEmptyState = document.getElementById("outputEmptyState");
const btnExportJson = document.getElementById("btnExportJson");

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  initSettingsModal();
  initTabs();
  initUploadDropzone();
  fetchPresets();

  btnRunScreen.addEventListener("click", handleRunScreening);
  presetSelect.addEventListener("change", handlePresetChange);
  criterionFilterSelect.addEventListener("change", handleCriterionFilterChange);
  btnExportJson.addEventListener("click", handleExportJson);
});

// ============================================================================
// 1. Settings & Modal Controls
// ============================================================================
function initSettingsModal() {
  const btnOpen = document.getElementById("btnOpenSettings");
  const btnClose = document.getElementById("btnCloseSettings");
  const btnSave = document.getElementById("btnSaveSettings");
  const modal = document.getElementById("settingsModal");

  btnOpen.addEventListener("click", () => modal.classList.add("open"));
  btnClose.addEventListener("click", () => modal.classList.remove("open"));
  btnSave.addEventListener("click", () => modal.classList.remove("open"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("open");
  });
}

// ============================================================================
// 2. Tabs & Navigation
// ============================================================================
function activateTab(tabId) {
  if (!outputTabNav) return;
  const tabBtns = outputTabNav.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    if (btn.getAttribute("data-tab") === tabId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  document.querySelectorAll(".tab-content").forEach((content) => {
    content.style.display = content.id === tabId ? "block" : "none";
  });
}

function initTabs() {
  // Input CV tabs (Text vs Upload)
  const tabCvText = document.getElementById("tabCvText");
  const tabCvUpload = document.getElementById("tabCvUpload");
  const cvTextInputWrapper = document.getElementById("cvTextInputWrapper");
  const cvUploadWrapper = document.getElementById("cvUploadWrapper");

  tabCvText.addEventListener("click", () => {
    tabCvText.classList.add("active");
    tabCvUpload.classList.remove("active");
    cvTextInputWrapper.style.display = "block";
    cvUploadWrapper.style.display = "none";
  });

  tabCvUpload.addEventListener("click", () => {
    tabCvUpload.classList.add("active");
    tabCvText.classList.remove("active");
    cvTextInputWrapper.style.display = "none";
    cvUploadWrapper.style.display = "block";
  });

  // Output Tabs (Scorecard, Evidence, Trace, E2)
  const tabBtns = outputTabNav.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      activateTab(targetId);
    });
  });
}

// ============================================================================
// 3. Presets & Upload Handling
// ============================================================================
async function fetchPresets() {
  try {
    const res = await fetch("/api/presets");
    currentPresets = await res.json();
    presetSelect.innerHTML = '<option value="">-- Nhập tùy chỉnh (Dán hoặc Tải file) --</option>';
    currentPresets.forEach((caseItem) => {
      const opt = document.createElement("option");
      opt.value = caseItem.id;
      opt.textContent = caseItem.title;
      presetSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to fetch presets:", err);
  }
}

function handlePresetChange() {
  const selectedId = presetSelect.value;
  const found = currentPresets.find((c) => c.id === selectedId);

  if (found) {
    scenarioBox.style.display = "block";
    scenarioDesc.textContent = found.scenario_desc;
    expectedBranch.textContent = found.expected_branch;
    cvTextEl.value = found.cv_text;
    jdTextEl.value = found.jd_text;
    document.getElementById("presetBadge").textContent = "Đã nạp Preset";
    rubricSelect.value = "backend_engineer";
  } else {
    scenarioBox.style.display = "none";
    document.getElementById("presetBadge").textContent = "Tùy chỉnh";
  }
}

function initUploadDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const uploadStatus = document.getElementById("uploadStatus");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      await uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", async () => {
    if (fileInput.files.length > 0) {
      await uploadFile(fileInput.files[0]);
    }
  });

  async function uploadFile(file) {
    uploadStatus.textContent = `Đang tải lên & trích xuất ${file.name}...`;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      cvTextEl.value = data.text;
      uploadStatus.textContent = `✓ Đã trích xuất ${data.char_count} ký tự từ ${data.filename}`;
      uploadStatus.style.color = "#059669";
    } catch (err) {
      uploadStatus.textContent = `Lỗi: ${err.message}`;
      uploadStatus.style.color = "#DC2626";
    }
  }
}

// ============================================================================
// 4. Run Screening & SSE Real-Time Streaming
// ============================================================================
async function handleRunScreening() {
  const cvText = cvTextEl.value.trim();
  const jdText = jdTextEl.value.trim();
  currentCvText = cvText;
  streamFailed = false;

  if (!cvText || !jdText) {
    alert("Vui lòng cung cấp đầy đủ cả CV ứng viên và Mô tả công việc (JD).");
    return;
  }

  // Read settings
  const guard = document.getElementById("toggleGuard").checked;
  const mustHaveGate = document.getElementById("toggleMustHave").checked;
  const grayZone = document.getElementById("toggleGrayZone").checked;
  const modelName = document.getElementById("modelSelect").value;
  const apiKey = document.getElementById("apiKeyInput").value.trim();
  const rubricPreset = rubricSelect.value === "backend_engineer" ? "backend_engineer" : null;

  // Prepare UI for execution
  btnRunScreen.disabled = true;
  btnRunScreen.innerHTML = "<span>⏳ Agent đang thực thi...</span>";
  pipelineStatusBadge.textContent = "Đang chạy";
  pipelineStatusBadge.className = "badge-tag indigo";

  stepperContainer.innerHTML = "";
  if (outputEmptyState) outputEmptyState.style.display = "none";
  verdictBanner.style.display = "none";
  rejectedReasonAlert.style.display = "none";
  outputTabNav.style.display = "flex";
  activateTab("traceTab");

  valLatency.textContent = "0 ms";
  valTokens.textContent = "0";
  valLlmCalls.textContent = "0";

  // Immediately render active starting step
  renderStepper([], "ingest", false);

  const requestBody = {
    cv_text: cvText,
    jd_text: jdText,
    rubric_preset: rubricPreset,
    guard: guard,
    must_have_gate: mustHaveGate,
    gray_zone: grayZone,
    model_name: modelName,
    api_key: apiKey || null,
  };

  try {
    const response = await fetch("/api/screen/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // Keep partial chunk

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6);
          try {
            const eventData = JSON.parse(jsonStr);
            handleStreamEvent(eventData);
          } catch (e) {
            console.error("Failed to parse SSE line:", line, e);
          }
        }
      }
    }

    if (!streamFailed) {
      pipelineStatusBadge.textContent = "Hoàn thành";
      pipelineStatusBadge.className = "badge-tag";
    }
    if (latestScreeningState) {
      renderStepper(latestScreeningState.node_traces, null, true);
    }
  } catch (err) {
    console.error("Streaming error:", err);
    pipelineStatusBadge.textContent = "Thất bại";
    pipelineStatusBadge.className = "badge-tag";
    alert(`Thực thi Screener Agent thất bại: ${err.message}`);
  } finally {
    btnRunScreen.disabled = false;
    btnRunScreen.innerHTML = "<span>🚀 Chạy Screener Agent</span>";
  }
}

function handleStreamEvent(event) {
  if (event.type === "step") {
    latestScreeningState = event;
    updateTelemetry(event.node_traces);
    renderStepper(event.node_traces, event.next_node, false);

    if (event.result) {
      renderFinalDecision(event);
    }
  } else if (event.type === "done") {
    if (latestScreeningState) {
      renderStepper(latestScreeningState.node_traces, null, true);
      if (latestScreeningState.result) {
        renderFinalDecision(latestScreeningState);
      }
    }
  } else if (event.type === "error") {
    // The stream itself succeeded, so the caller's catch never fires; without this
    // the badge would still settle on "Hoàn thành" for a run that failed.
    streamFailed = true;
    pipelineStatusBadge.textContent = "Thất bại";
    pipelineStatusBadge.className = "badge-tag";
    alert(`Agent Execution Error: ${event.message}`);
  }
}

// ============================================================================
// 5. Telemetry & Stepper Timeline Rendering
// ============================================================================
function updateTelemetry(nodeTraces) {
  if (!nodeTraces) return;
  const totalLatency = nodeTraces.reduce((sum, t) => sum + (t.latency_ms || 0), 0);
  const totalTokens = nodeTraces.reduce((sum, t) => sum + (t.prompt_tokens || 0) + (t.completion_tokens || 0), 0);
  const totalLlmCalls = nodeTraces.reduce((sum, t) => sum + (t.llm_calls || 0), 0);
  valLatency.textContent = `${Math.round(totalLatency)} ms`;
  valTokens.textContent = totalTokens.toLocaleString();
  valLlmCalls.textContent = totalLlmCalls;
}

function renderStepper(nodeTraces, nextNode, isDone = false) {
  if (!nodeTraces) nodeTraces = [];

  stepperContainer.innerHTML = "";

  // 1. Render all completed nodes in nodeTraces
  nodeTraces.forEach((trace, idx) => {
    const node = trace.node;
    const explanation = NODE_EXPLANATIONS[node] || "Processing node logic...";

    let statusTag = "✓ HOÀN THÀNH";
    let extraClass = "completed";

    if (node === "quarantine") {
      extraClass = "shortcut-quarantine";
      statusTag = "🛡️ QUARANTINE";
    } else if (node === "reject_fast") {
      extraClass = "shortcut-reject";
      statusTag = "⚡ FAST REJECT";
    } else if (node === "deep_review") {
      extraClass = "shortcut-review";
      statusTag = "🔍 DEEP REVIEW";
    }

    const card = document.createElement("div");
    card.className = `step-card ${extraClass}`;

    const llmChip = trace.llm_calls > 0 ? `<span class="chip">🤖 ${trace.llm_calls} LLM</span>` : '<span class="chip">⚙️ Deterministic</span>';
    const tokenSum = (trace.prompt_tokens || 0) + (trace.completion_tokens || 0);
    const tokenChip = tokenSum > 0 ? `<span class="chip">🪙 ${tokenSum} tok</span>` : "";
    const latencyChip = `<span class="chip">⏱️ ${Math.round(trace.latency_ms)}ms</span>`;
    const noteHtml = trace.note ? `<div class="step-note">${escapeHtml(trace.note)}</div>` : "";

    card.innerHTML = `
      <div class="step-header">
        <div class="step-title-group">
          <div class="step-num-badge">${idx + 1}</div>
          <div class="step-node-name">${escapeHtml(node)}</div>
        </div>
        <span style="font-size: 0.72rem; font-weight: 700;">${statusTag}</span>
      </div>
      <div class="step-desc">${explanation}</div>
      <div class="step-metrics">
        ${latencyChip}
        ${tokenChip}
        ${llmChip}
      </div>
      ${noteHtml}
    `;

    stepperContainer.appendChild(card);
  });

  // 2. If the pipeline is running and there is a next active node, render RUNNING card
  if (!isDone && nextNode) {
    const runningCard = document.createElement("div");
    runningCard.className = "step-card active";
    const explanation = NODE_EXPLANATIONS[nextNode] || "Đang thực thi logic node của Agent...";

    runningCard.innerHTML = `
      <div class="step-header">
        <div class="step-title-group">
          <div class="step-num-badge">${nodeTraces.length + 1}</div>
          <div class="step-node-name">${escapeHtml(nextNode)}</div>
        </div>
        <span style="font-size: 0.72rem; font-weight: 700; color: var(--primary);">⏳ ĐANG CHẠY</span>
      </div>
      <div class="step-desc">${explanation}</div>
      <div class="step-metrics">
        <span class="chip">Đang xử lý...</span>
      </div>
    `;

    stepperContainer.appendChild(runningCard);
  }

  // Auto scroll stepper to bottom
  stepperContainer.scrollTop = stepperContainer.scrollHeight;
}

// ============================================================================
// 6. Decision & Output Panel
// ============================================================================
function renderFinalDecision(state) {
  const res = state.result;
  if (!res) return;

  // With `ablations.guard` off, a poisoned CV is flagged *and* scored all the way
  // through. `quarantined` only records the detection; `path_taken` records whether
  // the graph actually stopped at the quarantine node.
  const stoppedAtQuarantine = (state.path_taken || []).includes("quarantine");

  outputEmptyState.style.display = "none";
  outputTabNav.style.display = "flex";
  btnExportJson.style.display = "inline-flex";

  // 1. Verdict Banner
  verdictBanner.style.display = "flex";
  let bannerClass = "nofit";
  let bannerIcon = "✕";
  let labelTitle = res.label;

  if (stoppedAtQuarantine) {
    bannerClass = "quarantined";
    bannerIcon = "🛡️";
    labelTitle = "Quarantined";
  } else if (res.label === "Good Fit") {
    bannerClass = "good";
    bannerIcon = "✓";
  } else if (res.label === "Potential Fit") {
    bannerClass = "potential";
    bannerIcon = "⚡";
  }

  verdictBanner.className = `verdict-banner ${bannerClass}`;
  verdictIcon.textContent = bannerIcon;
  verdictLabel.textContent = labelTitle;
  verdictScore.textContent = `${(res.overall_score * 100).toFixed(1)}% Match`;

  // Rejected reason
  if (res.rejected_reason) {
    rejectedReasonAlert.style.display = "block";
    rejectedReasonText.textContent = res.rejected_reason;
  } else {
    rejectedReasonAlert.style.display = "none";
  }

  // 2. Scorecard Tab
  renderScorecardTab(res.criterion_scores, state);

  // 3. Evidence Tab
  currentEvidences = [];
  if (res.criterion_scores) {
    res.criterion_scores.forEach((cs) => {
      if (cs.evidence) {
        cs.evidence.forEach((ev) => currentEvidences.push({ ...ev, criterion_id: cs.criterion_id, type: "evidence" }));
      }
    });
  }

  // Check injection spans if quarantined
  currentInjectionSpans = [];
  if (state.quarantined && state.injection_findings && state.injection_findings.length > 0) {
    state.injection_findings.forEach((f) => {
      if (f.evidence) {
        currentInjectionSpans.push({
          start: f.evidence.start,
          end: f.evidence.end,
          quote: f.evidence.quote,
          criterion_id: `Security: ${f.rule_id}`,
          type: "injection",
        });
      }
    });
  }

  renderEvidenceFilterOptions(res.criterion_scores, currentInjectionSpans);

  if (state.quarantined && currentInjectionSpans.length > 0) {
    selectedCriterionId = "injection";
    criterionFilterSelect.value = "injection";
  } else {
    selectedCriterionId = "all";
  }

  renderHighlightedCV(currentCvText, currentEvidences, currentInjectionSpans, selectedCriterionId);

  // 4. E2 Value Tab
  renderE2Tab(state);

  // Open Scorecard tab by default, or Evidence tab if the run stopped at quarantine
  if (stoppedAtQuarantine) {
    activateTab("evidenceTab");
  } else {
    activateTab("scorecardTab");
  }
}

function renderScorecardTab(criterionScores, state) {
  scorecardList.innerHTML = "";

  // Only a run that really stopped has no scores to show; a guard-ablated run has
  // a full scorecard and discarding it would hide the ablation's whole point.
  if (state && (state.path_taken || []).includes("quarantine")) {
    const flagsStr = state.injection_flags ? state.injection_flags.join(", ") : "detected";
    scorecardList.innerHTML = `
      <div class="scenario-info-box" style="background-color: #F5F3FF; border-color: #DDD6FE; color: #5B21B6;">
        🛡️ <strong>Cách ly tại Guard Layer (Quarantined):</strong>
        <div style="font-size: 0.78rem; margin-top: 0.25rem;">
          Tài liệu chứa hành vi Prompt Injection (<code>${escapeHtml(flagsStr)}</code>). Tiến trình chấm điểm downstream đã được dừng để ngăn chặn đầu độc LLM.
        </div>
      </div>
    `;
    return;
  }

  if (!criterionScores || criterionScores.length === 0) {
    scorecardList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">Không có điểm số tiêu chí (do kích hoạt đường tắt / shortcut trong pipeline).</div>';
    return;
  }

  criterionScores.forEach((cs) => {
    const card = document.createElement("div");
    card.className = "criterion-card";
    const scorePct = Math.round(cs.score * 100);

    card.innerHTML = `
      <div class="criterion-header">
        <span class="criterion-name">${escapeHtml(cs.criterion_id)}</span>
        <span class="criterion-score-badge">${scorePct}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width: ${scorePct}%;"></div>
      </div>
      <div class="criterion-reason">${escapeHtml(cs.reasoning || "")}</div>
      ${cs.tool_used ? `<div style="font-size: 0.72rem; color: #4F46E5; margin-top: 0.35rem;">🛠️ Đánh giá bởi tool: <code>${escapeHtml(cs.tool_used)}</code></div>` : ""}
    `;

    // Clicking a criterion jumps to evidence tab and filters highlight
    card.addEventListener("click", () => {
      document.querySelectorAll(".criterion-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");

      // Switch to evidence tab
      outputTabNav.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      outputTabNav.querySelector('[data-tab="evidenceTab"]').classList.add("active");
      document.querySelectorAll(".tab-content").forEach((c) => (c.style.display = "none"));
      document.getElementById("evidenceTab").style.display = "block";

      criterionFilterSelect.value = cs.criterion_id;
      selectedCriterionId = cs.criterion_id;
      renderHighlightedCV(currentCvText, currentEvidences, currentInjectionSpans, selectedCriterionId);
    });

    scorecardList.appendChild(card);
  });
}

function renderEvidenceFilterOptions(criterionScores, injectionSpans) {
  criterionFilterSelect.innerHTML = '<option value="all">-- Tất cả bằng chứng (Evidence) --</option>';

  if (injectionSpans && injectionSpans.length > 0) {
    const injOpt = document.createElement("option");
    injOpt.value = "injection";
    injOpt.textContent = `🛡️ Đoạn Prompt Injection (${injectionSpans.length})`;
    criterionFilterSelect.appendChild(injOpt);
  }

  if (!criterionScores) return;

  criterionScores.forEach((cs) => {
    const opt = document.createElement("option");
    opt.value = cs.criterion_id;
    opt.textContent = `${cs.criterion_id} (${cs.evidence ? cs.evidence.length : 0} trích dẫn)`;
    criterionFilterSelect.appendChild(opt);
  });
}

function handleCriterionFilterChange() {
  selectedCriterionId = criterionFilterSelect.value;
  renderHighlightedCV(currentCvText, currentEvidences, currentInjectionSpans, selectedCriterionId);
}

function renderHighlightedCV(cvText, evidences, injectionSpans, filterCritId) {
  if (!cvText) {
    cvDocumentViewer.innerHTML = "<em>Chưa có nội dung văn bản CV.</em>";
    evidenceQuotesList.innerHTML = "";
    return;
  }

  // Filter relevant spans
  let candidateSpans = [];

  if (filterCritId === "injection") {
    candidateSpans = [...injectionSpans];
  } else if (filterCritId && filterCritId !== "all") {
    candidateSpans = evidences.filter((ev) => ev.criterion_id === filterCritId);
  } else {
    candidateSpans = [...evidences, ...injectionSpans];
  }

  // Collect quotes list
  if (candidateSpans.length > 0) {
    evidenceQuotesList.innerHTML = `<strong>Các đoạn trích dẫn được Highlight (${candidateSpans.length}):</strong><br>` +
      candidateSpans.map((ev) => {
        const badge = ev.type === "injection" ? '<span style="color: #DC2626; font-weight: 700;">[AN TOÀN BẢO MẬT]</span> ' : "";
        return `• ${badge}<em>"${escapeHtml(ev.quote)}"</em> &nbsp; <code>[ký tự: ${ev.start}-${ev.end}]</code>`;
      }).join("<br>");
  } else {
    evidenceQuotesList.innerHTML = "<em>Không tìm thấy đoạn trích dẫn evidence nào cho tiêu chí này.</em>";
  }

  // Sort spans by start offset and remove overlaps
  const sorted = [...candidateSpans].sort((a, b) => a.start - b.start);
  const cleanSpans = [];
  let lastEnd = 0;
  for (const s of sorted) {
    if (s.start >= lastEnd && s.end <= cvText.length && s.start < s.end) {
      cleanSpans.push(s);
      lastEnd = s.end;
    }
  }

  // Build segmented HTML string
  let html = "";
  let curr = 0;
  for (const span of cleanSpans) {
    if (span.start > curr) {
      html += escapeHtml(cvText.slice(curr, span.start));
    }
    const quoteText = escapeHtml(cvText.slice(span.start, span.end));
    const cssClass = span.type === "injection" ? "injection-highlight" : "evidence-highlight";
    html += `<mark class="${cssClass}" title="${escapeHtml(span.criterion_id)}">${quoteText}</mark>`;
    curr = span.end;
  }

  if (curr < cvText.length) {
    html += escapeHtml(cvText.slice(curr));
  }

  cvDocumentViewer.innerHTML = html;
}

function renderE2Tab(state) {
  const e2LlmExp = document.getElementById("e2LlmExp");
  const e2ToolExp = document.getElementById("e2ToolExp");
  const e2ExpDiff = document.getElementById("e2ExpDiff");
  const e2SkillsBox = document.getElementById("e2SkillsBox");
  const e2ShortcutBox = document.getElementById("e2ShortcutBox");

  if (state.profile) {
    const llmYears = state.profile.llm_declared_years != null ? state.profile.llm_declared_years.toFixed(1) : "--";
    const toolYears = state.profile.total_experience_years != null ? state.profile.total_experience_years.toFixed(1) : "--";
    e2LlmExp.textContent = `${llmYears} năm`;
    e2ToolExp.textContent = `${toolYears} năm`;

    if (state.profile.llm_declared_years != null && state.profile.total_experience_years != null) {
      const diff = state.profile.total_experience_years - state.profile.llm_declared_years;
      e2ExpDiff.textContent = diff !== 0 ? `Chênh lệch: ${diff > 0 ? "+" : ""}${diff.toFixed(1)} năm` : "Trùng khớp";
    }

    if (state.profile.skills && state.profile.skills.length > 0) {
      e2SkillsBox.innerHTML = state.profile.skills.map((s) => `<span class="chip" style="color: #4F46E5;">${escapeHtml(s)}</span>`).join(" ");
    } else {
      e2SkillsBox.innerHTML = '<span style="color: var(--text-muted); font-size: 0.78rem;">Không trích xuất được kỹ năng chuẩn hóa nào.</span>';
    }
  }

  // Shortcut token efficiency
  if (state.path_taken.includes("reject_fast")) {
    e2ShortcutBox.innerHTML = `
      <div class="scenario-info-box" style="background-color: #FEF2F2; border-color: #FECACA; color: #991B1B;">
        ⚡ <strong>Kích hoạt nhánh tắt (reject_fast):</strong> Thiếu tiêu chí Must-Have bắt buộc nên bỏ qua các bước chấm điểm sau. Tiết kiệm ~3,200 tokens & 2.4s độ trễ.
      </div>
    `;
  } else if (state.path_taken.includes("quarantine")) {
    e2ShortcutBox.innerHTML = `
      <div class="scenario-info-box" style="background-color: #F5F3FF; border-color: #DDD6FE; color: #5B21B6;">
        🛡️ <strong>Kích hoạt cách ly bảo mật (Quarantine):</strong> Chỉ thị độc hại bị chặn tại node Guard. Ngăn chặn đầu độc LLM context và tiết kiệm ~4,500 tokens.
      </div>
    `;
  } else {
    e2ShortcutBox.innerHTML = `
      <div class="scenario-info-box" style="background-color: #ECFDF5; border-color: #A7F3D0; color: #065F46;">
        ✓ <strong>Phân tích đa bước toàn diện:</strong> Ứng viên vượt qua mọi cổng kiểm soát và được đánh giá chi tiết qua hệ thống multi-tool.
      </div>
    `;
  }
}

function handleExportJson() {
  if (!latestScreeningState || !latestScreeningState.result) return;
  const jsonStr = JSON.stringify(latestScreeningState.result, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `screening_result_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
