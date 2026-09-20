"""Generate a focused, high-resolution architecture diagram PNG of the HR CV Screener Agent.

Contains only the architectural graph framework, clearly displaying node types (LLM, Tool, Hybrid,
Terminal/Exit), tools, conditional edges, loops, and clean light-theme styling.
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_PNG = Path("docs/diagrams/agent_architecture.png")
OUTPUT_HTML = Path("docs/diagrams/agent_architecture.html")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>HR CV Screener Agent — Sơ đồ Kiến trúc Graph</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  body {
    width: 1680px;
    height: 1640px;
    background-color: #f8fafc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
    overflow: hidden;
    position: relative;
  }

  /* Subtle background dot pattern */
  .bg-pattern {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: radial-gradient(#cbd5e1 1.2px, transparent 1.2px);
    background-size: 28px 28px;
    opacity: 0.55;
    pointer-events: none;
  }

  /* Header & Integrated Legend Bar */
  .header-container {
    position: absolute;
    top: 28px;
    left: 45px;
    right: 45px;
    height: 96px;
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    z-index: 10;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .header-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #2563eb, #6366f1);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    font-weight: 800;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
  }

  .header-title-box h1 {
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    line-height: 1.2;
  }

  .header-title-box p {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
    margin-top: 2px;
  }

  /* Integrated Legend */
  .legend-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #f8fafc;
    padding: 6px 14px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
  }

  .tag-llm {
    background: #f5f3ff;
    color: #6d28d9;
    border: 1px solid #ddd6fe;
    padding: 3px 8px;
    border-radius: 6px;
  }

  .tag-tool {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 3px 8px;
    border-radius: 6px;
  }

  .tag-hybrid {
    background: #eef2ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
    padding: 3px 8px;
    border-radius: 6px;
  }

  .tag-exit {
    background: #fff1f2;
    color: #be123c;
    border: 1px solid #fecaca;
    padding: 3px 8px;
    border-radius: 6px;
  }

  .legend-divider {
    width: 1px;
    height: 20px;
    background: #cbd5e1;
    margin: 0 4px;
  }

  .line-sample-solid {
    width: 22px;
    height: 0;
    border-top: 2.5px solid #64748b;
  }

  .line-sample-dashed {
    width: 22px;
    height: 0;
    border-top: 2.5px dashed #059669;
  }

  /* Main SVG Diagram Canvas */
  .diagram-svg {
    position: absolute;
    top: 136px;
    left: 45px;
    width: 1590px;
    height: 1480px;
    z-index: 5;
  }
</style>
</head>
<body>

<div class="bg-pattern"></div>

<!-- Header & Legend Bar -->
<div class="header-container">
  <div class="header-left">
    <div class="header-logo">HR</div>
    <div class="header-title-box">
      <h1>HR CV Screener Agent — Sơ đồ Kiến trúc Graph (LangGraph)</h1>
      <p>13 Nodes • 4 Conditional Edges • 5 Deterministic Tools • Bắt đầu tại __start__ và trả kết quả tại __end__</p>
    </div>
  </div>
  
  <div class="legend-bar">
    <div class="legend-item tag-llm">🤖 LLM Node</div>
    <div class="legend-item tag-tool">⚙️ Tool / Rule Node</div>
    <div class="legend-item tag-hybrid">🧬 Hybrid (LLM + Tool)</div>
    <div class="legend-item tag-exit">🛑 Terminal Exit</div>
    <div class="legend-divider"></div>
    <div class="legend-item" style="color: #475569;">
      <div class="line-sample-solid"></div>
      <span>Chuẩn</span>
    </div>
    <div class="legend-item" style="color: #059669;">
      <div class="line-sample-dashed"></div>
      <span>Rẽ nhánh điều kiện</span>
    </div>
  </div>
</div>

<!-- Main Architecture SVG -->
<svg class="diagram-svg" viewBox="0 0 1590 1480" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Soft Drop Shadow Filter -->
    <filter id="nodeShadow" x="-10%" y="-10%" width="125%" height="125%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.06"/>
      <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.04"/>
    </filter>

    <!-- Arrow Markers -->
    <marker id="arrow-solid" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>
    <marker id="arrow-emerald" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#059669"/>
    </marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#d97706"/>
    </marker>
    <marker id="arrow-rose" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#e11d48"/>
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7c3aed"/>
    </marker>
  </defs>

  <!-- ==================================================================== -->
  <!-- CONNECTION EDGES                                                     -->
  <!-- ==================================================================== -->

  <!-- 1. __start__ -> ingest -->
  <path d="M 760,43 L 760,100" stroke="#64748b" stroke-width="2.5" fill="none" marker-end="url(#arrow-solid)"/>

  <!-- 2. ingest -> guard -->
  <path d="M 760,180 L 760,225" stroke="#64748b" stroke-width="2.5" fill="none" marker-end="url(#arrow-solid)"/>

  <!-- 3. guard -> quarantine (Conditional 1a: Injection or Empty) -->
  <path d="M 920,265 L 1100,265" stroke="#e11d48" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-rose)"/>
  <g transform="translate(1010, 250)">
    <rect x="-85" y="-14" width="170" height="22" rx="5" fill="#fff1f2" stroke="#fecaca" stroke-width="1"/>
    <text x="0" y="1" text-anchor="middle" font-size="10.5" font-weight="700" fill="#be123c">Bẩn / Injection (quarantine)</text>
  </g>

  <!-- 4. guard -> extract (Conditional 1b: Safe document) -->
  <path d="M 760,305 L 760,365" stroke="#059669" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-emerald)"/>
  <g transform="translate(760, 335)">
    <rect x="-70" y="-12" width="140" height="22" rx="5" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1"/>
    <text x="0" y="3" text-anchor="middle" font-size="10.5" font-weight="700" fill="#047857">Sạch / Hợp lệ (extract)</text>
  </g>

  <!-- 5. quarantine -> __end__ (Terminal trunk line down right side) -->
  <path d="M 1380,265 L 1480,265 L 1480,1470 L 880,1470" stroke="#f43f5e" stroke-width="2" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-rose)"/>

  <!-- 6. extract -> repair (Conditional 2a: Unparsed dates & attempts < 2) -->
  <path d="M 920,405 L 1100,405" stroke="#d97706" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-amber)"/>
  <g transform="translate(1010, 390)">
    <rect x="-88" y="-14" width="176" height="22" rx="5" fill="#fffbeb" stroke="#fde68a" stroke-width="1"/>
    <text x="0" y="1" text-anchor="middle" font-size="10.5" font-weight="700" fill="#b45309">Thiếu/Lỗi date (tries &lt; 2)</text>
  </g>

  <!-- 7. repair -> repair (Self-loop capped by max attempts) -->
  <path d="M 1380,385 C 1445,360 1445,450 1380,425" stroke="#d97706" stroke-width="2.5" stroke-dasharray="4,4" fill="none" marker-end="url(#arrow-amber)"/>
  <g transform="translate(1460, 405)">
    <rect x="-40" y="-12" width="80" height="22" rx="5" fill="#fffbeb" stroke="#fde68a" stroke-width="1"/>
    <text x="0" y="3" text-anchor="middle" font-size="10" font-weight="700" fill="#b45309">loop (&lt; 2)</text>
  </g>

  <!-- 8. repair -> load_rubric (Fixed or cap reached) -->
  <path d="M 1240,445 L 1240,545 L 920,545" stroke="#059669" stroke-width="2" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-emerald)"/>
  <g transform="translate(1050, 532)">
    <rect x="-72" y="-12" width="144" height="22" rx="5" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1"/>
    <text x="0" y="3" text-anchor="middle" font-size="10" font-weight="700" fill="#047857">Đã sửa / Đạt giới hạn</text>
  </g>

  <!-- 9. extract -> load_rubric (Conditional 2b: Valid profile) -->
  <path d="M 760,445 L 760,505" stroke="#059669" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-emerald)"/>
  <g transform="translate(760, 475)">
    <rect x="-60" y="-12" width="120" height="22" rx="5" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1"/>
    <text x="0" y="3" text-anchor="middle" font-size="10.5" font-weight="700" fill="#047857">Hồ sơ đầy đủ</text>
  </g>

  <!-- 10. load_rubric -> must_have_check -->
  <path d="M 760,585 L 760,635" stroke="#64748b" stroke-width="2.5" fill="none" marker-end="url(#arrow-solid)"/>

  <!-- 11. must_have_check -> reject_fast (Conditional 3a: Missing must-have) -->
  <path d="M 920,675 L 1100,675" stroke="#e11d48" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-rose)"/>
  <g transform="translate(1010, 660)">
    <rect x="-90" y="-14" width="180" height="22" rx="5" fill="#fff1f2" stroke="#fecaca" stroke-width="1"/>
    <text x="0" y="1" text-anchor="middle" font-size="10.5" font-weight="700" fill="#be123c">Thiếu Must-Have (gate=true)</text>
  </g>

  <!-- 12. reject_fast -> __end__ (Connects to right trunk) -->
  <path d="M 1380,675 L 1480,675" stroke="#f43f5e" stroke-width="2" stroke-dasharray="6,4" fill="none"/>

  <!-- 13. must_have_check -> score_criteria (Conditional 3b: Passed must-haves) -->
  <path d="M 760,715 L 760,775" stroke="#059669" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-emerald)"/>
  <g transform="translate(760, 745)">
    <rect x="-75" y="-12" width="150" height="22" rx="5" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1"/>
    <text x="0" y="3" text-anchor="middle" font-size="10.5" font-weight="700" fill="#047857">Đạt tiêu chí cứng (tiếp tục)</text>
  </g>

  <!-- 14. score_criteria -> aggregate -->
  <path d="M 760,865 L 760,925" stroke="#64748b" stroke-width="2.5" fill="none" marker-end="url(#arrow-solid)"/>

  <!-- 15. aggregate -> deep_review (Conditional 4a: Score in gray zone) -->
  <path d="M 600,965 L 320,965 L 320,1065" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-purple)"/>
  <g transform="translate(420, 950)">
    <rect x="-75" y="-14" width="150" height="22" rx="5" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1"/>
    <text x="0" y="1" text-anchor="middle" font-size="10.5" font-weight="700" fill="#6d28d9">Vùng xám (Gray-zone)</text>
  </g>

  <!-- 16. aggregate -> decide (Conditional 4b: Decisive score) -->
  <path d="M 760,1005 L 760,1175" stroke="#059669" stroke-width="2.5" stroke-dasharray="6,4" fill="none" marker-end="url(#arrow-emerald)"/>
  <g transform="translate(760, 1085)">
    <rect x="-75" y="-12" width="150" height="22" rx="5" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1"/>
    <text x="0" y="3" text-anchor="middle" font-size="10.5" font-weight="700" fill="#047857">Điểm rõ ràng (decide)</text>
  </g>

  <!-- 17. deep_review -> decide -->
  <path d="M 320,1145 L 320,1215 L 600,1215" stroke="#7c3aed" stroke-width="2.5" fill="none" marker-end="url(#arrow-purple)"/>
  <g transform="translate(460, 1200)">
    <rect x="-60" y="-12" width="120" height="22" rx="5" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1"/>
    <text x="0" y="3" text-anchor="middle" font-size="10" font-weight="700" fill="#6d28d9">Cập nhật điểm số</text>
  </g>

  <!-- 18. decide -> rank -->
  <path d="M 760,1255 L 760,1315" stroke="#64748b" stroke-width="2.5" fill="none" marker-end="url(#arrow-solid)"/>

  <!-- 19. rank -> __end__ -->
  <path d="M 760,1395 L 760,1445" stroke="#64748b" stroke-width="2.5" fill="none" marker-end="url(#arrow-solid)"/>

  <!-- ==================================================================== -->
  <!-- NODE CARDS                                                           -->
  <!-- ==================================================================== -->

  <!-- NODE 0: __start__ -->
  <g transform="translate(760, 25)" filter="url(#nodeShadow)">
    <rect x="-80" y="-18" width="160" height="36" rx="18" fill="#1e293b" stroke="#334155" stroke-width="2"/>
    <text x="0" y="5" text-anchor="middle" font-size="13" font-weight="800" fill="#ffffff" font-family="'JetBrains Mono', monospace">__start__</text>
  </g>

  <!-- NODE 1: ingest (TOOL / RULE) -->
  <g transform="translate(760, 140)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#10b981" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#ecfdf5"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#ecfdf5"/>
    <rect x="55" y="-35" width="95" height="18" rx="9" fill="#059669"/>
    <text x="102" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">⚙️ TOOL / RULE</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#047857" font-family="'JetBrains Mono', monospace">ingest</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">Tiếp nhận raw text (CV &amp; JD)</text>
    <text x="-145" y="24" font-size="10.5" fill="#64748b">Bảo toàn character offset • Kiểm tra rỗng</text>
  </g>

  <!-- NODE 2: guard (TOOL NODE: scan_injection) -->
  <g transform="translate(760, 265)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#10b981" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#ecfdf5"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#ecfdf5"/>
    <rect x="55" y="-35" width="95" height="18" rx="9" fill="#059669"/>
    <text x="102" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">⚙️ TOOL NODE</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#047857" font-family="'JetBrains Mono', monospace">guard</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">Quét bảo mật Prompt Injection</text>
    <text x="-145" y="24" font-size="10.5" font-weight="600" fill="#0284c7" font-family="'JetBrains Mono', monospace">Tool: scan_injection (regex patterns)</text>
  </g>

  <!-- NODE 3: quarantine (TERMINAL EXIT) -->
  <g transform="translate(1240, 265)" filter="url(#nodeShadow)">
    <rect x="-140" y="-40" width="280" height="80" rx="12" fill="#ffffff" stroke="#f43f5e" stroke-width="2"/>
    <rect x="-140" y="-40" width="280" height="26" rx="12" fill="#fff1f2"/>
    <rect x="-140" y="-20" width="280" height="6" fill="#fff1f2"/>
    <rect x="35" y="-35" width="95" height="18" rx="9" fill="#e11d48"/>
    <text x="82" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">🛑 TERMINAL EXIT</text>
    <text x="-125" y="-22" font-size="14" font-weight="800" fill="#be123c" font-family="'JetBrains Mono', monospace">quarantine</text>
    <text x="-125" y="5" font-size="11.5" font-weight="700" fill="#991b1b">Cách ly tài liệu độc hại</text>
    <text x="-125" y="24" font-size="10.5" fill="#64748b">Gán nhãn NO_FIT + ghi injection_flags</text>
  </g>

  <!-- NODE 4: extract (HYBRID: LLM + TOOL) -->
  <g transform="translate(760, 405)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#6366f1" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#eef2ff"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#eef2ff"/>
    <rect x="35" y="-35" width="115" height="18" rx="9" fill="#4f46e5"/>
    <text x="92" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">🧬 HYBRID: LLM+TOOL</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#4338ca" font-family="'JetBrains Mono', monospace">extract</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">🤖 Trích xuất CandidateProfile (LLM)</text>
    <text x="-145" y="24" font-size="10.5" font-weight="600" fill="#0284c7" font-family="'JetBrains Mono', monospace">⚙️ Tool: calculate_experience (đo overlap)</text>
  </g>

  <!-- NODE 5: repair (HYBRID: LLM REPAIR LOOP) -->
  <g transform="translate(1240, 405)" filter="url(#nodeShadow)">
    <rect x="-140" y="-40" width="280" height="80" rx="12" fill="#ffffff" stroke="#f59e0b" stroke-width="2"/>
    <rect x="-140" y="-40" width="280" height="26" rx="12" fill="#fffbeb"/>
    <rect x="-140" y="-20" width="280" height="6" fill="#fffbeb"/>
    <rect x="30" y="-35" width="100" height="18" rx="9" fill="#d97706"/>
    <text x="80" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">🔄 LLM REPAIR LOOP</text>
    <text x="-125" y="-22" font-size="14" font-weight="800" fill="#b45309" font-family="'JetBrains Mono', monospace">repair</text>
    <text x="-125" y="5" font-size="11.5" font-weight="600" fill="#1e293b">🤖 Sửa lỗi date/thiếu trường (LLM)</text>
    <text x="-125" y="24" font-size="10.5" fill="#64748b">Capped max 2 lần • Tool tính lại năm KN</text>
  </g>

  <!-- NODE 6: load_rubric (HYBRID: CACHE / LLM) -->
  <g transform="translate(760, 545)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#8b5cf6" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#f5f3ff"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#f5f3ff"/>
    <rect x="30" y="-35" width="120" height="18" rx="9" fill="#7c3aed"/>
    <text x="90" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">🧬 CACHE / LLM FALLBACK</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#6d28d9" font-family="'JetBrains Mono', monospace">load_rubric</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">Đọc YAML cache / LLM sinh tiêu chí</text>
    <text x="-145" y="24" font-size="10.5" fill="#64748b">4-8 tiêu chí trọng số • Cờ must_have &amp; skills</text>
  </g>

  <!-- NODE 7: must_have_check (TOOL / RULE GATE) -->
  <g transform="translate(760, 675)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#10b981" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#ecfdf5"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#ecfdf5"/>
    <rect x="55" y="-35" width="95" height="18" rx="9" fill="#059669"/>
    <text x="102" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">⚙️ RULE GATE</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#047857" font-family="'JetBrains Mono', monospace">must_have_check</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">Kiểm tra tiêu chí bắt buộc (No LLM)</text>
    <text x="-145" y="24" font-size="10.5" font-weight="600" fill="#0284c7" font-family="'JetBrains Mono', monospace">Tools: skills_match, search_evidence</text>
  </g>

  <!-- NODE 8: reject_fast (TERMINAL EXIT / FAST PATH) -->
  <g transform="translate(1240, 675)" filter="url(#nodeShadow)">
    <rect x="-140" y="-40" width="280" height="80" rx="12" fill="#ffffff" stroke="#f43f5e" stroke-width="2"/>
    <rect x="-140" y="-40" width="280" height="26" rx="12" fill="#fff1f2"/>
    <rect x="-140" y="-20" width="280" height="6" fill="#fff1f2"/>
    <rect x="40" y="-35" width="90" height="18" rx="9" fill="#e11d48"/>
    <text x="85" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">🛑 FAST REJECT</text>
    <text x="-125" y="-22" font-size="14" font-weight="800" fill="#be123c" font-family="'JetBrains Mono', monospace">reject_fast</text>
    <text x="-125" y="5" font-size="11.5" font-weight="700" fill="#991b1b">Loại sớm, tiết kiệm Token</text>
    <text x="-125" y="24" font-size="10.5" fill="#64748b">Thiếu tiêu chí cứng • Vẫn lưu vết giải thích</text>
  </g>

  <!-- NODE 9: score_criteria (CORE HYBRID: LLM + MULTI-TOOL) -->
  <g transform="translate(760, 820)" filter="url(#nodeShadow)">
    <rect x="-160" y="-45" width="320" height="90" rx="12" fill="#ffffff" stroke="#6366f1" stroke-width="2"/>
    <rect x="-160" y="-45" width="320" height="26" rx="12" fill="#eef2ff"/>
    <rect x="-160" y="-25" width="320" height="6" fill="#eef2ff"/>
    <rect x="25" y="-40" width="125" height="18" rx="9" fill="#4f46e5"/>
    <text x="87" y="-27" text-anchor="middle" font-size="9" font-weight="800" fill="#ffffff">🧬 HYBRID: LLM + TOOLS</text>
    <text x="-145" y="-27" font-size="14" font-weight="800" fill="#4338ca" font-family="'JetBrains Mono', monospace">score_criteria</text>
    <text x="-145" y="-3" font-size="11.5" font-weight="600" fill="#1e293b">🤖 Chấm điểm 0.0-1.0 + Trích dẫn quote (LLM)</text>
    <text x="-145" y="16" font-size="10.5" font-weight="600" fill="#0284c7" font-family="'JetBrains Mono', monospace">⚙️ Tool: search_evidence (neo offset ký tự E1)</text>
    <text x="-145" y="33" font-size="10.5" font-weight="600" fill="#059669" font-family="'JetBrains Mono', monospace">⚙️ Tool: calculate_experience &amp; expand_skill</text>
  </g>

  <!-- NODE 10: aggregate (TOOL / MATH SCORING) -->
  <g transform="translate(760, 965)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#10b981" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#ecfdf5"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#ecfdf5"/>
    <rect x="50" y="-35" width="100" height="18" rx="9" fill="#059669"/>
    <text x="100" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">⚙️ TOOL: MATH</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#047857" font-family="'JetBrains Mono', monospace">aggregate</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">Tổng hợp điểm có trọng số</text>
    <text x="-145" y="24" font-size="10.5" font-weight="600" fill="#0284c7" font-family="'JetBrains Mono', monospace">Tool: aggregate_scorecard (phát hiện gray-zone)</text>
  </g>

  <!-- NODE 11: deep_review (PURE LLM: 2ND PASS ESCALATION) -->
  <g transform="translate(320, 1105)" filter="url(#nodeShadow)">
    <rect x="-145" y="-40" width="290" height="80" rx="12" fill="#ffffff" stroke="#9333ea" stroke-width="2"/>
    <rect x="-145" y="-40" width="290" height="26" rx="12" fill="#faf5ff"/>
    <rect x="-145" y="-20" width="290" height="6" fill="#faf5ff"/>
    <rect x="40" y="-35" width="95" height="18" rx="9" fill="#7e22ce"/>
    <text x="87" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">🤖 PURE LLM PASS</text>
    <text x="-130" y="-22" font-size="14" font-weight="800" fill="#6b21a8" font-family="'JetBrains Mono', monospace">deep_review</text>
    <text x="-130" y="5" font-size="11.5" font-weight="700" fill="#581c87">Đánh giá kỹ hồ sơ vùng xám</text>
    <text x="-130" y="24" font-size="10.5" fill="#64748b">LLM phản biện tiêu chí tiệm cận ngưỡng cắt</text>
  </g>

  <!-- NODE 12: decide (DETERMINISTIC LOGIC) -->
  <g transform="translate(760, 1215)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#0ea5e9" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#f0f9ff"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#f0f9ff"/>
    <rect x="55" y="-35" width="95" height="18" rx="9" fill="#0284c7"/>
    <text x="102" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">⚙️ LOGIC NODE</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#0369a1" font-family="'JetBrains Mono', monospace">decide</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">Quyết định nhãn phù hợp</text>
    <text x="-145" y="24" font-size="10.5" fill="#64748b">Good Fit / Potential Fit / No Fit &amp; Trace</text>
  </g>

  <!-- NODE 13: rank (DETERMINISTIC RANKER) -->
  <g transform="translate(760, 1355)" filter="url(#nodeShadow)">
    <rect x="-160" y="-40" width="320" height="80" rx="12" fill="#ffffff" stroke="#0ea5e9" stroke-width="2"/>
    <rect x="-160" y="-40" width="320" height="26" rx="12" fill="#f0f9ff"/>
    <rect x="-160" y="-20" width="320" height="6" fill="#f0f9ff"/>
    <rect x="45" y="-35" width="105" height="18" rx="9" fill="#0284c7"/>
    <text x="97" y="-22" text-anchor="middle" font-size="9.5" font-weight="800" fill="#ffffff">⚙️ EXPLAINABILITY</text>
    <text x="-145" y="-22" font-size="14" font-weight="800" fill="#0369a1" font-family="'JetBrains Mono', monospace">rank</text>
    <text x="-145" y="5" font-size="11.5" font-weight="600" fill="#1e293b">Xếp hạng tiêu chí theo đóng góp</text>
    <text x="-145" y="24" font-size="10.5" fill="#64748b">Xếp weighted_contribution (điểm mạnh / lỗ hổng)</text>
  </g>

  <!-- NODE 14: __end__ -->
  <g transform="translate(760, 1470)" filter="url(#nodeShadow)">
    <rect x="-120" y="-25" width="240" height="50" rx="25" fill="#0f172a" stroke="#334155" stroke-width="2"/>
    <text x="0" y="5" text-anchor="middle" font-size="15" font-weight="800" fill="#ffffff" font-family="'JetBrains Mono', monospace">__end__ (ScreeningResult)</text>
  </g>

</svg>

</body>
</html>
"""

def main():
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"Wrote HTML template to {OUTPUT_HTML}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Viewport matching diagram dimensions
        page = browser.new_page(viewport={"width": 1680, "height": 1640}, device_scale_factor=2)
        page.goto(f"file:///{OUTPUT_HTML.resolve().as_posix()}")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUTPUT_PNG), full_page=True)
        browser.close()

    print(f"Successfully generated focused architecture PNG: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
