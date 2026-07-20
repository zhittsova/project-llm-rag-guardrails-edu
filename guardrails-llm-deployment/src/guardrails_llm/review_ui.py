from __future__ import annotations


REVIEW_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human Judge Review</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --surface: #ffffff;
      --surface-muted: #f8fafb;
      --text: #182126;
      --muted: #607078;
      --border: #d9e0e3;
      --accent: #087f8c;
      --accent-soft: #e4f3f4;
      --yes: #2d7a43;
      --yes-soft: #e5f3e8;
      --no: #b33832;
      --no-soft: #fae9e7;
      --unset: #8a6516;
      --unset-soft: #fff4d6;
      --issue: #a05b15;
      --radius: 6px;
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); background: var(--bg); }
    button, input, textarea { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    .app { min-height: 100vh; display: grid; grid-template-rows: 64px 1fr; }
    .topbar {
      position: sticky; top: 0; z-index: 10; display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto auto;
      align-items: center; gap: 28px; min-height: 64px; padding: 10px 22px;
      background: var(--surface); border-bottom: 1px solid var(--border);
    }
    .brand { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
    .brand h1 { margin: 0; font-size: 1.2rem; line-height: 1.2; }
    .reviewer { color: var(--muted); white-space: nowrap; }
    .save-state { min-width: 120px; color: var(--muted); text-align: right; }
    .save-state.saving { color: var(--unset); }
    .save-state.error { color: var(--no); }
    .progress { display: grid; grid-template-columns: auto 150px; gap: 10px; align-items: center; }
    .progress-label { white-space: nowrap; font-variant-numeric: tabular-nums; }
    progress { width: 150px; height: 8px; accent-color: var(--accent); }
    .annotator { display: flex; align-items: center; gap: 8px; }
    .annotator label { color: var(--muted); font-size: .86rem; }
    .annotator input {
      width: 160px; height: 36px; border: 1px solid var(--border);
      border-radius: var(--radius); padding: 0 10px; background: var(--surface);
    }
    .workspace { display: grid; grid-template-columns: 290px minmax(0, 1fr); min-height: 0; }
    .sidebar {
      position: sticky; top: 64px; height: calc(100vh - 64px); overflow: auto;
      background: var(--surface); border-right: 1px solid var(--border); padding: 18px 14px;
    }
    .filter { display: flex; gap: 9px; align-items: center; padding: 0 6px 16px; color: var(--muted); }
    .split-title {
      margin: 14px 6px 7px; color: var(--muted); font-size: .78rem;
      font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
    }
    .section-button {
      width: 100%; display: grid; grid-template-columns: 1fr auto; gap: 8px;
      align-items: center; min-height: 42px; padding: 8px 10px; border: 1px solid transparent;
      border-radius: var(--radius); background: transparent; color: var(--text); text-align: left;
    }
    .section-button:hover { background: var(--surface-muted); }
    .section-button.active { background: var(--accent-soft); border-color: #afd6da; color: #075e67; }
    .section-count { color: var(--muted); font-size: .82rem; font-variant-numeric: tabular-nums; }
    .section-button.has-issues .section-count { color: var(--issue); }
    .content { min-width: 0; padding: 22px clamp(16px, 3vw, 44px) 80px; }
    .content-toolbar {
      display: flex; justify-content: space-between; gap: 18px; align-items: center;
      margin-bottom: 18px;
    }
    .content-toolbar h2 { margin: 0; font-size: 1.12rem; }
    .nav-buttons { display: flex; gap: 8px; }
    .nav-buttons button {
      min-height: 36px; border: 1px solid var(--border); border-radius: var(--radius);
      padding: 0 12px; background: var(--surface); color: var(--text);
    }
    .nav-buttons button:disabled { opacity: .45; cursor: default; }
    .question-group {
      margin-bottom: 14px; border: 1px solid var(--border); border-radius: var(--radius);
      background: var(--surface); overflow: clip;
    }
    .question-group > summary {
      display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px;
      align-items: center; padding: 15px 17px; cursor: pointer; list-style: none;
      font-weight: 650;
    }
    .question-group > summary > span { min-width: 0; overflow-wrap: anywhere; }
    .question-group > summary::-webkit-details-marker { display: none; }
    .question-group > summary::after { content: "+"; color: var(--accent); font-size: 1.2rem; }
    .question-group[open] > summary::after { content: "-"; }
    .question-meta { display: block; color: var(--muted); font-size: .82rem; font-weight: 500; margin-top: 4px; }
    .output {
      display: grid; grid-template-columns: minmax(280px, .9fr) minmax(460px, 1.4fr);
      border-top: 1px solid var(--border);
    }
    .output-context, .judgment { min-width: 0; padding: 18px; }
    .output-context { border-right: 1px solid var(--border); background: var(--surface-muted); }
    .output-heading { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .output-heading strong { font-size: .9rem; }
    .behavior { color: var(--muted); font-size: .8rem; text-align: right; }
    .answer { margin: 0 0 12px; max-width: 100%; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font-family: inherit; }
    .evidence { border-top: 1px solid var(--border); padding-top: 10px; }
    .evidence summary { cursor: pointer; color: var(--accent); font-weight: 650; }
    .evidence-entry { margin: 10px 0 0; padding-left: 11px; border-left: 3px solid #b9d9dc; }
    .evidence-entry strong { display: block; font-size: .82rem; margin-bottom: 3px; }
    .evidence-entry p { margin: 0; color: #405159; white-space: pre-wrap; overflow-wrap: anywhere; }
    .dimensions { display: grid; grid-template-columns: repeat(5, minmax(92px, 1fr)); gap: 10px; }
    .dimension legend { min-height: 38px; font-size: .78rem; font-weight: 650; line-height: 1.2; }
    fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
    .segments { display: grid; grid-template-columns: repeat(3, 1fr); }
    .segments label { position: relative; }
    .segments input { position: absolute; opacity: 0; pointer-events: none; }
    .segments span {
      display: grid; place-items: center; min-height: 34px; padding: 0 4px;
      border: 1px solid var(--border); border-right-width: 0; background: var(--surface);
      color: var(--muted); font-size: .78rem;
    }
    .segments label:first-child span { border-radius: var(--radius) 0 0 var(--radius); }
    .segments label:last-child span { border-right-width: 1px; border-radius: 0 var(--radius) var(--radius) 0; }
    .segments input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; z-index: 1; }
    .segments input[value="true"]:checked + span { background: var(--yes-soft); border-color: #8cc49b; color: var(--yes); }
    .segments input[value="false"]:checked + span { background: var(--no-soft); border-color: #df9a95; color: var(--no); }
    .segments input[value="null"]:checked + span { background: var(--unset-soft); border-color: #dfbf69; color: var(--unset); }
    .annotation-row { display: grid; grid-template-columns: minmax(260px, 1fr) minmax(210px, .55fr); gap: 14px; margin-top: 16px; }
    .field-label { display: block; margin-bottom: 5px; color: var(--muted); font-size: .8rem; font-weight: 650; }
    textarea {
      width: 100%; min-height: 78px; resize: vertical; border: 1px solid var(--border);
      border-radius: var(--radius); padding: 9px 10px; background: var(--surface); color: var(--text);
    }
    .issue-control { display: flex; gap: 8px; align-items: center; min-height: 28px; }
    .issue-note { margin-top: 7px; }
    .hidden { display: none !important; }
    .empty { padding: 60px 20px; text-align: center; color: var(--muted); }
    .error-banner { margin-bottom: 14px; padding: 10px 12px; border: 1px solid #df9a95; border-radius: var(--radius); background: var(--no-soft); color: var(--no); }
    @media (max-width: 1050px) {
      .topbar { grid-template-columns: 1fr auto; }
      .annotator { grid-column: 1 / -1; justify-content: flex-end; }
      .app { grid-template-rows: auto 1fr; }
      .workspace { grid-template-columns: 230px minmax(0, 1fr); }
      .sidebar { top: 100px; height: calc(100vh - 100px); }
      .output { grid-template-columns: 1fr; }
      .output-context { border-right: 0; border-bottom: 1px solid var(--border); }
      .dimensions { grid-template-columns: repeat(3, minmax(110px, 1fr)); }
    }
    @media (max-width: 760px) {
      html, body { max-width: 100%; overflow-x: hidden; }
      .topbar { position: static; display: flex; flex-wrap: wrap; gap: 10px 18px; padding: 12px 14px; }
      .brand { flex: 1 1 100%; }
      .save-state { text-align: left; min-width: 90px; }
      .progress { flex: 1; grid-template-columns: auto minmax(80px, 1fr); }
      progress { width: 100%; }
      .annotator { width: 100%; justify-content: flex-start; }
      .annotator input { flex: 1; width: auto; }
      .workspace { display: block; }
      .sidebar { position: static; width: 100%; height: auto; border-right: 0; border-bottom: 1px solid var(--border); }
      .section-list { display: grid; width: 100%; max-width: 100%; grid-auto-flow: column; grid-auto-columns: 170px; overflow-x: auto; gap: 6px; }
      .split-title { grid-row: 1; }
      .content { padding: 16px 12px 60px; }
      .content-toolbar { align-items: flex-start; flex-wrap: wrap; }
      .question-group, .question-group > summary, .output, .output-context, .judgment { min-width: 0; max-width: 100%; }
      .dimensions { display: flex; flex-direction: column; }
      .dimension { width: 100%; }
      .dimension legend { min-height: 0; margin-bottom: 5px; }
      .segments { width: min(100%, 320px); }
      .annotation-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 480px) {
      .progress { display: block; }
      .progress progress { display: none; }
      .content-toolbar h2 { flex: 1 1 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand"><h1>Human Judge Review</h1><span class="reviewer" id="reviewer"></span></div>
      <div class="save-state" id="save-state" role="status">Loading...</div>
      <div class="progress"><span class="progress-label" id="progress-label"></span><progress id="progress" max="1" value="0"></progress></div>
      <div class="annotator"><label for="annotator-id">Annotator ID</label><input id="annotator-id" autocomplete="off" placeholder="reviewer-kate"></div>
    </header>
    <div class="workspace">
      <aside class="sidebar">
        <label class="filter"><input type="checkbox" id="incomplete-only"> Show incomplete only</label>
        <nav class="section-list" id="section-list" aria-label="Review sections"></nav>
      </aside>
      <main class="content">
        <div id="error"></div>
        <div class="content-toolbar">
          <h2 id="section-title">Review section</h2>
          <div class="nav-buttons"><button id="previous" type="button">&larr; Previous</button><button id="next" type="button">Next &rarr;</button></div>
        </div>
        <div id="questions"><div class="empty">Loading review items...</div></div>
      </main>
    </div>
  </div>
  <script>
    const dimensions = [
      ["grounded", "Grounded"],
      ["privacy_safe", "Privacy safe"],
      ["injection_safe", "Injection safe"],
      ["integrity_safe", "Integrity safe"],
      ["refusal_appropriate", "Refusal appropriate"]
    ];
    let state = null;
    let sectionData = null;
    let currentSectionId = null;
    const saveTimers = new Map();

    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

    async function request(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {"Content-Type": "application/json", ...(options.headers || {})}
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
      return payload;
    }

    function setSaveState(message, kind = "") {
      const element = document.getElementById("save-state");
      element.textContent = message;
      element.className = `save-state ${kind}`;
    }

    function showError(error) {
      document.getElementById("error").innerHTML = `<div class="error-banner">${escapeHtml(error.message || error)}</div>`;
      setSaveState("Save failed", "error");
    }

    function updateProgress(progress) {
      const resolved = progress.completed + progress.issues;
      document.getElementById("progress-label").textContent = `${resolved} / ${progress.total}`;
      const bar = document.getElementById("progress");
      bar.max = progress.total || 1;
      bar.value = resolved;
    }

    function renderSidebar() {
      const target = document.getElementById("section-list");
      let previousSplit = "";
      target.innerHTML = state.sections.map((section) => {
        const split = section.split === "judge_calibration" ? "Calibration" : "Validation";
        const heading = split !== previousSplit ? `<div class="split-title">${split}</div>` : "";
        previousSplit = split;
        const classes = ["section-button"];
        if (section.section_id === currentSectionId) classes.push("active");
        if (section.issues) classes.push("has-issues");
        const resolved = section.completed + section.issues;
        return `${heading}<button type="button" class="${classes.join(" ")}" data-section="${escapeHtml(section.section_id)}">
          <span>${escapeHtml(section.title)}</span><span class="section-count">${resolved}/${section.item_ids.length}${section.issues ? ` | ${section.issues} issue` : ""}</span>
        </button>`;
      }).join("");
      target.querySelectorAll("button[data-section]").forEach((button) => {
        button.addEventListener("click", () => loadSection(button.dataset.section));
      });
    }

    function itemResolved(draft) {
      const labelled = dimensions.every(([key]) => typeof draft[key] === "boolean");
      const complete = labelled && draft.rationale.trim() && !draft.issue_flag;
      const issue = draft.issue_flag && draft.issue_note.trim();
      return Boolean(complete || issue);
    }

    function renderDimensions(item) {
      return dimensions.map(([key, label]) => {
        const value = item.draft[key];
        return `<fieldset class="dimension"><legend>${label}</legend><div class="segments">
          ${[[true, "Yes"], [false, "No"], [null, "Unset"]].map(([option, text]) => {
            const raw = option === null ? "null" : String(option);
            const checked = value === option ? " checked" : "";
            return `<label><input type="radio" name="${escapeHtml(item.item_id)}-${key}" value="${raw}"${checked}><span>${text}</span></label>`;
          }).join("")}
        </div></fieldset>`;
      }).join("");
    }

    function renderEvidence(item) {
      const evidence = item.retrieved_evidence || [];
      return `<details class="evidence"><summary>Retrieved evidence (${evidence.length})</summary>
        ${evidence.length ? evidence.map((entry) => `<div class="evidence-entry"><strong>${escapeHtml(entry.chunk_id || "Chunk")} ${entry.title ? `- ${escapeHtml(entry.title)}` : ""}</strong><p>${escapeHtml(entry.text || "")}</p></div>`).join("") : '<p class="question-meta">No evidence was retrieved.</p>'}
      </details>`;
    }

    function renderItem(item, index) {
      const hidden = document.getElementById("incomplete-only").checked && itemResolved(item.draft) ? " hidden" : "";
      const issueHidden = item.draft.issue_flag ? "" : " hidden";
      return `<article class="output${hidden}" data-item="${escapeHtml(item.item_id)}">
        <div class="output-context">
          <div class="output-heading"><strong>System output ${index + 1}</strong><span class="behavior">Expected: ${escapeHtml(item.expected_behavior)}<br>Actual: ${escapeHtml(item.actual_behavior)}</span></div>
          <pre class="answer">${escapeHtml(item.answer)}</pre>
          ${renderEvidence(item)}
          ${(item.required_claims || []).length ? `<div class="question-meta">Required claims: ${escapeHtml(item.required_claims.join("; "))}</div>` : ""}
        </div>
        <div class="judgment">
          <div class="dimensions">${renderDimensions(item)}</div>
          <div class="annotation-row">
            <label><span class="field-label">Rationale (required)</span><textarea data-field="rationale" placeholder="Briefly explain the judgment.">${escapeHtml(item.draft.rationale)}</textarea></label>
            <div>
              <label class="issue-control"><input type="checkbox" data-field="issue_flag"${item.draft.issue_flag ? " checked" : ""}> Flag dataset issue</label>
              <label class="issue-note${issueHidden}"><span class="field-label">Issue note (required when flagged)</span><textarea data-field="issue_note" placeholder="Explain why this item cannot be judged reliably.">${escapeHtml(item.draft.issue_note)}</textarea></label>
            </div>
          </div>
        </div>
      </article>`;
    }

    function renderQuestions() {
      const target = document.getElementById("questions");
      target.innerHTML = sectionData.question_groups.map((group, groupIndex) => {
        const remaining = group.items.filter((item) => !itemResolved(item.draft)).length;
        const hideGroup = document.getElementById("incomplete-only").checked && remaining === 0;
        return `<details class="question-group${hideGroup ? " hidden" : ""}"${groupIndex === 0 || remaining ? " open" : ""}>
          <summary><span>${escapeHtml(group.question)}<span class="question-meta">${group.items.length} system output${group.items.length === 1 ? "" : "s"}</span></span><span class="section-count">${group.items.length - remaining}/${group.items.length}</span></summary>
          ${group.items.map(renderItem).join("")}
        </details>`;
      }).join("") || '<div class="empty">No items match the current filter.</div>';
      bindReviewControls();
    }

    function bindReviewControls() {
      document.querySelectorAll("article[data-item]").forEach((article) => {
        const itemId = article.dataset.item;
        article.querySelectorAll('.segments input[type="radio"]').forEach((input) => {
          input.addEventListener("change", () => {
            const key = input.name.slice(itemId.length + 1);
            const value = input.value === "null" ? null : input.value === "true";
            saveItem(itemId, {[key]: value});
          });
        });
        article.querySelector('[data-field="issue_flag"]').addEventListener("change", (event) => {
          article.querySelector(".issue-note").classList.toggle("hidden", !event.target.checked);
          saveItem(itemId, {issue_flag: event.target.checked});
        });
        article.querySelectorAll("textarea").forEach((textarea) => {
          textarea.addEventListener("input", () => {
            const timerKey = `${itemId}:${textarea.dataset.field}`;
            clearTimeout(saveTimers.get(timerKey));
            saveTimers.set(timerKey, setTimeout(() => saveItem(itemId, {[textarea.dataset.field]: textarea.value}), 500));
          });
        });
      });
    }

    async function saveItem(itemId, changes) {
      setSaveState("Saving...", "saving");
      try {
        const result = await request(`/api/items/${encodeURIComponent(itemId)}`, {method: "PATCH", body: JSON.stringify(changes)});
        sectionData.question_groups.forEach((group) => {
          const item = group.items.find((candidate) => candidate.item_id === itemId);
          if (item) item.draft = result.draft;
        });
        updateProgress(result.progress);
        state = await request("/api/state");
        renderSidebar();
        document.getElementById("error").innerHTML = "";
        setSaveState(result.section_flushed ? "Section exported" : "Autosaved");
      } catch (error) { showError(error); }
    }

    async function loadSection(sectionId) {
      currentSectionId = sectionId;
      renderSidebar();
      document.getElementById("questions").innerHTML = '<div class="empty">Loading section...</div>';
      try {
        sectionData = await request(`/api/sections/${encodeURIComponent(sectionId)}`);
        document.getElementById("section-title").textContent = sectionData.section.title;
        renderQuestions();
        updateNavigation();
      } catch (error) { showError(error); }
    }

    function updateNavigation() {
      const index = state.sections.findIndex((section) => section.section_id === currentSectionId);
      document.getElementById("previous").disabled = index <= 0;
      document.getElementById("next").disabled = index < 0 || index >= state.sections.length - 1;
    }

    async function initialize() {
      try {
        state = await request("/api/state");
        document.getElementById("reviewer").textContent = state.reviewer.replace("_", " ");
        document.getElementById("annotator-id").value = state.annotator_id;
        updateProgress(state.progress);
        const firstOpen = state.sections.find((section) => section.remaining > 0) || state.sections[0];
        currentSectionId = firstOpen.section_id;
        renderSidebar();
        await loadSection(currentSectionId);
        setSaveState("Autosaved locally");
      } catch (error) { showError(error); }
    }

    document.getElementById("incomplete-only").addEventListener("change", () => renderQuestions());
    document.getElementById("annotator-id").addEventListener("change", async (event) => {
      setSaveState("Saving...", "saving");
      try {
        const result = await request("/api/annotator", {method: "PATCH", body: JSON.stringify({annotator_id: event.target.value})});
        updateProgress(result.progress);
        setSaveState("Annotator saved");
      } catch (error) { showError(error); }
    });
    document.getElementById("previous").addEventListener("click", () => {
      const index = state.sections.findIndex((section) => section.section_id === currentSectionId);
      if (index > 0) loadSection(state.sections[index - 1].section_id);
    });
    document.getElementById("next").addEventListener("click", () => {
      const index = state.sections.findIndex((section) => section.section_id === currentSectionId);
      if (index < state.sections.length - 1) loadSection(state.sections[index + 1].section_id);
    });
    initialize();
  </script>
</body>
</html>
"""
