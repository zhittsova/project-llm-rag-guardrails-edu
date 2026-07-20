from __future__ import annotations


POLICY_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Guardrail Policy Manager</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8; --surface: #fff; --surface-muted: #f0f3f4;
      --text: #172126; --muted: #627178; --border: #d7dfe2;
      --accent: #087f8c; --accent-soft: #e4f3f4; --danger: #a63b32;
      --danger-soft: #fbeceb; --ok: #217a46; --ok-soft: #e9f5ed;
      --warn: #8a6415; --warn-soft: #fff6db; --radius: 6px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    button, input, textarea, select { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    .topbar {
      position: sticky; top: 0; z-index: 10; min-height: 64px; display: flex;
      align-items: center; justify-content: space-between; gap: 20px;
      padding: 10px 22px; background: var(--surface); border-bottom: 1px solid var(--border);
    }
    .brand h1 { margin: 0; font-size: 1.15rem; }
    .brand p { margin: 3px 0 0; color: var(--muted); font-size: .8rem; }
    .actions { display: flex; align-items: center; gap: 8px; }
    .status { min-width: 100px; color: var(--muted); font-size: .84rem; text-align: right; }
    .button {
      min-height: 36px; padding: 0 12px; border: 1px solid var(--border);
      border-radius: var(--radius); background: var(--surface); color: var(--text);
    }
    .button:hover { border-color: #aebbc0; }
    .button.primary { border-color: var(--accent); background: var(--accent); color: #fff; }
    .button.danger { border-color: #dcaaa5; color: var(--danger); }
    .button:disabled { opacity: .45; cursor: default; }
    .shell { display: grid; grid-template-columns: 230px minmax(0, 1fr) 340px; min-height: calc(100vh - 64px); }
    .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 18px 12px; }
    .nav-button {
      width: 100%; min-height: 42px; padding: 0 12px; border: 0;
      border-radius: var(--radius); background: transparent; color: var(--text); text-align: left;
    }
    .nav-button:hover { background: var(--surface-muted); }
    .nav-button.active { background: var(--accent-soft); color: #075f68; font-weight: 700; }
    .source { margin: 22px 8px 0; color: var(--muted); font-size: .75rem; overflow-wrap: anywhere; }
    .editor { min-width: 0; padding: 24px clamp(18px, 3vw, 40px) 80px; }
    .editor-header { display: flex; align-items: start; justify-content: space-between; gap: 18px; margin-bottom: 18px; }
    .editor-header h2 { margin: 0; font-size: 1.35rem; }
    .editor-header p { margin: 5px 0 0; color: var(--muted); }
    .validation { margin-bottom: 18px; padding: 12px 14px; border-left: 4px solid var(--ok); background: var(--ok-soft); }
    .validation.invalid { border-color: var(--danger); background: var(--danger-soft); }
    .validation ul { margin: 7px 0 0; padding-left: 20px; }
    .validation .warnings { color: var(--warn); }
    .section { padding: 18px 0; border-top: 1px solid var(--border); }
    .section:first-child { border-top: 0; padding-top: 0; }
    .section-heading { display: flex; justify-content: space-between; align-items: center; gap: 14px; margin-bottom: 13px; }
    .section-heading h3 { margin: 0; font-size: 1rem; }
    .section-heading p { margin: 3px 0 0; color: var(--muted); font-size: .8rem; }
    .rule {
      margin-bottom: 10px; padding: 14px; border: 1px solid var(--border);
      border-radius: var(--radius); background: var(--surface);
    }
    .rule-grid { display: grid; grid-template-columns: minmax(150px, .4fr) minmax(260px, 1fr) auto; gap: 12px; align-items: end; }
    .field label { display: block; margin-bottom: 5px; color: var(--muted); font-size: .78rem; font-weight: 650; }
    input[type="text"], input[type="number"], textarea, select {
      width: 100%; border: 1px solid var(--border); border-radius: var(--radius);
      background: var(--surface); color: var(--text); padding: 8px 9px;
    }
    textarea { min-height: 76px; resize: vertical; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .threshold { display: grid; grid-template-columns: 1fr 62px; gap: 8px; align-items: center; margin-top: 10px; }
    .list-input { min-height: 58px; }
    .inline { display: flex; align-items: center; gap: 9px; }
    .switch { width: 18px; height: 18px; accent-color: var(--accent); }
    .coverage-row { display: grid; grid-template-columns: 150px 150px 170px minmax(220px, 1fr) auto; gap: 10px; margin-bottom: 10px; align-items: end; }
    .runtime-table { width: 100%; border-collapse: collapse; background: var(--surface); }
    .runtime-table th, .runtime-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
    .runtime-table th { width: 34%; color: var(--muted); font-weight: 650; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem; overflow-wrap: anywhere; }
    details.diff { margin-top: 16px; border-top: 1px solid var(--border); padding-top: 13px; }
    details.diff summary { cursor: pointer; color: var(--accent); font-weight: 650; }
    details.diff pre { max-height: 360px; overflow: auto; padding: 12px; background: #152126; color: #dce8e9; white-space: pre-wrap; }
    .history-row { display: grid; grid-template-columns: 1fr auto; gap: 12px; padding: 12px 0; border-bottom: 1px solid var(--border); }
    .simulator { position: sticky; top: 64px; align-self: start; height: calc(100vh - 64px); overflow: auto; padding: 22px 18px; background: var(--surface); border-left: 1px solid var(--border); }
    .simulator h2 { margin: 0 0 4px; font-size: 1.05rem; }
    .simulator > p { margin: 0 0 16px; color: var(--muted); font-size: .82rem; }
    .simulator textarea { min-height: 120px; }
    .sim-actions { display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 9px; }
    .result { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--border); }
    .disposition { font-size: 1.15rem; font-weight: 750; text-transform: capitalize; }
    .trigger { display: inline-block; margin: 6px 5px 0 0; padding: 3px 6px; border: 1px solid #b8d9dc; background: var(--accent-soft); border-radius: 3px; font-size: .75rem; }
    .score { margin-top: 8px; padding: 8px 0; border-top: 1px solid var(--border); font-size: .8rem; }
    .empty { padding: 32px 0; color: var(--muted); text-align: center; }
    .hidden { display: none !important; }
    @media (max-width: 1120px) {
      .shell { grid-template-columns: 210px minmax(0, 1fr); }
      .simulator { position: static; grid-column: 2; height: auto; border-left: 0; border-top: 1px solid var(--border); }
    }
    @media (max-width: 760px) {
      .topbar { position: static; flex-wrap: wrap; padding: 12px 14px; }
      .actions { width: 100%; flex-wrap: wrap; }
      .status { text-align: left; }
      .shell { display: block; }
      .sidebar { display: flex; gap: 6px; overflow-x: auto; border-right: 0; border-bottom: 1px solid var(--border); }
      .nav-button { width: auto; white-space: nowrap; }
      .source { display: none; }
      .editor { padding: 20px 14px 50px; }
      .simulator { padding: 20px 14px 50px; }
      .rule-grid, .coverage-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><h1>Guardrail Policy Manager</h1><p>Local instructor workspace</p></div>
    <div class="actions">
      <span class="status" id="save-status" role="status">Loading...</span>
      <button class="button hidden" id="reload-source" type="button">Reload published file</button>
      <button class="button" id="save-now" type="button">Save draft</button>
      <button class="button primary" id="publish" type="button">Publish policy</button>
    </div>
  </header>
  <div class="shell">
    <nav class="sidebar" aria-label="Policy sections">
      <button class="nav-button active" data-view="input">Input guards</button>
      <button class="nav-button" data-view="context">Context guards</button>
      <button class="nav-button" data-view="retrieval">Retrieval & output</button>
      <button class="nav-button" data-view="coverage">Coverage cases</button>
      <button class="nav-button" data-view="messages">Messages</button>
      <button class="nav-button" data-view="runtime">Runtime controls</button>
      <button class="nav-button" data-view="history">History</button>
      <div class="source" id="source"></div>
    </nav>
    <main class="editor">
      <div class="editor-header"><div><h2 id="view-title">Input guards</h2><p id="view-description"></p></div></div>
      <div id="validation"></div>
      <div id="editor-content"><div class="empty">Loading policy...</div></div>
      <details class="diff"><summary>Review draft diff</summary><pre id="diff">No changes.</pre></details>
    </main>
    <aside class="simulator">
      <h2>Test request</h2>
      <p>Offline preview. No model API calls are made.</p>
      <textarea id="simulation-text" placeholder="Enter a request or retrieved context..."></textarea>
      <div class="sim-actions"><select id="simulation-stage"><option value="input">Input</option><option value="context">Context</option><option value="output">Output</option></select><button class="button primary" id="simulate" type="button">Run test</button></div>
      <div class="result" id="simulation-result"><span class="empty">Run a test to inspect matched rules.</span></div>
    </aside>
  </div>
  <script>
    let state = null;
    let activeView = "input";
    let saveTimer = null;

    const viewMeta = {
      input: ["Input guards", "Inspect requests before retrieval or generation."],
      context: ["Context guards", "Detect instructions hidden in retrieved course content."],
      retrieval: ["Retrieval & output", "Constrain document visibility and generated responses."],
      coverage: ["Coverage cases", "Keep one direct, variant, and benign near-miss case per rule family."],
      messages: ["User messages", "Control safe block, abstention, and tutoring responses."],
      runtime: ["Runtime controls", "Read-only deployed model and threshold provenance."],
      history: ["Published history", "Restore a previous policy snapshot when needed."]
    };

    const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;").replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
    const lines = (value) => Array.isArray(value) ? value.join("\n") : "";
    const fromLines = (value) => value.split("\n").map(item => item.trim()).filter(Boolean);

    async function request(path, options = {}) {
      const response = await fetch(path, {headers: {"Content-Type": "application/json"}, ...options});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
      return payload;
    }

    async function loadState() {
      state = await request("/api/state");
      render();
    }

    function render() {
      const [title, description] = viewMeta[activeView];
      document.getElementById("view-title").textContent = title;
      document.getElementById("view-description").textContent = description;
      document.querySelectorAll(".nav-button").forEach(button => button.classList.toggle("active", button.dataset.view === activeView));
      document.getElementById("source").innerHTML = `<strong>Published file</strong><br>${escapeHtml(state.policy_path)}<br><br><strong>SHA-256</strong><br><span class="mono">${escapeHtml(state.policy_sha256)}</span>`;
      renderChrome();
      renderActiveView();
    }

    function renderChrome() {
      renderValidation();
      document.getElementById("diff").textContent = state.diff || "No changes.";
      document.getElementById("publish").disabled = !state.validation.valid || state.source_changed;
      document.getElementById("reload-source").classList.toggle("hidden", !state.source_changed);
      setStatus(state.dirty ? "Draft saved" : "Published");
    }

    function renderValidation() {
      const report = state.validation;
      const errors = report.errors.map(item => `<li>${escapeHtml(item)}</li>`).join("");
      const warnings = report.warnings.map(item => `<li>${escapeHtml(item)}</li>`).join("");
      const external = state.source_changed ? '<li>The published file changed outside this manager. Reload it to discard this draft and use the external version.</li>' : "";
      const valid = report.valid && !state.source_changed;
      document.getElementById("validation").innerHTML = `<div class="validation ${valid ? "" : "invalid"}"><strong>${valid ? "Draft is valid" : "Publishing is blocked"}</strong>${errors || external ? `<ul>${errors}${external}</ul>` : ""}${warnings ? `<ul class="warnings">${warnings}</ul>` : ""}</div>`;
    }

    function renderActiveView() {
      const content = document.getElementById("editor-content");
      if (activeView === "input") content.innerHTML = renderRuleSection("input", true);
      if (activeView === "context") content.innerHTML = renderRuleSection("context", false);
      if (activeView === "retrieval") content.innerHTML = renderRetrievalOutput();
      if (activeView === "coverage") content.innerHTML = renderCoverage();
      if (activeView === "messages") content.innerHTML = renderMessages();
      if (activeView === "runtime") content.innerHTML = renderRuntime();
      if (activeView === "history") content.innerHTML = renderHistory();
      bindEditors();
    }

    function renderRuleSection(section, includeSimilarity) {
      const data = state.draft[section] || {};
      const types = [["rules", "Regex rules", "patterns"], ["fuzzy_rules", "Fuzzy rules", "phrases"]];
      if (includeSimilarity) types.push(["similarity_rules", "Embedding similarity rules", "examples"]);
      const blocking = section === "input" ? `<div class="section"><div class="section-heading"><div><h3>Blocking triggers</h3><p>One trigger name per line. Academic integrity normally redirects instead.</p></div></div><textarea class="list-input" data-path="input.blocking_triggers">${escapeHtml(lines(data.blocking_triggers || []))}</textarea></div>` : "";
      return blocking + types.map(([type, title, listKey]) => {
        const rules = data[type] || [];
        const rendered = rules.map((rule, index) => `<div class="rule"><div class="rule-grid"><div class="field"><label>Trigger</label><input type="text" data-path="${section}.${type}.${index}.trigger" value="${escapeHtml(rule.trigger)}"></div><div class="field"><label>${listKey === "patterns" ? "Patterns" : listKey === "phrases" ? "Phrases" : "Examples"} · one per line</label><textarea data-path="${section}.${type}.${index}.${listKey}">${escapeHtml(lines(rule[listKey]))}</textarea></div><button class="button danger remove-rule" data-section="${section}" data-type="${type}" data-index="${index}" type="button">Remove</button></div>${type !== "rules" ? `<div class="threshold"><input type="range" min="0.01" max="1" step="0.01" data-path="${section}.${type}.${index}.threshold" value="${rule.threshold ?? (type === "fuzzy_rules" ? .88 : .5)}"><input type="number" min="0.01" max="1" step="0.01" data-path="${section}.${type}.${index}.threshold" value="${rule.threshold ?? (type === "fuzzy_rules" ? .88 : .5)}"></div>` : ""}</div>`).join("");
        return `<section class="section"><div class="section-heading"><div><h3>${title}</h3><p>${type === "similarity_rules" ? "BGE-M3 thresholds must be calibrated outside this preview." : "Rules execute before model classification."}</p></div><button class="button add-rule" data-section="${section}" data-type="${type}" data-list-key="${listKey}" type="button">Add rule</button></div>${rendered || '<div class="empty">No custom rules in this section.</div>'}</section>`;
      }).join("");
    }

    function renderRetrievalOutput() {
      const visibility = state.draft.retrieval?.allowed_visibility || ["public"];
      const requireCitations = state.draft.output?.require_citations ?? true;
      return `<section class="section"><div class="section-heading"><div><h3>Allowed document visibility</h3><p>Only matching corpus documents may enter retrieval.</p></div></div><textarea class="list-input" data-path="retrieval.allowed_visibility">${escapeHtml(lines(visibility))}</textarea></section><section class="section"><div class="section-heading"><div><h3>Citation boundary</h3><p>Require supporting course citations before returning an answer.</p></div></div><label class="inline"><input class="switch" type="checkbox" data-path="output.require_citations" ${requireCitations ? "checked" : ""}> Require citations</label></section>${renderRuleSection("output", false)}`;
    }

    function renderCoverage() {
      const rows = (state.draft.coverage_cases || []).map((item, index) => `<div class="coverage-row"><div class="field"><label>Case ID</label><input type="text" data-path="coverage_cases.${index}.case_id" value="${escapeHtml(item.case_id)}"></div><div class="field"><label>Family</label><input type="text" data-path="coverage_cases.${index}.family" value="${escapeHtml(item.family)}"></div><div class="field"><label>Role</label><select data-path="coverage_cases.${index}.coverage_role">${["positive_direct", "positive_variant", "benign_near_miss"].map(role => `<option ${item.coverage_role === role ? "selected" : ""}>${role}</option>`).join("")}</select></div><div class="field"><label>Test request</label><textarea data-path="coverage_cases.${index}.text">${escapeHtml(item.text)}</textarea><label>Expected triggers · one per line</label><textarea data-path="coverage_cases.${index}.expected_triggers">${escapeHtml(lines(item.expected_triggers))}</textarea></div><button class="button danger remove-coverage" data-index="${index}" type="button">Remove</button></div>`).join("");
      return `<section class="section"><div class="section-heading"><div><h3>Rule-family checks</h3><p>Coverage is a publish gate. It prevents attack recall from improving by simply over-blocking safe requests.</p></div><div><button class="button" id="run-coverage" type="button">Run coverage preview</button> <button class="button" id="add-coverage" type="button">Add case</button></div></div><div id="coverage-result"></div>${rows || '<div class="empty">No coverage cases defined.</div>'}</section>`;
    }

    function renderMessages() {
      const defaults = {input_block: "Input block", output_block: "Output block", ungrounded: "Insufficient evidence", integrity_safe: "Academic integrity redirect"};
      return `<section class="section">${Object.entries(defaults).map(([key, label]) => `<div class="field" style="margin-bottom:16px"><label>${label}</label><textarea data-path="messages.${key}">${escapeHtml(state.draft.messages?.[key] || "")}</textarea></div>`).join("")}</section>`;
    }

    function renderRuntime() {
      const runtime = state.runtime;
      const rows = [
        ["Embedding", runtime.models.embedding], ["Answer", runtime.models.answer],
        ["Classifier", runtime.models.classifier], ["Entailment", runtime.models.entailment],
        ["Judge", runtime.models.judge], ["Evidence threshold", runtime.thresholds.evidence_min_score],
        ["Classifier confidence", runtime.thresholds.classifier_min_confidence],
        ["Entailment confidence", runtime.thresholds.entailment_min_confidence],
        ["Retrieval top-k", runtime.retrieval.top_k], ["Config SHA-256", runtime.sha256]
      ];
      return `<section class="section"><table class="runtime-table"><tbody>${rows.map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td class="mono">${escapeHtml(value)}</td></tr>`).join("")}</tbody></table><p class="source" style="display:block;margin:14px 0">Runtime controls are versioned separately. Restart commands after publishing a runtime-config change.</p></section>`;
    }

    function renderHistory() {
      return `<section class="section">${state.versions.map(version => `<div class="history-row"><div><strong>Version ${version.version_id}</strong><br><span class="mono">${escapeHtml(version.sha256)}</span><br><small>${escapeHtml(version.created_at)} · ${escapeHtml(version.reason)}</small></div><button class="button rollback" data-version="${version.version_id}" type="button">Restore</button></div>`).join("") || '<div class="empty">No published snapshots yet.</div>'}</section>`;
    }

    function bindEditors() {
      document.querySelectorAll("[data-path]").forEach(element => element.addEventListener("input", () => {
        let value = element.type === "checkbox" ? element.checked : element.value;
        if (element.tagName === "TEXTAREA" && ["patterns", "phrases", "examples", "blocking_triggers", "allowed_visibility", "expected_triggers"].includes(element.dataset.path.split(".").at(-1))) value = fromLines(value);
        if (element.type === "range" || element.type === "number") value = Number(value);
        setPath(state.draft, element.dataset.path, value);
        if (element.type === "range") {
          const peer = [...document.querySelectorAll(`[data-path="${element.dataset.path}"]`)].find(item => item !== element);
          if (peer) peer.value = value;
        }
        scheduleSave();
      }));
      document.querySelectorAll(".add-rule").forEach(button => button.addEventListener("click", () => {
        const rule = {trigger: "new_guardrail", [button.dataset.listKey]: [""]};
        if (button.dataset.type !== "rules") rule.threshold = button.dataset.type === "fuzzy_rules" ? .88 : .5;
        state.draft[button.dataset.section] ||= {};
        state.draft[button.dataset.section][button.dataset.type] ||= [];
        state.draft[button.dataset.section][button.dataset.type].push(rule);
        renderActiveView(); scheduleSave();
      }));
      document.querySelectorAll(".remove-rule").forEach(button => button.addEventListener("click", () => {
        state.draft[button.dataset.section][button.dataset.type].splice(Number(button.dataset.index), 1);
        renderActiveView(); scheduleSave();
      }));
      document.getElementById("add-coverage")?.addEventListener("click", () => {
        state.draft.coverage_cases ||= [];
        state.draft.coverage_cases.push({case_id: "new-case", family: "new_guardrail", coverage_role: "positive_direct", text: "", expected_triggers: ["new_guardrail"]});
        renderActiveView(); scheduleSave();
      });
      document.getElementById("run-coverage")?.addEventListener("click", runCoverage);
      document.querySelectorAll(".remove-coverage").forEach(button => button.addEventListener("click", () => {
        state.draft.coverage_cases.splice(Number(button.dataset.index), 1);
        renderActiveView(); scheduleSave();
      }));
      document.querySelectorAll(".rollback").forEach(button => button.addEventListener("click", () => rollback(Number(button.dataset.version))));
    }

    function setPath(target, path, value) {
      const keys = path.split("."); let cursor = target;
      keys.slice(0, -1).forEach((key, index) => {
        const normalizedKey = Number.isNaN(Number(key)) ? key : Number(key);
        if (cursor[normalizedKey] === undefined) {
          const nextKey = keys[index + 1];
          cursor[normalizedKey] = Number.isNaN(Number(nextKey)) ? {} : [];
        }
        cursor = cursor[normalizedKey];
      });
      const finalKey = keys.at(-1); cursor[Number.isNaN(Number(finalKey)) ? finalKey : Number(finalKey)] = value;
    }

    function scheduleSave() {
      setStatus("Unsaved changes"); clearTimeout(saveTimer);
      saveTimer = setTimeout(saveDraft, 500);
    }

    async function saveDraft() {
      clearTimeout(saveTimer); setStatus("Saving...");
      try { state = await request("/api/draft", {method: "POST", body: JSON.stringify({document: state.draft})}); renderChrome(); }
      catch (error) { setStatus(error.message, true); }
    }

    async function publish() {
      await saveDraft();
      if (!state.validation.valid) return;
      if (!window.confirm("Publish this validated draft and create a rollback snapshot?")) return;
      try { state = await request("/api/publish", {method: "POST", body: "{}"}); render(); }
      catch (error) { setStatus(error.message, true); }
    }

    async function rollback(versionId) {
      if (!window.confirm(`Restore policy version ${versionId}? The current policy will also be snapshotted.`)) return;
      try { state = await request("/api/rollback", {method: "POST", body: JSON.stringify({version_id: versionId})}); render(); }
      catch (error) { setStatus(error.message, true); }
    }

    async function reloadSource() {
      if (!window.confirm("Discard the current draft and reload the published policy file?")) return;
      try { state = await request("/api/reload", {method: "POST", body: "{}"}); render(); }
      catch (error) { setStatus(error.message, true); }
    }

    async function simulate() {
      const resultElement = document.getElementById("simulation-result");
      resultElement.textContent = "Running local checks...";
      try {
        const result = await request("/api/simulate", {method: "POST", body: JSON.stringify({text: document.getElementById("simulation-text").value, stage: document.getElementById("simulation-stage").value})});
        const triggers = result.triggers.map(item => `<span class="trigger">${escapeHtml(item)}</span>`).join("") || "<span>No guard trigger matched.</span>";
        const scores = (result.similarity || []).map(item => `<div class="score"><strong>${escapeHtml(item.trigger)}</strong><br>${item.score} / ${item.threshold} ${item.matched ? "· matched" : ""}</div>`).join("");
        resultElement.innerHTML = `<div class="disposition">${escapeHtml(result.disposition)}</div><div>${triggers}</div>${scores}<p><small>${result.remote_calls} remote calls${result.similarity_provider ? ` · ${escapeHtml(result.similarity_provider)}` : ""}</small></p>`;
      } catch (error) { resultElement.innerHTML = `<div class="validation invalid">${escapeHtml(error.message)}</div>`; }
    }

    async function runCoverage() {
      await saveDraft();
      const element = document.getElementById("coverage-result");
      if (!element) return;
      element.innerHTML = '<div class="validation">Running local coverage preview...</div>';
      try {
        const report = await request("/api/coverage", {method: "POST", body: "{}"});
        const failures = report.results.filter(item => !item.passed).map(item => `<li>${escapeHtml(item.case_id)}: expected ${escapeHtml(item.expected_triggers.join(", ") || "no triggers")}, got ${escapeHtml(item.actual_triggers.join(", ") || "no triggers")}</li>`).join("");
        element.innerHTML = `<div class="validation ${report.failed ? "invalid" : ""}"><strong>${report.passed}/${report.total} local preview cases passed</strong>${failures ? `<ul>${failures}</ul>` : ""}<p><small>${report.remote_calls} remote calls · ${escapeHtml(report.similarity_provider)}</small></p></div>`;
      } catch (error) { element.innerHTML = `<div class="validation invalid">${escapeHtml(error.message)}</div>`; }
    }

    function setStatus(message, error = false) { const element = document.getElementById("save-status"); element.textContent = message; element.style.color = error ? "var(--danger)" : ""; }

    document.querySelectorAll(".nav-button").forEach(button => button.addEventListener("click", () => { activeView = button.dataset.view; render(); }));
    document.getElementById("save-now").addEventListener("click", saveDraft);
    document.getElementById("reload-source").addEventListener("click", reloadSource);
    document.getElementById("publish").addEventListener("click", publish);
    document.getElementById("simulate").addEventListener("click", simulate);
    loadState().catch(error => setStatus(error.message, true));
  </script>
</body>
</html>
"""
