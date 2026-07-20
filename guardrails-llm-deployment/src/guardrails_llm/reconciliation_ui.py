from __future__ import annotations


RECONCILIATION_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Judge Review Reconciliation</title>
  <style>
    :root {
      --bg: #f4f6f7; --surface: #fff; --muted: #5d6b72; --text: #172126;
      --border: #d7dfe2; --accent: #087f8c; --soft: #e8f4f5;
      --yes: #276b3d; --no: #a8322d; --radius: 6px;
      font: 15px/1.45 Inter, ui-sans-serif, system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text); background: var(--bg); }
    button, input, textarea { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    header {
      position: sticky; top: 0; z-index: 4; display: flex; flex-wrap: wrap;
      align-items: center; gap: 18px; min-height: 62px; padding: 10px 20px;
      background: var(--surface); border-bottom: 1px solid var(--border);
    }
    header h1 { margin: 0; font-size: 1.15rem; }
    .status { margin-left: auto; color: var(--muted); }
    .layout { display: grid; grid-template-columns: 260px minmax(0, 1fr); }
    aside {
      position: sticky; top: 62px; height: calc(100vh - 62px); overflow: auto;
      padding: 16px 12px; background: var(--surface); border-right: 1px solid var(--border);
    }
    aside label { display: block; margin: 0 6px 12px; color: var(--muted); }
    .section {
      width: 100%; display: flex; justify-content: space-between; gap: 8px;
      margin: 3px 0; padding: 9px; border: 1px solid transparent;
      border-radius: var(--radius); background: transparent; text-align: left;
    }
    .section.active { background: var(--soft); border-color: #acd3d7; }
    main { min-width: 0; padding: 22px clamp(14px, 3vw, 40px) 70px; }
    .item {
      margin-bottom: 14px; border: 1px solid var(--border);
      border-radius: var(--radius); background: var(--surface);
    }
    .item > summary { padding: 14px 16px; cursor: pointer; font-weight: 650; }
    .item-body { padding: 0 16px 18px; }
    .context {
      display: grid; grid-template-columns: minmax(0, .75fr) minmax(0, 1.25fr);
      gap: 14px; padding: 14px; background: #f8fafb; border: 1px solid var(--border);
    }
    pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }
    .evidence { margin-top: 8px; }
    .evidence pre { margin-top: 7px; padding: 9px; border-left: 3px solid #afd5d8; color: #405159; }
    .reviews { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .review { padding: 12px; border: 1px solid var(--border); border-radius: var(--radius); }
    .review h3 { margin: 0 0 8px; font-size: .9rem; }
    .labels { display: grid; grid-template-columns: 1fr auto; gap: 4px 8px; font-size: .82rem; }
    .yes { color: var(--yes); } .no { color: var(--no); }
    .rationale { margin-top: 8px; color: var(--muted); font-size: .82rem; white-space: pre-wrap; }
    .adjudication { margin-top: 12px; padding: 12px; border: 2px solid #acd3d7; border-radius: var(--radius); }
    .dimensions { display: grid; grid-template-columns: repeat(5, minmax(110px, 1fr)); gap: 8px; }
    fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
    legend { min-height: 38px; font-size: .78rem; font-weight: 650; }
    .choice { display: grid; grid-template-columns: 1fr 1fr; }
    .choice label { position: relative; }
    .choice input { position: absolute; opacity: 0; }
    .choice span { display: grid; place-items: center; min-height: 32px; border: 1px solid var(--border); }
    .choice input:checked + span { background: var(--soft); border-color: var(--accent); }
    .adjudication-fields { display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 10px; margin-top: 10px; }
    .adjudication-fields input, .adjudication-fields textarea {
      width: 100%; min-height: 38px; padding: 8px; border: 1px solid var(--border); border-radius: var(--radius);
    }
    .adjudication-fields textarea { min-height: 70px; resize: vertical; }
    .hidden { display: none !important; }
    @media (max-width: 1050px) {
      .reviews { grid-template-columns: 1fr; }
      .dimensions { grid-template-columns: repeat(3, minmax(110px, 1fr)); }
    }
    @media (max-width: 760px) {
      .layout { display: block; } aside { position: static; height: auto; border-right: 0; }
      .context { grid-template-columns: 1fr; } .dimensions { grid-template-columns: 1fr; }
      .adjudication-fields { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header><h1>Judge Review Reconciliation</h1><span id="progress"></span><span class="status" id="status">Loading...</span></header>
  <div class="layout">
    <aside><label><input type="checkbox" id="disagreements-only" checked> Show disagreements only</label><nav id="sections"></nav></aside>
    <main><h2 id="title">Reconciliation section</h2><div id="items"></div></main>
  </div>
  <script>
    const dimensions = [
      ["grounded", "Grounded"], ["privacy_safe", "Privacy safe"],
      ["injection_safe", "Injection safe"], ["integrity_safe", "Integrity safe"],
      ["refusal_appropriate", "Refusal appropriate"]
    ];
    let state = null;
    let section = null;
    let currentSection = null;
    const timers = new Map();
    const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
    async function request(path, options = {}) {
      const response = await fetch(path, {headers: {"Content-Type": "application/json"}, ...options});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }
    function reviewCard(title, review) {
      const labels = dimensions.map(([key, label]) =>
        `<span>${label}</span><strong class="${review[key] ? "yes" : "no"}">${review[key] ? "Yes" : "No"}</strong>`
      ).join("");
      return `<div class="review"><h3>${escapeHtml(title)}</h3><div class="labels">${labels}</div><div class="rationale">${escapeHtml(review.rationale || "No rationale provided.")}</div></div>`;
    }
    function evidence(item) {
      const rows = item.retrieved_evidence || [];
      return `<details class="evidence"><summary>Retrieved evidence (${rows.length})</summary>${rows.map((row) =>
        `<pre><strong>${escapeHtml(row.chunk_id || "Chunk")}</strong>\n${escapeHtml(row.text || "")}</pre>`
      ).join("")}</details>`;
    }
    function adjudication(item) {
      if (!item.requires_adjudication) return "";
      const value = item.adjudication || {};
      const controls = dimensions.map(([key, label]) => `<fieldset><legend>${label}</legend><div class="choice">
        <label><input type="radio" name="${item.item_id}-${key}" value="true"${value[key] === true ? " checked" : ""}><span>Yes</span></label>
        <label><input type="radio" name="${item.item_id}-${key}" value="false"${value[key] === false ? " checked" : ""}><span>No</span></label>
      </div></fieldset>`).join("");
      return `<div class="adjudication"><strong>Final adjudication</strong><div class="dimensions">${controls}</div>
        <div class="adjudication-fields"><input data-field="adjudicator_id" placeholder="Adjudicator ID" value="${escapeHtml(value.adjudicator_id || "")}">
        <textarea data-field="rationale" placeholder="Adjudication rationale (required)">${escapeHtml(value.rationale || "")}</textarea></div></div>`;
    }
    function render() {
      const only = document.getElementById("disagreements-only").checked;
      document.getElementById("items").innerHTML = section.items.map((item) => {
        if (only && !item.requires_adjudication) return "";
        return `<details class="item" data-item="${escapeHtml(item.item_id)}" open><summary>${escapeHtml(item.question)} · expected ${escapeHtml(item.expected_behavior)} / actual ${escapeHtml(item.actual_behavior)}</summary>
          <div class="item-body"><div class="context"><div><strong>System answer</strong><pre>${escapeHtml(item.answer)}</pre></div><div>${evidence(item)}</div></div>
          <div class="reviews">${reviewCard("Reviewer A", item.reviewer_a)}${reviewCard("Reviewer B", item.reviewer_b)}${reviewCard("Rubric recommendation (not ground truth)", item.recommendation)}</div>
          ${adjudication(item)}</div></details>`;
      }).join("") || "<p>No items match this filter.</p>";
      bind();
    }
    function bind() {
      document.querySelectorAll(".item[data-item]").forEach((node) => {
        const itemId = node.dataset.item;
        node.querySelectorAll('.adjudication input[type="radio"]').forEach((input) => input.addEventListener("change", () => {
          const key = input.name.slice(itemId.length + 1);
          save(itemId, {[key]: input.value === "true"});
        }));
        node.querySelectorAll(".adjudication [data-field]").forEach((input) => input.addEventListener("input", () => {
          const key = `${itemId}:${input.dataset.field}`;
          clearTimeout(timers.get(key));
          timers.set(key, setTimeout(() => save(itemId, {[input.dataset.field]: input.value}), 450));
        }));
      });
    }
    async function save(itemId, changes) {
      document.getElementById("status").textContent = "Saving...";
      try {
        const result = await request(`/api/adjudications/${encodeURIComponent(itemId)}`, {method: "PATCH", body: JSON.stringify(changes)});
        const index = section.items.findIndex((item) => item.item_id === itemId);
        section.items[index] = result.item;
        state.progress = result.progress;
        updateProgress();
        document.getElementById("status").textContent = "Autosaved locally";
      } catch (error) { document.getElementById("status").textContent = error.message; }
    }
    function updateProgress() {
      document.getElementById("progress").textContent = `${state.progress.completed} / ${state.progress.total} disagreements adjudicated`;
    }
    async function loadSection(sectionId) {
      currentSection = sectionId;
      section = await request(`/api/sections/${encodeURIComponent(sectionId)}`);
      document.getElementById("title").textContent = section.section.title;
      document.querySelectorAll(".section").forEach((button) => button.classList.toggle("active", button.dataset.section === sectionId));
      render();
    }
    async function initialize() {
      state = await request("/api/state");
      updateProgress();
      document.getElementById("sections").innerHTML = state.sections.map((entry) =>
        `<button class="section" data-section="${entry.section_id}"><span>${entry.title}</span><span>${entry.remaining}/${entry.disagreements}</span></button>`
      ).join("");
      document.querySelectorAll(".section").forEach((button) => button.addEventListener("click", () => loadSection(button.dataset.section)));
      const first = state.sections.find((entry) => entry.remaining) || state.sections[0];
      await loadSection(first.section_id);
      document.getElementById("status").textContent = "Autosaved locally";
    }
    document.getElementById("disagreements-only").addEventListener("change", render);
    initialize();
  </script>
</body>
</html>
"""
