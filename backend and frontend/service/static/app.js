const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const modeSelect = document.querySelector("#mode");
const topKSelect = document.querySelector("#top-k");
const rerankInput = document.querySelector("#rerank");
const statusBox = document.querySelector("#status");
const resultsBox = document.querySelector("#results");
const submitButton = document.querySelector("#submit");
const sourceMenu = document.querySelector("#source-menu");
const filterTrigger = document.querySelector("#filter-trigger");

const escapeHtml = (value) => {
  const node = document.createElement("div");
  node.textContent = String(value ?? "");
  return node.innerHTML;
};

function selectedSources() {
  return [...sourceMenu.querySelectorAll("input:checked")].map((input) => input.value);
}

function updateSourceLabel() {
  const selected = selectedSources();
  const total = sourceMenu.querySelectorAll("input").length;
  filterTrigger.textContent = selected.length === 0 || selected.length === total
    ? "All sources"
    : `${selected.length} selected`;
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error((await response.json()).detail || "Configuration unavailable");
    const config = await response.json();
    modeSelect.innerHTML = config.modes.map((mode) =>
      `<option value="${escapeHtml(mode.value)}">${escapeHtml(mode.label)}</option>`
    ).join("");
    modeSelect.value = document.body.dataset.defaultMode || config.default_mode;
    rerankInput.disabled = !config.reranker_enabled;
    rerankInput.title = config.reranker_enabled
      ? "Rerank the candidate pool with a cross-encoder"
      : "Enable RERANKER_ENABLED=true on the backend to use this";
    sourceMenu.innerHTML = config.sources.map((source) => `
      <label class="source-option">
        <input type="checkbox" value="${escapeHtml(source)}" checked />
        <span>${escapeHtml(source)}</span>
      </label>
    `).join("");
    sourceMenu.addEventListener("change", updateSourceLabel);
    updateSourceLabel();
  } catch (error) {
    statusBox.className = "status error";
    statusBox.textContent = error.message;
  }
}

filterTrigger.addEventListener("click", () => {
  sourceMenu.hidden = !sourceMenu.hidden;
});
document.addEventListener("click", (event) => {
  if (!sourceMenu.contains(event.target) && event.target !== filterTrigger) sourceMenu.hidden = true;
});

function renderResults(payload) {
  if (!payload.results.length) {
    resultsBox.innerHTML = "";
    statusBox.textContent = `No results · ${payload.latency_ms.toFixed(1)} ms`;
    return;
  }
  statusBox.className = "status";
  statusBox.textContent = `${payload.total} results · ${payload.mode} · ${payload.latency_ms.toFixed(1)} ms`;
  resultsBox.innerHTML = payload.results.map((item) => `
    <article class="result-card">
      <div class="rank">${item.rank}</div>
      <div>
        <h2>${escapeHtml(item.title)}</h2>
        <div class="meta">
          <span class="source">${escapeHtml(item.source)}</span>
          <span>${escapeHtml(item.doc_id)}</span>
          ${item.chunk_id ? `<span>chunk ${escapeHtml(item.chunk_id)}</span>` : ""}
          ${item.n_chunks ? `<span>${item.n_chunks} chunks</span>` : ""}
          ${item.char_len ? `<span>${item.char_len.toLocaleString()} chars</span>` : ""}
        </div>
        ${item.snippet ? `<p class="snippet">${escapeHtml(item.snippet)}</p>` : ""}
      </div>
      <div class="score">score ${Number(item.score).toFixed(5)}</div>
    </article>
  `).join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  statusBox.className = "status";
  statusBox.textContent = "Searching…";
  resultsBox.innerHTML = "";
  const sources = selectedSources();
  const allSources = sourceMenu.querySelectorAll("input").length;
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryInput.value,
        mode: modeSelect.value,
        top_k: Number(topKSelect.value),
        sources: sources.length === allSources ? null : sources,
        rerank: rerankInput.checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Search failed");
    renderResults(payload);
  } catch (error) {
    statusBox.className = "status error";
    statusBox.textContent = error.message;
  } finally {
    submitButton.disabled = false;
  }
});

loadConfig();
