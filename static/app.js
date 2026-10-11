/**
 * Churnable — Single-Column ChatGPT-Style AI Chatbot Engine
 * Powers conversational customer churn investigation, tool execution traces,
 * interactive in-chat graphs (Chart.js), predictive tables, and full dataset browsing.
 */

let conversationHistory = [];
let isSending = false;
let messageCounter = 0;

// Preview Modal State
let previewPage = 1;
let previewSearch = "";
let previewTotalPages = 1;
let previewSearchTimeout = null;

document.addEventListener("DOMContentLoaded", () => {
  initChatApp();
});

function initChatApp() {
  setupNavbarActions();
  setupChatForm();
  setupPromptCards();
  setupModals();
  loadDatasetMeta();
}

/** -------------------------------------------------------------------------
 * Navbar Actions
 * -------------------------------------------------------------------------- */
function setupNavbarActions() {
  const btnNewChat = document.getElementById("btn-new-chat");
  if (btnNewChat) {
    btnNewChat.addEventListener("click", () => {
      startNewInvestigation();
    });
  }
}

function startNewInvestigation() {
  conversationHistory = [];
  const stream = document.getElementById("messages-stream");
  const hero = document.getElementById("chat-hero");
  if (stream) stream.innerHTML = "";
  if (hero) hero.style.display = "flex";
  showToast("Started new churn investigation.");
}

/** -------------------------------------------------------------------------
 * Prompt Cards Setup (Hero State)
 * -------------------------------------------------------------------------- */
function setupPromptCards() {
  document.querySelectorAll(".prompt-card").forEach(card => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      if (prompt && !isSending) {
        submitUserQuery(prompt);
      }
    });
  });
}

/** -------------------------------------------------------------------------
 * Chat Form & Textarea
 * -------------------------------------------------------------------------- */
function setupChatForm() {
  const form = document.getElementById("chat-form");
  const textarea = document.getElementById("chat-textarea");

  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      handleFormSubmit();
    });
  }

  if (textarea) {
    textarea.addEventListener("input", () => {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
      const btn = document.getElementById("btn-send-message");
      if (btn) btn.disabled = !textarea.value.trim();
    });

    textarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleFormSubmit();
      }
    });
  }
}

function handleFormSubmit() {
  const textarea = document.getElementById("chat-textarea");
  if (!textarea || isSending) return;
  const message = textarea.value.trim();
  if (!message) return;

  textarea.value = "";
  textarea.style.height = "auto";
  const btn = document.getElementById("btn-send-message");
  if (btn) btn.disabled = true;

  submitUserQuery(message);
}

/** -------------------------------------------------------------------------
 * Query Submission & Multi-Turn Execution
 * -------------------------------------------------------------------------- */
async function submitUserQuery(text) {
  if (isSending) return;
  isSending = true;

  // Hide hero if visible
  const hero = document.getElementById("chat-hero");
  if (hero) hero.style.display = "none";

  // Append User message row
  appendUserMessage(text);

  // Show thinking state
  const thinkingState = document.getElementById("thinking-state");
  if (thinkingState) thinkingState.style.display = "flex";
  scrollToBottom();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: conversationHistory
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Agent investigation request failed");
    }

    const data = await res.json();

    // Hide thinking state
    if (thinkingState) thinkingState.style.display = "none";

    // Append Assistant response row with interactive graphs & table
    appendAgentResponse(data);

    // Save to multi-turn conversation memory
    conversationHistory.push(
      { role: "user", content: text },
      { role: "assistant", content: data.reply }
    );

  } catch (error) {
    if (thinkingState) thinkingState.style.display = "none";
    appendErrorMessage(error.message);
    showToast(`Error: ${error.message}`);
  } finally {
    isSending = false;
    scrollToBottom();
  }
}

/** -------------------------------------------------------------------------
 * DOM Rendering: User & Agent Messages
 * -------------------------------------------------------------------------- */
function appendUserMessage(text) {
  const stream = document.getElementById("messages-stream");
  const row = document.createElement("div");
  row.className = "message-row user-row";
  row.innerHTML = `
    <div class="user-bubble-capsule">${escapeHtml(text)}</div>
  `;
  stream.appendChild(row);
}

function appendAgentResponse(data) {
  messageCounter++;
  const msgId = messageCounter;
  const stream = document.getElementById("messages-stream");
  const row = document.createElement("div");
  row.className = "message-row agent-row";

  // 1. Tool Trace Accordion
  let toolTraceHtml = "";
  if (data.activity_log && data.activity_log.length > 0) {
    const toolCount = data.activity_log.length;
    const stepsHtml = data.activity_log.map(s => `
      <div class="trace-step-row">
        <span class="trace-step-tag">${escapeHtml(s.tool || "tool")}</span>
        <span class="trace-step-detail">${escapeHtml(s.summary || JSON.stringify(s.args || {}))}</span>
      </div>
    `).join("");

    toolTraceHtml = `
      <div class="tool-trace-card">
        <div class="tool-trace-head" onclick="toggleTraceAccordion(this)">
          <div class="tool-trace-head-left">
            <span class="trace-chevron">›</span>
            <span>Executed ${toolCount} investigative tool(s) with LangGraph</span>
          </div>
          <span style="font-size: 11px; color: var(--text-muted);">Details</span>
        </div>
        <div class="tool-trace-steps" style="display: none;">
          ${stepsHtml}
        </div>
      </div>
    `;
  }

  // 2. In-Chat Interactive Graphs Grid (from agent's charts array)
  let chartsHtml = "";
  const agentCharts = data.charts || [];
  const hasRiskDist = !!(data.risk_distribution && Object.keys(data.risk_distribution).length > 0);
  const hasFactors = !!(data.factors_breakdown && Object.keys(data.factors_breakdown).length > 0);

  if (agentCharts.length > 0) {
    chartsHtml = `
      <div class="in-chat-charts-grid">
        ${agentCharts.map((chart, idx) => `
          <div class="in-chat-chart-card">
            <div class="chart-card-heading">${escapeHtml(chart.title || `Chart ${idx + 1}`)}</div>
            <div class="chart-canvas-box">
              <canvas id="chart-${chart.id}-${msgId}"></canvas>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  } else if (hasRiskDist || hasFactors) {
    // Fallback to old format for backward compatibility
    chartsHtml = `
      <div class="in-chat-charts-grid">
        <div class="in-chat-chart-card">
          <div class="chart-card-heading">Risk Cohort Distribution</div>
          <div class="chart-canvas-box">
            <canvas id="chart-risk-${msgId}"></canvas>
          </div>
        </div>
        <div class="in-chat-chart-card">
          <div class="chart-card-heading">Primary Churn Factors Prevalence</div>
          <div class="chart-canvas-box">
            <canvas id="chart-factors-${msgId}"></canvas>
          </div>
        </div>
      </div>
    `;
  }

  // 3. In-Chat Customer Churn Table
  let tableHtml = "";
  if (data.agent_predictions && data.agent_predictions.length > 0) {
    const rowsHtml = data.agent_predictions.slice(0, 10).map(c => {
      const cid = c.customer_id || "N/A";
      const score = Math.round(Number(c.risk_score || 0));
      const tier = c.risk_category || (score >= 60 ? "High" : (score >= 35 ? "Medium" : "Low"));
      const tierClass = tier.toLowerCase();
      const drivers = c.primary_drivers || "Telemetry drop";
      const action = c.recommended_action || "Targeted retention outreach";
      const days = c.days_since_last_transaction !== undefined ? `${c.days_since_last_transaction}d` : "--";

      return `
        <tr class="clickable-row" onclick="inquireCustomerInChat('${cid}')">
          <td><strong>${cid}</strong></td>
          <td><strong>${score}</strong><span style="font-size: 11px; color: var(--text-muted);">/100</span></td>
          <td><span class="risk-pill ${tierClass}">${tier}</span></td>
          <td>${days}</td>
          <td style="max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(drivers)}">${escapeHtml(drivers)}</td>
          <td style="max-width: 190px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #4B5563;" title="${escapeHtml(action)}">${escapeHtml(action)}</td>
        </tr>
      `;
    }).join("");

    tableHtml = `
      <div class="in-chat-table-wrapper">
        <table class="clean-table">
          <thead>
            <tr>
              <th>Customer ID</th>
              <th>Risk Score</th>
              <th>Tier</th>
              <th>Inactivity</th>
              <th>Primary Churn Factors</th>
              <th>Recommended Retention Action</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      </div>
    `;
  }

  // 4. Follow-up suggestion chips
  const followUpHtml = `
    <div class="follow-up-chips-row">
      <button class="follow-up-chip" onclick="submitUserQuery('Deep dive into the top at-risk customer account: examine transactions, balances, and retention plan.')">
        Investigate top at-risk account
      </button>
      <button class="follow-up-chip" onclick="submitUserQuery('What retention offer or campaign should we send to accounts with sharp transaction declines?')">
        Formulate retention campaign
      </button>
      <button class="follow-up-chip" onclick="submitUserQuery('Compare churn risk between active accounts versus accounts with payment failures.')">
        Compare failure vs active cohorts
      </button>
    </div>
  `;

  row.innerHTML = `
    <div class="agent-row-header">
      <div class="agent-avatar-circle">CB</div>
      <div class="agent-author-name">Churnable</div>
    </div>
    ${toolTraceHtml}
    <div class="agent-message-content">
      ${formatMarkdown(data.reply)}
    </div>
    ${chartsHtml}
    ${tableHtml}
    ${followUpHtml}
  `;

  stream.appendChild(row);

  // Render Chart.js canvases after DOM append
  // 1. Structured charts array from agent response
  if (agentCharts.length > 0) {
    agentCharts.forEach(chart => {
      const canvasId = `chart-${chart.id}-${msgId}`;
      renderChartFromConfig(canvasId, chart);
    });
  } else {
    // Fallback to old format
    if (hasRiskDist) {
      renderInChatRiskChart(`chart-risk-${msgId}`, data.risk_distribution);
    }
    if (hasFactors) {
      renderInChatFactorsChart(`chart-factors-${msgId}`, data.factors_breakdown);
    }
  }

  // 2. Render any inline chart canvases extracted from the reply text
  renderInlineCharts(row);
}

function renderInChatRiskChart(canvasId, riskDist) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const highCnt = riskDist.high_risk?.count || 0;
  const medCnt = riskDist.medium_risk?.count || 0;
  const lowCnt = riskDist.low_risk?.count || 0;

  new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["High Risk (≥60)", "Medium Risk (35–59)", "Low Risk (<35)"],
      datasets: [{
        data: [highCnt, medCnt, lowCnt],
        backgroundColor: ["#EF4444", "#F59E0B", "#10B981"],
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "right",
          labels: {
            boxWidth: 10,
            font: { size: 11, family: "Inter" },
            color: "#4B5563"
          }
        }
      }
    }
  });
}

function renderPieChart(canvasId, data, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const defaults = {
    type: "pie",
    data: data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: {
            boxWidth: 10,
            font: { size: 11, family: "Inter" },
            color: "#4B5563"
          }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.raw} (${((ctx.raw / ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0)) * 100).toFixed(1)}%)`
          }
        }
      }
    }
  };

  const config = { ...defaults, ...options };
  config.options = { ...defaults.options, ...options.options };
  config.options.plugins = { ...defaults.options.plugins, ...options.options?.plugins };

  new Chart(canvas, config);
}

function renderBarChart(canvasId, data, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const defaults = {
    type: "bar",
    data: data,
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: "#F3F4F6" },
          ticks: { font: { size: 10 }, color: "#6B7280" }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10 }, color: "#374151" }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.raw}`
          }
        }
      }
    }
  };

  const config = { ...defaults, ...options };
  config.options = { ...defaults.options, ...options.options };
  config.options.scales = { ...defaults.options.scales, ...options.options?.scales };
  config.options.plugins = { ...defaults.options.plugins, ...options.options?.plugins };

  new Chart(canvas, config);
}

function renderLineChart(canvasId, data, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const defaults = {
    type: "line",
    data: data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      tension: 0.3,
      scales: {
        x: {
          grid: { color: "#F3F4F6" },
          ticks: { font: { size: 10 }, color: "#6B7280" }
        },
        y: {
          beginAtZero: true,
          grid: { color: "#F3F4F6" },
          ticks: { font: { size: 10 }, color: "#6B7280" }
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: { font: { size: 11 }, color: "#4B5563" }
        }
      }
    }
  };

  const config = { ...defaults, ...options };
  config.options = { ...defaults.options, ...options.options };
  config.options.scales = { ...defaults.options.scales, ...options.options?.scales };
  config.options.plugins = { ...defaults.options.plugins, ...options.options?.plugins };

  new Chart(canvas, config);
}

function renderInChatFactorsChart(canvasId, factors) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const labels = Object.keys(factors);
  const data = labels.map(k => factors[k].percent || 0);

  renderBarChart(canvasId, {
    labels: labels,
    datasets: [{
      label: "% Accounts Affected",
      data: data,
      backgroundColor: "#6366F1",
      borderRadius: 4
    }]
  }, {
    options: {
      scales: {
        x: { max: 100, ticks: { callback: v => `${v}%` } }
      },
      plugins: {
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.raw}% of evaluated accounts` }
        }
      }
    }
  });
}

function parseChartOptions(options) {
  if (!options) return {};
  const parsed = JSON.parse(JSON.stringify(options), (key, value) => {
    if (typeof value === 'string' && value.startsWith('function')) {
      try {
        return new Function('return ' + value)();
      } catch (e) {
        return value;
      }
    }
    return value;
  });
  return parsed;
}

function renderChartFromConfig(canvasId, chartConfig) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const type = chartConfig.type || "bar";
  const data = chartConfig.data || { labels: [], datasets: [] };
  const options = parseChartOptions(chartConfig.options || {});

  switch (type) {
    case "pie":
    case "doughnut":
      renderPieChart(canvasId, data, options);
      break;
    case "bar":
      renderBarChart(canvasId, data, options);
      break;
    case "line":
      renderLineChart(canvasId, data, options);
      break;
    default:
      renderBarChart(canvasId, data, options);
  }
}

function appendErrorMessage(msg) {
  const stream = document.getElementById("messages-stream");
  const row = document.createElement("div");
  row.className = "message-row agent-row";
  row.innerHTML = `
    <div class="agent-row-header">
      <div class="agent-avatar-circle">!</div>
      <div class="agent-author-name">Churnable Notice</div>
    </div>
    <div class="agent-message-content" style="color: #DC2626;">
      <p>Investigation could not complete: ${escapeHtml(msg)}</p>
    </div>
  `;
  stream.appendChild(row);
}

window.toggleTraceAccordion = function(headElem) {
  const steps = headElem.parentElement.querySelector(".tool-trace-steps");
  const chevron = headElem.querySelector(".trace-chevron");
  if (!steps) return;
  const isHidden = steps.style.display === "none";
  steps.style.display = isHidden ? "flex" : "none";
  if (chevron) chevron.textContent = isHidden ? "v" : "›";
};

window.inquireCustomerInChat = function(customerId) {
  submitUserQuery(`Deep dive into customer ${customerId}: examine transaction velocity, balance trend, and provide tailored retention steps.`);
};

function scrollToBottom() {
  const container = document.getElementById("chat-scroll-container");
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

/** -------------------------------------------------------------------------
 * Load Overview & Meta
 * -------------------------------------------------------------------------- */
async function loadDatasetMeta() {
  try {
    const res = await fetch("/api/overview");
    if (!res.ok) throw new Error("Failed to load overview");
    const data = await res.json();

    const ds = data.dataset_info;
    const model = data.gemini_model;

    const datasetPill = document.getElementById("dataset-pill");
    const engineBadge = document.getElementById("engine-badge");

    if (datasetPill && ds) {
      datasetPill.textContent = `${ds.total_rows.toLocaleString()} customers • data/customers.csv`;
    }
    if (engineBadge && model) {
      engineBadge.textContent = `LangGraph • ${model}`;
    }
  } catch (e) {
    console.error("Meta load error:", e);
  }
}

/** -------------------------------------------------------------------------
 * Full Dataset Browser Modal (All Customers with Pagination & Search)
 * -------------------------------------------------------------------------- */
async function fetchPreviewData(page = 1, search = "") {
  const thead = document.getElementById("preview-thead");
  const tbody = document.getElementById("preview-tbody");
  const pageInfo = document.getElementById("preview-page-info");
  const btnPrev = document.getElementById("btn-page-prev");
  const btnNext = document.getElementById("btn-page-next");

  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="15" class="table-empty">Loading customer records...</td></tr>`;
  }

  try {
    const url = `/api/data?page=${page}&page_size=50&search=${encodeURIComponent(search)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch customer data");
    const data = await res.json();

    previewPage = data.page;
    previewTotalPages = data.total_pages;

    if (pageInfo) {
      pageInfo.textContent = `Page ${data.page} of ${data.total_pages} (${data.total_records.toLocaleString()} accounts)`;
    }
    if (btnPrev) btnPrev.disabled = data.page <= 1;
    if (btnNext) btnNext.disabled = data.page >= data.total_pages;

    const cols = data.columns || [];
    const records = data.records || [];

    if (thead) {
      thead.innerHTML = `<tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
    }

    if (tbody) {
      if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="${cols.length || 10}" class="table-empty">No matching customer records found.</td></tr>`;
      } else {
        tbody.innerHTML = records.map(r => `
          <tr>${cols.map(c => `<td>${r[c] !== null && r[c] !== undefined ? escapeHtml(String(r[c])) : ""}</td>`).join("")}</tr>
        `).join("");
      }
    }

  } catch (e) {
    console.error("Fetch data error:", e);
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="15" class="table-empty">Error loading data: ${e.message}</td></tr>`;
    }
  }
}

function setupModals() {
  // Preview Data Modal
  const btnPreview = document.getElementById("btn-preview-data");
  const modalPreview = document.getElementById("modal-preview");
  const btnPreviewClose = document.getElementById("modal-preview-close");
  const searchInput = document.getElementById("preview-search-input");
  const btnPrev = document.getElementById("btn-page-prev");
  const btnNext = document.getElementById("btn-page-next");

  if (btnPreview && modalPreview) {
    btnPreview.addEventListener("click", () => {
      modalPreview.style.display = "flex";
      previewPage = 1;
      previewSearch = "";
      if (searchInput) searchInput.value = "";
      fetchPreviewData(1, "");
    });
  }

  if (btnPreviewClose && modalPreview) {
    btnPreviewClose.addEventListener("click", () => {
      modalPreview.style.display = "none";
    });
  }

  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (previewPage > 1) {
        fetchPreviewData(previewPage - 1, previewSearch);
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (previewPage < previewTotalPages) {
        fetchPreviewData(previewPage + 1, previewSearch);
      }
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      clearTimeout(previewSearchTimeout);
      previewSearchTimeout = setTimeout(() => {
        previewSearch = e.target.value.trim();
        fetchPreviewData(1, previewSearch);
      }, 250);
    });
  }

  // Upload Modal
  const btnUpload = document.getElementById("btn-upload-modal");
  const modalUpload = document.getElementById("modal-upload");
  const btnUploadClose = document.getElementById("modal-upload-close");
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("csv-file-input");
  const uploadSelectedBox = document.getElementById("upload-selected-box");
  const selectedFileName = document.getElementById("selected-file-name");
  const btnSubmitUpload = document.getElementById("btn-submit-upload");

  let selectedFile = null;

  if (btnUpload && modalUpload) {
    btnUpload.addEventListener("click", () => {
      modalUpload.style.display = "flex";
    });
  }

  if (btnUploadClose && modalUpload) {
    btnUploadClose.addEventListener("click", () => {
      modalUpload.style.display = "none";
    });
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "#6366F1";
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.style.borderColor = "#CBD5E1";
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "#CBD5E1";
      if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) {
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    if (!file.name.endsWith(".csv")) {
      showToast("Only CSV files are accepted.");
      return;
    }
    selectedFile = file;
    if (selectedFileName) selectedFileName.textContent = file.name;
    if (uploadSelectedBox) uploadSelectedBox.style.display = "flex";
  }

  if (btnSubmitUpload) {
    btnSubmitUpload.addEventListener("click", async () => {
      if (!selectedFile) return;
      const formData = new FormData();
      formData.append("file", selectedFile);

      btnSubmitUpload.textContent = "Uploading...";
      btnSubmitUpload.disabled = true;

      try {
        const res = await fetch("/api/upload-csv", { method: "POST", body: formData });
        if (!res.ok) throw new Error("Upload failed");
        modalUpload.style.display = "none";
        showToast("Dataset uploaded successfully.");
        loadDatasetMeta();
        if (modalPreview && modalPreview.style.display === "flex") {
          fetchPreviewData(1, "");
        }
      } catch (err) {
        showToast(`Upload error: ${err.message}`);
      } finally {
        btnSubmitUpload.textContent = "Upload";
        btnSubmitUpload.disabled = false;
      }
    });
  }

  // Regenerate Button
  const btnRegen = document.getElementById("btn-regenerate");
  if (btnRegen) {
    btnRegen.addEventListener("click", async () => {
      btnRegen.textContent = "Regenerating...";
      btnRegen.disabled = true;
      try {
        const res = await fetch("/api/regenerate-data", { method: "POST" });
        if (!res.ok) throw new Error("Regeneration failed");
        showToast("Synthetic dataset regenerated.");
        loadDatasetMeta();
        if (modalPreview && modalPreview.style.display === "flex") {
          fetchPreviewData(1, "");
        }
      } catch (e) {
        showToast(`Error: ${e.message}`);
      } finally {
        btnRegen.textContent = "Regenerate";
        btnRegen.disabled = false;
      }
    });
  }
}

/** -------------------------------------------------------------------------
 * Markdown Formatting Helper
 * -------------------------------------------------------------------------- */
function formatMarkdown(text) {
  if (!text) return "";
  let html = text;

  // --- Intercept ```json chart blocks BEFORE HTML escaping ---
  // Replace them with an inline <canvas> placeholder so Chart.js can render them
  let inlineChartIdx = 0;
  html = html.replace(/```json\s*([\s\S]*?)```/gi, (match, jsonStr) => {
    let parsed;
    try { parsed = JSON.parse(jsonStr.trim()); } catch (e) { return match; }
    // Only intercept if it looks like a chart config
    if (!parsed.chart_type && !parsed.type && !parsed.data) return match;
    const type = (parsed.chart_type || parsed.type || "bar").toLowerCase();
    const title = parsed.title || "";
    const canvasId = `inline-chart-${Date.now()}-${inlineChartIdx++}`;
    const configAttr = encodeURIComponent(JSON.stringify(parsed));
    return `<div class="in-chat-chart-card" style="margin:12px 0;">${title ? `<div class="chart-card-heading">${title}</div>` : ""}<div class="chart-canvas-box"><canvas id="${canvasId}" data-chart-config="${configAttr}"></canvas></div></div>`;
  });

  // Also handle ``` fenced code blocks (non-chart) as <pre><code>
  html = html.replace(/```([\w]*)\n?([\s\S]*?)```/gi, (match, lang, code) => {
    return `<pre><code>${code.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}</code></pre>`;
  });

  // Escape HTML tags in remaining plain text
  // (split around already-injected HTML to avoid double-escaping)
  html = escapeHtmlOutsideTags(html);

  // Headers (##, ###)
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  html = html.replace(/^# (.*$)/gim, "<h2>$1</h2>");

  // Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bullet Lists
  html = html.replace(/^\s*[-*]\s+(.*)$/gim, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gms, "<ul>$1</ul>");

  // Paragraphs
  html = html.replace(/\n\n+/g, "</p><p>");
  html = `<p>${html}</p>`;
  html = html.replace(/<p><\/p>/g, "");

  return html;
}

/** Escape HTML only outside already-injected HTML tags */
function escapeHtmlOutsideTags(str) {
  // Split on HTML tags, escape text nodes only
  return str.replace(/(<[^>]+>)|([^<]+)/g, (match, tag, text) => {
    if (tag) return tag;
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  });
}

/** Walk the rendered row and fire Chart.js for any inline canvas placeholders */
function renderInlineCharts(rowElem) {
  rowElem.querySelectorAll("canvas[data-chart-config]").forEach(canvas => {
    try {
      const config = JSON.parse(decodeURIComponent(canvas.dataset.chartConfig));
      const type = (config.chart_type || config.type || "bar").toLowerCase();
      const rawData = config.data || {};

      if (type === "pie" || type === "doughnut") {
        // Support { "Label": value, ... } flat data format
        let labels, values;
        if (Array.isArray(rawData.labels)) {
          labels = rawData.labels;
          values = (rawData.datasets && rawData.datasets[0] && rawData.datasets[0].data) || [];
        } else {
          labels = Object.keys(rawData);
          values = Object.values(rawData);
        }
        const colors = [
          "#6366f1","#8b5cf6","#a78bfa","#c4b5fd",
          "#06b6d4","#10b981","#f59e0b","#ef4444"
        ];
        new Chart(canvas, {
          type: type === "doughnut" ? "doughnut" : "pie",
          data: { labels, datasets: [{ data: values, backgroundColor: colors.slice(0, labels.length), borderWidth: 0 }] },
          options: { responsive: true, plugins: { legend: { position: "bottom", labels: { color: "#e2e8f0", font: { size: 11 } } } } }
        });
      } else if (type === "bar") {
        let labels, values;
        if (Array.isArray(rawData.labels)) {
          labels = rawData.labels;
          values = (rawData.datasets && rawData.datasets[0] && rawData.datasets[0].data) || [];
        } else {
          labels = Object.keys(rawData);
          values = Object.values(rawData);
        }
        new Chart(canvas, {
          type: "bar",
          data: { labels, datasets: [{ data: values, backgroundColor: "#6366f1", borderRadius: 4 }] },
          options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: "#94a3b8" } }, y: { ticks: { color: "#94a3b8" } } } }
        });
      }
    } catch(e) {
      console.warn("Inline chart render failed:", e);
    }
  });
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

function showToast(message) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 3500);
}
