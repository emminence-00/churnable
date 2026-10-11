# 🛡️ ChurnGuard AI — Agentic Customer Churn Prediction (LangGraph Edition)

> An Agentic AI investigation system for digital banking, mobile wallets, and FinTech applications built with **LangGraph**, **Google Gemini (`langchain-google-genai`)**, **FastAPI**, **Pandas**, and a clean, responsive single-page dashboard.

---

## 🎓 University Project Overview

**ChurnGuard AI** is powered by a stateful **LangGraph** multi-step agent workflow. Rather than a standard chatbot or static pipeline, it utilizes a cyclical state graph:

```
          [ START ]
              │
              ▼
       ┌──────────────┐
       │  agent_node  │ ◄───────┐
       └──────┬───────┘         │
              │ (conditional)   │
      tool_calls?               │
        ├── Yes ──► ┌───────────┴┐
        │           │ tools_node │
        └── No      └────────────┘
         │
         ▼
      [ END ]
```

When given an objective (e.g., *"Why are customers becoming inactive?"* or *"Are falling balances causing churn?"*), the LangGraph agent:

1. **Inspects** dataset dimensions, schema, and column availability.
2. **Audits data quality**, identifying missing columns, null values, or absence of historical churn labels.
3. **Calls approved Python analytical tools dynamically** through LangGraph tool nodes (up to 6 investigative rounds).
4. **Inspects tool returns** and conditionally decides what to analyze next.
5. **Synthesizes an evidence-based final report** detailing empirical findings, methodological limitations, and actionable retention strategies.

---

## 🛠️ Technology Stack

- **Agent Framework**: LangGraph (`StateGraph`, cyclical execution, state memory)
- **AI Agent**: Google Gemini via `langchain-google-genai` (official SDK backend)
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Data Engine**: Pandas & NumPy (clean behavioral analytics)
- **Frontend**: Clean Vanilla HTML5, CSS3 (FinTech design system), Vanilla JavaScript, and Chart.js

---

## ⚡ Quick Start (Hassle-Free Setup)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional but Recommended)
Copy the example environment file:
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API key (obtainable for free at [Google AI Studio](https://aistudio.google.com/)):
```ini
GEMINI_API_KEY=AIzaSy...
```
*(Note: If no API key is provided, ChurnGuard AI automatically runs in its built-in Python heuristic mode, allowing you to test and present the full dashboard and tools anytime without setup blockers!)*

### 3. Launch the Application
Run this single command:
```bash
python3 main.py
```
Or with Uvicorn live reload:
```bash
uvicorn main:app --reload --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 📂 Project Architecture

```
churnguard/
├── main.py              # FastAPI endpoints, static file mounting, file upload handling
├── agent.py             # Agentic investigation loop using google-genai & tool calling
├── tools.py             # 5 core Python analytical tools
├── data_generator.py    # Synthetic banking customer CSV generator (750 records)
├── data/
│   └── customers.csv    # Active synthetic banking dataset
├── static/
│   ├── index.html       # Single-page FinTech dashboard
│   ├── style.css        # Clean charcoal & subtle emerald design system
│   └── app.js           # Live tool trace, charts, and API interactions
├── requirements.txt     # Python package requirements
├── .env.example         # Template for environment configuration
└── README.md            # Project documentation and guide
```

---

## 🔍 The 5 Approved Python Tools

Gemini is given strict access to approved Python functions:

1. **`inspect_data()`**: Returns row counts, column types, missing values, duplicates, and statistical distribution summaries.
2. **`check_data_quality()`**: Audits data integrity, detects anomalies, warns about missing values, and identifies which analyses are supported vs unsupported.
3. **`analyze_customers()`**: Compares 30-day transaction velocity, quantifies drop-off rates, measures inactivity recency, and measures friction (failed transactions & complaints).
4. **`calculate_churn_risk()`**: Computes a transparent, explainable 0–100 heuristic risk score (Low < 35, Medium 35–59, High ≥ 60) based solely on customer behavioral factors without historical label leakage.
5. **`summarize_churn_patterns()`**: Evaluates historical churn rates and cross-tabulations when the `churned` ground-truth label exists.

---

## 🧪 Key Agentic Behaviors Demonstrated

- **Handling Missing Columns**: When asked about a metric that does not exist in the dataset (such as balance trends on a file without balance columns), Gemini reports the data gap rather than hallucinating findings.
- **Handling Data Quality Issues**: Detects missing values and suspicious data before calculating metrics.
- **Data Without Churn Labels**: Distinguishes between heuristic risk scoring and historical predictive models.
- **Live Tool Trace**: The UI displays the exact sequence of tools invoked, their input parameters, and output summaries.

---

## ⚠️ Academic Disclaimer

All data generated and analyzed by ChurnGuard AI is synthetic. It is designed purely for educational and investigative modeling demonstrations and does not contain real banking credentials or confidential customer records.
