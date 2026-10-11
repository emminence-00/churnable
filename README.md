# Churnable — Agentic Customer Churn Intelligence

> An autonomous AI analyst that investigates customer behavioral telemetry to predict churn risk, identify root causes, and prescribe retention actions. Built with LangGraph, Google Gemini, FastAPI, and a real-time dashboard.

---

## What It Does

Churnable is not a dashboard — it's an analyst. Give it a question like *"Who's churning and why?"* and it autonomously:

1. **Inspects** your customer dataset (schema, quality, gaps)
2. **Analyzes** transaction velocity, dormancy, failed payments, support friction, balance trends
3. **Scores** each account 0–100 with causal reasoning (not black-box ML)
4. **Visualizes** risk cohorts and factor breakdowns as interactive charts
5. **Recommends** specific retention plays per risk tier

All reasoning is traceable — every tool call, parameter, and output is logged in the chat.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key (optional — runs in heuristic mode without it)
cp .env.example .env
# Edit .env with your key from https://aistudio.google.com

# 3. Run
python3 main.py
# Or with hot reload: uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`

---

## Architecture

```
churnable/
├── main.py              # FastAPI app, endpoints, static serving
├── agent.py             # LangGraph agent with Gemini, tool orchestration
├── tools.py             # 5 analytical tools (Pandas/NumPy)
├── data_generator.py    # Synthetic data generator (750 customers)
├── data/
│   └── customers.csv    # Active dataset
├── static/
│   ├── index.html       # Single-page chat UI
│   ├── style.css        # Clean FinTech design system
│   └── app.js           # Chat engine, Chart.js rendering, modals
├── requirements.txt
├── .env.example         # Config template
└── README.md
```

---

## The Analytical Tools

| Tool | Purpose |
|------|---------|
| `inspect_data()` | Dataset dimensions, columns, dtypes, missing values, stats |
| `check_data_quality()` | Integrity audit, anomaly detection, analysis capability report |
| `analyze_customers()` | Transaction velocity, drop-off rates, inactivity, friction signals |
| `evaluate_cohort_churn()` | Agentic 0–100 risk scoring with causal drivers per customer |
| `summarize_churn_patterns()` | Historical churn correlations (when `churned` label exists) |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the dashboard |
| `/api/chat` | POST | Conversational agent (multi-turn, history-aware) |
| `/api/investigate` | POST | One-shot investigation task |
| `/api/predict-churn` | POST | Direct cohort risk evaluation |
| `/api/overview` | GET | Dataset metadata, quality, risk summary |
| `/api/data` | GET | Paginated, searchable customer records |
| `/api/upload-csv` | POST | Replace active dataset |
| `/api/regenerate-data` | POST | Generate fresh synthetic data |

---

## Chat Examples

```
"Show me the high-risk cohort"
"Why is CUST-1563 churning?"
"Analyze churn patterns across all customers"
"Compare dormant vs active accounts"
"What retention plays for accounts with failed payments?"
"Build a pie chart of risk distribution"
```

The agent responds with markdown reports, interactive tables, and live Chart.js graphs (pie, bar, line).

---

## Design Principles

- **No black boxes** — Every prediction includes primary drivers and recommended action
- **Conversational first** — Chat interface, not static reports
- **Traceable reasoning** — Expandable tool trace shows exact investigative steps
- **Works without API keys** — Heuristic fallback for demos/offline use
- **Extensible** — Add tools in `tools.py`, register in `agent.py`

---

## Requirements

- Python 3.10+
- Google Gemini API key (free tier works)
- 750-row synthetic dataset included; upload your own CSV anytime

---

## License

MIT