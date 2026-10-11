"""
Agentic Investigation & Predictive Churn Engine for Churnable using LangGraph & Google Gemini.
Implements a stateful, cyclic investigation graph with Google Gemini (ChatGoogleGenerativeAI),
agentic churn prediction nodes, customer cohort telemetry extraction, dynamic tool execution,
and structured evidence synthesis.
"""

import os
import json
import warnings
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Suppress minor pydantic compatibility notices on python 3.14
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure SSL CA bundle is loaded for macOS Python environments
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import tools

SYSTEM_INSTRUCTION = """You are **Churnable**, a witty, insightful AI churn analyst with a sharp mind and a bit of attitude. You don't just report numbers — you tell the story behind the data.

## Personality:
- **Sharp & conversational**: You speak like a seasoned analyst who's seen it all. Direct, occasionally dry humor, never robotic.
- **Opinionated but evidence-backed**: You have takes on what drives churn, but you back them with data.
- **Financially fluent**: You understand unit economics, LTV, CAC, and why a 5% churn rate kills a business.
- **Action-oriented**: Every insight comes with a "so what?" — what should the team actually *do*?

## CRITICAL BEHAVIOR RULES:
1. **For casual greetings and general questions** (e.g., "hi", "hello", "what can you do?", "how does this work?") — respond conversationally and naturally WITHOUT calling any tools. Introduce yourself with personality.
2. **For analytical or data questions** (e.g., "analyze customers", "show churn risk", "who is at risk?", "show me graphs", "predict churn") — use your tools autonomously to investigate and answer with data.
3. **Never run an unsolicited full investigation** just because someone greeted you.

## Your Analytical Capabilities (use tools only when relevant):
- inspect_data(): Dataset dimensions, columns, and statistics.
- check_data_quality(): Data health and missing values.
- analyze_customers(): Transaction velocity, inactivity, friction indicators.
- query_customer_cohort(cohort_type, limit): Customer behavioral profiles for cohorts.
- inspect_customer_profile(customer_id): Full telemetry for a specific customer.
- evaluate_cohort_churn(cohort_type, limit): Predictive churn risk scores for a customer cohort.
- record_churn_determinations(predictions): Record your churn predictions.
- summarize_churn_patterns(): Historical churn correlation patterns.

## CHART & VISUALIZATION RULES — READ CAREFULLY:

When your analysis contains comparative data, distributions, or metrics that benefit from visualization, you MUST embed a chart using this EXACT format:

```json
{
  "type": "bar",
  "title": "Chart Title Here",
  "data": {
    "labels": ["Label A", "Label B", "Label C"],
    "datasets": [{ "label": "Series Name", "data": [42, 28, 15] }]
  }
}
```

Supported types: `"bar"`, `"pie"`, `"doughnut"`, `"line"`.

For horizontal bar charts add: `"options": { "indexAxis": "y" }`

**Example — Risk distribution as pie:**
```json
{
  "type": "pie",
  "title": "Churn Risk Cohort Distribution",
  "data": {
    "labels": ["High Risk", "Medium Risk", "Low Risk"],
    "datasets": [{ "data": [8, 4, 3] }]
  }
}
```

**Example — Churn factors as horizontal bar:**
```json
{
  "type": "bar",
  "title": "Primary Churn Signal Prevalence (%)",
  "options": { "indexAxis": "y" },
  "data": {
    "labels": ["Velocity Drop", "Inactivity >30d", "Failed Transfers", "Complaints"],
    "datasets": [{ "label": "% Affected", "data": [67, 53, 40, 27] }]
  }
}
```

**Example — Metric comparison as bar:**
```json
{
  "type": "bar",
  "title": "Churned vs Retained: Key Metrics",
  "data": {
    "labels": ["Avg Tx Last 30d", "Avg Logins", "Avg Inactive Days"],
    "datasets": [
      { "label": "Churned", "data": [2.3, 1.4, 43] },
      { "label": "Retained", "data": [16.7, 17.8, 7] }
    ]
  }
}
```

## STRICT FORMATTING RULES:
- **NEVER use markdown tables** (pipes `|`, dashes `---`). They do not render — they show as raw text.
- Use bullet points (`-`) for lists of metrics or comparisons instead of tables.
- Embed ```json chart blocks for ANY visual comparison that would normally go in a table.
- Use markdown headers (##, ###) and bold (**text**) for structure.

## When producing analytical reports, include:
- Risk scores and tiers (High/Medium/Low)
- Key churn drivers per customer
- Retention recommendations
- Concise, executive-level formatting with markdown headers and bullet points.
- **Always embed chart blocks for data distributions, metric comparisons, and factor breakdowns.**

## Tone: Sharp, conversational, financially literate. No emojis. Think "senior analyst who gives a damn."
"""


def extract_text_from_message(content: Any) -> str:
    """Extracts raw string text safely from str, list, or dict message content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _generate_tool_summary(tool_name: str, result: Dict[str, Any]) -> str:
    """Produces a concise, user-friendly summary of tool execution for the UI activity feed."""
    if "error" in result:
        return f"Error: {result['error']}"

    if tool_name == "inspect_data":
        rows = result.get("total_rows", 0)
        cols = result.get("total_columns", 0)
        missing = len(result.get("missing_values_summary", {}))
        return f"Inspected dataset: {rows} customers, {cols} columns. {missing} column(s) with missing values."

    elif tool_name == "check_data_quality":
        score = result.get("data_quality_score", 0)
        warnings_cnt = len(result.get("warnings", []))
        has_churn = result.get("has_historical_churn_label", False)
        return (
            f"Quality Score: {score}/100. Warnings: {warnings_cnt}. "
            f"Historical churn label: {'Present' if has_churn else 'Missing'}."
        )

    elif tool_name == "analyze_customers":
        trends = result.get("transaction_trends", {})
        sharp_drop = trends.get("pct_customers_with_sharp_decline", 0)
        dormant = result.get("inactivity_breakdown", {}).get("severely_inactive_over_60_days", {}).get("percent", 0)
        return f"Analyzed activity: {sharp_drop}% showed sharp transaction drops (>40%); {dormant}% dormant >60 days."

    elif tool_name == "query_customer_cohort":
        cnt = result.get("extracted_count", 0)
        cohort = result.get("cohort_type", "candidates")
        return f"Queried {cnt} customer profiles ({cohort}) for behavioral inspection."

    elif tool_name == "inspect_customer_profile":
        cid = result.get("customer_profile", {}).get("customer_id", "")
        return f"Retrieved detailed record for customer {cid}."

    elif tool_name in ["record_churn_determinations", "evaluate_cohort_churn"]:
        cnt = result.get("recorded_count", result.get("evaluated_count", len(result.get("predictions", []))))
        return f"Recorded churn risk determinations for {cnt} customer accounts."

    elif tool_name == "summarize_churn_patterns":
        if result.get("status") == "unavailable":
            return "Historical churn patterns unavailable (no 'churned' label)."
        rate = result.get("historical_churn_rate_pct", 0)
        return f"Evaluated historical churn: Overall churn rate is {rate}%."

    return f"Completed {tool_name} successfully."


def predict_churn_with_gemini(
    llm,
    customer_profiles: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Directly invokes Google Gemini to evaluate customer behavioral telemetry and predict churn.
    Returns customer records with AI-predicted risk score, category, causal reasoning, and retention action.
    """
    if not customer_profiles:
        return []

    profile_lookup = {str(p["customer_id"]): p for p in customer_profiles}

    # Compact telemetry representation for prompt efficiency
    compact_profiles = []
    for p in customer_profiles:
        compact_profiles.append({
            "customer_id": str(p.get("customer_id")),
            "tenure_months": p.get("tenure_months"),
            "tx_last_30d": p.get("transactions_last_30_days"),
            "tx_prev_30d": p.get("transactions_previous_30_days"),
            "tx_change_pct": p.get("transaction_change_pct"),
            "days_inactive": p.get("days_since_last_transaction"),
            "logins_last_30d": p.get("logins_last_30_days"),
            "failed_transactions": p.get("failed_transactions"),
            "support_complaints": p.get("support_complaints"),
            "balance_change_pct": p.get("balance_change_percentage"),
        })

    prompt = f"""You are Churnable, a Senior Predictive Churn Intelligence Agent.
Evaluate the following customer records and DETERMINE / PREDICT their churn risk using agentic reasoning.
For EACH customer, evaluate their behavioral factors:
- Transaction velocity decline and momentum
- Days of inactivity / dormancy
- Friction signals (failed transactions, support complaints)
- Digital logins and balance shifts

Return a valid JSON array of objects. Each object MUST have:
- "customer_id": string (matching input ID)
- "risk_score": integer between 0 and 100 (probability/risk of churn determined by you)
- "risk_category": "High" (score >= 60) | "Medium" (score 35-59) | "Low" (score < 35)
- "churn_prediction": "Likely to Churn" | "At Risk" | "Retained"
- "primary_drivers": concise synthesized causal reasoning explaining WHY they are predicted to churn (max 12 words)
- "recommended_action": personalized retention intervention (max 10 words)

Customer Records:
{json.dumps(compact_profiles, indent=2)}

Output ONLY the raw JSON array starting with [ and ending with ]. Do not include markdown codeblocks or explanation."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert AI predictive churn analyst. You output only valid JSON."),
            HumanMessage(content=prompt)
        ])

        raw_text = response.content
        if isinstance(raw_text, list):
            raw_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_text])

        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        start_idx = clean_text.find("[")
        end_idx = clean_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx:end_idx + 1]

        predictions = json.loads(clean_text)

        results = []
        for pred in predictions:
            # Guard: skip non-dict items (e.g. model returned strings)
            if not isinstance(pred, dict):
                continue
            cid = str(pred.get("customer_id", ""))
            prof = profile_lookup.get(cid, {})
            score = int(pred.get("risk_score", 50))
            score = max(0, min(100, score))
            cat = pred.get("risk_category")
            if not cat:
                cat = "High" if score >= 60 else ("Medium" if score >= 35 else "Low")

            results.append({
                "customer_id": cid,
                "risk_score": score,
                "risk_category": cat,
                "churn_prediction": pred.get("churn_prediction", "At Risk"),
                "primary_drivers": pred.get("primary_drivers", "Behavioral risk indicators identified"),
                "recommended_action": pred.get("recommended_action", "Proactive retention engagement"),
                "days_since_last_transaction": prof.get("days_since_last_transaction"),
                "transactions_last_30_days": prof.get("transactions_last_30_days"),
                "transactions_previous_30_days": prof.get("transactions_previous_30_days"),
                "failed_transactions": prof.get("failed_transactions"),
                "support_complaints": prof.get("support_complaints"),
                "balance_change_percentage": prof.get("balance_change_percentage"),
                "prediction_source": "Gemini AI Agent"
            })

        # If all items were non-dict, fall through to fallback
        if not results:
            raise ValueError("Model returned no valid prediction objects (received non-dict items)")

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results


    except Exception as e:
        print(f"Error in predict_churn_with_gemini: {str(e)}")
        # Graceful fallback: produce behavioral-informed predictions
        results = []
        for p in customer_profiles:
            cid = str(p.get("customer_id", ""))
            days = p.get("days_since_last_transaction", 0) or 0
            fails = p.get("failed_transactions", 0) or 0
            comps = p.get("support_complaints", 0) or 0
            tx_last = p.get("transactions_last_30_days", 0) or 0
            tx_prev = p.get("transactions_previous_30_days", 0) or 0
            
            score = 20
            drivers = []
            if days >= 30:
                score += 30
                drivers.append(f"{days}d inactive")
            if tx_prev > 0 and tx_last <= tx_prev * 0.5:
                score += 25
                drivers.append("sharp tx drop")
            if fails >= 2 or comps >= 1:
                score += 25
                drivers.append("friction & complaints")
            score = min(98, score)

            cat = "High" if score >= 60 else ("Medium" if score >= 35 else "Low")
            results.append({
                "customer_id": cid,
                "risk_score": score,
                "risk_category": cat,
                "churn_prediction": "Likely to Churn" if cat == "High" else ("At Risk" if cat == "Medium" else "Retained"),
                "primary_drivers": ", ".join(drivers) if drivers else "Stable activity",
                "recommended_action": "Reach out with tailored incentive",
                "days_since_last_transaction": days,
                "transactions_last_30_days": tx_last,
                "transactions_previous_30_days": tx_prev,
                "failed_transactions": fails,
                "support_complaints": comps,
                "balance_change_percentage": p.get("balance_change_percentage"),
                "prediction_source": "Heuristic Rule (Fallback)"
            })
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results


# LangGraph State Definition
class InvestigationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    dataset_path: str
    activity_log: list
    round_count: int
    final_report: str
    quality_data: dict
    risk_data: dict
    agent_predictions: list


def build_langgraph_investigator(llm, dataset_path: str = "data/customers.csv"):
    """
    Constructs the LangGraph state graph with Gemini and analytical tools:
    START -> agent_node -> (conditional) -> tools_node -> agent_node ... -> END
    """
    @tool
    def inspect_data() -> dict:
        """Inspects dataset structure, dimensions, columns, dtypes, missing values, duplicates, and stats."""
        return tools.inspect_data(dataset_path)

    @tool
    def check_data_quality() -> dict:
        """Checks data integrity, detects anomalies, missing or suspicious values, and reports analysis capability."""
        return tools.check_data_quality(dataset_path)

    @tool
    def analyze_customers(filter_segment: str = "all", metric_focus: str = "all") -> dict:
        """Analyzes customer activity, transaction velocities, dormancy lengths, logins, and friction."""
        return tools.analyze_customers(dataset_path=dataset_path, filter_segment=filter_segment, metric_focus=metric_focus)

    @tool
    def query_customer_cohort(cohort_type: str = "candidates", limit: int = 15) -> dict:
        """Queries customer behavioral profiles across cohorts (candidates, high_friction, dormant, sharp_drop, balance_stressed, loyal_benchmark, all). Returns actual customer telemetry."""
        return tools.extract_customer_cohorts(dataset_path=dataset_path, cohort_type=cohort_type, limit=limit)

    @tool
    def inspect_customer_profile(customer_id: str) -> dict:
        """Retrieves full behavioral telemetry for a specific customer ID."""
        return tools.get_customer_profile(dataset_path=dataset_path, customer_id=customer_id)

    @tool
    def record_churn_determinations(predictions: list) -> dict:
        """Records agentic churn risk determinations made by the agent. Each prediction object must contain: customer_id, risk_score (0-100), risk_category (High/Medium/Low), churn_prediction (Likely to Churn/At Risk/Retained), primary_drivers, and recommended_action."""
        return {
            "status": "success",
            "recorded_count": len(predictions),
            "predictions": predictions,
            "summary": f"Recorded churn risk determinations for {len(predictions)} customer accounts."
        }

    @tool
    def evaluate_cohort_churn(cohort_type: str = "candidates", limit: int = 15) -> dict:
        """Evaluates customer behavioral telemetry for a cohort (candidates, high_friction, dormant, sharp_drop) and produces predictive churn risk scores, categories, causal drivers, and retention actions."""
        cohort_res = tools.extract_customer_cohorts(dataset_path=dataset_path, cohort_type=cohort_type, limit=limit)
        profiles = cohort_res.get("profiles", [])
        preds = predict_churn_with_gemini(llm, profiles)
        return {
            "status": "success",
            "cohort_type": cohort_type,
            "evaluated_count": len(preds),
            "predictions": preds,
            "summary": f"Evaluated {len(preds)} customer records ({cohort_type}) and determined predictive churn risk scores and drivers."
        }

    @tool
    def summarize_churn_patterns() -> dict:
        """Examines historical churn rates across customer groups and identifies associations when 'churned' label is present."""
        return tools.summarize_churn_patterns(dataset_path=dataset_path)

    tools_list = [
        inspect_data,
        check_data_quality,
        analyze_customers,
        query_customer_cohort,
        inspect_customer_profile,
        record_churn_determinations,
        evaluate_cohort_churn,
        summarize_churn_patterns
    ]
    tool_map = {t.name: t for t in tools_list}
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: InvestigationState) -> Dict[str, Any]:
        """Agent node: Prompts Gemini with current messages to decide tool calls or final answer."""
        current_rounds = state.get("round_count", 0)
        messages = list(state["messages"])

        if current_rounds >= 6:
            messages.append(
                HumanMessage(
                    content="You have completed all tool rounds. Provide your final evidence-based report now, synthesizing your agentic churn predictions, data limitations, and retention playbook."
                )
            )

        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "round_count": current_rounds + 1
        }

    def tools_node(state: InvestigationState) -> Dict[str, Any]:
        """Tools node: Safely executes requested tools, records trace, and creates ToolMessages."""
        last_message = state["messages"][-1]
        tool_messages = []
        new_activity_entries = []

        quality_data = dict(state.get("quality_data", {}))
        risk_data = dict(state.get("risk_data", {}))
        agent_predictions = list(state.get("agent_predictions", []))
        round_idx = state.get("round_count", 1)

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                call_id = tc.get("id", f"call_{len(tool_messages)}")

                if tool_name not in tool_map:
                    result = {"error": f"Tool '{tool_name}' is not in approved registry."}
                else:
                    target_tool = tool_map[tool_name]
                    try:
                        result = target_tool.invoke(tool_args)
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {str(e)}"}

                # Cache quality and risk data
                if tool_name in ["record_churn_determinations", "evaluate_cohort_churn"]:
                    preds = result.get("predictions", [])
                    if preds:
                        agent_predictions = preds
                        risk_data["top_high_risk_customers"] = preds
                elif tool_name == "check_data_quality":
                    quality_data = result

                summary = _generate_tool_summary(tool_name, result)
                new_activity_entries.append({
                    "step": len(state.get("activity_log", [])) + len(new_activity_entries) + 1,
                    "round": round_idx,
                    "tool": tool_name,
                    "args": tool_args,
                    "summary": summary,
                    "data": result
                })

                tool_messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        name=tool_name,
                        content=json.dumps(result, default=str)
                    )
                )

        return {
            "messages": tool_messages,
            "activity_log": state.get("activity_log", []) + new_activity_entries,
            "quality_data": quality_data,
            "risk_data": risk_data,
            "agent_predictions": agent_predictions
        }

    def should_continue(state: InvestigationState) -> str:
        """Determines whether to continue calling tools or finish investigation."""
        last_message = state["messages"][-1]
        round_count = state.get("round_count", 0)

        if hasattr(last_message, "tool_calls") and last_message.tool_calls and round_count <= 6:
            return "tools_node"
        return END

    graph_builder = StateGraph(InvestigationState)
    graph_builder.add_node("agent_node", agent_node)
    graph_builder.add_node("tools_node", tools_node)

    graph_builder.add_edge(START, "agent_node")
    graph_builder.add_conditional_edges("agent_node", should_continue, {
        "tools_node": "tools_node",
        END: END
    })
    graph_builder.add_edge("tools_node", "agent_node")

    return graph_builder.compile()


def run_heuristic_investigation(task: str, dataset_path: Optional[str] = None, reason: str = "") -> Dict[str, Any]:
    """
    Fallback investigator: Executes analytical tools directly when GEMINI_API_KEY is not configured
    or Gemini service is unreachable.
    """
    d_path = dataset_path or "data/customers.csv"
    activity_log = []
    executed_tools = {}

    t1_res = tools.inspect_data(d_path)
    executed_tools["inspect_data"] = t1_res
    activity_log.append({
        "step": 1,
        "tool": "inspect_data",
        "args": {},
        "summary": _generate_tool_summary("inspect_data", t1_res),
        "data": t1_res
    })

    t2_res = tools.check_data_quality(d_path)
    executed_tools["check_data_quality"] = t2_res
    activity_log.append({
        "step": 2,
        "tool": "check_data_quality",
        "args": {},
        "summary": _generate_tool_summary("check_data_quality", t2_res),
        "data": t2_res
    })

    t3_res = tools.analyze_customers(dataset_path=d_path)
    executed_tools["analyze_customers"] = t3_res
    activity_log.append({
        "step": 3,
        "tool": "analyze_customers",
        "args": {},
        "summary": _generate_tool_summary("analyze_customers", t3_res),
        "data": t3_res
    })

    t4_res = tools.calculate_churn_risk(dataset_path=d_path, limit=15)
    executed_tools["calculate_churn_risk"] = t4_res
    activity_log.append({
        "step": 4,
        "tool": "calculate_churn_risk",
        "args": {"limit": 15},
        "summary": _generate_tool_summary("calculate_churn_risk", t4_res),
        "data": t4_res
    })

    if t2_res.get("has_historical_churn_label", False):
        t5_res = tools.summarize_churn_patterns(dataset_path=d_path)
        executed_tools["summarize_churn_patterns"] = t5_res
        activity_log.append({
            "step": 5,
            "tool": "summarize_churn_patterns",
            "args": {},
            "summary": _generate_tool_summary("summarize_churn_patterns", t5_res),
            "data": t5_res
        })

    rows = t1_res.get("total_rows", 0)
    cols = t1_res.get("total_columns", 0)
    q_score = t2_res.get("data_quality_score", 0)
    risk_dist = t4_res.get("risk_distribution", {})

    final_report = f"""## Investigation Objective & Scope
The agent evaluated the objective: **"{task}"** across {rows} digital banking customers and {cols} behavioral attributes.
*(Notice: {reason or "Running in fallback mode. Ensure GEMINI_API_KEY is configured for active Gemini 3.8 Flash agentic intelligence"}.)*

## Data Quality & Identified Limitations
- **Dataset Completeness**: Overall quality index rated at **{q_score}/100**.
- **Methodological Limitation**: Analysis evaluated synthetic FinTech customer records. Baseline risk rankings reflect behavioral indicators.

## Key Empirical Findings
- **Transaction Drop**: {t3_res.get('transaction_trends', {}).get('pct_customers_with_sharp_decline', 0)}% of customers exhibited a sharp transaction decline (>40% drop).
- **Inactivity & Dormancy**: {t3_res.get('inactivity_breakdown', {}).get('severely_inactive_over_60_days', {}).get('percent', 0)}% of accounts have been inactive for over 60 days.
- **Friction Points**: {t3_res.get('friction_indicators', {}).get('pct_customers_with_complaints', 0)}% of users logged customer service complaints.

## Actionable Retention Recommendations
1. **Targeted Inactivity Re-engagement**: Deploy automated push notifications with transaction fee discounts for accounts reaching 20-30 days of inactivity.
2. **Payment Failure Recovery**: Trigger real-time customer care alerts when digital transfers fail >=2 times.
3. **Complaint Priority Escalation**: Route accounts with filed complaints to senior retention representatives within 4 hours.
"""

    return {
        "status": "success",
        "agent_mode": "heuristic_fallback",
        "task": task,
        "activity_log": activity_log,
        "final_report": final_report,
        "quality_score": q_score,
        "high_risk_customers": t4_res.get("top_high_risk_customers", []),
        "risk_distribution": risk_dist,
        "average_risk_score": t4_res.get("average_risk_score", 0)
    }


def run_gemini_investigation(task: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the Agentic Churn Investigation & Prediction loop using LangGraph and Google Gemini.
    The agent inspects data, analyzes signals, evaluates customer profiles, and DETERMINES churn
    using cognitive generative reasoning.
    """
    d_path = dataset_path or "data/customers.csv"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return run_heuristic_investigation(
            task, d_path,
            reason="GEMINI_API_KEY is not set in environment or .env. Running in heuristic mode"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    preferred_models = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
    ]
    seen = set()
    models_to_try = [m for m in preferred_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2
            )

            # Build initial dataset profile
            initial_inspection = tools.inspect_data(d_path)
            col_list = initial_inspection.get("columns", [])
            total_rows = initial_inspection.get("total_rows", 0)

            user_content = f"""Investigation Request:
"{task}"

Current Dataset Profile:
- Total Customers: {total_rows}
- Columns Available: {', '.join(col_list)}

Available Investigative Tools:
- inspect_data(): Dataset dimensions, columns, dtypes, missing values, duplicates, and statistics.
- check_data_quality(): Evaluates data health, anomalies, missing fields, and supported analyses.
- analyze_customers(): Compares transaction velocity, inactivity lengths, logins, and friction indicators.
- query_customer_cohort(cohort_type, limit): Queries customer records (candidates, high_friction, dormant, sharp_drop, balance_stressed, loyal_benchmark, all).
- inspect_customer_profile(customer_id): Retrieves detailed behavioral telemetry for a specific customer ID.
- record_churn_determinations(predictions): Records your churn predictions with scores (0-100), tiers, causal drivers, and retention actions.
- evaluate_cohort_churn(cohort_type, limit): Evaluates customer records for a cohort and determines churn risk scores.
- summarize_churn_patterns(): Analyzes historical churn correlations if 'churned' label is present.

Instructions:
1. Formulate your investigative approach based on the objective.
2. Query data and inspect customer behavioral records as you deem necessary.
3. Determine customer churn risk and record evaluations with record_churn_determinations().
4. Deliver an evidence-backed investigation report explaining your findings and recommendations."""

            # Compile LangGraph with tools
            graph = build_langgraph_investigator(llm, dataset_path=d_path)

            initial_state: InvestigationState = {
                "messages": [
                    SystemMessage(content=SYSTEM_INSTRUCTION),
                    HumanMessage(content=user_content)
                ],
                "task": task,
                "dataset_path": d_path,
                "activity_log": [],
                "round_count": 0,
                "final_report": "",
                "quality_data": {},
                "risk_data": {},
                "agent_predictions": []
            }

            # Invoke the LangGraph execution graph
            final_state = graph.invoke(initial_state)

            # Extract final report from the last AI message
            final_message = final_state["messages"][-1]
            final_report_text = final_message.content if isinstance(final_message, AIMessage) else ""
            if isinstance(final_report_text, list):
                final_report_text = "\n".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in final_report_text])

            cached_quality_data = final_state.get("quality_data", {})
            if not cached_quality_data:
                cached_quality_data = tools.check_data_quality(d_path)

            agent_predictions = final_state.get("agent_predictions", [])

            # If agent did not explicitly call predict_cohort_churn during graph execution,
            # perform Gemini Agentic Churn Determination now on the top candidate cohort!
            if not agent_predictions:
                candidate_cohort = tools.extract_customer_cohorts(dataset_path=d_path, cohort_type="candidates", limit=15)
                agent_predictions = predict_churn_with_gemini(llm, candidate_cohort.get("profiles", []))

            # Calculate risk distribution and average score directly from Gemini Agent determinations
            if agent_predictions:
                scores = [c["risk_score"] for c in agent_predictions]
                avg_score = round(float(np.mean(scores)), 1)
                high_count = sum(1 for c in agent_predictions if c.get("risk_category") == "High")
                med_count = sum(1 for c in agent_predictions if c.get("risk_category") == "Medium")
                low_count = sum(1 for c in agent_predictions if c.get("risk_category") == "Low")
                tot = len(agent_predictions)
                risk_dist = {
                    "high_risk": {"count": high_count, "percent": round(high_count / tot * 100, 1)},
                    "medium_risk": {"count": med_count, "percent": round(med_count / tot * 100, 1)},
                    "low_risk": {"count": low_count, "percent": round(low_count / tot * 100, 1)}
                }
            else:
                base_risk = tools.calculate_churn_risk(dataset_path=d_path, limit=15)
                agent_predictions = base_risk.get("top_high_risk_customers", [])
                risk_dist = base_risk.get("risk_distribution", {})
                avg_score = base_risk.get("average_risk_score", 0)

            # Build charts for investigation response
            high_cnt = risk_dist.get("high_risk", {}).get("count", 0)
            med_cnt = risk_dist.get("medium_risk", {}).get("count", 0)
            low_cnt = risk_dist.get("low_risk", {}).get("count", 0)
            charts = [
                {
                    "id": "risk_pie",
                    "type": "pie",
                    "title": "Risk Cohort Distribution",
                    "data": {
                        "labels": ["High Risk", "Medium Risk", "Low Risk"],
                        "datasets": [{
                            "data": [high_cnt, med_cnt, low_cnt],
                            "backgroundColor": ["#EF4444", "#F59E0B", "#10B981"]
                        }]
                    }
                }
            ] if (high_cnt + med_cnt + low_cnt) > 0 else []

            return {
                "status": "success",
                "agent_mode": "agentic_gemini_predictive",
                "model_used": model_name,
                "prediction_engine": f"Agentic AI ({model_name})",
                "task": task,
                "activity_log": final_state.get("activity_log", []),
                "final_report": final_report_text or "Agentic investigation and churn determination completed with findings above.",
                "quality_score": cached_quality_data.get("data_quality_score", 90),
                "high_risk_customers": agent_predictions,
                "risk_distribution": risk_dist,
                "charts": charts,
                "average_risk_score": avg_score
            }

        except Exception as e:
            last_error = e
            print(f"Model {model_name} encountered error: {str(e)}. Trying next model...")
            continue

    print(f"All Gemini models exhausted. Error: {str(last_error)}. Running heuristic fallback.")
    return run_heuristic_investigation(
        task, d_path,
        reason=f"Gemini API error ({str(last_error)}). Ran analytical tools directly"
    )


def run_direct_agent_churn_prediction(
    cohort_type: str = "candidates",
    limit: int = 15,
    dataset_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Directly triggers the Gemini AI Agent to evaluate customer profiles and determine churn risk.
    """
    d_path = dataset_path or "data/customers.csv"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        base = tools.calculate_churn_risk(d_path, limit=limit)
        return {
            "status": "success",
            "prediction_engine": "Heuristic Rule Fallback (GEMINI_API_KEY not configured)",
            "cohort_type": cohort_type,
            "evaluated_count": len(base.get("top_high_risk_customers", [])),
            "predictions": base.get("top_high_risk_customers", []),
            "risk_distribution": base.get("risk_distribution", {}),
            "average_risk_score": base.get("average_risk_score", 0)
        }

    from langchain_google_genai import ChatGoogleGenerativeAI
    preferred_models = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
    ]
    seen = set()
    models_to_try = [m for m in preferred_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1
            )

            cohort_data = tools.extract_customer_cohorts(dataset_path=d_path, cohort_type=cohort_type, limit=limit)
            profiles = cohort_data.get("profiles", [])
            predictions = predict_churn_with_gemini(llm, profiles)

            scores = [c["risk_score"] for c in predictions] if predictions else [0]
            avg_score = round(float(np.mean(scores)), 1)
            high_cnt = sum(1 for c in predictions if c.get("risk_category") == "High")
            med_cnt = sum(1 for c in predictions if c.get("risk_category") == "Medium")
            low_cnt = sum(1 for c in predictions if c.get("risk_category") == "Low")
            tot = len(predictions) if predictions else 1

            risk_dist = {
                "high_risk": {"count": high_cnt, "percent": round(high_cnt / tot * 100, 1)},
                "medium_risk": {"count": med_cnt, "percent": round(med_cnt / tot * 100, 1)},
                "low_risk": {"count": low_cnt, "percent": round(low_cnt / tot * 100, 1)}
            }

            charts = [
                {
                    "id": "risk_pie",
                    "type": "pie",
                    "title": "Risk Cohort Distribution",
                    "data": {
                        "labels": ["High Risk", "Medium Risk", "Low Risk"],
                        "datasets": [{
                            "data": [high_cnt, med_cnt, low_cnt],
                            "backgroundColor": ["#EF4444", "#F59E0B", "#10B981"]
                        }]
                    }
                }
            ] if (high_cnt + med_cnt + low_cnt) > 0 else []

            return {
                "status": "success",
                "prediction_engine": f"Agentic AI ({model_name})",
                "cohort_type": cohort_type,
                "evaluated_count": len(predictions),
                "predictions": predictions,
                "risk_distribution": risk_dist,
                "charts": charts,
                "average_risk_score": avg_score
            }

        except Exception as e:
            last_error = e
            print(f"Direct prediction with {model_name} failed: {str(e)}. Trying next...")
            continue

    base = tools.calculate_churn_risk(d_path, limit=limit)
    risk_dist = base.get("risk_distribution", {})
    high_cnt = risk_dist.get("high_risk", {}).get("count", 0)
    med_cnt = risk_dist.get("medium_risk", {}).get("count", 0)
    low_cnt = risk_dist.get("low_risk", {}).get("count", 0)
    charts = [
        {
            "id": "risk_pie",
            "type": "pie",
            "title": "Risk Cohort Distribution",
            "data": {
                "labels": ["High Risk", "Medium Risk", "Low Risk"],
                "datasets": [{
                    "data": [high_cnt, med_cnt, low_cnt],
                    "backgroundColor": ["#EF4444", "#F59E0B", "#10B981"]
                }]
            }
        }
    ] if (high_cnt + med_cnt + low_cnt) > 0 else []
    return {
        "status": "error",
        "message": str(last_error),
        "prediction_engine": "Heuristic Rule Fallback",
        "cohort_type": cohort_type,
        "evaluated_count": len(base.get("top_high_risk_customers", [])),
        "predictions": base.get("top_high_risk_customers", []),
        "risk_distribution": risk_dist,
        "charts": charts,
        "average_risk_score": base.get("average_risk_score", 0)
    }


def chat_with_agent(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    dataset_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Conversational Agentic Chatbot Engine powered by LangGraph and Google Gemini.
    Maintains multi-turn context, invokes investigative and prediction tools autonomously,
    and returns conversational responses alongside live risk predictions.
    """
    d_path = dataset_path or "data/customers.csv"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    history = history or []

    # Conversational shortcut: if the message is a casual greeting, skip all tooling
    casual_phrases = ["hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon", "good evening", "sup", "yo"]
    is_casual = message.strip().lower().rstrip("!?.") in casual_phrases or len(message.strip().split()) <= 2 and not any(w in message.lower() for w in ["data", "churn", "risk", "predict", "customer", "analyze", "show", "graph", "chart"])

    if not api_key:
        if is_casual:
            return {
                "status": "success",
                "reply": "Hey. I'm **Churnable** — your churn analyst who actually reads the data.\n\nI don't do generic dashboards. I investigate:\n- **Who's leaving** — risk scores with causal reasoning, not vibes\n- **Why they're leaving** — transaction drops, dormancy, failed payments, complaints\n- **What to do about it** — retention plays tailored to each risk tier\n- **Visual proof** — pie charts, bar charts, trend lines on demand\n\nTry: *\"Show me the high-risk cohort\"* or *\"Why is customer CUST-042 churning?\"*",
                "activity_log": [],
                "agent_predictions": [],
                "risk_distribution": {},
                "average_risk_score": 0,
                "model_used": "Churnable"
            }
        heuristic_res = run_heuristic_investigation(message, dataset_path=d_path, reason="API key not set")
        return {
            "status": "success",
            "reply": f"Here is my analysis of your query:\n\n{heuristic_res['final_report']}",
            "activity_log": heuristic_res.get("activity_log", []),
            "agent_predictions": heuristic_res.get("high_risk_customers", []),
            "risk_distribution": heuristic_res.get("risk_distribution", {}),
            "average_risk_score": heuristic_res.get("average_risk_score", 0),
            "model_used": "Heuristic Rule Engine"
        }

    from langchain_google_genai import ChatGoogleGenerativeAI

    preferred_models = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
    ]
    seen = set()
    models_to_try = [m for m in preferred_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2
            )

            # Build LangGraph graph with tools
            graph = build_langgraph_investigator(llm, dataset_path=d_path)

            # Profile dataset for current turn
            initial_inspection = tools.inspect_data(d_path)
            col_list = initial_inspection.get("columns", [])
            total_rows = initial_inspection.get("total_rows", 0)

            system_prompt = SYSTEM_INSTRUCTION + f"""

## Active Context:
- Dataset: {d_path} ({total_rows} customer accounts, {len(col_list)} behavioral columns: {', '.join(col_list)})
- You are in live chat mode with a financial analyst or retention team member.
- IMPORTANT: Only invoke tools when the user's message specifically asks for data analysis, predictions, customer insights, or churn investigation. For greetings or general questions, respond conversationally without tools.
"""

            langchain_messages: List[BaseMessage] = [
                SystemMessage(content=system_prompt)
            ]

            # Reconstruct past conversation turns
            for turn in history[-10:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))

            # Current user query
            langchain_messages.append(HumanMessage(content=message))

            initial_state: InvestigationState = {
                "messages": langchain_messages,
                "task": message,
                "dataset_path": d_path,
                "activity_log": [],
                "round_count": 0,
                "final_report": "",
                "quality_data": {},
                "risk_data": {},
                "agent_predictions": []
            }

            final_state = graph.invoke(initial_state)

            final_message = final_state["messages"][-1]
            reply_text = extract_text_from_message(final_message.content if hasattr(final_message, "content") else "")

            if not reply_text.strip():
                reply_text = "I have examined the data and updated the risk determinations."

            agent_preds = final_state.get("agent_predictions", [])
            if not agent_preds:
                # If the user specifically asked for churn evaluation or high risk accounts
                lowered = message.lower()
                if any(w in lowered for w in ["risk", "churn", "predict", "cohort", "account", "customer", "dormant", "friction"]):
                    candidate_cohort = tools.extract_customer_cohorts(dataset_path=d_path, cohort_type="candidates", limit=15)
                    agent_preds = predict_churn_with_gemini(llm, candidate_cohort.get("profiles", []))

            risk_dist = {}
            factors_breakdown = {}
            avg_score = 0.0

            if agent_preds and isinstance(agent_preds, list):
                # Ensure all items in agent_preds are dicts
                valid_preds = [c for c in agent_preds if isinstance(c, dict)]
                if valid_preds:
                    scores = [int(c.get("risk_score", 50)) for c in valid_preds]
                    avg_score = round(float(np.mean(scores)), 1)
                    high_cnt = sum(1 for c in valid_preds if c.get("risk_category") == "High")
                    med_cnt = sum(1 for c in valid_preds if c.get("risk_category") == "Medium")
                    low_cnt = sum(1 for c in valid_preds if c.get("risk_category") == "Low")
                    tot = len(valid_preds)
                    risk_dist = {
                        "high_risk": {"count": high_cnt, "percent": round(high_cnt / tot * 100, 1)},
                        "medium_risk": {"count": med_cnt, "percent": round(med_cnt / tot * 100, 1)},
                        "low_risk": {"count": low_cnt, "percent": round(low_cnt / tot * 100, 1)}
                    }

                    drop_cnt = sum(1 for c in valid_preds if (c.get("transactions_previous_30_days") or 0) > 0 and (c.get("transactions_last_30_days") or 0) <= (c.get("transactions_previous_30_days") or 0) * 0.6)
                    dorm_cnt = sum(1 for c in valid_preds if (c.get("days_since_last_transaction") or 0) >= 30)
                    fail_cnt = sum(1 for c in valid_preds if (c.get("failed_transactions") or 0) >= 1)
                    comp_cnt = sum(1 for c in valid_preds if (c.get("support_complaints") or 0) >= 1)
                    bal_cnt = sum(1 for c in valid_preds if (c.get("balance_change_percentage") or 0) < -10)
                    factors_breakdown = {
                        "Velocity Drop": {"count": drop_cnt, "percent": round(drop_cnt / tot * 100, 1)},
                        "Inactivity (>30d)": {"count": dorm_cnt, "percent": round(dorm_cnt / tot * 100, 1)},
                        "Failed Transfers": {"count": fail_cnt, "percent": round(fail_cnt / tot * 100, 1)},
                        "Support Complaints": {"count": comp_cnt, "percent": round(comp_cnt / tot * 100, 1)},
                        "Balance Decline": {"count": bal_cnt, "percent": round(bal_cnt / tot * 100, 1)}
                    }
                    agent_preds = valid_preds

                    # Build chart data for frontend
                    factor_labels = list(factors_breakdown.keys())
                    factor_data = [f["percent"] for f in factors_breakdown.values()]
                    charts = [
                        {
                            "id": "risk_pie",
                            "type": "pie",
                            "title": "Risk Cohort Distribution",
                            "data": {
                                "labels": ["High Risk", "Medium Risk", "Low Risk"],
                                "datasets": [{
                                    "data": [high_cnt, med_cnt, low_cnt],
                                    "backgroundColor": ["#EF4444", "#F59E0B", "#10B981"]
                                }]
                            }
                        },
                        {
                            "id": "factors_bar",
                            "type": "bar",
                            "title": "Primary Churn Factors",
                            "data": {
                                "labels": factor_labels,
                                "datasets": [{
                                    "label": "% Accounts Affected",
                                    "data": factor_data,
                                    "backgroundColor": "#6366F1",
                                    "borderRadius": 4
                                }]
                            },
                            "options": {
                                "indexAxis": "y",
                                "scales": { "x": { "max": 100, "ticks": { "callback": "function(v) { return v + '%'; }" } } }
                            }
                        }
                    ]

            return {
                "status": "success",
                "reply": reply_text,
                "activity_log": final_state.get("activity_log", []),
                "agent_predictions": agent_preds,
                "risk_distribution": risk_dist,
                "factors_breakdown": factors_breakdown,
                "charts": charts if 'charts' in locals() else [],
                "average_risk_score": avg_score,
                "model_used": f"Gemini Agent ({model_name})"
            }

        except Exception as e:
            last_error = e
            print(f"Chat turn with {model_name} failed: {str(e)}. Trying next...")
            continue

    # Fallback to analytical rules if all LLM models encounter issues
    if is_casual:
        return {
            "status": "success",
            "reply": "Hey. I'm **Churnable** — your churn analyst who actually reads the data.\n\nI don't do generic dashboards. I investigate:\n- **Who's leaving** — risk scores with causal reasoning, not vibes\n- **Why they're leaving** — transaction drops, dormancy, failed payments, complaints\n- **What to do about it** — retention plays tailored to each risk tier\n- **Visual proof** — pie charts, bar charts, trend lines on demand\n\nTry: *\"Show me the high-risk cohort\"* or *\"Why is customer CUST-042 churning?\"*",
            "activity_log": [],
            "agent_predictions": [],
            "risk_distribution": {},
            "charts": [],
            "average_risk_score": 0,
            "model_used": "Churnable"
        }
    heuristic_res = run_heuristic_investigation(message, dataset_path=d_path, reason=str(last_error))
    return {
        "status": "success",
        "reply": f"Analysis based on analytical behavioral rules:\n\n{heuristic_res['final_report']}",
        "activity_log": heuristic_res.get("activity_log", []),
        "agent_predictions": heuristic_res.get("high_risk_customers", []),
        "risk_distribution": heuristic_res.get("risk_distribution", {}),
        "charts": [],
        "average_risk_score": heuristic_res.get("average_risk_score", 0),
        "model_used": "Heuristic Rule Fallback",
        "warning": f"Gemini capacity notice: {str(last_error)}"
    }

