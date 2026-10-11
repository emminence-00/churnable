"""
Core Python Tools for ChurnGuard AI.
Implements the 5 required analytical tools for customer churn investigation.
All tools operate on pandas DataFrames and return structured, JSON-serializable dictionaries.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "data/customers.csv"


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads dataset from path with fallback to default path."""
    target_path = filepath or DEFAULT_DATA_PATH
    if not os.path.exists(target_path):
        from data_generator import ensure_data_directory
        ensure_data_directory(target_path)
    return pd.read_csv(target_path)


def inspect_data(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspects the dataset structure, dimensions, column names, data types,
    missing values, duplicate records, and basic descriptive statistics.
    """
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    total_rows, total_cols = df.shape
    columns = list(df.columns)
    
    # Missing values
    missing = df.isnull().sum()
    missing_dict = {
        col: {"count": int(count), "percent": round(float(count / total_rows * 100), 2)}
        for col, count in missing.items() if count > 0
    }

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["customer_id"].duplicated().sum()) if "customer_id" in df.columns else 0

    # Data types
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            numeric_summary[col] = {
                "mean": round(float(series.mean()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "median": round(float(series.median()), 2),
            }

    # Sample preview (first 3 rows with NaN sanitized to None)
    clean_preview = df.head(3).replace({np.nan: None})
    preview = clean_preview.to_dict(orient="records")

    return {
        "status": "success",
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": columns,
        "data_types": dtypes_dict,
        "missing_values_summary": missing_dict,
        "has_missing_values": len(missing_dict) > 0,
        "duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": duplicate_ids,
        "numeric_summary": numeric_summary,
        "sample_preview": preview,
        "summary_text": (
            f"Dataset contains {total_rows} customers and {total_cols} attributes. "
            f"{len(missing_dict)} column(s) contain missing values. "
            f"Duplicate records: {duplicate_rows}."
        )
    }


def check_data_quality(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Checks data integrity, detects anomalies, missing or suspicious values,
    and reports which analyses are valid vs unsupported.
    """
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    total_rows = len(df)
    issues: List[str] = []
    warnings: List[str] = []

    # 1. Check Key Columns
    has_customer_id = "customer_id" in df.columns
    has_churn_label = "churned" in df.columns
    has_balance = "balance_change_percentage" in df.columns
    has_transactions = (
        "transactions_last_30_days" in df.columns and 
        "transactions_previous_30_days" in df.columns
    )
    has_inactivity = "days_since_last_transaction" in df.columns
    has_friction = "failed_transactions" in df.columns and "support_complaints" in df.columns

    # 2. Check Missing Values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct = round((count / total_rows) * 100, 1)
            warnings.append(f"Column '{col}' has {count} missing values ({pct}%).")

    # 3. Check Anomalies & Negative values in non-negative counts
    non_negative_fields = [
        "transactions_last_30_days", "transactions_previous_30_days",
        "days_since_last_transaction", "logins_last_30_days",
        "failed_transactions", "support_complaints", "app_usage_minutes",
        "customer_tenure_months"
    ]
    for field in non_negative_fields:
        if field in df.columns:
            negative_count = int((df[field] < 0).sum())
            if negative_count > 0:
                issues.append(f"Anomalous negative values found in '{field}' ({negative_count} records).")

    # 4. Check duplicate IDs
    if has_customer_id:
        dup_ids = int(df["customer_id"].duplicated().sum())
        if dup_ids > 0:
            issues.append(f"Duplicate customer IDs detected ({dup_ids} duplicates).")

    # 5. Determine Supported vs Unsupported Analyses
    supported = []
    unsupported = []

    if has_transactions and has_inactivity:
        supported.append("Transaction Velocity & Recency Analysis")
        supported.append("Heuristic Churn Risk Scoring")
    else:
        unsupported.append("Transaction Velocity Analysis (missing transaction or inactivity fields)")

    if has_churn_label:
        supported.append("Historical Churn Pattern & Correlation Analysis")
    else:
        unsupported.append("Historical Churn Model Evaluation (churned label column missing)")

    if has_balance:
        missing_bal = int(df["balance_change_percentage"].isnull().sum())
        if missing_bal > 0:
            supported.append(f"Balance Trend Analysis (note: {missing_bal} rows missing balance data)")
        else:
            supported.append("Balance Trend Analysis")
    else:
        unsupported.append("Balance Trend & Capital Flow Analysis (column 'balance_change_percentage' missing)")

    if has_friction:
        supported.append("Friction & Support Experience Analysis")
    else:
        unsupported.append("Friction Analysis (failed transactions or complaints columns missing)")

    # Quality Score Calculation (0-100)
    score = 100
    score -= len(issues) * 15
    score -= min(30, len(warnings) * 5)
    quality_score = max(20, min(100, score))

    return {
        "status": "success",
        "data_quality_score": quality_score,
        "is_healthy": len(issues) == 0,
        "has_historical_churn_label": bool(has_churn_label),
        "has_balance_data": bool(has_balance),
        "issues_found": issues,
        "warnings": warnings,
        "supported_analyses": supported,
        "unsupported_analyses": unsupported,
        "recommendation": (
            "Data is healthy for behavioral investigation."
            if quality_score >= 80 else
            "Data contains gaps or anomalies. Agent should report limitations when drawing conclusions."
        )
    }


def analyze_customers(
    dataset_path: Optional[str] = None,
    filter_segment: str = "all",
    metric_focus: str = "all"
) -> Dict[str, Any]:
    """
    Analyzes customer engagement, compares recent vs previous transactions,
    quantifies activity drops, inactivity length, logins, and friction factors.
    """
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    total_customers = len(df)
    results: Dict[str, Any] = {
        "status": "success",
        "total_analyzed": total_customers
    }

    # 1. Transaction Velocity Comparison
    if "transactions_last_30_days" in df.columns and "transactions_previous_30_days" in df.columns:
        tx_last = df["transactions_last_30_days"]
        tx_prev = df["transactions_previous_30_days"]
        
        # Transaction drop: customers whose transactions fell by > 40%
        # Handle zero previous transactions safely
        valid_prev = tx_prev > 0
        drop_pct = pd.Series(0.0, index=df.index)
        drop_pct[valid_prev] = (tx_prev[valid_prev] - tx_last[valid_prev]) / tx_prev[valid_prev]

        sharp_drop_customers = df[drop_pct >= 0.40]
        zero_activity_customers = df[(tx_last == 0) & valid_prev]

        results["transaction_trends"] = {
            "avg_transactions_last_30_days": round(float(tx_last.mean()), 2),
            "avg_transactions_previous_30_days": round(float(tx_prev.mean()), 2),
            "customers_with_sharp_decline_over_40pct": int(len(sharp_drop_customers)),
            "pct_customers_with_sharp_decline": round(float(len(sharp_drop_customers) / total_customers * 100), 2),
            "customers_dropped_to_zero_transactions": int(len(zero_activity_customers)),
        }

    # 2. Inactivity Breakdown
    if "days_since_last_transaction" in df.columns:
        days = df["days_since_last_transaction"]
        inactive_7d = int((days <= 7).sum())
        inactive_8_30d = int(((days > 7) & (days <= 30)).sum())
        inactive_31_60d = int(((days > 30) & (days <= 60)).sum())
        inactive_over_60d = int((days > 60).sum())

        results["inactivity_breakdown"] = {
            "active_within_7_days": {"count": inactive_7d, "percent": round(inactive_7d / total_customers * 100, 1)},
            "inactive_8_to_30_days": {"count": inactive_8_30d, "percent": round(inactive_8_30d / total_customers * 100, 1)},
            "dormant_31_to_60_days": {"count": inactive_31_60d, "percent": round(inactive_31_60d / total_customers * 100, 1)},
            "severely_inactive_over_60_days": {"count": inactive_over_60d, "percent": round(inactive_over_60d / total_customers * 100, 1)},
            "median_days_inactive": round(float(days.median()), 1),
            "max_days_inactive": int(days.max())
        }

    # 3. Logins and Digital Engagement
    if "logins_last_30_days" in df.columns:
        logins = df["logins_last_30_days"]
        low_login_count = int((logins <= 3).sum())
        results["login_engagement"] = {
            "avg_logins_last_30_days": round(float(logins.mean()), 1),
            "customers_with_very_low_logins_le_3": low_login_count,
            "pct_very_low_logins": round(float(low_login_count / total_customers * 100), 1)
        }

    # 4. Friction Metrics
    friction_summary = {}
    if "failed_transactions" in df.columns:
        failed = df["failed_transactions"]
        high_failures = int((failed >= 3).sum())
        friction_summary["customers_with_high_failed_tx_ge_3"] = high_failures
        friction_summary["avg_failed_transactions"] = round(float(failed.mean()), 2)

    if "support_complaints" in df.columns:
        complaints = df["support_complaints"]
        with_complaints = int((complaints >= 1).sum())
        friction_summary["customers_with_complaints"] = with_complaints
        friction_summary["pct_customers_with_complaints"] = round(float(with_complaints / total_customers * 100), 1)

    if friction_summary:
        results["friction_indicators"] = friction_summary

    # 5. Balance Trends (if available)
    if "balance_change_percentage" in df.columns:
        bal = df["balance_change_percentage"].dropna()
        falling_bal = int((bal < -25.0).sum())
        results["balance_dynamics"] = {
            "evaluated_rows": int(len(bal)),
            "customers_with_falling_balance_drop_gt_25pct": falling_bal,
            "pct_falling_balance": round(float(falling_bal / len(bal) * 100), 1),
            "avg_balance_change_pct": round(float(bal.mean()), 1)
        }

    return results


def calculate_churn_risk(
    dataset_path: Optional[str] = None,
    limit: int = 15
) -> Dict[str, Any]:
    """
    Calculates transparent heuristic churn risk scores (0-100) using
    behavioral indicators only (zero data leakage from historical labels).
    Assigns categories: Low (<35), Medium (35-59), High (>=60).
    """
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    total = len(df)
    scores = pd.Series(0.0, index=df.index)
    risk_drivers_list = [[] for _ in range(total)]

    # 1. Inactivity Factor (Up to 30 pts)
    if "days_since_last_transaction" in df.columns:
        days = df["days_since_last_transaction"].fillna(0)
        c1 = days >= 45
        c2 = (days >= 20) & (days < 45)
        c3 = (days >= 10) & (days < 20)

        scores += np.where(c1, 30.0, np.where(c2, 18.0, np.where(c3, 8.0, 0.0)))
        
        for idx in df[c1].index:
            risk_drivers_list[idx].append(f"Severe inactivity ({int(days[idx])} days)")
        for idx in df[c2].index:
            risk_drivers_list[idx].append(f"Noticeable inactivity ({int(days[idx])} days)")

    # 2. Transaction Velocity Drop Factor (Up to 25 pts)
    if "transactions_last_30_days" in df.columns and "transactions_previous_30_days" in df.columns:
        tx_last = df["transactions_last_30_days"].fillna(0)
        tx_prev = df["transactions_previous_30_days"].fillna(0)
        
        zero_drop = (tx_prev > 0) & (tx_last == 0)
        drop_ratio = np.where(tx_prev > 0, (tx_prev - tx_last) / tx_prev, 0.0)
        sharp_drop = (tx_prev > 0) & (drop_ratio >= 0.5) & (~zero_drop)
        mod_drop = (tx_prev > 0) & (drop_ratio >= 0.25) & (drop_ratio < 0.5)

        scores += np.where(zero_drop, 25.0, np.where(sharp_drop, 20.0, np.where(mod_drop, 10.0, 0.0)))

        for idx in df[zero_drop].index:
            risk_drivers_list[idx].append("Transactions ceased completely to 0")
        for idx in df[sharp_drop].index:
            pct = int(round(drop_ratio[idx] * 100))
            risk_drivers_list[idx].append(f"Sharp transaction drop (-{pct}%)")

    # 3. Low Digital Login Activity (Up to 15 pts)
    if "logins_last_30_days" in df.columns:
        logins = df["logins_last_30_days"].fillna(0)
        v_low = logins <= 2
        mod_low = (logins > 2) & (logins <= 6)
        scores += np.where(v_low, 15.0, np.where(mod_low, 8.0, 0.0))

        for idx in df[v_low].index:
            risk_drivers_list[idx].append(f"Infrequent digital logins ({int(logins[idx])}/mo)")

    # 4. Failed Transactions / Operational Friction (Up to 15 pts)
    if "failed_transactions" in df.columns:
        failed = df["failed_transactions"].fillna(0)
        high_fail = failed >= 3
        mod_fail = (failed >= 1) & (failed < 3)
        scores += np.where(high_fail, 15.0, np.where(mod_fail, 7.0, 0.0))

        for idx in df[high_fail].index:
            risk_drivers_list[idx].append(f"High failed transactions ({int(failed[idx])} failed)")

    # 5. Customer Complaints / Support Friction (Up to 15 pts)
    if "support_complaints" in df.columns:
        comp = df["support_complaints"].fillna(0)
        high_comp = comp >= 2
        mod_comp = comp == 1
        scores += np.where(high_comp, 15.0, np.where(mod_comp, 8.0, 0.0))

        for idx in df[high_comp | mod_comp].index:
            risk_drivers_list[idx].append(f"Unresolved complaints filed ({int(comp[idx])})")

    # Cap score at 100
    scores = np.clip(scores, 0.0, 100.0).round(1)
    
    # Categorize
    categories = pd.Series("Low", index=df.index)
    categories[scores >= 35.0] = "Medium"
    categories[scores >= 60.0] = "High"

    df_scored = df.copy()
    df_scored["risk_score"] = scores
    df_scored["risk_category"] = categories
    df_scored["primary_drivers"] = [", ".join(d[:2]) if d else "Stable activity" for d in risk_drivers_list]

    high_count = int((categories == "High").sum())
    med_count = int((categories == "Medium").sum())
    low_count = int((categories == "Low").sum())

    # Sort descending by risk score
    top_risk_df = df_scored.sort_values(by="risk_score", ascending=False).head(limit)
    
    selected_cols = [
        "customer_id", "risk_score", "risk_category", "primary_drivers",
        "days_since_last_transaction", "transactions_last_30_days",
        "transactions_previous_30_days", "failed_transactions", "support_complaints"
    ]
    avail_cols = [c for c in selected_cols if c in top_risk_df.columns]
    clean_top_df = top_risk_df[avail_cols].replace({np.nan: None})
    top_customers = clean_top_df.to_dict(orient="records")

    return {
        "status": "success",
        "total_evaluated": total,
        "scoring_methodology": "Transparent Rule-Based Heuristic (0-100 pts, zero historical label leakage)",
        "risk_distribution": {
            "high_risk": {"count": high_count, "percent": round(high_count / total * 100, 1)},
            "medium_risk": {"count": med_count, "percent": round(med_count / total * 100, 1)},
            "low_risk": {"count": low_count, "percent": round(low_count / total * 100, 1)}
        },
        "average_risk_score": round(float(scores.mean()), 1),
        "top_high_risk_customers": top_customers,
        "primary_churn_drivers_summary": [
            "Prolonged transaction inactivity (>30 days)",
            "Sharp decrease in monthly transaction volume (>40%)",
            "Severe drop in digital app login frequency",
            "Friction spikes from failed payment attempts and customer complaints"
        ]
    }


def extract_customer_cohorts(
    dataset_path: Optional[str] = None,
    cohort_type: str = "candidates",
    limit: int = 20
) -> Dict[str, Any]:
    """
    Extracts rich customer behavioral profiles for Agentic AI churn determination.
    Enables Gemini to inspect customer-level telemetry without data leakage from labels.
    """
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    total = len(df)
    limit = max(1, min(int(limit), 50))
    c_type = (cohort_type or "candidates").lower().strip()

    tx_last = df["transactions_last_30_days"].fillna(0) if "transactions_last_30_days" in df.columns else pd.Series(0, index=df.index)
    tx_prev = df["transactions_previous_30_days"].fillna(0) if "transactions_previous_30_days" in df.columns else pd.Series(0, index=df.index)
    days_inactive = df["days_since_last_transaction"].fillna(0) if "days_since_last_transaction" in df.columns else pd.Series(0, index=df.index)
    failed_tx = df["failed_transactions"].fillna(0) if "failed_transactions" in df.columns else pd.Series(0, index=df.index)
    complaints = df["support_complaints"].fillna(0) if "support_complaints" in df.columns else pd.Series(0, index=df.index)
    balance_change = df["balance_change_percentage"] if "balance_change_percentage" in df.columns else pd.Series(0.0, index=df.index)

    drop_ratio = np.where(tx_prev > 0, (tx_prev - tx_last) / tx_prev, 0.0)

    if c_type == "high_friction":
        mask = (failed_tx >= 2) | (complaints >= 1)
        sub_df = df[mask].sort_values(by=["failed_transactions", "support_complaints"], ascending=False)
    elif c_type == "dormant":
        mask = days_inactive >= 30
        sub_df = df[mask].sort_values(by="days_since_last_transaction", ascending=False)
    elif c_type == "sharp_drop":
        mask = drop_ratio >= 0.40
        sub_df = df[mask].copy()
        sub_df["_drop"] = drop_ratio[mask]
        sub_df = sub_df.sort_values(by="_drop", ascending=False)
    elif c_type == "balance_stressed":
        mask = balance_change < -20.0
        sub_df = df[mask].sort_values(by="balance_change_percentage", ascending=True)
    elif c_type == "loyal_benchmark":
        mask = (days_inactive <= 7) & (drop_ratio <= 0.0) & (failed_tx == 0) & (complaints == 0)
        sub_df = df[mask].head(limit)
    elif c_type == "candidates":
        stress_score = (
            np.where(days_inactive >= 30, 30, np.where(days_inactive >= 15, 15, 0)) +
            np.where(drop_ratio >= 0.5, 30, np.where(drop_ratio >= 0.25, 15, 0)) +
            np.where(failed_tx >= 2, 20, 0) +
            np.where(complaints >= 1, 20, 0)
        )
        cand_df = df.copy()
        cand_df["_stress"] = stress_score
        top_stressed = cand_df.sort_values(by="_stress", ascending=False).head(max(limit - 2, 1))
        controls = cand_df[cand_df["_stress"] <= 15].head(2)
        sub_df = pd.concat([top_stressed, controls]).drop_duplicates(subset=["customer_id"])
    else:
        sub_df = df.head(limit)

    selected = sub_df.head(limit)
    profiles = []
    for _, row in selected.iterrows():
        tx_p = int(row.get("transactions_previous_30_days", 0))
        tx_l = int(row.get("transactions_last_30_days", 0))
        change_pct = round(((tx_l - tx_p) / tx_p * 100), 1) if tx_p > 0 else 0.0
        bal = row.get("balance_change_percentage")
        bal_val = None if (pd.isna(bal) or bal is None) else round(float(bal), 1)
        app_m = row.get("app_usage_minutes")
        app_val = None if (pd.isna(app_m) or app_m is None) else int(app_m)

        profiles.append({
            "customer_id": str(row.get("customer_id", "")),
            "tenure_months": int(row.get("customer_tenure_months", 0)),
            "transactions_last_30_days": tx_l,
            "transactions_previous_30_days": tx_p,
            "transaction_change_pct": change_pct,
            "days_since_last_transaction": int(row.get("days_since_last_transaction", 0)),
            "average_transaction_amount": round(float(row.get("average_transaction_amount", 0)), 2),
            "balance_change_percentage": bal_val,
            "logins_last_30_days": int(row.get("logins_last_30_days", 0)),
            "failed_transactions": int(row.get("failed_transactions", 0)),
            "support_complaints": int(row.get("support_complaints", 0)),
            "app_usage_minutes": app_val
        })

    return {
        "status": "success",
        "cohort_type": c_type,
        "total_in_cohort": len(sub_df),
        "extracted_count": len(profiles),
        "profiles": profiles,
        "summary": f"Extracted {len(profiles)} customer profiles (cohort: '{c_type}') for agentic AI churn determination."
    }


def get_customer_profile(dataset_path: Optional[str] = None, customer_id: str = "") -> Dict[str, Any]:
    """Retrieves full behavioral telemetry for a specific customer ID."""
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}
    
    match = df[df["customer_id"] == customer_id.strip()]
    if len(match) == 0:
        return {"error": f"Customer ID '{customer_id}' not found in dataset."}
    
    row = match.iloc[0].to_dict()
    clean_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    return {"status": "success", "customer_profile": clean_row}


def summarize_churn_patterns(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Examines historical churn rates across customer segments and identifies
    associations with customer behavioral metrics when 'churned' label is available.
    """
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    if "churned" not in df.columns:
        return {
            "status": "unavailable",
            "error": (
                "Historical churn label 'churned' is missing from dataset. "
                "Historical validation and churn rate associations cannot be computed. "
                "Heuristic risk scoring via calculate_churn_risk() should be used instead."
            )
        }

    total = len(df)
    churn_series = df["churned"].dropna()
    churn_count = int((churn_series == 1).sum())
    overall_churn_rate = round(float(churn_count / total * 100), 2)

    breakdowns: Dict[str, Any] = {}

    # 1. Inactivity vs Churn
    if "days_since_last_transaction" in df.columns:
        inactive_group = np.where(
            df["days_since_last_transaction"] > 30, "Inactive (>30 days)", "Active (<=30 days)"
        )
        inact_churn = df.groupby(inactive_group)["churned"].agg(["count", "mean"])
        breakdowns["by_activity_recency"] = {
            str(group): {
                "customers": int(row["count"]),
                "churn_rate_pct": round(float(row["mean"] * 100), 1)
            }
            for group, row in inact_churn.iterrows()
        }

    # 2. Transaction Change vs Churn
    if "transactions_last_30_days" in df.columns and "transactions_previous_30_days" in df.columns:
        diff = df["transactions_previous_30_days"] - df["transactions_last_30_days"]
        decline_group = np.where(diff > 5, "Declining Transactions", "Stable or Increasing")
        tx_churn = df.groupby(decline_group)["churned"].agg(["count", "mean"])
        breakdowns["by_transaction_trend"] = {
            str(group): {
                "customers": int(row["count"]),
                "churn_rate_pct": round(float(row["mean"] * 100), 1)
            }
            for group, row in tx_churn.iterrows()
        }

    # 3. Complaints vs Churn
    if "support_complaints" in df.columns:
        comp_group = np.where(df["support_complaints"] >= 1, "With Complaints", "No Complaints")
        comp_churn = df.groupby(comp_group)["churned"].agg(["count", "mean"])
        breakdowns["by_support_complaints"] = {
            str(group): {
                "customers": int(row["count"]),
                "churn_rate_pct": round(float(row["mean"] * 100), 1)
            }
            for group, row in comp_churn.iterrows()
        }

    # 4. Failed Transactions vs Churn
    if "failed_transactions" in df.columns:
        fail_group = np.where(df["failed_transactions"] >= 2, "2+ Failed Transactions", "<2 Failed Transactions")
        fail_churn = df.groupby(fail_group)["churned"].agg(["count", "mean"])
        breakdowns["by_failed_transactions"] = {
            str(group): {
                "customers": int(row["count"]),
                "churn_rate_pct": round(float(row["mean"] * 100), 1)
            }
            for group, row in fail_churn.iterrows()
        }

    # 5. Balance Change vs Churn (if available)
    if "balance_change_percentage" in df.columns:
        valid_bal = df.dropna(subset=["balance_change_percentage"])
        bal_group = np.where(valid_bal["balance_change_percentage"] < -20.0, "Falling Balance (<-20%)", "Stable / Growing Balance")
        bal_churn = valid_bal.groupby(bal_group)["churned"].agg(["count", "mean"])
        breakdowns["by_balance_trend"] = {
            str(group): {
                "customers": int(row["count"]),
                "churn_rate_pct": round(float(row["mean"] * 100), 1)
            }
            for group, row in bal_churn.iterrows()
        }

    # 6. Mean Comparison between Churned vs Retained
    metric_cols = [
        "days_since_last_transaction", "transactions_last_30_days",
        "logins_last_30_days", "failed_transactions", "support_complaints"
    ]
    available_metrics = [c for c in metric_cols if c in df.columns]
    comparison = {}
    churned_df = df[df["churned"] == 1]
    retained_df = df[df["churned"] == 0]

    for col in available_metrics:
        c_mean = churned_df[col].dropna().mean()
        r_mean = retained_df[col].dropna().mean()
        comparison[col] = {
            "churned_mean": round(float(c_mean), 2),
            "retained_mean": round(float(r_mean), 2),
            "difference": round(float(c_mean - r_mean), 2)
        }

    return {
        "status": "success",
        "total_records": total,
        "historical_churn_rate_pct": overall_churn_rate,
        "churned_count": churn_count,
        "retained_count": total - churn_count,
        "segment_churn_rates": breakdowns,
        "metric_comparison_churned_vs_retained": comparison,
        "key_insight": (
            f"Overall historical churn is {overall_churn_rate}%. "
            "Inactivity (>30 days), falling transactions, and customer complaints exhibit "
            "the strongest statistical association with customer churn."
        )
    }
