"""
Synthetic Banking Customer Churn Data Generator.
Generates realistic customer behavior data for digital banking, mobile wallets, and FinTech.
Uses reproducible random seeds and realistic behavioral archetypes.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_customers(n_customers: int = 750, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic customer records with realistic behavioral archetypes:
    - Active loyal customers
    - Declining engagement customers
    - Dormant / inactive customers
    - High-friction customers (failed transactions, complaints)
    - Balance-stressed customers
    - Ambiguous edge cases
    """
    rng = np.random.default_rng(seed)

    records = []
    
    # Archetype distribution:
    # 0: Loyal & Active (~45%)
    # 1: Declining Engagement (~20%)
    # 2: Inactive / Dormant (~15%)
    # 3: High Friction / Frustrated (~12%)
    # 4: Ambiguous / Borderline (~8%)
    archetypes = rng.choice(
        [0, 1, 2, 3, 4],
        size=n_customers,
        p=[0.45, 0.20, 0.15, 0.12, 0.08]
    )

    for i in range(n_customers):
        cust_id = f"CUST-{1000 + i + 1}"
        archetype = archetypes[i]
        
        # Tenure
        tenure = int(rng.integers(1, 49))

        if archetype == 0:
            # Loyal & Active
            tx_prev = int(rng.integers(12, 45))
            tx_last = int(rng.integers(max(10, tx_prev - 5), tx_prev + 8))
            days_inactive = int(rng.integers(0, 7))
            avg_tx_amt = round(float(rng.uniform(25.0, 180.0)), 2)
            balance_change = round(float(rng.normal(5.0, 12.0)), 1)
            logins = int(rng.integers(15, 50))
            failed_tx = int(rng.choice([0, 1], p=[0.92, 0.08]))
            complaints = int(rng.choice([0, 1], p=[0.96, 0.04]))
            app_minutes = int(rng.integers(120, 420))
            active_last_month = 1
            # Low churn chance: ~4%
            churn_prob = 0.04

        elif archetype == 1:
            # Declining Engagement
            tx_prev = int(rng.integers(15, 40))
            # 40% to 80% drop in transactions
            drop_factor = rng.uniform(0.15, 0.55)
            tx_last = int(max(1, round(tx_prev * drop_factor)))
            days_inactive = int(rng.integers(12, 35))
            avg_tx_amt = round(float(rng.uniform(15.0, 120.0)), 2)
            balance_change = round(float(rng.normal(-25.0, 15.0)), 1)
            logins = int(rng.integers(3, 14))
            failed_tx = int(rng.choice([0, 1, 2], p=[0.75, 0.20, 0.05]))
            complaints = int(rng.choice([0, 1, 2], p=[0.80, 0.16, 0.04]))
            app_minutes = int(rng.integers(30, 110))
            active_last_month = 1
            # Medium-High churn chance: ~62%
            churn_prob = 0.62

        elif archetype == 2:
            # Inactive / Dormant
            tx_prev = int(rng.integers(2, 18))
            tx_last = 0
            days_inactive = int(rng.integers(32, 90))
            avg_tx_amt = round(float(rng.uniform(10.0, 95.0)), 2)
            balance_change = round(float(rng.normal(-45.0, 20.0)), 1)
            logins = int(rng.integers(0, 4))
            failed_tx = 0
            complaints = int(rng.choice([0, 1], p=[0.88, 0.12]))
            app_minutes = int(rng.integers(0, 25))
            active_last_month = 0
            # High churn chance: ~82%
            churn_prob = 0.82

        elif archetype == 3:
            # High Friction / Frustrated
            tx_prev = int(rng.integers(10, 32))
            tx_last = int(rng.integers(2, max(3, tx_prev - 4)))
            days_inactive = int(rng.integers(4, 25))
            avg_tx_amt = round(float(rng.uniform(20.0, 160.0)), 2)
            balance_change = round(float(rng.normal(-15.0, 22.0)), 1)
            logins = int(rng.integers(8, 30))
            failed_tx = int(rng.integers(3, 8))
            complaints = int(rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10]))
            app_minutes = int(rng.integers(40, 180))
            active_last_month = 1
            # High churn chance: ~68%
            churn_prob = 0.68

        else:
            # Ambiguous / Borderline
            tx_prev = int(rng.integers(8, 25))
            tx_last = int(rng.integers(6, 28))
            days_inactive = int(rng.integers(8, 22))
            avg_tx_amt = round(float(rng.uniform(15.0, 220.0)), 2)
            balance_change = round(float(rng.normal(-5.0, 18.0)), 1)
            logins = int(rng.integers(6, 20))
            failed_tx = int(rng.choice([0, 1, 2], p=[0.70, 0.22, 0.08]))
            complaints = int(rng.choice([0, 1], p=[0.85, 0.15]))
            app_minutes = int(rng.integers(50, 160))
            active_last_month = 1
            # Ambiguous churn chance: ~28%
            churn_prob = 0.28

        # Churn label with noise
        is_churned = int(rng.uniform(0, 1) < churn_prob)

        records.append({
            "customer_id": cust_id,
            "customer_tenure_months": tenure,
            "transactions_last_30_days": tx_last,
            "transactions_previous_30_days": tx_prev,
            "days_since_last_transaction": days_inactive,
            "average_transaction_amount": avg_tx_amt,
            "balance_change_percentage": balance_change,
            "logins_last_30_days": logins,
            "failed_transactions": failed_tx,
            "support_complaints": complaints,
            "app_usage_minutes": app_minutes,
            "last_month_active": active_last_month,
            "churned": is_churned
        })

    df = pd.DataFrame(records)

    # Intentionally insert a few realistic missing values (~2-3% in optional/noisy columns)
    # to demonstrate that the agent inspects data and detects missing values
    missing_indices_balance = rng.choice(n_customers, size=int(n_customers * 0.025), replace=False)
    df.loc[missing_indices_balance, "balance_change_percentage"] = np.nan

    missing_indices_app = rng.choice(n_customers, size=int(n_customers * 0.02), replace=False)
    df.loc[missing_indices_app, "app_usage_minutes"] = np.nan

    return df


def ensure_data_directory(filepath: str = "data/customers.csv") -> str:
    """Ensures data directory exists and generates dataset if absent."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        df = generate_synthetic_customers()
        df.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    path = ensure_data_directory()
    df = pd.read_csv(path)
    print(f"Generated {len(df)} rows at {path}")
    print(df.head(2))
