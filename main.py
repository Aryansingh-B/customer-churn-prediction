"""
main.py
-------
End-to-end pipeline for Customer Churn Prediction.

Usage:
    python main.py

Steps:
    1. Load / generate dataset
    2. Clean & preprocess data
    3. Engineer features & run hypothesis tests
    4. Train Logistic Regression with hyperparameter tuning
    5. Evaluate model & generate all charts
    6. Load data into SQLite & run business SQL queries
"""

import sys
import os
import sqlite3
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from src.data_loader import load_raw_data
from src.preprocessing import clean_data, encode_features, save_processed_data
from src.feature_engineering import add_engineered_features, run_hypothesis_tests
from src.model import (
    prepare_features, split_data, handle_class_imbalance,
    build_pipeline, tune_hyperparameters, save_model
)
from src.evaluate import print_classification_report, generate_all_plots

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        📡  CUSTOMER CHURN PREDICTION PIPELINE  📡           ║
║            Telecom Dataset · Logistic Regression            ║
╚══════════════════════════════════════════════════════════════╝
"""


def load_to_sqlite(df_raw: pd.DataFrame, db_path: str = "data/telecom_churn.db") -> None:
    """Loads raw dataset into SQLite for SQL analysis."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df_raw.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[✓] SQLite DB created → {db_path}  (table: customers)")


def run_sql_insights(db_path: str = "data/telecom_churn.db") -> None:
    """Runs key business SQL queries and prints results."""
    conn = sqlite3.connect(db_path)

    queries = {
        "Overall Churn Rate": """
            SELECT
                COUNT(*) AS total_customers,
                SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS churned,
                ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
            FROM customers
        """,
        "Churn by Contract Type": """
            SELECT Contract,
                   COUNT(*) AS total,
                   ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_pct
            FROM customers GROUP BY Contract ORDER BY churn_pct DESC
        """,
        "Revenue at Risk": """
            SELECT
                ROUND(SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END), 2) AS revenue_at_risk,
                ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges ELSE 0 END) / SUM(MonthlyCharges), 2) AS pct
            FROM customers
        """
    }

    print("\n" + "─" * 60)
    print("  🗃️  SQL BUSINESS INSIGHTS")
    print("─" * 60)

    for title, query in queries.items():
        print(f"\n📌 {title}:")
        result = pd.read_sql_query(query, conn)
        print(result.to_string(index=False))

    conn.close()


def main():
    print(BANNER)

    # ── Step 1: Load Data ──────────────────────────────────────
    print("\n[STEP 1] Loading dataset...")
    df_raw = load_raw_data()

    # ── Step 2: Clean & Preprocess ────────────────────────────
    print("\n[STEP 2] Cleaning & preprocessing...")
    df_clean = clean_data(df_raw)

    # ── Step 3: Feature Engineering & Hypothesis Tests ─────────
    print("\n[STEP 3] Feature engineering & hypothesis testing...")
    df_eng = add_engineered_features(df_clean)
    run_hypothesis_tests(df_clean)

    # Encode after engineering
    df_encoded = encode_features(df_eng)
    save_processed_data(df_encoded)

    # ── Step 4: Train Model ────────────────────────────────────
    print("\n[STEP 4] Training Logistic Regression model...")
    X, y = prepare_features(df_encoded)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_res, y_res = handle_class_imbalance(X_train, y_train)

    pipeline = build_pipeline()
    print("\n[→] Tuning hyperparameters via GridSearchCV...")
    best_model = tune_hyperparameters(pipeline, X_res, y_res)
    best_model.fit(X_res, y_res)
    save_model(best_model)

    # ── Step 5: Evaluate ──────────────────────────────────────
    print("\n[STEP 5] Evaluating model...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print_classification_report(y_test, y_pred, y_prob)
    generate_all_plots(best_model, X_test, y_test, y_pred, y_prob, df_raw)

    # ── Step 6: SQL Analysis ───────────────────────────────────
    print("\n[STEP 6] Loading data to SQLite & running SQL queries...")
    load_to_sqlite(df_raw)
    run_sql_insights()

    print("\n" + "═" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("  📁 Outputs saved in /reports/   |   /models/   |   /data/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
