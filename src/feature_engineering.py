"""
feature_engineering.py
-----------------------
Creates derived features and runs statistical hypothesis tests
to validate business assumptions about churn drivers.
"""

import pandas as pd
import numpy as np
from scipy import stats


# ─────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-driven engineered features to the dataset.

    New features:
    - tenure_group      : Customer lifecycle segment (New / Growing / Loyal / Champion)
    - avg_monthly_spend : TotalCharges / tenure (handles tenure=0 edge case)
    - services_count    : Number of add-on services subscribed
    - is_high_value     : MonthlyCharges > 75th percentile

    Args:
        df: Cleaned, raw (pre-encoded) DataFrame.

    Returns:
        DataFrame with new feature columns appended.
    """
    df = df.copy()

    # 1. Tenure group — lifecycle segmentation
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New (0-12m)", "Growing (1-2yr)", "Loyal (2-4yr)", "Champion (4yr+)"]
    )

    # 2. Average monthly spend (handles new customers with tenure=0)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # 3. Count of add-on services subscribed
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    existing_addons = [c for c in addon_cols if c in df.columns]
    df["services_count"] = df[existing_addons].apply(
        lambda row: sum(row == "Yes"), axis=1
    )

    # 4. High-value customer flag
    threshold = df["MonthlyCharges"].quantile(0.75)
    df["is_high_value"] = (df["MonthlyCharges"] > threshold).astype(int)

    print("[✓] Engineered features added: tenure_group, avg_monthly_spend, services_count, is_high_value")
    return df


# ─────────────────────────────────────────────
#  Hypothesis Testing
# ─────────────────────────────────────────────

def run_hypothesis_tests(df: pd.DataFrame) -> None:
    """
    Runs statistical hypothesis tests to validate key business assumptions.

    Tests:
    1. Do month-to-month customers churn significantly more than others?
       → Chi-Square test: Contract Type vs Churn
    2. Is there a significant difference in monthly charges between churned/retained customers?
       → Independent t-test: MonthlyCharges by Churn
    3. Do customers with shorter tenure churn more?
       → Point-biserial correlation: tenure vs Churn

    Args:
        df: Raw (pre-encoded) DataFrame with 'Churn' as 'Yes'/'No'.
    """
    print("\n" + "=" * 60)
    print("  📊 HYPOTHESIS TESTING RESULTS")
    print("=" * 60)

    churn_binary = (df["Churn"] == "Yes").astype(int)

    # ── Test 1: Contract Type vs Churn (Chi-Square) ──────────────
    if "Contract" in df.columns:
        contingency = pd.crosstab(df["Contract"], df["Churn"])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        print(f"\n🔬 Test 1: Contract Type → Churn  (Chi-Square)")
        print(f"   H₀: Contract type has NO effect on churn")
        print(f"   χ² = {chi2:.2f}  |  p-value = {p_val:.2e}  |  df = {dof}")
        print(f"   ✅ REJECT H₀ — Contract type significantly affects churn" if p_val < 0.05
              else f"   ❌ FAIL TO REJECT H₀")

    # ── Test 2: Monthly Charges by Churn (t-test) ─────────────────
    churned = df.loc[df["Churn"] == "Yes", "MonthlyCharges"]
    retained = df.loc[df["Churn"] == "No", "MonthlyCharges"]
    t_stat, p_val = stats.ttest_ind(churned, retained)
    print(f"\n🔬 Test 2: Monthly Charges → Churn  (Independent t-test)")
    print(f"   H₀: No difference in monthly charges between churned/retained")
    print(f"   Mean (churned) = ${churned.mean():.2f}  |  Mean (retained) = ${retained.mean():.2f}")
    print(f"   t = {t_stat:.2f}  |  p-value = {p_val:.2e}")
    print(f"   ✅ REJECT H₀ — Churned customers pay significantly more" if p_val < 0.05
          else f"   ❌ FAIL TO REJECT H₀")

    # ── Test 3: Tenure vs Churn (Point-biserial correlation) ──────
    corr, p_val = stats.pointbiserialr(churn_binary, df["tenure"])
    print(f"\n🔬 Test 3: Tenure → Churn  (Point-Biserial Correlation)")
    print(f"   H₀: Tenure is NOT correlated with churn")
    print(f"   r = {corr:.3f}  |  p-value = {p_val:.2e}")
    print(f"   ✅ REJECT H₀ — Shorter tenure customers churn more" if p_val < 0.05
          else f"   ❌ FAIL TO REJECT H₀")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data

    df = load_raw_data()
    df_clean = clean_data(df)

    # Hypothesis tests on raw (pre-encoded) data
    run_hypothesis_tests(df_clean)

    # Feature engineering
    df_eng = add_engineered_features(df_clean)
    print("\nEngineered feature preview:")
    print(df_eng[["tenure", "tenure_group", "avg_monthly_spend", "services_count", "is_high_value"]].head(10))
