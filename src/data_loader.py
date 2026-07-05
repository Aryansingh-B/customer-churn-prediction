"""
data_loader.py
--------------
Generates a synthetic telecom churn dataset and provides loading utilities.
Mirrors the structure of real-world telecom datasets (e.g., IBM Telco).
"""

import pandas as pd
import numpy as np
import os

SEED = 42
N_CUSTOMERS = 7043


def generate_telecom_dataset(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    """
    Generates a realistic synthetic telecom customer churn dataset.

    Args:
        n: Number of customer records to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with telecom customer features and churn label.
    """
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, n)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n)
    )

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_security = np.where(internet_service == "No", "No internet service",
                               rng.choice(["Yes", "No"], n))
    online_backup = np.where(internet_service == "No", "No internet service",
                             rng.choice(["Yes", "No"], n))
    device_protection = np.where(internet_service == "No", "No internet service",
                                 rng.choice(["Yes", "No"], n))
    tech_support = np.where(internet_service == "No", "No internet service",
                            rng.choice(["Yes", "No"], n))
    streaming_tv = np.where(internet_service == "No", "No internet service",
                            rng.choice(["Yes", "No"], n))
    streaming_movies = np.where(internet_service == "No", "No internet service",
                                rng.choice(["Yes", "No"], n))

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    monthly_charges = np.round(rng.uniform(18, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure + rng.uniform(0, 50, n), 2)

    # Churn probability influenced by real-world drivers
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet_service == "Fiber optic")
        + 0.10 * (payment_method == "Electronic check")
        + 0.08 * (tenure < 6)
        - 0.05 * (tenure > 36)
        - 0.05 * (contract == "Two year")
        + rng.uniform(-0.05, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    return df


def save_raw_data(df: pd.DataFrame, path: str = "data/raw/telecom_churn.csv") -> None:
    """Saves the raw dataset to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] Raw dataset saved → {path}  ({len(df):,} records)")


def load_raw_data(path: str = "data/raw/telecom_churn.csv") -> pd.DataFrame:
    """Loads the raw dataset from CSV."""
    if not os.path.exists(path):
        print("[!] Dataset not found. Generating synthetic data...")
        df = generate_telecom_dataset()
        save_raw_data(df, path)
        return df
    df = pd.read_csv(path)
    print(f"[✓] Loaded dataset → {path}  ({len(df):,} records, {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    df = generate_telecom_dataset()
    save_raw_data(df)
    print(df.head())
    print(f"\nChurn rate: {(df['Churn'] == 'Yes').mean():.1%}")
