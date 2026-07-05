"""
preprocessing.py
----------------
Handles data cleaning, type casting, and encoding for the churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw telecom churn data:
    - Converts TotalCharges to numeric
    - Fills missing values
    - Drops irrelevant columns

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Convert TotalCharges (may have spaces as NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with MonthlyCharges * tenure
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]

    # Drop customer ID (non-predictive)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    print(f"[✓] Cleaning complete. Shape: {df.shape} | Missing values: {df.isna().sum().sum()}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features using Label Encoding for binary columns
    and One-Hot Encoding for multi-class columns.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Encoded DataFrame ready for modeling.
    """
    df = df.copy()

    # Binary columns → Label Encode
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Multi-class → One-Hot Encode
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=[c for c in multi_cols if c in df.columns], drop_first=True)

    print(f"[✓] Encoding complete. Final shape: {df.shape}")
    return df


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list = None
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardizes numeric features using StandardScaler (fit on train only).

    Args:
        X_train: Training feature set.
        X_test: Test feature set.
        numeric_cols: Columns to scale. Defaults to tenure, MonthlyCharges, TotalCharges.

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler).
    """
    if numeric_cols is None:
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"[✓] Scaled columns: {numeric_cols}")
    return X_train, X_test, scaler


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/churn_processed.csv") -> None:
    """Saves processed data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] Processed data saved → {path}")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_raw_data

    df = load_raw_data()
    df = clean_data(df)
    df = encode_features(df)
    save_processed_data(df)
    print(df.head())
