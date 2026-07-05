"""
test_preprocessing.py
---------------------
Unit tests for data cleaning and encoding functions.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_telecom_dataset
from src.preprocessing import clean_data, encode_features


@pytest.fixture(scope="module")
def raw_df():
    """Shared raw dataset across all tests."""
    return generate_telecom_dataset(n=500, seed=99)


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="module")
def encoded_df(clean_df):
    return encode_features(clean_df)


# ─────────────────────────────────────────────
#  Data Loading Tests
# ─────────────────────────────────────────────

class TestDataLoader:
    def test_dataset_shape(self, raw_df):
        assert raw_df.shape == (500, 21), "Dataset should have 500 rows and 21 columns"

    def test_required_columns(self, raw_df):
        required = ["customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges", "Contract"]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_churn_values(self, raw_df):
        assert set(raw_df["Churn"].unique()).issubset({"Yes", "No"}), "Churn must be 'Yes' or 'No'"

    def test_tenure_range(self, raw_df):
        assert raw_df["tenure"].between(0, 72).all(), "Tenure should be between 0 and 72"

    def test_churn_rate_reasonable(self, raw_df):
        churn_rate = (raw_df["Churn"] == "Yes").mean()
        assert 0.10 <= churn_rate <= 0.50, f"Churn rate {churn_rate:.1%} looks unrealistic"


# ─────────────────────────────────────────────
#  Cleaning Tests
# ─────────────────────────────────────────────

class TestCleaning:
    def test_no_missing_values(self, clean_df):
        assert clean_df.isna().sum().sum() == 0, "Cleaned data should have no missing values"

    def test_customer_id_dropped(self, clean_df):
        assert "customerID" not in clean_df.columns, "customerID should be dropped after cleaning"

    def test_total_charges_numeric(self, clean_df):
        assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"]), \
            "TotalCharges should be numeric after cleaning"

    def test_monthly_charges_positive(self, clean_df):
        assert (clean_df["MonthlyCharges"] > 0).all(), "MonthlyCharges should all be positive"


# ─────────────────────────────────────────────
#  Encoding Tests
# ─────────────────────────────────────────────

class TestEncoding:
    def test_all_columns_numeric(self, encoded_df):
        non_numeric = encoded_df.select_dtypes(exclude=[np.number]).columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns remaining: {non_numeric}"

    def test_churn_binary(self, encoded_df):
        assert set(encoded_df["Churn"].unique()).issubset({0, 1}), "Churn must be 0 or 1 after encoding"

    def test_no_missing_after_encoding(self, encoded_df):
        assert encoded_df.isna().sum().sum() == 0, "No missing values after encoding"

    def test_expanded_columns(self, clean_df, encoded_df):
        # One-hot encoding should increase column count
        assert encoded_df.shape[1] >= clean_df.shape[1], \
            "Encoded df should have more or equal columns than cleaned df"

    def test_contract_columns_exist(self, encoded_df):
        # After OHE, Contract_Month-to-month or Contract_One year should exist
        contract_cols = [c for c in encoded_df.columns if "Contract" in c]
        assert len(contract_cols) > 0, "No Contract OHE columns found"


# ─────────────────────────────────────────────
#  Integration Test
# ─────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_runs(self):
        df = generate_telecom_dataset(n=200, seed=7)
        df_clean = clean_data(df)
        df_enc = encode_features(df_clean)

        assert df_enc.shape[0] == 200
        assert "Churn" in df_enc.columns
        assert df_enc.select_dtypes(exclude=[np.number]).shape[1] == 0

    def test_reproducibility(self):
        df1 = generate_telecom_dataset(n=100, seed=42)
        df2 = generate_telecom_dataset(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
