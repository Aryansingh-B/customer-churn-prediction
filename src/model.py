"""
model.py
--------
Trains and persists a Logistic Regression churn prediction model
with cross-validation and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


MODEL_PATH = "models/logistic_regression_churn.pkl"


def prepare_features(df: pd.DataFrame, target_col: str = "Churn") -> tuple:
    """
    Splits encoded DataFrame into features (X) and target (y).

    Args:
        df: Fully encoded DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple (X, y) as DataFrames.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop any remaining non-numeric columns (e.g., tenure_group if not encoded)
    df_numeric = df.select_dtypes(include=[np.number])

    X = df_numeric.drop(columns=[target_col], errors="ignore")
    y = df_numeric[target_col]

    print(f"[✓] Features: {X.shape[1]} columns | Target: '{target_col}' | Samples: {len(y):,}")
    print(f"    Class distribution → Churn=1: {y.mean():.1%}  |  No Churn=0: {(1-y.mean()):.1%}")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[✓] Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def handle_class_imbalance(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Applies SMOTE to handle class imbalance in training data.

    Args:
        X_train, y_train: Training data.

    Returns:
        Resampled (X_train_res, y_train_res).
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[✓] SMOTE applied → Churn=1: {y_res.sum():,} | No Churn=0: {(y_res == 0).sum():,}")
    return X_res, y_res


def build_pipeline() -> Pipeline:
    """
    Builds a Scikit-learn Pipeline:
    StandardScaler → LogisticRegression

    Returns:
        Untrained Pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ))
    ])
    return pipeline


def tune_hyperparameters(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Runs GridSearchCV to find optimal regularization strength (C).

    Args:
        pipeline: Sklearn Pipeline.
        X_train, y_train: Training data.

    Returns:
        Best estimator Pipeline.
    """
    param_grid = {
        "clf__C": [0.01, 0.1, 1, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"[✓] Best params: {grid_search.best_params_}")
    print(f"[✓] Best CV F1 Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Runs 5-fold stratified cross-validation.

    Returns:
        Dict of metric name → (mean, std).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {}

    for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=-1)
        metrics[metric] = (scores.mean(), scores.std())
        print(f"   {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

    return metrics


def save_model(model: Pipeline, path: str = MODEL_PATH) -> None:
    """Persists trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[✓] Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> Pipeline:
    """Loads a persisted model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[✓] Model loaded ← {path}")
    return model


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data, encode_features

    df = load_raw_data()
    df = clean_data(df)
    df = encode_features(df)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_res, y_train_res = handle_class_imbalance(X_train, y_train)

    pipeline = build_pipeline()

    print("\n[→] Running 5-fold cross-validation...")
    cross_validate_model(pipeline, X_train_res, y_train_res)

    print("\n[→] Tuning hyperparameters...")
    best_model = tune_hyperparameters(pipeline, X_train_res, y_train_res)

    save_model(best_model)
