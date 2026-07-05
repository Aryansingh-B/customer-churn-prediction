"""
evaluate.py
-----------
Evaluates model performance and generates all visualizations:
- Confusion Matrix
- ROC Curve
- Feature Importances
- Churn Rate by Segment
- Tenure Distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

REPORTS_DIR = "reports"
PALETTE = {"Churned": "#E84545", "Retained": "#2B9348"}

# ── Style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def print_classification_report(y_test, y_pred, y_prob) -> dict:
    """Prints a full classification report and returns metric dict."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    print("\n" + "=" * 50)
    print("  📈 MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print("=" * 50)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    return metrics


# ─────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, save: bool = True) -> None:
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax, linewidths=0.5
    )
    ax.set_title("Confusion Matrix", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    if save:
        fig.savefig(f"{REPORTS_DIR}/confusion_matrix.png")
        print(f"[✓] Saved → {REPORTS_DIR}/confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_test, y_prob, save: bool = True) -> None:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2B9348", lw=2, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2B9348")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        fig.savefig(f"{REPORTS_DIR}/roc_curve.png")
        print(f"[✓] Saved → {REPORTS_DIR}/roc_curve.png")
    plt.close()


def plot_feature_importance(model: Pipeline, feature_names: list, top_n: int = 15, save: bool = True) -> None:
    """
    Horizontal bar chart of top-N feature importances from Logistic Regression coefficients.
    """
    coefs = model.named_steps["clf"].coef_[0]
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.abs(coefs),
        "Direction": np.where(coefs > 0, "Increases Churn", "Decreases Churn")
    }).sort_values("Importance", ascending=False).head(top_n)

    colors = importance_df["Direction"].map({
        "Increases Churn": "#E84545",
        "Decreases Churn": "#2B9348"
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color=colors[::-1], edgecolor="white"
    )
    ax.set_xlabel("Coefficient Magnitude (Absolute)")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#E84545", label="Increases Churn Risk"),
        Patch(color="#2B9348", label="Decreases Churn Risk")
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    if save:
        fig.savefig(f"{REPORTS_DIR}/feature_importance.png")
        print(f"[✓] Saved → {REPORTS_DIR}/feature_importance.png")
    plt.close()


def plot_churn_by_segment(df_raw: pd.DataFrame, save: bool = True) -> None:
    """
    Multi-panel chart showing churn rates across key business segments:
    contract type, tenure group, internet service, payment method.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Churn Rate by Customer Segment", fontsize=16, fontweight="bold", y=1.01)

    segments = [
        ("Contract", "Contract Type"),
        ("InternetService", "Internet Service"),
        ("PaymentMethod", "Payment Method"),
    ]

    # Tenure group
    df_raw = df_raw.copy()
    df_raw["tenure_group"] = pd.cut(
        df_raw["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "1-2yr", "2-4yr", "4yr+"]
    )

    all_segments = segments + [("tenure_group", "Tenure Group")]
    axes_flat = axes.flatten()

    for i, (col, title) in enumerate(all_segments):
        if col not in df_raw.columns:
            continue
        churn_rate = df_raw.groupby(col)["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        churn_rate.columns = [col, "Churn Rate (%)"]
        churn_rate = churn_rate.sort_values("Churn Rate (%)", ascending=False)

        bars = axes_flat[i].bar(
            churn_rate[col].astype(str),
            churn_rate["Churn Rate (%)"],
            color="#E84545", alpha=0.85, edgecolor="white"
        )
        axes_flat[i].set_title(title, fontweight="bold")
        axes_flat[i].set_ylabel("Churn Rate (%)")
        axes_flat[i].tick_params(axis="x", rotation=20)
        axes_flat[i].set_ylim(0, churn_rate["Churn Rate (%)"].max() * 1.25)

        # Value labels on bars
        for bar in bars:
            h = bar.get_height()
            axes_flat[i].text(
                bar.get_x() + bar.get_width() / 2.,
                h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()
    if save:
        fig.savefig(f"{REPORTS_DIR}/churn_by_segment.png")
        print(f"[✓] Saved → {REPORTS_DIR}/churn_by_segment.png")
    plt.close()


def plot_tenure_distribution(df_raw: pd.DataFrame, save: bool = True) -> None:
    """Overlapping KDE plots of tenure for churned vs retained customers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, color in [("Yes", "#E84545"), ("No", "#2B9348")]:
        subset = df_raw[df_raw["Churn"] == label]["tenure"]
        name = "Churned" if label == "Yes" else "Retained"
        subset.plot.kde(ax=ax, label=name, color=color, lw=2)
        ax.fill_between(
            np.linspace(subset.min(), subset.max(), 200),
            0,
            [ax.lines[-1].get_ydata()[j] for j in range(200)],
            alpha=0.1, color=color
        )

    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Density")
    ax.set_title("Tenure Distribution: Churned vs Retained", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(f"{REPORTS_DIR}/tenure_distribution.png")
        print(f"[✓] Saved → {REPORTS_DIR}/tenure_distribution.png")
    plt.close()


def generate_all_plots(model, X_test, y_test, y_pred, y_prob, df_raw: pd.DataFrame) -> None:
    """Generates and saves all evaluation plots."""
    print("\n[→] Generating evaluation charts...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_feature_importance(model, list(X_test.columns))
    plot_churn_by_segment(df_raw)
    plot_tenure_distribution(df_raw)
    print(f"\n[✓] All charts saved to /{REPORTS_DIR}/")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data, encode_features
    from src.model import prepare_features, split_data, handle_class_imbalance, build_pipeline, tune_hyperparameters

    df_raw = load_raw_data()
    df = clean_data(df_raw)
    df_enc = encode_features(df)

    X, y = prepare_features(df_enc)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_res, y_res = handle_class_imbalance(X_train, y_train)

    pipeline = build_pipeline()
    best_model = tune_hyperparameters(pipeline, X_res, y_res)
    best_model.fit(X_res, y_res)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print_classification_report(y_test, y_pred, y_prob)
    generate_all_plots(best_model, X_test, y_test, y_pred, y_prob, df_raw)
