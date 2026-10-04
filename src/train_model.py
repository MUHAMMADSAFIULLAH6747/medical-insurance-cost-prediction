"""Train Multiple Linear Regression and persist model + metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from download_data import download_dataset
from preprocess import FEATURE_COLUMNS, clean_dataframe

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "insurance.csv"
MODEL_PATH = ROOT / "models" / "insurance_cost_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIGURES_DIR = ROOT / "docs" / "figures"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        download_dataset()
    return pd.read_csv(DATA_PATH)


def train() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    df = clean_dataframe(raw)

    X = df[FEATURE_COLUMNS]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "n_rows_raw": int(len(raw)),
        "n_rows_clean": int(len(df)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "mae": float(mean_absolute_error(y_test, y_pred_test)),
        "mse": float(mean_squared_error(y_test, y_pred_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "intercept": float(model.intercept_),
        "coefficients": {
            name: float(coef) for name, coef in zip(FEATURE_COLUMNS, model.coef_)
        },
        "sample_predictions": [
            {
                "actual": float(a),
                "predicted": float(p),
                "error": float(p - a),
                "abs_error": float(abs(p - a)),
            }
            for a, p in zip(y_test.values[:10], y_pred_test[:10])
        ],
    }

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": {
                "r2_test": metrics["r2_test"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            },
        },
        MODEL_PATH,
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Evaluation figures for README / documentation
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axes[0].scatter(
        y_test, y_pred_test, alpha=0.65, color="#1E3A8A", edgecolors="k", linewidths=0.4
    )
    axes[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Ideal Fit (y = x)",
    )
    axes[0].set_title("Actual vs Predicted Medical Charges")
    axes[0].set_xlabel("Actual Charges ($)")
    axes[0].set_ylabel("Predicted Charges ($)")
    axes[0].legend(loc="upper left")

    residuals = y_test - y_pred_test
    sns.histplot(residuals, kde=True, color="#2563EB", bins=25, ax=axes[1])
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    axes[1].set_title("Residual Error Distribution")
    axes[1].set_xlabel("Residual Error ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "evaluation_plots.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print("=" * 55)
    print("MODEL TRAINING COMPLETE")
    print("=" * 55)
    print(f"Training R2: {metrics['r2_train']:.4f}")
    print(f"Testing  R2: {metrics['r2_test']:.4f}")
    print(f"MAE:         ${metrics['mae']:,.2f}")
    print(f"RMSE:        ${metrics['rmse']:,.2f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Figure:      {fig_path}")
    return metrics


if __name__ == "__main__":
    train()
