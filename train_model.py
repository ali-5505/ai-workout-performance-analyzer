"""
train_model.py

AI Workout Performance Analyzer
--------------------------------
Predicts next-session workout performance using training load, sleep,
macro/nutrition adherence, and recovery metrics (resting heart rate, soreness).

Pipeline steps:
1. Load data (generates synthetic data if no CSV is present).
2. Feature engineering: rolling averages, trend deltas, macro adherence.
3. Train/test split.
4. Train Ridge Regression and Random Forest models.
5. Evaluate with cross-validation, MAE, and R^2.
6. Produce diagnostic plots: predicted vs actual, residuals.

Run:
    python train_model.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from generate_data import generate_workout_data

FEATURE_COLUMNS = [
    "sleep_hours",
    "training_load",
    "protein_g",
    "calories",
    "resting_hr",
    "soreness",
    "macro_adherence",
    "sleep_7d_avg",
    "load_7d_avg",
    "load_trend_delta",
    "resting_hr_7d_avg",
    "resting_hr_delta",
]
TARGET_COLUMN = "next_session_performance"


def load_data(csv_path: str = "workout_data.csv") -> pd.DataFrame:
    """Load data from CSV if present, otherwise generate and save it."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["date"])
    df = generate_workout_data()
    df.to_csv(csv_path, index=False)
    return df


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    print(f"\n--- {name} ---")
    print(f"Test MAE:        {mae:.2f}")
    print(f"Test R^2:        {r2:.3f}")
    print(f"5-fold CV MAE:   {cv_mae:.2f} (+/- {cv_scores.std():.2f})")

    return {"name": name, "model": model, "preds": preds, "mae": mae, "r2": r2}


def plot_diagnostics(results, y_test, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(len(results), 2, figsize=(11, 4.5 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for i, res in enumerate(results):
        preds = res["preds"]
        residuals = y_test - preds

        # Predicted vs actual
        ax1 = axes[i][0]
        ax1.scatter(y_test, preds, alpha=0.5, edgecolor="k", linewidth=0.3)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax1.plot(lims, lims, "r--", linewidth=1)
        ax1.set_xlabel("Actual performance score")
        ax1.set_ylabel("Predicted performance score")
        ax1.set_title(f"{res['name']}: Predicted vs Actual")

        # Residuals
        ax2 = axes[i][1]
        ax2.scatter(preds, residuals, alpha=0.5, edgecolor="k", linewidth=0.3)
        ax2.axhline(0, color="r", linestyle="--", linewidth=1)
        ax2.set_xlabel("Predicted performance score")
        ax2.set_ylabel("Residual (actual - predicted)")
        ax2.set_title(f"{res['name']}: Residual Analysis")

    plt.tight_layout()
    out_path = os.path.join(outdir, "diagnostics.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved diagnostic plots -> {out_path}")


def main():
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=6, random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    plot_diagnostics(results, y_test)

    # Feature importance from the Random Forest model, since that's the
    # one that exposes it directly
    rf_result = next(r for r in results if r["name"] == "Random Forest")
    importances = pd.Series(
        rf_result["model"].feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print("\n--- Random Forest Feature Importances ---")
    print(importances.to_string())


if __name__ == "__main__":
    main()
