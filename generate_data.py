"""
generate_data.py

Generates a synthetic dataset simulating daily workout, sleep, nutrition,
and recovery metrics for a single athlete over time. Used as sample data
so the pipeline in train_model.py has something realistic to run against.

In a real deployment, this would be replaced with logs pulled from a
fitness tracker, a training app export, or manually logged data.
"""

import numpy as np
import pandas as pd


def generate_workout_data(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    # Base daily signals with some autocorrelation (today depends partly on yesterday)
    sleep_hours = np.clip(rng.normal(7, 1.1, n_days), 3, 10)
    training_load = np.clip(rng.normal(60, 20, n_days), 0, 150)  # arbitrary load units
    protein_g = np.clip(rng.normal(140, 30, n_days), 40, 260)
    calories = np.clip(rng.normal(2600, 400, n_days), 1500, 4200)
    resting_hr = np.clip(rng.normal(58, 5, n_days), 42, 80)
    soreness = np.clip(rng.normal(4, 2, n_days), 0, 10)  # self-reported 0-10

    # Macro adherence: how close intake was to a target (protein target = 150g)
    macro_adherence = 1 - (np.abs(protein_g - 150) / 150)

    # Rolling features (7-day windows) computed after generation, see below
    df = pd.DataFrame({
        "date": dates,
        "sleep_hours": sleep_hours,
        "training_load": training_load,
        "protein_g": protein_g,
        "calories": calories,
        "resting_hr": resting_hr,
        "soreness": soreness,
        "macro_adherence": macro_adherence,
    })

    # Rolling averages / trend deltas — same feature engineering style
    # described in the resume (rolling averages, trend deltas, macro adherence)
    df["sleep_7d_avg"] = df["sleep_hours"].rolling(7, min_periods=1).mean()
    df["load_7d_avg"] = df["training_load"].rolling(7, min_periods=1).mean()
    df["load_trend_delta"] = df["training_load"] - df["load_7d_avg"]
    df["resting_hr_7d_avg"] = df["resting_hr"].rolling(7, min_periods=1).mean()
    df["resting_hr_delta"] = df["resting_hr"] - df["resting_hr_7d_avg"]

    # Synthetic "next-session performance score" (0-100) as the prediction target.
    # Built from a weighted combination of recovery-related inputs plus noise,
    # so the model has real signal to find rather than pure randomness.
    performance = (
        50
        + 3.0 * (df["sleep_7d_avg"] - 7)
        - 0.15 * df["load_trend_delta"]
        + 8.0 * (df["macro_adherence"] - 0.5)
        - 1.2 * (df["resting_hr_delta"])
        - 0.8 * (df["soreness"] - 4)
        + rng.normal(0, 4, n_days)
    )
    df["next_session_performance"] = np.clip(performance, 0, 100)

    return df


if __name__ == "__main__":
    data = generate_workout_data()
    data.to_csv("workout_data.csv", index=False)
    print(f"Generated {len(data)} rows -> workout_data.csv")
    print(data.head())
