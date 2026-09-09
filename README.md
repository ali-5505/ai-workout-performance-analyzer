# AI Workout Performance Analyzer

A machine learning pipeline that predicts next-session workout performance
using training load, sleep, nutrition (macro adherence), and recovery
metrics (resting heart rate, soreness).

## What it does

- Engineers features from raw daily logs: 7-day rolling averages, trend
  deltas (today vs. recent average), and macro adherence relative to a
  protein target.
- Trains and compares two models: **Ridge Regression** and
  **Random Forest Regressor**.
- Evaluates both with a held-out test set (MAE, R²) and 5-fold
  cross-validation.
- Produces diagnostic plots (predicted vs. actual, residual analysis) to
  sanity-check model behavior rather than trusting a single metric.
- Surfaces Random Forest feature importances to show which inputs actually
  drive next-session performance.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib.

## Project structure

```
ai-workout-performance-analyzer/
├── generate_data.py   # Synthetic sample dataset generator
├── train_model.py     # Feature engineering, training, evaluation, plots
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python train_model.py
```

This generates a synthetic `workout_data.csv` on first run (or uses one
you supply, in the same format), trains both models, prints evaluation
metrics to the console, and saves diagnostic plots to `plots/diagnostics.png`.

## Sample output

```
--- Ridge Regression ---
Test MAE:        3.49
Test R^2:        0.673
5-fold CV MAE:   3.24 (+/- 0.21)

--- Random Forest ---
Test MAE:        4.32
Test R^2:        0.523
5-fold CV MAE:   3.54 (+/- 0.32)
```

## Notes

The included dataset is synthetically generated to have realistic
correlations (e.g. poor sleep and elevated resting heart rate reduce
predicted performance) so the pipeline has real signal to learn from.
Swap in real training-log data with the same column structure to use
this on an actual dataset.
