# ============================================================
# KPT PREDICTION — BASELINE vs RELIABILITY-WEIGHTED EXPERIMENT
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("orders_with_reliability.csv")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

le = LabelEncoder()
df["merchant_id_encoded"] = le.fit_transform(df["merchant_id"])

FEATURES = ["item_count", "is_peak_hour", "hour_of_day", "day", "merchant_id_encoded"]
TARGET_TRAIN = "observed_prep_duration"
TARGET_EVAL  = "true_prep_duration"
WEIGHT_COL   = "FinalReliabilityScore"

# ============================================================
# TEMPORAL TRAIN / TEST SPLIT
# ============================================================

train_df = df[df["day"] <= 24].copy()
test_df  = df[df["day"] >  24].copy()

X_train        = train_df[FEATURES]
y_train        = train_df[TARGET_TRAIN]
train_weights  = train_df[WEIGHT_COL]

X_test         = test_df[FEATURES]
y_eval         = test_df[TARGET_EVAL]

print("=" * 60)
print("  KPT MODEL EXPERIMENT — SETUP")
print("=" * 60)
print(f"  Train size : {len(train_df):,} orders  (day <= 24)")
print(f"  Test size  : {len(test_df):,} orders  (day  > 24)")
print(f"  Features   : {FEATURES}")
print(f"  Train target : {TARGET_TRAIN}")
print(f"  Eval target  : {TARGET_EVAL}")

# ============================================================
# MODEL 1 — BASELINE (no weights)
# ============================================================

print("\n  Training baseline model ...")
baseline_model = GradientBoostingRegressor(random_state=42)
baseline_model.fit(X_train, y_train)

# ============================================================
# MODEL 2 — RELIABILITY-WEIGHTED
# ============================================================

print("  Training reliability-weighted model ...")
weighted_model = GradientBoostingRegressor(random_state=42)
weighted_model.fit(X_train, y_train, sample_weight=train_weights)

# ============================================================
# EVALUATION vs TRUE PREP DURATION
# ============================================================

baseline_preds = baseline_model.predict(X_test)
weighted_preds = weighted_model.predict(X_test)

# Absolute Errors
baseline_abs_error = np.abs(y_eval - baseline_preds)
weighted_abs_error = np.abs(y_eval - weighted_preds)

# MAE
baseline_mae = mean_absolute_error(y_eval, baseline_preds)
weighted_mae = mean_absolute_error(y_eval, weighted_preds)

# RMSE
baseline_rmse = np.sqrt(mean_squared_error(y_eval, baseline_preds))
weighted_rmse = np.sqrt(mean_squared_error(y_eval, weighted_preds))

# Percentiles
baseline_p50 = np.percentile(baseline_abs_error, 50)
weighted_p50 = np.percentile(weighted_abs_error, 50)

baseline_p90 = np.percentile(baseline_abs_error, 90)
weighted_p90 = np.percentile(weighted_abs_error, 90)

# Improvements
mae_improvement  = ((baseline_mae  - weighted_mae)  / baseline_mae)  * 100
rmse_improvement = ((baseline_rmse - weighted_rmse) / baseline_rmse) * 100
p50_improvement  = ((baseline_p50  - weighted_p50)  / baseline_p50)  * 100
p90_improvement  = ((baseline_p90  - weighted_p90)  / baseline_p90)  * 100

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("  KPT MODEL EXPERIMENT — RESULTS")
print("=" * 70)
print(f"  {'Metric':<30} {'Baseline':>12} {'Weighted':>12} {'Improvement':>12}")
print("  " + "-" * 70)

print(f"  {'MAE':<30} {baseline_mae:>12.4f} {weighted_mae:>12.4f} {mae_improvement:>11.2f}%")
print(f"  {'RMSE':<30} {baseline_rmse:>12.4f} {weighted_rmse:>12.4f} {rmse_improvement:>11.2f}%")
print(f"  {'P50 Absolute Error':<30} {baseline_p50:>12.4f} {weighted_p50:>12.4f} {p50_improvement:>11.2f}%")
print(f"  {'P90 Absolute Error':<30} {baseline_p90:>12.4f} {weighted_p90:>12.4f} {p90_improvement:>11.2f}%")

print("=" * 70)

# Simple interpretation
if mae_improvement > 0:
    print(f"\n  ✅ Reliability weighting reduced overall MAE by {mae_improvement:.2f}%")
else:
    print(f"\n  ⚠️ Reliability weighting increased MAE by {abs(mae_improvement):.2f}%")

if p90_improvement > 0:
    print(f"  ✅ Tail error (P90) reduced by {p90_improvement:.2f}% — improved stability")
else:
    print(f"  ⚠️ Tail error (P90) increased by {abs(p90_improvement):.2f}%")
