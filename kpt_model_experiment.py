# ============================================================
# KPT PREDICTION — BASELINE vs RELIABILITY-WEIGHTED EXPERIMENT
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("/mnt/user-data/uploads/orders_with_reliability.csv")

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

baseline_mae  = mean_absolute_error(y_eval, baseline_preds)
weighted_mae  = mean_absolute_error(y_eval, weighted_preds)

baseline_rmse = root_mean_squared_error(y_eval, baseline_preds)
weighted_rmse = root_mean_squared_error(y_eval, weighted_preds)

mae_improvement  = ((baseline_mae  - weighted_mae)  / baseline_mae)  * 100
rmse_improvement = ((baseline_rmse - weighted_rmse) / baseline_rmse) * 100

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("  KPT MODEL EXPERIMENT — RESULTS")
print("=" * 60)
print(f"  {'Metric':<30} {'Baseline':>10}  {'Weighted':>10}")
print("  " + "-" * 54)
print(f"  {'MAE  (vs true_prep_duration)':<30} {baseline_mae:>10.4f}  {weighted_mae:>10.4f}")
print(f"  {'RMSE (vs true_prep_duration)':<30} {baseline_rmse:>10.4f}  {weighted_rmse:>10.4f}")
print("  " + "-" * 54)
print(f"  {'MAE  Improvement (%)':<30} {mae_improvement:>+10.2f}%")
print(f"  {'RMSE Improvement (%)':<30} {rmse_improvement:>+10.2f}%")
print("=" * 60)

if mae_improvement > 0:
    print(f"\n  ✅ Reliability weighting reduced MAE by {mae_improvement:.2f}%")
else:
    print(f"\n  ⚠️  Reliability weighting increased MAE by {abs(mae_improvement):.2f}%")
