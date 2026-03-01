# ============================================================
# MERCHANT RELIABILITY SCORING ENGINE
# KPT Label Governance — Production Style
# ============================================================

import numpy as np
import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

orders    = pd.read_csv("/mnt/user-data/uploads/orders_latest.csv")
merchants = pd.read_csv("/mnt/user-data/uploads/merchants_latest.csv")

# ============================================================
# STEP 1 — Duration Features
# ============================================================

orders["rider_arrival_duration"] = (
    orders["rider_arrival_time"] - orders["order_confirm_time"]
)

orders["observed_prep_duration"] = (
    orders["observed_label"] - orders["order_confirm_time"]
)

orders["residual"] = (
    orders["observed_prep_duration"] - orders["rider_arrival_duration"]
)

# ============================================================
# STEP 2 — Merchant-Level Aggregations
# ============================================================

# --- std of residual (coupling signal) ---
agg_std_residual = (
    orders.groupby("merchant_id")["residual"]
    .std()
    .rename("std_residual")
)

# --- correlation between item_count and observed_prep_duration ---
agg_corr_complexity = (
    orders.groupby("merchant_id")
    .apply(
        lambda g: g["item_count"].corr(g["observed_prep_duration"]),
        include_groups=False
    )
    .rename("corr_complexity")
)

# --- std of observed_prep_duration ---
agg_std_observed = (
    orders.groupby("merchant_id")["observed_prep_duration"]
    .std()
    .rename("std_observed")
)

# --- peak / off-peak mean observed_prep_duration ---
agg_mean_peak = (
    orders[orders["is_peak_hour"] == 1]
    .groupby("merchant_id")["observed_prep_duration"]
    .mean()
    .rename("mean_peak")
)

agg_mean_off_peak = (
    orders[orders["is_peak_hour"] == 0]
    .groupby("merchant_id")["observed_prep_duration"]
    .mean()
    .rename("mean_off_peak")
)

# --- order count ---
agg_order_count = (
    orders.groupby("merchant_id")["order_id"]
    .count()
    .rename("order_count")
)

# --- Combine all aggregations ---
merchant_stats = pd.concat(
    [agg_std_residual, agg_corr_complexity, agg_std_observed,
     agg_mean_peak, agg_mean_off_peak, agg_order_count],
    axis=1
).reset_index()

# Fill merchants that have no peak or off-peak orders with column mean
merchant_stats["mean_peak"]     = merchant_stats["mean_peak"].fillna(merchant_stats["mean_peak"].mean())
merchant_stats["mean_off_peak"] = merchant_stats["mean_off_peak"].fillna(merchant_stats["mean_off_peak"].mean())

merchant_stats["peak_diff"] = (
    merchant_stats["mean_peak"] - merchant_stats["mean_off_peak"]
).abs()

# ============================================================
# STEP 3 — Behavioral Scores (bounded 0–1, smooth scaling)
# ============================================================

# CouplingScore: high std_residual → rider coupling → less reliable
merchant_stats["CouplingScore"] = (
    merchant_stats["std_residual"] /
    (merchant_stats["std_residual"] + 1)
)

# ComplexityScore: how well item_count explains observed prep (floor at 0)
merchant_stats["ComplexityScore"] = merchant_stats["corr_complexity"].clip(lower=0)

# VarianceScore: high variance in observed prep → noisy labels
merchant_stats["VarianceScore"] = (
    merchant_stats["std_observed"] /
    (merchant_stats["std_observed"] + 5)
)

# PeakScore: merchants sensitive to peak hours show expected behavioral shift
merchant_stats["PeakScore"] = (
    merchant_stats["peak_diff"] /
    (merchant_stats["peak_diff"] + 3)
)

# Handle any residual NaNs in behavioral scores before weighting
for col in ["CouplingScore", "ComplexityScore", "VarianceScore", "PeakScore"]:
    merchant_stats[col] = merchant_stats[col].fillna(0).clip(0, 1)

# ============================================================
# STEP 4 — BehaviorScore (weighted linear combination)
# ============================================================

merchant_stats["BehaviorScore"] = (
    0.40 * merchant_stats["CouplingScore"]
    + 0.20 * merchant_stats["ComplexityScore"]
    + 0.20 * merchant_stats["VarianceScore"]
    + 0.20 * merchant_stats["PeakScore"]
).clip(0, 1)

# ============================================================
# STEP 5 — Volume Confidence
# ============================================================

log_counts     = np.log(merchant_stats["order_count"])
log_max        = np.log(merchant_stats["order_count"].max())

merchant_stats["VolumeConfidence"] = (log_counts / log_max).clip(0, 1)

# ============================================================
# STEP 6 — FinalReliabilityScore
# ============================================================

merchant_stats["FinalReliabilityScore"] = (
    merchant_stats["BehaviorScore"] * merchant_stats["VolumeConfidence"]
).clip(0, 1)

# ============================================================
# STEP 7 — Merge Back into Orders
# ============================================================

orders = orders.merge(
    merchant_stats[["merchant_id", "FinalReliabilityScore"]],
    on="merchant_id",
    how="left"
)

# ============================================================
# VALIDATION
# ============================================================

assert orders["FinalReliabilityScore"].isnull().sum() == 0, \
    "Null FinalReliabilityScore values detected in orders!"

assert orders["FinalReliabilityScore"].between(0, 1).all(), \
    "FinalReliabilityScore out of [0, 1] bounds!"

# Enforce numeric dtypes
orders["FinalReliabilityScore"] = orders["FinalReliabilityScore"].astype(float)

# ============================================================
# STEP 8 — Save Output
# ============================================================

orders.to_csv("/mnt/user-data/outputs/orders_with_reliability.csv", index=False)

# ============================================================
# PRINT DIAGNOSTICS
# ============================================================

# Merge merchant_type into merchant_stats for display
merchant_stats = merchant_stats.merge(
    merchants[["merchant_id", "merchant_type"]],
    on="merchant_id",
    how="left"
)

display_cols = ["merchant_id", "merchant_type", "order_count",
                "BehaviorScore", "VolumeConfidence", "FinalReliabilityScore"]

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)

print("=" * 80)
print("  MERCHANT RELIABILITY SCORING ENGINE — OUTPUT REPORT")
print("=" * 80)

print("\n--- Top 10 Merchants by FinalReliabilityScore ---")
print(
    merchant_stats.nlargest(10, "FinalReliabilityScore")[display_cols]
    .to_string(index=False)
)

print("\n--- Bottom 10 Merchants by FinalReliabilityScore ---")
print(
    merchant_stats.nsmallest(10, "FinalReliabilityScore")[display_cols]
    .to_string(index=False)
)

print("\n--- FinalReliabilityScore Summary Statistics ---")
print(merchant_stats["FinalReliabilityScore"].describe())

print("\n--- Score Distribution by Merchant Type ---")
print(
    merchant_stats.groupby("merchant_type")["FinalReliabilityScore"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(4)
)

print(f"\n✅ orders_with_reliability.csv saved — shape: {orders.shape}")
print(f"   Columns: {list(orders.columns)}")
print(f"   Null values: {orders.isnull().sum().sum()}")
