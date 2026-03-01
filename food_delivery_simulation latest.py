# ============================================================
# SECTION 1 — Imports & Seed
# ============================================================

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)


# ============================================================
# SECTION 2 — Merchant Table
# ============================================================

N_MERCHANTS = 1000

# Merchant type distribution
type_probs = {"A": 0.40, "B": 0.15, "C": 0.15, "D": 0.10, "E": 0.10, "F": 0.10}
merchant_types = np.random.choice(
    list(type_probs.keys()),
    size=N_MERCHANTS,
    p=list(type_probs.values()),
    replace=True,
)

# Base attributes
base_prep_time = np.random.uniform(8, 18, size=N_MERCHANTS)
peak_factor = np.random.uniform(1.5, 3.0, size=N_MERCHANTS)

# Heavy-tailed volume weights using log-normal distribution
raw_volume = np.random.lognormal(mean=0, sigma=1.2, size=N_MERCHANTS)

# Force Type F merchants to have very small volume weight
f_mask = merchant_types == "F"
raw_volume[f_mask] *= 0.05  # significantly reduce Type F volume

# Normalize to sum to 1
volume_weight = raw_volume / raw_volume.sum()

# Behavior subtype for all merchants; Type F gets randomly assigned from A/B/D/E
behavior_subtypes = merchant_types.copy()
behavior_subtypes[f_mask] = np.random.choice(["A", "B", "D", "E"], size=f_mask.sum())

merchants = pd.DataFrame(
    {
        "merchant_id": np.arange(1, N_MERCHANTS + 1),
        "merchant_type": merchant_types,
        "base_prep_time": base_prep_time,
        "peak_factor": peak_factor,
        "volume_weight": volume_weight,
        "behavior_subtype": behavior_subtypes,
    }
)

print("=== Merchant Type Counts ===")
print(merchants["merchant_type"].value_counts().sort_index())
print("\n=== Top 10 Merchants by Volume Weight ===")
print(
    merchants.nlargest(10, "volume_weight")[
        ["merchant_id", "merchant_type", "volume_weight", "behavior_subtype"]
    ].to_string(index=False)
)

# Sanity check: top 10% merchant order share
top10_pct_idx = merchants.nlargest(int(N_MERCHANTS * 0.1), "volume_weight").index
top10_share = merchants.loc[top10_pct_idx, "volume_weight"].sum()
print(f"\nTop 10% merchants capture {top10_share:.1%} of order volume")


# ============================================================
# SECTION 3 — Order Generation
# ============================================================

N_ORDERS = 100_000
PEAK_HOURS = {12, 13, 14, 19, 20, 21, 22}

# Sample merchant IDs using volume weights
sampled_merchant_ids = np.random.choice(
    merchants["merchant_id"].values,
    size=N_ORDERS,
    p=merchants["volume_weight"].values,
    replace=True,
)

day = np.random.randint(1, 31, size=N_ORDERS)          # 1–30
hour_of_day = np.random.randint(0, 24, size=N_ORDERS)  # 0–23
is_peak_hour = np.isin(hour_of_day, list(PEAK_HOURS))
item_count = np.random.randint(1, 9, size=N_ORDERS)    # 1–8

random_minute = np.random.uniform(0, 60, size=N_ORDERS)
order_confirm_time = (day - 1) * 1440 + hour_of_day * 60 + random_minute

orders = pd.DataFrame(
    {
        "order_id": np.arange(1, N_ORDERS + 1),
        "merchant_id": sampled_merchant_ids,
        "day": day,
        "hour_of_day": hour_of_day,
        "is_peak_hour": is_peak_hour,
        "item_count": item_count,
        "order_confirm_time": order_confirm_time,
    }
)

# Merge merchant attributes into orders
orders = orders.merge(
    merchants[["merchant_id", "merchant_type", "base_prep_time", "peak_factor", "behavior_subtype"]],
    on="merchant_id",
    how="left",
)


# ============================================================
# SECTION 4 — True Prep Duration
# ============================================================

noise_prep = np.random.normal(0, 1, size=N_ORDERS)
peak_addition = np.where(orders["is_peak_hour"], orders["peak_factor"], 0.0)

true_prep_duration = (
    orders["base_prep_time"]
    + 2 * orders["item_count"]
    + peak_addition
    + noise_prep
)

# Ensure all durations are positive
true_prep_duration = np.maximum(true_prep_duration, 0.5)

orders["true_prep_duration"] = true_prep_duration
orders["true_prep_ready_time"] = orders["order_confirm_time"] + orders["true_prep_duration"]


# ============================================================
# SECTION 5 — Dispatch & Rider Arrival
# ============================================================

noise_kpt = np.random.normal(0, 1, size=N_ORDERS)
predicted_kpt = orders["true_prep_duration"] + noise_kpt
predicted_kpt = np.maximum(predicted_kpt, 0.5)

orders["predicted_kpt"] = predicted_kpt

dispatch_buffer = np.random.uniform(3, 7, size=N_ORDERS)
dispatch_time_raw = orders["order_confirm_time"] + orders["predicted_kpt"] - dispatch_buffer
orders["dispatch_time"] = np.maximum(dispatch_time_raw, orders["order_confirm_time"])

noise_rider = np.random.normal(0, 2.5, size=N_ORDERS)
orders["rider_arrival_time"] = orders["dispatch_time"] + noise_rider


# ============================================================
# SECTION 6 — Observed Label (Merchant Marked Ready Time)
# ============================================================

def compute_observed_label(row_slice, mtype_col, subtype_col,
                            true_ready_col, rider_col,
                            is_peak_col):
    """Vectorized label corruption per merchant type."""

    n = len(row_slice)
    mtype = row_slice[mtype_col].values
    subtype = row_slice[subtype_col].values
    true_ready = row_slice[true_ready_col].values
    rider = row_slice[rider_col].values
    is_peak = row_slice[is_peak_col].values

    observed = np.empty(n)
    observed[:] = np.nan

    # Type A — clean label
    mask_A = mtype == "A"
    observed[mask_A] = true_ready[mask_A]

    # Type B — systematic late marking
    mask_B = mtype == "B"
    observed[mask_B] = true_ready[mask_B] + np.random.uniform(2, 3, size=mask_B.sum())

    # Type C — marks ready only when rider arrives
    mask_C = mtype == "C"
    observed[mask_C] = rider[mask_C]

    # Type D — noisy during peak, clean otherwise
    mask_D = mtype == "D"
    mask_D_peak = mask_D & is_peak
    mask_D_off = mask_D & ~is_peak
    observed[mask_D_peak] = true_ready[mask_D_peak] + np.random.normal(0, 2, size=mask_D_peak.sum())
    observed[mask_D_off] = true_ready[mask_D_off]

    # Type E — noisy label
    mask_E = mtype == "E"
    observed[mask_E] = true_ready[mask_E] + np.random.normal(0, 3, size=mask_E.sum())

    # Type F — delegate to behavior_subtype logic
    mask_F = mtype == "F"
    if mask_F.any():
        sub = subtype[mask_F]
        tr_f = true_ready[mask_F]
        ri_f = rider[mask_F]
        ip_f = is_peak[mask_F]
        obs_f = np.empty(mask_F.sum())

        sub_A = sub == "A"
        obs_f[sub_A] = tr_f[sub_A]

        sub_B = sub == "B"
        obs_f[sub_B] = tr_f[sub_B] + np.random.uniform(2, 3, size=sub_B.sum())

        sub_D = sub == "D"
        sub_D_peak = sub_D & ip_f
        sub_D_off  = sub_D & ~ip_f
        obs_f[sub_D_peak] = tr_f[sub_D_peak] + np.random.normal(0, 2, size=sub_D_peak.sum())
        obs_f[sub_D_off]  = tr_f[sub_D_off]

        sub_E = sub == "E"
        obs_f[sub_E] = tr_f[sub_E] + np.random.normal(0, 3, size=sub_E.sum())

        observed[mask_F] = obs_f

    return observed


orders["observed_label"] = compute_observed_label(
    row_slice=orders,
    mtype_col="merchant_type",
    subtype_col="behavior_subtype",
    true_ready_col="true_prep_ready_time",
    rider_col="rider_arrival_time",
    is_peak_col="is_peak_hour",
)

# Derived observed prep duration (for ML targets)
orders["observed_prep_duration"] = orders["observed_label"] - orders["order_confirm_time"]


# ============================================================
# SECTION 7 — Final Checks
# ============================================================

print("\n=== orders.head() ===")
print(orders.head().to_string())

print("\n=== Null Value Check ===")
print(orders.isnull().sum())

print("\n=== True Prep Duration Summary Stats ===")
print(orders["true_prep_duration"].describe())

print("\n=== Distribution of (observed_label - rider_arrival_time) ===")
diff = orders["observed_label"] - orders["rider_arrival_time"]
print(diff.describe())
print(f"  Skewness : {diff.skew():.4f}")
print(f"  Kurtosis : {diff.kurt():.4f}")
print(f"  % positive (label after rider): {(diff > 0).mean():.2%}")


# ============================================================
# SECTION 8 — Save Files
# ============================================================

# --- Merchants ---
merchants_out = merchants.copy()
merchants_out["behavior_subtype"] = merchants_out["behavior_subtype"].fillna("N/A")
merchants_out = merchants_out.astype(
    {
        "merchant_id": int,
        "base_prep_time": float,
        "peak_factor": float,
        "volume_weight": float,
    }
)

# --- Orders ---
orders_out = orders.copy()
orders_out["behavior_subtype"] = orders_out["behavior_subtype"].fillna("N/A")
orders_out["is_peak_hour"] = orders_out["is_peak_hour"].astype(int)  # bool → 0/1 for sklearn
orders_out = orders_out.astype(
    {
        "order_id": int,
        "merchant_id": int,
        "day": int,
        "hour_of_day": int,
        "item_count": int,
        "order_confirm_time": float,
        "true_prep_duration": float,
        "true_prep_ready_time": float,
        "predicted_kpt": float,
        "dispatch_time": float,
        "rider_arrival_time": float,
        "observed_label": float,
        "observed_prep_duration": float,
    }
)

# Final null assertion
assert merchants_out.isnull().sum().sum() == 0, "merchants has nulls!"
assert orders_out.isnull().sum().sum() == 0, "orders has nulls!"

merchants_out.to_csv("/mnt/user-data/outputs/merchants.csv", index=False)
orders_out.to_csv("/mnt/user-data/outputs/orders.csv", index=False)

print("\n✅ merchants.csv saved — shape:", merchants_out.shape)
print("✅ orders.csv saved    — shape:", orders_out.shape)
print("\nColumn list (orders):")
print(orders_out.dtypes.to_string())
