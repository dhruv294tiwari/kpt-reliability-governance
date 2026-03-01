
# KPT Reliability Governance Framework

## Overview

This repository presents a signal-level reliability governance framework designed to improve Kitchen Prep Time (KPT) prediction accuracy in large-scale food delivery systems.

Instead of modifying the predictive model itself, this approach focuses on improving the reliability of training labels derived from merchant-marked “Food Ready” (FOR) signals. By detecting and down-weighting unreliable behavioral patterns, we isolate the impact of label quality on prediction performance.

In a controlled synthetic marketplace simulation (1000 merchants, 100,000 orders, 30 days), the reliability-weighted model achieved:

- **3.38% reduction in MAE**
- **2.81% reduction in RMSE**

using the same model architecture and feature set as baseline.

---

## Problem Context

In large-scale delivery ecosystems, Kitchen Prep Time (KPT) prediction depends heavily on merchant-marked readiness signals. However, these signals often suffer from:

- Rider-coupled marking behavior
- Manual operational bias
- Hidden kitchen rush (non-platform orders)
- Order complexity variability

Model improvements alone cannot correct corrupted labels. Therefore, this framework introduces a reliability scoring layer that governs how much trust each merchant’s historical signals should receive during training.

---

## Solution Architecture

The system introduces a **Merchant Reliability Scoring Engine** that:

1. Detects rider-coupled label behavior using duration residual analysis
2. Measures prep-time sensitivity to order complexity
3. Evaluates variance health of prep durations
4. Captures peak-hour behavioral consistency
5. Adjusts confidence based on merchant volume

The resulting **FinalReliabilityScore (0–1)** is applied as `sample_weight` during model training.

The predictive model itself remains unchanged to isolate the impact of label reliability governance.

---

## Repository Structure
