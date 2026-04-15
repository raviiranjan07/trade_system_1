# Retrain Pipeline Build Plan

**Goal:** Replace the current ad-hoc training process with a proper MLOps pipeline runnable via a single `dvc repro` command.

**Why:**
- Current `feature_cache.parquet` builder is missing → cannot retrain on new data
- Current training scripts have data leakage (random split across all years, scaler fit on full dataset) → reported OOS metrics are not honest
- Need a reproducible, versioned pipeline before any future retrain

**Outcome:**
- `dvc repro` rebuilds features → labels → trains model with honest split → registers in MLflow → runs comparison backtest
- Human gate before promotion to `@production` (correct for trading $$$)

---

## Phase 1 — Build features stage (the keystone)

**Deliverable:** `src/engine/build_features.py` + DVC stage `build_features`

**Scope:**
- Reads `data/raw/BTCUSDT_15m_ohlcv.parquet`
- Computes 23 columns matching `configs/data_cards/direction_feature_cache.yaml` spec:
  - OHLCV (5): open, high, low, close, volume
  - Direction (6): rsi7, range_position, sma200_dist_pct, roc5, momentum10, rsi
  - Magnitude (8): atr_pct, atr_percentile, ema_separation, range_bps, body_bps, dist_from_high20_pct, hour_utc, volume_ratio
  - Analysis (4): ema20_slope, ema50_dist_pct, sma200_slope, range_position_50
- Writes `data/features/direction_prediction/feature_cache.parquet`

**Validation:**
- Shape: ~210k rows, 23 columns
- No NaN in core indicators after warm-up (first 200 bars expected NaN)
- Distribution sanity check vs current parquet (means/std of each column close)

**Wired as:** DVC stage `build_features` in `dvc.yaml`

**Estimated time:** 2-3 hours

---

## Phase 2 — Honest training stage (fix the leakage)

**Deliverable:** Refactored `src/engine/ml_train.py`

**Scope:**
- **Date-based split** (replaces random 90/10):
  - Train: 2020-2023
  - Val: 2024 (early stopping)
  - Test: 2025 (true OOS, never touched during training)
- **Scaler fit on train only** (replaces fit on full dataset)
- Auto-logs to MLflow run with:
  - Params (architecture, features, hyperparams)
  - Honest OOS metrics (test_accuracy, confident_accuracy, distribution)
  - Tags (git commit, data hash from .dvc, train date)
- Auto-registers as `direction_v15` v2 with `@staging` alias
- Existing v1 stays as `@production` (no swap until human approves)

**Replaces:** Current leaky `ml_train.py` (keeps `retrain_mlp_v15_honest.py` as reference, then delete)

**Wired as:** Updated DVC stage `train_mlp_v15`

**Estimated time:** 1.5 hours

---

## Phase 3 — Backtest comparison stage

**Deliverable:** `scripts/mlops/compare_models.py` + DVC stage `backtest_compare`

**Scope:**
- Loads `direction_v15@staging` and `direction_v15@production` from MLflow registry
- Runs identical OOS backtest on both (2025 data, V3 exit rules)
- Outputs `data/reports/model_comparison.json`:
  ```json
  {
    "production": {"version": 1, "trades": 1767, "win_pct": 53.3, "total_bps": 18207, "pf": 1.5},
    "staging":    {"version": 2, "trades": ?,    "win_pct": ?,    "total_bps": ?,     "pf": ?},
    "diff":       {"trades_pct": ?, "bps_pct": ?, "pf_diff": ?},
    "recommendation": "PROMOTE | KEEP_PRODUCTION | INCONCLUSIVE"
  }
  ```
- Prints clear summary to console for human review

**Wired as:** DVC stage `backtest_compare` (final stage)

**Estimated time:** 1-1.5 hours

---

## Phase 4 — Wire dvc.yaml end-to-end

**Deliverable:** Updated `dvc.yaml` with full 4-stage pipeline

```
build_features    → feature_cache.parquet
build_labels      → labels.parquet
train_mlp_v15     → models/direction_v15/ + MLflow @staging
backtest_compare  → data/reports/model_comparison.json

train_attention   (frozen, Colab — unchanged)
```

**Validation:** Run `dvc repro` end-to-end on a clean clone, confirm all stages execute and outputs match expected.

**Estimated time:** 30 min

---

## Phase 5 — Document the workflow

**Deliverable:** `docs/RETRAIN.md`

**Scope:**
- "How to retrain on new data" 5-step guide
- Manual promotion command for swapping `@staging` → `@production`
- Rollback procedure (alias swap)
- What to check in `model_comparison.json` before promoting

**Estimated time:** 30 min

---

## Total estimate: ~6 hours

---

## Approval gates

After each phase, pause and show:
- What was built
- Smoke test result
- Any deviations from this plan

User approves before next phase begins.

---

## Out of scope (Stage 3 / Stage 4 work — not in this plan)

- Live model monitoring / drift detection
- Auto-promotion (always manual gate here)
- A/B testing / shadow mode
- Feature store
- CI/CD integration of pipeline
- Attention model retraining pipeline (still Colab-only)

These are deliberately deferred to keep scope manageable. Add later when needed.

---

## Risks / known gaps

1. **build_features output may not match existing parquet exactly.** Acceptable — we're building v2 features. Old model (v1) stays in production untouched until human approves v2 promotion.
2. **Honest split may produce worse-looking metrics.** Expected. The leaky metrics were inflated. Honest baseline = real baseline.
3. **Attention model retains its leakage.** Out of scope for this plan; tracked separately.

---

_Plan written: 2026-04-16_
_Status: awaiting approval to start Phase 1_
