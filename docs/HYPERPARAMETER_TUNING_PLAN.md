# ML V3 Hyperparameter Tuning Plan

**Date:** 2026-04-18
**Model:** ML V3 (exit-aware labels + snapshot features)
**Current baseline:** 614 trades, +7,721 bps, PF 2.42 on 2025 test (at default params)

---

## Tuning Rule

**Tune on validation (2024). Confirm on test (2025). Never tune on test.**

```
For each config:
  1. Run backtest on 2024 (val)  → pick best
  2. Run best on 2025 (test)    → confirm it holds
  3. Lock config in params.yaml
```

---

## Level 1: Threshold Sweep (NO retraining, ~24-72 min)

### Parameters:

| Parameter | Current | Sweep values | What it controls |
|-----------|:-------:|-------------|-----------------|
| conf_long | 0.40 | 0.35, 0.37, 0.40, 0.42, 0.45, 0.50 | Min P(LONG) to fire signal |
| conf_short | 0.40 | 0.35, 0.37, 0.40, 0.42, 0.45, 0.50 | Min P(SHORT) to fire signal |

### Implementation:

```
Script: src/engine/sweep_thresholds.py

What it does:
  1. Load trained V3 model (no retraining)
  2. Run inference on ALL val bars (2024) — get probabilities once
  3. For each threshold combo:
     - Filter signals where P(LONG) > threshold or P(SHORT) > threshold
     - Run those signals through backtest (2024 only)
     - Record: trades, bps, PF, stop_rate
  4. Rank by PF (min 100 trades)
  5. Take top 3, confirm each on test (2025)
  6. Pick the one that holds best on 2025

Input:
  - models/ML_V3_staging/v3_model.onnx (existing, no retrain)
  - data/raw/BTCUSDT_15m_ohlcv.parquet
  - data/raw/BTCUSDT_1m_ohlcv.parquet

Output:
  - data/reports/threshold_sweep_results.json
  - Best config → update configs/params.yaml

MLflow:
  - Experiment: "ml_v3_threshold_sweep"
  - One run per threshold combo (12-36 runs)
```

### Sweep strategy:

**Option A: Independent sweep (12 runs, ~24 min)**
```
Fix conf_short=0.40, sweep conf_long: 6 runs
Fix conf_long=best, sweep conf_short: 6 runs
```

**Option B: Grid sweep (36 runs, ~72 min)**
```
All 6 × 6 = 36 combinations
```

### Evaluation metric:
- Primary: Profit Factor on val (2024)
- Secondary: total_bps, stop_rate
- Constraint: must have > 100 trades

---

## Level 2: Training Params (~90 min, requires retraining)

### Parameters:

| Parameter | Current | Sweep values | What it controls |
|-----------|:-------:|-------------|-----------------|
| loss_weight_pnl | 1.0 | 0.5, 1.0, 2.0 | How much LSTM focuses on P&L prediction |
| loss_weight_dir | 1.0 | 0.5, 1.0, 2.0 | How much LSTM focuses on direction prediction |
| learning_rate | 0.001 | 0.0005, 0.001, 0.002 | How fast model learns |
| batch_size | 2048 | 512, 1024, 2048 | How many bars per weight update |

### Why each matters:

**loss_weight_pnl vs loss_weight_dir:**
```
pnl=2.0, dir=0.5 → "Focus on predicting P&L, direction secondary"
                    LSTM learns representation optimized for profit prediction

pnl=0.5, dir=2.0 → "Focus on direction, P&L secondary"
                    LSTM learns representation optimized for classification

pnl=1.0, dir=1.0 → "Try both equally" (current)
                    Might not be optimal for either
```
V2 failed because MFE loss had 4× direction weight — balance matters.

**learning_rate:**
```
0.0005 → learns slowly, might not converge in 100 epochs
0.001  → current, balanced
0.002  → learns fast but might overshoot, unstable
```

**batch_size:**
```
512  → noisy updates, explores more, slower per epoch
1024 �� moderate
2048 → stable updates, converges faster, might miss subtle patterns
```

### Implementation:

```
Script: src/engine/sweep_training_params.py

What it does:
  Step 1 — Loss weight sweep (9 configs):
    1. For each combo of loss_weight_pnl × loss_weight_dir:
       - Train fresh model on 2020-2023 (same labels, same features)
       - Evaluate on val 2024 (accuracy + backtest)
       - Record: val_PF, val_bps, val_stop_rate
    2. Pick best loss weight combo

  Step 2 — LR + batch_size sweep (9 configs):
    1. Fix loss weights from Step 1
    2. For each combo of lr × batch_size:
       - Train fresh model
       - Evaluate on val 2024
    3. Pick best combo

  Step 3 — Confirm:
    1. Train final model with all best params
    2. Run backtest on test 2025
    3. Compare to baseline (PF 2.42)

Input:
  - data/features/direction_prediction/feature_cache.parquet
  - data/features/direction_prediction/exit_aware_labels.parquet
  - data/raw/ (for backtest)

Output:
  - data/reports/training_param_sweep_results.json
  - Best config → update configs/params.yaml

MLflow:
  - Experiment: "ml_v3_training_sweep"
  - 18 runs total (9 + 9)
  - Each logs: params, val_accuracy, val_PF, val_bps

Runtime: 18 × 5 min = ~90 min
```

---

## Level 3: Architecture Params (~120 min, only if needed)

### Parameters:

| Parameter | Current | Sweep values | What it controls |
|-----------|:-------:|-------------|-----------------|
| hidden_size | 128 | 64, 128, 180, 256 | LSTM capacity — how many patterns it can learn |
| temperature | 0.5 | 0.1, 0.3, 0.5, 0.7, 1.0 | Attention sharpness — focus on few steps vs blend all |
| dropout | 0.5 | 0.3, 0.4, 0.5, 0.6 | Regularization — prevents overfitting |

### Why each matters:

**hidden_size:**
```
64  → small model, fast, might underfit (can't learn enough patterns)
128 → current — balanced
180 → larger, can learn more complex patterns
256 → risk overfitting on 140K training bars
```

**temperature:**
```
0.1 → almost hard attention — focuses on 1-2 steps, ignores rest
0.3 → sharp focus on important steps
0.5 → current — moderate
0.7 → more blended
1.0 → uniform — all steps weighted equally (like simple average)
```

**dropout:**
```
0.3 → less regularization — model memorizes more (risk overfit)
0.5 → current — balanced
0.6 → more regularization — forces robustness (risk underfit)
```

### Implementation:

```
Script: src/engine/sweep_architecture.py

What it does:
  Step 1 — Hidden + Temperature sweep (20 configs):
    1. Fix best from Level 1 + 2
    2. For each combo of hidden_size × temperature:
       - Train fresh model
       - Evaluate on val 2024 (accuracy + backtest)
    3. Pick best combo

  Step 2 — Dropout sweep (4 configs):
    1. Fix best hidden + temp from Step 1
    2. For each dropout value:
       - Train fresh model
       - Evaluate on val 2024
    3. Pick best

  Step 3 — Confirm on test 2025

Input: same as Level 2

Output:
  - data/reports/architecture_sweep_results.json
  - Best config → update configs/params.yaml

MLflow:
  - Experiment: "ml_v3_architecture_sweep"
  - 24 runs total (20 + 4)

Runtime: 24 × 5 min = ~120 min
```

---

## Final Step: Retrain on ALL Data

After all 3 levels complete:

```
Script: src/engine/train_v3_final.py

What it does:
  1. Load locked config from params.yaml (all best values from Level 1-3)
  2. Train on FULL dataset (2020-2025) — no val/test split
  3. Export to models/ML_V3/
  4. Register in MLflow @production
  5. Deploy to EC2

No testing possible (all data used for training).
Live paper trading = the test.
Monitor via Expected vs Actual dashboard.
```

---

## Execution Order

```
Phase 1: Level 1 — Threshold sweep (~24 min)
  → Lock best thresholds in params.yaml
  → Confirm on 2025 test

Phase 2: Level 2 — Training params (~90 min)
  → Lock best loss weights, lr, batch_size
  → Retrain with best config
  → Confirm on 2025 test

Phase 3: Level 3 — Architecture (~120 min, only if needed)
  → Lock best hidden, temp, dropout
  → Retrain with best config
  → Confirm on 2025 test

Phase 4: Final retrain on ALL data (2020-2025)
  → Use locked config from Phase 1-3
  → Deploy to production
```

---

## DVC Pipeline Stages

```yaml
  sweep_thresholds:
    cmd: python -m engine.sweep_thresholds
    deps:
      - models/ML_V3_staging/v3_model.onnx
      - data/raw/BTCUSDT_15m_ohlcv.parquet
      - data/raw/BTCUSDT_1m_ohlcv.parquet
      - src/engine/sweep_thresholds.py
    outs:
      - data/reports/threshold_sweep_results.json

  sweep_training:
    cmd: python -m engine.sweep_training_params
    deps:
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/reports/threshold_sweep_results.json
      - src/engine/sweep_training_params.py
    outs:
      - data/reports/training_sweep_results.json

  sweep_architecture:
    cmd: python -m engine.sweep_architecture
    deps:
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/reports/training_sweep_results.json
      - src/engine/sweep_architecture.py
    outs:
      - data/reports/architecture_sweep_results.json

  train_v3_final:
    cmd: python -m engine.train_v3_final
    deps:
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/reports/architecture_sweep_results.json
      - src/engine/train_v3_final.py
    outs:
      - models/ML_V3/
```

---

## Scripts to Create

| Script | Level | Configs tested | Runtime |
|--------|:-----:|:--------------:|:-------:|
| src/engine/sweep_thresholds.py | 1 | 12-36 | ~24-72 min |
| src/engine/sweep_training_params.py | 2 | 18 | ~90 min |
| src/engine/sweep_architecture.py | 3 | 24 | ~120 min |
| src/engine/train_v3_final.py | Final | 1 (best) | ~5 min |

---

## Baseline to Beat

All metrics on 2025 test (ML V3 signals only):

| Metric | Value |
|--------|------:|
| Trades | 614 |
| Win % | 48.5% |
| Net bps | +7,721 |
| PF | 2.42 |
| Stop % | 48.2% |
| Max DD | -258 |
| Avg/trade | +12.6 |

Any tuned config must beat this on PF AND maintain > 200 trades.
