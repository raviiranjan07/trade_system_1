# ML V3 Model Plan — Exit-Aware Direction Prediction

**Date:** 2026-04-17
**Status:** Planning
**Predecessor:** ML V2 Attention (LSTM+Attention, 32 diff features, H8 first-hit labels)

---

## Problem Statement

ML V2 has 50% stop rate because:
1. Labels say "LONG" for bars where price eventually hits +15 bps, even if STOP_LOSS fires at -10 first
2. MFE auxiliary heads learn volatility (not direction), wasting LSTM capacity
3. Model has no concept of SKIP — forced to choose LONG or SHORT, can't say "don't trade"

See: [docs/FLAWS.md](FLAWS.md) for full diagnosis.

---

## Core Idea

Replace the training label with **simulated trade outcomes** using our actual exit rules (PT_TARGET, PT_LOCK, MID_TRAIL, STOP_LOSS, TIME_EXIT) on 1m tick data.

The model learns: **"is this bar worth trading?"** instead of **"which direction moves 15 bps?"**

---

## Architecture

```
INPUT
─────
8 steps × 4 features (same diff features as V2)

Step 1: [roc_diff_1, rsi_diff_1, rp_diff_1, sma200_diff_1]
Step 2: [roc_diff_2, rsi_diff_2, rp_diff_2, sma200_diff_2]
...
Step 8: [roc_diff_8, rsi_diff_8, rp_diff_8, sma200_diff_8]


LAYER 1: LSTM
─────────────
Same cell runs 8 times. Carries memory from step to step.
At each step: 4 input features + previous hidden state → new hidden state.

LSTM(input_size=4, hidden_size=128, num_layers=1)

Step 1: [4 features] + [no memory]  → hidden_1 (128)
Step 2: [4 features] + [hidden_1]   → hidden_2 (128)
...
Step 8: [4 features] + [hidden_7]   → hidden_8 (128)

Result: 8 hidden states, each 128 numbers.
Hidden size 128 is a hyperparameter (can test 64, 128, 180, 256).


LAYER 2: ATTENTION
──────────────────
Score each hidden state to determine importance.

Linear(128 → 1) applied to each hidden state → 8 raw scores
Softmax(scores / temperature) → 8 weights that sum to 1

temperature = 0.5 (controls sharpness: lower = more focused)

Weighted sum of hidden states:
attended = w1 × hidden_1 + w2 × hidden_2 + ... + w8 × hidden_8
         = 128 numbers (single market summary)

Dropout(0.5)


LAYER 3: TRADE PnL HEADS (replaces MFE heads)
──────────────────────────────────────────────
Two linear layers predict actual trade profit for each direction.

attended (128) → Linear(128 → 1) → pred_long_pnl
attended (128) → Linear(128 → 1) → pred_short_pnl

These heads predict: "if you entered LONG/SHORT here with our exit rules,
how many net bps would the trade make?"

Unlike MFE (which predicted volatility), these predict DIRECTIONAL trade
outcomes — one positive, one negative — forcing the LSTM to learn
directional patterns.


LAYER 4: DIRECTION HEAD (expanded to 3-class)
──────────────────────────────────────────────
Receives the market summary + both PnL predictions.

concat[attended(128) + pred_long_pnl(1) + pred_short_pnl(1)] = 130
Linear(130 → 3) → softmax → [P(LONG), P(SHORT), P(SKIP)]


OUTPUT / SIGNAL GENERATION
──────────────────────────
If P(LONG) > conf_long threshold → fire ML_V3_LONG
If P(SHORT) > conf_short threshold → fire ML_V3_SHORT
Otherwise → no signal (SKIP)
```

---

## Labels (Exit-Aware)

For every 15m bar in the dataset, simulate TWO trades using 1m tick data
and the actual V2 exit rules (PT_TARGET, PT_LOCK, MID_TRAIL, STOP_LOSS,
TIME_EXIT/NO_ZONE at bar 6):

1. Enter LONG at next bar's open → run exit rules on 1m ticks → record net_profit_bps
2. Enter SHORT at next bar's open → run exit rules on 1m ticks → record net_profit_bps

### Label columns:
```
long_net_bps:   actual P&L if entered LONG (after 8 bps fees)
short_net_bps:  actual P&L if entered SHORT (after 8 bps fees)
direction:      derived from above (see logic below)
```

### Direction logic:
| long_net_bps | short_net_bps | Label | Why |
|:---:|:---:|:---:|---|
| +52.0 | -18.0 | LONG | LONG profitable, SHORT stopped |
| -18.0 | +21.5 | SHORT | SHORT profitable, LONG stopped |
| -18.0 | -4.8 | SKIP | Both lose — no good entry |
| +21.5 | +8.3 | LONG | Both win, LONG wins more |
| +8.3 | +21.5 | SHORT | Both win, SHORT wins more |

### Key difference from V2 labels:
```
V2 label:  "does +15 hit before -15 within 8 bars?"
           → LONG even if STOP_LOSS fires at -10 on the way to +15
           → fixed 8-bar horizon
           → binary (LONG or SHORT only)

V3 label:  "simulate actual trade with real exit rules on 1m ticks"
           → SKIP if STOP_LOSS fires (correctly labeled as bad entry)
           → dynamic horizon (trade ends when exit rule fires: bar 0 to bar 6)
           → 3-class (LONG, SHORT, or SKIP)
```

---

## Loss Function

```
loss = MSE(pred_long_pnl, actual_long_pnl)
     + MSE(pred_short_pnl, actual_short_pnl)
     + CrossEntropy(direction_pred, direction_label)
```

Loss weights: 1.0 : 1.0 : 1.0 (equal, unlike V2 which had 4:1 MFE:direction imbalance).
Can tune if needed, but start equal.

---

## Exit Rules Used for Label Generation

V2 exit rules (V1 minus LOCKED_PROFIT):
- PT_TARGET: arm at peak >= 60 bps by bar 5, exit at tick >= 80 bps
- PT_LOCK: arm at peak >= 60 bps by bar 5, exit at tick <= 60 bps (idealized)
- MID_TRAIL: arm at peak >= 25 bps (not PT-armed), trail 10 bps width (idealized)
- STOP_LOSS: exit at tick <= -10 bps (idealized)
- TIME_EXIT: bar 6 if pnl < 0
- NO_ZONE: bar 6 if pnl >= 0
- Fees: 8 bps round-trip

Source: src/engine/position_manager.py (V12PositionManager, exit_version="v2")

---

## Features (unchanged from V2)

32 diff features: 4 channels × 8 lookback steps

| Channel | Lookback n | Formula |
|---------|:---:|---|
| roc_diff | 1-8 | (close[t] - close[t-n]) / close[t-n] × 10000 |
| rsi_diff | 1-8 | rsi7[t] - rsi7[t-n] |
| rp_diff | 1-8 | range_position[t] - range_position[t-n] |
| sma200_diff | 1-8 | sma200_dist_pct[t] - sma200_dist_pct[t-n] |

Reshaped to [8, 4] for LSTM input. Scaled with train-set mean/std.

Future: consider adding volume features or absolute features if V3 labels
show improved accuracy with same features.

---

## Data Split

| Split | Range | Purpose |
|-------|-------|---------|
| Train | 2020-01-01 to 2023-12-31 | Model training |
| Val | 2024-01-01 to 2024-12-31 | Early stopping + hyperparameter selection |
| Test | 2025-01-01 to 2025-12-31 | Final evaluation (never used for decisions) |

1-month gap between splits to prevent label leakage (trades near boundary
could span into next split).

---

## Hyperparameters

| Parameter | Default | Sweep range |
|-----------|:-------:|-------------|
| hidden_size | 128 | 64, 128, 180 |
| dropout | 0.5 | 0.3, 0.5 |
| temperature | 0.5 | 0.3, 0.5, 0.7 |
| lr | 0.001 | 0.0005, 0.001 |
| batch_size | 2048 | 1024, 2048 |
| max_epochs | 100 | — |
| patience | 10 | — |
| loss_weights | 1:1:1 | test 1:1:2 (direction-heavy) |

---

## Expected Label Distribution (estimate)

Based on analysis showing ~50% of entries get stopped:
```
LONG:  ~25% (entries where LONG trade profits)
SHORT: ~25% (entries where SHORT trade profits)
SKIP:  ~50% (entries where both sides lose — mostly stopped)
```

This is a HARDER classification problem than V2 (50/50 binary).
But it's the HONEST problem — the model learns the real distribution.

---

## Evaluation Metrics

### Accuracy metrics:
- Overall accuracy (3-class)
- Per-class accuracy: LONG recall, SHORT recall, SKIP recall
- Confident accuracy (above threshold)
- N confident signals (how many tradeable signals)

### Trading metrics (backtest with V2 exits, 2024-2025):
- Total trades
- Total bps
- Profit factor
- Stop rate (should be LOWER than V2's 50%)
- Max drawdown
- Average bps per trade

### Key comparison: V3 vs V2 (current production)
| Metric | V2 (baseline) | V3 (target) |
|--------|:---:|:---:|
| Stop rate | 50.4% | < 40% (goal) |
| PF | 2.28 | > 2.5 |
| Total bps | +10,129 | maintain or improve |

---

## MLOps Pipeline

### Overview

```
Stage 1:  build_features        (existing, cached)
              |
Stage 2:  build_exit_labels     (NEW — simulate LONG+SHORT trades per bar)
              |
Stage 3:  verify_labels         (NEW — GATE: 1000 samples must match backtest)
              |
Stage 4:  train_v3              (NEW — 3-class LSTM+Attention)
              |
Stage 4b: verify_model          (NEW — GATE: accuracy + sanity thresholds)
              |
Stage 5:  backtest_v3           (NEW — full trading simulation)
              |
Stage 5b: verify_backtest       (NEW — GATE: PF > 1, stop_rate < V2)
              |
Stage 6:  compare               (NEW — V3 vs V2 side-by-side table)
              |
Stage 7:  promote               (existing — manual decision to deploy)
```

Run: `PYTHONPATH=src dvc repro` — stages 1-6 run automatically, stage 7 is manual.
Any GATE failure stops the pipeline. DVC caches stages where inputs haven't changed.

### Stage 1: build_features (existing, unchanged)

```
Script:  src/engine/build_features.py (already exists)
Input:   data/raw/BTCUSDT_15m_ohlcv.parquet
Output:  data/features/direction_prediction/feature_cache.parquet
         (23 columns: OHLCV + rsi7 + range_position + sma200_dist_pct + ...)

Status:  Already built and cached. No changes needed.
DVC:     Already tracked.
```

### Stage 2: build_exit_labels (NEW)

This is the core new stage. For every 15m bar, simulate LONG and SHORT
trades using 1m tick data and the actual V2 exit rules.

```
Script:  src/engine/build_exit_labels.py (NEW)
Input:   data/raw/BTCUSDT_15m_ohlcv.parquet   (15m bars — entry prices)
         data/raw/BTCUSDT_1m_ohlcv.parquet    (1m ticks — exit execution)
         src/engine/position_manager.py         (exit rules — V2)
         src/engine/config/settings.yaml        (exit parameters)
Output:  data/features/direction_prediction/exit_aware_labels.parquet

Process:
  For each 15m bar (index i):
    1. entry_price = open of bar i+1 (same as backtest)
    2. Get 1m ticks from bar i+1 through bar i+6 (max 6 bars = 90 min)
    3. Create fresh V12PositionManager(exit_version="v2")
    4. Open LONG at entry_price
    5. Feed 1m ticks via pm.on_tick() until exit fires or bar 6 closes
    6. Record: long_net_bps, long_exit_reason, long_exit_bar
    7. Repeat steps 3-6 for SHORT
    8. Compute direction label from long_net_bps vs short_net_bps

Output columns:
  long_net_bps      float    actual LONG P&L after fees
  short_net_bps     float    actual SHORT P&L after fees
  long_exit_reason  string   PT_TARGET / PT_LOCK / MID_TRAIL / STOP_LOSS / TIME_EXIT / NO_ZONE
  short_exit_reason string   same
  long_exit_bar     int      0-6, which bar the exit fired
  short_exit_bar    int      0-6
  direction         int      0=LONG, 1=SHORT, 2=SKIP

Runtime: ~30-60 min (210K bars × 2 trades × ~90 1m ticks each)
Caching: DVC tracked. Only regenerate if exit rules or raw data change.
```

### Stage 3: verify_labels (NEW — gate stage)

Verifies that labels match actual backtest outcomes. Pipeline STOPS if
verification fails. This ensures label generator and backtest use the
same logic.

```
Script:  src/engine/verify_exit_labels.py (NEW)
Input:   data/features/direction_prediction/exit_aware_labels.parquet
         data/raw/BTCUSDT_15m_ohlcv.parquet
         data/raw/BTCUSDT_1m_ohlcv.parquet
Output:  data/reports/label_verification.json

Process:
  1. Sample 1000 random bars from labels
  2. For each bar, run REAL backtest (V12PositionManager + 1m ticks)
     for both LONG and SHORT
  3. Compare backtest P&L to label P&L
  4. PASS if all 1000 match within 0.01 bps tolerance
  5. FAIL (exit code 1) if any mismatch → pipeline stops

Also reports label distribution:
  - LONG count / %
  - SHORT count / %
  - SKIP count / %
  - Per-exit-reason breakdown
  - Per-year breakdown
  - Average P&L per class

Gate: train_v3 ONLY runs if this stage passes.
```

### Stage 4: train_v3 (NEW)

Train the V3 model with exit-aware labels.

```
Script:  src/engine/train_v3.py (NEW)
Input:   data/features/direction_prediction/feature_cache.parquet (features)
         data/features/direction_prediction/exit_aware_labels.parquet (labels)
Output:  models/ML_V3_staging/v3_model.onnx
         models/ML_V3_staging/v3_model.pt (torch checkpoint)
         models/ML_V3_staging/scaler.npz

Architecture:
  LSTM(4→128) × 8 steps → Attention → attended(128)
  → h_long_pnl(128→1) + h_short_pnl(128→1)
  → concat(130) → h_dir(130→3) → softmax → [P(LONG), P(SHORT), P(SKIP)]

Loss:
  MSE(pred_long, actual_long_pnl) + MSE(pred_short, actual_short_pnl)
  + CrossEntropy(direction, label)
  Weights: 1.0 : 1.0 : 1.0

Training:
  - Train on 2020-2023, validate on 2024, test on 2025
  - Early stopping on val loss (patience=10)
  - 3 seeds for stability check (42, 43, 44)
  - Best seed selected by val loss

MLflow:
  Experiment: "ml_v3_exit_aware"
  Logged: all hyperparams, train/val/test metrics, model artifact
  Alias: @staging on best run

Protocol: configs/protocols/direction_prediction_v3.yaml (NEW)
  Required metrics:
    - overall_accuracy_3class
    - long_recall, short_recall, skip_recall
    - confident_accuracy (at prod thresholds)
    - n_confident_signals
    - test_pnl_mse (how well PnL heads predict)
```

### Stage 4b: verify_model (NEW — gate stage)

Checks that the trained model is not degenerate before running expensive backtest.

```
Script:  src/engine/verify_v3_model.py (NEW)
Input:   models/ML_V3_staging/v3_model.onnx
         models/ML_V3_staging/scaler.npz
         data/features/direction_prediction/feature_cache.parquet
         data/features/direction_prediction/exit_aware_labels.parquet
Output:  data/reports/v3_model_verification.json

Checks (ALL must pass):
  1. Overall 3-class accuracy > 40%     (not trivial "always SKIP")
  2. LONG recall > 10%                  (model predicts LONG sometimes)
  3. SHORT recall > 10%                 (model predicts SHORT sometimes)
  4. SKIP recall > 10%                  (model doesn't ignore SKIP)
  5. N confident signals > 100          (enough signals to trade)
  6. Seed stability: std < 2% across 3 seeds
  7. PnL head MSE < baseline            (PnL predictions not random)

FAILS pipeline (exit code 1) if ANY check fails.
```

### Stage 5: backtest_v3 (NEW)

Full backtest using V3 model as signal generator.

```
Script:  src/engine/backtest.py (extended with --model v3 flag)
Input:   models/ML_V3_staging/v3_model.onnx
         models/ML_V3_staging/scaler.npz
         data/raw/BTCUSDT_15m_ohlcv.parquet
         data/raw/BTCUSDT_1m_ohlcv.parquet
Output:  data/reports/backtest_staging_v3.json
         data/reports/backtest_staging_v3_trades.parquet

Process:
  1. V3 model generates signals (LONG / SHORT / SKIP)
  2. V1.4 signals also generated (same as current)
  3. Trades executed with V2 exit rules + 1m ticks
  4. Results computed: total_bps, PF, stop_rate, DD, per-signal breakdown

Key: V3 signals replace V2 signals. V1.4 signals unchanged.
     Exit rules unchanged. Only the ML signal source changes.
```

### Stage 5b: verify_backtest (NEW — gate stage)

Checks that backtest results are sane and V3 improves over V2.

```
Script:  src/engine/verify_v3_backtest.py (NEW)
Input:   data/reports/backtest_staging_v3.json
         data/reports/backtest_staging_ml_v2_attention_exitv2.json (V2 baseline)
Output:  data/reports/v3_backtest_verification.json

Checks (ALL must pass):
  1. Total trades > 50                  (enough trades for significance)
  2. PF > 1.0                          (at least profitable)
  3. Stop rate < 50.4%                  (better than V2 baseline)
  4. No NaN or impossible values        (sanity)
  5. Both years have trades             (not concentrated in one period)
  6. Max DD < -2000 bps                 (not catastrophic)

INFORMATIONAL (logged but doesn't fail):
  - PF improvement vs V2
  - Stop rate reduction vs V2
  - Total bps comparison
  - Per-signal-type breakdown

FAILS pipeline (exit code 1) if ANY required check fails.
```

### Stage 6: compare (NEW)

Side-by-side comparison of V2 (current) vs V3 (new).

```
Script:  src/engine/compare_models.py (NEW)
Input:   data/reports/backtest_staging_v3.json
         data/reports/backtest_staging_ml_v2_attention_exitv2.json (existing V2 baseline)
Output:  data/reports/v3_vs_v2_comparison.json
         Printed summary table

Comparison table:
  | Metric      | V2 (current) | V3 (new) | Delta |
  |-------------|:---:|:---:|:---:|
  | Trades      |     |     |     |
  | Total bps   |     |     |     |
  | PF          |     |     |     |
  | Stop rate   |     |     |     |
  | Max DD      |     |     |     |
  | Win rate    |     |     |     |

Gate: V3 must beat V2 on BOTH total_bps AND stop_rate to proceed.
```

### Stage 7: promote (existing)

```
Script:  scripts/mlops/promote.py (already exists)
Process:
  1. Copy models/ML_V3_staging/ → models/ML_V3/
  2. MLflow: set @production alias on V3 model version
  3. Update bot.py to use ML_V3 signal generator
  4. Tag + push → GitHub Actions → EC2 deploy
```

---

## DVC Pipeline (dvc.yaml stages)

```yaml
stages:
  build_features:
    cmd: PYTHONPATH=src python -m engine.build_features
    deps:
      - data/raw/BTCUSDT_15m_ohlcv.parquet
      - src/engine/build_features.py
    outs:
      - data/features/direction_prediction/feature_cache.parquet

  build_exit_labels:
    cmd: PYTHONPATH=src python -m engine.build_exit_labels
    deps:
      - data/raw/BTCUSDT_15m_ohlcv.parquet
      - data/raw/BTCUSDT_1m_ohlcv.parquet
      - src/engine/position_manager.py
      - src/engine/config/settings.yaml
      - src/engine/build_exit_labels.py
    outs:
      - data/features/direction_prediction/exit_aware_labels.parquet

  verify_labels:
    cmd: PYTHONPATH=src python -m engine.verify_exit_labels
    deps:
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/raw/BTCUSDT_15m_ohlcv.parquet
      - data/raw/BTCUSDT_1m_ohlcv.parquet
      - src/engine/verify_exit_labels.py
    outs:
      - data/reports/label_verification.json

  train_v3:
    cmd: PYTHONPATH=src python -m engine.train_v3
    deps:
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/reports/label_verification.json
      - src/engine/train_v3.py
      - configs/params.yaml
    outs:
      - models/ML_V3_staging/v3_model.onnx
      - models/ML_V3_staging/v3_model.pt
      - models/ML_V3_staging/scaler.npz
    metrics:
      - data/reports/v3_train_metrics.json:
          cache: false

  verify_model:
    cmd: PYTHONPATH=src python -m engine.verify_v3_model
    deps:
      - models/ML_V3_staging/v3_model.onnx
      - models/ML_V3_staging/scaler.npz
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - src/engine/verify_v3_model.py
    outs:
      - data/reports/v3_model_verification.json

  backtest_v3:
    cmd: PYTHONPATH=src python -m engine.backtest --model v3
    deps:
      - models/ML_V3_staging/v3_model.onnx
      - models/ML_V3_staging/scaler.npz
      - data/raw/BTCUSDT_15m_ohlcv.parquet
      - data/raw/BTCUSDT_1m_ohlcv.parquet
      - data/reports/v3_model_verification.json
    outs:
      - data/reports/backtest_staging_v3.json
      - data/reports/backtest_staging_v3_trades.parquet

  verify_backtest:
    cmd: PYTHONPATH=src python -m engine.verify_v3_backtest
    deps:
      - data/reports/backtest_staging_v3.json
      - data/reports/backtest_staging_ml_v2_attention_exitv2.json
      - src/engine/verify_v3_backtest.py
    outs:
      - data/reports/v3_backtest_verification.json

  compare:
    cmd: PYTHONPATH=src python -m engine.compare_models
    deps:
      - data/reports/backtest_staging_v3.json
      - data/reports/backtest_staging_ml_v2_attention_exitv2.json
      - data/reports/v3_backtest_verification.json
    outs:
      - data/reports/v3_vs_v2_comparison.json
```

---

## Verification Strategy

### Label verification (Stage 3):
- 1000 random bars sampled
- Real backtest run for each (LONG and SHORT)
- Labels must match within 0.01 bps
- Pipeline FAILS if any mismatch

### Backtest verification:
- V3 backtest uses SAME position_manager + exit rules as label generator
- Compare V3 backtest stop_rate to label SKIP percentage
  - If labels say 50% SKIP but backtest shows 30% stop → model learned something
  - If labels say 50% SKIP and backtest shows 50% stop → model didn't learn

### Cross-validation:
- 3 seeds per training run
- Results must be stable across seeds (std < 2% on confident accuracy)

---

## Implementation Order

### Phase 1: Label pipeline
1. **build_exit_labels.py** — generate labels (30-60 min compute)
2. **verify_exit_labels.py** — confirm labels match backtest (5 min)
3. Inspect label distribution — check LONG/SHORT/SKIP split, per-year, sanity

### Phase 2: Training pipeline
4. **train_v3.py** — new architecture, 3 seeds, MLflow logging
5. **verify_v3_model.py** — accuracy gates + sanity checks

### Phase 3: Evaluation pipeline
6. **backtest.py** — extend with `--model v3` flag
7. **verify_v3_backtest.py** — backtest gates (PF > 1, stop < V2)
8. **compare_models.py** — V3 vs V2 comparison table

### Phase 4: Integration
9. Wire **dvc.yaml** with all stages + deps
10. Run `dvc repro` end-to-end
11. If V3 wins → promote, deploy, monitor on dashboard (Expected vs Actual)

### Files to create:

**Pipeline scripts:**
```
src/engine/build_exit_labels.py     (Stage 2)  ← DONE
src/engine/verify_exit_labels.py    (Stage 3)
src/engine/train_v3.py              (Stage 4)
src/engine/verify_v3_model.py       (Stage 4b)
src/engine/verify_v3_backtest.py    (Stage 5b)
src/engine/compare_models.py        (Stage 6)
src/engine/signals/ml_v3.py         (V3 signal generator for bot + backtest)
```

**MLOps configs (create BEFORE code):**
```
configs/model_cards/ML_V3.yaml                  ← model documentation
configs/protocols/direction_prediction_v3.yaml  ← validation protocol (3-class metrics)
configs/data_cards/exit_aware_labels.yaml       ← label dataset documentation
configs/params.yaml                             ← ADD ml_v3 section
```

### Files to modify:
```
src/engine/backtest.py              (add --model v3 support)
dvc.yaml                            (add all new stages)
```

---

## MLOps Configs

### Model Card: configs/model_cards/ML_V3.yaml

Documents the V3 model for reproducibility and auditing.

```
Contents:
  - Model name: ML_V3
  - Architecture: LSTM(4→128) + Attention + PnL heads + 3-class direction
  - Input: 32 diff features [8, 4] (same as V2)
  - Output: P(LONG), P(SHORT), P(SKIP)
  - Labels: exit-aware (simulated trade P&L with V2 exit rules)
  - Training: date-based split, 2020-2023 train, 2024 val, 2025 test
  - Loss: MSE(long_pnl) + MSE(short_pnl) + CrossEntropy(direction)
  - Predecessor: ML_V2_ATTENTION (binary direction, MFE heads)
  - Key change: labels encode actual trade outcomes instead of ±15 first-hit
  - Hyperparameters: hidden=128, dropout=0.5, temp=0.5, lr=0.001
  - Performance: [filled after training]
  - Known limitations: labels coupled to V2 exit rules, same feature set as V2
```

### Protocol: configs/protocols/direction_prediction_v3.yaml

Defines required metrics for V3 validation. Extended from V1 protocol
to handle 3-class output and PnL regression.

```
Required metrics (model accuracy):
  - overall_accuracy_3class         (> 40% — not trivial)
  - long_precision                  (of signals that say LONG, how many correct?)
  - long_recall                     (of actual LONGs, how many detected?)
  - short_precision
  - short_recall
  - skip_recall                     (> 10% — model uses SKIP class)
  - confident_accuracy              (accuracy at prod thresholds)
  - n_confident_signals             (> 100 — enough to trade)
  - pnl_head_mse_long              (regression quality)
  - pnl_head_mse_short
  - seed_stability_std              (< 2% across 3 seeds)

Required metrics (trading — from backtest):
  - total_trades
  - total_bps
  - profit_factor                   (> 1.0)
  - stop_rate                       (< 50.4% V2 baseline)
  - max_drawdown_bps                (< -2000)
  - win_rate
  - avg_bps_per_trade
  - per_signal_type_bps
  - per_year_bps

Required artifacts:
  - v3_model.onnx
  - v3_model.pt
  - scaler.npz
  - backtest_staging_v3.json
  - backtest_staging_v3_trades.parquet
  - v3_vs_v2_comparison.json
```

### Data Card: configs/data_cards/exit_aware_labels.yaml

Documents the exit-aware label dataset.

```
Contents:
  - Name: exit_aware_labels
  - Description: per-bar trade simulation outcomes using V2 exit rules
  - Source: generated by build_exit_labels.py from 15m + 1m OHLCV
  - Rows: ~210K (one per 15m bar, 2020-2025)
  - Columns:
      long_net_bps (float)      — LONG trade P&L after 8 bps fees
      short_net_bps (float)     — SHORT trade P&L after 8 bps fees
      long_exit_reason (string) — which exit rule fired
      short_exit_reason (string)
      long_exit_bar (int)       — bar 0-6
      short_exit_bar (int)
      direction (int)           — 0=LONG, 1=SHORT, 2=SKIP
  - Dependencies:
      data/raw/BTCUSDT_15m_ohlcv.parquet
      data/raw/BTCUSDT_1m_ohlcv.parquet
      src/engine/position_manager.py (V2 exit rules)
      src/engine/config/settings.yaml (exit parameters)
  - Regeneration: required if exit rules or raw data change
  - Verified by: verify_exit_labels.py (1000-sample backtest match)
  - Expected distribution: ~25% LONG, ~25% SHORT, ~50% SKIP
  - Split: train 2020-2023, val 2024, test 2025 (date-based)
```

### params.yaml: ml_v3 section

```yaml
ml_v3:
  inference:
    conf_long: 0.50        # threshold for P(LONG) — tune after training
    conf_short: 0.50       # threshold for P(SHORT) — tune after training
  training:
    hidden: 128
    dropout: 0.5
    temperature: 0.5
    lr: 0.001
    batch_size: 2048
    max_epochs: 100
    patience: 10
    seed: 42
    loss_weight_pnl: 1.0
    loss_weight_dir: 1.0
  split:
    train_start: '2020-01-01'
    train_end: '2023-12-31'
    val_start: '2024-01-01'
    val_end: '2024-12-31'
    test_start: '2025-01-01'
    test_end: '2025-12-31'
```

---

## Risk / What Could Go Wrong

1. **SKIP dominates:** ~50% of labels may be SKIP. Model might learn "always predict SKIP" (trivial solution). Mitigation: class weights in CrossEntropy loss, undersample SKIP, or focal loss.

2. **Labels coupled to exit params:** if exit rules change, labels must be regenerated. Accepted — exit rules are stable (V2, frozen). DVC deps on position_manager.py and settings.yaml auto-detect changes.

3. **Compute cost:** 210K bars × 2 trades × ~90 ticks = ~38M tick operations. Estimated 30-60 min on CPU. Runs once, cached by DVC.

4. **Same features, same ceiling?** Features might not contain enough information to distinguish LONG from SKIP even with better labels. If accuracy doesn't improve, next step is adding new features (volume, multi-timeframe) — not changing architecture.

5. **Label leakage at split boundaries:** a trade simulation starting at the last bar of 2023 looks into 2024 data (1m ticks). Mitigation: 1-month gap between splits. Labels for bars within 6 bars of split boundary are excluded.
