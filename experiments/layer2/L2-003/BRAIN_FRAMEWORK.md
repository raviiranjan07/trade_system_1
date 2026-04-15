# Brain Pipeline v1.1 — End-to-End Framework

## Pipeline Overview

```
DATA INGESTION → DATA PROCESSING → FEATURE SELECTION → LABEL GENERATION → DATASET CREATION → TRAINING → EVALUATION → OUTPUT → PAPER TRADING
```

Every step is configurable. Change one config, rerun the pipeline, compare results.

---

## Step 1: DATA INGESTION

**What:** Load raw market data and validate

**Config:**
```yaml
ingestion:
  symbol: BTCUSDT
  timeframe: 15m              # configurable: 1m, 5m, 15m, 1h, 4h
  source: data/ohlcv/         # path to parquet files
  date_range:
    start: 2020-01-01
    end: 2025-12-31
  validation:
    check_gaps: true           # detect missing bars
    max_gap_bars: 3            # alert if gap > N bars
    fill_method: forward       # forward fill small gaps
    remove_weekends: false     # crypto trades 24/7
  warm_up:
    min_bars: 50               # skip first N bars (need history for indicators + S/R)
                               # RSI needs ~14, ATR needs ~14, S/R needs 8+8=16, range_position needs 50
```

**Output:** Validated OHLCV dataframe (open, high, low, close, volume)
**Validation report:** total bars, gaps found, gaps filled, date range confirmed, warm-up bars skipped

---

## Step 2: DATA PROCESSING

**What:** Compute all features from raw data

**Config:**
```yaml
features:
  price_action:
    enabled: true
    roc: true
    rsi7: true
    range_position: true

  market_structure:
    enabled: true
    sr_method: v5_confirmed      # v1, v2, v3, v5
    sr_min_touches: 2            # minimum touches for confirmed S/R
    sr_lookback: 8               # bars for S/R detection
    sr_carry_forward: true       # use previous S/R when not found
    features:
      - support_range_low
      - support_range_high
      - resistance_range_low
      - resistance_range_high
      - zone_width
      - support_retest
      - resistance_retest
      - distance_to_support
      - distance_to_resistance
      - recovery_up
      - recovery_down
      - window_low
      - window_high

  volatility:
    enabled: true
    atr_pct: true                # ATR as % of price (14-bar)
    atr_percentile: true         # ATR rank vs last 100 bars
    ema_separation: true         # EMA9 vs EMA21 distance

  bar_characteristics:
    enabled: true
    range_bps: true              # current bar high-low in bps
    body_bps: true               # current bar open-close in bps
    dist_from_high20_pct: true   # price position in 20-bar range

  activity:
    enabled: true
    volume_ratio: true           # volume vs 20-bar average

  time:
    enabled: true
    hour_utc: true               # hour of day (0-23)
```

### All Features (23 per snapshot)

| # | Category | Feature | Source |
|---|----------|---------|--------|
| 1 | Price Action | roc | computed |
| 2 | Price Action | rsi7 | computed |
| 3 | Price Action | range_position | computed |
| 4 | Market Structure | support_range_low | S/R V5 |
| 5 | Market Structure | support_range_high | S/R V5 |
| 6 | Market Structure | resistance_range_low | S/R V5 |
| 7 | Market Structure | resistance_range_high | S/R V5 |
| 8 | Market Structure | zone_width | S/R V5 |
| 9 | Market Structure | support_retest | S/R V5 |
| 10 | Market Structure | resistance_retest | S/R V5 |
| 11 | Market Structure | distance_to_support | S/R V5 |
| 12 | Market Structure | distance_to_resistance | S/R V5 |
| 13 | Market Structure | recovery_up | S/R V5 |
| 14 | Market Structure | recovery_down | S/R V5 |
| 15 | Market Structure | window_low | S/R V5 |
| 16 | Market Structure | window_high | S/R V5 |
| 17 | Volatility | atr_pct | L2-001 5/5 validated |
| 18 | Volatility | atr_percentile | L2-001 5/5 validated |
| 19 | Volatility | ema_separation | L2-001 5/5 validated |
| 20 | Bar Characteristics | range_bps | L2-001 5/5 validated |
| 21 | Bar Characteristics | body_bps | L2-001 5/5 validated |
| 22 | Bar Characteristics | dist_from_high20_pct | L2-001 5/5 validated |
| 23 | Activity | volume_ratio | L2-001 5/5 validated |

Note: hour_utc (L2-001 5/5 validated) is available but may need special encoding (cyclical or one-hot). Configurable to enable/disable.

### S/R Per-Snapshot Computation

Each snapshot computes its own S/R from its own lookback window:

```
For bar t with 8 snapshots (t-7 to t):
  Snapshot at t-7: S/R computed from bars t-14 to t-7
  Snapshot at t-6: S/R computed from bars t-13 to t-6
  Snapshot at t-5: S/R computed from bars t-12 to t-5
  ...
  Snapshot at t:   S/R computed from bars t-7 to t
```

Each snapshot sees a DIFFERENT S/R — the brain watches the zone form and evolve.

### S/R Carry-Forward Logic

```
At each snapshot:
  1. Try to find confirmed S/R (2+ touches) in current 8-bar window
  2. If found: use it, update carry-forward state
  3. If NOT found: use previous snapshot's S/R (carry forward)
  4. If no previous exists (start of data): S/R features = NaN
```

### Raw Price Normalization

Raw price features (support_range_low/high, resistance_range_low/high, window_low/high) need special handling because BTC went from 7k (2020) to 100k (2025):

```
Option A: Convert to relative (bps from current price)
  - support_range_low_rel = (current_price - support_range_low) / current_price * 10000
  - This makes 7k and 100k comparable

Option B: Normalize within each snapshot window
  - All prices in the window divided by current close
  - Becomes ratio (0.99, 1.01, etc.)

Option C: Let the brain learn raw + use per-sample normalization
  - Normalize each sample independently (mean/std of that sample)

Decision: TBD from testing. Start with Option A (relative bps).
```

**Output:** Feature dataframe (every bar has all enabled features)

---

## Step 2.5: FEATURE SELECTION

**What:** Test which features help and which hurt before training

**Config:**
```yaml
feature_selection:
  enabled: true
  method: ablation              # ablation, correlation, importance
  ablation:
    baseline: all_features      # train with all, then remove one at a time
    metric: confident_accuracy  # which metric to optimize
  correlation:
    threshold: 0.95             # flag features with >95% correlation
    action: report              # report, auto_remove
  importance:
    method: permutation         # permutation, gradient, shap
    report: true
```

**Process:**
1. Check feature correlations — flag redundant pairs
2. Train baseline with all features
3. Ablation: remove one feature at a time, measure impact
4. Report: which features improve accuracy, which are neutral, which hurt

**Output:** Recommended feature set, feature importance ranking

---

## Step 3: LABEL GENERATION

**What:** Compute what we're predicting

**Config:**
```yaml
labels:
  direction:
    enabled: true
    horizon: 8                # configurable: 1, 4, 8, 16, 32, 96
    threshold_bps: 15         # which +/-15bps hits first
    classes: [LONG, SHORT]    # exclude SKIP and BOTH
  mfe:
    enabled: true
    horizons: [1, 2, 3, 4, 5, 6, 7, 8]
  class_balance:
    method: none              # none, undersample, oversample, class_weights
```

**Data leakage prevention:**
```
S/R features use:  bars t-14 to t    (past only)
Label uses:        bars t+1 to t+8   (future only)
NO OVERLAP — verified by design
```

**Output:** Label dataframe (direction, mfe_up_H, mfe_down_H per bar)

---

## Step 4: DATASET CREATION

**What:** Create snapshots, split train/test, normalize

**Config:**
```yaml
dataset:
  snapshot_count: 8            # how many past bars per sample
  snapshot_gap: 1              # gap between snapshots (1 = consecutive)

  split:
    train: [2020-01-01, 2022-12-31]
    val: [2023-01-01, 2023-12-31]       # validation for early stopping (carved from train)
    test: [2024-01-01, 2025-12-31]
    # Validation is NEVER used for training — only for early stopping + hyperparameter selection
    # Test is NEVER seen until final evaluation

  normalization:
    method: zscore             # zscore, minmax, none
    fit_on: train              # only fit on train data
    raw_price_method: relative_bps   # how to handle raw prices (see Step 2)
    nan_handling: zero         # zero, drop, forward_fill

  filtering:
    remove_skip: true          # remove bars where direction = SKIP/BOTH
    min_zone_width: 0          # configurable: filter narrow zones
    require_zone: false        # only include bars with full zone
```

### Handling Missing S/R in Snapshot Sequences

When one snapshot in an 8-snapshot sequence has S/R but another doesn't:

```
Snapshot sequence for bar t:
  t-7: S/R found     -> use it
  t-6: S/R found     -> use it
  t-5: S/R NOT found -> carry forward from t-6
  t-4: S/R NOT found -> carry forward from t-6
  t-3: S/R found     -> use it (new zone)
  t-2: S/R found     -> use it
  t-1: S/R found     -> use it
  t:   S/R found     -> use it
```

The brain sees the zone disappear and reappear — this is information.

**Output:** Train dataset, Test dataset (X = snapshots x features, Y = labels)

---

## Step 5: TRAINING

**What:** Train the brain

**Config:**
```yaml
training:
  architecture: lstm           # mlp, lstm, gru, transformer, hybrid

  model:
    hidden_size: 128
    num_layers: 1
    dropout: 0.2
    # hybrid-specific
    snapshot_dense: 64         # dense layer per snapshot before LSTM

  tasks:
    direction:
      enabled: true
      loss: cross_entropy
      weight: 1.0
    mfe:
      enabled: true
      loss: mse
      weight: 5.0              # weight of MFE loss vs direction loss

  optimizer:
    type: adam
    lr: 0.001
    weight_decay: 0.0001

  schedule:
    batch_size: 512
    max_epochs: 200
    patience: 10               # early stopping on validation loss

  hyperparameter_tuning:
    enabled: false             # manual first, systematic later
    method: grid               # grid, random, bayesian
    search_space:
      lr: [0.0005, 0.001, 0.002]
      hidden_size: [64, 128, 256]
      dropout: [0.1, 0.2, 0.3]
      mfe_weight: [1.0, 3.0, 5.0]

  device: cuda                 # cuda, cpu
  seed: 42                     # reproducibility

  environment:
    primary: colab_t4            # colab_t4, colab_a100, local_gpu, local_cpu
    # Steps 1-4 (data processing) run locally
    # Step 5 (training) runs on Colab GPU
    # Steps 6-8 (evaluation, output, live) run locally
    # Pipeline exports dataset to .npz/.parquet for Colab upload
    export_dataset: true         # save processed dataset for Colab
    export_path: experiments/brain/datasets/
```

### Training Environment

```
LOCAL MACHINE:
  Step 1: Data Ingestion        (load OHLCV parquet)
  Step 2: Data Processing       (compute features)
  Step 2.5: Feature Selection   (correlation check)
  Step 3: Label Generation      (compute labels)
  Step 4: Dataset Creation      (snapshots, normalize, split)
  --> Export dataset (.npz) for Colab upload

COLAB T4 GPU:
  Step 5: Training              (train model on GPU)
  Step 6: Evaluation            (metrics on GPU)
  --> Download model weights + metrics

LOCAL MACHINE:
  Step 7: Output                (inference)
  Step 8: Paper Trading         (live bot)
```

### Loss Function

```
total_loss = direction_weight * CrossEntropy(direction_pred, direction_true)
           + mfe_weight * MSE(mfe_up_pred, mfe_up_true)
           + mfe_weight * MSE(mfe_down_pred, mfe_down_true)
```

Multi-task learning: direction + MFE together. Proved +3-4% improvement over direction-only.

**Output:** Trained model weights + training history (loss, accuracy per epoch)

---

## Step 6: EVALUATION

**What:** Measure how good the brain is

**Config:**
```yaml
evaluation:
  metrics:
    direction:
      - overall_accuracy         # all bars
      - confident_accuracy       # bars where confidence > threshold
      - per_class_accuracy       # LONG vs SHORT separately
    sr_specific:
      - bounce_rate_at_support
      - bounce_rate_at_resistance
      - bounce_by_touch_count
      - bounce_by_zone_width
    mfe:
      - mfe_up_mae               # mean absolute error
      - mfe_down_mae
      - mfe_direction_accuracy   # does max(mfe_up) > max(mfe_down) match direction?
    feature:
      - feature_importance        # which features contributed most
      - feature_correlation       # check for redundancy
    consistency:
      - train_test_gap            # overfitting check (max 3%)
      - yearly_breakdown          # per-year accuracy
      - monthly_stability         # no single month should be terrible

  confidence_thresholds: [0.55, 0.60, 0.65, 0.70]

  sr_position:
    method: data_driven           # data_driven, fixed
    # data_driven: "at support" = price within support_range_low to support_range_high
    # fixed: "at support" = distance_to_support <= threshold
    fixed_threshold_bps: 5        # only used if method=fixed

  backtest:
    enabled: true
    entry_rule: confidence > threshold
    exit_rules:
      trailing_stop_long: 20     # bps
      trailing_stop_short: 30
      tighten_after_bar: 5
      tighten_to: 8
      time_exit: 10
    report:
      - total_trades
      - win_rate
      - total_bps
      - profit_factor
      - max_drawdown
      - long_vs_short_split
      - per_year_breakdown

  comparison:
    baselines:
      - name: random
        accuracy: 50.0
      - name: current_mlp_v15
        accuracy: 57.8
      - name: sr_only_v5
        accuracy: 55.1
    format: side_by_side_table
```

### Evaluation Report Format

Every experiment produces a standard report:

```
=== Experiment: {config_name} ===
Config: {path to yaml}
Date: {timestamp}
Git hash: {commit}
Features: {list of enabled features} (version: {feature_version})
Architecture: {model type}

--- Direction ---
Overall accuracy:   XX.X%
Confident (>0.60):  XX.X% (N bars)
LONG accuracy:      XX.X%
SHORT accuracy:     XX.X%

--- S/R ---
At support bounce:  XX.X%
At resistance bounce: XX.X%

--- MFE ---
MFE up MAE:   XX.X bps
MFE down MAE: XX.X bps

--- Feature Importance ---
Top 5: feature1 (XX%), feature2 (XX%), ...
Bottom 5: featureN (X%), ...

--- Consistency ---
Train: XX.X%  Test: XX.X%  Gap: X.X%
2020: XX.X%  2021: XX.X%  2022: XX.X%  2023: XX.X%  2024: XX.X%  2025: XX.X%

--- Backtest ---
Trades: XXX  Win: XX.X%  Bps: +XXXX  PF: X.XX  DD: -XXX

--- vs Baselines ---
| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Random | 50.0% | +X.X% |
| Current MLP | 57.8% | +X.X% |
| This model | XX.X% | -- |
```

---

## Step 7: OUTPUT

**What:** What the brain produces for each bar

```python
{
    "direction": "LONG",         # LONG / SHORT / SKIP
    "confidence": 0.73,          # 0.0 to 1.0
    "mfe_up": 28.5,              # predicted upward MFE in bps
    "mfe_down": 12.3,            # predicted downward MFE in bps
    "context": {
        "support_range": [8649, 8653],
        "resistance_range": [8670, 8677],
        "zone_width": 21.5,
        "position": "at_support",
        "support_touches": 5,
        "resistance_touches": 3
    }
}
```

**When brain says SKIP:**
- Confidence is below both LONG and SHORT thresholds
- Bot does nothing -- no trade
- Log the decision for analysis

---

## Step 8: PAPER TRADING INTEGRATION

**What:** Connect brain to live bot

**Config:**
```yaml
live:
  model_path: models/brain_v1.pt
  scaler_path: models/scaler.npz
  config_path: models/config.yaml      # SAME config used in training

  thresholds:
    long_confidence: 0.60
    short_confidence: 0.65              # asymmetric (SHORT needs more confidence)

  wallet: ml_brain                      # separate wallet
  position_size: 0.001                  # BTC

  state:
    sr_carry_forward: true              # persist S/R state between bars
    state_file: data/v12_trades/brain_state.json

  monitoring:
    log_every_bar: true                 # log prediction on every bar
    log_file: data/risk_logs/brain/decisions.csv
    dashboard: true                     # push brain output to web dashboard
    alerts:
      accuracy_below: 0.50              # alert if rolling accuracy drops
      rolling_window: 50                # last N trades
      max_consecutive_losses: 8

  error_handling:
    on_model_error: skip                # skip, use_previous, halt
    on_data_error: skip                 # skip (no trade), halt
    on_missing_bar: wait                # wait for next bar
    log_errors: true
    error_log: data/risk_logs/brain/errors.csv
```

**Live Pipeline (every 15min bar):**
```
1. New bar arrives (OHLCV)
2. Validate bar data (not null, reasonable price range)
3. Compute features (same code as training pipeline)
4. Load carry-forward S/R state
5. Create snapshot (current bar's features)
6. Feed 8 snapshots to brain
7. Brain outputs direction + confidence + MFE + context
8. If confidence > threshold -> signal entry
9. Update S/R state, save to state_file
10. Push to dashboard (direction, confidence, S/R zones, position)
11. Log decision
12. Existing exit rules handle open positions
```

**Dashboard Integration:**
- Show brain's current prediction (LONG/SHORT/SKIP + confidence)
- Show S/R zones on price chart (support range, resistance range)
- Show zone width and touch counts
- Show price position relative to zones
- Show rolling accuracy of brain predictions

**Model Retraining Schedule:**
```yaml
retraining:
  trigger: manual                       # manual, monthly, quarterly
  retrain_window: [2020-01-01, latest]  # expanding window
  validation: last_3_months             # hold out for validation
  auto_deploy: false                    # always manual review before deploy
  keep_previous: true                   # keep old model as fallback
  compare_before_deploy: true           # must beat current model on OOS
```

---

## Experiment Tracking

Every experiment is logged:

**Config:**
```yaml
tracking:
  experiment_dir: experiments/brain/
  log_format: yaml
  auto_log: true
```

**Experiment log structure:**
```
experiments/brain/
  registry.csv                          # master log of all experiments
  EXP-B001/
    config.yaml                         # full config used
    metrics.json                        # evaluation results
    training_history.csv                # loss/accuracy per epoch
    model.pt                            # saved model weights
    notes.md                            # what we learned
  EXP-B002/
    ...
```

**registry.csv columns:**
```
exp_id, date, config, architecture, features, feature_version,
accuracy, confident_acc, pf, total_bps, train_acc, test_acc, gap,
status, notes
```

---

## Reproducibility

```yaml
reproducibility:
  seed: 42
  feature_version: v5                   # track which feature computation version
  label_version: v1                     # track which label computation version
  torch_deterministic: true
  log_git_hash: true                    # log current git commit
```

---

## Feature Versioning

When features change, track the version:

| Version | Features | Count | Changes |
|---------|----------|-------|---------|
| v1 | roc1-8, rsi7, range_position | 10 | Original MLP (V1.5) |
| v2 | roc, rsi7, range_position (per snapshot) | 3 | Snapshot-based |
| v3 | v2 + 13 S/R features | 16 | S/R added |
| v4 | v3 + atr_pct, ema_separation, volume_ratio | 19 | Volatility + activity |
| v5 | v4 + atr_percentile, range_bps, body_bps, dist_from_high20_pct | 23 | All L2-001 validated features |

---

## Config System

**One base config + one config per experiment.** Experiment configs only override what's different.

### File Structure

```
configs/
  base.yaml                    # default: all features, BTCUSDT, 15m, LSTM

  # Architecture experiments
  exp_mlp.yaml                 # architecture: mlp
  exp_lstm.yaml                # architecture: lstm
  exp_gru.yaml                 # architecture: gru
  exp_transformer.yaml         # architecture: transformer
  exp_hybrid.yaml              # architecture: hybrid

  # Feature experiments
  exp_base_only.yaml           # price_action only (3 features)
  exp_sr_only.yaml             # market_structure only
  exp_base_sr.yaml             # price_action + market_structure
  exp_full.yaml                # all features (23)

  # Timeframe experiments
  exp_5m.yaml                  # timeframe: 5m
  exp_1h.yaml                  # timeframe: 1h
  exp_4h.yaml                  # timeframe: 4h
  exp_multi_tf.yaml            # timeframe: [15m, 1h]

  # Symbol experiments
  exp_eth.yaml                 # symbol: ETHUSDT
  exp_gold.yaml                # symbol: XAUUSD

  # Label experiments
  exp_h4_label.yaml            # horizon: 4
  exp_h16_label.yaml           # horizon: 16
```

### Config Inheritance

Each experiment config inherits from base and only overrides what's different:

```yaml
# configs/base.yaml (full default config)
ingestion:
  symbol: BTCUSDT
  timeframe: 15m
  source: data/ohlcv/
  date_range:
    start: 2020-01-01
    end: 2025-12-31
  validation:
    check_gaps: true
    max_gap_bars: 3
    fill_method: forward
  warm_up:
    min_bars: 50

features:
  price_action: { enabled: true, roc: true, rsi7: true, range_position: true }
  market_structure: { enabled: true, sr_method: v5_confirmed, sr_min_touches: 2 }
  volatility: { enabled: true, atr_pct: true, atr_percentile: true, ema_separation: true }
  bar_characteristics: { enabled: true, range_bps: true, body_bps: true, dist_from_high20_pct: true }
  activity: { enabled: true, volume_ratio: true }

labels:
  direction: { horizon: 8, threshold_bps: 15 }
  mfe: { enabled: true, horizons: [1,2,3,4,5,6,7,8] }

dataset:
  snapshot_count: 8
  split: { train: [2020-01-01, 2022-12-31], val: [2023-01-01, 2023-12-31], test: [2024-01-01, 2025-12-31] }
  normalization: { method: zscore, fit_on: train }

training:
  architecture: lstm
  model: { hidden_size: 128, num_layers: 1, dropout: 0.2 }
  tasks: { direction: { weight: 1.0 }, mfe: { weight: 5.0 } }
  optimizer: { type: adam, lr: 0.001, weight_decay: 0.0001 }
  schedule: { batch_size: 512, max_epochs: 200, patience: 10 }
  seed: 42
```

```yaml
# configs/exp_mlp.yaml (only overrides architecture)
inherit: base.yaml
training:
  architecture: mlp
```

```yaml
# configs/exp_sr_only.yaml (only S/R features)
inherit: base.yaml
features:
  price_action: { enabled: false }
  volatility: { enabled: false }
  bar_characteristics: { enabled: false }
  activity: { enabled: false }
```

```yaml
# configs/exp_eth.yaml (different symbol)
inherit: base.yaml
ingestion:
  symbol: ETHUSDT
  source: data/ohlcv/
```

```yaml
# configs/exp_multi_tf.yaml (multiple timeframes)
inherit: base.yaml
ingestion:
  timeframe: [15m, 1h]
```

### How To Run

**Single experiment:**
```bash
PYTHONPATH=src python -m brain.pipeline --config configs/exp_mlp.yaml
```

**Multiple experiments in sequence:**
```bash
PYTHONPATH=src python -m brain.pipeline --config configs/exp_mlp.yaml configs/exp_lstm.yaml configs/exp_gru.yaml
```

**Compare experiments:**
```bash
PYTHONPATH=src python -m brain.compare --exp EXP-B001 EXP-B002 EXP-B003
```

**Deploy to paper trading:**
```bash
PYTHONPATH=src python -m brain.deploy --exp EXP-B001 --confirm
```

---

## What's Next

1. Build the pipeline code (src/brain/)
2. Test with base features only (3 features x 8 snapshots) -- establish baseline
3. Add S/R features -- measure improvement
4. Add volatility + bar + activity features -- measure improvement
5. Feature selection -- which features help, which hurt
6. Test architectures (MLP vs LSTM vs Hybrid)
7. Hyperparameter tuning on best architecture
8. Best model -> paper trading
9. Monitor and iterate
