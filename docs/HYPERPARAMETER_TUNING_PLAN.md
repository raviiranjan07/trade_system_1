# Hyperparameter Tuning Plan

**Date:** 2026-04-18
**Applies to:** Any ML model (V1, V2, V3, future models)
**Current focus:** ML V3 (exit-aware labels + snapshot features)
**Baseline:** 614 trades, +7,721 bps, PF 2.42 on 2025 test (at default params)

---

## Design Principles

1. **Config-driven, not hardcoded** — all sweep ranges live in `configs/params.yaml` under each model's `sweep:` section. Scripts read from config.
2. **Model-agnostic** — same sweep scripts work for any model via `--model` flag.
3. **Tune on validation (2024). Confirm on test (2025). Never tune on test.**

```
For each config:
  1. Run backtest on 2024 (val)  → pick best
  2. Run best on 2025 (test)    → confirm it holds
  3. Lock config in params.yaml
```

---

## Config Structure (configs/params.yaml)

Each model has an `inference`, `training`, and `sweep` section:

```yaml
ml_v3:
  inference:
    conf_long: 0.40              # current production value
    conf_short: 0.40             # updated by sweep winner
  training:
    hidden: 128
    dropout: 0.5
    temperature: 0.5
    lr: 0.001
    batch_size: 2048
    loss_weight_pnl: 1.0
    loss_weight_dir: 1.0
    seed: 42
  sweep:
    # Level 1: inference thresholds (no retraining)
    conf_long: [0.35, 0.37, 0.40, 0.42, 0.45, 0.50]
    conf_short: [0.35, 0.37, 0.40, 0.42, 0.45, 0.50]
    strategy: independent        # "independent" or "grid"
    # Level 2: training params (retraining required)
    learning_rate: [0.0005, 0.001, 0.002]
    loss_weight_pnl: [0.5, 1.0, 2.0]
    loss_weight_dir: [0.5, 1.0, 2.0]
    batch_size: [512, 1024, 2048]
    # Level 3: architecture (retraining required)
    hidden_size: [64, 128, 180, 256]
    temperature: [0.1, 0.3, 0.5, 0.7, 1.0]
    dropout: [0.3, 0.4, 0.5, 0.6]
```

Same structure applies to ml_v1, ml_v2_attention, or any future model —
just add a `sweep:` section with the ranges to test.

---

## Script Design (model-agnostic)

All sweep scripts accept `--model` flag and read config from params.yaml:

```bash
# Same script, different models:
python -m engine.sweep_thresholds --model ml_v3
python -m engine.sweep_thresholds --model ml_v1
python -m engine.sweep_thresholds --model ml_v2_attention

# Level 2:
python -m engine.sweep_training_params --model ml_v3

# Level 3:
python -m engine.sweep_architecture --model ml_v3
```

Script internals:
```python
# sweep_thresholds.py
model_name = args.model                           # "ml_v3"
sweep_cfg = params[model_name]["sweep"]            # from params.yaml
long_values = sweep_cfg["conf_long"]               # [0.35, 0.37, ...]
short_values = sweep_cfg["conf_short"]             # [0.35, 0.37, ...]
strategy = sweep_cfg.get("strategy", "independent")
model_class, model_dir = ML_GENERATORS[model_name] # from backtest registry
```

---

## Level 1: Threshold Sweep (NO retraining)

**Cost:** ~2 min per config (just re-run backtest with existing model)
**Priority:** HIGH — most impact, zero cost

### Parameters (read from params.yaml → model.sweep):

| Parameter | Current | Sweep values (from config) | What it controls |
|-----------|:-------:|---------------------------|-----------------|
| conf_long | 0.40 | params[model]["sweep"]["conf_long"] | Min P(LONG) to fire signal |
| conf_short | 0.40 | params[model]["sweep"]["conf_short"] | Min P(SHORT) to fire signal |
| strategy | — | params[model]["sweep"]["strategy"] | "independent" or "grid" |

### Implementation:

```
Script: src/engine/sweep_thresholds.py --model {model_name}

What it does:
  1. Read sweep config from params.yaml → {model_name}.sweep
  2. Load trained model from ML_GENERATORS registry (no retraining)
  3. Run inference on ALL val bars (2024) — get probabilities once
  4. For each threshold combo (based on strategy):
     - Filter signals where P(LONG) > threshold or P(SHORT) > threshold
     - Run those signals through backtest (2024 only)
     - Record: trades, bps, PF, stop_rate
  5. Rank by PF (min 100 trades)
  6. Take top 3, confirm each on test (2025)
  7. Pick the one that holds best on 2025
  8. Save winner to output JSON

Input:
  - Model ONNX + scaler (from ML_GENERATORS registry)
  - data/raw/BTCUSDT_15m_ohlcv.parquet
  - data/raw/BTCUSDT_1m_ohlcv.parquet
  - configs/params.yaml → {model_name}.sweep

Output:
  - data/reports/{model_name}_threshold_sweep.json
  - Best config logged (user updates params.yaml manually)

MLflow:
  - Experiment: "{model_name}_threshold_sweep"
  - One run per threshold combo

Reusable: same script works for ml_v1, ml_v2_attention, ml_v3.
```

### Sweep strategies:

**Independent (default, from config strategy="independent"):**
```
Fix conf_short at current, sweep conf_long: N runs
Fix conf_long at best, sweep conf_short: N runs
Total: 2 × N runs
```

**Grid (from config strategy="grid"):**
```
All conf_long × conf_short combinations
Total: N × N runs
```

### Evaluation metric:
- Primary: Profit Factor on val (2024)
- Secondary: total_bps, stop_rate
- Constraint: must have > 100 trades

---

## Level 2: Training Params (retraining required)

**Cost:** ~5 min per config (retrain model)
**Priority:** MEDIUM — do after Level 1

### Parameters (read from params.yaml → model.sweep):

| Parameter | Current | Config key | What it controls |
|-----------|:-------:|-----------|-----------------|
| loss_weight_pnl | 1.0 | sweep.loss_weight_pnl | How much LSTM focuses on P&L prediction |
| loss_weight_dir | 1.0 | sweep.loss_weight_dir | How much LSTM focuses on direction prediction |
| learning_rate | 0.001 | sweep.learning_rate | How fast model learns |
| batch_size | 2048 | sweep.batch_size | How many bars per weight update |

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
Script: src/engine/sweep_training_params.py --model {model_name}

What it does:
  1. Read sweep config from params.yaml → {model_name}.sweep
  2. Step 1 — Loss weight sweep:
     - For each combo of loss_weight_pnl × loss_weight_dir (from config):
       - Train fresh model on train split (same labels, same features)
       - Evaluate on val (accuracy + backtest)
       - Record: val_PF, val_bps, val_stop_rate
     - Pick best loss weight combo

  3. Step 2 — LR + batch_size sweep:
     - Fix loss weights from Step 1
     - For each combo of lr × batch_size (from config):
       - Train fresh model
       - Evaluate on val
     - Pick best combo

  4. Step 3 — Confirm:
     - Train final model with all best params
     - Run backtest on test (2025)
     - Compare to baseline

Input:
  - data/features/direction_prediction/feature_cache.parquet
  - data/features/direction_prediction/exit_aware_labels.parquet
  - data/raw/ (for backtest)
  - configs/params.yaml → {model_name}.sweep

Output:
  - data/reports/{model_name}_training_sweep.json
  - Best config logged (user updates params.yaml manually)

MLflow:
  - Experiment: "{model_name}_training_sweep"
  - Each run logs: params, val_accuracy, val_PF, val_bps

Reusable: same script works for any model.
```

---

## Level 3: Architecture Params (only if needed)

**Cost:** ~5 min per config (retrain model)
**Priority:** LOW — only if Level 1+2 don't satisfy

### Parameters (read from params.yaml → model.sweep):

| Parameter | Current | Config key | What it controls |
|-----------|:-------:|-----------|-----------------|
| hidden_size | 128 | sweep.hidden_size | LSTM capacity — how many patterns it can learn |
| temperature | 0.5 | sweep.temperature | Attention sharpness — focus on few steps vs blend all |
| dropout | 0.5 | sweep.dropout | Regularization — prevents overfitting |

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
Script: src/engine/sweep_architecture.py --model {model_name}

What it does:
  1. Read sweep config from params.yaml → {model_name}.sweep
  2. Step 1 — Hidden + Temperature sweep:
     - Fix best from Level 1 + 2
     - For each combo of hidden_size × temperature (from config):
       - Train fresh model
       - Evaluate on val (accuracy + backtest)
     - Pick best combo

  3. Step 2 — Dropout sweep:
     - Fix best hidden + temp from Step 1
     - For each dropout value (from config):
       - Train fresh model
       - Evaluate on val
     - Pick best

  4. Step 3 — Confirm on test

Input: same as Level 2

Output:
  - data/reports/{model_name}_architecture_sweep.json
  - Best config logged (user updates params.yaml manually)

MLflow:
  - Experiment: "{model_name}_architecture_sweep"

Reusable: same script works for any model.
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

Model-agnostic stages — replace `{model}` with actual model name (e.g., `ml_v3`):

```yaml
  sweep_thresholds_{model}:
    cmd: cmd /c "set PYTHONPATH=src && python -m engine.sweep_thresholds --model {model}"
    deps:
      - models/{MODEL_DIR}/         # model ONNX + scaler
      - data/raw/BTCUSDT_15m_ohlcv.parquet
      - data/raw/BTCUSDT_1m_ohlcv.parquet
      - src/engine/sweep_thresholds.py
    params:
      - configs/params.yaml:
          - {model}.sweep
    outs:
      - data/reports/{model}_threshold_sweep.json

  sweep_training_{model}:
    cmd: cmd /c "set PYTHONPATH=src && python -m engine.sweep_training_params --model {model}"
    deps:
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/reports/{model}_threshold_sweep.json
      - src/engine/sweep_training_params.py
    params:
      - configs/params.yaml:
          - {model}.sweep
    outs:
      - data/reports/{model}_training_sweep.json

  sweep_architecture_{model}:
    cmd: cmd /c "set PYTHONPATH=src && python -m engine.sweep_architecture --model {model}"
    deps:
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/reports/{model}_training_sweep.json
      - src/engine/sweep_architecture.py
    params:
      - configs/params.yaml:
          - {model}.sweep
    outs:
      - data/reports/{model}_architecture_sweep.json

  train_{model}_final:
    cmd: cmd /c "set PYTHONPATH=src && python -m engine.train_v3_final --model {model}"
    deps:
      - data/features/direction_prediction/feature_cache.parquet
      - data/features/direction_prediction/exit_aware_labels.parquet
      - data/reports/{model}_architecture_sweep.json
    outs:
      - models/{MODEL_DIR}/
```

Note: DVC doesn't support template variables natively. When adding stages
for a specific model, replace `{model}` with the actual name (e.g., `ml_v3`)
and `{MODEL_DIR}` with the directory (e.g., `ML_V3_staging`).

---

## Scripts to Create

All scripts are model-agnostic via `--model` flag:

| Script | Level | Flag | Reads config from |
|--------|:-----:|------|------------------|
| src/engine/sweep_thresholds.py | 1 | --model ml_v3 | params.yaml → ml_v3.sweep.conf_long/short |
| src/engine/sweep_training_params.py | 2 | --model ml_v3 | params.yaml → ml_v3.sweep.learning_rate/loss_weights/batch_size |
| src/engine/sweep_architecture.py | 3 | --model ml_v3 | params.yaml → ml_v3.sweep.hidden_size/temperature/dropout |
| src/engine/train_v3_final.py | Final | --model ml_v3 | params.yaml → ml_v3.training (locked best values) |

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
