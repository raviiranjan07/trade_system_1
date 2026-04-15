# L2-003: Magnitude Gate

**Date:** 2026-02-28
**Timeframe:** 15-min BTCUSDT
**Train:** 2020-2023 | **OOS:** 2024-2025

## Objective

Determine which bars are worth trading — direction-agnostic.

**Question**: "Does this bar have enough OPPORTUNITY to be worth trading?"

**Gate Input:** A bar (with all its feature values)
**Gate Output:** YES (trade this bar) or NO (skip this bar)

---

## How the Gate Learns

### Input Features (lookback — what the gate uses to decide)

695 magnitude features computed on each bar. These capture past + present market conditions.

### Labels (lookahead — used to TRAIN the gate, NOT used by the gate itself)

For each bar, compute **MFE (Max Favorable Excursion)** in EITHER direction over the next N bars:

- **max_up**: highest upward move over next N bars (in bps)
- **max_down**: highest downward move over next N bars (in bps)
- **MFE = max(max_up, max_down)** — whichever direction moved more

**Lookahead horizons** (same as lookback horizons from L2-001c):
[1, 2, 3, 5, 10, 20, 32, 96] bars

**Bar labeling:**
- MFE >= 12bps → **GOOD bar** (enough movement to profit — Rule #1)
- MFE < 12bps → **BAD bar** (noise, not enough movement)

**Key principle:** Lookahead data is ONLY used during training to label bars. In live trading, the gate only uses lookback features (no peeking at the future).

---

## Input Features: 695 Magnitude Features

### Continuous base features (14) — threshold gates + derivatives

| Category | Features | Count |
|----------|----------|-------|
| Volatility | keltner_width, std20, donchian_width, atr_percentile | 4 |
| Bar Size | range_bps, body_bps | 2 |
| Position | dist_from_high20_pct, dist_from_low20_pct | 2 |
| Trend | ema_separation | 1 |
| Volume | volume_trend, volume_ratio | 2 |
| Microstructure | ll_count5, down_bars5, hh_count5 | 3 |

### Binary/categorical base features (9) — equality gates, no derivatives

| Feature | Gate logic |
|---------|-----------|
| price_above_sma200 | == 1 (bull) or == 0 (bear) |
| rsi_extreme_oversold | == 1 or == 0 |
| rsi_oversold_zone | == 1 or == 0 |
| is_weekend | == 1 or == 0 |
| session_asia_night | == 1 or == 0 |
| session_europe | == 1 or == 0 |
| session_us | == 1 or == 0 |
| day_of_week | == specific day (0-6) |
| session | == specific session code |

### Derivatives (672)

14 continuous magnitude features × 6 types × 8 lookbacks = 672

- Types: slope, acceleration, zscore, percentile_rank, dist_from_max, dist_from_min
- Lookbacks: [1, 2, 3, 5, 10, 20, 32, 96] bars

### Total: 23 base + 672 derivatives = 695 magnitude features

---

## Rule #1: Minimum Profitable Move = 12bps

- Fees: 8bps (round-trip, limit orders)
- 12bps minimum move to profit (8bps fees + 4bps net)
- This is proven from economics, NOT an assumption
- Any bar where MFE < 12bps = impossible to profit from

## Rule #2: Tradeable Move Threshold = 15bps

- 15bps minimum move for meaningful profit (8bps fees + 7bps net)
- Used for Stage 2 direction labeling
- Which direction hits 15bps FIRST determines the label
- 4 labels: LONG, SHORT, BOTH, SKIP

**Label logic (per horizon H):**
- Entry = open of next bar after current bar closes
- Scan future bars one by one (bar i+1, i+2, ..., i+H):
  - `first_up_bar` = first bar where `high >= entry + 15bps`
  - `first_down_bar` = first bar where `low <= entry - 15bps`
- **LONG**: first_up_bar < first_down_bar (up hit 15bps first)
- **SHORT**: first_down_bar < first_up_bar (down hit 15bps first)
- **BOTH**: first_up_bar == first_down_bar (same bar hit 15bps in both directions)
- **SKIP**: neither direction hits 15bps within H bars

---

## Methodology

### Stage 1: Single-Feature Screening (TRAIN only)

**Objective:** Find which individual features separate GOOD bars from BAD bars.
Test each feature alone. No combinations. Understand each feature's power independently before combining them.

Each of 695 features tested ALONE — one at a time. No combos in Stage 1.

**Step 1: Label every TRAIN bar**
- Compute MFE = max(max_up, max_down) over next H bars
- H = [1, 2, 3, 5, 10, 20, 32, 96] bars (same as lookback horizons)
- MFE = the MAXIMUM bps price reached at any point during the next H bars (peak, not close)
- MFE >= 12bps → GOOD bar (Rule #1)
- MFE < 12bps → BAD bar
- Each bar keeps its date, time, feature values, and label — nothing moves

**Step 2: For each feature, measure separation between GOOD and BAD bars**

How: **Binning** — group bars by feature value, count GOOD% per group.

For **continuous features** (14 base + 672 derivatives = 686):
- Take all ~140,000 TRAIN bars
- Group them into 10 equal-sized groups (deciles) based on that feature's value
  - Group 1: bars with lowest 10% of feature values (~14,000 bars)
  - Group 2: next 10%
  - ...
  - Group 10: bars with highest 10% of feature values
  - Group boundaries come from the DATA (not chosen by us)
- For each group: count what % of bars are GOOD
- If GOOD% changes across groups → feature separates GOOD from BAD → useful
- If GOOD% is the same across all groups → feature doesn't help → useless for gate
- A feature might be useful at some horizons and useless at others

For **binary features** (is_weekend, session_europe, price_above_sma200, etc.):
- Only 2 values: 0 or 1
- Compare GOOD% when feature == 1 vs GOOD% when feature == 0
- No binning needed — just two groups

For **categorical features** (day_of_week has 7 values, session has 6 values):
- Test each value separately as binary: "is it Monday?" (yes/no), "is it Tuesday?" (yes/no), etc.

**Step 3: Find natural thresholds**
- For each useful feature, find where GOOD% jumps most between groups
- The feature value at that group boundary = the natural threshold
- Data decides: the cutoff value, the direction (higher=better or lower=better), and which features matter

**Step 4: Rank features**
- Which features have the strongest separation between GOOD and BAD bars?
- Metric: **spread** = max GOOD% across groups minus min GOOD% across groups
- Large spread = feature strongly separates GOOD from BAD
- Small spread = feature doesn't help
- Top features = best gate candidates

**Note on L2-001a validation:**
- L2-001a validated base features for "how FAR does price move?" (MFE Q4/Q1 ratio)
- L2-003 validates for "does price move AT LEAST 12bps?" — different question
- L2-003 is also the first real validation for the 672 derivatives (never tested before)

### Stage 2: Build the Unified Decision Brain (TRAIN only, validate on OOS)

**Objective:** Train 7 LightGBM models that read the full market journey (11,178 input columns) and learn which market states resolve as LONG / SHORT / SKIP. Model learns HOW STATES RESOLVE from historical evidence — not prediction.

---

## INPUT

#### Total Features Given to Brain

| Feature Set | Base | Derivatives | Total |
|-------------|------|-------------|-------|
| Magnitude (HOW FAR) | 23 | 672 (14 × 6 types × 8 lookbacks) | 695 |
| Direction (WHICH WAY) | 11 | 528 (11 × 6 types × 8 lookbacks) | 539 |
| **All features** | **34** | **1,200** | **1,234** |

At each snapshot, we also capture:
- OHLCV: 5 values
- Time (hour_ist, day_of_week, month): 3 values — IST = UTC+5:30

**Per snapshot: 1,234 + 5 + 3 = 1,242 values**

**9 snapshots × 1,242 = 11,178 columns per bar** → one row in training table

Snapshots: [t-96, t-32, t-20, t-10, t-5, t-3, t-2, t-1, t]
Column naming: `t96_{feature}`, `t32_{feature}`, ..., `t1_{feature}`, `t_{feature}`

---

## TEACHER (Outcome Labels — from Lookahead)

7 structured outcome metrics computed from lookahead data (historical truth):

| Label | Type | Definition |
|-------|------|-----------|
| direction | multiclass (0/1/2) | LONG=0 if max_up > max_down AND max_up >= 12bps; SHORT=1 if max_down > max_up AND max_down >= 12bps; else SKIP=2 |
| mfe_up_bps | regression | Max upward move over next 96 bars (entry = next bar open) |
| mfe_down_bps | regression | Max downward move over next 96 bars |
| time_to_peak | regression | Horizon [1,2,3,5,10,20,32,96] where peak MFE was achieved |
| persistence | regression | Count of 8 horizons where direction matches primary (0-8) |
| vol_expansion | binary (0/1) | ATR at time_to_peak bar > ATR at current bar |
| volume_expansion | binary (0/1) | Volume at time_to_peak bar > volume MA at current bar |

These are **historical facts** — computed from real data, used as labels (teacher), NEVER as model input in live trading.

---

## 7 LightGBM Models

One model per label. All share the same 11,178 input columns.

| Model | Objective | Primary metric |
|-------|-----------|---------------|
| direction | multiclass (3 classes) | OOS accuracy |
| mfe_up_bps | regression | OOS MAE |
| mfe_down_bps | regression | OOS MAE |
| time_to_peak | regression | OOS MAE |
| persistence | regression | OOS MAE |
| vol_expansion | binary | OOS AUC |
| volume_expansion | binary | OOS AUC |

**Common LightGBM params:**
- n_estimators: 1000, learning_rate: 0.05, num_leaves: 63
- feature_fraction: 0.1 (each tree sees ~1,118 of 11,178 features)
- bagging_fraction: 0.8, early_stopping_rounds: 50
- reg_alpha: 0.1 (L1), reg_lambda: 0.1 (L2)

---

## File Structure

```
experiments/layer2/L2-003/
├── PLAN.md               (this file)
├── learning.md           (conceptual explanation)
├── L2_003_stage1.py      (DONE)
├── L2_003_stage2.py      (Stage 2 script)
└── models/
    ├── direction.txt
    ├── mfe_up.txt
    ├── mfe_down.txt
    ├── time_to_peak.txt
    ├── persistence.txt
    ├── vol_expansion.txt
    └── volume_expansion.txt
```

---

## Expected Output

### Stage 1 (DONE)
- Feature ranking by separation strength (GOOD vs BAD bars)
- Natural thresholds from data

### Stage 2
- 7 trained LightGBM model files (`.txt`)
- `L2_003_stage2_results.csv` — train vs OOS metrics per model
- `L2_003_stage2_feature_importance.csv` — top features by gain
- Direction model: target >60% OOS accuracy (baseline: 33% random)
- Feature importance: which snapshots and feature types matter most

---

## Stage 2 Conclusion: Why LightGBM Was Not Ideal

**Result:** 97.9% train, 50.5% OOS — failed.

**Three reasons LightGBM is wrong for this problem:**

1. **Cannot learn sequences** — flattens 9 snapshots into 11,124 flat numbers, loses the time order. Can't learn "feature trajectory A → B → C leads to price crossing 15bps UP"

2. **Cannot learn multiple outputs together** — separate model per target, each learns alone, no shared understanding between direction/magnitude/timing

3. **Requires manual combination logic** — with separate models, humans must write rules to combine outputs, reintroducing the same bias ML was supposed to remove

**Direction label was also wrong:**
- H=96 too wide — both directions cross 12bps, label is near-random
- Picked LARGER move, not FIRST move — not tradeable
- Rule #2 (15bps first-hit with LONG/SHORT/BOTH/SKIP) is the corrected label

**Next step:** Sequential neural network with Rule #2 labels.

---

### Stage 3: RNN Decision Brain (NEXT)

**Objective:** Train a vanilla RNN that reads the 9-snapshot feature sequence and learns how feature trajectories resolve into price movements.

#### Labels (21 targets — lookahead, never used in live)

| Label | Type | Count | Definition |
|-------|------|-------|-----------|
| `mfe_up_1,2,3,5,10,20,32,96` | regression | 8 | Max upward move from entry within H bars (bps) |
| `mfe_down_1,2,3,5,10,20,32,96` | regression | 8 | Max downward move from entry within H bars (bps) |
| `time_to_first_up_hit` | regression | 1 | First bar where `high >= entry + 15bps` (0 if never within 96) |
| `time_to_first_down_hit` | regression | 1 | First bar where `low <= entry - 15bps` (0 if never within 96) |
| `direction` | multiclass (0/1/2/3) | 1 | LONG=0, SHORT=1, BOTH=2, SKIP=3 — derived from first-hit timing |
| `vol_expansion` | binary (0/1) | 1 | ATR at peak bar > ATR at current bar |
| `volume_expansion` | binary (0/1) | 1 | Volume at peak bar > volume MA at current bar |

**Entry:** `open[i+1]` — next bar's open price. All labels measured from this price.

**Direction label logic (Rule #2):**
- `time_to_first_up_hit < time_to_first_down_hit` (both > 0) → LONG
- `time_to_first_down_hit < time_to_first_up_hit` (both > 0) → SHORT
- both > 0 and equal → BOTH
- both == 0 → SKIP

#### Input Shape

```
(batch, 9, 1236)
```
- 9 timesteps: [t-96, t-32, t-20, t-10, t-5, t-3, t-2, t-1, t]
- 1236 per timestep: 1234 features + hour_ist + day_of_week
- Sequence preserved — RNN sees feature trajectories, not flat numbers

#### Architecture

```
Input: (batch, 9, 1236)
  → RNN(hidden=128, layers=1, batch_first=True)
  → Last hidden state: (batch, 128)
  → Dropout(0.3)
  → Dense(256) → ReLU
  → Dense(128) → ReLU
  → 21 output heads:
      mfe_up_1,2,3,5,10,20,32,96    → Dense(1) each  [regression × 8]
      mfe_down_1,2,3,5,10,20,32,96  → Dense(1) each  [regression × 8]
      time_to_first_up_hit          → Dense(1)        [regression]
      time_to_first_down_hit        → Dense(1)        [regression]
      direction                     → Dense(4) → Softmax  [multiclass: LONG/SHORT/BOTH/SKIP]
      vol_expansion                 → Dense(1) → Sigmoid  [binary]
      volume_expansion              → Dense(1) → Sigmoid  [binary]
```

#### Training

- Framework: PyTorch, run on Google Colab T4 GPU
- Optimizer: Adam, lr=0.001
- Loss: MSELoss (regression) + BCELoss (binary) — summed equally
- Early stopping: patience=10, monitor validation loss
- Max epochs: 200, batch size: 512
- Data split: Train 2020-2023 | Val 2024 | OOS/Test 2025
- Normalization: z-score per feature, fit on train only

#### Files

```
experiments/layer2/L2-003/
├── L2_003_stage3_labels.py      ← Step 1: compute 7 labels, save labels.parquet
├── L2_003_stage3_train.py       ← Steps 2-4: normalize, build RNN, train
├── L2_003_stage3_eval.py        ← Step 5: OOS eval, derive direction
├── labels.parquet               ← output of step 1
└── models/
    └── rnn_model.pt             ← saved PyTorch model
```

#### Success Criteria

- Direction accuracy OOS > 60% (baseline: 33% random)
- LONG bars predicted higher mfe_up than SKIP bars
- Train vs OOS gap < 10pp
- No data leakage (normalization fit on 2020-2023 train only)

---

L2-003 brain → L2-004 Exit Gate (when to close the trade)

---

## Status Update (2026-03-24)

### DEPRECATED: Magnitude Gate (Original Objective)

The original "Magnitude Gate" concept — predicting if a bar has enough opportunity (MFE >= 15bps) — is **DEPRECATED**.

**Reason:** The V1.5 ML direction model already handles this implicitly. When the model is confident (prob > 0.60 or < 0.35), the underlying features are at extreme values (large ROC, extreme RSI, low range_position) — which naturally correlates with high magnitude bars. When features are normal, the model outputs ~0.50 and we skip. The confidence threshold IS the magnitude gate.

A separate magnitude gate would be redundant with the ML model's confidence score.

### What Was Completed

| Item | Status | Result |
|------|--------|--------|
| Stage 1: Feature screening | DONE | 695 features ranked by GOOD/BAD separation |
| Stage 2: LightGBM | DONE, FAILED | 97.9% train, 50.5% OOS — massive overfit |
| Stage 3: Direction prediction | DONE (V1.5) | MLP 10 feat, 57-58% conf acc, deployed to production |
| Direction label bug fix | DONE | SKIP mislabeled — 23K bars fixed |
| LSTM/GRU investigation | DONE | Memory doesn't help — temporal signal only 1-2% |
| V1.4 filter test | DONE | ML agrees=62.9% win, disagrees=48.9% — filter works |
| Standalone backtest | DONE | +6,096 bps (H96), +3,815 bps (H8) |
| V1.5 integration | DONE | ML_LONG/ML_SHORT in bot, separate wallet, dashboard |

### What Was Left Behind (for future work)

| Item | Status | Note |
|------|--------|------|
| Magnitude Gate | DEPRECATED | ML model confidence handles this |
| MFE prediction | ABANDONED | Worked (pred_mean ≈ actual_mean) but not integrated — could set dynamic exits |
| Time-to-peak prediction | NOT DONE | Could inform hold duration |
| Multi-task brain (21 outputs) | ABANDONED | Direction failed, other outputs worked but unused |
| Support/resistance features | NOT DONE | Planned for next iteration |
| Price reaction probability map | NOT DONE | Data shows 62% LONG after crash+RSI<20 — systematic model not built |
| L2-004 Exit Gate | NOT STARTED | Next phase — smarter exits than fixed trailing stop |

### Key Findings (see memory/l2_challenges.md for full details)

- Direction accuracy ceiling: 57-58% on confident bars with OHLCV features
- MLP > LSTM/GRU for indicator-based features (indicators already encode temporal info)
- Technical indicators all derive from same OHLCV source — ceiling is in the data
- 95.7% of bars have opposite-direction twin (same features, opposite outcome)
- Temporal signal adds only 1-2% beyond single-bar features
- Weight decay blocks LSTM gate learning; without it, gates shut down memory
- sma200_dist_pct normalization bug found and fixed (NaN contamination)
- roc1 + roc2 carry 45% of directional signal; range_position 17%; rsi7 6%

### Production Model (V1.5)

```
Architecture: MLP 10 → 128 → 128 → 1
Features: roc1-roc8 + range_position + rsi7
Label: H96 binary direction (which ±15bps hits first)
Thresholds: LONG > 0.60, SHORT < 0.35
Backtest (2024-2025): 439 trades, +11,096 bps, PF 2.44
ML_LONG: 220 trades, 73.6% win, +7,035 bps
Live paper trading since 2026-03-19
Paper results (as of Mar 24): 4 trades, 3W 1L, +140.5 bps, wallet $5→$6.98
```

---

## Stage 4: Controlled Learning Experiments (NEXT)

**Problem:** The model predicts mfe_up ≈ mfe_down for every bar (can't distinguish directional asymmetry). All architectures (MLP, LSTM, GRU) and all feature combinations converge to ~57-58% confident accuracy. The model learns magnitude but not which direction is stronger.

**Goal:** Control HOW the model learns to force it to find directional patterns.

**Setup:** 4 diff features (roc, rsi, rp, sma200) × 8 lookbacks = 32 inputs. H8 labels (MFE at H1-H8 + direction_h8). Multi-task with connected architecture (direction sees MFE predictions).

### Experiment 4A: Curriculum Learning
Train on easy bars first, gradually add harder ones.
```
Phase 1: Only bars where |mfe_up - mfe_down| > 50 bps (clear direction, ~20% of data)
Phase 2: Add bars with |diff| > 30 bps
Phase 3: Add bars with |diff| > 15 bps
Phase 4: All bars
```
**Hypothesis:** Model learns clear directional patterns first, then refines. Currently it averages across all bars and the clear patterns get diluted by noise.

### Experiment 4B: Attention Mechanism
Add attention layer on top of LSTM hidden states.
```
LSTM processes 8 steps → 8 hidden states [h1, h2, ..., h8]
Attention: learn which steps matter for each bar
→ weighted combination → MFE + direction heads
```
**Hypothesis:** Different bars need different lookback emphasis. Crash bars need recent steps, trending bars need all steps. Fixed last-hidden-state approach misses this.

### Experiment 4C: Separate Encoders with Comparison
Force the model to explicitly compare up vs down.
```
shared hidden → encoder_up → mfe_up prediction
shared hidden → encoder_down → mfe_down prediction
[encoder_up_output, encoder_down_output] → comparison layer → direction
```
**Hypothesis:** Current architecture has independent mfe_up and mfe_down heads that don't interact. A comparison layer forces the model to learn the RELATIONSHIP between them.

### Experiment 4D: Contrastive Learning
Train the model to produce DIFFERENT representations for LONG vs SHORT bars.
```
Take pairs: bar_A (strong LONG) and bar_B (strong SHORT)
Loss: representations of A and B must be far apart
      representations of similar-direction bars must be close
```
**Hypothesis:** Currently the model maps LONG and SHORT bars to similar hidden representations (because features are similar). Contrastive loss forces separation in representation space.

### Experiment 4E: Custom Asymmetry Loss
Penalize the model for predicting mfe_up ≈ mfe_down when they're actually different.
```
Standard MSE: (pred_up - actual_up)² + (pred_down - actual_down)²
Add penalty: if |actual_up - actual_down| > 20bps AND |pred_up - pred_down| < 5bps → extra loss
```
**Hypothesis:** MSE treats pred_up=55, pred_down=55 as acceptable when actual is up=80, down=20 (total error is moderate). The asymmetry penalty specifically targets the directional failure case.

### Experiment Order
1. 4E (Custom Asymmetry Loss) — simplest change, one line in loss function
2. 4A (Curriculum Learning) — data ordering change, no architecture change
3. 4C (Separate Encoders) — architecture change, tests comparison mechanism
4. 4B (Attention) — architecture change, tests dynamic step weighting
5. 4D (Contrastive) — most complex, requires pair sampling

### Success Criteria
- Direction confident accuracy > 60% on OOS (current: 57-58%)
- MFE-derived direction > 52% (current: 50%)
- pred_mfe_up ≠ pred_mfe_down when actual differs (current: pred_up ≈ pred_down always)
- Train-test gap < 10% on confident accuracy

### Stage 4 Results (2026-03-26) — COMPLETED

**None of the experiments broke the 58% ceiling or achieved MFE direction > 50%.**

| Experiment | Dir conf | MFE dir | Notes |
|------------|----------|---------|-------|
| 4E: Asymmetry Loss | 57.1% | 50.4% | Penalty doesn't create signal |
| 4A: Curriculum (loose) | 57.6% | 50.0% | Easy patterns unlearned when noise added |
| 4A: Curriculum (strict) | 54.9% | 49.9% | Too few bars in Phase 1 |
| 4C: Separate Encoders | 56.5% | 50.0% | Same input → same encoders |
| 4B: Attention temp=0.5 | 57.8% | 50.1% | Best balance (1814 bars) |
| 4B: Attention temp=0.05 | 58.8% | 50.3% | Best accuracy (816 bars) |

**Key insights:**
- Attention is the only technique that showed improvement (+1-2%)
- LONG attends to medium-term steps (5-7), SHORT to short-term (1-2)
- Attention weights stay near uniform — the model can't find strongly different patterns per step
- All techniques are VALIDATED tools, not failures — they may work with different data sources
- The bottleneck is confirmed: OHLCV-derived features lack directional asymmetry information
- 4D (Contrastive Learning) not tested — saved for future work with new data

**Conclusion:** Stage 4 confirms the direction accuracy ceiling at 57-58% with OHLCV features.
Next steps should focus on:
1. New data sources (funding rate, open interest, liquidations)
2. Layer 3: Adaptive Exit (use MFE predictions for dynamic stops)
3. Support/resistance features
4. Price reaction probability map
