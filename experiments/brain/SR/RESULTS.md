# S/R Brain Experiments — Results & Analysis

## Stage 1: S/R Zone Features → FAILED

**Setup:** Biggest gap clustering (V5), 8-bar then 25-bar lookback.
**Features:** 13 raw → 16 zone-relative → 11 deduplicated.
**Labels tested:** Binary (first-hit 15bps), score-based (MFE-MAE dominance), 3-class (bounce/break/chop).
**Models tested:** XGBoost, MLP, LSTM.

**Results:**
- All models ~50% test accuracy
- Features show 0.2-4% separation between bounce and break
- No model could learn — XGBoost, MLP, LSTM all failed

**Key findings:**
- Label was noisy — first-hit captures micro-moves, not real bounces
- Switched to score-based (MFE-MAE dominance) — still 50%
- Zone structure features look identical for bounce and break bars

**Conclusion:** S/R zone structure features alone cannot predict bounce vs break.

---

## Stage 2: Level Memory → FAILED

**Setup:** Zone registry tracking 2,263 zones with bounce/break/chop history.
**Features:** 6 static memory + 11 dynamic = 17 total (later reduced to 15).
**Touch detection:** Entry-only (first bar of each visit).
**Label:** 3-class (MFE>0.70, MAE>0.70, else chop) + 15bps minimum.

**Results:**
- 10,500 events with biggest gap detection
- Memory features show 0.1-4% separation
- Model accuracy: 32.8% (random for 3-class)
- Zone memory doesn't predict — past bounce/break ratio doesn't predict next outcome

**Bugs found & fixed:**
- Zone death was killing 89% of events → removed (deactivate/reactivate instead)
- Entry-only detection key used zone.id only → fixed to (zone.id, role)
- Event gap = 1 (consecutive bars from different zones) → identified as data quality issue

**Conclusion:** Zone memory doesn't help. A zone that bounced 80% before is equally likely to bounce or break next time.

---

## Stage 3: KDE S/R Detection → VALIDATED

**Problem:** Biggest gap (V5) produced poor zones — below random bounce rate.

### Bounce Rate Comparison (all methods, 100-bar lookback, bw=0.03)

| # | Method | Support | Bars | Resistance | Bars |
|---|--------|---------|------|------------|------|
| 1 | Biggest Gap (V5) | 48.3% | 468 | 44.7% | 568 |
| 2 | Plain KDE | 52.8% | 549 | 55.8% | 692 |
| 3 | Swing + KDE | 50.2% | 265 | 55.6% | 349 |
| 4 | Recency weighted KDE | 52.6% | 656 | **57.5%** | **803** |
| 5 | Reaction weighted KDE | **55.2%** | 210 | 58.2% | 306 |
| 6 | Combined (rec × react) | 57.5% | 233 | 56.3% | 318 |
| 7 | Plain + Recency | 52.2% | 575 | 57.3% | 721 |
| 8 | All three | 57.4% | 235 | 56.5% | 324 |
| 9 | **Hybrid (react sup + recency res)** | **55.2%** | **210** | **57.5%** | **803** |
| - | Baseline | 50% | - | 50% | - |

### Hybrid KDE chosen: Reaction for support + Recency for resistance
- Support: where buyers stepped in STRONGLY (reaction)
- Resistance: where price was RECENTLY rejected (recency)

### Why Hybrid wins
- Recency support weak (52.6%) — recent lows in downtrend are not real support
- Reaction support strong (55.2%) — strong bounces = real buyers
- Recency resistance strong (57.5%) — recent highs = fresh rejection levels

### Lookback Comparison (Hybrid KDE)

| Lookback | Support | Bars | Resistance | Bars |
|----------|---------|------|------------|------|
| 25 bars  | **55.8%** | **678** | 57.0% | **1450** |
| 100 bars | 55.2% | 210 | **57.5%** | 803 |

25-bar gives 3x more signals with nearly same accuracy.

### Bandwidth Testing
- Tested: 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15
- Auto (Scott/Silverman) too smooth — merges everything
- 0.03 best — precise zones without over-splitting

### Zone Registry with KDE
- 2,263 zones (vs 394 with biggest gap)
- Avg 14.4 touches per zone
- 1,253 zones with 10+ touches
- 0% bars without S/R (vs 11.1% before)

### Note on bounce rate measurement
Above bounce rates use SIMPLE label (next bar close up/down).
This is NOT the same as the MFE-based label used for training.

---

## Stage 4: Rebuild with KDE → Feature Separation Still Weak

### Entry-only mode (run_stage4.py)
- 31,410 events
- Feature separation: 0.2-7.3% (all features)
- Bounce vs Break rate: 50.2%

### Every-bar mode (sr_dataset.py)
- 67,047 events out of 210,000 bars (32% at S/R)
- Train: 32,817 | Val: 8,877 | Test: 23,586
- BOUNCE: 21,522 (32.1%) | BREAK: 21,658 (32.3%) | CHOP: 23,867 (35.6%)
- Bounce vs Break rate: 49.8%

### Feature Separation (every_bar, 67K events)

**Dynamic:**

| Feature | Bounce | Break | Chop | MaxDiff% |
|---------|--------|-------|------|----------|
| dist_to_zone_pct | 0.113 | 0.113 | 0.113 | 0.2% |
| support_width_pct | 0.025 | 0.025 | 0.026 | 2.3% |
| res_width_pct | 0.027 | 0.028 | 0.028 | 3.2% |
| support_retest | 1.674 | 1.726 | 1.693 | 3.0% |
| resistance_retest | 2.574 | 2.615 | 2.663 | 3.4% |
| zone_width | 5.451 | 5.454 | 5.473 | 0.4% |
| recovery_up_pct | 0.315 | 0.314 | 0.317 | 1.0% |
| recovery_down_pct | 0.192 | 0.190 | 0.199 | 4.9% |
| speed_short | 0.165 | 0.162 | 0.160 | 3.1% |
| speed_mid | 0.199 | 0.200 | 0.192 | 4.3% |
| speed_long | 0.207 | 0.207 | 0.210 | 1.5% |

**Static:**

| Feature | Bounce | Break | Chop | MaxDiff% |
|---------|--------|-------|------|----------|
| bounce_ratio | 0.386 | 0.381 | 0.378 | 2.0% |
| recent_bounce_ratio | 0.505 | 0.514 | 0.491 | 4.4% |
| pressure | 1.325 | 1.369 | 1.366 | 3.2% |
| bars_since_touch | 2.754 | 2.784 | 2.678 | 3.8% |
| touch_count_scaled | 2.256 | 2.333 | 2.262 | 3.3% |
| level_type_binary | 0.374 | 0.408 | 0.407 | 8.4% |

**All features under 8.4% separation. Target: >10-20% for model to learn.**

### Key Finding: Label Mismatch
- KDE bounce test (simple label): 55.8% bounce rate
- MFE label (25 bars, 70%, 15bps): 49.8% bounce rate
- The short-term edge (55.8%) doesn't persist over 25 bars with strict thresholds
- Label parameters may need tuning

---

---

## Stage 5: Label Validation → NO IMPROVEMENT

**Objective:** Test if different label parameters improve feature separation.

### Test Matrix (12 combinations)
- Horizons: 5, 10, 15, 25 bars
- Dominance: 60%, 70%, 80%
- Min move: 15bps (fixed)
- Same 32,817 events (train split)

### Results

| Horizon | Dominance | Bounce | Break | Rate | Max Sep | Avg Sep | Best Feature |
|---------|-----------|--------|-------|------|---------|---------|-------------|
| 5 | 0.60 | 13,577 | 13,248 | 50.6% | 8.0% | 1.9% | level_type_binary |
| 5 | 0.70 | 10,591 | 10,257 | 50.8% | 8.4% | 2.1% | level_type_binary |
| 5 | 0.80 | 7,361 | 7,192 | 50.6% | 9.1% | 2.3% | level_type_binary |
| 10 | 0.60 | 13,559 | 13,346 | 50.4% | 8.1% | 1.7% | level_type_binary |
| 10 | 0.70 | 10,586 | 10,308 | 50.7% | 8.3% | 2.0% | level_type_binary |
| 10 | 0.80 | 7,427 | 7,260 | 50.6% | 8.8% | 2.3% | level_type_binary |
| 15 | 0.60 | 13,532 | 13,409 | 50.2% | 7.1% | 1.5% | level_type_binary |
| 15 | 0.70 | 10,547 | 10,371 | 50.4% | 9.1% | 2.1% | level_type_binary |
| 15 | 0.80 | 7,512 | 7,315 | 50.7% | 8.3% | 2.4% | level_type_binary |
| 25 | 0.60 | 13,363 | 13,325 | 50.1% | 6.2% | 1.2% | level_type_binary |
| 25 | 0.70 | 10,486 | 10,389 | 50.2% | 8.4% | 1.9% | level_type_binary |
| 25 | 0.80 | 7,474 | 7,381 | 50.3% | 8.6% | 2.7% | level_type_binary |

### Key Findings
- All 12 combinations: bounce rate 50.1-50.8% (essentially 50/50)
- Max separation: 6.2-9.1% (all below 10% target)
- Best feature is ALWAYS level_type_binary (support vs resistance)
- Average separation across all features: 1.2-2.7%
- No label parameter changes the result

### Conclusion
**The problem is not the label. The problem is the features.**
S/R zone features (structure + memory + speed) do not predict bounce vs break regardless of label design.

The only signal: support bounces slightly more than resistance (~8-9% separation on level_type_binary).

---

## Overall Conclusion

S/R features alone CANNOT predict bounce/break. Tested with:
- 5 S/R detection methods (V1-V5, KDE)
- 4 model architectures (XGBoost, MLP, LSTM, LSTM+Static)
- 3 label types (binary, score-based, 3-class)
- 12 label parameter combinations
- 2 touch modes (entry-only, every-bar)
- Zone memory (bounce_ratio, pressure, history)
- Multiple normalizations (z-score, zone-relative, clipped)

All results: ~50% accuracy, 0-9% feature separation.

---

## Stage 6: S/R + Base Features Combined

### Test 1: MLP with MFE 3-class label (stage6_combined.py)
- Features: 20 (11 S/R dynamic + 6 S/R static + 3 base: roc, rsi7, range_position)
- Label: MFE 3-class (bounce/break/chop, H25, 70% dominance, 15bps min)
- Result: **34.4%** (predicts all CHOP)
- ROC showed 179.8% separation — but inflated by near-zero values

### Test 2: End-to-end model with H25 direction label (stage6_train.py)
- Architecture: S/R(17) -> Dense(8) -> Dense(3) -> zone_context + base(3) -> Dense(32) -> Dense(4)
- Label: H25 direction (LONG/SHORT/BOTH/SKIP)
- End-to-end result: **45.0%**
- Base-only result: **44.7%**
- S/R improvement: **+0.3%** (negligible)
- Model biased toward SHORT (67% SHORT accuracy, 22% LONG)

### Test 3: Base features only with H25 binary label (test_base_h25.py)
- Features: 10 (roc1-8 + rsi7 + range_position) — same as V1.5
- Label: H25 binary (LONG/SHORT only, removed BOTH/SKIP)
- Model: LSTM, 8 snapshots
- Result: **50.9%** (random)
- V1.5 reference with H96: 57-58%

### Key Finding
Base features that got 57-58% with H96 get only 50.9% with H25.
The directional signal needs 96-bar horizon to be predictable.
H25 is too short — too noisy for direction prediction.

### Note on Test 3 setup
Used roc1-8 (V1.5 style) instead of single roc per snapshot.
Should have used 3 features (roc, rsi7, range_position) × 8 snapshots.
Need to retest with correct snapshot setup.

---

## Overall Conclusion

S/R features alone CANNOT predict bounce/break. Tested with:
- 5 S/R detection methods (V1-V5, KDE)
- 4 model architectures (XGBoost, MLP, LSTM, LSTM+Static)
- 3 label types (binary, score-based, 3-class)
- 12 label parameter combinations
- 2 touch modes (entry-only, every-bar)
- Zone memory (bounce_ratio, pressure, history)
- Multiple normalizations (z-score, zone-relative, clipped)
- Combined with base features (end-to-end model)

All S/R-only results: ~50% accuracy, 0-9% feature separation.
Combined S/R+base: +0.3% improvement (negligible).
H25 label itself is unpredictable with base features (50.9%).

---

## Stage 7: Connected LSTM + Multi-task

### V1.5 MLP Replication (test_base_h25.py)
- MLP 10->128->128->2, roc1-8 + rsi7 + range_position, H8 binary
- Overall: 52.1%, Confident >0.60: **58.2%**, >0.65: **61.1%**
- Confirms V1.5 still works. 57-58% is **confident accuracy**, not overall.

### LSTM with snapshots (test_base_h25.py)
- 8 snapshots × 3 features (roc, rsi7, range_position), H8 and H25
- LSTM(64) and LSTM(128+128+ReLU) both tested
- Result: **50.6-50.9%** — LSTM with snapshots can't learn
- roc1-8 as flat features works, roc per snapshot doesn't

### Connected LSTM + Multi-task baseline (test_connected_h25.py)
- 25 snapshots × 4 features (roc, rsi7, range_position, sma200_dist)
- Connected architecture: MFE predictions feed into direction head
- Multi-task: direction + MFE loss (1.0 × CE + 5.0 × MSE)
- H25 binary (LONG/SHORT)
- Overall: 51.9%, Confident >0.60: **56.7%**, >0.65: **60.1%**
- **Multi-task + connection makes LSTM learn** (vs 50.6% without)

### Combined Architecture (test_combined_arch.py)
- 3 paths: Base(LSTM 25×4) + S/R Dynamic(Dense 11→4) + S/R Static(6 direct)
- MFE connection + all paths combine at direction head (44 inputs)
- H25 binary

| Model | Overall | Conf >0.55 | Conf >0.60 | Conf >0.65 |
|---|---|---|---|---|
| Combined (S/R + base) | **52.5%** | 54.3% | 55.7% | 58.4% |
| Connected LSTM base-only | 51.9% | 53.9% | **56.7%** | **60.1%** |
| V1.5 MLP (H8) | 52.1% | 55.1% | **58.2%** | **61.1%** |

### Key Finding
- S/R improved overall accuracy slightly (+0.6%)
- But **hurt confident accuracy** (-1.0% at >0.60, -1.7% at >0.65)
- S/R adds noise at high confidence — base-only is better for confident predictions
- S/R features still don't help direction prediction

---

## Final Conclusion

After exhaustive testing across 7 stages:
- S/R zone features do NOT improve direction prediction
- S/R zones are real (55-57% simple bounce rate with KDE)
- But predicting WHICH direction at a zone is as hard as predicting anywhere else
- Base features (roc, rsi7, range_position, sma200_dist) carry the directional signal
- Connected LSTM + multi-task is the best architecture for learning from snapshots

---

## Stage 8: S/R Advisor Standalone (test_stage8.py)

### Architecture
Separate dynamic/static paths (first time tested):
```
DYNAMIC (9) -> Dense(9->4) -> ReLU -> 4 numbers
STATIC (6)  -> Dense(6->4) -> ReLU -> 4 numbers
Combine: 4+4=8 -> Dense(8->2) -> [bounce_score, break_score]
```

### Results

| Test | Accuracy | Bounce acc | Break acc |
|------|----------|-----------|-----------|
| 8a (no MFE) | **51.7%** | 75.8% | 25.9% |
| 8b (with MFE) | **51.7%** | 79.9% | 21.4% |
| Previous S/R best | ~50% | - | - |
| Target | >52% | - | - |

**51.7% is the best S/R-only result ever.** Separate paths extracted more signal than any previous approach. But model biased toward bounce (76-80%).

MFE head didn't improve accuracy (51.7% both).

### Deep Analysis

**By level type:**
- Support: 50.4% bounce (basically random)
- Resistance: 52.7% bounce (slight edge)

**By historical bounce_ratio:**
- Low history (0-25%): 50.1% bounce (random)
- High history (75-100%): 53.4% bounce (+3.4% edge)
- Zones with strong bounce history DO bounce more — small but real signal

**By pressure:**
- pressure=0: 52.3%
- pressure=3+: 51.3%
- No clear pattern

**Feature separation (resistance only):**
- speed_long: 5.5% diff (best dynamic)
- pressure: 6.1% diff (best static)
- bounce_ratio: 2.4%

**Feature separation (support only):**
- All features under 2.5% — nearly nothing

### Key Insights
1. Resistance has more signal than support
2. Zone bounce history (bounce_ratio) is the only feature with real predictive power
3. Support is nearly random — may need different approach
4. Separate paths architecture is better than dumping features together

---

## Stage 8c: Enriched Static Memory (14 features)

### Results

| Test | Static | MFE head | Accuracy | Bounce acc | Break acc |
|------|--------|----------|----------|-----------|-----------|
| 8a | 6 | No | 51.7% | 75.8% | 25.9% |
| 8b | 6 | Yes | 51.7% | 79.9% | 21.4% |
| 8c-A | 14 | No | 51.4% | 70.0% | 31.4% |
| 8c-B | **14** | **Yes** | **52.0%** | 86.4% | 14.9% |

8c-B hit 52.0% — first time above 52%. But model predicts bounce 86% of the time.

### Deep Analysis: Is the model really learning?

Checked every feature: "when value is high vs low, does bounce rate change?"

| Feature | Low value | High value | Max edge |
|---------|-----------|------------|----------|
| bounce_ratio | 50.1% | 53.4% | 3.3% |
| pressure=0 vs 3+ | 52.3% | 51.3% | 1.0% |
| avg_bounce > avg_break | 51.5% | 51.7% | 0.2% |
| Support vs Resistance | 50.4% | 52.7% | 2.3% |
| Last outcome bounce vs break | 51.7% | 51.5% | 0.2% |
| Low chop vs high chop | 52.8% | 52.3% | 0.5% |
| MFE trend positive vs negative | 52.1% | 51.4% | 0.7% |

**No feature changes bounce rate by more than 3.4%.**

### Conclusion
The 52.0% is NOT real learning. The model predicts bounce for almost everything because bounce is 52.3% of the data. It found the base rate, not a pattern.

The features genuinely don't predict bounce vs break:
- Every feature individually: 50-54% bounce rate regardless of value
- The maximum edge from any single feature: 3.3% (bounce_ratio high vs low)
- The model correctly found this and always predicts bounce — optimal strategy given the data

### Root Cause Analysis
The problem is not the features or the model. The problem is:
- S/R zone features describe WHERE price is
- They don't predict WHAT will happen
- Bounce/break at a zone depends on forces outside the zone (trend, momentum, volume)
- Current architecture processes features but doesn't guide the model on HOW to reason about them

### Next Direction
Hierarchical architecture — process features in the ORDER a trader thinks:
1. What zone am I at? (level_type)
2. How strong historically? (bounce_ratio, touch_count)
3. What's happening recently? (pressure, last_outcome, streak)
4. How is price approaching? (speed, distance)
5. Decide: bounce or break

## Open Questions
1. Test hierarchical architecture (guided reasoning order)
2. Whether S/R works as a **filter** (only trade at zones, use base model for direction)
3. Whether the 55.8% simple edge is tradeable after fees

---

## Stage 9A: Static Memory Ablation Diagnostics (2026-04-08)

### Goal
Isolate what signal exists in static memory before mixing larger feature groups.

### Dataset
- `datasets_stage9a_static`
- Thresholds: support `<= 0.10`, resistance `>= 0.90`
- Samples: Train `11,909`, Val `3,245`, Test `10,527`
- Label: binary `BREAK(0)` / `BOUNCE(1)` with 25-bar lookahead

### Position / Memory Ablation

| Feature Set | Overall | Bounce acc | Break acc | Pred bounce |
|---|---:|---:|---:|---:|
| position only (`price_position`) | 50.0% | 100.0% | 0.0% | 100.0% |
| memory only (baseline old) | 51.2% | 67.9% | 34.4% | 66.7% |
| position + memory | 49.7% | 93.7% | 5.6% | 94.0% |

**Conclusion:** `price_position` hurts this setup; memory carries weak signal, but still collapses toward bounce.

### Memory Feature Replacement Grid (memory-only)

| Memory Features | Overall | Bounce acc | Break acc | Pred bounce |
|---|---:|---:|---:|---:|
| `bounce_ratio,touch_count_scaled,recent_bounce_ratio,pressure` | 50.5% | 88.0% | 13.0% | 87.5% |
| `bounce_ratio,touch_count_scaled,recent_bounce_ratio,bounce_streak` | 50.2% | 84.1% | 16.3% | 83.9% |
| `bounce_ratio,bars_since_touch,recent_bounce_ratio,pressure` | 49.7% | 95.6% | 3.7% | 95.9% |
| `bounce_ratio,touch_count_scaled,chop_ratio,pressure` | 50.6% | 93.0% | 8.1% | 92.5% |
| `bounce_ratio,touch_count_scaled,last_outcome,bounce_streak` | 50.4% | 70.9% | 29.9% | 70.5% |

**Best memory-only (break discrimination):**
- `bounce_ratio + touch_count_scaled + last_outcome + bounce_streak`

### Minimal Trajectory Add-On

Using `memory_best = bounce_ratio,touch_count_scaled,last_outcome,bounce_streak`:

| Model | Overall | Bounce acc | Break acc | Pred bounce |
|---|---:|---:|---:|---:|
| memory only | 50.2% | 74.8% | 25.6% | 74.6% |
| memory + speed (`speed_short,mid,long`) | 50.5% | 70.4% | 30.5% | 70.0% |

**Conclusion:** minimal speed features give a small but real improvement in break detection and reduce bounce-collapse.

### Current Read
- Static memory alone is insufficient for this label (around ~50% overall).
- The failure mode is class-collapse toward bounce.
- `last_outcome` and `bounce_streak` are more useful than `recent_bounce_ratio` in the current setup.
- Adding a small trajectory block helps more than adding position in this formulation.
