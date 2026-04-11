# Brain S/R Experiments — Plan

## Overall Objective
Test if a model can learn to predict **bounce or break at support/resistance** on BTC 15-minute.

---

## Stage 1: S/R Zone Features Only

**Objective:** Can zone structure features alone predict bounce vs break?
**Features:** 13 raw S/R → 16 zone-relative → 11 deduplicated
**Labels:** Binary → score-based → 3-class
**Models:** XGBoost, MLP, LSTM
**Success:** > 55% accuracy
**Result:** FAILED — see RESULTS.md

---

## Stage 2: Level Memory

**Objective:** Does zone history (past bounces/breaks) predict next outcome?

### Zone Registry Schema
```
zone = {
    id, center, width, original_center,
    role: "support" or "resistance",
    touches, bounces, breaks,
    support_bounces, support_breaks,
    resistance_bounces, resistance_breaks,
    history: deque(maxlen=10),
    last_touch_bar, created_bar
}
```

### Zone Registry Rules
- **Matching:** Range overlap → tolerance floor (10bps) → closest center wins
- **Drift:** Smooth update (0.9 old + 0.1 new), cap at 0.5 × original width
- **No death:** Zones deactivate/reactivate (never permanently deleted)
- **Role flip:** Price-based (close crosses zone by threshold), no future info
- **Chop:** touches++ but bounces/breaks unchanged
- **Cold start:** Bayesian smoothing (bounces+1)/(touches+2), touch_count for trust

### Static Memory Features (6)
1. bounce_ratio — Bayesian smoothed
2. recent_bounce_ratio — last 5 touches
3. pressure — consecutive breaks from end
4. bars_since_touch — log(1 + bars)
5. touch_count_scaled — log(1 + touches)
6. level_type_binary — 1=support, 0=resistance

**Result:** FAILED — see RESULTS.md

---

## Stage 3: KDE S/R Detection

**Objective:** Replace biggest gap with KDE for better zone detection.

### Hybrid KDE (CHOSEN)
- **Support:** Reaction-weighted KDE on lows (strong bounces = real support)
- **Resistance:** Recency-weighted KDE on highs (recent rejections = fresh resistance)
- **Bandwidth:** 0.03
- **Lookback:** 25 bars

### Config
```yaml
sr_detection:
  method: hybrid_kde
  lookback: 25
  bandwidth: 0.03
  support_weight: reaction
  resistance_weight: recency
  peak_threshold: 0.2
```

**Result:** VALIDATED — 55.8% support, 57.0% resistance (simple label). See RESULTS.md

---

## Stage 4: Rebuild with KDE + Train

**Objective:** Rebuild dataset with KDE zones, check feature separation, train model.

### Current Features (15 total)

**Dynamic (9):**
1. dist_to_zone_pct
2. support_retest
3. resistance_retest
4. zone_width (log)
5. recovery_up_pct
6. recovery_down_pct
7. speed_short
8. speed_mid
9. speed_long

**Static (6):** bounce_ratio, recent_bounce_ratio, pressure, bars_since_touch, touch_count_scaled, level_type_binary

### Current Label
3-class: BOUNCE (fav_pct>0.70 AND fav>15bps), BREAK (adv_pct>0.70 AND adv>15bps), CHOP (else)
Horizon: 25 bars. Direction normalized for support vs resistance.

### Touch Modes
- **every_bar:** record every bar at zone (more samples, some duplication)
- **entry_only:** record first bar of each visit (fewer but independent samples)

### Data
- Both support and resistance touches
- Zone width >= 30bps
- Touch threshold: dist/zone_width <= 0.2
- Split: Train 2020-2022, Val 2023, Test 2024-2025
- Clip zone-relative features to [-3, 3]

### Datasets
- `datasets_entry_only/` — entry-only mode
- `datasets_every_bar/` — every-bar mode (with bar numbers + zone registry)

### Pipeline
1. Run hybrid KDE → find S/R zones
2. Run zone registry → track zones, memory
3. Compute 9 dynamic + 6 static features per touch
4. Create labels
5. Check feature separation (target > 10%)
6. Train MLP (15 → 32 → 3)
7. Check accuracy > 55%

**Result:** Feature separation still 0.2-8.4%. Bounce rate 49.8% with MFE label. See RESULTS.md

---

## Stage 5: Label Validation (NEXT)

**Objective:** Find which label parameters give the best feature separation.

### Test Matrix (12 combinations)

| Parameter | Values |
|-----------|--------|
| Horizon | 5, 10, 15, 25 bars |
| Dominance | 60%, 70%, 80% |
| Min move | 15bps (fixed) |

### Method
- Use same 67K events (every_bar dataset with saved bar numbers)
- Recompute label for each combination
- Check feature separation for each
- Find which combination gives highest separation

### Success Criteria
- Find a label combination with feature separation > 10%
- Bounce/break balanced (not all chop)

---

---

## Stage 6: S/R + Base Features Combined

### Objective
Test if combining S/R features (WHERE) with base features (WHAT) improves prediction.

### Features (18 total)

**S/R Features (15):**
- Dynamic (9): dist_to_zone_pct, support_retest, resistance_retest, zone_width, recovery_up_pct, recovery_down_pct, speed_short, speed_mid, speed_long
- Static (6): bounce_ratio, recent_bounce_ratio, pressure, bars_since_touch, touch_count_scaled, level_type_binary

**Base Features (3):**
- roc (from feature_cache.parquet)
- rsi7 (from feature_cache.parquet)
- range_position (from feature_cache.parquet)

### Data
- Same 67K events (every_bar, hybrid KDE)
- Bar numbers saved → load base features from experiments/layer2/L2-003/feature_cache.parquet
- Label: 3-class (bounce/break/chop), H25, 70% dominance, 15bps min

### Tests
**Test 1: No lookback (MLP)**
- Input: 18 features per bar (single bar)
- Model: MLP (18 → 32 → 3)

**Test 2: With lookback (LSTM)**
- Input: 25 bars × 18 features (snapshot sequence)
- Model: LSTM(64) → Dense(3)

### Steps
1. Load 67K dataset (has bar numbers)
2. Load base features from feature_cache.parquet for those bars
3. Combine: 15 S/R + 3 base = 18 features
4. Check feature separation on all 18
5. Run Test 1 (MLP, no lookback)
6. Run Test 2 (LSTM, 25-bar lookback)
7. Compare both vs S/R-only baseline (~33%)

### Success Criteria
- Feature separation > 10% on base features
- Test accuracy > 55%
- Improvement over S/R-only

**Result:** FAILED — see RESULTS.md Stage 6

---

## Stage 7: Correct Setup Testing (COMPLETE)

### Key Finding from Stage 6
- H25 label is unpredictable even with base features (50.9%)
- Need to test with H96 (proven to work at 57-58%)
- Need correct snapshot setup (3 features × 8 snapshots, not roc1-8)

### Planned Tests
1. Base features (roc, rsi7, range_position) × 8 snapshots, H96, binary → establish baseline
2. Add S/R zone_context on top → check if accuracy improves
3. S/R as filter: only predict at S/R zones using base model

---

---

## Stage 8: S/R Advisor — Standalone Test

### Objective
Test if separate dynamic/static path architecture can predict bounce/break.
Two-part test: without and with MFE head.

### Stage 8a: Without MFE head

Architecture:
```
DYNAMIC (9) -> Dense(9->4) + ReLU -> 4 numbers
STATIC (6)  -> Dense(6->4) + ReLU -> 4 numbers
Combine: 4+4 = 8 -> Dense(8->2) -> [bounce_score, break_score]

Loss = CrossEntropy(bounce/break)
```

### Stage 8b: With MFE head (connected)

Architecture:
```
DYNAMIC (9) -> Dense(9->4) + ReLU -> 4 numbers
STATIC (6)  -> Dense(6->4) + ReLU -> 4 numbers
Combine: 4+4 = 8 -> MFE head: Dense(8->2) -> [mfe_up, mfe_down]
                  -> Bounce head: Dense(8+2=10 -> 2) -> [bounce_score, break_score]

Loss = 1.0 x CrossEntropy(bounce/break) + 5.0 x MSE(mfe)
```

### Data
- 67K events (every_bar, hybrid KDE)
- S/R dynamic (9) + static (6) features
- Both support and resistance (~40%/60%)
- level_type_binary tells model which side
- Label: bounce/break (direction normalized per side)
  - Support: bounce=UP, break=DOWN
  - Resistance: bounce=DOWN, break=UP
- H25 horizon for MFE computation
- Binary only (remove BOTH/SKIP/CHOP)

### Success Criteria
- 8a accuracy > 52%
- 8b accuracy > 8a (MFE helps)
- If both fail at ~50%: S/R has no signal -- stop research

### Stage 8c: Hierarchical S/R Advisor with Gated Fusion

Processes features in trader's reasoning order. Gated fusion lets model decide
whether to trust history or recent state.

**Static features (14):**

History (used in Step 2):
1. bounce_ratio
2. touch_count_scaled
3. avg_bounce_mfe_pct (normalized by zone_width)
4. avg_break_mfe_pct (normalized by zone_width)
5. max_bounce_mfe_pct (normalized by zone_width)
6. max_break_mfe_pct (normalized by zone_width)

Recent (used in Step 3):
7. recent_bounce_ratio
8. pressure
9. last_outcome (1=bounce, 0=break, 0.5=chop)
10. bounce_streak
11. bounce_mfe_trend
12. chop_ratio

Zone position (used in Step 1):
13. level_type_binary
14. dist_to_zone_pct

**Architecture:**
```
STEP 1: Zone Position (no Dense)
  level_type_binary (1) + dist_to_zone_pct (1) = 2 numbers
  Pass directly to Step 2

STEP 2: History Branch
  Step 1 (2) + bounce_ratio + touch_count + 4 MFE_pct = 8
  Dense(8->4) + ReLU -> history_signal (4)

STEP 3: Recent Branch
  recent_bounce + pressure + last_outcome + bounce_streak + bounce_mfe_trend + chop_ratio = 6
  Dense(6->4) + ReLU -> recent_signal (4)

GATED FUSION
  history(4) + recent(4) = 8 -> Dense(8->4) + Sigmoid -> gate (4)
  zone_state = gate * recent + (1-gate) * history -> (4)

STEP 4: Approach Context
  zone_state (4) + 11 dynamic features = 15
  Dense(15->4) + ReLU -> approach_context (4)

STEP 5: Decision
  approach_context (4) -> Dense(4->2) -> [bounce, break]
```

**Normalization:** MFE features / zone_width. Others: z-score.
**Data:** 67K events, H25 bounce/break label
**Compare:** vs 8a (51.7%), 8c-A (51.4%), 8c-B (52.0%)

**Previous 8c-A/8c-B results (simple separate paths):**
- 8c-A (14 static, no MFE): 51.4%
- 8c-B (14 static, with MFE): 52.0%

### Next
- If passes: Stage 9 -- integrate as additive boost to base model
- If fails: Accept S/R features cannot predict, use S/R as filter only

---

## Stage 9: Redesigned Hierarchical S/R Advisor (NEXT)

### Why
Stage 8c hit 51.6-52.0% ceiling. Diagnostic revealed:
- Step 4 (approach features) tested with last-frame only — sequence info thrown away
- Raw MFE averages unreliable with small N (one outlier dominates)
- "Recent touches" features double-counted what's already in history
- Dense bottlenecks (→4) over-compressed; gated fusion gate had std=0.02 (basically fixed)
- Combinations like `high bnc MFE + low brk MFE → 57%` exist but only in <2% of bars

Conclusion: model wasn't taught wrong, it was BUILT wrong. Rebuild from first principles.

### New Hierarchy (3 steps, not 5)

**Step 1 — Position (1 feature, no transformation)**
- `price_position` ∈ [0, 1]: `(close - support_range_high) / (resistance_range_low - support_range_high)`, clipped
- 0=at support, 0.5=middle, 1=at resistance
- Replaces `level_type_binary` + `dist_to_zone_pct`
- Self-resets on breakouts (KDE recomputes zones)

**Step 2 — History / Long-term zone character (4 features)**
- `bounce_ratio` (last N=10 strict-edge touches)
- `touch_count` (confidence)
- `strong_bounce_share` = bounces with MFE > 0.5×zone_width / total bounces
- `strong_break_share` = breaks with MFE > 0.5×zone_width / total breaks
- Drops the 4 raw MFE averages and `bounce_mfe_trend` (averages problem)

**Step 3 — Trajectory (sequence model)**
- Uses dynamic memory as a SEQUENCE (25 bars × 11 features), not a single frame
- Conv1D, not Dense — captures motion patterns (speed-up, slow-down, hover, re-test)
- Replaces and absorbs the old "Step 4 (approach)"
- Old "Step 3 (recent touches)" is DELETED — redundant with Step 2

### Architecture

```
STEP 1+2 fusion:
  [price_position(1), 4 history features] = 5 dim
       → Linear(5→8) + ReLU
       → step2_output (8d)

STEP 3 trajectory:
  dynamic memory (25 bars × 11 features)
       → Conv1D(11→8, kernel=3) + ReLU
       → Conv1D(8→8,  kernel=3) + ReLU
       → AvgPool over time
       → step3_output (8d)

FINAL FUSION + CLASSIFIER:
  concat [step2_output(8), step3_output(8)] = 16 dim
       → Linear(16→8) + ReLU
       → Linear(8→2) → bounce / break
```

**Total params:** ~674

### Key Design Decisions (locked)
1. Step 1 has NO Dense layer — `price_position` passed raw
2. No "expansion" layers — only fusions that justify themselves
3. Dynamic features processed as a sequence, not flattened
4. Concat-then-fuse (simplest; multi-task deferred to v2)
5. CHOP class dropped — binary bounce/break only
6. Strict touch: `price_position ≤ 0.05` or `≥ 0.95`
7. Last N=10 touches for history
8. "Strong" MFE = 50% of zone_width (relative)

### Open Items Before Code
1. Dataset rebuild needed: new `price_position`, `strong_bounce_share`, `strong_break_share`, re-derived `bounce_ratio` under strict-touch + last-N=10
2. Filter to bounce/break only
3. Reuse existing train/val/test splits

### Success Criteria
- OOS accuracy > 55% (binary bounce/break)
- Step 3 measurably contributes lift over Step 2 alone
- If still ~52%: next iteration adds market regime as Step 4

### Deferred
- **Step 4 (market regime)** — only if Stage 9 caps at ~52%
- **Multi-task auxiliary losses** — only if concat-fuse proves insufficient

---

## Stage 9A: Minimal Static Memory Test (NEXT)

### Why
Before adding more architecture or feature groups, isolate the smallest
zone-memory context that could plausibly predict bounce vs break.

Question:
- Given the zone already exists and current `price_position` is known,
  does minimal static memory alone contain useful bounce/break signal?

### Features

**Position (kept separate):**
- `price_position`

**Static memory (minimal core):**
- `bounce_ratio`
- `touch_count_scaled`
- `recent_bounce_ratio`
- `pressure`

### Data / Label
- Same event definition: valid S/R touch event at bar `t`
- Same target: bounce/break from the next 25 bars
- Same train/val/test date split
- No dynamic trajectory features in this test
- No snapshots in input; memory features are already history summaries

### Architecture

```text
INPUT A: Position
  price_position -> [1]

INPUT B: Static Memory
  bounce_ratio
  touch_count_scaled
  recent_bounce_ratio
  pressure -> [4]

POSITION BRANCH:
  pass raw -> position_state [1]

MEMORY BRANCH:
  [4]
   -> Linear(4->8) + ReLU
   -> Linear(8->4) + ReLU
   -> memory_state [4]

FUSION:
  concat[position_state(1), memory_state(4)] -> [5]

DECISION HEAD:
  [5]
   -> Linear(5->8) + ReLU
   -> Dropout(0.1)
   -> Linear(8->2)
   -> [BREAK, BOUNCE]
```

### Purpose
- Test whether static memory itself carries bounce/break context
- Avoid mixing memory with trajectory/geometry/recovery too early
- Establish a clean baseline before adding more features

### Expansion Order If Signal Exists
1. Add `bars_since_touch`
2. Add `last_outcome`
3. Add `chop_ratio`
4. Reintroduce dynamic/trajectory features as a separate test

### Success Criteria
- OOS accuracy above bounce-base-rate behavior
- Break accuracy not collapsing to near-random
- Class predictions materially less biased than previous Stage 8/9 runs

---

## Stage 10: S/R Context For Trade Timing (NEXT)

### Why
Stage 8/9 showed S/R memory alone is weak for standalone bounce/break prediction.
Use S/R as context on top of the base model to improve timing and trade selection.

Core idea:
- Base model answers direction.
- S/R context answers whether this is a good location to act now.

### Objective
Improve entry quality, not build a standalone S/R direction model.

Expected effect:
- Better decisions near support/resistance zones
- Better second-touch handling
- Fewer low-quality trades where base direction conflicts with zone context

### S/R Context Features (apart from base features)
- `support_range_low`
- `support_range_high`
- `resistance_range_low`
- `resistance_range_high`
- `zone_width`
- `support_retest`
- `resistance_retest`

### Normalization
Convert price ranges to relative form per bar:
- `sl_norm = sl/close - 1`
- `sh_norm = sh/close - 1`
- `rl_norm = rl/close - 1`
- `rh_norm = rh/close - 1`
- `zone_width_norm = (rl - sh)/close`

Fit scaler on train only, apply to val/test.

### Model Pattern
1. Keep base branch unchanged.
2. Add a small SR context branch.
3. Late-fuse base state + SR state for final direction logits.
4. Optional second-touch gate:
- Near support + second touch: favor LONG
- Near resistance + second touch: favor SHORT

### What This Stage Should Teach
- Whether S/R context helps the model decide when to take trades
- Whether second-touch context improves quality over base-only
- Whether S/R should be used as learned context or only as a hard execution filter

### Success Criteria
- Consistent lift vs base-only on S/R-relevant subsets
- No major degradation in full test metrics
- Improved trade-quality metrics on second-touch zones

---

## Future Stages (Not Started)
- **Tradeability test** — backtest the 55.8% simple edge after fees
- **Multi-timeframe** — test on 1H, 4H
- **Other assets** — ETH, Gold, Silver (config-driven)
