# SURVIVE Phase — Final Locked Specification

## 1. Purpose (Non-Negotiable)

**SURVIVE = Futures survivability layer for Case 2 & Case 3**

```
GIVEN exposure is already allowed by WHEN
→ Can leverage + margin survive the expected adverse path?
```

**SURVIVE is not about profit.**
**SURVIVE is not about prediction.**
**SURVIVE is about staying alive.**

---

## 2. What SURVIVE Explicitly Does NOT Do

SURVIVE does NOT:
- Filter Case 1 (WHEN already did that)
- Predict direction
- Improve win rate
- Optimize PnL
- Choose entries or exits for alpha

**If any of the above appear → architecture is broken.**

---

## 3. Inputs to SURVIVE (Strict)

SURVIVE only runs after `WHEN = PASS`

**Required inputs:**
- Condition ID (from WHEN)
- Case label (2 or 3 only)
- MAE path (full adverse excursion)
- Time-at-risk (bars underwater)
- Recovery time (if recovered)
- Exchange liquidation rules
- Funding + fee model

**No indicators.**
**No signals.**
**No state vectors.**

---

## 4. Core Question SURVIVE Answers

> "How much leverage can survive this market reality without liquidation or capital lock?"

That's it. One question.

---

## 5. Survival Metrics (Authoritative)

These are the **only** metrics SURVIVE is allowed to care about.

### 5.1 MAE Distribution (by Case)

For each WHEN-approved condition:

| Metric | Meaning |
|--------|---------|
| Median MAE | Typical pain |
| P75 MAE | Normal stress |
| P90 MAE | High stress |
| P95 MAE | Survival boundary |
| P99 MAE | Extreme tail |

**⚠️ P95 is the design point.**
Anything beyond is considered unacceptable risk.

### 5.2 Time-at-Risk Distribution (CRITICAL)

For each case separately:

```
Time_at_Risk = number of bars unrealized PnL < 0
```

Measure:
- Median
- P75
- P90
- P95

This is **capital lock time**, not drawdown.

### 5.3 Recovery Time (Informational, not optimization)

Only for understanding exposure duration.

SURVIVE does NOT optimize around it.

---

## 6. Case Separation (MANDATORY)

SURVIVE must treat cases differently.

### Case 2 — Fast Recovery
- Short time-at-risk
- Mean-reverting
- Usually leverage-tolerant

### Case 3 — Slow Recovery
- Long underwater time
- Funding bleed risk
- Capital lock risk
- Often leverage-intolerant

**Rules MUST be derived separately, even if merged later.**

---

## 7. Leverage Survival Model (Corrected)

### 7.1 Effective Liquidation Threshold

Theoretical liquidation is useless alone.

Define:
```
Effective_Liq_MAE =
  (1 / leverage)
  - maintenance_margin
  - safety_buffer
  - fees
  - funding
```

**Safety buffer is empirical, not guessed.**

### 7.2 Survival Condition

A leverage level is SURVIVABLE if:
```
P95_MAE < Effective_Liq_MAE
AND
P95_Time_at_Risk < Max_Allowed_Time
```

Fail either → leverage is unsafe.

### 7.3 Output: Safe Leverage Bands

SURVIVE outputs **ranges**, not exact numbers.

Example:
```
SAFE:        3x
CAUTION:     5x
DANGEROUS:   ≥10x
```

**No precision illusion.**

---

## 8. Forced Exit Rules (NOT Stop Loss)

**This is not optimization.**

### Definition

A forced exit exists ONLY to prevent liquidation or permanent capital lock.

### Forced Exit Triggers (Example)
```
IF
  MAE > P95_MAE
AND
  Time_at_Risk > P95_Time
THEN
  EXIT (SURVIVAL)
```

**No early cutting.**
**No micro stops.**
**No PnL logic.**

---

## 9. Analyses to Run (Final)

### S1 — MAE Distribution by Case

**File:** `scripts/debug/survive_mae_distribution.py`

- Case 2 vs Case 3
- Conditioned on WHEN filters

### S2 — Time-at-Risk Distribution

**File:** `scripts/debug/survive_time_at_risk.py`

- Case 2 vs Case 3
- Identify capital lock regimes

### S3 — Leverage Survival Matrix

**File:** `scripts/debug/survive_leverage_matrix.py`

| Leverage | Case 2 Survive % | Case 3 Survive % |
|----------|-----------------|-----------------|
| 3x | ? | ? |
| 5x | ? | ? |
| 10x | ? | ? |
| 20x | ? | ? |

Liquidation = failure. No gray area.

### S4 — Forced Exit Stress Test

**File:** `scripts/debug/survive_forced_exit.py`

- With vs without forced exit
- Measure:
  - Liquidation rate
  - Capital freed
  - **NOT expected value**

### S5 — Condition-Specific Survival

**File:** `scripts/debug/survive_by_condition.py`

Run S1–S4 for:
- Baseline
- ATR >75%
- Trend >1%
- ATR >75% + Trend >1%

---

## 10. SURVIVE Outputs (Contract)

SURVIVE produces **rules**, not predictions.

```json
{
  "condition_id": "ATR>75% + Trend>1%",
  "case2": {
    "p95_mae_bp": 45,
    "p95_time_at_risk": 18
  },
  "case3": {
    "p95_mae_bp": 130,
    "p95_time_at_risk": 95
  },
  "leverage_rules": {
    "safe": "3x",
    "caution": "5x",
    "unsafe": ">=10x"
  },
  "forced_exit": {
    "mae_bp": 150,
    "time_at_risk": 120
  }
}
```

**This is what EXECUTE consumes.**

---

## 11. Phase Lock Rule

Once SURVIVE is frozen:
- No tuning
- No optimization
- No re-interpretation

Changes require:
- New data
- New full re-run

---

## 12. Final Architecture (Clean)

```
WHAT     → Reality measurement
WHEN     → Structural risk filter (Case 1)
SURVIVE  → Futures survivability (Case 2/3)
EXECUTE  → Entry, sizing, execution
```

Each layer is:
- Independent
- Non-overlapping
- Testable

---

## Final Statement (Important)

> **WHAT tells you what exists**
> **WHEN tells you when not to play**
> **SURVIVE tells you how much you can afford to be wrong**
> **EXECUTE is just mechanics**

---

## SURVIVE Phase Status: IN PROGRESS

---

## Analysis Results

### S1 — MAE Distribution by Case
**Status: COMPLETED** (2026-01-16)

**Parameters:** Target=25bp, H=30, Train data (2020-2023)

#### CASE 2 — Fast Recovery (P95 is design point)

| Condition | Count | Median | P75 | P90 | **P95** | P99 |
|-----------|-------|--------|-----|-----|---------|-----|
| Baseline | 825,049 | 9.1bp | 19.8bp | 37.0bp | **52.7bp** | 101.9bp |
| ATR >75% | 344,232 | 14.2bp | 29.6bp | 52.3bp | **72.2bp** | 133.7bp |
| Trend >1% | 59,131 | 17.3bp | 38.8bp | 73.6bp | **106.2bp** | 206.5bp |
| ATR>75% + Trend>1% | 55,781 | 18.1bp | 40.4bp | 76.1bp | **109.6bp** | 212.8bp |

#### CASE 3 — Slow Recovery (P95 is design point)

| Condition | Count | Median | P75 | P90 | **P95** | P99 |
|-----------|-------|--------|-----|-----|---------|-----|
| Baseline | 889,123 | 38.3bp | 74.9bp | 131.9bp | **180.5bp** | 323.3bp |
| ATR >75% | 123,011 | 91.9bp | 154.0bp | 242.7bp | **325.0bp** | 565.4bp |
| Trend >1% | 14,127 | 110.2bp | 192.8bp | 333.4bp | **462.6bp** | 974.1bp |
| ATR>75% + Trend>1% | 11,876 | 124.4bp | 210.4bp | 369.5bp | **494.7bp** | 994.7bp |

#### Case 2 vs Case 3 Comparison (P95)

| Condition | Case 2 P95 | Case 3 P95 | Ratio (C3/C2) |
|-----------|------------|------------|---------------|
| Baseline | 52.7bp | 180.5bp | **3.4x** |
| ATR >75% | 72.2bp | 325.0bp | **4.5x** |
| Trend >1% | 106.2bp | 462.6bp | **4.4x** |
| ATR>75% + Trend>1% | 109.6bp | 494.7bp | **4.5x** |

#### CRITICAL FINDING: Trade-off Between P(Case1) and MAE

| Condition | P(Case1) | Case 2 P95 MAE | Case 3 P95 MAE |
|-----------|----------|----------------|----------------|
| Baseline | 16.2% | 52.7bp | 180.5bp |
| ATR>75% + Trend>1% | 6.6% | 109.6bp | 494.7bp |

**Interpretation:**
- WHEN filters reduce P(Case1) from 16.2% to 6.6% (good)
- BUT increase MAE when wrong: Case 2 from 53bp to 110bp, Case 3 from 181bp to 495bp
- **Why?** High volatility = bigger moves in BOTH directions
- When you're right more often, but when wrong, you're MORE wrong

**This is NOT bad.** It's a clear trade-off:
- Baseline: 16% structural failure, smaller MAE when wrong
- WHEN-filtered: 7% structural failure, larger MAE when wrong

**Leverage Implications (Preliminary):**
- Case 2 P95 = 109.6bp for ATR>75%+Trend>1% condition
- Case 3 P95 = 494.7bp for ATR>75%+Trend>1% condition
- Case 3 is the binding constraint for leverage

---

### S2 - Time-at-Risk Distribution
**Status: COMPLETED** (2026-01-16)

**Parameters:** Target=25bp, H=30, Train data (2020-2023)

#### CASE 2 - Time-at-Risk (bars underwater)

| Condition | Count | Median | P75 | P90 | **P95** | P99 |
|-----------|-------|--------|-----|-----|---------|-----|
| Baseline | 825,049 | 4 | 9 | 16 | **20** | 26 |
| ATR >75% | 344,232 | 3 | 9 | 16 | **21** | 27 |
| Trend >1% | 59,131 | 2 | 7 | 14 | **20** | 27 |
| ATR>75% + Trend>1% | 55,781 | 2 | 7 | 14 | **20** | 27 |

#### CASE 3 - Time-at-Risk (bars underwater)

| Condition | Count | Median | P75 | P90 | **P95** | P99 |
|-----------|-------|--------|-----|-----|---------|-----|
| Baseline | 889,123 | 67 | 155 | 286 | **361** | 454 |
| ATR >75% | 123,011 | 72 | 158 | 298 | **376** | 470 |
| Trend >1% | 14,127 | 73 | 155 | 287 | **367** | 466 |
| ATR>75% + Trend>1% | 11,876 | 73 | 155 | 281 | **368** | 466 |

#### Recovery Time (bars to hit target)

**Case 2:**
| Condition | Median | P75 | P90 | P95 |
|-----------|--------|-----|-----|-----|
| Baseline | 10 | 18 | 25 | 27 |
| ATR>75% + Trend>1% | 3 | 8 | 17 | **22** |

**Case 3:**
| Condition | Median | P75 | P90 | P95 |
|-----------|--------|-----|-----|-----|
| Baseline | 100 | 201 | 336 | 405 |
| ATR>75% + Trend>1% | 77 | 158 | 285 | **372** |

#### Case 2 vs Case 3 Comparison (P95 Time-at-Risk)

| Condition | Case 2 P95 | Case 3 P95 | Ratio (C3/C2) |
|-----------|------------|------------|---------------|
| Baseline | 20 | 361 | **18.1x** |
| ATR>75% + Trend>1% | 20 | 368 | **18.4x** |

#### CRITICAL FINDING: Time is the Bigger Difference

| Metric | Case 2 | Case 3 | Ratio |
|--------|--------|--------|-------|
| P95 MAE | 109.6bp | 494.7bp | 4.5x |
| P95 Time-at-Risk | 20 bars | 368 bars | **18.4x** |

- MAE difference: 4.5x (Case 3 is 4.5x worse)
- Time difference: **18.4x** (Case 3 is 18x longer underwater)

**Capital Lock Implications (ATR>75% + Trend>1%):**
- Case 2: P95 = 20 minutes underwater - negligible funding cost
- Case 3: P95 = 368 minutes (~6 hours) underwater - significant funding cost

**Funding Cost Example (typical 0.01% per 8 hours):**
- Case 2: ~0.0004% funding cost (negligible)
- Case 3: ~0.0075% funding cost per occurrence

**Conclusion:**
- Time-at-Risk is MORE different between cases than MAE
- Case 3 is dangerous not just for MAE, but for capital lock
- WHEN filters do NOT significantly reduce time-at-risk (all ~368 bars at P95)

---

### S3 - Leverage Survival Matrix
**Status: COMPLETED** (2026-01-16)

**Parameters:** Target=25bp, H=30, Train data (2020-2023)

#### Effective Liquidation Thresholds (with 20% safety buffer)

| Leverage | Theoretical | Effective Threshold |
|----------|-------------|---------------------|
| 3x | 3333bp | 2619bp |
| 5x | 2000bp | 1552bp |
| 10x | 1000bp | 752bp |
| 20x | 500bp | 352bp |
| 50x | 200bp | 112bp |
| 100x | 100bp | 32bp |

Assumptions: Maintenance margin 0.4%, Safety buffer 20%, Fees 0.08%

#### Survival Matrix (ATR>75% + Trend>1%)

| Leverage | Liq Threshold | Case 2 Survive | Case 3 Survive | Classification |
|----------|---------------|----------------|----------------|----------------|
| 3x | 2619bp | 100.0% | 100.0% | **SAFE** |
| 5x | 1552bp | 100.0% | 99.7% | **SAFE** |
| 10x | 752bp | 99.9% | 97.7% | **CAUTION** |
| 20x | 352bp | 99.8% | 89.2% | DANGEROUS |
| 50x | 112bp | 95.2% | 44.5% | DANGEROUS |
| 100x | 32bp | 68.0% | 2.9% | DANGEROUS |

#### Baseline Comparison

| Leverage | Case 2 Survive (Baseline) | Case 3 Survive (Baseline) |
|----------|---------------------------|---------------------------|
| 10x | 100.0% | 99.9% |
| 20x | 100.0% | 99.2% |
| 50x | 99.2% | 86.4% |
| 100x | 87.2% | 43.3% |

#### CRITICAL FINDING: Case 3 is the Binding Constraint

| Metric | Case 2 | Case 3 | Implication |
|--------|--------|--------|-------------|
| P95 MAE | 109.6bp | 494.7bp | Case 3 is 4.5x worse |
| 10x Survival | 99.9% | 97.7% | Case 3 drives leverage limit |
| 20x Survival | 99.8% | 89.2% | 10.8% liquidation rate |

**Recommended Leverage Rules (ATR>75% + Trend>1%):**
```
SAFE:      <= 5x  (99.7%+ survival for both cases)
CAUTION:   6x-10x (97.7% Case 3 survival)
DANGEROUS: >= 20x (89.2% Case 3 survival = 10.8% liquidation)
```

**Key Insight:**
- WHEN filters reduce P(Case1) but do NOT improve leverage survival
- High volatility conditions have HIGHER MAE, requiring LOWER leverage
- Trade-off: Better structural risk, worse leverage capacity

---

### S4 - Forced Exit Stress Test
**Status: COMPLETED** (2026-01-16)

**Parameters:** Target=25bp, H=30, 10x leverage, Effective liquidation=752bp

#### Baseline (No Forced Exit)

| Metric | Value |
|--------|-------|
| Total trades | 74,712 |
| Target hit | 93.0% |
| Liquidated | 1.4% |
| Timeout | 5.6% |
| Avg bars held | 54 |

#### MAE-Based Forced Exit

| MAE Threshold | Target % | Liquidated % | Forced Exit % | Avg Bars |
|---------------|----------|--------------|---------------|----------|
| 100bp | 79.2% | **0.0%** | 20.7% | 14 |
| 150bp | 85.0% | **0.0%** | 14.6% | 22 |
| 200bp | 88.2% | **0.0%** | 10.9% | 28 |
| 300bp | 90.9% | **0.0%** | 7.0% | 38 |

#### Time-Based Forced Exit

| Time Threshold | Target % | Liquidated % | Forced Exit % | Avg Bars |
|----------------|----------|--------------|---------------|----------|
| 50 bars | 82.7% | 0.3% | 17.0% | 16 |
| 100 bars | 87.0% | 0.5% | 12.4% | 23 |
| 150 bars | 89.1% | 0.7% | 10.3% | 29 |

#### Combined Forced Exit (MAE + Time)

| Config | Target % | Liquidated % | Forced Exit % | Bars Saved | Liq Prevention |
|--------|----------|--------------|---------------|------------|----------------|
| 150bp/120bars | 83.2% | **0.0%** | 16.8% | 37 (68%) | +1.39pp |
| 200bp/150bars | 86.2% | **0.0%** | 13.8% | 33 (60%) | +1.39pp |
| 300bp/200bars | 88.9% | **0.0%** | 11.1% | 26 (48%) | +1.39pp |

#### CRITICAL FINDINGS

**1. MAE-based exit eliminates liquidation:**
- ANY MAE threshold < liquidation threshold (752bp) reduces liquidation to 0%
- Trade-off: More forced exits in exchange for zero liquidations

**2. MAE is more effective than Time:**
- MAE directly relates to liquidation risk
- Time only indirectly relates through capital lock

**3. Combined exit provides best protection:**
- Catches both deep drawdowns AND long locks
- Recommended: 150bp MAE + 120 bars time

**Recommended Forced Exit Rule:**
```
IF MAE > 150bp AND Time_at_Risk > 120 bars
THEN EXIT (SURVIVAL)
```

This is NOT a stop-loss. This is a survival mechanism.

---

### S5 - Condition-Specific Survival
**Status: COMPLETED** (2026-01-16)

**Consolidated comparison across all WHEN conditions**

#### Case Distribution by Condition

| Condition | Total | Case 0 | Case 1 | Case 2 | Case 3 |
|-----------|-------|--------|--------|--------|--------|
| Baseline | 2,099,697 | 2.2% | 16.2% | 39.3% | 42.3% |
| ATR >75% | 525,102 | 2.7% | 8.3% | 65.6% | 23.4% |
| Trend >1% | 81,236 | 2.8% | 7.0% | 72.8% | 17.4% |
| ATR>75% + Trend>1% | 74,712 | 2.8% | **6.6%** | 74.7% | 15.9% |

#### Survival Scorecard

| Condition | P(Case1) | MAE Score | TAR Score | 10x Survival | TOTAL |
|-----------|----------|-----------|-----------|--------------|-------|
| Baseline | 1 (worst) | 4 (best) | 4 (best) | 4 (best) | 13 |
| ATR>75% + Trend>1% | 4 (best) | 1 (worst) | 2 | 1 (worst) | 8 |

**Key Trade-off Identified:**
- Baseline: High structural failure (16.2%), low MAE when wrong
- ATR>75%+Trend>1%: Low structural failure (6.6%), high MAE when wrong

**Interpretation:** You can't have both. Choose based on priority:
- If avoiding structural failure is priority: Use ATR>75%+Trend>1% with lower leverage
- If leverage is priority: Use baseline conditions

---

## 10. Final SURVIVE Outputs (Contract)

**SURVIVE produces rules, not predictions.**

```json
{
  "condition_id": "ATR>75% + Trend>1%",
  "case2": {
    "p95_mae_bp": 110,
    "p95_time_at_risk": 20
  },
  "case3": {
    "p95_mae_bp": 495,
    "p95_time_at_risk": 368
  },
  "leverage_rules": {
    "safe": "5x",
    "caution": "10x",
    "unsafe": ">=20x"
  },
  "forced_exit": {
    "mae_bp": 150,
    "time_at_risk": 120
  }
}
```

**This is what EXECUTE consumes.**

---

## SURVIVE Phase Status: COMPLETE - READY FOR LOCK

**All 5 analyses completed:**
- S1: MAE Distribution - COMPLETED
- S2: Time-at-Risk Distribution - COMPLETED
- S3: Leverage Survival Matrix - COMPLETED
- S4: Forced Exit Stress Test - COMPLETED
- S5: Condition-Specific Survival - COMPLETED

**Key Findings Summary:**

| Finding | Value | Implication |
|---------|-------|-------------|
| Case 3 P95 MAE | 495bp | Binding constraint for leverage |
| Case 3 P95 TAR | 368 bars (~6h) | Capital lock risk |
| Safe leverage | <= 5x | 99.7%+ survival rate |
| Forced exit rule | 150bp/120bars | Eliminates liquidation |
| Trade-off | P(Case1) vs MAE | Can't optimize both |

---

### S6 - Multi-Parameter Validation
**Status: COMPLETED** (2026-01-16)

**Validation: 33/36 combinations = 91.7% ROBUST**

Tested same 36 parameter combinations as WHEN phase:
- Targets: 12bp, 15bp, 20bp, 25bp, 30bp, 50bp
- Horizons: 5, 10, 15, 30, 60, 120 bars

#### Case 3 MAE P95 (Train vs Test)

**TRAIN (2020-2023):**
| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |
|--------|-----|------|------|------|------|-------|
| 25bp | 321bp | 386bp | 433bp | 495bp | 605bp | 742bp |
| 50bp | 347bp | 389bp | 420bp | 473bp | 565bp | 674bp |

**TEST (2024-2025):**
| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |
|--------|-----|------|------|------|------|-------|
| 25bp | 220bp | 254bp | 279bp | 318bp | 348bp | 373bp |
| 50bp | 222bp | 239bp | 253bp | 293bp | 318bp | 337bp |

#### 10x Leverage Survival Rate (Train vs Test)

**TRAIN (2020-2023):**
| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |
|--------|-----|------|------|------|------|-------|
| 25bp | 99.1% | 98.7% | 98.4% | 97.7% | 96.4% | 95.1% |

**TEST (2024-2025):**
| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |
|--------|-----|------|------|------|------|-------|
| 25bp | 99.8% | 99.8% | 99.7% | 99.6% | 99.4% | 98.8% |

#### CRITICAL FINDING: Test Period Was EASIER

| Metric | Train (2020-2023) | Test (2024-2025) |
|--------|-------------------|------------------|
| Case 3 P95 MAE (T=25, H=30) | 495bp | 318bp |
| 10x Survival Rate | 97.7% | 99.6% |

**Interpretation:**
- 2024-2025 was LESS volatile than 2020-2023 for Case 3 scenarios
- Survival rates IMPROVED in test period
- **Conservative rules derived from train data will perform BETTER in calmer markets**

#### Validation Summary

| Result | Count | Rate |
|--------|-------|------|
| Pattern Holds | 33 | 91.7% |
| Pattern Breaks | 3 | 8.3% |
| **Status** | **ROBUST** | >= 90% |

All 3 failures were at H=120 (longest horizon) where train had ~750-820bp MAE but test had ~370-380bp.

**Next Phase:** EXECUTE (Entry, sizing, execution mechanics)
