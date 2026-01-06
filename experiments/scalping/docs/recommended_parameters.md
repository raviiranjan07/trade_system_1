# Scalping - Recommended Parameters

**Status:** NOT YET TESTED

---

## Grid Search Setup (Colab)

Use `scalping_grid_search_colab.ipynb` to run full grid search on Google Colab Free.

| Batch | Tests | Change `BATCH_NUMBER` to |
|-------|-------|-------------------------|
| 1 | 1-100 | 1 |
| 2 | 101-200 | 2 |
| 3 | 201-300 | 3 |
| 4 | 301-400 | 4 |
| 5 | 401-500 | 5 |
| 6 | 501-600 | 6 |
| 7 | 601-648 | 7 |

**Time per batch:** ~45 minutes on T4 GPU

---

## Overview

These are recommended parameters to test for scalping strategy (H=1-2 minutes).

---

## Core Parameters

### 1. Horizon (H)

| Value | Description |
|-------|-------------|
| H=1 | 1-minute forward prediction |
| H=2 | 2-minute forward prediction |

**Recommendation:** Start with H=2, then try H=1 if results are good.

---

### 2. Sample Interval (si)

| Value | Description |
|-------|-------------|
| si=1 | Check every bar (1 minute) |

**Recommendation:** si=1 is required for scalping. Cannot skip bars.

---

### 3. Normalization Window

| Value | Pros | Cons |
|-------|------|------|
| 200 | Fast adaptation | More noise |
| 300 | Balance | Moderate |
| 500 | More stable | Slower adaptation |

**Recommendation:** Start with 300, test 200 and 500.

---

### 4. min_expectancy

| Value | Expected Effect |
|-------|-----------------|
| 0.001 | May be too loose for short horizon |
| 0.002 | Balanced |
| 0.003 | Stricter, fewer trades |
| 0.005 | Very strict, rare trades |

**Recommendation:** Start with 0.002, may need to go higher.

---

### 5. min_consensus

| Value | Description |
|-------|-------------|
| 0.60 | Default (may be risky for scalping) |
| 0.70 | Moderate confidence |
| 0.75 | Higher confidence |
| 0.80 | Very high confidence |

**Recommendation:** 0.75+ for real-time safety.

---

### 6. k (Number of Neighbors)

| Value | Description |
|-------|-------------|
| 100 | Faster, less data |
| 150 | Moderate |
| 200 | Current default |

**Recommendation:** 100-150 for speed.

---

## Suggested Test Matrix

### Phase 1: Basic Horizon Test

| Test | H | si | min_exp | window |
|------|---|----|---------|--------|
| S1 | 2 | 1 | 0.002 | 300 |
| S2 | 1 | 1 | 0.002 | 300 |

### Phase 2: Expectancy Tuning (best H from Phase 1)

| Test | min_exp |
|------|---------|
| S3 | 0.001 |
| S4 | 0.002 |
| S5 | 0.003 |
| S6 | 0.005 |

### Phase 3: Window Tuning

| Test | window |
|------|--------|
| S7 | 200 |
| S8 | 300 |
| S9 | 500 |

### Phase 4: Consensus Tuning

| Test | min_consensus |
|------|---------------|
| S10 | 0.70 |
| S11 | 0.75 |
| S12 | 0.80 |

---

## Expected Challenges

1. **Higher noise at short horizons**: Prices are more random at 1-2 minute scale
2. **Execution latency**: Backtests don't account for real-world delays
3. **Spread impact**: Need to test with realistic spread assumptions
4. **Fewer similar neighbors**: Short-term patterns may be less repeatable

---

## Comparison to Day Trading

| Parameter | Day Trading (Current) | Scalping (Target) |
|-----------|----------------------|-------------------|
| Horizon | 5 min | 1-2 min |
| Sample Interval | 15 | 1 |
| Window | 2000 | 200-500 |
| min_expectancy | 0.001 | 0.002+ |
| Trades/Year | ~77 | ~500+ (expected) |

---

## Success Criteria

- Win rate > 60%
- Positive cumulative return
- Acceptable drawdown (<5%)
- Reasonable trade count (not excessive)
