# Swing Trading - Tested Parameters

**Status:** PARTIALLY TESTED (Results Negative)

---

## Overview

Swing trading with longer horizons has been tested but not successfully.

---

## Tested Horizons

### H=15 (15 minutes)

| Parameter | Value |
|-----------|-------|
| Horizon | 15 |
| Sample Interval | 15 |
| min_expectancy | 0.001 |
| Window | 2000 |

**Result:**
| Metric | Value |
|--------|-------|
| PnL | -$40.62 |
| Win Rate | 51% |
| Status | LOSS |

**Analysis:** Near 50% win rate indicates no edge at this horizon.

---

### H=30 (30 minutes)

| Parameter | Value |
|-----------|-------|
| Horizon | 30 |
| Sample Interval | 15 |
| min_expectancy | 0.001 |
| Window | 2000 |

**Result:**
| Metric | Value |
|--------|-------|
| PnL | -$1,457 |
| Win Rate | 99% |
| Status | SIGNIFICANT LOSS |

**Analysis:** Despite 99% win rate, large losses indicate:
- Winning trades have small gains
- Losing trades have large losses
- Poor risk/reward at this horizon

---

## Parameters NOT Yet Tested

### Horizons to Test

| Horizon | Notes |
|---------|-------|
| H=60 | 1 hour hold |
| H=120 | 2 hour hold |
| H=240 | 4 hour hold |

### Normalization Windows to Test

| Window | Notes |
|--------|-------|
| 5000 | ~3.5 days of 1-min data |
| 7500 | ~5 days |
| 10000 | ~7 days |

### Sample Intervals to Test

| si | Notes |
|----|-------|
| 30 | Check every 30 min |
| 60 | Check every hour |

### min_expectancy to Test

| Value | Notes |
|-------|-------|
| 0.002 | Double current threshold |
| 0.003 | 3x current threshold |
| 0.005 | 5x current threshold |

---

## Suggested Test Sequence

### Phase 1: Baseline at Longer Horizons

| Test | H | si | min_exp | window | Expected |
|------|---|----|---------|--------|----------|
| SW1 | 60 | 30 | 0.002 | 2000 | Baseline |
| SW2 | 120 | 30 | 0.002 | 2000 | Baseline |

### Phase 2: Window Tuning

| Test | H | window | Notes |
|------|---|--------|-------|
| SW3 | best | 5000 | Longer history |
| SW4 | best | 7500 | Even longer |
| SW5 | best | 10000 | Very long |

### Phase 3: Expectancy Tuning

| Test | min_exp |
|------|---------|
| SW6 | 0.003 |
| SW7 | 0.005 |
| SW8 | 0.008 |

---

## Key Learnings from Failed Tests

1. **99% WR with losses at H=30**: Model predicts direction correctly but:
   - Take profit too small
   - Stop loss too tight
   - Doesn't capture full move

2. **51% WR at H=15**: Edge disappears at 15-min horizon with current features

3. **Possible root cause**: Features are computed on 1-min data, optimized for 5-min predictions. Longer horizons may need different feature engineering.

---

## Recommendations Before Further Testing

1. **Analyze H=30 trades**: Why does 99% WR produce losses?
2. **Check MFE distribution**: How far do winning trades actually go?
3. **Consider retraining**: May need separate model for swing trading
4. **Feature review**: Current features may not capture swing dynamics
