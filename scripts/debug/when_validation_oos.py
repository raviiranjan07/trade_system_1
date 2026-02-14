"""
WHEN PHASE: Out-of-Sample Validation (2024-2025)

Verify that WHEN filters discovered on train data (2020-2023)
hold on unseen data (2024-2025).

Run: .venv/Scripts/python.exe scripts/debug/when_validation_oos.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_BPS = 25
HORIZON = 30
EXTENDED_H = 500

ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE: OUT-OF-SAMPLE VALIDATION (2024-2025)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

# Split data
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()

print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")
print(f"Test date range: {test.index.min()} to {test.index.max()}")

# =============================================================================
# CALCULATE INDICATORS FOR TEST DATA
# =============================================================================
print("\nCalculating indicators for test data...")

# ATR
def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

test['atr'] = calculate_atr(test, ATR_PERIOD)
test['atr_pct'] = test['atr'] / test['close'] * 100
test['atr_percentile'] = test['atr_pct'].rank(pct=True) * 100

# EMAs
test['ema50'] = test['close'].ewm(span=EMA_FAST, adjust=False).mean()
test['ema200'] = test['close'].ewm(span=EMA_SLOW, adjust=False).mean()
test['ema_separation'] = np.abs(test['ema50'] - test['ema200']) / test['close'] * 100

# RSI
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

test['rsi'] = calculate_rsi(test['close'], 14)
test['hour'] = test.index.hour

print("Indicators calculated.")


# =============================================================================
# CASE CLASSIFICATION FUNCTION
# =============================================================================
@njit
def classify_single_bar(entry, highs, lows, target_pct, H, extended_H):
    n = len(highs)
    if n == 0:
        return -1, 0.0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False
    max_adverse_bps = 0.0

    for j in range(min(H, n)):
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse
        if lows[j] < entry:
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            break

    if went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if highs[j] >= target_price:
                hit_extended = True
                break

    if not went_below and hit_within_H:
        return 0, max_adverse_bps
    elif went_below and hit_within_H:
        return 2, max_adverse_bps
    elif went_below and not hit_within_H and hit_extended:
        return 3, max_adverse_bps
    elif went_below and not hit_within_H and not hit_extended:
        return 1, max_adverse_bps
    elif not went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                if went_below:
                    return 3, max_adverse_bps
                else:
                    return 0, max_adverse_bps
        return 1, max_adverse_bps

    return -1, max_adverse_bps


# =============================================================================
# ANALYZE FUNCTION
# =============================================================================
def analyze_condition(df, condition_mask, close, high, low, target_pct, H, extended_H):
    indices = np.where(condition_mask)[0]
    valid_indices = indices[(indices >= EMA_SLOW + 10) & (indices < len(df) - extended_H)]

    if len(valid_indices) < 50:
        return None

    cases = []
    maes = []

    for i in valid_indices:
        entry = close[i]
        future_highs = high[i+1:i+1+extended_H]
        future_lows = low[i+1:i+1+extended_H]

        case, mae = classify_single_bar(entry, future_highs, future_lows, target_pct, H, extended_H)
        if case >= 0:
            cases.append(case)
            maes.append(mae)

    if len(cases) < 50:
        return None

    cases_arr = np.array(cases)
    maes_arr = np.array(maes)
    total = len(cases_arr)

    return {
        'count': total,
        'p_case0': np.sum(cases_arr == 0) / total * 100,
        'p_case1': np.sum(cases_arr == 1) / total * 100,
        'p_case2': np.sum(cases_arr == 2) / total * 100,
        'p_case3': np.sum(cases_arr == 3) / total * 100,
        'mae_median': np.median(maes_arr),
        'mae_95': np.percentile(maes_arr, 95)
    }


# =============================================================================
# RUN VALIDATION
# =============================================================================
close = test['close'].values
high = test['high'].values
low = test['low'].values
target_pct = TARGET_BPS / 10000

# Baseline
print("\n" + "=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)

baseline_mask = np.ones(len(test), dtype=bool)
baseline = analyze_condition(test, baseline_mask, close, high, low, target_pct, HORIZON, EXTENDED_H)

if baseline:
    print(f"\nTest Baseline P(Case1): {baseline['p_case1']:.1f}%")
    print(f"Test Baseline count: {baseline['count']:,}")
    print(f"\n(Train Baseline was 16.2%)")


# =============================================================================
# VALIDATE KEY FILTERS
# =============================================================================
print("\n" + "=" * 80)
print("KEY FILTER VALIDATION: TRAIN vs TEST")
print("=" * 80)

filters_to_test = [
    ("ATR <10% (AVOID)", test['atr_percentile'] < 10, 33.8),
    ("ATR >90% (PREFER)", test['atr_percentile'] > 90, 6.5),
    ("ATR >75% (PREFER)", test['atr_percentile'] > 75, 8.3),
    ("Trend <0.5% (AVOID)", test['ema_separation'] < 0.5, 17.3),
    ("Trend >2% (PREFER)", test['ema_separation'] > 2.0, 4.0),
    ("Trend >1% (PREFER)", test['ema_separation'] > 1.0, 7.7),
    ("00-04 UTC (AVOID)", test['hour'].isin([0, 1, 2, 3]), 21.6),
    ("08-12 UTC (PREFER)", test['hour'].isin([8, 9, 10, 11]), 13.2),
    ("RSI >80 (AVOID)", test['rsi'] > 80, 21.1),
]

print("\n| Filter | Train P(Case1) | Test P(Case1) | Test Count | Status |")
print("|--------|----------------|---------------|------------|--------|")

validation_results = {}

for name, mask, train_pcase1 in filters_to_test:
    result = analyze_condition(test, mask.values, close, high, low, target_pct, HORIZON, EXTENDED_H)

    if result:
        diff = result['p_case1'] - train_pcase1

        # Check if pattern holds (within 5pp)
        if abs(diff) <= 5:
            status = "VALID"
        elif (train_pcase1 > 15 and result['p_case1'] > 15) or (train_pcase1 < 15 and result['p_case1'] < 15):
            status = "VALID (direction)"
        else:
            status = "FAILED"

        validation_results[name] = {
            'train': train_pcase1,
            'test': result['p_case1'],
            'count': result['count'],
            'status': status
        }

        print(f"| {name:25s} | {train_pcase1:13.1f}% | {result['p_case1']:12.1f}% | {result['count']:10,} | {status:6s} |")
    else:
        print(f"| {name:25s} | {train_pcase1:13.1f}% | N/A | <50 samples | SKIP |")


# =============================================================================
# VALIDATE COMBINED FILTERS
# =============================================================================
print("\n" + "=" * 80)
print("COMBINED FILTER VALIDATION")
print("=" * 80)

combined_filters = [
    ("ATR >75% + Trend >1%",
     (test['atr_percentile'] > 75) & (test['ema_separation'] > 1.0),
     6.6),
    ("NOT choppy + NOT low ATR",
     (test['atr_percentile'] > 25) & (test['ema_separation'] > 0.5),
     9.3),
    ("ATR <10% + Night hours",
     (test['atr_percentile'] < 10) & (test['hour'].isin([0, 1, 2, 3])),
     43.2),
]

print("\n| Combination | Train P(Case1) | Test P(Case1) | Test Count | Status |")
print("|-------------|----------------|---------------|------------|--------|")

for name, mask, train_pcase1 in combined_filters:
    result = analyze_condition(test, mask.values, close, high, low, target_pct, HORIZON, EXTENDED_H)

    if result:
        diff = result['p_case1'] - train_pcase1

        if abs(diff) <= 5:
            status = "VALID"
        elif (train_pcase1 > baseline['p_case1'] and result['p_case1'] > baseline['p_case1']) or \
             (train_pcase1 < baseline['p_case1'] and result['p_case1'] < baseline['p_case1']):
            status = "VALID (direction)"
        else:
            status = "FAILED"

        print(f"| {name:30s} | {train_pcase1:13.1f}% | {result['p_case1']:12.1f}% | {result['count']:10,} | {status:6s} |")
    else:
        print(f"| {name:30s} | {train_pcase1:13.1f}% | N/A | <50 samples | SKIP |")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

valid_count = sum(1 for v in validation_results.values() if 'VALID' in v['status'])
total_count = len(validation_results)

print(f"\nFilters validated: {valid_count}/{total_count}")

print("\n### Key Findings:")
print("-" * 40)

for name, result in validation_results.items():
    if 'VALID' in result['status']:
        print(f"  {name}: Train={result['train']:.1f}% -> Test={result['test']:.1f}% [OK]")
    else:
        print(f"  {name}: Train={result['train']:.1f}% -> Test={result['test']:.1f}% [FAILED]")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if valid_count >= total_count * 0.7:
    print("""
VALIDATION PASSED: Most WHEN filters hold on out-of-sample data.

The patterns discovered on 2020-2023 data generalize to 2024-2025.
WHEN phase filters are VALID for live trading.
""")
else:
    print("""
VALIDATION FAILED: WHEN filters do not hold on out-of-sample data.

The patterns may have been overfitted to training data.
Need to re-evaluate the WHEN phase approach.
""")
