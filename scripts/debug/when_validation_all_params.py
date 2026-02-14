"""
WHEN PHASE: Full Validation (All Parameters)

Validate WHEN filters across ALL parameter combinations:
- Targets: 15bp, 25bp
- Horizons: H=10, H=30, H=60

Run: .venv/Scripts/python.exe scripts/debug/when_validation_all_params.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGETS = [15, 25]  # basis points
HORIZONS = [10, 30, 60]  # bars
EXTENDED_H = 500

ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE: FULL VALIDATION (ALL PARAMETERS)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

# Split data
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()

print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\nCalculating indicators...")

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

for df in [train, test]:
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    df['atr_pct'] = df['atr'] / df['close'] * 100
    df['atr_percentile'] = df['atr_pct'].rank(pct=True) * 100
    df['ema50'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['ema_separation'] = np.abs(df['ema50'] - df['ema200']) / df['close'] * 100
    df['hour'] = df.index.hour

print("Indicators calculated.")


# =============================================================================
# CASE CLASSIFICATION FUNCTION
# =============================================================================
@njit
def classify_single_bar(entry, highs, lows, target_pct, H, extended_H):
    n = len(highs)
    if n == 0:
        return -1

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False

    for j in range(min(H, n)):
        if lows[j] < entry:
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            break

    if went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            if highs[j] >= target_price:
                hit_extended = True
                break

    if not went_below and hit_within_H:
        return 0
    elif went_below and hit_within_H:
        return 2
    elif went_below and not hit_within_H and hit_extended:
        return 3
    elif went_below and not hit_within_H and not hit_extended:
        return 1
    elif not went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                if went_below:
                    return 3
                else:
                    return 0
        return 1

    return -1


# =============================================================================
# ANALYZE FUNCTION
# =============================================================================
def analyze_condition(df, condition_mask, target_bps, H):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    target_pct = target_bps / 10000

    indices = np.where(condition_mask)[0]
    valid_indices = indices[(indices >= EMA_SLOW + 10) & (indices < len(df) - EXTENDED_H)]

    if len(valid_indices) < 50:
        return None

    cases = []
    for i in valid_indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]
        case = classify_single_bar(entry, future_highs, future_lows, target_pct, H, EXTENDED_H)
        if case >= 0:
            cases.append(case)

    if len(cases) < 50:
        return None

    cases_arr = np.array(cases)
    total = len(cases_arr)
    return {
        'count': total,
        'p_case1': np.sum(cases_arr == 1) / total * 100
    }


# =============================================================================
# KEY FILTERS TO VALIDATE
# =============================================================================
filters = {
    "Baseline": lambda df: pd.Series(np.ones(len(df), dtype=bool), index=df.index),
    "ATR <10% (AVOID)": lambda df: df['atr_percentile'] < 10,
    "ATR >75% (PREFER)": lambda df: df['atr_percentile'] > 75,
    "Trend <0.5% (AVOID)": lambda df: df['ema_separation'] < 0.5,
    "Trend >1% (PREFER)": lambda df: df['ema_separation'] > 1.0,
    "00-04 UTC (AVOID)": lambda df: df['hour'].isin([0, 1, 2, 3]),
    "ATR>75% + Trend>1%": lambda df: (df['atr_percentile'] > 75) & (df['ema_separation'] > 1.0),
}


# =============================================================================
# RUN VALIDATION FOR ALL COMBINATIONS
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION RESULTS BY TARGET AND HORIZON")
print("=" * 80)

all_results = {}

for target in TARGETS:
    for H in HORIZONS:
        print(f"\n### Target={target}bp, H={H} bars ({H} minutes)")
        print("-" * 60)

        key = f"T{target}_H{H}"
        all_results[key] = {}

        print(f"\n| Filter | Train P(Case1) | Test P(Case1) | Diff | Status |")
        print("|--------|----------------|---------------|------|--------|")

        for filter_name, filter_func in filters.items():
            train_mask = filter_func(train).values
            test_mask = filter_func(test).values

            train_result = analyze_condition(train, train_mask, target, H)
            test_result = analyze_condition(test, test_mask, target, H)

            if train_result and test_result:
                diff = test_result['p_case1'] - train_result['p_case1']

                # Determine status
                if filter_name == "Baseline":
                    status = "BASE"
                elif abs(diff) <= 5:
                    status = "VALID"
                elif (train_result['p_case1'] > 15 and test_result['p_case1'] > 15):
                    status = "VALID"  # Both still high
                elif (train_result['p_case1'] < 15 and test_result['p_case1'] < 15):
                    status = "VALID"  # Both still low
                else:
                    status = "CHECK"

                all_results[key][filter_name] = {
                    'train': train_result['p_case1'],
                    'test': test_result['p_case1'],
                    'diff': diff,
                    'status': status
                }

                print(f"| {filter_name:22s} | {train_result['p_case1']:13.1f}% | {test_result['p_case1']:12.1f}% | {diff:+5.1f}pp | {status:6s} |")
            else:
                print(f"| {filter_name:22s} | N/A | N/A | N/A | SKIP |")


# =============================================================================
# SUMMARY: DOES PATTERN HOLD ACROSS ALL PARAMS?
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: PATTERN CONSISTENCY ACROSS ALL PARAMETERS")
print("=" * 80)

# Check if key patterns hold
patterns_to_check = [
    ("ATR <10% (AVOID)", "above baseline", lambda train, test, base_train, base_test:
     train > base_train and test > base_test),
    ("ATR >75% (PREFER)", "below baseline", lambda train, test, base_train, base_test:
     train < base_train and test < base_test),
    ("Trend >1% (PREFER)", "below baseline", lambda train, test, base_train, base_test:
     train < base_train and test < base_test),
    ("00-04 UTC (AVOID)", "above baseline", lambda train, test, base_train, base_test:
     train > base_train and test > base_test),
]

print("\n| Pattern | T15_H10 | T15_H30 | T15_H60 | T25_H10 | T25_H30 | T25_H60 |")
print("|---------|---------|---------|---------|---------|---------|---------|")

for pattern_name, expected, check_func in patterns_to_check:
    row = f"| {pattern_name:20s} |"

    for target in TARGETS:
        for H in HORIZONS:
            key = f"T{target}_H{H}"

            if key in all_results and pattern_name in all_results[key] and "Baseline" in all_results[key]:
                p = all_results[key][pattern_name]
                b = all_results[key]["Baseline"]

                holds = check_func(p['train'], p['test'], b['train'], b['test'])
                row += f" {'YES':^7s} |" if holds else f" {'NO':^7s} |"
            else:
                row += f" {'N/A':^7s} |"

    print(row)


# =============================================================================
# FINAL CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Count how many combinations each pattern holds
print("\nPattern validity across all 6 parameter combinations:")
print("-" * 50)

for pattern_name, expected, check_func in patterns_to_check:
    valid_count = 0
    total_count = 0

    for target in TARGETS:
        for H in HORIZONS:
            key = f"T{target}_H{H}"

            if key in all_results and pattern_name in all_results[key] and "Baseline" in all_results[key]:
                p = all_results[key][pattern_name]
                b = all_results[key]["Baseline"]
                total_count += 1

                if check_func(p['train'], p['test'], b['train'], b['test']):
                    valid_count += 1

    pct = (valid_count / total_count * 100) if total_count > 0 else 0
    status = "ROBUST" if pct >= 80 else "PARTIAL" if pct >= 50 else "WEAK"
    print(f"  {pattern_name}: {valid_count}/{total_count} ({pct:.0f}%) - {status}")
