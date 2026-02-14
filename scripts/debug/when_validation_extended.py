"""
WHEN PHASE: Extended Validation (All Targets & Horizons)

Validate WHEN filters across EXTENDED parameter combinations:
- Targets: 12bp, 15bp, 20bp, 25bp, 30bp, 50bp
- Horizons: H=5, H=10, H=15, H=30, H=60, H=120

Run: .venv/Scripts/python.exe scripts/debug/when_validation_extended.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGETS = [12, 15, 20, 25, 30, 50]  # basis points
HORIZONS = [5, 10, 15, 30, 60, 120]  # bars (minutes)
EXTENDED_H = 500

ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE: EXTENDED VALIDATION (ALL TARGETS & HORIZONS)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

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
    "ATR <10%": lambda df: df['atr_percentile'] < 10,
    "ATR >75%": lambda df: df['atr_percentile'] > 75,
    "Trend <0.5%": lambda df: df['ema_separation'] < 0.5,
    "Trend >1%": lambda df: df['ema_separation'] > 1.0,
    "00-04 UTC": lambda df: df['hour'].isin([0, 1, 2, 3]),
    "ATR>75%+Trend>1%": lambda df: (df['atr_percentile'] > 75) & (df['ema_separation'] > 1.0),
}


# =============================================================================
# RUN VALIDATION - COLLECT ALL RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING EXTENDED VALIDATION...")
print("=" * 80)

all_results = {}

total_combos = len(TARGETS) * len(HORIZONS)
combo_count = 0

for target in TARGETS:
    for H in HORIZONS:
        combo_count += 1
        print(f"  Processing T={target}bp, H={H}... ({combo_count}/{total_combos})")

        key = f"T{target}_H{H}"
        all_results[key] = {}

        for filter_name, filter_func in filters.items():
            train_mask = filter_func(train).values
            test_mask = filter_func(test).values

            train_result = analyze_condition(train, train_mask, target, H)
            test_result = analyze_condition(test, test_mask, target, H)

            if train_result and test_result:
                all_results[key][filter_name] = {
                    'train': train_result['p_case1'],
                    'test': test_result['p_case1'],
                }


# =============================================================================
# DISPLAY RESULTS: ATR <10% (AVOID)
# =============================================================================
print("\n" + "=" * 80)
print("FILTER: ATR <10% (Should be AVOID - above baseline)")
print("=" * 80)

print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for H in HORIZONS:
        key = f"T{target}_H{H}"
        if key in all_results and "ATR <10%" in all_results[key] and "Baseline" in all_results[key]:
            atr = all_results[key]["ATR <10%"]
            base = all_results[key]["Baseline"]
            # Check if ATR<10% is above baseline in both train AND test
            train_above = atr['train'] > base['train']
            test_above = atr['test'] > base['test']
            if train_above and test_above:
                row += f" YES |"
            else:
                row += f" NO  |"
        else:
            row += f" N/A |"
    print(row)


# =============================================================================
# DISPLAY RESULTS: ATR >75% (PREFER)
# =============================================================================
print("\n" + "=" * 80)
print("FILTER: ATR >75% (Should be PREFER - below baseline)")
print("=" * 80)

print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for H in HORIZONS:
        key = f"T{target}_H{H}"
        if key in all_results and "ATR >75%" in all_results[key] and "Baseline" in all_results[key]:
            atr = all_results[key]["ATR >75%"]
            base = all_results[key]["Baseline"]
            train_below = atr['train'] < base['train']
            test_below = atr['test'] < base['test']
            if train_below and test_below:
                row += f" YES |"
            else:
                row += f" NO  |"
        else:
            row += f" N/A |"
    print(row)


# =============================================================================
# DISPLAY RESULTS: Trend >1% (PREFER)
# =============================================================================
print("\n" + "=" * 80)
print("FILTER: Trend >1% (Should be PREFER - below baseline)")
print("=" * 80)

print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for H in HORIZONS:
        key = f"T{target}_H{H}"
        if key in all_results and "Trend >1%" in all_results[key] and "Baseline" in all_results[key]:
            trend = all_results[key]["Trend >1%"]
            base = all_results[key]["Baseline"]
            train_below = trend['train'] < base['train']
            test_below = trend['test'] < base['test']
            if train_below and test_below:
                row += f" YES |"
            else:
                row += f" NO  |"
        else:
            row += f" N/A |"
    print(row)


# =============================================================================
# DISPLAY RESULTS: 00-04 UTC (AVOID)
# =============================================================================
print("\n" + "=" * 80)
print("FILTER: 00-04 UTC (Should be AVOID - above baseline)")
print("=" * 80)

print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for H in HORIZONS:
        key = f"T{target}_H{H}"
        if key in all_results and "00-04 UTC" in all_results[key] and "Baseline" in all_results[key]:
            time = all_results[key]["00-04 UTC"]
            base = all_results[key]["Baseline"]
            train_above = time['train'] > base['train']
            test_above = time['test'] > base['test']
            if train_above and test_above:
                row += f" YES |"
            else:
                row += f" NO  |"
        else:
            row += f" N/A |"
    print(row)


# =============================================================================
# DISPLAY RESULTS: Combined Filter (PREFER)
# =============================================================================
print("\n" + "=" * 80)
print("FILTER: ATR>75% + Trend>1% (Should be PREFER - below baseline)")
print("=" * 80)

print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for H in HORIZONS:
        key = f"T{target}_H{H}"
        if key in all_results and "ATR>75%+Trend>1%" in all_results[key] and "Baseline" in all_results[key]:
            combo = all_results[key]["ATR>75%+Trend>1%"]
            base = all_results[key]["Baseline"]
            train_below = combo['train'] < base['train']
            test_below = combo['test'] < base['test']
            if train_below and test_below:
                row += f" YES |"
            else:
                row += f" NO  |"
        else:
            row += f" N/A |"
    print(row)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: VALIDATION SCORES")
print("=" * 80)

filter_checks = {
    "ATR <10% (AVOID)": ("ATR <10%", "above"),
    "ATR >75% (PREFER)": ("ATR >75%", "below"),
    "Trend >1% (PREFER)": ("Trend >1%", "below"),
    "00-04 UTC (AVOID)": ("00-04 UTC", "above"),
    "ATR>75%+Trend>1% (PREFER)": ("ATR>75%+Trend>1%", "below"),
}

print("\n| Filter | Valid / Total | Percentage | Status |")
print("|--------|---------------|------------|--------|")

for filter_label, (filter_key, direction) in filter_checks.items():
    valid = 0
    total = 0

    for target in TARGETS:
        for H in HORIZONS:
            key = f"T{target}_H{H}"
            if key in all_results and filter_key in all_results[key] and "Baseline" in all_results[key]:
                f = all_results[key][filter_key]
                b = all_results[key]["Baseline"]
                total += 1

                if direction == "above":
                    if f['train'] > b['train'] and f['test'] > b['test']:
                        valid += 1
                else:  # below
                    if f['train'] < b['train'] and f['test'] < b['test']:
                        valid += 1

    pct = (valid / total * 100) if total > 0 else 0
    status = "ROBUST" if pct >= 90 else "STRONG" if pct >= 75 else "PARTIAL" if pct >= 50 else "WEAK"
    print(f"| {filter_label:28s} | {valid:5d} / {total:5d} | {pct:9.1f}% | {status:7s} |")


# =============================================================================
# FINAL CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"""
Tested {len(TARGETS)} targets x {len(HORIZONS)} horizons = {len(TARGETS) * len(HORIZONS)} parameter combinations

Validation criteria:
- ROBUST (>=90%): Filter works in almost all conditions
- STRONG (>=75%): Filter works in most conditions
- PARTIAL (>=50%): Filter works in some conditions
- WEAK (<50%): Filter is unreliable

If all key filters score ROBUST or STRONG, WHEN phase is validated.
""")
