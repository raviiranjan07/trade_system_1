"""
WHAT PHASE: Range Expansion Analysis (W-EXP1)

Measure how ATR/Range evolves over time:
- Does range expand or contract after entry?
- How does initial range relate to outcome?
- What's the expansion pattern by case?

Parameters:
- Horizons: 5, 10, 15, 30, 60, 120 bars
- Targets: 12, 15, 20, 25, 30, 50 bp

Run: .venv/Scripts/python.exe scripts/debug/what_range_expansion.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import itertools

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGETS = [12, 15, 20, 25, 30, 50]  # basis points
HORIZONS = [5, 10, 15, 30, 60, 120]  # bars
EXTENDED_H = 500

ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHAT PHASE: RANGE EXPANSION ANALYSIS (W-EXP1)")
print(f"Testing {len(TARGETS)} targets x {len(HORIZONS)} horizons = {len(TARGETS)*len(HORIZONS)} combinations")
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
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['ema50'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

print("Indicators calculated.")


# =============================================================================
# CASE LABELING FUNCTION
# =============================================================================
@njit
def label_case(entry, highs, lows, target_pct, H, extended_H):
    """Returns case (0, 1, 2, 3) and metrics."""
    n = len(highs)
    if n == 0:
        return -1, 0.0, 0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False
    hit_time = 0

    for j in range(min(H, n)):
        if lows[j] < entry:
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            hit_time = j + 1
            break

    if not hit_within_H:
        for j in range(H, min(extended_H, n)):
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                hit_extended = True
                hit_time = j + 1
                break

    if not went_below and hit_within_H:
        case = 0
    elif not hit_within_H and not hit_extended:
        case = 1
    elif went_below and hit_within_H:
        case = 2
    elif went_below and hit_extended:
        case = 3
    else:
        case = -1

    return case, 0.0, hit_time


# =============================================================================
# RANGE EXPANSION ANALYSIS FUNCTION
# =============================================================================
def analyze_range_expansion(df, target_bps, H):
    """Analyze how range expands/contracts by case."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    range_bps = df['range_bps'].values
    atr_pct = df['atr_pct'].values
    target_pct = target_bps / 10000

    indices = np.arange(EMA_SLOW + 10, len(df) - EXTENDED_H)

    if len(indices) < 100:
        return None

    results = {
        'case0_entry_range': [], 'case0_future_range': [], 'case0_expansion': [],
        'case1_entry_range': [], 'case1_future_range': [], 'case1_expansion': [],
        'case2_entry_range': [], 'case2_future_range': [], 'case2_expansion': [],
        'case3_entry_range': [], 'case3_future_range': [], 'case3_expansion': [],
    }

    for i in indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, _, hit_time = label_case(
            entry, future_highs, future_lows, target_pct, H, EXTENDED_H
        )

        if case < 0:
            continue

        # Entry range (current bar)
        entry_range = range_bps[i]

        # Future range (average over horizon H)
        future_range_arr = range_bps[i+1:i+1+H]
        if len(future_range_arr) > 0:
            future_range = np.mean(future_range_arr)
        else:
            continue

        # Expansion ratio
        expansion = future_range / entry_range if entry_range > 0 else 0

        results[f'case{case}_entry_range'].append(entry_range)
        results[f'case{case}_future_range'].append(future_range)
        results[f'case{case}_expansion'].append(expansion)

    # Calculate statistics
    output = {
        'total': len(indices),
        'target': target_bps,
        'horizon': H,
    }

    for case in [0, 1, 2, 3]:
        entry_arr = np.array(results[f'case{case}_entry_range'])
        future_arr = np.array(results[f'case{case}_future_range'])
        expansion_arr = np.array(results[f'case{case}_expansion'])

        output[f'case{case}_count'] = len(entry_arr)

        if len(entry_arr) >= 50:
            output[f'case{case}_entry_range_median'] = np.median(entry_arr)
            output[f'case{case}_future_range_median'] = np.median(future_arr)
            output[f'case{case}_expansion_median'] = np.median(expansion_arr)
            output[f'case{case}_expansion_p75'] = np.percentile(expansion_arr, 75)
            output[f'case{case}_expansion_p25'] = np.percentile(expansion_arr, 25)
        else:
            output[f'case{case}_entry_range_median'] = None
            output[f'case{case}_future_range_median'] = None
            output[f'case{case}_expansion_median'] = None
            output[f'case{case}_expansion_p75'] = None
            output[f'case{case}_expansion_p25'] = None

    return output


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING RANGE EXPANSION ANALYSIS")
print("=" * 80)

train_results = []
test_results = []

total_combos = len(TARGETS) * len(HORIZONS)
combo_num = 0

for target, horizon in itertools.product(TARGETS, HORIZONS):
    combo_num += 1
    print(f"\rProcessing {combo_num}/{total_combos}: T={target}bp, H={horizon}...", end="", flush=True)

    # Train
    train_result = analyze_range_expansion(train, target, horizon)
    if train_result:
        train_result['dataset'] = 'train'
        train_results.append(train_result)

    # Test
    test_result = analyze_range_expansion(test, target, horizon)
    if test_result:
        test_result['dataset'] = 'test'
        test_results.append(test_result)

print("\n\nAnalysis complete.")


# =============================================================================
# DISPLAY RESULTS: EXPANSION BY CASE (T=25bp reference)
# =============================================================================
print("\n" + "=" * 80)
print("RANGE EXPANSION BY CASE (T=25bp, H=30, Baseline)")
print("=" * 80)

print("\n### TRAIN DATA (2020-2023)")
print("\n| Case | Count | Entry Range | Future Range | Expansion Ratio |")
print("|------|-------|-------------|--------------|-----------------|")

for case in [0, 1, 2, 3]:
    for r in train_results:
        if r['target'] == 25 and r['horizon'] == 30:
            count = r[f'case{case}_count']
            if r[f'case{case}_expansion_median'] is not None:
                entry = r[f'case{case}_entry_range_median']
                future = r[f'case{case}_future_range_median']
                expansion = r[f'case{case}_expansion_median']
                print(f"| Case {case} | {count:,} | {entry:.1f}bp | {future:.1f}bp | {expansion:.2f}x |")
            else:
                print(f"| Case {case} | {count:,} | N/A | N/A | N/A |")
            break


# =============================================================================
# DISPLAY: EXPANSION BY TARGET/HORIZON (Case 0 vs Case 1)
# =============================================================================
print("\n" + "=" * 80)
print("CASE 0 vs CASE 1 ENTRY RANGE COMPARISON (TRAIN)")
print("=" * 80)

print("\n| Target | H | Case 0 Entry Range | Case 1 Entry Range | Diff |")
print("|--------|---|-------------------|-------------------|------|")

for target in [15, 25, 50]:
    for horizon in [10, 30, 60]:
        match = [r for r in train_results if r['target'] == target and r['horizon'] == horizon]
        if match:
            r = match[0]
            c0 = r['case0_entry_range_median']
            c1 = r['case1_entry_range_median']
            if c0 is not None and c1 is not None:
                diff = c0 - c1
                print(f"| {target}bp | {horizon} | {c0:.1f}bp | {c1:.1f}bp | {diff:+.1f}bp |")


# =============================================================================
# DISPLAY: EXPANSION RATIO BY CASE
# =============================================================================
print("\n" + "=" * 80)
print("EXPANSION RATIO BY CASE (T=25bp, TRAIN)")
print("=" * 80)
print("Expansion > 1 = Range expanding (volatility increase)")
print("Expansion < 1 = Range contracting (volatility decrease)")

print("\n| Horizon | Case 0 | Case 1 | Case 2 | Case 3 |")
print("|---------|--------|--------|--------|--------|")

for horizon in HORIZONS:
    match = [r for r in train_results if r['target'] == 25 and r['horizon'] == horizon]
    if match:
        r = match[0]
        c0 = r['case0_expansion_median']
        c1 = r['case1_expansion_median']
        c2 = r['case2_expansion_median']
        c3 = r['case3_expansion_median']
        row = f"| H={horizon} |"
        for c in [c0, c1, c2, c3]:
            if c is not None:
                row += f" {c:.2f}x |"
            else:
                row += " N/A |"
        print(row)


# =============================================================================
# TRAIN vs TEST COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("TRAIN vs TEST: CASE 1 ENTRY RANGE (T=25bp)")
print("=" * 80)

print("\n| Horizon | Train Entry Range | Test Entry Range | Diff |")
print("|---------|-------------------|------------------|------|")

for horizon in HORIZONS:
    train_match = [r for r in train_results if r['target'] == 25 and r['horizon'] == horizon]
    test_match = [r for r in test_results if r['target'] == 25 and r['horizon'] == horizon]

    if train_match and test_match:
        train_range = train_match[0]['case1_entry_range_median']
        test_range = test_match[0]['case1_entry_range_median']
        if train_range and test_range:
            diff = test_range - train_range
            print(f"| H={horizon} | {train_range:.1f}bp | {test_range:.1f}bp | {diff:+.1f}bp |")


# =============================================================================
# KEY INSIGHT: LOW ENTRY RANGE = HIGHER CASE 1?
# =============================================================================
print("\n" + "=" * 80)
print("ENTRY RANGE QUARTILE vs CASE DISTRIBUTION (T=25bp, H=30)")
print("=" * 80)

# Analyze if low entry range correlates with more Case 1
close = train['close'].values
high = train['high'].values
low = train['low'].values
range_bps = train['range_bps'].values
target_pct = 25 / 10000

indices = np.arange(EMA_SLOW + 10, len(train) - EXTENDED_H)

entry_ranges = []
cases = []

for i in indices:
    entry = close[i]
    future_highs = high[i+1:i+1+EXTENDED_H]
    future_lows = low[i+1:i+1+EXTENDED_H]

    case, _, _ = label_case(entry, future_highs, future_lows, target_pct, 30, EXTENDED_H)

    if case >= 0:
        entry_ranges.append(range_bps[i])
        cases.append(case)

entry_ranges = np.array(entry_ranges)
cases = np.array(cases)

# Quartile analysis
q25 = np.percentile(entry_ranges, 25)
q50 = np.percentile(entry_ranges, 50)
q75 = np.percentile(entry_ranges, 75)

print(f"\nEntry Range Quartiles:")
print(f"  Q25 = {q25:.1f}bp")
print(f"  Q50 = {q50:.1f}bp")
print(f"  Q75 = {q75:.1f}bp")

print("\n| Quartile | Range | Case 0 % | Case 1 % | Case 2 % | Case 3 % |")
print("|----------|-------|----------|----------|----------|----------|")

for qname, qmin, qmax in [('Q1 (Low)', 0, q25), ('Q2', q25, q50), ('Q3', q50, q75), ('Q4 (High)', q75, 999)]:
    mask = (entry_ranges >= qmin) & (entry_ranges < qmax)
    subset_cases = cases[mask]
    total = len(subset_cases)
    if total > 0:
        c0_pct = (subset_cases == 0).sum() / total * 100
        c1_pct = (subset_cases == 1).sum() / total * 100
        c2_pct = (subset_cases == 2).sum() / total * 100
        c3_pct = (subset_cases == 3).sum() / total * 100
        print(f"| {qname} | {qmin:.0f}-{qmax:.0f}bp | {c0_pct:.1f}% | {c1_pct:.1f}% | {c2_pct:.1f}% | {c3_pct:.1f}% |")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("RANGE EXPANSION ANALYSIS SUMMARY")
print("=" * 80)

print("""
Key Findings:

1. Entry Range = Current bar's high-low range in bps
   - Measures immediate volatility at entry
   - May correlate with case outcomes

2. Expansion Ratio = Future Range / Entry Range
   - Ratio > 1: Volatility expanding (market moving)
   - Ratio < 1: Volatility contracting (market calming)

3. Case-specific patterns:
   - Case 0: What entry range leads to clean wins?
   - Case 1: Does low range correlate with structural failure?
   - Case 2/3: Does high range correlate with recovery?

4. Implications for WHEN phase:
   - Should we filter by entry range?
   - Is there a "sweet spot" range for trading?
   - Does expansion ratio predict outcome?
""")


# =============================================================================
# SAVE RESULTS
# =============================================================================
output_path = Path("experiments/what_range_expansion.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

all_results = train_results + test_results
df_results = pd.DataFrame(all_results)
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
