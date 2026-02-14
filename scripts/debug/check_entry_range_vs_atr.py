"""
Check if Entry Bar Range adds value beyond ATR

Questions to answer:
1. How correlated are Entry Bar Range and ATR?
2. Does Entry Bar Range predict Case 1 WITHIN ATR bins?
3. Does combining them improve over ATR alone?

Run: .venv/Scripts/python.exe scripts/debug/check_entry_range_vs_atr.py
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


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("CHECK: Entry Bar Range vs ATR - Independence Test")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")


# =============================================================================
# CALCULATE FEATURES
# =============================================================================
print("\nCalculating features...")

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

train['atr'] = calculate_atr(train, ATR_PERIOD)
train['atr_pct'] = train['atr'] / train['close'] * 100
train['atr_percentile'] = train['atr_pct'].rolling(window=1000, min_periods=100).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
)
train['range_bps'] = (train['high'] - train['low']) / train['close'] * 10000

print("Features calculated.")


# =============================================================================
# CASE LABELING FUNCTION
# =============================================================================
@njit
def label_case(entry, highs, lows, target_pct, H, extended_H):
    """Returns case (0, 1, 2, 3)."""
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

    if not hit_within_H:
        for j in range(H, min(extended_H, n)):
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                hit_extended = True
                break

    if not went_below and hit_within_H:
        return 0
    elif not hit_within_H and not hit_extended:
        return 1
    elif went_below and hit_within_H:
        return 2
    elif went_below and hit_extended:
        return 3
    else:
        return -1


# =============================================================================
# COLLECT DATA
# =============================================================================
print("\nCollecting data for analysis...")

close = train['close'].values
high = train['high'].values
low = train['low'].values
range_bps = train['range_bps'].values
atr_pctl = train['atr_percentile'].values
target_pct = TARGET_BPS / 10000

start_idx = 1010  # Skip initial bars for ATR percentile warmup
end_idx = len(train) - EXTENDED_H

entry_ranges = []
atr_pctls = []
cases = []

for i in range(start_idx, end_idx):
    if np.isnan(atr_pctl[i]) or np.isnan(range_bps[i]):
        continue

    entry = close[i]
    future_highs = high[i+1:i+1+EXTENDED_H]
    future_lows = low[i+1:i+1+EXTENDED_H]

    case = label_case(entry, future_highs, future_lows, target_pct, HORIZON, EXTENDED_H)

    if case >= 0:
        entry_ranges.append(range_bps[i])
        atr_pctls.append(atr_pctl[i])
        cases.append(case)

entry_ranges = np.array(entry_ranges)
atr_pctls = np.array(atr_pctls)
cases = np.array(cases)

print(f"Collected {len(cases):,} samples")


# =============================================================================
# QUESTION 1: CORRELATION
# =============================================================================
print("\n" + "=" * 80)
print("QUESTION 1: How correlated are Entry Bar Range and ATR?")
print("=" * 80)

correlation = np.corrcoef(entry_ranges, atr_pctls)[0, 1]
print(f"\nPearson correlation: {correlation:.3f}")

if abs(correlation) > 0.7:
    print("Interpretation: HIGH correlation - likely redundant")
elif abs(correlation) > 0.4:
    print("Interpretation: MODERATE correlation - partially overlapping")
else:
    print("Interpretation: LOW correlation - mostly independent")


# =============================================================================
# QUESTION 2: Entry Range within ATR bins
# =============================================================================
print("\n" + "=" * 80)
print("QUESTION 2: Does Entry Bar Range predict Case 1 WITHIN ATR bins?")
print("=" * 80)

# Define ATR bins
atr_bins = [
    ('ATR 0-25%', 0, 25),
    ('ATR 25-50%', 25, 50),
    ('ATR 50-75%', 50, 75),
    ('ATR 75-100%', 75, 100)
]

# Define Range quartiles (from earlier analysis)
range_q25 = np.percentile(entry_ranges, 25)
range_q75 = np.percentile(entry_ranges, 75)

print(f"\nRange quartile boundaries: Q25={range_q25:.1f}bp, Q75={range_q75:.1f}bp")
print("\n| ATR Bin | Range Q1 P(Case1) | Range Q4 P(Case1) | Diff | Independent? |")
print("|---------|-------------------|-------------------|------|--------------|")

independent_count = 0
for atr_name, atr_min, atr_max in atr_bins:
    atr_mask = (atr_pctls >= atr_min) & (atr_pctls < atr_max)

    # Within this ATR bin, check Range Q1 vs Q4
    range_q1_mask = atr_mask & (entry_ranges < range_q25)
    range_q4_mask = atr_mask & (entry_ranges >= range_q75)

    q1_cases = cases[range_q1_mask]
    q4_cases = cases[range_q4_mask]

    if len(q1_cases) > 100 and len(q4_cases) > 100:
        q1_case1 = (q1_cases == 1).sum() / len(q1_cases) * 100
        q4_case1 = (q4_cases == 1).sum() / len(q4_cases) * 100
        diff = q1_case1 - q4_case1
        independent = "YES" if diff > 5 else "NO"
        if diff > 5:
            independent_count += 1
        print(f"| {atr_name} | {q1_case1:.1f}% | {q4_case1:.1f}% | {diff:+.1f}pp | {independent} |")
    else:
        print(f"| {atr_name} | N/A | N/A | N/A | Insufficient data |")

print(f"\nIndependent in {independent_count}/4 ATR bins")


# =============================================================================
# QUESTION 3: Combined vs ATR alone
# =============================================================================
print("\n" + "=" * 80)
print("QUESTION 3: Does combining them improve over ATR alone?")
print("=" * 80)

# ATR alone (best condition: ATR > 75%)
atr_high_mask = atr_pctls >= 75
atr_high_cases = cases[atr_high_mask]
atr_high_case1 = (atr_high_cases == 1).sum() / len(atr_high_cases) * 100

# Combined: ATR > 75% AND Range Q4
combined_mask = (atr_pctls >= 75) & (entry_ranges >= range_q75)
combined_cases = cases[combined_mask]
combined_case1 = (combined_cases == 1).sum() / len(combined_cases) * 100

# Combined: ATR > 75% AND Range Q1 (worst range within good ATR)
worst_combined_mask = (atr_pctls >= 75) & (entry_ranges < range_q25)
worst_combined_cases = cases[worst_combined_mask]
worst_combined_case1 = (worst_combined_cases == 1).sum() / len(worst_combined_cases) * 100

print("\n| Condition | Count | P(Case1) | vs ATR alone |")
print("|-----------|-------|----------|--------------|")
print(f"| ATR > 75% (alone) | {len(atr_high_cases):,} | {atr_high_case1:.1f}% | baseline |")
print(f"| ATR > 75% + Range Q4 | {len(combined_cases):,} | {combined_case1:.1f}% | {combined_case1 - atr_high_case1:+.1f}pp |")
print(f"| ATR > 75% + Range Q1 | {len(worst_combined_cases):,} | {worst_combined_case1:.1f}% | {worst_combined_case1 - atr_high_case1:+.1f}pp |")

improvement = atr_high_case1 - combined_case1
print(f"\n**Improvement from adding Range Q4 filter:** {improvement:.1f}pp")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Does Entry Bar Range add value beyond ATR?")
print("=" * 80)

print(f"""
1. Correlation: {correlation:.3f} ({'HIGH' if abs(correlation) > 0.7 else 'MODERATE' if abs(correlation) > 0.4 else 'LOW'})

2. Independent within ATR bins: {independent_count}/4 bins show Entry Range effect

3. Combined improvement: {improvement:.1f}pp better than ATR alone
""")

if abs(correlation) < 0.7 and independent_count >= 2 and improvement > 2:
    print("**CONCLUSION: Entry Bar Range ADDS INDEPENDENT VALUE beyond ATR**")
    print("  - Keep both features")
    print("  - They measure different aspects of volatility")
elif abs(correlation) >= 0.7 or (independent_count < 2 and improvement < 2):
    print("**CONCLUSION: Entry Bar Range is REDUNDANT with ATR**")
    print("  - Can drop Entry Bar Range")
    print("  - ATR captures the same information")
else:
    print("**CONCLUSION: MARGINAL value - Entry Bar Range adds small benefit**")
    print("  - Optional to keep")
    print("  - Depends on complexity preference")
