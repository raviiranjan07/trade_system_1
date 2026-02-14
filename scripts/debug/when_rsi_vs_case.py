"""
WHEN PHASE: W1 - RSI vs Case Distribution

Question: Does RSI bin change P(Case1), REGARDLESS of trade direction?

IMPORTANT: No direction conditioning. We analyze RSI bins for ALL trades.
We're looking for conditions where P(Case1) differs from baseline (~10%).

Run: .venv/Scripts/python.exe scripts/debug/when_rsi_vs_case.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
RSI_PERIOD = 14
RSI_BINS = [(0, 20), (20, 30), (30, 40), (40, 60), (60, 70), (70, 80), (80, 100)]
TARGET_BPS = 25  # Focus on main target
HORIZONS = [10, 30, 60]
EXTENDED_H = 500
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE W1: RSI vs CASE DISTRIBUTION")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

# =============================================================================
# CALCULATE RSI
# =============================================================================
print("\nCalculating RSI...")

def calculate_rsi(close, period=14):
    """Calculate RSI using exponential moving average."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

train['rsi'] = calculate_rsi(train['close'], RSI_PERIOD)
print(f"RSI calculated. NaN count: {train['rsi'].isna().sum()}")

# =============================================================================
# CASE CLASSIFICATION FUNCTION (from case_labeler.py)
# =============================================================================
@njit
def classify_single_bar(entry, highs, lows, target_pct, H, extended_H):
    """Classify a single bar into Case 0/1/2/3."""
    n = len(highs)
    if n == 0:
        return -1, 0.0

    target_price = entry * (1 + target_pct)

    went_below = False
    hit_within_H = False
    hit_extended = False
    max_adverse_bps = 0.0

    # Check within H bars
    for j in range(min(H, n)):
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse
        if lows[j] < entry:
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            break

    # If went below and didn't hit within H, check extended
    if went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if highs[j] >= target_price:
                hit_extended = True
                break

    # Classify
    if not went_below and hit_within_H:
        return 0, max_adverse_bps  # Case 0: Clean Win
    elif went_below and hit_within_H:
        return 2, max_adverse_bps  # Case 2: Quick Recovery
    elif went_below and not hit_within_H and hit_extended:
        return 3, max_adverse_bps  # Case 3: Slow Recovery
    elif went_below and not hit_within_H and not hit_extended:
        return 1, max_adverse_bps  # Case 1: Wrong Direction
    elif not went_below and not hit_within_H:
        # Check extended
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
# ANALYZE RSI BINS
# =============================================================================
print("\nAnalyzing RSI bins vs Case distribution...")

close = train['close'].values
high = train['high'].values
low = train['low'].values
rsi = train['rsi'].values
n = len(train)

target_pct = TARGET_BPS / 10000

results = {}

for H in HORIZONS:
    print(f"\n  Processing H={H}...")

    results[H] = {}

    # Sample indices
    np.random.seed(42)
    valid_start = RSI_PERIOD + 10
    sample_idx = np.random.choice(
        range(valid_start, n - EXTENDED_H),
        size=min(SAMPLE_SIZE, n - EXTENDED_H - valid_start),
        replace=False
    )

    for rsi_low, rsi_high in RSI_BINS:
        bin_key = f"{rsi_low}-{rsi_high}"
        cases = []
        maes = []

        for i in sample_idx:
            # Check if RSI in this bin
            if pd.isna(rsi[i]):
                continue
            if not (rsi_low <= rsi[i] < rsi_high):
                continue

            entry = close[i]
            future_highs = high[i+1:i+1+EXTENDED_H]
            future_lows = low[i+1:i+1+EXTENDED_H]

            case, mae = classify_single_bar(entry, future_highs, future_lows,
                                            target_pct, H, EXTENDED_H)
            if case >= 0:
                cases.append(case)
                maes.append(mae)

        # Calculate statistics
        if len(cases) > 0:
            cases_arr = np.array(cases)
            maes_arr = np.array(maes)

            total = len(cases_arr)
            p_case0 = np.sum(cases_arr == 0) / total * 100
            p_case1 = np.sum(cases_arr == 1) / total * 100
            p_case2 = np.sum(cases_arr == 2) / total * 100
            p_case3 = np.sum(cases_arr == 3) / total * 100

            results[H][bin_key] = {
                'count': total,
                'p_case0': p_case0,
                'p_case1': p_case1,
                'p_case2': p_case2,
                'p_case3': p_case3,
                'mae_median': np.median(maes_arr),
                'mae_75': np.percentile(maes_arr, 75)
            }

            print(f"    RSI {bin_key}: n={total}, P(Case1)={p_case1:.1f}%")

print("\nAnalysis complete!")


# =============================================================================
# OUTPUT: MARKDOWN FORMAT
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: RSI vs Case Distribution")
print("=" * 80)

for H in HORIZONS:
    print(f"\n### Target={TARGET_BPS}bp, Horizon H={H}")
    print("\n| RSI Bin | Count | P(Case0) | P(Case1) | P(Case2) | P(Case3) | Median MAE |")
    print("|---------|-------|----------|----------|----------|----------|------------|")

    for bin_key in [f"{lo}-{hi}" for lo, hi in RSI_BINS]:
        if bin_key in results[H]:
            r = results[H][bin_key]
            print(f"| {bin_key:7s} | {r['count']:5d} | {r['p_case0']:7.1f}% | "
                  f"{r['p_case1']:7.1f}% | {r['p_case2']:7.1f}% | {r['p_case3']:7.1f}% | "
                  f"{r['mae_median']:9.1f}bp |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: RSI vs P(Case1)")
print("=" * 80)

# Compare extreme RSI bins to neutral (40-60)
H = 30  # Focus on main horizon
if "40-60" in results[H]:
    baseline = results[H]["40-60"]['p_case1']
    print(f"\nBaseline P(Case1) at RSI 40-60: {baseline:.1f}%")

    print("\nDeviation from baseline:")
    for bin_key in [f"{lo}-{hi}" for lo, hi in RSI_BINS]:
        if bin_key in results[H] and bin_key != "40-60":
            r = results[H][bin_key]
            diff = r['p_case1'] - baseline
            if abs(diff) >= 1:  # Show only meaningful differences
                direction = "HIGHER" if diff > 0 else "LOWER"
                print(f"  RSI {bin_key}: {r['p_case1']:.1f}% ({diff:+.1f}pp {direction})")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
If P(Case1) is significantly HIGHER in certain RSI bins, those are conditions to AVOID.
If P(Case1) is significantly LOWER in certain RSI bins, those are conditions to PREFER.

Threshold for "significant": >3pp difference from baseline (to account for noise).
""")
