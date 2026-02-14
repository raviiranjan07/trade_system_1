"""
WHEN PHASE: W2 - Volatility (ATR) vs Case Distribution

Question: Does ATR percentile affect P(Case 1)?

IMPORTANT: No direction conditioning. We analyze ATR bins for ALL trades.
We're looking for conditions where P(Case1) differs from baseline (~15%).

Run: .venv/Scripts/python.exe scripts/debug/when_volatility_vs_case.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
ATR_PERIOD = 14
ATR_PERCENTILE_BINS = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100)]
TARGET_BPS = 25  # Focus on main target
HORIZONS = [10, 30, 60]
EXTENDED_H = 500
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE W2: VOLATILITY (ATR) vs CASE DISTRIBUTION")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

# =============================================================================
# CALCULATE ATR
# =============================================================================
print("\nCalculating ATR...")

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
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

# Calculate ATR as percentage of price (normalized)
train['atr_pct'] = train['atr'] / train['close'] * 100

# Calculate percentile ranks
train['atr_percentile'] = train['atr_pct'].rank(pct=True) * 100

print(f"ATR calculated. NaN count: {train['atr'].isna().sum()}")
print(f"ATR percentile range: {train['atr_percentile'].min():.1f} - {train['atr_percentile'].max():.1f}")

# =============================================================================
# CASE CLASSIFICATION FUNCTION
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
# ANALYZE ATR BINS
# =============================================================================
print("\nAnalyzing ATR percentile bins vs Case distribution...")

close = train['close'].values
high = train['high'].values
low = train['low'].values
atr_pct = train['atr_percentile'].values
n = len(train)

target_pct = TARGET_BPS / 10000

results = {}

for H in HORIZONS:
    print(f"\n  Processing H={H}...")

    results[H] = {}

    # Sample indices
    np.random.seed(42)
    valid_start = ATR_PERIOD + 10
    sample_idx = np.random.choice(
        range(valid_start, n - EXTENDED_H),
        size=min(SAMPLE_SIZE, n - EXTENDED_H - valid_start),
        replace=False
    )

    for pct_low, pct_high in ATR_PERCENTILE_BINS:
        bin_key = f"{pct_low}-{pct_high}"
        cases = []
        maes = []

        for i in sample_idx:
            # Check if ATR percentile in this bin
            if pd.isna(atr_pct[i]):
                continue
            if not (pct_low <= atr_pct[i] < pct_high):
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
                'mae_75': np.percentile(maes_arr, 75),
                'mae_95': np.percentile(maes_arr, 95)
            }

            print(f"    ATR {bin_key}%: n={total}, P(Case1)={p_case1:.1f}%, Median MAE={np.median(maes_arr):.1f}bp")

print("\nAnalysis complete!")


# =============================================================================
# OUTPUT: MARKDOWN FORMAT
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: ATR Percentile vs Case Distribution")
print("=" * 80)

for H in HORIZONS:
    print(f"\n### Target={TARGET_BPS}bp, Horizon H={H}")
    print("\n| ATR Pctl | Count | P(Case0) | P(Case1) | P(Case2) | P(Case3) | Med MAE | 95th MAE |")
    print("|----------|-------|----------|----------|----------|----------|---------|----------|")

    for bin_key in [f"{lo}-{hi}" for lo, hi in ATR_PERCENTILE_BINS]:
        if bin_key in results[H]:
            r = results[H][bin_key]
            print(f"| {bin_key:8s} | {r['count']:5d} | {r['p_case0']:7.1f}% | "
                  f"{r['p_case1']:7.1f}% | {r['p_case2']:7.1f}% | {r['p_case3']:7.1f}% | "
                  f"{r['mae_median']:6.1f}bp | {r['mae_95']:7.1f}bp |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: ATR vs P(Case1)")
print("=" * 80)

# Compare to baseline (25-75 percentile = "normal" volatility)
H = 30  # Focus on main horizon
if "25-50" in results[H] and "50-75" in results[H]:
    baseline_low = results[H]["25-50"]['p_case1']
    baseline_high = results[H]["50-75"]['p_case1']
    baseline = (baseline_low + baseline_high) / 2
    print(f"\nBaseline P(Case1) at ATR 25-75%: {baseline:.1f}%")

    print("\nDeviation from baseline:")
    for bin_key in [f"{lo}-{hi}" for lo, hi in ATR_PERCENTILE_BINS]:
        if bin_key in results[H]:
            r = results[H][bin_key]
            diff = r['p_case1'] - baseline
            if abs(diff) >= 1:  # Show only meaningful differences
                direction = "HIGHER" if diff > 0 else "LOWER"
                marker = "***" if abs(diff) >= 3 else ""
                print(f"  ATR {bin_key}%: {r['p_case1']:.1f}% ({diff:+.1f}pp {direction}) {marker}")

# MAE Analysis
print("\n" + "-" * 40)
print("MAE by ATR Percentile (H=30):")
print("-" * 40)
for bin_key in [f"{lo}-{hi}" for lo, hi in ATR_PERCENTILE_BINS]:
    if bin_key in results[30]:
        r = results[30][bin_key]
        print(f"  ATR {bin_key}%: Median={r['mae_median']:.1f}bp, 95th={r['mae_95']:.1f}bp")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Analyzing how volatility (ATR) affects P(Case1):
- If HIGH volatility has higher P(Case1) -> AVOID high ATR periods
- If LOW volatility has higher P(Case1) -> AVOID low ATR periods
- If no significant difference -> ATR alone is not a useful filter

Also check MAE distribution - high ATR may mean larger drawdowns even if P(Case1) is similar.
""")
