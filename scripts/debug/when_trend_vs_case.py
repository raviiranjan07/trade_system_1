"""
WHEN PHASE: W4 - Trend STRENGTH vs Case Distribution

Question: Does trend STRENGTH affect P(Case 1)?

IMPORTANT: We measure STRENGTH, not direction alignment.
- NOT: "LONG aligned with uptrend" (directional assumption)
- YES: "Strong trend vs weak/choppy trend" (regime descriptor)

Metrics:
- EMA separation: |EMA50 - EMA200| / price
- Trend persistence: how long in same regime

Run: .venv/Scripts/python.exe scripts/debug/when_trend_vs_case.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_FAST = 50
EMA_SLOW = 200

# Trend strength bins (EMA separation as % of price)
STRENGTH_BINS = [
    (0, 0.5, "Very weak (0-0.5%)"),
    (0.5, 1.0, "Weak (0.5-1%)"),
    (1.0, 2.0, "Moderate (1-2%)"),
    (2.0, 4.0, "Strong (2-4%)"),
    (4.0, 100, "Very strong (>4%)")
]

TARGET_BPS = 25
HORIZONS = [10, 30, 60]
EXTENDED_H = 500
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE W4: TREND STRENGTH vs CASE DISTRIBUTION")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

# =============================================================================
# CALCULATE TREND INDICATORS
# =============================================================================
print("\nCalculating trend indicators...")

train['ema50'] = train['close'].ewm(span=EMA_FAST, adjust=False).mean()
train['ema200'] = train['close'].ewm(span=EMA_SLOW, adjust=False).mean()

# Trend strength = |EMA50 - EMA200| / price * 100 (as percentage)
train['ema_separation'] = np.abs(train['ema50'] - train['ema200']) / train['close'] * 100

# Trend direction (for reference, but NOT used for filtering)
train['trend_up'] = train['ema50'] > train['ema200']

print(f"EMA separation range: {train['ema_separation'].min():.2f}% - {train['ema_separation'].max():.2f}%")
print(f"Median EMA separation: {train['ema_separation'].median():.2f}%")

# Distribution of trend strength
print("\nTrend strength distribution:")
for low, high, label in STRENGTH_BINS:
    count = ((train['ema_separation'] >= low) & (train['ema_separation'] < high)).sum()
    pct = count / len(train) * 100
    print(f"  {label}: {count:,} ({pct:.1f}%)")


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
# ANALYZE BY TREND STRENGTH
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS BY TREND STRENGTH")
print("=" * 80)

close = train['close'].values
high = train['high'].values
low = train['low'].values
ema_sep = train['ema_separation'].values
n = len(train)

target_pct = TARGET_BPS / 10000

# Sample indices
np.random.seed(42)
valid_start = EMA_SLOW + 10
sample_idx = np.random.choice(
    range(valid_start, n - EXTENDED_H),
    size=min(SAMPLE_SIZE, n - EXTENDED_H - valid_start),
    replace=False
)

results = {}

for H in HORIZONS:
    print(f"\n  Processing H={H}...")
    results[H] = {}

    for low_thresh, high_thresh, label in STRENGTH_BINS:
        cases = []
        maes = []

        for i in sample_idx:
            if pd.isna(ema_sep[i]):
                continue
            if not (low_thresh <= ema_sep[i] < high_thresh):
                continue

            entry = close[i]
            future_highs = high[i+1:i+1+EXTENDED_H]
            future_lows = low[i+1:i+1+EXTENDED_H]

            case, mae = classify_single_bar(entry, future_highs, future_lows,
                                            target_pct, H, EXTENDED_H)
            if case >= 0:
                cases.append(case)
                maes.append(mae)

        if len(cases) > 0:
            cases_arr = np.array(cases)
            maes_arr = np.array(maes)

            total = len(cases_arr)
            p_case1 = np.sum(cases_arr == 1) / total * 100

            results[H][label] = {
                'count': total,
                'p_case0': np.sum(cases_arr == 0) / total * 100,
                'p_case1': p_case1,
                'p_case2': np.sum(cases_arr == 2) / total * 100,
                'p_case3': np.sum(cases_arr == 3) / total * 100,
                'mae_median': np.median(maes_arr),
                'mae_95': np.percentile(maes_arr, 95)
            }

            print(f"    {label}: n={total}, P(Case1)={p_case1:.1f}%")


# =============================================================================
# ANALYZE CHOPPY vs TRENDING
# =============================================================================
print("\n" + "=" * 80)
print("CHOPPY vs TRENDING MARKET")
print("=" * 80)

# Choppy = EMA separation < 1%
# Trending = EMA separation > 2%

H = 30
regime_results = {}

for regime, (low_t, high_t) in [("Choppy (<1%)", (0, 1)), ("Moderate (1-2%)", (1, 2)), ("Trending (>2%)", (2, 100))]:
    cases = []
    maes = []

    for i in sample_idx:
        if pd.isna(ema_sep[i]):
            continue
        if not (low_t <= ema_sep[i] < high_t):
            continue

        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, mae = classify_single_bar(entry, future_highs, future_lows,
                                        target_pct, H, EXTENDED_H)
        if case >= 0:
            cases.append(case)
            maes.append(mae)

    if len(cases) > 0:
        cases_arr = np.array(cases)
        maes_arr = np.array(maes)

        total = len(cases_arr)
        p_case1 = np.sum(cases_arr == 1) / total * 100

        regime_results[regime] = {
            'count': total,
            'p_case1': p_case1,
            'mae_median': np.median(maes_arr)
        }

        print(f"  {regime}: n={total}, P(Case1)={p_case1:.1f}%")


# =============================================================================
# OUTPUT: MARKDOWN FORMAT
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Trend Strength vs Case Distribution")
print("=" * 80)

for H in HORIZONS:
    print(f"\n### Target={TARGET_BPS}bp, Horizon H={H}")
    print("\n| Trend Strength | Count | P(Case0) | P(Case1) | P(Case2) | P(Case3) | Med MAE | 95th MAE |")
    print("|----------------|-------|----------|----------|----------|----------|---------|----------|")

    for _, _, label in STRENGTH_BINS:
        if label in results[H]:
            r = results[H][label]
            print(f"| {label:20s} | {r['count']:5d} | {r['p_case0']:7.1f}% | "
                  f"{r['p_case1']:7.1f}% | {r['p_case2']:7.1f}% | {r['p_case3']:7.1f}% | "
                  f"{r['mae_median']:6.1f}bp | {r['mae_95']:7.1f}bp |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: Trend Strength vs P(Case1)")
print("=" * 80)

H = 30
all_p_case1 = [r['p_case1'] for r in results[H].values()]
baseline = np.mean(all_p_case1)
print(f"\nBaseline P(Case1) (avg): {baseline:.1f}%")

print("\nDeviation from baseline:")
for _, _, label in STRENGTH_BINS:
    if label in results[H]:
        r = results[H][label]
        diff = r['p_case1'] - baseline
        if abs(diff) >= 1:
            direction = "HIGHER" if diff > 0 else "LOWER"
            marker = "***" if abs(diff) >= 3 else ""
            print(f"  {label}: {r['p_case1']:.1f}% ({diff:+.1f}pp {direction}) {marker}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Analyzing how trend STRENGTH (not direction) affects P(Case1):
- If weak/choppy markets have higher P(Case1) -> AVOID choppy markets
- If strong trends have lower P(Case1) -> PREFER strong trends
- This is direction-neutral - we're not predicting up/down
""")
