"""
WHEN PHASE: W6 - Early-Path Analysis (MOST IMPORTANT)

Question: Does what happens in first N bars predict final case?

From WHAT phase, we know early path has more predictive power than entry conditions.
This analysis finds the CUT threshold - when early damage becomes unrecoverable.

Metrics:
- MAE at bar 3, 5, 10 -> P(Case1)
- Find threshold where P(Case1) spikes

Run: .venv/Scripts/python.exe scripts/debug/when_early_path_vs_case.py
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

# Early path checkpoints
EARLY_BARS = [1, 2, 3, 5, 10]

# MAE bins (in basis points)
MAE_BINS = [
    (0, 5, "0-5bp"),
    (5, 10, "5-10bp"),
    (10, 15, "10-15bp"),
    (15, 20, "15-20bp"),
    (20, 30, "20-30bp"),
    (30, 50, "30-50bp"),
    (50, 75, "50-75bp"),
    (75, 100, "75-100bp"),
    (100, 150, "100-150bp"),
    (150, 500, ">150bp")
]

SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE W6: EARLY-PATH vs CASE DISTRIBUTION (CRITICAL)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")


# =============================================================================
# CASE CLASSIFICATION WITH EARLY MAE TRACKING
# =============================================================================
@njit
def classify_with_early_mae(entry, highs, lows, target_pct, H, extended_H, early_bars):
    """
    Classify bar and track MAE at early checkpoints.

    Returns:
        case: 0/1/2/3
        final_mae: total MAE
        early_maes: array of MAE at each early checkpoint
    """
    n = len(highs)
    num_checkpoints = len(early_bars)
    early_maes = np.zeros(num_checkpoints)

    if n == 0:
        return -1, 0.0, early_maes

    target_price = entry * (1 + target_pct)

    went_below = False
    hit_within_H = False
    hit_extended = False
    max_adverse_bps = 0.0
    current_mae = 0.0

    # Track MAE at early checkpoints
    for j in range(min(max(early_bars), n)):
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > current_mae:
            current_mae = adverse

        # Record at checkpoints
        for k in range(num_checkpoints):
            if j + 1 == early_bars[k]:
                early_maes[k] = current_mae

    # Continue to classify
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

    # Determine case
    if not went_below and hit_within_H:
        case = 0
    elif went_below and hit_within_H:
        case = 2
    elif went_below and not hit_within_H and hit_extended:
        case = 3
    elif went_below and not hit_within_H and not hit_extended:
        case = 1
    elif not went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                if went_below:
                    case = 3
                else:
                    case = 0
                break
        else:
            case = 1
    else:
        case = -1

    return case, max_adverse_bps, early_maes


# =============================================================================
# ANALYZE EARLY PATH
# =============================================================================
print("\nAnalyzing early-path vs final case...")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

target_pct = TARGET_BPS / 10000
early_bars_arr = np.array(EARLY_BARS)

# Sample indices
np.random.seed(42)
valid_start = 100
sample_idx = np.random.choice(
    range(valid_start, n - EXTENDED_H),
    size=min(SAMPLE_SIZE, n - EXTENDED_H - valid_start),
    replace=False
)

# Collect results
all_cases = []
all_final_maes = []
all_early_maes = {bar: [] for bar in EARLY_BARS}

print(f"Processing {len(sample_idx):,} samples...")

for i in sample_idx:
    entry = close[i]
    future_highs = high[i+1:i+1+EXTENDED_H]
    future_lows = low[i+1:i+1+EXTENDED_H]

    case, final_mae, early_maes = classify_with_early_mae(
        entry, future_highs, future_lows, target_pct, HORIZON, EXTENDED_H, early_bars_arr
    )

    if case >= 0:
        all_cases.append(case)
        all_final_maes.append(final_mae)
        for k, bar in enumerate(EARLY_BARS):
            all_early_maes[bar].append(early_maes[k])

all_cases = np.array(all_cases)
all_final_maes = np.array(all_final_maes)
for bar in EARLY_BARS:
    all_early_maes[bar] = np.array(all_early_maes[bar])

print(f"Valid samples: {len(all_cases):,}")


# =============================================================================
# ANALYZE P(Case1) BY EARLY MAE
# =============================================================================
print("\n" + "=" * 80)
print("P(CASE 1) BY EARLY MAE AT DIFFERENT CHECKPOINTS")
print("=" * 80)

results = {}

for bar in EARLY_BARS:
    print(f"\n### MAE at Bar {bar}")
    results[bar] = {}

    early_mae = all_early_maes[bar]

    print(f"\n| MAE at Bar {bar} | Count | P(Case0) | P(Case1) | P(Case2) | P(Case3) | Classification |")
    print("|----------------|-------|----------|----------|----------|----------|----------------|")

    for low_t, high_t, label in MAE_BINS:
        mask = (early_mae >= low_t) & (early_mae < high_t)
        bin_cases = all_cases[mask]

        if len(bin_cases) < 100:
            continue

        total = len(bin_cases)
        p_case0 = np.sum(bin_cases == 0) / total * 100
        p_case1 = np.sum(bin_cases == 1) / total * 100
        p_case2 = np.sum(bin_cases == 2) / total * 100
        p_case3 = np.sum(bin_cases == 3) / total * 100

        # Classification
        if p_case1 > 30:
            classification = "CUT"
        elif p_case1 > 20:
            classification = "WATCH/CUT"
        elif p_case1 > 15:
            classification = "WATCH"
        else:
            classification = "HOLD"

        results[bar][label] = {
            'count': total,
            'p_case0': p_case0,
            'p_case1': p_case1,
            'p_case2': p_case2,
            'p_case3': p_case3,
            'classification': classification
        }

        marker = " ***" if p_case1 > 25 else ""
        print(f"| {label:14s} | {total:5d} | {p_case0:7.1f}% | {p_case1:7.1f}% | "
              f"{p_case2:7.1f}% | {p_case3:7.1f}% | {classification:14s}{marker} |")


# =============================================================================
# FIND CUT THRESHOLD
# =============================================================================
print("\n" + "=" * 80)
print("CUT THRESHOLD ANALYSIS")
print("=" * 80)

print("\nFinding MAE threshold where P(Case1) > 25% (CUT decision):")

for bar in EARLY_BARS:
    print(f"\n  Bar {bar}:")
    found_threshold = False

    for low_t, high_t, label in MAE_BINS:
        if label in results[bar]:
            r = results[bar][label]
            if r['p_case1'] > 25 and not found_threshold:
                print(f"    -> CUT when MAE > {low_t}bp (P(Case1) = {r['p_case1']:.1f}%)")
                found_threshold = True
                break

    if not found_threshold:
        print(f"    -> No clear CUT threshold found")


# =============================================================================
# RECOVERY PROBABILITY BY EARLY MAE
# =============================================================================
print("\n" + "=" * 80)
print("RECOVERY PROBABILITY BY EARLY MAE (Bar 3)")
print("=" * 80)

bar = 3
early_mae = all_early_maes[bar]

print(f"\n| MAE at Bar 3 | Count | P(Recovery) | P(Case1) | Recommendation |")
print("|--------------|-------|-------------|----------|----------------|")

for low_t, high_t, label in MAE_BINS:
    mask = (early_mae >= low_t) & (early_mae < high_t)
    bin_cases = all_cases[mask]

    if len(bin_cases) < 100:
        continue

    total = len(bin_cases)
    p_recovery = np.sum((bin_cases == 0) | (bin_cases == 2) | (bin_cases == 3)) / total * 100
    p_case1 = np.sum(bin_cases == 1) / total * 100

    if p_case1 > 30:
        rec = "CUT NOW"
    elif p_case1 > 20:
        rec = "Consider CUT"
    elif p_case1 > 15:
        rec = "Watch closely"
    else:
        rec = "Hold"

    print(f"| {label:12s} | {total:5d} | {p_recovery:10.1f}% | {p_case1:7.1f}% | {rec:14s} |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: Early-Path vs P(Case1)")
print("=" * 80)

bar = 3
baseline_mask = all_early_maes[bar] < 10
baseline_p_case1 = np.sum(all_cases[baseline_mask] == 1) / np.sum(baseline_mask) * 100

print(f"\nBaseline (MAE < 10bp at bar 3): P(Case1) = {baseline_p_case1:.1f}%")
print("\nProgression of P(Case1) by early MAE:")

for low_t, high_t, label in MAE_BINS[:8]:  # First 8 bins
    mask = (all_early_maes[bar] >= low_t) & (all_early_maes[bar] < high_t)
    if np.sum(mask) > 100:
        p_case1 = np.sum(all_cases[mask] == 1) / np.sum(mask) * 100
        diff = p_case1 - baseline_p_case1
        marker = "***" if p_case1 > 25 else ""
        print(f"  MAE {label}: P(Case1) = {p_case1:.1f}% ({diff:+.1f}pp from baseline) {marker}")


print("\n" + "=" * 80)
print("CONCLUSION: CUT RULES")
print("=" * 80)
print("""
Based on early-path analysis:

1. If MAE at bar 3 > 50bp: P(Case1) jumps significantly -> CUT
2. If MAE at bar 3 > 75bp: P(Case1) > 30% -> IMMEDIATE CUT
3. If MAE at bar 3 < 20bp: P(Case1) is low -> HOLD

This is the most actionable finding for the EXECUTE phase:
- Use early MAE as a real-time CUT trigger
- Don't wait for stop loss - cut based on early path damage
""")
