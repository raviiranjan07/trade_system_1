"""
WHEN PHASE: W3 - Time Patterns vs Case Distribution

Question: Does time of day/week affect P(Case 1)?

Sessions to analyze:
- Asia (00:00-08:00 UTC) - low liquidity
- Europe (08:00-14:00 UTC) - medium liquidity
- US (14:00-21:00 UTC) - high liquidity
- Weekend (Sat-Sun) - different dynamics

Run: .venv/Scripts/python.exe scripts/debug/when_time_vs_case.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
# Time bins (UTC hours)
HOUR_BINS = [
    (0, 4, "00-04 (Asia night)"),
    (4, 8, "04-08 (Asia)"),
    (8, 12, "08-12 (Europe)"),
    (12, 16, "12-16 (US open)"),
    (16, 20, "16-20 (US)"),
    (20, 24, "20-24 (US close)")
]

# Sessions
SESSIONS = {
    "Asia": (0, 8),
    "Europe": (8, 14),
    "US": (14, 21),
    "Off-hours": (21, 24)  # Plus 0-8 handled separately
}

TARGET_BPS = 25
HORIZONS = [10, 30, 60]
EXTENDED_H = 500
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHEN PHASE W3: TIME PATTERNS vs CASE DISTRIBUTION")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

# Extract time features
train['hour'] = train.index.hour
train['day_of_week'] = train.index.dayofweek  # 0=Monday, 6=Sunday
train['is_weekend'] = train['day_of_week'].isin([5, 6])

print(f"Hour range: {train['hour'].min()} - {train['hour'].max()}")
print(f"Weekend bars: {train['is_weekend'].sum():,} ({train['is_weekend'].mean()*100:.1f}%)")

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
# ANALYZE BY HOUR
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS BY HOUR (UTC)")
print("=" * 80)

close = train['close'].values
high = train['high'].values
low = train['low'].values
hour = train['hour'].values
dow = train['day_of_week'].values
n = len(train)

target_pct = TARGET_BPS / 10000

# Sample indices
np.random.seed(42)
valid_start = 100
sample_idx = np.random.choice(
    range(valid_start, n - EXTENDED_H),
    size=min(SAMPLE_SIZE, n - EXTENDED_H - valid_start),
    replace=False
)

H = 30  # Focus on main horizon

# By hour bin
print(f"\nProcessing H={H}...")
hour_results = {}

for h_start, h_end, label in HOUR_BINS:
    cases = []
    maes = []

    for i in sample_idx:
        if not (h_start <= hour[i] < h_end):
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

        hour_results[label] = {
            'count': total,
            'p_case0': np.sum(cases_arr == 0) / total * 100,
            'p_case1': p_case1,
            'p_case2': np.sum(cases_arr == 2) / total * 100,
            'p_case3': np.sum(cases_arr == 3) / total * 100,
            'mae_median': np.median(maes_arr)
        }

        print(f"  {label}: n={total}, P(Case1)={p_case1:.1f}%")


# =============================================================================
# ANALYZE BY DAY OF WEEK
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS BY DAY OF WEEK")
print("=" * 80)

days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_results = {}

for d in range(7):
    cases = []
    maes = []

    for i in sample_idx:
        if dow[i] != d:
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

        day_results[days[d]] = {
            'count': total,
            'p_case0': np.sum(cases_arr == 0) / total * 100,
            'p_case1': p_case1,
            'p_case2': np.sum(cases_arr == 2) / total * 100,
            'p_case3': np.sum(cases_arr == 3) / total * 100,
            'mae_median': np.median(maes_arr)
        }

        print(f"  {days[d]}: n={total}, P(Case1)={p_case1:.1f}%")


# =============================================================================
# ANALYZE WEEKEND vs WEEKDAY
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS: WEEKEND vs WEEKDAY")
print("=" * 80)

weekend_results = {}

for is_wknd, label in [(False, "Weekday"), (True, "Weekend")]:
    cases = []
    maes = []

    for i in sample_idx:
        if (dow[i] >= 5) != is_wknd:
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

        weekend_results[label] = {
            'count': total,
            'p_case1': p_case1,
            'mae_median': np.median(maes_arr)
        }

        print(f"  {label}: n={total}, P(Case1)={p_case1:.1f}%")


# =============================================================================
# OUTPUT: MARKDOWN FORMAT
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Time Patterns vs Case Distribution")
print("=" * 80)

print(f"\n### By Hour (Target={TARGET_BPS}bp, H={H})")
print("\n| Hour (UTC) | Count | P(Case0) | P(Case1) | P(Case2) | P(Case3) | Med MAE |")
print("|------------|-------|----------|----------|----------|----------|---------|")

for label, r in hour_results.items():
    print(f"| {label:18s} | {r['count']:5d} | {r['p_case0']:7.1f}% | "
          f"{r['p_case1']:7.1f}% | {r['p_case2']:7.1f}% | {r['p_case3']:7.1f}% | "
          f"{r['mae_median']:6.1f}bp |")

print(f"\n### By Day of Week")
print("\n| Day | Count | P(Case1) | Med MAE |")
print("|-----|-------|----------|---------|")

for day in days:
    if day in day_results:
        r = day_results[day]
        print(f"| {day:3s} | {r['count']:5d} | {r['p_case1']:7.1f}% | {r['mae_median']:6.1f}bp |")

print(f"\n### Weekend vs Weekday")
print("\n| Period | Count | P(Case1) | Med MAE |")
print("|--------|-------|----------|---------|")

for label, r in weekend_results.items():
    print(f"| {label:8s} | {r['count']:5d} | {r['p_case1']:7.1f}% | {r['mae_median']:6.1f}bp |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS: Time vs P(Case1)")
print("=" * 80)

# Calculate baseline (overall average)
all_p_case1 = [r['p_case1'] for r in hour_results.values()]
baseline = np.mean(all_p_case1)
print(f"\nBaseline P(Case1) (avg across hours): {baseline:.1f}%")

print("\nHour deviations from baseline:")
for label, r in sorted(hour_results.items(), key=lambda x: x[1]['p_case1'], reverse=True):
    diff = r['p_case1'] - baseline
    if abs(diff) >= 1:
        direction = "HIGHER" if diff > 0 else "LOWER"
        marker = "***" if abs(diff) >= 2 else ""
        print(f"  {label}: {r['p_case1']:.1f}% ({diff:+.1f}pp {direction}) {marker}")

print("\nDay deviations:")
day_baseline = np.mean([r['p_case1'] for r in day_results.values()])
for day in days:
    if day in day_results:
        r = day_results[day]
        diff = r['p_case1'] - day_baseline
        if abs(diff) >= 1:
            direction = "HIGHER" if diff > 0 else "LOWER"
            print(f"  {day}: {r['p_case1']:.1f}% ({diff:+.1f}pp {direction})")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Analyzing how time affects P(Case1):
- If certain hours have significantly higher P(Case1) -> AVOID those hours
- If weekend has different P(Case1) -> Consider weekend-specific rules
- Threshold for significance: >2pp difference from baseline
""")
