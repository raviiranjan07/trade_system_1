"""
ANALYSIS: EMA Bounce/Drop Magnitude

Questions:
1. When price touches EMA (support test in uptrend), how big is the bounce?
2. When price touches EMA (resistance test in downtrend), how big is the drop?
3. After bounce/drop, how big is the subsequent correction?

This measures the PATH PATTERN, not just direction.

Run: .venv/Scripts/python.exe scripts/debug/analysis_ema_bounce_magnitude.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [20, 50, 100, 200]
NEAR_THRESHOLD_BPS = [5, 10, 15, 20]  # How close to EMA to count as "touch"
BOUNCE_HORIZONS = [3, 5, 10, 15, 30, 60]  # How far to measure bounce
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE/DROP MAGNITUDE")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

# =============================================================================
# CALCULATE EMAs
# =============================================================================
print("\nCalculating EMAs...")
emas = {}
for period in EMA_PERIODS:
    emas[period] = train['close'].ewm(span=period, adjust=False).mean().values
print("EMAs calculated.")

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_ema_bounce(ema_period, near_bp, bounce_horizon):
    """
    Analyze bounce/drop magnitude when price touches EMA.

    Support Test (Uptrend): Price above EMA, dips to touch EMA, bounces UP
    Resistance Test (Downtrend): Price below EMA, rises to touch EMA, drops DOWN

    Returns: bounce magnitude stats, correction stats
    """
    ema = emas[ema_period]
    near_pct = near_bp / 10000

    support_bounces = []  # MFE after support test
    support_corrections = []  # MAE after support test (subsequent pullback)

    resistance_drops = []  # MFE (down) after resistance test
    resistance_corrections = []  # MAE (up) after resistance test

    # Sample random indices
    np.random.seed(42)
    max_h = bounce_horizon + 100
    valid_start = max(EMA_PERIODS) + 10

    sample_idx = np.random.choice(
        range(valid_start, n - max_h),
        size=min(SAMPLE_SIZE, n - max_h - valid_start),
        replace=False
    )

    for i in sample_idx:
        entry = close[i]
        ema_val = ema[i]

        # Check if price is "near" EMA
        distance_pct = abs(entry - ema_val) / ema_val
        if distance_pct > near_pct:
            continue

        # Determine trend direction (using EMA slope over last 10 bars)
        if i < 10:
            continue
        ema_slope = (ema[i] - ema[i-10]) / ema[i-10] * 10000  # in bps

        # Support Test: Price near EMA, EMA sloping UP (uptrend)
        # We expect price to bounce UP
        if ema_slope > 2:  # Uptrend (EMA rising)
            # Measure MFE (max favorable = max UP move)
            mfe = 0
            mae = 0
            for j in range(1, bounce_horizon + 1):
                if i + j >= n:
                    break
                # Max UP move (bounce)
                up_move = (high[i + j] - entry) / entry * 10000
                mfe = max(mfe, up_move)
                # Max DOWN move (subsequent correction)
                down_move = (entry - low[i + j]) / entry * 10000
                mae = max(mae, down_move)

            support_bounces.append(mfe)
            support_corrections.append(mae)

        # Resistance Test: Price near EMA, EMA sloping DOWN (downtrend)
        # We expect price to drop DOWN
        elif ema_slope < -2:  # Downtrend (EMA falling)
            # Measure MFE (max favorable = max DOWN move)
            mfe = 0
            mae = 0
            for j in range(1, bounce_horizon + 1):
                if i + j >= n:
                    break
                # Max DOWN move (drop)
                down_move = (entry - low[i + j]) / entry * 10000
                mfe = max(mfe, down_move)
                # Max UP move (subsequent correction)
                up_move = (high[i + j] - entry) / entry * 10000
                mae = max(mae, up_move)

            resistance_drops.append(mfe)
            resistance_corrections.append(mae)

    return {
        'support': {
            'count': len(support_bounces),
            'bounce_median': np.median(support_bounces) if support_bounces else 0,
            'bounce_75': np.percentile(support_bounces, 75) if support_bounces else 0,
            'bounce_90': np.percentile(support_bounces, 90) if support_bounces else 0,
            'correction_median': np.median(support_corrections) if support_corrections else 0,
            'correction_75': np.percentile(support_corrections, 75) if support_corrections else 0,
        },
        'resistance': {
            'count': len(resistance_drops),
            'drop_median': np.median(resistance_drops) if resistance_drops else 0,
            'drop_75': np.percentile(resistance_drops, 75) if resistance_drops else 0,
            'drop_90': np.percentile(resistance_drops, 90) if resistance_drops else 0,
            'correction_median': np.median(resistance_corrections) if resistance_corrections else 0,
            'correction_75': np.percentile(resistance_corrections, 75) if resistance_corrections else 0,
        }
    }


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\nRunning analysis...")

results = {}

# Focus on key combinations
key_combos = [
    (20, 10, 10),   # EMA20, 10bp near, H=10
    (20, 10, 30),   # EMA20, 10bp near, H=30
    (50, 10, 10),   # EMA50, 10bp near, H=10
    (50, 10, 30),   # EMA50, 10bp near, H=30
    (50, 15, 30),   # EMA50, 15bp near, H=30
    (50, 20, 60),   # EMA50, 20bp near, H=60
    (100, 15, 30),  # EMA100, 15bp near, H=30
    (100, 20, 60),  # EMA100, 20bp near, H=60
    (200, 20, 30),  # EMA200, 20bp near, H=30
    (200, 20, 60),  # EMA200, 20bp near, H=60
]

for ema_period, near_bp, horizon in key_combos:
    key = f"EMA{ema_period}_near{near_bp}_H{horizon}"
    results[key] = analyze_ema_bounce(ema_period, near_bp, horizon)
    print(f"  {key} done")

print("\nAnalysis complete!")


# =============================================================================
# OUTPUT: SUPPORT TEST (Bounce in Uptrend)
# =============================================================================
print("\n" + "=" * 80)
print("SUPPORT TEST: Bounce Magnitude in UPTREND")
print("(Price near EMA, EMA rising -> expect bounce UP)")
print("=" * 80)

print("\n### Support Test Results (Bounce UP from EMA)")
print("\n| EMA | Near | H | Count | Bounce Median | Bounce 75th | Bounce 90th | Correction Median |")
print("|-----|------|---|-------|---------------|-------------|-------------|-------------------|")

for key, data in results.items():
    parts = key.split('_')
    ema = parts[0]
    near = parts[1].replace('near', '')
    h = parts[2]
    s = data['support']
    if s['count'] > 100:
        print(f"| {ema} | {near}bp | {h} | {s['count']} | {s['bounce_median']:.1f}bp | {s['bounce_75']:.1f}bp | {s['bounce_90']:.1f}bp | {s['correction_median']:.1f}bp |")


# =============================================================================
# OUTPUT: RESISTANCE TEST (Drop in Downtrend)
# =============================================================================
print("\n" + "=" * 80)
print("RESISTANCE TEST: Drop Magnitude in DOWNTREND")
print("(Price near EMA, EMA falling -> expect drop DOWN)")
print("=" * 80)

print("\n### Resistance Test Results (Drop DOWN from EMA)")
print("\n| EMA | Near | H | Count | Drop Median | Drop 75th | Drop 90th | Correction Median |")
print("|-----|------|---|-------|-------------|-----------|-----------|-------------------|")

for key, data in results.items():
    parts = key.split('_')
    ema = parts[0]
    near = parts[1].replace('near', '')
    h = parts[2]
    r = data['resistance']
    if r['count'] > 100:
        print(f"| {ema} | {near}bp | {h} | {r['count']} | {r['drop_median']:.1f}bp | {r['drop_75']:.1f}bp | {r['drop_90']:.1f}bp | {r['correction_median']:.1f}bp |")


# =============================================================================
# DETAILED ANALYSIS: EMA50 with varying horizons
# =============================================================================
print("\n" + "=" * 80)
print("DETAILED: EMA50 Bounce/Drop by Horizon")
print("=" * 80)

detailed_results = {}
for H in BOUNCE_HORIZONS:
    key = f"EMA50_H{H}"
    detailed_results[key] = analyze_ema_bounce(50, 15, H)

print("\n### EMA50 Support Test (15bp near) by Horizon")
print("\n| Horizon | Count | Bounce Median | Bounce 75th | Bounce 90th | Correction Median |")
print("|---------|-------|---------------|-------------|-------------|-------------------|")

for H in BOUNCE_HORIZONS:
    key = f"EMA50_H{H}"
    s = detailed_results[key]['support']
    if s['count'] > 50:
        print(f"| H={H:2d} | {s['count']:5d} | {s['bounce_median']:12.1f}bp | {s['bounce_75']:10.1f}bp | {s['bounce_90']:10.1f}bp | {s['correction_median']:16.1f}bp |")

print("\n### EMA50 Resistance Test (15bp near) by Horizon")
print("\n| Horizon | Count | Drop Median | Drop 75th | Drop 90th | Correction Median |")
print("|---------|-------|-------------|-----------|-----------|-------------------|")

for H in BOUNCE_HORIZONS:
    key = f"EMA50_H{H}"
    r = detailed_results[key]['resistance']
    if r['count'] > 50:
        print(f"| H={H:2d} | {r['count']:5d} | {r['drop_median']:10.1f}bp | {r['drop_75']:8.1f}bp | {r['drop_90']:8.1f}bp | {r['correction_median']:16.1f}bp |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Calculate average bounce/drop across key combos
support_bounces = [r['support']['bounce_median'] for r in results.values() if r['support']['count'] > 100]
resistance_drops = [r['resistance']['drop_median'] for r in results.values() if r['resistance']['count'] > 100]

print("\n1. Average bounce magnitude (support tests):")
if support_bounces:
    print(f"   Median bounce: {np.mean(support_bounces):.1f}bp")

print("\n2. Average drop magnitude (resistance tests):")
if resistance_drops:
    print(f"   Median drop: {np.mean(resistance_drops):.1f}bp")

print("\n3. Bounce vs Correction ratio:")
for key, data in results.items():
    s = data['support']
    if s['count'] > 100 and s['correction_median'] > 0:
        ratio = s['bounce_median'] / s['correction_median']
        print(f"   {key}: Bounce/Correction = {ratio:.2f}x")
        break

print("\n4. Key finding:")
print("   - Bounce/Drop magnitude tells us expected move size after EMA touch")
print("   - Correction magnitude tells us expected pullback")
print("   - Ratio of bounce:correction tells us risk:reward of EMA bounce trade")
