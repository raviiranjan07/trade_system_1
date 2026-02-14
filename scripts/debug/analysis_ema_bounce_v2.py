"""
ANALYSIS: EMA Bounce/Rejection Magnitude (V2 - Corrected)

Properly detects:
1. BOUNCE: Price comes DOWN to EMA, touches, bounces UP
2. REJECTION: Price comes UP to EMA, touches, falls DOWN

Key improvement: Confirms actual bounce/rejection before measuring magnitude.

Run: .venv/Scripts/python.exe scripts/debug/analysis_ema_bounce_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [5, 9, 10, 12, 20, 50, 100, 200]
TOUCH_THRESHOLD_BPS = 15  # How close to EMA to count as "touch"
APPROACH_BARS = 5  # How many bars to look back for approach
BOUNCE_CONFIRM_BPS = 10  # Min move away from EMA to confirm bounce/rejection
MEASURE_HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]  # All tested horizons
SAMPLE_SIZE = 300000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE/REJECTION MAGNITUDE (V2 - CORRECTED)")
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
# DETECT BOUNCE/REJECTION PATTERNS
# =============================================================================
def detect_ema_patterns(ema_period):
    """
    Detect actual EMA bounce and rejection patterns.

    BOUNCE (Support):
    - Price was ABOVE EMA (last N bars)
    - Price comes DOWN and touches EMA
    - Price bounces UP (moves away from EMA upward)

    REJECTION (Resistance):
    - Price was BELOW EMA (last N bars)
    - Price comes UP and touches EMA
    - Price falls DOWN (moves away from EMA downward)
    """
    ema = emas[ema_period]
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    confirm_pct = BOUNCE_CONFIRM_BPS / 10000

    bounces = []  # (index, bounce_magnitude by horizon, subsequent_correction)
    rejections = []  # (index, rejection_magnitude by horizon, subsequent_correction)

    # Sample random indices
    np.random.seed(42)
    max_h = max(MEASURE_HORIZONS) + 50
    valid_start = max(EMA_PERIODS) + APPROACH_BARS + 10

    sample_idx = np.random.choice(
        range(valid_start, n - max_h),
        size=min(SAMPLE_SIZE, n - max_h - valid_start),
        replace=False
    )

    for i in sample_idx:
        current_close = close[i]
        current_ema = ema[i]

        # Check if price is "touching" EMA
        distance_pct = (current_close - current_ema) / current_ema
        if abs(distance_pct) > touch_pct:
            continue  # Not touching EMA

        # Check approach direction (where did price come from?)
        # Look at average position relative to EMA over last APPROACH_BARS
        approach_positions = []
        for j in range(1, APPROACH_BARS + 1):
            if i - j >= 0:
                rel_pos = (close[i - j] - ema[i - j]) / ema[i - j]
                approach_positions.append(rel_pos)

        if not approach_positions:
            continue

        avg_approach = np.mean(approach_positions)

        # Determine pattern type
        # BOUNCE: Approached from ABOVE (avg_approach > 0), now touching
        # REJECTION: Approached from BELOW (avg_approach < 0), now touching

        if avg_approach > touch_pct:  # Came from ABOVE - potential BOUNCE
            # Check if price actually bounces UP
            # Look for confirmation: price moves UP from EMA
            bounce_confirmed = False
            confirm_bar = None

            for j in range(1, 11):  # Look up to 10 bars for confirmation
                if i + j >= n:
                    break
                future_rel = (close[i + j] - ema[i + j]) / ema[i + j]
                if future_rel > confirm_pct:  # Moved UP away from EMA
                    bounce_confirmed = True
                    confirm_bar = j
                    break
                elif future_rel < -confirm_pct:  # Broke DOWN through EMA
                    break  # Not a bounce - it's a break

            if bounce_confirmed:
                # Measure bounce magnitude from the touch point
                entry = current_close
                results = {'confirm_bar': confirm_bar}

                for H in MEASURE_HORIZONS:
                    if i + H >= n:
                        continue

                    # Max UP move (bounce magnitude)
                    mfe = 0
                    mae = 0
                    for j in range(1, H + 1):
                        if i + j >= n:
                            break
                        up_move = (high[i + j] - entry) / entry * 10000
                        down_move = (entry - low[i + j]) / entry * 10000
                        mfe = max(mfe, up_move)
                        mae = max(mae, down_move)

                    results[f'bounce_H{H}'] = mfe
                    results[f'correction_H{H}'] = mae

                bounces.append(results)

        elif avg_approach < -touch_pct:  # Came from BELOW - potential REJECTION
            # Check if price actually falls DOWN
            rejection_confirmed = False
            confirm_bar = None

            for j in range(1, 11):
                if i + j >= n:
                    break
                future_rel = (close[i + j] - ema[i + j]) / ema[i + j]
                if future_rel < -confirm_pct:  # Moved DOWN away from EMA
                    rejection_confirmed = True
                    confirm_bar = j
                    break
                elif future_rel > confirm_pct:  # Broke UP through EMA
                    break  # Not a rejection - it's a break

            if rejection_confirmed:
                # Measure rejection magnitude
                entry = current_close
                results = {'confirm_bar': confirm_bar}

                for H in MEASURE_HORIZONS:
                    if i + H >= n:
                        continue

                    # Max DOWN move (rejection magnitude)
                    mfe = 0
                    mae = 0
                    for j in range(1, H + 1):
                        if i + j >= n:
                            break
                        down_move = (entry - low[i + j]) / entry * 10000
                        up_move = (high[i + j] - entry) / entry * 10000
                        mfe = max(mfe, down_move)
                        mae = max(mae, up_move)

                    results[f'drop_H{H}'] = mfe
                    results[f'correction_H{H}'] = mae

                rejections.append(results)

    return bounces, rejections


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\nDetecting EMA patterns...")

all_results = {}
for ema_period in EMA_PERIODS:
    print(f"  Analyzing EMA{ema_period}...")
    bounces, rejections = detect_ema_patterns(ema_period)
    all_results[ema_period] = {
        'bounces': bounces,
        'rejections': rejections
    }
    print(f"    Found {len(bounces):,} bounces, {len(rejections):,} rejections")

print("\nAnalysis complete!")


# =============================================================================
# OUTPUT: BOUNCE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("EMA BOUNCE RESULTS (Price came from ABOVE, touched EMA, bounced UP)")
print("=" * 80)

print("\n### Bounce Magnitude by EMA and Horizon")
print("\n| EMA | H | Count | Bounce Median | Bounce 75th | Bounce 90th | Correction Median | Bounce:Correction |")
print("|-----|---|-------|---------------|-------------|-------------|-------------------|-------------------|")

for ema_period in EMA_PERIODS:
    bounces = all_results[ema_period]['bounces']
    if len(bounces) < 100:
        continue

    for H in MEASURE_HORIZONS:
        key = f'bounce_H{H}'
        corr_key = f'correction_H{H}'

        bounce_vals = [b[key] for b in bounces if key in b]
        corr_vals = [b[corr_key] for b in bounces if corr_key in b]

        if len(bounce_vals) < 100:
            continue

        bounce_med = np.median(bounce_vals)
        bounce_75 = np.percentile(bounce_vals, 75)
        bounce_90 = np.percentile(bounce_vals, 90)
        corr_med = np.median(corr_vals)
        ratio = bounce_med / corr_med if corr_med > 0 else 0

        print(f"| EMA{ema_period} | H{H} | {len(bounce_vals):,} | {bounce_med:.1f}bp | {bounce_75:.1f}bp | {bounce_90:.1f}bp | {corr_med:.1f}bp | {ratio:.2f}:1 |")


# =============================================================================
# OUTPUT: REJECTION RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("EMA REJECTION RESULTS (Price came from BELOW, touched EMA, fell DOWN)")
print("=" * 80)

print("\n### Rejection (Drop) Magnitude by EMA and Horizon")
print("\n| EMA | H | Count | Drop Median | Drop 75th | Drop 90th | Correction Median | Drop:Correction |")
print("|-----|---|-------|-------------|-----------|-----------|-------------------|-----------------|")

for ema_period in EMA_PERIODS:
    rejections = all_results[ema_period]['rejections']
    if len(rejections) < 100:
        continue

    for H in MEASURE_HORIZONS:
        key = f'drop_H{H}'
        corr_key = f'correction_H{H}'

        drop_vals = [r[key] for r in rejections if key in r]
        corr_vals = [r[corr_key] for r in rejections if corr_key in r]

        if len(drop_vals) < 100:
            continue

        drop_med = np.median(drop_vals)
        drop_75 = np.percentile(drop_vals, 75)
        drop_90 = np.percentile(drop_vals, 90)
        corr_med = np.median(corr_vals)
        ratio = drop_med / corr_med if corr_med > 0 else 0

        print(f"| EMA{ema_period} | H{H} | {len(drop_vals):,} | {drop_med:.1f}bp | {drop_75:.1f}bp | {drop_90:.1f}bp | {corr_med:.1f}bp | {ratio:.2f}:1 |")


# =============================================================================
# BOUNCE vs BREAK RATE
# =============================================================================
print("\n" + "=" * 80)
print("BOUNCE vs BREAK RATE")
print("(When price touches EMA from above, does it bounce or break through?)")
print("=" * 80)

def calculate_bounce_break_rate(ema_period):
    """Calculate what % of EMA touches result in bounce vs break."""
    ema = emas[ema_period]
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    confirm_pct = BOUNCE_CONFIRM_BPS / 10000

    np.random.seed(42)
    max_h = 20
    valid_start = max(EMA_PERIODS) + APPROACH_BARS + 10

    sample_idx = np.random.choice(
        range(valid_start, n - max_h),
        size=min(SAMPLE_SIZE, n - max_h - valid_start),
        replace=False
    )

    from_above_bounce = 0
    from_above_break = 0
    from_above_neutral = 0

    from_below_reject = 0
    from_below_break = 0
    from_below_neutral = 0

    for i in sample_idx:
        current_close = close[i]
        current_ema = ema[i]

        distance_pct = (current_close - current_ema) / current_ema
        if abs(distance_pct) > touch_pct:
            continue

        approach_positions = []
        for j in range(1, APPROACH_BARS + 1):
            if i - j >= 0:
                rel_pos = (close[i - j] - ema[i - j]) / ema[i - j]
                approach_positions.append(rel_pos)

        if not approach_positions:
            continue

        avg_approach = np.mean(approach_positions)

        # Check outcome
        outcome = None
        for j in range(1, 11):
            if i + j >= n:
                break
            future_rel = (close[i + j] - ema[i + j]) / ema[i + j]
            if future_rel > confirm_pct:
                outcome = 'up'
                break
            elif future_rel < -confirm_pct:
                outcome = 'down'
                break

        if avg_approach > touch_pct:  # Came from ABOVE
            if outcome == 'up':
                from_above_bounce += 1
            elif outcome == 'down':
                from_above_break += 1
            else:
                from_above_neutral += 1

        elif avg_approach < -touch_pct:  # Came from BELOW
            if outcome == 'down':
                from_below_reject += 1
            elif outcome == 'up':
                from_below_break += 1
            else:
                from_below_neutral += 1

    return {
        'from_above': {
            'bounce': from_above_bounce,
            'break': from_above_break,
            'neutral': from_above_neutral,
            'total': from_above_bounce + from_above_break + from_above_neutral
        },
        'from_below': {
            'reject': from_below_reject,
            'break': from_below_break,
            'neutral': from_below_neutral,
            'total': from_below_reject + from_below_break + from_below_neutral
        }
    }

print("\n### When price comes FROM ABOVE and touches EMA:")
print("\n| EMA | Total | Bounce | Break | Neutral | Bounce% | Break% |")
print("|-----|-------|--------|-------|---------|---------|--------|")

for ema_period in EMA_PERIODS:
    rates = calculate_bounce_break_rate(ema_period)
    fa = rates['from_above']
    if fa['total'] > 100:
        bounce_pct = fa['bounce'] / fa['total'] * 100
        break_pct = fa['break'] / fa['total'] * 100
        print(f"| EMA{ema_period} | {fa['total']:,} | {fa['bounce']:,} | {fa['break']:,} | {fa['neutral']:,} | {bounce_pct:.1f}% | {break_pct:.1f}% |")

print("\n### When price comes FROM BELOW and touches EMA:")
print("\n| EMA | Total | Reject | Break | Neutral | Reject% | Break% |")
print("|-----|-------|--------|-------|---------|---------|--------|")

for ema_period in EMA_PERIODS:
    rates = calculate_bounce_break_rate(ema_period)
    fb = rates['from_below']
    if fb['total'] > 100:
        reject_pct = fb['reject'] / fb['total'] * 100
        break_pct = fb['break'] / fb['total'] * 100
        print(f"| EMA{ema_period} | {fb['total']:,} | {fb['reject']:,} | {fb['break']:,} | {fb['neutral']:,} | {reject_pct:.1f}% | {break_pct:.1f}% |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. Bounce/Break Rate:")
for ema_period in [50, 200]:
    rates = calculate_bounce_break_rate(ema_period)
    fa = rates['from_above']
    fb = rates['from_below']
    if fa['total'] > 100 and fb['total'] > 100:
        bounce_pct = fa['bounce'] / fa['total'] * 100
        reject_pct = fb['reject'] / fb['total'] * 100
        print(f"   EMA{ema_period}: Bounce {bounce_pct:.1f}%, Reject {reject_pct:.1f}%")

print("\n2. Confirmed Bounce Magnitude (EMA50, H=30):")
bounces = all_results[50]['bounces']
if len(bounces) > 100:
    bounce_vals = [b['bounce_H30'] for b in bounces if 'bounce_H30' in b]
    corr_vals = [b['correction_H30'] for b in bounces if 'correction_H30' in b]
    if bounce_vals:
        print(f"   Median bounce: {np.median(bounce_vals):.1f}bp")
        print(f"   Median correction: {np.median(corr_vals):.1f}bp")
        print(f"   Ratio: {np.median(bounce_vals)/np.median(corr_vals):.2f}:1")

print("\n3. Confirmed Rejection Magnitude (EMA50, H=30):")
rejections = all_results[50]['rejections']
if len(rejections) > 100:
    drop_vals = [r['drop_H30'] for r in rejections if 'drop_H30' in r]
    corr_vals = [r['correction_H30'] for r in rejections if 'correction_H30' in r]
    if drop_vals:
        print(f"   Median drop: {np.median(drop_vals):.1f}bp")
        print(f"   Median correction: {np.median(corr_vals):.1f}bp")
        print(f"   Ratio: {np.median(drop_vals)/np.median(corr_vals):.2f}:1")

print("\n4. Key finding:")
print("   - Bounce/Rejection RATE tells us: How often does EMA act as support/resistance?")
print("   - Bounce/Rejection MAGNITUDE tells us: When it works, how far does it go?")
print("   - These are CONFIRMED patterns (not just price near EMA)")
