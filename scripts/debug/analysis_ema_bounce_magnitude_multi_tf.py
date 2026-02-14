"""
RE-ANALYSIS-16: EMA Bounce Magnitude on Multiple Timeframes

Measures:
1. How often does EMA act as support/resistance? (Bounce rate)
2. When it works, how far does price bounce? (Magnitude)
3. How far does price go against before bouncing? (Correction)

Timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes
EMA calculated on EACH timeframe's candles (correct method)

Run: python scripts/debug/analysis_ema_bounce_magnitude_multi_tf.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES_MINUTES = [3, 5, 10, 15, 30, 60, 120, 240, 480]
EMA_PERIODS = [9, 20, 50, 100, 200]  # Test all EMAs to find best for each timeframe
TOUCH_THRESHOLD_BPS = 15  # How close to EMA to count as "touch"
APPROACH_BARS = 5  # How many bars to look back for approach direction
BOUNCE_CONFIRM_BPS = 10  # Min move away from EMA to confirm bounce/rejection
MEASURE_HORIZONS = [3, 5, 10, 20]  # Bars to measure bounce/correction

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RE-ANALYSIS-16: EMA BOUNCE MAGNITUDE ON MULTIPLE TIMEFRAMES")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

# Use ALL data for distribution analysis (not train/test - this is descriptive)
all_data = ohlcv_1m.copy()
print(f"Full data: {len(all_data):,} 1-minute candles ({all_data.index.min().date()} to {all_data.index.max().date()})")


# =============================================================================
# RESAMPLE
# =============================================================================
def resample_ohlcv(df, minutes):
    tf_str = f'{minutes}min'
    return df.resample(tf_str).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


print("\nResampling...")
timeframe_data = {}
for tf_min in TIMEFRAMES_MINUTES:
    timeframe_data[tf_min] = resample_ohlcv(all_data, tf_min)
    print(f"  {tf_min}-minute: {len(timeframe_data[tf_min]):,} candles")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_ema_bounce(df, ema_period, H_bars):
    """
    Analyze EMA bounce/rejection for given EMA and horizon.
    Returns bounce rate, bounce magnitude, correction magnitude.
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    ema = df['close'].ewm(span=ema_period, adjust=False).mean().values
    n = len(df)

    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    confirm_pct = BOUNCE_CONFIRM_BPS / 10000

    # Support test (price from above touches EMA)
    support_touches = 0
    support_bounces = 0
    support_breaks = 0
    bounce_magnitudes = []
    bounce_corrections = []

    # Resistance test (price from below touches EMA)
    resist_touches = 0
    resist_rejects = 0
    resist_breaks = 0
    reject_magnitudes = []
    reject_corrections = []

    valid_start = max(ema_period, APPROACH_BARS) + 10
    max_end = n - H_bars - 5

    for i in range(valid_start, max_end):
        current_close = close[i]
        current_ema = ema[i]

        # Check if touching EMA
        distance_pct = abs(current_close - current_ema) / current_ema
        if distance_pct > touch_pct:
            continue

        # Check approach direction
        above_count = sum(1 for k in range(1, APPROACH_BARS + 1) if i-k >= 0 and close[i-k] > ema[i-k])
        below_count = sum(1 for k in range(1, APPROACH_BARS + 1) if i-k >= 0 and close[i-k] < ema[i-k])

        # SUPPORT TEST: Price came from above
        if above_count >= APPROACH_BARS * 0.6:
            support_touches += 1

            # Measure bounce (up) and correction (down) over H bars
            max_up = 0
            max_down = 0
            for j in range(1, H_bars + 1):
                if i + j >= n:
                    break
                up_move = (high[i + j] - current_close) / current_close * 10000
                down_move = (current_close - low[i + j]) / current_close * 10000
                max_up = max(max_up, up_move)
                max_down = max(max_down, down_move)

            # Did it bounce (confirm move UP)?
            if max_up >= confirm_pct * 10000:
                support_bounces += 1
                bounce_magnitudes.append(max_up)
                bounce_corrections.append(max_down)
            # Did it break (move DOWN through EMA)?
            elif max_down >= confirm_pct * 10000:
                support_breaks += 1

        # RESISTANCE TEST: Price came from below
        elif below_count >= APPROACH_BARS * 0.6:
            resist_touches += 1

            # Measure rejection (down) and correction (up) over H bars
            max_down = 0
            max_up = 0
            for j in range(1, H_bars + 1):
                if i + j >= n:
                    break
                down_move = (current_close - low[i + j]) / current_close * 10000
                up_move = (high[i + j] - current_close) / current_close * 10000
                max_down = max(max_down, down_move)
                max_up = max(max_up, up_move)

            # Did it reject (confirm move DOWN)?
            if max_down >= confirm_pct * 10000:
                resist_rejects += 1
                reject_magnitudes.append(max_down)
                reject_corrections.append(max_up)
            # Did it break (move UP through EMA)?
            elif max_up >= confirm_pct * 10000:
                resist_breaks += 1

    return {
        'support_touches': support_touches,
        'support_bounces': support_bounces,
        'support_breaks': support_breaks,
        'support_bounce_rate': support_bounces / support_touches * 100 if support_touches > 0 else 0,
        'bounce_median': np.median(bounce_magnitudes) if bounce_magnitudes else 0,
        'bounce_75th': np.percentile(bounce_magnitudes, 75) if bounce_magnitudes else 0,
        'correction_median': np.median(bounce_corrections) if bounce_corrections else 0,
        'resist_touches': resist_touches,
        'resist_rejects': resist_rejects,
        'resist_breaks': resist_breaks,
        'resist_reject_rate': resist_rejects / resist_touches * 100 if resist_touches > 0 else 0,
        'reject_median': np.median(reject_magnitudes) if reject_magnitudes else 0,
        'reject_75th': np.percentile(reject_magnitudes, 75) if reject_magnitudes else 0,
        'reject_correction_median': np.median(reject_corrections) if reject_corrections else 0,
    }


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING EMA BOUNCE ANALYSIS ON ALL TIMEFRAMES...")
print("=" * 80)

all_results = []

for tf_min in TIMEFRAMES_MINUTES:
    print(f"\n{'='*60}")
    print(f"TIMEFRAME: {tf_min} MINUTES")
    print(f"{'='*60}")

    df = timeframe_data[tf_min]

    print(f"\n### SUPPORT TEST (Bounce UP from EMA)")
    print(f"| EMA | H(bars) | H(time) | Touches | Bounces | Bounce% | Bounce Med | Corr Med | R:R |")
    print(f"|-----|---------|---------|---------|---------|---------|------------|----------|-----|")

    for ema_period in EMA_PERIODS:
        for H_bars in [5, 10, 20]:
            r = analyze_ema_bounce(df, ema_period, H_bars)
            H_time = H_bars * tf_min

            if r['support_touches'] >= 100:
                rr = r['bounce_median'] / r['correction_median'] if r['correction_median'] > 0 else 0
                print(f"| {ema_period} | {H_bars}b | {H_time}min | {r['support_touches']:,} | "
                      f"{r['support_bounces']:,} | {r['support_bounce_rate']:.1f}% | "
                      f"{r['bounce_median']:.1f}bp | {r['correction_median']:.1f}bp | {rr:.2f} |")

                all_results.append({
                    'tf_min': tf_min,
                    'ema': ema_period,
                    'H_bars': H_bars,
                    'H_time': H_time,
                    'type': 'support',
                    'touches': r['support_touches'],
                    'success': r['support_bounces'],
                    'success_rate': r['support_bounce_rate'],
                    'magnitude': r['bounce_median'],
                    'correction': r['correction_median'],
                    'rr_ratio': rr
                })

    print(f"\n### RESISTANCE TEST (Reject DOWN from EMA)")
    print(f"| EMA | H(bars) | H(time) | Touches | Rejects | Reject% | Drop Med | Corr Med | R:R |")
    print(f"|-----|---------|---------|---------|---------|---------|----------|----------|-----|")

    for ema_period in EMA_PERIODS:
        for H_bars in [5, 10, 20]:
            r = analyze_ema_bounce(df, ema_period, H_bars)
            H_time = H_bars * tf_min

            if r['resist_touches'] >= 100:
                rr = r['reject_median'] / r['reject_correction_median'] if r['reject_correction_median'] > 0 else 0
                print(f"| {ema_period} | {H_bars}b | {H_time}min | {r['resist_touches']:,} | "
                      f"{r['resist_rejects']:,} | {r['resist_reject_rate']:.1f}% | "
                      f"{r['reject_median']:.1f}bp | {r['reject_correction_median']:.1f}bp | {rr:.2f} |")

                all_results.append({
                    'tf_min': tf_min,
                    'ema': ema_period,
                    'H_bars': H_bars,
                    'H_time': H_time,
                    'type': 'resistance',
                    'touches': r['resist_touches'],
                    'success': r['resist_rejects'],
                    'success_rate': r['resist_reject_rate'],
                    'magnitude': r['reject_median'],
                    'correction': r['reject_correction_median'],
                    'rr_ratio': rr
                })


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: BEST EMA BOUNCE/REJECTION BY TIMEFRAME")
print("=" * 80)

print("\n| Timeframe | Best Setup | Success Rate | Bounce/Drop | Correction | R:R |")
print("|-----------|------------|--------------|-------------|------------|-----|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_results = [r for r in all_results if r['tf_min'] == tf_min]
    if tf_results:
        # Sort by success rate
        tf_results.sort(key=lambda x: x['success_rate'], reverse=True)
        best = tf_results[0]
        setup = f"EMA{best['ema']} H={best['H_bars']}b ({best['type'][:3]})"
        print(f"| {tf_min}min | {setup} | {best['success_rate']:.1f}% | "
              f"{best['magnitude']:.1f}bp | {best['correction']:.1f}bp | {best['rr_ratio']:.2f} |")


# =============================================================================
# BEST EMA FOR EACH TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("BEST EMA FOR EACH TIMEFRAME (By Success Rate)")
print("=" * 80)

print("\n### SUPPORT (Bounce UP) - Best EMA per Timeframe")
print("| Timeframe | Best EMA | Success Rate | Bounce | Correction | R:R | Sample |")
print("|-----------|----------|--------------|--------|------------|-----|--------|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_support = [r for r in all_results if r['tf_min'] == tf_min and r['type'] == 'support']
    if tf_support:
        # Find best by success rate
        tf_support.sort(key=lambda x: x['success_rate'], reverse=True)
        best = tf_support[0]
        print(f"| {tf_min}min | EMA{best['ema']} | {best['success_rate']:.1f}% | "
              f"{best['magnitude']:.1f}bp | {best['correction']:.1f}bp | {best['rr_ratio']:.2f} | {best['touches']:,} |")

print("\n### RESISTANCE (Reject DOWN) - Best EMA per Timeframe")
print("| Timeframe | Best EMA | Success Rate | Drop | Correction | R:R | Sample |")
print("|-----------|----------|--------------|------|------------|-----|--------|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_resist = [r for r in all_results if r['tf_min'] == tf_min and r['type'] == 'resistance']
    if tf_resist:
        tf_resist.sort(key=lambda x: x['success_rate'], reverse=True)
        best = tf_resist[0]
        print(f"| {tf_min}min | EMA{best['ema']} | {best['success_rate']:.1f}% | "
              f"{best['magnitude']:.1f}bp | {best['correction']:.1f}bp | {best['rr_ratio']:.2f} | {best['touches']:,} |")


# =============================================================================
# BEST EMA BY R:R RATIO (Alternative ranking)
# =============================================================================
print("\n" + "=" * 80)
print("BEST EMA FOR EACH TIMEFRAME (By R:R Ratio)")
print("=" * 80)

print("\n### SUPPORT (Bounce UP) - Best R:R per Timeframe")
print("| Timeframe | Best EMA | R:R | Success Rate | Bounce | Correction | Sample |")
print("|-----------|----------|-----|--------------|--------|------------|--------|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_support = [r for r in all_results if r['tf_min'] == tf_min and r['type'] == 'support']
    if tf_support:
        # Find best by R:R ratio
        tf_support.sort(key=lambda x: x['rr_ratio'], reverse=True)
        best = tf_support[0]
        print(f"| {tf_min}min | EMA{best['ema']} | {best['rr_ratio']:.2f} | {best['success_rate']:.1f}% | "
              f"{best['magnitude']:.1f}bp | {best['correction']:.1f}bp | {best['touches']:,} |")

print("\n### RESISTANCE (Reject DOWN) - Best R:R per Timeframe")
print("| Timeframe | Best EMA | R:R | Success Rate | Drop | Correction | Sample |")
print("|-----------|----------|-----|--------------|------|------------|--------|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_resist = [r for r in all_results if r['tf_min'] == tf_min and r['type'] == 'resistance']
    if tf_resist:
        tf_resist.sort(key=lambda x: x['rr_ratio'], reverse=True)
        best = tf_resist[0]
        print(f"| {tf_min}min | EMA{best['ema']} | {best['rr_ratio']:.2f} | {best['success_rate']:.1f}% | "
              f"{best['magnitude']:.1f}bp | {best['correction']:.1f}bp | {best['touches']:,} |")


# =============================================================================
# KEY FINDINGS
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("""
ANALYSIS-12 CORRECTED: EMA BOUNCE MAGNITUDE ON ALL TIMEFRAMES

1. WHAT WAS MEASURED:
   - EMA calculated on EACH timeframe's candles (correct method)
   - Support test: Price from above touches EMA, check if bounces UP
   - Resistance test: Price from below touches EMA, check if rejects DOWN
   - Bounce/Drop magnitude: How far price moves in expected direction
   - Correction: How far price moves against expected direction

2. KEY METRICS:
   - Success Rate: % of touches that resulted in bounce/rejection
   - R:R Ratio: Bounce magnitude / Correction magnitude
   - R:R > 1 means bounce is larger than correction (favorable)

3. INTERPRETATION:
   - High success rate (>70%) = EMA acts as reliable support/resistance
   - R:R > 1.5 = Good risk:reward for trading bounces
   - Look for combinations of HIGH success rate AND HIGH R:R

4. EMA MEANING BY TIMEFRAME:
   - EMA20 on 3-min = 60 minutes of price history
   - EMA20 on 15-min = 300 minutes of price history
   - EMA20 on 60-min = 1200 minutes (20 hours) of price history
""")
