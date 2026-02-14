"""
ANALYSIS-14 RE-RUN: Range Position on Multiple Timeframes
==========================================================
Range Position calculated on EACH timeframe's candles (correct method).

Question: When price is near support/resistance, does it reverse?
- Near Low (< 20%) = Support zone -> expect UP
- Near High (> 80%) = Resistance zone -> expect DOWN

Data: Full dataset (2019-2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES_MINUTES = [3, 5, 10, 15, 30, 60, 120, 240, 480]
LOOKBACK_PERIODS = [10, 20, 50]  # Bars to calculate range
SUPPORT_LEVELS = [10, 20, 30]  # % from low
RESISTANCE_LEVELS = [70, 80, 90]  # % from low (high = 100%)
H_BARS = [5, 10, 20]  # Horizon in bars of that timeframe

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-14 RE-RUN: RANGE POSITION (MULTI-TIMEFRAME)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")
print(f"Date range: {ohlcv_1m.index.min().date()} to {ohlcv_1m.index.max().date()}")

# =============================================================================
# RESAMPLE FUNCTION
# =============================================================================
def resample_ohlcv(df, minutes):
    """Resample 1-min data to larger timeframe."""
    if minutes == 1:
        return df.copy()
    return df.resample(f'{minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

def calculate_range_position(df, lookback):
    """Calculate range position (0-100%) within recent high/low range."""
    rolling_high = df['high'].rolling(window=lookback, min_periods=lookback).max()
    rolling_low = df['low'].rolling(window=lookback, min_periods=lookback).min()

    range_size = rolling_high - rolling_low
    position = (df['close'] - rolling_low) / range_size * 100

    # Handle zero range (flat market)
    position = position.replace([np.inf, -np.inf], np.nan)

    return position

# =============================================================================
# RESAMPLE ALL TIMEFRAMES
# =============================================================================
print("\nResampling...")
timeframe_data = {}
for tf_min in TIMEFRAMES_MINUTES:
    timeframe_data[tf_min] = resample_ohlcv(ohlcv_1m, tf_min)
    print(f"  {tf_min}-minute: {len(timeframe_data[tf_min]):,} candles")

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_range_reversal(df, lookback, support_level, resistance_level, H_bars):
    """
    Analyze range position mean reversion.
    Returns stats for support (expect UP) and resistance (expect DOWN).
    """
    df = df.copy()
    df['range_pos'] = calculate_range_position(df, lookback)
    df = df.dropna()

    if len(df) < H_bars + 100:
        return None

    close = df['close'].values
    range_pos = df['range_pos'].values

    # Support analysis (range_pos < support_level -> expect UP)
    support_up = 0
    support_down = 0

    # Resistance analysis (range_pos > resistance_level -> expect DOWN)
    resistance_up = 0
    resistance_down = 0

    for i in range(len(df) - H_bars - 1):
        entry_price = close[i]
        future_close = close[i + H_bars]

        # Support zone
        if range_pos[i] < support_level:
            if future_close > entry_price:
                support_up += 1
            else:
                support_down += 1

        # Resistance zone
        if range_pos[i] > resistance_level:
            if future_close < entry_price:
                resistance_down += 1
            else:
                resistance_up += 1

    results = {}

    # Support results
    if support_up + support_down >= 50:
        total = support_up + support_down
        up_rate = support_up / total * 100
        results['support'] = {
            'count': total,
            'up_rate': up_rate,
            'edge': up_rate - 50
        }

    # Resistance results
    if resistance_up + resistance_down >= 50:
        total = resistance_up + resistance_down
        down_rate = resistance_down / total * 100
        results['resistance'] = {
            'count': total,
            'down_rate': down_rate,
            'edge': down_rate - 50
        }

    return results if results else None

# =============================================================================
# RUN ANALYSIS ON ALL TIMEFRAMES
# =============================================================================
print("\n" + "=" * 80)
print("RANGE POSITION ANALYSIS BY TIMEFRAME")
print("=" * 80)

all_results = []

for tf_min in TIMEFRAMES_MINUTES:
    df = timeframe_data[tf_min]

    print(f"\n{'='*60}")
    print(f"TIMEFRAME: {tf_min} MINUTES")
    print(f"{'='*60}")

    print(f"\n### SUPPORT ZONE (Range < threshold -> expect UP)")
    print(f"| Lookback | Level | H(bars) | H(time) | Count | UP% | Edge |")
    print(f"|----------|-------|---------|---------|-------|-----|------|")

    for lookback in LOOKBACK_PERIODS:
        for support in SUPPORT_LEVELS:
            for H in H_BARS:
                result = analyze_range_reversal(df, lookback, support, 100, H)
                if result and 'support' in result:
                    r = result['support']
                    h_time = H * tf_min
                    time_str = f"{h_time}min" if h_time < 60 else f"{h_time//60}h"
                    print(f"| {lookback}b | <{support}% | {H}b | {time_str} | {r['count']:,} | {r['up_rate']:.1f}% | {r['edge']:+.1f}% |")

                    all_results.append({
                        'tf': tf_min,
                        'lookback': lookback,
                        'type': 'support',
                        'level': support,
                        'H': H,
                        'count': r['count'],
                        'accuracy': r['up_rate'],
                        'edge': r['edge']
                    })

    print(f"\n### RESISTANCE ZONE (Range > threshold -> expect DOWN)")
    print(f"| Lookback | Level | H(bars) | H(time) | Count | DOWN% | Edge |")
    print(f"|----------|-------|---------|---------|-------|-------|------|")

    for lookback in LOOKBACK_PERIODS:
        for resistance in RESISTANCE_LEVELS:
            for H in H_BARS:
                result = analyze_range_reversal(df, lookback, 0, resistance, H)
                if result and 'resistance' in result:
                    r = result['resistance']
                    h_time = H * tf_min
                    time_str = f"{h_time}min" if h_time < 60 else f"{h_time//60}h"
                    print(f"| {lookback}b | >{resistance}% | {H}b | {time_str} | {r['count']:,} | {r['down_rate']:.1f}% | {r['edge']:+.1f}% |")

                    all_results.append({
                        'tf': tf_min,
                        'lookback': lookback,
                        'type': 'resistance',
                        'level': resistance,
                        'H': H,
                        'count': r['count'],
                        'accuracy': r['down_rate'],
                        'edge': r['edge']
                    })

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: BEST RANGE POSITION SETUP PER TIMEFRAME")
print("=" * 80)

results_df = pd.DataFrame(all_results)

print("\n### BEST SUPPORT (expect UP) by Timeframe")
print("| Timeframe | Lookback | Level | H | Accuracy | Edge |")
print("|-----------|----------|-------|---|----------|------|")

for tf in TIMEFRAMES_MINUTES:
    tf_data = results_df[(results_df['tf'] == tf) & (results_df['type'] == 'support')]
    if len(tf_data) > 0:
        best = tf_data.loc[tf_data['edge'].idxmax()]
        print(f"| {tf}min | {int(best['lookback'])}b | <{int(best['level'])}% | {int(best['H'])}b | {best['accuracy']:.1f}% | {best['edge']:+.1f}% |")

print("\n### BEST RESISTANCE (expect DOWN) by Timeframe")
print("| Timeframe | Lookback | Level | H | Accuracy | Edge |")
print("|-----------|----------|-------|---|----------|------|")

for tf in TIMEFRAMES_MINUTES:
    tf_data = results_df[(results_df['tf'] == tf) & (results_df['type'] == 'resistance')]
    if len(tf_data) > 0:
        best = tf_data.loc[tf_data['edge'].idxmax()]
        print(f"| {tf}min | {int(best['lookback'])}b | >{int(best['level'])}% | {int(best['H'])}b | {best['accuracy']:.1f}% | {best['edge']:+.1f}% |")

# Overall best
print("\n### OVERALL BEST SETUPS")
if len(results_df) > 0:
    print("\nTop 5 Support:")
    support_df = results_df[results_df['type'] == 'support'].nlargest(5, 'edge')
    for _, row in support_df.iterrows():
        print(f"  {row['tf']}min LB={int(row['lookback'])} <{int(row['level'])}% H={int(row['H'])}b: {row['accuracy']:.1f}% ({row['edge']:+.1f}% edge)")

    print("\nTop 5 Resistance:")
    resistance_df = results_df[results_df['type'] == 'resistance'].nlargest(5, 'edge')
    for _, row in resistance_df.iterrows():
        print(f"  {row['tf']}min LB={int(row['lookback'])} >{int(row['level'])}% H={int(row['H'])}b: {row['accuracy']:.1f}% ({row['edge']:+.1f}% edge)")

# =============================================================================
# KEY FINDINGS
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

avg_support_edge = results_df[results_df['type'] == 'support']['edge'].mean()
avg_resistance_edge = results_df[results_df['type'] == 'resistance']['edge'].mean()
max_edge = results_df['edge'].max()

print(f"""
1. Range Position calculated on EACH timeframe's candles (correct method)
2. Data: Full dataset 2019-2025

3. Average support edge: {avg_support_edge:+.1f}%
4. Average resistance edge: {avg_resistance_edge:+.1f}%
5. Best edge found: {max_edge:+.1f}%

6. TRADEABLE?
   - Need 58%+ accuracy for 50bp target after 8bp fees
   - Best Range Position accuracy: {results_df['accuracy'].max():.1f}%
   - Gap: {results_df['accuracy'].max() - 58:.1f}%
""")
