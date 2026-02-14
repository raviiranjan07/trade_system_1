"""
ANALYSIS-13 RE-RUN: RSI Mean Reversion on Multiple Timeframes
=============================================================
RSI calculated on EACH timeframe's candles (correct method).

Question: When RSI is extreme, does price reverse?
- RSI < 30 (oversold) → expect UP
- RSI > 70 (overbought) → expect DOWN

Data: Full dataset (2019-2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES_MINUTES = [3, 5, 10, 15, 30, 60, 120, 240, 480]
RSI_PERIODS = [7, 14, 21]  # Common RSI periods
OVERSOLD_LEVELS = [20, 30]
OVERBOUGHT_LEVELS = [70, 80]
H_BARS = [5, 10, 20]  # Horizon in bars of that timeframe

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-13 RE-RUN: RSI MEAN REVERSION (MULTI-TIMEFRAME)")
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

def calculate_rsi(df, period):
    """Calculate RSI on close prices."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
def analyze_rsi_reversal(df, rsi_period, oversold, overbought, H_bars):
    """
    Analyze RSI mean reversion.
    Returns stats for oversold (expect UP) and overbought (expect DOWN).
    """
    # Calculate RSI
    df = df.copy()
    df['rsi'] = calculate_rsi(df, rsi_period)
    df = df.dropna()

    if len(df) < H_bars + 100:
        return None

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values

    # Oversold analysis (RSI < oversold → expect UP)
    oversold_up = 0
    oversold_down = 0
    oversold_returns = []

    # Overbought analysis (RSI > overbought → expect DOWN)
    overbought_up = 0
    overbought_down = 0
    overbought_returns = []

    for i in range(len(df) - H_bars - 1):
        entry_price = close[i]
        future_close = close[i + H_bars]
        ret_bp = (future_close / entry_price - 1) * 10000

        # Oversold
        if rsi[i] < oversold:
            oversold_returns.append(ret_bp)
            if future_close > entry_price:
                oversold_up += 1
            else:
                oversold_down += 1

        # Overbought
        if rsi[i] > overbought:
            overbought_returns.append(ret_bp)
            if future_close < entry_price:
                overbought_down += 1
            else:
                overbought_up += 1

    results = {}

    # Oversold results
    if oversold_up + oversold_down >= 50:
        total = oversold_up + oversold_down
        up_rate = oversold_up / total * 100
        results['oversold'] = {
            'count': total,
            'up_rate': up_rate,
            'edge': up_rate - 50,
            'avg_return': np.mean(oversold_returns) if oversold_returns else 0
        }

    # Overbought results
    if overbought_up + overbought_down >= 50:
        total = overbought_up + overbought_down
        down_rate = overbought_down / total * 100
        results['overbought'] = {
            'count': total,
            'down_rate': down_rate,
            'edge': down_rate - 50,
            'avg_return': -np.mean(overbought_returns) if overbought_returns else 0
        }

    return results if results else None

# =============================================================================
# RUN ANALYSIS ON ALL TIMEFRAMES
# =============================================================================
print("\n" + "=" * 80)
print("RSI MEAN REVERSION ANALYSIS BY TIMEFRAME")
print("=" * 80)

all_results = []

for tf_min in TIMEFRAMES_MINUTES:
    df = timeframe_data[tf_min]

    print(f"\n{'='*60}")
    print(f"TIMEFRAME: {tf_min} MINUTES")
    print(f"{'='*60}")

    print(f"\n### OVERSOLD (RSI < threshold -> expect UP)")
    print(f"| RSI | Level | H(bars) | H(time) | Count | UP% | Edge |")
    print(f"|-----|-------|---------|---------|-------|-----|------|")

    tf_best_oversold = {'edge': 0}
    tf_best_overbought = {'edge': 0}

    for rsi_period in RSI_PERIODS:
        for oversold in OVERSOLD_LEVELS:
            for H in H_BARS:
                result = analyze_rsi_reversal(df, rsi_period, oversold, 100, H)
                if result and 'oversold' in result:
                    r = result['oversold']
                    h_time = H * tf_min
                    time_str = f"{h_time}min" if h_time < 60 else f"{h_time//60}h"
                    print(f"| RSI{rsi_period} | <{oversold} | {H}b | {time_str} | {r['count']:,} | {r['up_rate']:.1f}% | {r['edge']:+.1f}% |")

                    all_results.append({
                        'tf': tf_min,
                        'rsi_period': rsi_period,
                        'type': 'oversold',
                        'level': oversold,
                        'H': H,
                        'count': r['count'],
                        'accuracy': r['up_rate'],
                        'edge': r['edge']
                    })

                    if r['edge'] > tf_best_oversold['edge']:
                        tf_best_oversold = {'edge': r['edge'], 'rsi': rsi_period, 'level': oversold, 'H': H, 'rate': r['up_rate']}

    print(f"\n### OVERBOUGHT (RSI > threshold -> expect DOWN)")
    print(f"| RSI | Level | H(bars) | H(time) | Count | DOWN% | Edge |")
    print(f"|-----|-------|---------|---------|-------|-------|------|")

    for rsi_period in RSI_PERIODS:
        for overbought in OVERBOUGHT_LEVELS:
            for H in H_BARS:
                result = analyze_rsi_reversal(df, rsi_period, 0, overbought, H)
                if result and 'overbought' in result:
                    r = result['overbought']
                    h_time = H * tf_min
                    time_str = f"{h_time}min" if h_time < 60 else f"{h_time//60}h"
                    print(f"| RSI{rsi_period} | >{overbought} | {H}b | {time_str} | {r['count']:,} | {r['down_rate']:.1f}% | {r['edge']:+.1f}% |")

                    all_results.append({
                        'tf': tf_min,
                        'rsi_period': rsi_period,
                        'type': 'overbought',
                        'level': overbought,
                        'H': H,
                        'count': r['count'],
                        'accuracy': r['down_rate'],
                        'edge': r['edge']
                    })

                    if r['edge'] > tf_best_overbought['edge']:
                        tf_best_overbought = {'edge': r['edge'], 'rsi': rsi_period, 'level': overbought, 'H': H, 'rate': r['down_rate']}

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: BEST RSI SETUP PER TIMEFRAME")
print("=" * 80)

# Convert to DataFrame for analysis
results_df = pd.DataFrame(all_results)

print("\n### BEST OVERSOLD (expect UP) by Timeframe")
print("| Timeframe | Best RSI | Level | H | Accuracy | Edge |")
print("|-----------|----------|-------|---|----------|------|")

for tf in TIMEFRAMES_MINUTES:
    tf_data = results_df[(results_df['tf'] == tf) & (results_df['type'] == 'oversold')]
    if len(tf_data) > 0:
        best = tf_data.loc[tf_data['edge'].idxmax()]
        print(f"| {tf}min | RSI{int(best['rsi_period'])} | <{int(best['level'])} | {int(best['H'])}b | {best['accuracy']:.1f}% | {best['edge']:+.1f}% |")

print("\n### BEST OVERBOUGHT (expect DOWN) by Timeframe")
print("| Timeframe | Best RSI | Level | H | Accuracy | Edge |")
print("|-----------|----------|-------|---|----------|------|")

for tf in TIMEFRAMES_MINUTES:
    tf_data = results_df[(results_df['tf'] == tf) & (results_df['type'] == 'overbought')]
    if len(tf_data) > 0:
        best = tf_data.loc[tf_data['edge'].idxmax()]
        print(f"| {tf}min | RSI{int(best['rsi_period'])} | >{int(best['level'])} | {int(best['H'])}b | {best['accuracy']:.1f}% | {best['edge']:+.1f}% |")

# Overall best
print("\n### OVERALL BEST RSI SETUPS")
if len(results_df) > 0:
    print("\nTop 5 Oversold:")
    oversold_df = results_df[results_df['type'] == 'oversold'].nlargest(5, 'edge')
    for _, row in oversold_df.iterrows():
        print(f"  {row['tf']}min RSI{int(row['rsi_period'])} <{int(row['level'])} H={int(row['H'])}b: {row['accuracy']:.1f}% ({row['edge']:+.1f}% edge)")

    print("\nTop 5 Overbought:")
    overbought_df = results_df[results_df['type'] == 'overbought'].nlargest(5, 'edge')
    for _, row in overbought_df.iterrows():
        print(f"  {row['tf']}min RSI{int(row['rsi_period'])} >{int(row['level'])} H={int(row['H'])}b: {row['accuracy']:.1f}% ({row['edge']:+.1f}% edge)")

# =============================================================================
# KEY FINDINGS
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

avg_oversold_edge = results_df[results_df['type'] == 'oversold']['edge'].mean()
avg_overbought_edge = results_df[results_df['type'] == 'overbought']['edge'].mean()
max_edge = results_df['edge'].max()

print(f"""
1. RSI calculated on EACH timeframe's candles (correct method)
2. Data: Full dataset 2019-2025

3. Average oversold edge: {avg_oversold_edge:+.1f}%
4. Average overbought edge: {avg_overbought_edge:+.1f}%
5. Best edge found: {max_edge:+.1f}%

6. TRADEABLE?
   - Need 58%+ accuracy for 50bp target after 8bp fees
   - Best RSI accuracy: {results_df['accuracy'].max():.1f}%
   - Gap: {results_df['accuracy'].max() - 58:.1f}%
""")
