"""
ANALYSIS-11 RE-RUN: Edge Filter Backtest on Multiple Timeframes

Tests momentum-based edge filter on timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes

Run: python scripts/debug/analysis11_multi_timeframe.py
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
HOLD_BARS = [5, 10, 20]  # Bars to hold on each timeframe
TARGETS_BPS = [15, 25, 50]
SAMPLE_SIZE = 30000
MAE_CUT_THRESHOLD = 50  # Exit if MAE > 50bp
FEE_BPS = 8
MOMENTUM_LOOKBACK = 20  # 20 bars lookback for momentum
MOMENTUM_THRESHOLD = 5  # Need >5bp momentum for signal

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-11 RE-RUN: EDGE FILTER ON MULTIPLE TIMEFRAMES")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

train_1m = ohlcv_1m[ohlcv_1m.index <= "2023-12-31"].copy()
print(f"Train data: {len(train_1m):,} 1-minute candles")


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
    timeframe_data[tf_min] = resample_ohlcv(train_1m, tf_min)
    print(f"  {tf_min}-minute: {len(timeframe_data[tf_min]):,} candles")


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================
def backtest_edge_filter(df, H_bars, target_bps):
    """
    Backtest with momentum edge filter on given timeframe.
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(df)

    target_pct = target_bps / 10000
    mae_cut_pct = MAE_CUT_THRESHOLD / 10000

    trades = 0
    wins = 0
    cuts = 0
    timeouts = 0
    total_pnl_bps = 0

    valid_start = MOMENTUM_LOOKBACK + 10
    max_end = n - H_bars - 5

    # Sample indices
    np.random.seed(42)
    available = range(valid_start, max_end)
    sample_size = min(SAMPLE_SIZE, len(available))
    sample_idx = np.random.choice(available, size=sample_size, replace=False)

    for i in sample_idx:
        # Calculate momentum signal on THIS timeframe
        momentum = (close[i] - close[i - MOMENTUM_LOOKBACK]) / close[i - MOMENTUM_LOOKBACK] * 10000

        # Filter: only trade if strong momentum
        if abs(momentum) < MOMENTUM_THRESHOLD:
            continue

        signal = "LONG" if momentum > MOMENTUM_THRESHOLD else "SHORT"
        trades += 1
        entry = close[i]

        if signal == "LONG":
            target_price = entry * (1 + target_pct)
            mae_cut_price = entry * (1 - mae_cut_pct)

            hit_target = False
            hit_cut = False

            for j in range(1, H_bars + 1):
                if i + j >= n:
                    break
                if low[i + j] <= mae_cut_price:
                    hit_cut = True
                    break
                if high[i + j] >= target_price:
                    hit_target = True
                    break

            if hit_target:
                wins += 1
                total_pnl_bps += (target_bps - FEE_BPS)
            elif hit_cut:
                cuts += 1
                total_pnl_bps -= (MAE_CUT_THRESHOLD + FEE_BPS)
            else:
                timeouts += 1
                exit_pnl = (close[min(i + H_bars, n-1)] - entry) / entry * 10000 - FEE_BPS
                total_pnl_bps += exit_pnl

        else:  # SHORT
            target_price = entry * (1 - target_pct)
            mae_cut_price = entry * (1 + mae_cut_pct)

            hit_target = False
            hit_cut = False

            for j in range(1, H_bars + 1):
                if i + j >= n:
                    break
                if high[i + j] >= mae_cut_price:
                    hit_cut = True
                    break
                if low[i + j] <= target_price:
                    hit_target = True
                    break

            if hit_target:
                wins += 1
                total_pnl_bps += (target_bps - FEE_BPS)
            elif hit_cut:
                cuts += 1
                total_pnl_bps -= (MAE_CUT_THRESHOLD + FEE_BPS)
            else:
                timeouts += 1
                exit_pnl = (entry - close[min(i + H_bars, n-1)]) / entry * 10000 - FEE_BPS
                total_pnl_bps += exit_pnl

    return {
        'trades': trades,
        'wins': wins,
        'cuts': cuts,
        'timeouts': timeouts,
        'total_pnl_bps': total_pnl_bps
    }


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING EDGE FILTER BACKTEST ON ALL TIMEFRAMES...")
print("=" * 80)

all_results = []

for tf_min in TIMEFRAMES_MINUTES:
    print(f"\nProcessing {tf_min}-minute timeframe...")
    df = timeframe_data[tf_min]

    for H_bars in HOLD_BARS:
        H_time = H_bars * tf_min  # Real time in minutes

        for target_bps in TARGETS_BPS:
            result = backtest_edge_filter(df, H_bars, target_bps)

            if result['trades'] > 0:
                win_pct = result['wins'] / result['trades'] * 100
                wc_ratio = result['wins'] / result['cuts'] if result['cuts'] > 0 else float('inf')
                avg_pnl = result['total_pnl_bps'] / result['trades']

                # Required W/C ratio
                net_win = target_bps - FEE_BPS
                net_loss = MAE_CUT_THRESHOLD + FEE_BPS
                required_wc = net_loss / net_win if net_win > 0 else float('inf')

                status = "GOOD" if wc_ratio >= required_wc else "BAD"

                all_results.append({
                    'tf_min': tf_min,
                    'H_bars': H_bars,
                    'H_time': H_time,
                    'target': target_bps,
                    'trades': result['trades'],
                    'win_pct': win_pct,
                    'wc_ratio': wc_ratio,
                    'required_wc': required_wc,
                    'avg_pnl': avg_pnl,
                    'total_pnl': result['total_pnl_bps'],
                    'status': status
                })


# =============================================================================
# RESULTS BY TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS BY TIMEFRAME")
print("=" * 80)

for tf_min in TIMEFRAMES_MINUTES:
    tf_results = [r for r in all_results if r['tf_min'] == tf_min]
    if not tf_results:
        continue

    print(f"\n### {tf_min}-MINUTE TIMEFRAME")
    print(f"Momentum lookback: {MOMENTUM_LOOKBACK} bars = {MOMENTUM_LOOKBACK * tf_min} minutes")

    print("\n| Target | H(bars) | H(time) | Trades | Win% | W/C Ratio | Req W/C | Avg PnL | Status |")
    print("|--------|---------|---------|--------|------|-----------|---------|---------|--------|")

    for r in tf_results:
        wc_str = f"{r['wc_ratio']:.1f}" if r['wc_ratio'] != float('inf') else "N/A"
        sign = "+" if r['avg_pnl'] > 0 else ""
        print(f"| {r['target']}bp | {r['H_bars']}b | {r['H_time']}min | {r['trades']:,} | "
              f"{r['win_pct']:.1f}% | {wc_str} | {r['required_wc']:.1f} | "
              f"{sign}{r['avg_pnl']:.2f}bp | {r['status']} |")


# =============================================================================
# SUMMARY: BEST BY TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: BEST RESULT PER TIMEFRAME")
print("=" * 80)

print("\n| Timeframe | Momentum LB | Best Target | Best PnL | Win% | Status | Profitable? |")
print("|-----------|-------------|-------------|----------|------|--------|-------------|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_results = [r for r in all_results if r['tf_min'] == tf_min]
    if tf_results:
        # Sort by avg_pnl (best first)
        tf_results.sort(key=lambda x: x['avg_pnl'], reverse=True)
        best = tf_results[0]

        momentum_lb_min = MOMENTUM_LOOKBACK * tf_min
        profitable = "YES" if best['avg_pnl'] > 0 else "NO"
        sign = "+" if best['avg_pnl'] > 0 else ""

        print(f"| {tf_min}min | {momentum_lb_min}min | T{best['target']}/H{best['H_bars']} | "
              f"{sign}{best['avg_pnl']:.2f}bp | {best['win_pct']:.1f}% | {best['status']} | {profitable} |")


# =============================================================================
# COMPARISON: MOMENTUM LOOKBACK MEANING BY TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("MOMENTUM LOOKBACK MEANING BY TIMEFRAME")
print("=" * 80)

print("\n| Timeframe | 20-bar Lookback Equals |")
print("|-----------|------------------------|")
for tf_min in TIMEFRAMES_MINUTES:
    lookback_min = MOMENTUM_LOOKBACK * tf_min
    if lookback_min >= 60:
        hours = lookback_min / 60
        print(f"| {tf_min}min | {lookback_min} min ({hours:.1f} hours) |")
    else:
        print(f"| {tf_min}min | {lookback_min} min |")


# =============================================================================
# KEY FINDINGS
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

profitable_count = sum(1 for r in all_results if r['avg_pnl'] > 0)
total_count = len(all_results)

print(f"""
1. TIMEFRAMES TESTED: {TIMEFRAMES_MINUTES}
   - Momentum calculated on EACH timeframe's candles
   - 20-bar lookback means different time on each TF

2. MOMENTUM LOOKBACK BY TIMEFRAME:
   - 3-min TF: 20 bars = 60 min lookback
   - 15-min TF: 20 bars = 300 min (5 hour) lookback
   - 60-min TF: 20 bars = 1200 min (20 hour) lookback

3. PROFITABILITY ACROSS ALL TIMEFRAMES:
   - Profitable combinations: {profitable_count} / {total_count}
   - Percentage: {profitable_count/total_count*100:.1f}%
""")

if profitable_count > 0:
    print("4. PROFITABLE SETUPS FOUND:")
    profitable_results = [r for r in all_results if r['avg_pnl'] > 0]
    profitable_results.sort(key=lambda x: x['avg_pnl'], reverse=True)
    for r in profitable_results[:5]:
        print(f"   - {r['tf_min']}min TF, T{r['target']}/H{r['H_bars']}: +{r['avg_pnl']:.2f}bp")
else:
    print("4. NO PROFITABLE SETUPS FOUND")
    print("   - Edge filter doesn't work on ANY timeframe")
    print("   - Confirms original ANALYSIS-11 conclusion")
