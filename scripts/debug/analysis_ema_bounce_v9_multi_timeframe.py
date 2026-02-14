"""
ANALYSIS: EMA Bounce - MULTI-TIMEFRAME (V9)

Tests EMA bounce on different chart timeframes:
- 1-minute
- 5-minute
- 15-minute
- 1-hour

Each timeframe has its own EMA values and bounce characteristics.

Run: python scripts/debug/analysis_ema_bounce_v9_multi_timeframe.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES = ['1min', '5min', '15min', '1H']
EMA_PERIODS = [9, 20, 50, 100, 200]
EMA_SLOPE_BARS = 5  # 5 bars on each timeframe
TOUCH_THRESHOLD_BPS = 10  # Within 10bp of EMA
HORIZONS_BARS = [3, 5, 10, 20, 30]  # Bars (not minutes) - relative to each TF
FEE_BPS = 8

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - MULTI-TIMEFRAME (V9)")
print("Testing EMA bounce on 1m, 5m, 15m, 1H charts")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

# Filter to train data
train_1m = ohlcv_1m[ohlcv_1m.index <= "2023-12-31"].copy()
print(f"Train data: {len(train_1m):,} 1-minute candles")


# =============================================================================
# RESAMPLE TO DIFFERENT TIMEFRAMES
# =============================================================================
def resample_ohlcv(df, timeframe):
    """Resample 1-minute OHLCV to higher timeframe."""
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled


print("\nResampling to different timeframes...")
timeframe_data = {}

for tf in TIMEFRAMES:
    if tf == '1min':
        timeframe_data[tf] = train_1m.copy()
    else:
        timeframe_data[tf] = resample_ohlcv(train_1m, tf)
    print(f"  {tf}: {len(timeframe_data[tf]):,} candles")


# =============================================================================
# NUMBA FUNCTIONS
# =============================================================================
@njit
def is_ema_sloping_up(ema_arr, idx, slope_bars):
    if idx < slope_bars:
        return False
    return ema_arr[idx] > ema_arr[idx - slope_bars]


@njit
def is_ema_sloping_down(ema_arr, idx, slope_bars):
    if idx < slope_bars:
        return False
    return ema_arr[idx] < ema_arr[idx - slope_bars]


@njit
def low_touches_ema(low_val, ema_val, threshold_pct):
    distance = (low_val - ema_val) / ema_val
    return distance <= threshold_pct and distance >= -threshold_pct * 2


@njit
def high_touches_ema(high_val, ema_val, threshold_pct):
    distance = (high_val - ema_val) / ema_val
    return distance >= -threshold_pct and distance <= threshold_pct * 2


@njit
def backtest_ema_bounce(open_arr, high_arr, low_arr, close_arr, ema_arr,
                        n, H, target_bps, stop_bps, touch_pct, slope_bars, fee_bps):
    """
    Backtest EMA bounce strategy.
    LONG: EMA sloping up, low touches EMA, close above EMA
    SHORT: EMA sloping down, high touches EMA, close below EMA
    """
    target_pct = target_bps / 10000.0
    stop_pct = stop_bps / 10000.0

    long_trades = 0
    long_wins = 0
    long_pnl = 0.0

    short_trades = 0
    short_wins = 0
    short_pnl = 0.0

    valid_start = max(250, slope_bars + 10)
    max_end = n - H - 5

    i = valid_start
    while i < max_end:
        ema_val = ema_arr[i]

        # LONG: Uptrend bounce
        if is_ema_sloping_up(ema_arr, i, slope_bars):
            if low_touches_ema(low_arr[i], ema_val, touch_pct):
                if close_arr[i] > ema_val:
                    # ENTER LONG
                    long_trades += 1
                    entry = close_arr[i]
                    target_price = entry * (1.0 + target_pct)
                    stop_price = entry * (1.0 - stop_pct)

                    trade_pnl = 0.0
                    won = False

                    for j in range(1, H + 1):
                        if i + j >= n:
                            break
                        if low_arr[i + j] <= stop_price:
                            trade_pnl = -stop_bps - fee_bps
                            break
                        if high_arr[i + j] >= target_price:
                            trade_pnl = target_bps - fee_bps
                            won = True
                            break
                    else:
                        exit_price = close_arr[min(i + H, n - 1)]
                        trade_pnl = (exit_price - entry) / entry * 10000.0 - fee_bps

                    long_pnl += trade_pnl
                    if won:
                        long_wins += 1

                    i += H
                    continue

        # SHORT: Downtrend rejection
        if is_ema_sloping_down(ema_arr, i, slope_bars):
            if high_touches_ema(high_arr[i], ema_val, touch_pct):
                if close_arr[i] < ema_val:
                    # ENTER SHORT
                    short_trades += 1
                    entry = close_arr[i]
                    target_price = entry * (1.0 - target_pct)
                    stop_price = entry * (1.0 + stop_pct)

                    trade_pnl = 0.0
                    won = False

                    for j in range(1, H + 1):
                        if i + j >= n:
                            break
                        if high_arr[i + j] >= stop_price:
                            trade_pnl = -stop_bps - fee_bps
                            break
                        if low_arr[i + j] <= target_price:
                            trade_pnl = target_bps - fee_bps
                            won = True
                            break
                    else:
                        exit_price = close_arr[min(i + H, n - 1)]
                        trade_pnl = (entry - exit_price) / entry * 10000.0 - fee_bps

                    short_pnl += trade_pnl
                    if won:
                        short_wins += 1

                    i += H
                    continue

        i += 1

    return (long_trades, long_wins, long_pnl,
            short_trades, short_wins, short_pnl)


# =============================================================================
# RUN BACKTEST ON EACH TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("BACKTEST RESULTS BY TIMEFRAME")
print("=" * 80)

touch_pct = TOUCH_THRESHOLD_BPS / 10000.0
all_results = []

for tf in TIMEFRAMES:
    print(f"\n### Timeframe: {tf}")
    print("-" * 60)

    df = timeframe_data[tf]
    open_arr = df['open'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    close_arr = df['close'].values.astype(np.float64)
    n = len(df)

    # Calculate real time per bar
    if tf == '1min':
        minutes_per_bar = 1
    elif tf == '5min':
        minutes_per_bar = 5
    elif tf == '15min':
        minutes_per_bar = 15
    elif tf == '1H':
        minutes_per_bar = 60
    else:
        minutes_per_bar = 1

    print(f"\n| EMA | H(bars) | H(time) | Target | Stop | L_Trades | L_Win% | L_Avg | S_Trades | S_Win% | S_Avg | Combined |")
    print("|-----|---------|---------|--------|------|----------|--------|-------|----------|--------|-------|----------|")

    for ema_period in [20, 50, 100]:
        # Calculate EMA on this timeframe
        ema = df['close'].ewm(span=ema_period, adjust=False).mean().values.astype(np.float64)

        for H_bars in [5, 10, 20]:
            H_time = H_bars * minutes_per_bar  # Real time in minutes

            for target in [20, 30]:
                stop = int(target * 0.5)  # 50% of target as stop

                (l_trades, l_wins, l_pnl,
                 s_trades, s_wins, s_pnl) = backtest_ema_bounce(
                    open_arr, high_arr, low_arr, close_arr, ema,
                    n, H_bars, target, stop, touch_pct, EMA_SLOPE_BARS, FEE_BPS)

                if l_trades > 0 or s_trades > 0:
                    l_win_rate = l_wins / l_trades * 100 if l_trades > 0 else 0
                    l_avg = l_pnl / l_trades if l_trades > 0 else 0
                    s_win_rate = s_wins / s_trades * 100 if s_trades > 0 else 0
                    s_avg = s_pnl / s_trades if s_trades > 0 else 0

                    total_trades = l_trades + s_trades
                    combined_avg = (l_pnl + s_pnl) / total_trades if total_trades > 0 else 0

                    all_results.append({
                        'tf': tf,
                        'ema': ema_period,
                        'H_bars': H_bars,
                        'H_time': H_time,
                        'target': target,
                        'stop': stop,
                        'l_trades': l_trades,
                        'l_win_rate': l_win_rate,
                        'l_avg': l_avg,
                        's_trades': s_trades,
                        's_win_rate': s_win_rate,
                        's_avg': s_avg,
                        'combined_avg': combined_avg
                    })

                    l_sign = "+" if l_avg > 0 else ""
                    s_sign = "+" if s_avg > 0 else ""
                    c_sign = "+" if combined_avg > 0 else ""

                    print(f"| EMA{ema_period} | {H_bars} | {H_time}min | {target}bp | {stop}bp | "
                          f"{l_trades:,} | {l_win_rate:.1f}% | {l_sign}{l_avg:.1f}bp | "
                          f"{s_trades:,} | {s_win_rate:.1f}% | {s_sign}{s_avg:.1f}bp | "
                          f"{c_sign}{combined_avg:.1f}bp |")


# =============================================================================
# SUMMARY: BEST BY TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: BEST RESULTS BY TIMEFRAME")
print("=" * 80)

for tf in TIMEFRAMES:
    tf_results = [r for r in all_results if r['tf'] == tf]
    if tf_results:
        tf_results.sort(key=lambda x: x['combined_avg'], reverse=True)
        best = tf_results[0]
        profitable = len([r for r in tf_results if r['combined_avg'] > 0])

        print(f"\n### {tf}")
        print(f"Profitable combinations: {profitable} / {len(tf_results)}")
        print(f"Best: EMA{best['ema']}, H={best['H_bars']} bars ({best['H_time']}min), "
              f"T={best['target']}bp, S={best['stop']}bp")
        sign = "+" if best['combined_avg'] > 0 else ""
        print(f"Combined Avg P&L: {sign}{best['combined_avg']:.2f}bp")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("""
1. TIMEFRAME COMPARISON:
   - 1-minute: Most trades, smallest moves, highest fee impact
   - 5-minute: Medium trades, medium moves
   - 15-minute: Fewer trades, larger moves
   - 1-hour: Fewest trades, largest moves, lower fee impact

2. EMA MEANING BY TIMEFRAME:
   - EMA20 on 1min = 20 minute EMA
   - EMA20 on 5min = 100 minute EMA
   - EMA20 on 15min = 300 minute EMA
   - EMA20 on 1H = 20 hour EMA

3. HORIZON MEANING BY TIMEFRAME:
   - H=5 on 1min = 5 minutes holding
   - H=5 on 5min = 25 minutes holding
   - H=5 on 15min = 75 minutes holding
   - H=5 on 1H = 5 hours holding
""")

# Sort all results
all_results.sort(key=lambda x: x['combined_avg'], reverse=True)
profitable_all = len([r for r in all_results if r['combined_avg'] > 0])

print(f"\n4. OVERALL RESULTS:")
print(f"   Total combinations tested: {len(all_results)}")
print(f"   Profitable: {profitable_all} ({profitable_all/len(all_results)*100:.1f}%)")

if all_results:
    best = all_results[0]
    print(f"\n5. BEST OVERALL:")
    print(f"   Timeframe: {best['tf']}")
    print(f"   EMA: {best['ema']}")
    print(f"   Horizon: {best['H_bars']} bars ({best['H_time']} minutes)")
    print(f"   Target: {best['target']}bp, Stop: {best['stop']}bp")
    sign = "+" if best['combined_avg'] > 0 else ""
    print(f"   Combined Avg P&L: {sign}{best['combined_avg']:.2f}bp")
