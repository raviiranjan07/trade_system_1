"""
ANALYSIS: EMA Bounce - ALL TIMEFRAMES (V10)

Tests EMA bounce on timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes
Each timeframe has EMA calculated on its own candles.

Run: python scripts/debug/analysis_ema_bounce_v10_all_timeframes.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES_MINUTES = [3, 5, 10, 15, 30, 60, 120, 240, 480]
EMA_PERIODS = [9, 20, 50, 100, 200]
EMA_SLOPE_BARS = 5
TOUCH_THRESHOLD_BPS = 10
HOLD_BARS = [3, 5, 10, 20]  # How many bars to hold after entry
TARGETS_BPS = [15, 20, 30, 50]
STOPS_BPS = [10, 15, 20, 30]
FEE_BPS = 8

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - ALL TIMEFRAMES (V10)")
print("Timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

train_1m = ohlcv_1m[ohlcv_1m.index <= "2023-12-31"].copy()
print(f"Train data: {len(train_1m):,} 1-minute candles")


# =============================================================================
# RESAMPLE TO DIFFERENT TIMEFRAMES
# =============================================================================
def resample_ohlcv(df, minutes):
    """Resample 1-minute OHLCV to higher timeframe."""
    tf_str = f'{minutes}min'
    resampled = df.resample(tf_str).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled


print("\nResampling to different timeframes...")
timeframe_data = {}

for tf_min in TIMEFRAMES_MINUTES:
    if tf_min == 1:
        timeframe_data[tf_min] = train_1m.copy()
    else:
        timeframe_data[tf_min] = resample_ohlcv(train_1m, tf_min)
    print(f"  {tf_min}-minute: {len(timeframe_data[tf_min]):,} candles")


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
    Backtest EMA bounce strategy on any timeframe.
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
# RUN BACKTEST ON ALL TIMEFRAMES
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING BACKTEST ON ALL TIMEFRAMES...")
print("=" * 80)

touch_pct = TOUCH_THRESHOLD_BPS / 10000.0
all_results = []

start_time = time.time()

for tf_min in TIMEFRAMES_MINUTES:
    print(f"\nProcessing {tf_min}-minute timeframe...")

    df = timeframe_data[tf_min]
    open_arr = df['open'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    close_arr = df['close'].values.astype(np.float64)
    n = len(df)

    for ema_period in EMA_PERIODS:
        # Calculate EMA on this timeframe's candles
        ema = df['close'].ewm(span=ema_period, adjust=False).mean().values.astype(np.float64)

        for hold_bars in HOLD_BARS:
            for target in TARGETS_BPS:
                for stop in STOPS_BPS:
                    (l_trades, l_wins, l_pnl,
                     s_trades, s_wins, s_pnl) = backtest_ema_bounce(
                        open_arr, high_arr, low_arr, close_arr, ema,
                        n, hold_bars, target, stop, touch_pct, EMA_SLOPE_BARS, FEE_BPS)

                    if l_trades > 0 or s_trades > 0:
                        l_win_rate = l_wins / l_trades * 100 if l_trades > 0 else 0
                        l_avg = l_pnl / l_trades if l_trades > 0 else 0
                        s_win_rate = s_wins / s_trades * 100 if s_trades > 0 else 0
                        s_avg = s_pnl / s_trades if s_trades > 0 else 0

                        total_trades = l_trades + s_trades
                        combined_avg = (l_pnl + s_pnl) / total_trades if total_trades > 0 else 0

                        # Real hold time in minutes
                        hold_time_min = hold_bars * tf_min

                        all_results.append({
                            'tf_min': tf_min,
                            'ema': ema_period,
                            'hold_bars': hold_bars,
                            'hold_time': hold_time_min,
                            'target': target,
                            'stop': stop,
                            'l_trades': l_trades,
                            'l_win_rate': l_win_rate,
                            'l_avg': l_avg,
                            's_trades': s_trades,
                            's_win_rate': s_win_rate,
                            's_avg': s_avg,
                            'total_trades': total_trades,
                            'combined_avg': combined_avg
                        })

print(f"\nBacktest completed in {time.time() - start_time:.1f}s")
print(f"Total combinations tested: {len(all_results)}")


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

    tf_results.sort(key=lambda x: x['combined_avg'], reverse=True)
    profitable = len([r for r in tf_results if r['combined_avg'] > 0])

    print(f"\n### {tf_min}-MINUTE TIMEFRAME")
    print(f"Combinations tested: {len(tf_results)}")
    print(f"Profitable: {profitable} ({profitable/len(tf_results)*100:.1f}%)")

    # Show top 5
    print("\nTop 5 combinations:")
    print("| EMA | Hold | HoldTime | Target | Stop | Trades | Win% | Avg PnL |")
    print("|-----|------|----------|--------|------|--------|------|---------|")

    for r in tf_results[:5]:
        # Calculate approximate win rate
        l_wins = int(r['l_trades'] * r['l_win_rate'] / 100) if r['l_trades'] > 0 else 0
        s_wins = int(r['s_trades'] * r['s_win_rate'] / 100) if r['s_trades'] > 0 else 0
        total_win_rate = (l_wins + s_wins) / r['total_trades'] * 100 if r['total_trades'] > 0 else 0

        sign = "+" if r['combined_avg'] > 0 else ""
        print(f"| {r['ema']} | {r['hold_bars']}bars | {r['hold_time']}min | "
              f"{r['target']}bp | {r['stop']}bp | {r['total_trades']:,} | "
              f"{total_win_rate:.1f}% | {sign}{r['combined_avg']:.2f}bp |")


# =============================================================================
# OVERALL BEST RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("OVERALL BEST RESULTS (ALL TIMEFRAMES)")
print("=" * 80)

all_results.sort(key=lambda x: x['combined_avg'], reverse=True)
profitable_all = len([r for r in all_results if r['combined_avg'] > 0])

print(f"\nTotal profitable: {profitable_all} / {len(all_results)} ({profitable_all/len(all_results)*100:.1f}%)")

print("\nTop 20 combinations across all timeframes:")
print("| TF | EMA | Hold | HoldTime | Target | Stop | Trades | Combined Avg |")
print("|----|-----|------|----------|--------|------|--------|--------------|")

for r in all_results[:20]:
    sign = "+" if r['combined_avg'] > 0 else ""
    print(f"| {r['tf_min']}min | {r['ema']} | {r['hold_bars']}b | {r['hold_time']}min | "
          f"{r['target']}bp | {r['stop']}bp | {r['total_trades']:,} | {sign}{r['combined_avg']:.2f}bp |")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: BEST RESULT PER TIMEFRAME")
print("=" * 80)

print("\n| Timeframe | Best EMA | Best Target/Stop | Trades | Best Avg PnL | Profitable? |")
print("|-----------|----------|------------------|--------|--------------|-------------|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_results = [r for r in all_results if r['tf_min'] == tf_min]
    if tf_results:
        tf_results.sort(key=lambda x: x['combined_avg'], reverse=True)
        best = tf_results[0]
        status = "YES" if best['combined_avg'] > 0 else "NO"
        sign = "+" if best['combined_avg'] > 0 else ""
        print(f"| {tf_min}min | EMA{best['ema']} | T{best['target']}/S{best['stop']} | "
              f"{best['total_trades']:,} | {sign}{best['combined_avg']:.2f}bp | {status} |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print(f"""
1. TIMEFRAME TESTING:
   - Tested {len(TIMEFRAMES_MINUTES)} timeframes: {TIMEFRAMES_MINUTES}
   - EMA calculated on EACH timeframe's candles (correct method)
   - Total {len(all_results)} combinations tested

2. EMA MEANING BY TIMEFRAME:
   - EMA20 on 3-min chart = 60 minutes of price history
   - EMA20 on 15-min chart = 300 minutes of price history
   - EMA20 on 60-min chart = 1200 minutes (20 hours) of price history

3. PROFITABLE COMBINATIONS:
   - Total profitable: {profitable_all} / {len(all_results)}
   - Percentage: {profitable_all/len(all_results)*100:.1f}%
""")

if profitable_all > 0:
    print("4. PROFITABLE SETUPS FOUND:")
    for r in all_results[:5]:
        if r['combined_avg'] > 0:
            print(f"   - {r['tf_min']}min TF, EMA{r['ema']}, T{r['target']}/S{r['stop']}: +{r['combined_avg']:.2f}bp")
else:
    print("4. NO PROFITABLE SETUPS FOUND")
    print("   - All combinations lose money after 8bp fees")
    print("   - EMA bounce may not have systematic edge")
