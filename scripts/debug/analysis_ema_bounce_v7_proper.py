"""
ANALYSIS: EMA Bounce - PROPER DETECTION (V7)

Proper filters based on how traders actually see bounces:
1. EMA Slope - EMA must be sloping UP for LONG (uptrend)
2. Touch Detection - Low/High actually touches EMA (not just close within X bp)
3. Engulfing Pattern - Bullish/Bearish engulfing at touch point

Run: python scripts/debug/analysis_ema_bounce_v7_proper.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [9, 12, 20, 50, 100, 200]
EMA_SLOPE_BARS = 5  # Bars to measure EMA slope
TOUCH_THRESHOLD_BPS = 5  # Tighter threshold - actual touch
HORIZONS = [5, 10, 15, 30, 60]
FEE_BPS = 8

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - PROPER DETECTION (V7)")
print("Filters: EMA Slope + Proper Touch + Engulfing Pattern")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

open_arr = train['open'].values.astype(np.float64)
high_arr = train['high'].values.astype(np.float64)
low_arr = train['low'].values.astype(np.float64)
close_arr = train['close'].values.astype(np.float64)
n = len(train)

# Calculate EMAs
print("\nCalculating EMAs...")
emas_dict = {}
for period in EMA_PERIODS:
    emas_dict[period] = train['close'].ewm(span=period, adjust=False).mean().values.astype(np.float64)
print("EMAs calculated.")


# =============================================================================
# NUMBA FUNCTIONS
# =============================================================================
@njit
def is_ema_sloping_up(ema_arr, idx, slope_bars):
    """Check if EMA is sloping UP (uptrend)."""
    if idx < slope_bars:
        return False
    return ema_arr[idx] > ema_arr[idx - slope_bars]


@njit
def is_ema_sloping_down(ema_arr, idx, slope_bars):
    """Check if EMA is sloping DOWN (downtrend)."""
    if idx < slope_bars:
        return False
    return ema_arr[idx] < ema_arr[idx - slope_bars]


@njit
def is_bullish_engulfing(open_arr, high_arr, low_arr, close_arr, idx):
    """
    Check if current bar is a bullish engulfing pattern.
    - Previous bar: bearish (close < open)
    - Current bar: bullish (close > open)
    - Current bar body engulfs previous bar body
    """
    if idx < 1:
        return False

    prev_open = open_arr[idx - 1]
    prev_close = close_arr[idx - 1]
    curr_open = open_arr[idx]
    curr_close = close_arr[idx]

    # Previous bar bearish
    if prev_close >= prev_open:
        return False

    # Current bar bullish
    if curr_close <= curr_open:
        return False

    # Current body engulfs previous body
    if curr_open <= prev_close and curr_close >= prev_open:
        return True

    return False


@njit
def is_bearish_engulfing(open_arr, high_arr, low_arr, close_arr, idx):
    """
    Check if current bar is a bearish engulfing pattern.
    - Previous bar: bullish (close > open)
    - Current bar: bearish (close < open)
    - Current bar body engulfs previous bar body
    """
    if idx < 1:
        return False

    prev_open = open_arr[idx - 1]
    prev_close = close_arr[idx - 1]
    curr_open = open_arr[idx]
    curr_close = close_arr[idx]

    # Previous bar bullish
    if prev_close <= prev_open:
        return False

    # Current bar bearish
    if curr_close >= curr_open:
        return False

    # Current body engulfs previous body
    if curr_open >= prev_close and curr_close <= prev_open:
        return True

    return False


@njit
def low_touches_ema(low_val, ema_val, threshold_pct):
    """Check if low touches or goes slightly below EMA."""
    distance = (low_val - ema_val) / ema_val
    # Low should be at or slightly below EMA (within threshold)
    return distance <= threshold_pct and distance >= -threshold_pct * 2


@njit
def high_touches_ema(high_val, ema_val, threshold_pct):
    """Check if high touches or goes slightly above EMA."""
    distance = (high_val - ema_val) / ema_val
    # High should be at or slightly above EMA (within threshold)
    return distance >= -threshold_pct and distance <= threshold_pct * 2


@njit
def backtest_proper_bounce(open_arr, high_arr, low_arr, close_arr, ema_arr,
                           n, H, target_bps, stop_bps, touch_pct, slope_bars,
                           fee_bps, use_engulfing):
    """
    Proper EMA bounce backtest with all filters.

    LONG entry conditions:
    1. EMA sloping UP (uptrend)
    2. Low touches EMA (actual touch)
    3. Bullish engulfing pattern (if use_engulfing=True)

    SHORT entry conditions:
    1. EMA sloping DOWN (downtrend)
    2. High touches EMA (actual touch)
    3. Bearish engulfing pattern (if use_engulfing=True)
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
    max_end = n - H - 10

    i = valid_start
    while i < max_end:
        ema_val = ema_arr[i]

        # =================================================================
        # LONG: Uptrend bounce
        # =================================================================
        if is_ema_sloping_up(ema_arr, i, slope_bars):
            # Check if low touches EMA
            if low_touches_ema(low_arr[i], ema_val, touch_pct):
                # Optional: Check for bullish engulfing
                if use_engulfing and not is_bullish_engulfing(open_arr, high_arr, low_arr, close_arr, i):
                    i += 1
                    continue

                # Close should be above EMA (bounce confirmed)
                if close_arr[i] <= ema_val:
                    i += 1
                    continue

                # ENTER LONG
                long_trades += 1
                entry = close_arr[i]
                target_price = entry * (1.0 + target_pct)
                stop_price = entry * (1.0 - stop_pct)

                won = False
                trade_pnl = 0.0

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

        # =================================================================
        # SHORT: Downtrend rejection
        # =================================================================
        if is_ema_sloping_down(ema_arr, i, slope_bars):
            # Check if high touches EMA
            if high_touches_ema(high_arr[i], ema_val, touch_pct):
                # Optional: Check for bearish engulfing
                if use_engulfing and not is_bearish_engulfing(open_arr, high_arr, low_arr, close_arr, i):
                    i += 1
                    continue

                # Close should be below EMA (rejection confirmed)
                if close_arr[i] >= ema_val:
                    i += 1
                    continue

                # ENTER SHORT
                short_trades += 1
                entry = close_arr[i]
                target_price = entry * (1.0 - target_pct)
                stop_price = entry * (1.0 + stop_pct)

                won = False
                trade_pnl = 0.0

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
# RUN TESTS
# =============================================================================
print("\nCompiling numba functions...")
start_time = time.time()

# Warm up
touch_pct = TOUCH_THRESHOLD_BPS / 10000.0
ema50 = emas_dict[50]
_ = backtest_proper_bounce(open_arr, high_arr, low_arr, close_arr, ema50,
                           n, 30, 25, 20, touch_pct, EMA_SLOPE_BARS, FEE_BPS, False)
print(f"Compilation done in {time.time() - start_time:.1f}s")


# =============================================================================
# TEST 1: Without Engulfing Filter
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: EMA Slope + Proper Touch (NO Engulfing Filter)")
print("=" * 80)

print("\n| EMA | H | Target | Stop | L_Trades | L_Win% | L_AvgPnL | S_Trades | S_Win% | S_AvgPnL | Combined |")
print("|-----|---|--------|------|----------|--------|----------|----------|--------|----------|----------|")

results_no_engulf = []

for ema_period in [20, 50, 100]:
    ema = emas_dict[ema_period]

    for H in [10, 30, 60]:
        for target in [15, 25, 35]:
            stop = int(target * 0.6)  # Stop = 60% of target

            (l_trades, l_wins, l_pnl,
             s_trades, s_wins, s_pnl) = backtest_proper_bounce(
                open_arr, high_arr, low_arr, close_arr, ema,
                n, H, target, stop, touch_pct, EMA_SLOPE_BARS, FEE_BPS, False)

            if l_trades > 0 or s_trades > 0:
                l_win_rate = l_wins / l_trades * 100 if l_trades > 0 else 0
                l_avg = l_pnl / l_trades if l_trades > 0 else 0
                s_win_rate = s_wins / s_trades * 100 if s_trades > 0 else 0
                s_avg = s_pnl / s_trades if s_trades > 0 else 0

                total_trades = l_trades + s_trades
                combined_avg = (l_pnl + s_pnl) / total_trades if total_trades > 0 else 0

                results_no_engulf.append({
                    'ema': ema_period, 'H': H, 'target': target, 'stop': stop,
                    'l_trades': l_trades, 'l_win_rate': l_win_rate, 'l_avg': l_avg,
                    's_trades': s_trades, 's_win_rate': s_win_rate, 's_avg': s_avg,
                    'combined_avg': combined_avg
                })

                l_sign = "+" if l_avg > 0 else ""
                s_sign = "+" if s_avg > 0 else ""
                c_sign = "+" if combined_avg > 0 else ""

                print(f"| EMA{ema_period} | {H} | {target}bp | {stop}bp | "
                      f"{l_trades:,} | {l_win_rate:.1f}% | {l_sign}{l_avg:.2f}bp | "
                      f"{s_trades:,} | {s_win_rate:.1f}% | {s_sign}{s_avg:.2f}bp | "
                      f"{c_sign}{combined_avg:.2f}bp |")


# =============================================================================
# TEST 2: With Engulfing Filter
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: EMA Slope + Proper Touch + ENGULFING PATTERN")
print("=" * 80)

print("\n| EMA | H | Target | Stop | L_Trades | L_Win% | L_AvgPnL | S_Trades | S_Win% | S_AvgPnL | Combined |")
print("|-----|---|--------|------|----------|--------|----------|----------|--------|----------|----------|")

results_engulf = []

for ema_period in [20, 50, 100]:
    ema = emas_dict[ema_period]

    for H in [10, 30, 60]:
        for target in [15, 25, 35]:
            stop = int(target * 0.6)

            (l_trades, l_wins, l_pnl,
             s_trades, s_wins, s_pnl) = backtest_proper_bounce(
                open_arr, high_arr, low_arr, close_arr, ema,
                n, H, target, stop, touch_pct, EMA_SLOPE_BARS, FEE_BPS, True)

            if l_trades > 0 or s_trades > 0:
                l_win_rate = l_wins / l_trades * 100 if l_trades > 0 else 0
                l_avg = l_pnl / l_trades if l_trades > 0 else 0
                s_win_rate = s_wins / s_trades * 100 if s_trades > 0 else 0
                s_avg = s_pnl / s_trades if s_trades > 0 else 0

                total_trades = l_trades + s_trades
                combined_avg = (l_pnl + s_pnl) / total_trades if total_trades > 0 else 0

                results_engulf.append({
                    'ema': ema_period, 'H': H, 'target': target, 'stop': stop,
                    'l_trades': l_trades, 'l_win_rate': l_win_rate, 'l_avg': l_avg,
                    's_trades': s_trades, 's_win_rate': s_win_rate, 's_avg': s_avg,
                    'combined_avg': combined_avg
                })

                l_sign = "+" if l_avg > 0 else ""
                s_sign = "+" if s_avg > 0 else ""
                c_sign = "+" if combined_avg > 0 else ""

                print(f"| EMA{ema_period} | {H} | {target}bp | {stop}bp | "
                      f"{l_trades:,} | {l_win_rate:.1f}% | {l_sign}{l_avg:.2f}bp | "
                      f"{s_trades:,} | {s_win_rate:.1f}% | {s_sign}{s_avg:.2f}bp | "
                      f"{c_sign}{combined_avg:.2f}bp |")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Sort by combined avg
results_no_engulf.sort(key=lambda x: x['combined_avg'], reverse=True)
results_engulf.sort(key=lambda x: x['combined_avg'], reverse=True)

profitable_no_engulf = len([r for r in results_no_engulf if r['combined_avg'] > 0])
profitable_engulf = len([r for r in results_engulf if r['combined_avg'] > 0])

print(f"\nWithout Engulfing: {profitable_no_engulf} / {len(results_no_engulf)} profitable")
print(f"With Engulfing: {profitable_engulf} / {len(results_engulf)} profitable")

if results_no_engulf:
    best = results_no_engulf[0]
    print(f"\nBest (No Engulfing): EMA{best['ema']} H={best['H']} T={best['target']} S={best['stop']}")
    print(f"   Combined Avg P&L: {best['combined_avg']:.2f}bp")

if results_engulf:
    best = results_engulf[0]
    print(f"\nBest (With Engulfing): EMA{best['ema']} H={best['H']} T={best['target']} S={best['stop']}")
    print(f"   Combined Avg P&L: {best['combined_avg']:.2f}bp")

print(f"\nTotal execution time: {time.time() - start_time:.1f}s")
