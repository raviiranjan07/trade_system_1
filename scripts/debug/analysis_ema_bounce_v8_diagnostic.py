"""
ANALYSIS: EMA Bounce - DIAGNOSTIC (V8)

Purpose: Understand WHY proper detection still doesn't work.

Tests:
1. Zero fees - does the strategy have raw edge?
2. Trade-by-trade analysis - what's happening?
3. Entry price sensitivity - does entry timing matter?

Run: python scripts/debug/analysis_ema_bounce_v8_diagnostic.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [20, 50, 100]
EMA_SLOPE_BARS = 5
TOUCH_THRESHOLD_BPS = 5
HORIZONS = [10, 30, 60]

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - DIAGNOSTIC (V8)")
print("Purpose: Understand why proper detection doesn't work")
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
# NUMBA FUNCTIONS (same as V7)
# =============================================================================
@njit
def is_ema_sloping_up(ema_arr, idx, slope_bars):
    if idx < slope_bars:
        return False
    return ema_arr[idx] > ema_arr[idx - slope_bars]


@njit
def is_bullish_engulfing(open_arr, high_arr, low_arr, close_arr, idx):
    if idx < 1:
        return False
    prev_open = open_arr[idx - 1]
    prev_close = close_arr[idx - 1]
    curr_open = open_arr[idx]
    curr_close = close_arr[idx]
    if prev_close >= prev_open:
        return False
    if curr_close <= curr_open:
        return False
    if curr_open <= prev_close and curr_close >= prev_open:
        return True
    return False


@njit
def low_touches_ema(low_val, ema_val, threshold_pct):
    distance = (low_val - ema_val) / ema_val
    return distance <= threshold_pct and distance >= -threshold_pct * 2


@njit
def collect_trade_outcomes(open_arr, high_arr, low_arr, close_arr, ema_arr,
                           n, H, touch_pct, slope_bars, use_engulfing):
    """
    Collect raw trade outcomes without target/stop.
    Returns arrays of: entry_idx, entry_price, max_gain, max_loss, exit_price
    """
    # Pre-allocate arrays (max possible size)
    max_trades = n // H
    entry_indices = np.zeros(max_trades, dtype=np.int64)
    entry_prices = np.zeros(max_trades, dtype=np.float64)
    max_gains = np.zeros(max_trades, dtype=np.float64)
    max_losses = np.zeros(max_trades, dtype=np.float64)
    exit_prices = np.zeros(max_trades, dtype=np.float64)
    final_pnls = np.zeros(max_trades, dtype=np.float64)

    trade_count = 0
    valid_start = max(250, slope_bars + 10)
    max_end = n - H - 10

    i = valid_start
    while i < max_end:
        ema_val = ema_arr[i]

        # Check all LONG conditions
        if not is_ema_sloping_up(ema_arr, i, slope_bars):
            i += 1
            continue

        if not low_touches_ema(low_arr[i], ema_val, touch_pct):
            i += 1
            continue

        if use_engulfing and not is_bullish_engulfing(open_arr, high_arr, low_arr, close_arr, i):
            i += 1
            continue

        if close_arr[i] <= ema_val:
            i += 1
            continue

        # ENTER LONG - track raw outcomes
        entry = close_arr[i]
        max_gain = 0.0
        max_loss = 0.0

        for j in range(1, H + 1):
            if i + j >= n:
                break
            gain = (high_arr[i + j] - entry) / entry * 10000.0
            loss = (entry - low_arr[i + j]) / entry * 10000.0
            if gain > max_gain:
                max_gain = gain
            if loss > max_loss:
                max_loss = loss

        exit_price = close_arr[min(i + H, n - 1)]
        final_pnl = (exit_price - entry) / entry * 10000.0

        entry_indices[trade_count] = i
        entry_prices[trade_count] = entry
        max_gains[trade_count] = max_gain
        max_losses[trade_count] = max_loss
        exit_prices[trade_count] = exit_price
        final_pnls[trade_count] = final_pnl
        trade_count += 1

        i += H

    return (entry_indices[:trade_count], entry_prices[:trade_count],
            max_gains[:trade_count], max_losses[:trade_count],
            exit_prices[:trade_count], final_pnls[:trade_count])


@njit
def backtest_with_fees(max_gains, max_losses, final_pnls, target_bps, stop_bps, fee_bps):
    """
    Simulate target/stop exits from raw outcome data.
    """
    n_trades = len(max_gains)
    wins = 0
    total_pnl = 0.0

    for i in range(n_trades):
        # Check which happens first (simplified - assume worst case)
        hit_stop = max_losses[i] >= stop_bps
        hit_target = max_gains[i] >= target_bps

        if hit_stop and hit_target:
            # Both happened - use 50/50 assumption (conservative)
            # In reality need bar-by-bar check
            if max_losses[i] > max_gains[i]:
                total_pnl += -stop_bps - fee_bps
            else:
                total_pnl += target_bps - fee_bps
                wins += 1
        elif hit_stop:
            total_pnl += -stop_bps - fee_bps
        elif hit_target:
            total_pnl += target_bps - fee_bps
            wins += 1
        else:
            total_pnl += final_pnls[i] - fee_bps

    return n_trades, wins, total_pnl


# =============================================================================
# TEST 1: RAW EDGE (NO FEES)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: RAW EDGE ANALYSIS (NO FEES)")
print("Check if strategy has edge before fees")
print("=" * 80)

touch_pct = TOUCH_THRESHOLD_BPS / 10000.0

print("\n### With Engulfing Filter")
print("\n| EMA | H | Trades | Avg Max Gain | Avg Max Loss | Avg Final PnL | Edge? |")
print("|-----|---|--------|--------------|--------------|---------------|-------|")

for ema_period in EMA_PERIODS:
    ema = emas_dict[ema_period]

    for H in HORIZONS:
        (indices, entries, gains, losses, exits, pnls) = collect_trade_outcomes(
            open_arr, high_arr, low_arr, close_arr, ema,
            n, H, touch_pct, EMA_SLOPE_BARS, True)

        if len(gains) > 0:
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            avg_pnl = np.mean(pnls)
            edge = "YES" if avg_pnl > 0 else "NO"
            sign = "+" if avg_pnl > 0 else ""

            print(f"| EMA{ema_period} | {H} | {len(gains):,} | {avg_gain:.1f}bp | "
                  f"{avg_loss:.1f}bp | {sign}{avg_pnl:.2f}bp | {edge} |")

print("\n### Without Engulfing Filter")
print("\n| EMA | H | Trades | Avg Max Gain | Avg Max Loss | Avg Final PnL | Edge? |")
print("|-----|---|--------|--------------|--------------|---------------|-------|")

for ema_period in EMA_PERIODS:
    ema = emas_dict[ema_period]

    for H in HORIZONS:
        (indices, entries, gains, losses, exits, pnls) = collect_trade_outcomes(
            open_arr, high_arr, low_arr, close_arr, ema,
            n, H, touch_pct, EMA_SLOPE_BARS, False)

        if len(gains) > 0:
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            avg_pnl = np.mean(pnls)
            edge = "YES" if avg_pnl > 0 else "NO"
            sign = "+" if avg_pnl > 0 else ""

            print(f"| EMA{ema_period} | {H} | {len(gains):,} | {avg_gain:.1f}bp | "
                  f"{avg_loss:.1f}bp | {sign}{avg_pnl:.2f}bp | {edge} |")


# =============================================================================
# TEST 2: OUTCOME DISTRIBUTION
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: OUTCOME DISTRIBUTION")
print("What percentage of trades are winners?")
print("=" * 80)

print("\n### EMA50, H=30, No Engulfing")
ema50 = emas_dict[50]
(indices, entries, gains, losses, exits, pnls) = collect_trade_outcomes(
    open_arr, high_arr, low_arr, close_arr, ema50,
    n, 30, touch_pct, EMA_SLOPE_BARS, False)

if len(pnls) > 0:
    # Distribution of final P&L
    pnls_sorted = np.sort(pnls)

    print(f"\nTotal trades: {len(pnls):,}")
    print(f"Winners (PnL > 0): {np.sum(pnls > 0):,} ({np.mean(pnls > 0)*100:.1f}%)")
    print(f"Losers (PnL < 0): {np.sum(pnls < 0):,} ({np.mean(pnls < 0)*100:.1f}%)")

    print(f"\nP&L Distribution:")
    print(f"  Min: {np.min(pnls):.1f}bp")
    print(f"  25th: {np.percentile(pnls, 25):.1f}bp")
    print(f"  50th: {np.percentile(pnls, 50):.1f}bp")
    print(f"  75th: {np.percentile(pnls, 75):.1f}bp")
    print(f"  Max: {np.max(pnls):.1f}bp")
    print(f"  Mean: {np.mean(pnls):.2f}bp")

    print(f"\nMax Gain Distribution:")
    print(f"  25th: {np.percentile(gains, 25):.1f}bp")
    print(f"  50th: {np.percentile(gains, 50):.1f}bp")
    print(f"  75th: {np.percentile(gains, 75):.1f}bp")

    print(f"\nMax Loss Distribution:")
    print(f"  25th: {np.percentile(losses, 25):.1f}bp")
    print(f"  50th: {np.percentile(losses, 50):.1f}bp")
    print(f"  75th: {np.percentile(losses, 75):.1f}bp")


# =============================================================================
# TEST 3: TARGET/STOP ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: TARGET/STOP OPTIMIZATION")
print("Find if any target/stop combination works")
print("=" * 80)

print("\n### EMA50, H=30, No Engulfing")
print("Testing different target/stop combinations with 0, 4, 8 bps fees")

targets = [10, 15, 20, 25, 30, 35]
stops = [10, 15, 20, 25, 30, 35]

print("\n| Target | Stop | Win% | Avg PnL (0bp) | Avg PnL (4bp) | Avg PnL (8bp) |")
print("|--------|------|------|---------------|---------------|---------------|")

best_combo = None
best_pnl = -999

for target in targets:
    for stop in stops:
        n_trades, wins, pnl_0 = backtest_with_fees(gains, losses, pnls, target, stop, 0)
        _, _, pnl_4 = backtest_with_fees(gains, losses, pnls, target, stop, 4)
        _, _, pnl_8 = backtest_with_fees(gains, losses, pnls, target, stop, 8)

        if n_trades > 0:
            win_rate = wins / n_trades * 100
            avg_0 = pnl_0 / n_trades
            avg_4 = pnl_4 / n_trades
            avg_8 = pnl_8 / n_trades

            s0 = "+" if avg_0 > 0 else ""
            s4 = "+" if avg_4 > 0 else ""
            s8 = "+" if avg_8 > 0 else ""

            if avg_0 > best_pnl:
                best_pnl = avg_0
                best_combo = (target, stop, win_rate, avg_0, avg_4, avg_8)

            # Only print interesting combinations
            if avg_0 > 0 or (target == stop):
                print(f"| {target}bp | {stop}bp | {win_rate:.1f}% | "
                      f"{s0}{avg_0:.2f}bp | {s4}{avg_4:.2f}bp | {s8}{avg_8:.2f}bp |")

if best_combo:
    print(f"\nBest combination (0 fees): T={best_combo[0]}bp S={best_combo[1]}bp")
    print(f"  Win rate: {best_combo[2]:.1f}%")
    print(f"  Avg PnL: {best_combo[3]:.2f}bp (0bp fee), {best_combo[4]:.2f}bp (4bp fee), {best_combo[5]:.2f}bp (8bp fee)")


# =============================================================================
# TEST 4: WHAT IF WE HOLD LONGER?
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: LONGER HORIZON TEST")
print("What if we just hold without target/stop?")
print("=" * 80)

print("\n| EMA | H | Trades | Win% | Avg PnL | After 8bp Fees |")
print("|-----|---|--------|------|---------|----------------|")

for ema_period in EMA_PERIODS:
    ema = emas_dict[ema_period]

    for H in [30, 60, 120, 240]:
        (indices, entries, gains, losses, exits, pnls) = collect_trade_outcomes(
            open_arr, high_arr, low_arr, close_arr, ema,
            n, H, touch_pct, EMA_SLOPE_BARS, False)

        if len(pnls) > 0:
            win_rate = np.mean(pnls > 0) * 100
            avg_pnl = np.mean(pnls)
            avg_net = avg_pnl - 8  # After fees

            sign = "+" if avg_pnl > 0 else ""
            net_sign = "+" if avg_net > 0 else ""

            print(f"| EMA{ema_period} | {H} | {len(pnls):,} | {win_rate:.1f}% | "
                  f"{sign}{avg_pnl:.2f}bp | {net_sign}{avg_net:.2f}bp |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("""
1. RAW EDGE CHECK:
   - Does the strategy have positive expected value BEFORE fees?
   - If Avg Final PnL > 0 (before fees), edge exists but fees kill it
   - If Avg Final PnL < 0 (before fees), no edge to begin with

2. GAIN vs LOSS:
   - Avg Max Gain vs Avg Max Loss shows theoretical R:R
   - If Avg Max Gain > Avg Max Loss, strategy has upside bias

3. OUTCOME DISTRIBUTION:
   - Win% at timeout (no target/stop) shows directional accuracy
   - P&L distribution shows if winners are bigger than losers

4. TARGET/STOP SENSITIVITY:
   - Shows which combinations work at different fee levels
   - If nothing works at 8bp but works at 4bp, strategy is marginal

5. CONCLUSIONS:
   - If 0 fee shows profit: Strategy has edge, need lower fees
   - If 0 fee shows loss: Strategy has no edge, need better filter
""")
