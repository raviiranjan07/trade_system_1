"""
ANALYSIS: EMA Bounce - FILTERED BACKTEST (V5 OPTIMIZED)

Pattern: Consecutive EMA Support (Trend Continuation)
Optimized with Numba for 10-100x faster execution.

Logic:
- Touch 1: Price touched EMA, bounced within H bars (support worked)
- Touch 2: Price comes back to SAME EMA
- Enter LONG at Touch 2
- Check: Does it bounce within same H bars?

Run: python scripts/debug/analysis_ema_bounce_v5_optimized.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit, prange
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [20, 50, 100, 200]
TOUCH_THRESHOLD_BPS = 15
APPROACH_BARS = 5
LOOKBACK_PERIODS = [30, 60, 120, 240]
HORIZONS = [5, 10, 15, 30, 60]
TARGETS_BPS = [10, 15, 20]
STOPS_BPS = [10, 15, 20, 25, 30]
FEE_BPS = 8

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - FILTERED BACKTEST (V5 OPTIMIZED)")
print("Pattern: Consecutive EMA Support (Trend Continuation)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values.astype(np.float64)
high = train['high'].values.astype(np.float64)
low = train['low'].values.astype(np.float64)
n = len(train)

# Calculate EMAs
print("\nCalculating EMAs...")
emas_dict = {}
for period in EMA_PERIODS:
    emas_dict[period] = train['close'].ewm(span=period, adjust=False).mean().values.astype(np.float64)
print("EMAs calculated.")


# =============================================================================
# NUMBA OPTIMIZED FUNCTIONS
# =============================================================================
@njit
def check_came_from_above(close_arr, ema_arr, idx, approach_bars, touch_pct):
    """Check if price came from above EMA (support test)."""
    above_count = 0
    for k in range(1, approach_bars + 1):
        if idx - k >= 0:
            if close_arr[idx - k] > ema_arr[idx - k]:
                above_count += 1
    return above_count >= approach_bars * 0.6


@njit
def find_recent_bounce(close_arr, high_arr, ema_arr, current_idx, lookback, H,
                       touch_pct, min_bounce_pct, approach_bars):
    """
    Check if EMA acted as support within lookback period.
    Returns True if found a successful bounce.
    """
    for search_idx in range(current_idx - 10, max(current_idx - lookback, approach_bars + 10), -1):
        # Check if this was a touch
        price = close_arr[search_idx]
        ema = ema_arr[search_idx]
        distance = abs(price - ema) / ema

        if distance > touch_pct:
            continue

        # Check if came from above
        if not check_came_from_above(close_arr, ema_arr, search_idx, approach_bars, touch_pct):
            continue

        # Check if bounced within H bars
        max_up = 0.0
        for j in range(1, min(H + 1, current_idx - search_idx)):
            up_move = (high_arr[search_idx + j] - price) / price
            if up_move > max_up:
                max_up = up_move

        if max_up >= min_bounce_pct:
            return True

    return False


@njit
def backtest_single_config(close_arr, high_arr, low_arr, ema_arr,
                           n, H, target_bps, stop_bps, lookback,
                           touch_pct, approach_bars, fee_bps):
    """
    Run backtest for a single configuration.
    Returns: (trades, wins, losses, timeouts, total_pnl)
    """
    min_bounce_pct = 0.001  # 10bp minimum bounce to confirm previous worked
    target_pct = target_bps / 10000.0
    stop_pct = stop_bps / 10000.0

    trades = 0
    wins = 0
    losses = 0
    timeouts = 0
    total_pnl = 0.0

    valid_start = 250 + lookback
    max_end = n - H - 10

    i = valid_start
    while i < max_end:
        current_close = close_arr[i]
        current_ema = ema_arr[i]

        # Check if touching EMA
        distance = abs(current_close - current_ema) / current_ema
        if distance > touch_pct:
            i += 1
            continue

        # Check if came from above
        if not check_came_from_above(close_arr, ema_arr, i, approach_bars, touch_pct):
            i += 1
            continue

        # KEY FILTER: Check for recent successful bounce
        if not find_recent_bounce(close_arr, high_arr, ema_arr, i, lookback, H,
                                  touch_pct, min_bounce_pct, approach_bars):
            i += 1
            continue

        # PASSED FILTER - ENTER LONG
        trades += 1
        entry = current_close
        target_price = entry * (1.0 + target_pct)
        stop_price = entry * (1.0 - stop_pct)

        outcome = 0  # 0=timeout, 1=win, 2=loss
        trade_pnl = 0.0

        for j in range(1, H + 1):
            if i + j >= n:
                break

            # Check stop first
            if low_arr[i + j] <= stop_price:
                outcome = 2
                trade_pnl = -stop_bps - fee_bps
                break

            # Check target
            if high_arr[i + j] >= target_price:
                outcome = 1
                trade_pnl = target_bps - fee_bps
                break

        if outcome == 0:
            exit_idx = min(i + H, n - 1)
            exit_price = close_arr[exit_idx]
            trade_pnl = (exit_price - entry) / entry * 10000.0 - fee_bps

        total_pnl += trade_pnl
        if outcome == 1:
            wins += 1
        elif outcome == 2:
            losses += 1
        else:
            timeouts += 1

        # Skip ahead to avoid overlapping
        i += H

    return trades, wins, losses, timeouts, total_pnl


@njit
def backtest_unfiltered(close_arr, high_arr, low_arr, ema_arr,
                        n, H, target_bps, stop_bps,
                        touch_pct, approach_bars, fee_bps):
    """
    Unfiltered backtest - enter at every touch from above.
    """
    target_pct = target_bps / 10000.0
    stop_pct = stop_bps / 10000.0

    trades = 0
    wins = 0
    total_pnl = 0.0

    valid_start = 250
    max_end = n - H - 10

    i = valid_start
    while i < max_end:
        current_close = close_arr[i]
        current_ema = ema_arr[i]

        distance = abs(current_close - current_ema) / current_ema
        if distance > touch_pct:
            i += 1
            continue

        if not check_came_from_above(close_arr, ema_arr, i, approach_bars, touch_pct):
            i += 1
            continue

        trades += 1
        entry = current_close
        target_price = entry * (1.0 + target_pct)
        stop_price = entry * (1.0 - stop_pct)

        for j in range(1, H + 1):
            if i + j >= n:
                break
            if low_arr[i + j] <= stop_price:
                total_pnl += -stop_bps - fee_bps
                break
            if high_arr[i + j] >= target_price:
                total_pnl += target_bps - fee_bps
                wins += 1
                break
        else:
            exit_price = close_arr[min(i + H, n - 1)]
            total_pnl += (exit_price - entry) / entry * 10000.0 - fee_bps

        i += H

    if trades == 0:
        return 0, 0.0, 0.0
    return trades, wins / trades * 100.0, total_pnl / trades


# =============================================================================
# RUN TESTS
# =============================================================================
print("\nCompiling numba functions (first run slower)...")
start_time = time.time()

touch_pct = TOUCH_THRESHOLD_BPS / 10000.0

# Warm up numba
ema50 = emas_dict[50]
_ = backtest_single_config(close, high, low, ema50, n, 30, 15, 15, 60,
                           touch_pct, APPROACH_BARS, FEE_BPS)
print(f"Compilation done in {time.time() - start_time:.1f}s")


print("\n" + "=" * 80)
print("TEST 1: Effect of Lookback Period")
print("(EMA50, Target=15bp, Stop=15bp)")
print("=" * 80)

print("\n| Lookback | H | Trades | Win% | Total P&L | Avg P&L | Status |")
print("|----------|---|--------|------|-----------|---------|--------|")

ema50 = emas_dict[50]
for H in [5, 10, 30]:
    for lookback in [30, 60, 120, 240]:
        trades, wins, losses, timeouts, total_pnl = backtest_single_config(
            close, high, low, ema50, n, H, 15, 15, lookback,
            touch_pct, APPROACH_BARS, FEE_BPS)

        if trades >= 20:
            win_rate = wins / trades * 100
            avg_pnl = total_pnl / trades
            status = "PROFIT" if total_pnl > 0 else "LOSS"
            print(f"| {lookback} | {H} | {trades:,} | {win_rate:.1f}% | "
                  f"{total_pnl:.0f}bp | {avg_pnl:.2f}bp | {status} |")


print("\n" + "=" * 80)
print("TEST 2: Different EMAs with H=5 (Short Horizon)")
print("(Lookback=60, Target=10bp, Stop=10bp)")
print("=" * 80)

print("\n| EMA | Trades | Win% | Total P&L | Avg P&L | Status |")
print("|-----|--------|------|-----------|---------|--------|")

for ema_period in EMA_PERIODS:
    ema = emas_dict[ema_period]
    trades, wins, losses, timeouts, total_pnl = backtest_single_config(
        close, high, low, ema, n, 5, 10, 10, 60,
        touch_pct, APPROACH_BARS, FEE_BPS)

    if trades >= 10:
        win_rate = wins / trades * 100
        avg_pnl = total_pnl / trades
        status = "PROFIT" if total_pnl > 0 else "LOSS"
        print(f"| EMA{ema_period} | {trades:,} | {win_rate:.1f}% | "
              f"{total_pnl:.0f}bp | {avg_pnl:.2f}bp | {status} |")


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================
print("\n" + "=" * 80)
print("PARAMETER OPTIMIZATION")
print("=" * 80)

all_results = []

for ema_period in EMA_PERIODS:
    print(f"Testing EMA{ema_period}...")
    ema = emas_dict[ema_period]

    for lookback in LOOKBACK_PERIODS:
        for H in HORIZONS:
            for target in TARGETS_BPS:
                for stop in STOPS_BPS:
                    trades, wins, losses, timeouts, total_pnl = backtest_single_config(
                        close, high, low, ema, n, H, target, stop, lookback,
                        touch_pct, APPROACH_BARS, FEE_BPS)

                    if trades >= 30:
                        win_rate = wins / trades * 100
                        avg_pnl = total_pnl / trades
                        all_results.append({
                            'ema': ema_period,
                            'lookback': lookback,
                            'H': H,
                            'target': target,
                            'stop': stop,
                            'trades': trades,
                            'wins': wins,
                            'win_rate': win_rate,
                            'total_pnl': total_pnl,
                            'avg_pnl': avg_pnl
                        })

# Sort by avg P&L
all_results.sort(key=lambda x: x['avg_pnl'], reverse=True)

profitable = len([r for r in all_results if r['total_pnl'] > 0])
total = len(all_results)

print(f"\nTotal combinations: {total}")
print(f"Profitable: {profitable} ({profitable/total*100:.1f}%)" if total > 0 else "No results")

print("\n### Top 20 Combinations by Avg P&L")
print("\n| Rank | EMA | Lookback | H | Target | Stop | Trades | Win% | Total P&L | Avg P&L |")
print("|------|-----|----------|---|--------|------|--------|------|-----------|---------|")

for i, r in enumerate(all_results[:20]):
    sign = "+" if r['avg_pnl'] > 0 else ""
    print(f"| {i+1} | EMA{r['ema']} | {r['lookback']} | {r['H']} | {r['target']}bp | {r['stop']}bp | "
          f"{r['trades']:,} | {r['win_rate']:.1f}% | {r['total_pnl']:.0f}bp | {sign}{r['avg_pnl']:.2f}bp |")


# =============================================================================
# COMPARISON: FILTERED VS UNFILTERED
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: FILTERED (V5) vs UNFILTERED (V4)")
print("=" * 80)

print("\n### EMA50, H=5, Target=10bp, Stop=10bp")
print("\n| Strategy | Trades | Win% | Avg P&L | Status |")
print("|----------|--------|------|---------|--------|")

# Unfiltered
ema50 = emas_dict[50]
trades_unf, win_rate_unf, avg_pnl_unf = backtest_unfiltered(
    close, high, low, ema50, n, 5, 10, 10,
    touch_pct, APPROACH_BARS, FEE_BPS)
status = "PROFIT" if avg_pnl_unf > 0 else "LOSS"
print(f"| Unfiltered | {trades_unf:,} | {win_rate_unf:.1f}% | {avg_pnl_unf:.2f}bp | {status} |")

# Filtered with different lookbacks
for lookback in [30, 60, 120]:
    trades, wins, losses, timeouts, total_pnl = backtest_single_config(
        close, high, low, ema50, n, 5, 10, 10, lookback,
        touch_pct, APPROACH_BARS, FEE_BPS)
    if trades > 0:
        win_rate = wins / trades * 100
        avg_pnl = total_pnl / trades
        status = "PROFIT" if total_pnl > 0 else "LOSS"
        print(f"| Filtered (LB={lookback}) | {trades:,} | {win_rate:.1f}% | {avg_pnl:.2f}bp | {status} |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

if all_results:
    best = all_results[0]
    print(f"\n1. Best combination:")
    print(f"   EMA{best['ema']}, Lookback={best['lookback']}, H={best['H']}")
    print(f"   Target={best['target']}bp, Stop={best['stop']}bp")
    print(f"   Trades: {best['trades']:,}, Win rate: {best['win_rate']:.1f}%")
    print(f"   Avg P&L: {best['avg_pnl']:.2f}bp per trade")

    print(f"\n2. Filter effectiveness:")
    print(f"   Profitable combinations: {profitable} / {total}")

print("\n3. Pattern tested:")
print("   Touch 1 → Bounced within H bars")
print("   Touch 2 → Enter LONG, check if bounces within same H")
print("   (Consecutive EMA support = trend continuation)")

print(f"\nTotal execution time: {time.time() - start_time:.1f}s")
