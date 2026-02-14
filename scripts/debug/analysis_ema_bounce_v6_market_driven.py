"""
ANALYSIS: EMA Bounce - MARKET-DRIVEN BACKTEST (V6)

Key difference from V4/V5:
- Target/Stop based on MEASURED bounce/correction from V2 analysis
- Not arbitrary fixed values

Logic:
- V2 measured: EMA50 H=30 bounces ~28bp median, correction ~19bp median
- So we use: Target ~25bp, Stop ~20bp (based on actual market behavior)

Run: python scripts/debug/analysis_ema_bounce_v6_market_driven.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [5, 9, 12, 20, 50, 100, 200]
TOUCH_THRESHOLD_BPS = 15
APPROACH_BARS = 5
FEE_BPS = 8

# =============================================================================
# MEASURED BOUNCE/CORRECTION FROM V2 (Market-Driven Parameters)
# Format: (EMA, H) -> (bounce_median, correction_median)
# =============================================================================
MEASURED_DATA = {
    # EMA5
    (5, 3): (27.0, 10.1),
    (5, 5): (35.6, 12.6),
    (5, 10): (48.6, 17.4),
    (5, 15): (57.0, 20.5),
    (5, 30): (72.1, 31.2),
    (5, 60): (91.0, 47.3),
    # EMA9
    (9, 3): (18.2, 7.7),
    (9, 5): (24.3, 9.8),
    (9, 10): (34.3, 13.6),
    (9, 15): (39.6, 16.9),
    (9, 30): (51.0, 25.5),
    (9, 60): (68.0, 39.6),
    # EMA12
    (12, 3): (15.2, 7.1),
    (12, 5): (20.4, 9.2),
    (12, 10): (29.0, 12.8),
    (12, 15): (34.1, 15.9),
    (12, 30): (43.8, 23.5),
    (12, 60): (59.0, 36.3),
    # EMA20
    (20, 3): (12.5, 6.1),
    (20, 5): (16.4, 8.0),
    (20, 10): (23.1, 11.6),
    (20, 15): (27.4, 14.6),
    (20, 30): (36.6, 21.9),
    (20, 60): (49.3, 34.0),
    # EMA50
    (50, 3): (9.0, 4.9),
    (50, 5): (11.9, 6.7),
    (50, 10): (17.2, 9.9),
    (50, 15): (20.4, 12.6),
    (50, 30): (28.3, 18.7),
    (50, 60): (39.1, 29.0),
    # EMA100
    (100, 3): (7.9, 4.3),
    (100, 5): (10.4, 5.8),
    (100, 10): (14.9, 8.7),
    (100, 15): (18.0, 11.1),
    (100, 30): (24.3, 16.9),
    (100, 60): (33.5, 26.3),
    # EMA200
    (200, 3): (7.2, 3.9),
    (200, 5): (9.7, 5.2),
    (200, 10): (13.6, 8.1),
    (200, 15): (16.4, 10.3),
    (200, 30): (22.3, 15.3),
    (200, 60): (30.8, 23.3),
}

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - MARKET-DRIVEN BACKTEST (V6)")
print("Target/Stop based on measured bounce/correction from V2")
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
# NUMBA OPTIMIZED BACKTEST
# =============================================================================
@njit
def check_came_from_above(close_arr, ema_arr, idx, approach_bars):
    """Check if price came from above EMA."""
    above_count = 0
    for k in range(1, approach_bars + 1):
        if idx - k >= 0:
            if close_arr[idx - k] > ema_arr[idx - k]:
                above_count += 1
    return above_count >= approach_bars * 0.6


@njit
def check_came_from_below(close_arr, ema_arr, idx, approach_bars):
    """Check if price came from below EMA."""
    below_count = 0
    for k in range(1, approach_bars + 1):
        if idx - k >= 0:
            if close_arr[idx - k] < ema_arr[idx - k]:
                below_count += 1
    return below_count >= approach_bars * 0.6


@njit
def backtest_market_driven(close_arr, high_arr, low_arr, ema_arr,
                           n, H, target_bps, stop_bps, touch_pct,
                           approach_bars, fee_bps):
    """
    Backtest with market-driven target/stop.
    Tests both LONG (bounce) and SHORT (rejection).
    """
    target_pct = target_bps / 10000.0
    stop_pct = stop_bps / 10000.0

    # LONG stats
    long_trades = 0
    long_wins = 0
    long_pnl = 0.0

    # SHORT stats
    short_trades = 0
    short_wins = 0
    short_pnl = 0.0

    valid_start = 250
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

        # LONG: Price came from above (support test)
        if check_came_from_above(close_arr, ema_arr, i, approach_bars):
            long_trades += 1
            entry = current_close
            target_price = entry * (1.0 + target_pct)
            stop_price = entry * (1.0 - stop_pct)

            trade_pnl = 0.0
            won = False

            for j in range(1, H + 1):
                if i + j >= n:
                    break
                # Check stop first
                if low_arr[i + j] <= stop_price:
                    trade_pnl = -stop_bps - fee_bps
                    break
                # Check target
                if high_arr[i + j] >= target_price:
                    trade_pnl = target_bps - fee_bps
                    won = True
                    break
            else:
                # Timeout - exit at close
                exit_price = close_arr[min(i + H, n - 1)]
                trade_pnl = (exit_price - entry) / entry * 10000.0 - fee_bps

            long_pnl += trade_pnl
            if won:
                long_wins += 1

            i += H
            continue

        # SHORT: Price came from below (resistance test)
        if check_came_from_below(close_arr, ema_arr, i, approach_bars):
            short_trades += 1
            entry = current_close
            target_price = entry * (1.0 - target_pct)
            stop_price = entry * (1.0 + stop_pct)

            trade_pnl = 0.0
            won = False

            for j in range(1, H + 1):
                if i + j >= n:
                    break
                # Check stop first
                if high_arr[i + j] >= stop_price:
                    trade_pnl = -stop_bps - fee_bps
                    break
                # Check target
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
# RUN BACKTEST WITH MARKET-DRIVEN PARAMETERS
# =============================================================================
print("\n" + "=" * 80)
print("MARKET-DRIVEN BACKTEST RESULTS")
print("(Target = 90% of measured bounce, Stop = measured correction)")
print("=" * 80)

touch_pct = TOUCH_THRESHOLD_BPS / 10000.0

results = []

print("\n| EMA | H | Bounce | Corr | Target | Stop | L_Trades | L_Win% | L_AvgPnL | S_Trades | S_Win% | S_AvgPnL |")
print("|-----|---|--------|------|--------|------|----------|--------|----------|----------|--------|----------|")

for ema_period in EMA_PERIODS:
    ema = emas_dict[ema_period]

    for H in [3, 5, 10, 15, 30, 60]:
        key = (ema_period, H)
        if key not in MEASURED_DATA:
            continue

        bounce_med, corr_med = MEASURED_DATA[key]

        # Market-driven parameters:
        # Target = 90% of median bounce (realistic, not too aggressive)
        # Stop = median correction (based on what market actually does)
        target_bps = int(bounce_med * 0.9)
        stop_bps = int(corr_med)

        # Skip if target <= fees (not tradeable)
        if target_bps <= FEE_BPS:
            continue

        # Run backtest
        (long_trades, long_wins, long_pnl,
         short_trades, short_wins, short_pnl) = backtest_market_driven(
            close, high, low, ema, n, H, target_bps, stop_bps,
            touch_pct, APPROACH_BARS, FEE_BPS)

        # Calculate stats
        l_win_rate = long_wins / long_trades * 100 if long_trades > 0 else 0
        l_avg_pnl = long_pnl / long_trades if long_trades > 0 else 0

        s_win_rate = short_wins / short_trades * 100 if short_trades > 0 else 0
        s_avg_pnl = short_pnl / short_trades if short_trades > 0 else 0

        total_trades = long_trades + short_trades
        total_pnl = long_pnl + short_pnl
        combined_avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        results.append({
            'ema': ema_period,
            'H': H,
            'bounce_med': bounce_med,
            'corr_med': corr_med,
            'target': target_bps,
            'stop': stop_bps,
            'long_trades': long_trades,
            'long_win_rate': l_win_rate,
            'long_avg_pnl': l_avg_pnl,
            'short_trades': short_trades,
            'short_win_rate': s_win_rate,
            'short_avg_pnl': s_avg_pnl,
            'combined_avg_pnl': combined_avg_pnl
        })

        l_status = "+" if l_avg_pnl > 0 else ""
        s_status = "+" if s_avg_pnl > 0 else ""

        print(f"| EMA{ema_period} | {H} | {bounce_med:.0f}bp | {corr_med:.0f}bp | "
              f"{target_bps}bp | {stop_bps}bp | {long_trades:,} | {l_win_rate:.1f}% | "
              f"{l_status}{l_avg_pnl:.2f}bp | {short_trades:,} | {s_win_rate:.1f}% | {s_status}{s_avg_pnl:.2f}bp |")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: COMBINED (LONG + SHORT)")
print("=" * 80)

print("\n| EMA | H | Target | Stop | Trades | Combined Avg P&L | Status |")
print("|-----|---|--------|------|--------|------------------|--------|")

results.sort(key=lambda x: x['combined_avg_pnl'], reverse=True)

for r in results:
    status = "PROFIT" if r['combined_avg_pnl'] > 0 else "LOSS"
    sign = "+" if r['combined_avg_pnl'] > 0 else ""
    total_trades = r['long_trades'] + r['short_trades']
    print(f"| EMA{r['ema']} | {r['H']} | {r['target']}bp | {r['stop']}bp | "
          f"{total_trades:,} | {sign}{r['combined_avg_pnl']:.2f}bp | {status} |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

profitable = [r for r in results if r['combined_avg_pnl'] > 0]
total = len(results)

print(f"\n1. Profitable combinations: {len(profitable)} / {total}")

if profitable:
    best = profitable[0]
    print(f"\n2. Best combination:")
    print(f"   EMA{best['ema']}, H={best['H']}")
    print(f"   Target={best['target']}bp (90% of {best['bounce_med']:.0f}bp bounce)")
    print(f"   Stop={best['stop']}bp (median correction)")
    print(f"   Avg P&L: {best['combined_avg_pnl']:.2f}bp per trade")

if results:
    worst = results[-1]
    print(f"\n3. Worst combination:")
    print(f"   EMA{worst['ema']}, H={worst['H']}")
    print(f"   Avg P&L: {worst['combined_avg_pnl']:.2f}bp per trade")

print("\n4. Market-driven approach:")
print("   - Target = 90% of measured median bounce")
print("   - Stop = measured median correction")
print("   - Based on actual market behavior, not arbitrary values")
