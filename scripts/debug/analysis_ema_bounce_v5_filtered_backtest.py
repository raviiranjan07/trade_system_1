"""
ANALYSIS: EMA Bounce - FILTERED BACKTEST (V5)

Pattern: Trend Continuation via Consecutive EMA Support

Logic:
- Touch 1: Price touched EMA, bounced within H bars (support worked)
- Touch 2: Price comes back to SAME EMA
- Enter LONG at Touch 2
- Check: Does it bounce within same H bars?

Run: python scripts/debug/analysis_ema_bounce_v5_filtered_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [20, 50, 100, 200]
TOUCH_THRESHOLD_BPS = 15  # Within 15bp of EMA = "touch"
APPROACH_BARS = 5  # Look back to determine approach direction

# Lookback periods to find previous bounce
LOOKBACK_PERIODS = [15, 30, 60, 120, 240, 360, 500]

# Trade parameters
HORIZONS = [5, 10, 15, 30, 60]
TARGETS_BPS = [10, 15, 20]
STOPS_BPS = [10, 15, 20, 25, 30]
FEE_BPS = 8  # Round-trip fee

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - FILTERED BACKTEST (V5)")
print("Pattern: Consecutive EMA Support (Trend Continuation)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

# Calculate EMAs
print("\nCalculating EMAs...")
emas = {}
for period in EMA_PERIODS:
    emas[period] = train['close'].ewm(span=period, adjust=False).mean().values
print("EMAs calculated.")


# =============================================================================
# HELPER: Find if EMA worked as support recently (within lookback)
# =============================================================================
def find_recent_successful_bounce(ema_values, current_idx, lookback, H, min_bounce_bps=10):
    """
    Check if this EMA acted as support within lookback period.

    A successful bounce means:
    1. Price touched EMA (came from above)
    2. Price bounced UP by at least min_bounce_bps within H bars

    Returns True if found, False otherwise.
    """
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    min_bounce_pct = min_bounce_bps / 10000

    # Search backwards for a previous touch that bounced
    for search_idx in range(current_idx - 10, current_idx - lookback, -1):
        if search_idx < APPROACH_BARS + 10:
            break

        # Check if this was a touch
        price_at_search = close[search_idx]
        ema_at_search = ema_values[search_idx]
        distance = abs(price_at_search - ema_at_search) / ema_at_search

        if distance > touch_pct:
            continue  # Not a touch

        # Check if price came from above (support test)
        above_count = 0
        for k in range(1, APPROACH_BARS + 1):
            if search_idx - k >= 0:
                if close[search_idx - k] > ema_values[search_idx - k]:
                    above_count += 1

        if above_count < APPROACH_BARS * 0.6:  # At least 60% of approach bars above EMA
            continue

        # Check if it bounced within H bars
        entry_price = price_at_search
        max_up_move = 0

        for j in range(1, H + 1):
            if search_idx + j >= current_idx:  # Don't look past current bar
                break
            up_move = (high[search_idx + j] - entry_price) / entry_price
            max_up_move = max(max_up_move, up_move)

        if max_up_move >= min_bounce_pct:
            return True  # Found a successful bounce!

    return False


# =============================================================================
# FILTERED BACKTEST
# =============================================================================
def backtest_consecutive_support(ema_period, H, target_bps, stop_bps, lookback):
    """
    Backtest consecutive EMA support pattern.

    Entry condition:
    1. Price touches EMA (from above)
    2. Same EMA had a successful bounce within lookback bars
    3. Enter LONG at touch
    """
    ema = emas[ema_period]
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    target_pct = target_bps / 10000
    stop_pct = stop_bps / 10000

    trades = 0
    wins = 0
    losses = 0
    timeouts = 0
    total_pnl = 0

    valid_start = max(EMA_PERIODS) + lookback + 50
    max_end = n - H - 10

    i = valid_start
    while i < max_end:
        current_close = close[i]
        current_ema = ema[i]

        # Check if touching EMA
        distance = abs(current_close - current_ema) / current_ema
        if distance > touch_pct:
            i += 1
            continue

        # Check if price came from above (support test)
        above_count = 0
        for k in range(1, APPROACH_BARS + 1):
            if i - k >= 0:
                if close[i - k] > ema[i - k]:
                    above_count += 1

        if above_count < APPROACH_BARS * 0.6:
            i += 1
            continue

        # KEY FILTER: Check for recent successful bounce on SAME EMA with SAME H
        if not find_recent_successful_bounce(ema, i, lookback, H):
            i += 1
            continue

        # PASSED FILTER - ENTER LONG
        trades += 1
        entry = current_close
        target_price = entry * (1 + target_pct)
        stop_price = entry * (1 - stop_pct)

        outcome = 'timeout'
        trade_pnl = 0

        for j in range(1, H + 1):
            if i + j >= n:
                break

            # Check stop first
            if low[i + j] <= stop_price:
                outcome = 'stop'
                trade_pnl = -stop_bps - FEE_BPS
                break

            # Check target
            if high[i + j] >= target_price:
                outcome = 'target'
                trade_pnl = target_bps - FEE_BPS
                break

        if outcome == 'timeout':
            exit_price = close[min(i + H, n - 1)]
            trade_pnl = (exit_price - entry) / entry * 10000 - FEE_BPS

        total_pnl += trade_pnl
        if outcome == 'target':
            wins += 1
        elif outcome == 'stop':
            losses += 1
        else:
            timeouts += 1

        # Skip ahead to avoid overlapping trades
        i += H

    if trades == 0:
        return None

    return {
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'timeouts': timeouts,
        'win_rate': wins / trades * 100,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / trades
    }


# =============================================================================
# RUN TESTS
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Effect of Lookback Period")
print("(EMA50, Target=15bp, Stop=15bp)")
print("=" * 80)

print("\n| Lookback | H | Trades | Win% | Total P&L | Avg P&L | Status |")
print("|----------|---|--------|------|-----------|---------|--------|")

for H in [5, 10, 30]:
    for lookback in [30, 60, 120, 240]:
        result = backtest_consecutive_support(50, H, 15, 15, lookback)
        if result and result['trades'] >= 20:
            status = "PROFIT" if result['total_pnl'] > 0 else "LOSS"
            print(f"| {lookback} | {H} | {result['trades']:,} | {result['win_rate']:.1f}% | "
                  f"{result['total_pnl']:.0f}bp | {result['avg_pnl']:.2f}bp | {status} |")


print("\n" + "=" * 80)
print("TEST 2: Different EMAs with H=5 (Short Horizon)")
print("(Lookback=60, Target=10bp, Stop=10bp)")
print("=" * 80)

print("\n| EMA | Trades | Win% | Total P&L | Avg P&L | Status |")
print("|-----|--------|------|-----------|---------|--------|")

for ema_period in EMA_PERIODS:
    result = backtest_consecutive_support(ema_period, 5, 10, 10, 60)
    if result and result['trades'] >= 10:
        status = "PROFIT" if result['total_pnl'] > 0 else "LOSS"
        print(f"| EMA{ema_period} | {result['trades']:,} | {result['win_rate']:.1f}% | "
              f"{result['total_pnl']:.0f}bp | {result['avg_pnl']:.2f}bp | {status} |")


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================
print("\n" + "=" * 80)
print("PARAMETER OPTIMIZATION")
print("=" * 80)

all_results = []

for ema_period in EMA_PERIODS:
    print(f"Testing EMA{ema_period}...")
    for lookback in [30, 60, 120, 240]:
        for H in HORIZONS:
            for target in TARGETS_BPS:
                for stop in STOPS_BPS:
                    result = backtest_consecutive_support(ema_period, H, target, stop, lookback)
                    if result and result['trades'] >= 30:
                        all_results.append({
                            'ema': ema_period,
                            'lookback': lookback,
                            'H': H,
                            'target': target,
                            'stop': stop,
                            **result
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

def backtest_unfiltered(ema_period, H, target_bps, stop_bps):
    """V4 style - no filter, enter at every touch"""
    ema = emas[ema_period]
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    target_pct = target_bps / 10000
    stop_pct = stop_bps / 10000

    trades = 0
    wins = 0
    total_pnl = 0

    valid_start = max(EMA_PERIODS) + APPROACH_BARS + 10
    max_end = n - H - 10

    i = valid_start
    while i < max_end:
        current_close = close[i]
        current_ema = ema[i]

        distance = abs(current_close - current_ema) / current_ema
        if distance > touch_pct:
            i += 1
            continue

        above_count = 0
        for k in range(1, APPROACH_BARS + 1):
            if i - k >= 0:
                if close[i - k] > ema[i - k]:
                    above_count += 1

        if above_count < APPROACH_BARS * 0.6:
            i += 1
            continue

        trades += 1
        entry = current_close
        target_price = entry * (1 + target_pct)
        stop_price = entry * (1 - stop_pct)

        for j in range(1, H + 1):
            if i + j >= n:
                break
            if low[i + j] <= stop_price:
                total_pnl += -stop_bps - FEE_BPS
                break
            if high[i + j] >= target_price:
                total_pnl += target_bps - FEE_BPS
                wins += 1
                break
        else:
            exit_price = close[min(i + H, n - 1)]
            total_pnl += (exit_price - entry) / entry * 10000 - FEE_BPS

        i += H

    if trades == 0:
        return None
    return {'trades': trades, 'win_rate': wins/trades*100, 'total_pnl': total_pnl, 'avg_pnl': total_pnl/trades}

print("\n### EMA50, H=5, Target=10bp, Stop=10bp")
print("\n| Strategy | Trades | Win% | Total P&L | Avg P&L |")
print("|----------|--------|------|-----------|---------|")

unf = backtest_unfiltered(50, 5, 10, 10)
if unf:
    status = "PROFIT" if unf['total_pnl'] > 0 else "LOSS"
    print(f"| Unfiltered | {unf['trades']:,} | {unf['win_rate']:.1f}% | {unf['total_pnl']:.0f}bp | {unf['avg_pnl']:.2f}bp | {status}")

for lookback in [30, 60, 120]:
    filt = backtest_consecutive_support(50, 5, 10, 10, lookback)
    if filt:
        status = "PROFIT" if filt['total_pnl'] > 0 else "LOSS"
        print(f"| Filtered (LB={lookback}) | {filt['trades']:,} | {filt['win_rate']:.1f}% | {filt['total_pnl']:.0f}bp | {filt['avg_pnl']:.2f}bp | {status}")


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
