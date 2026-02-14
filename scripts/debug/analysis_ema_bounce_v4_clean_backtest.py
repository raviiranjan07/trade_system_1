"""
ANALYSIS: EMA Bounce - CLEAN BACKTEST (V4)

Logic from user:
- Analysis showed EMA touches result in bounce/drop ~75% of time
- So when we see touch, we ENTER IMMEDIATELY
- No confirmation needed - analysis IS the confirmation

Entry rules:
- Price touches EMA from above → LONG
- Price touches EMA from below → SHORT

No look-ahead bias. Touch = Entry.

Run: python scripts/debug/analysis_ema_bounce_v4_clean_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [5, 9, 12, 20, 50, 100, 200]
TOUCH_THRESHOLD_BPS = 15  # Within 15bp of EMA = "touch"
APPROACH_BARS = 5  # Look back to determine approach direction
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
SAMPLE_SIZE = 300000

# Backtest parameters (informed by V2 analysis)
TARGETS_BPS = [15, 20, 25, 30, 35]
STOPS_BPS = [10, 15, 20, 25, 30]
FEE_BPS = 8  # Round-trip fee

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - CLEAN BACKTEST (V4)")
print("No look-ahead bias. Touch = Entry.")
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
# CLEAN BACKTEST FUNCTION
# =============================================================================
def backtest_ema_touch(ema_period, H, target_bps, stop_bps):
    """
    Clean backtest: Enter at EMA touch, no confirmation required.

    LONG: Price touches EMA from above (was above, now touching)
    SHORT: Price touches EMA from below (was below, now touching)
    """
    ema = emas[ema_period]
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    target_pct = target_bps / 10000
    stop_pct = stop_bps / 10000

    # Results
    long_trades = 0
    long_wins = 0
    long_losses = 0
    long_timeouts = 0
    long_pnl = 0

    short_trades = 0
    short_wins = 0
    short_losses = 0
    short_timeouts = 0
    short_pnl = 0

    np.random.seed(42)
    max_h = H + 50
    valid_start = max(EMA_PERIODS) + APPROACH_BARS + 10

    sample_idx = np.random.choice(
        range(valid_start, n - max_h),
        size=min(SAMPLE_SIZE, n - max_h - valid_start),
        replace=False
    )

    for i in sample_idx:
        current_close = close[i]
        current_ema = ema[i]

        # Check if price is "touching" EMA (within threshold)
        distance_pct = (current_close - current_ema) / current_ema
        if abs(distance_pct) > touch_pct:
            continue  # Not touching

        # Determine approach direction (where did price come from?)
        approach_positions = []
        for j in range(1, APPROACH_BARS + 1):
            if i - j >= 0:
                rel_pos = (close[i - j] - ema[i - j]) / ema[i - j]
                approach_positions.append(rel_pos)

        if not approach_positions:
            continue

        avg_approach = np.mean(approach_positions)

        # LONG: Price came from ABOVE (support test)
        if avg_approach > touch_pct:
            long_trades += 1
            entry = current_close
            target_price = entry * (1 + target_pct)
            stop_price = entry * (1 - stop_pct)

            outcome = 'timeout'
            trade_pnl = 0

            for j in range(1, H + 1):
                if i + j >= n:
                    break

                # Check stop first (conservative)
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

            long_pnl += trade_pnl
            if outcome == 'target':
                long_wins += 1
            elif outcome == 'stop':
                long_losses += 1
            else:
                long_timeouts += 1

        # SHORT: Price came from BELOW (resistance test)
        elif avg_approach < -touch_pct:
            short_trades += 1
            entry = current_close
            target_price = entry * (1 - target_pct)
            stop_price = entry * (1 + stop_pct)

            outcome = 'timeout'
            trade_pnl = 0

            for j in range(1, H + 1):
                if i + j >= n:
                    break

                # Check stop first
                if high[i + j] >= stop_price:
                    outcome = 'stop'
                    trade_pnl = -stop_bps - FEE_BPS
                    break

                # Check target
                if low[i + j] <= target_price:
                    outcome = 'target'
                    trade_pnl = target_bps - FEE_BPS
                    break

            if outcome == 'timeout':
                exit_price = close[min(i + H, n - 1)]
                trade_pnl = (entry - exit_price) / entry * 10000 - FEE_BPS

            short_pnl += trade_pnl
            if outcome == 'target':
                short_wins += 1
            elif outcome == 'stop':
                short_losses += 1
            else:
                short_timeouts += 1

    return {
        'long': {
            'trades': long_trades,
            'wins': long_wins,
            'losses': long_losses,
            'timeouts': long_timeouts,
            'win_rate': long_wins / long_trades * 100 if long_trades > 0 else 0,
            'total_pnl': long_pnl,
            'avg_pnl': long_pnl / long_trades if long_trades > 0 else 0
        },
        'short': {
            'trades': short_trades,
            'wins': short_wins,
            'losses': short_losses,
            'timeouts': short_timeouts,
            'win_rate': short_wins / short_trades * 100 if short_trades > 0 else 0,
            'total_pnl': short_pnl,
            'avg_pnl': short_pnl / short_trades if short_trades > 0 else 0
        },
        'combined': {
            'trades': long_trades + short_trades,
            'wins': long_wins + short_wins,
            'losses': long_losses + short_losses,
            'win_rate': (long_wins + short_wins) / (long_trades + short_trades) * 100 if (long_trades + short_trades) > 0 else 0,
            'total_pnl': long_pnl + short_pnl,
            'avg_pnl': (long_pnl + short_pnl) / (long_trades + short_trades) if (long_trades + short_trades) > 0 else 0
        }
    }


# =============================================================================
# RUN BACKTEST
# =============================================================================
print("\n" + "=" * 80)
print("BACKTEST RESULTS: LONG (Touch from above)")
print("=" * 80)

print("\n### EMA50 LONG Results")
print("\n| Target | Stop | H | Trades | Wins | Losses | Win% | Total P&L | Avg P&L | Status |")
print("|--------|------|---|--------|------|--------|------|-----------|---------|--------|")

for target in [20, 25, 30]:
    for stop in [15, 20, 25]:
        for H in [30, 60]:
            result = backtest_ema_touch(50, H, target, stop)
            r = result['long']
            if r['trades'] > 0:
                status = "PROFIT" if r['total_pnl'] > 0 else "LOSS"
                print(f"| {target}bp | {stop}bp | {H} | {r['trades']:,} | "
                      f"{r['wins']:,} | {r['losses']:,} | {r['win_rate']:.1f}% | "
                      f"{r['total_pnl']:.0f}bp | {r['avg_pnl']:.2f}bp | {status} |")


print("\n" + "=" * 80)
print("BACKTEST RESULTS: SHORT (Touch from below)")
print("=" * 80)

print("\n### EMA50 SHORT Results")
print("\n| Target | Stop | H | Trades | Wins | Losses | Win% | Total P&L | Avg P&L | Status |")
print("|--------|------|---|--------|------|--------|------|-----------|---------|--------|")

for target in [20, 25, 30]:
    for stop in [15, 20, 25]:
        for H in [30, 60]:
            result = backtest_ema_touch(50, H, target, stop)
            r = result['short']
            if r['trades'] > 0:
                status = "PROFIT" if r['total_pnl'] > 0 else "LOSS"
                print(f"| {target}bp | {stop}bp | {H} | {r['trades']:,} | "
                      f"{r['wins']:,} | {r['losses']:,} | {r['win_rate']:.1f}% | "
                      f"{r['total_pnl']:.0f}bp | {r['avg_pnl']:.2f}bp | {status} |")


print("\n" + "=" * 80)
print("BACKTEST RESULTS: COMBINED (LONG + SHORT)")
print("=" * 80)

print("\n### EMA50 Combined Results")
print("\n| Target | Stop | H | Trades | Win% | Total P&L | Avg P&L | Status |")
print("|--------|------|---|--------|------|-----------|---------|--------|")

for target in [20, 25, 30]:
    for stop in [15, 20, 25]:
        for H in [30, 60]:
            result = backtest_ema_touch(50, H, target, stop)
            r = result['combined']
            if r['trades'] > 0:
                status = "PROFIT" if r['total_pnl'] > 0 else "LOSS"
                print(f"| {target}bp | {stop}bp | {H} | {r['trades']:,} | {r['win_rate']:.1f}% | "
                      f"{r['total_pnl']:.0f}bp | {r['avg_pnl']:.2f}bp | {status} |")


# =============================================================================
# COMPARE ALL EMAs
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON ACROSS EMAs (Target=25bp, Stop=20bp, H=30)")
print("=" * 80)

print("\n| EMA | LONG Trades | LONG Win% | LONG P&L | SHORT Trades | SHORT Win% | SHORT P&L | Combined P&L |")
print("|-----|-------------|-----------|----------|--------------|------------|-----------|--------------|")

for ema_period in EMA_PERIODS:
    result = backtest_ema_touch(ema_period, 30, 25, 20)
    l = result['long']
    s = result['short']
    c = result['combined']

    l_status = "+" if l['total_pnl'] > 0 else ""
    s_status = "+" if s['total_pnl'] > 0 else ""
    c_status = "PROFIT" if c['total_pnl'] > 0 else "LOSS"

    print(f"| EMA{ema_period} | {l['trades']:,} | {l['win_rate']:.1f}% | {l_status}{l['total_pnl']:.0f}bp | "
          f"{s['trades']:,} | {s['win_rate']:.1f}% | {s_status}{s['total_pnl']:.0f}bp | "
          f"{c['total_pnl']:.0f}bp ({c_status}) |")


# =============================================================================
# FIND BEST PARAMETERS
# =============================================================================
print("\n" + "=" * 80)
print("PARAMETER OPTIMIZATION")
print("=" * 80)

all_results = []

for ema_period in [20, 50, 100]:
    for target in TARGETS_BPS:
        for stop in STOPS_BPS:
            for H in HORIZONS:
                result = backtest_ema_touch(ema_period, H, target, stop)
                c = result['combined']
                if c['trades'] >= 1000:
                    all_results.append({
                        'ema': ema_period,
                        'target': target,
                        'stop': stop,
                        'H': H,
                        **c
                    })

# Sort by total P&L
all_results.sort(key=lambda x: x['total_pnl'], reverse=True)

print("\n### Top 15 Most Profitable Combinations")
print("\n| Rank | EMA | Target | Stop | H | Trades | Win% | Total P&L | Avg P&L |")
print("|------|-----|--------|------|---|--------|------|-----------|---------|")

for i, r in enumerate(all_results[:15]):
    print(f"| {i+1} | EMA{r['ema']} | {r['target']}bp | {r['stop']}bp | {r['H']} | "
          f"{r['trades']:,} | {r['win_rate']:.1f}% | {r['total_pnl']:.0f}bp | {r['avg_pnl']:.2f}bp |")


# Count profitable vs unprofitable
profitable = len([r for r in all_results if r['total_pnl'] > 0])
total = len(all_results)

print(f"\n### Summary")
print(f"Profitable combinations: {profitable} / {total} ({profitable/total*100:.1f}%)")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

if all_results:
    best = all_results[0]
    worst = all_results[-1]

    print(f"\n1. Best combination:")
    print(f"   EMA{best['ema']}, Target={best['target']}bp, Stop={best['stop']}bp, H={best['H']}")
    print(f"   Win rate: {best['win_rate']:.1f}%")
    print(f"   Avg P&L: {best['avg_pnl']:.2f}bp per trade")
    print(f"   Total P&L: {best['total_pnl']:.0f}bp over {best['trades']:,} trades")

    print(f"\n2. Worst combination:")
    print(f"   EMA{worst['ema']}, Target={worst['target']}bp, Stop={worst['stop']}bp, H={worst['H']}")
    print(f"   Win rate: {worst['win_rate']:.1f}%")
    print(f"   Total P&L: {worst['total_pnl']:.0f}bp")

print("\n3. This is a CLEAN backtest:")
print("   - Entry at EMA touch (no confirmation)")
print("   - No look-ahead bias")
print("   - Fees included (8bp round-trip)")
print("   - Both LONG and SHORT tested")
