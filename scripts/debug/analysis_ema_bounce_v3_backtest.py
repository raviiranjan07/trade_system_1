"""
ANALYSIS: EMA Bounce - Path Sequence & P&L Backtest (V3)

Part 1: PATH SEQUENCE
- When does bounce happen vs correction?
- Does profit come before or after drawdown?

Part 2: P&L BACKTEST
- Simulate trades with specific target/stop/fees
- Calculate actual profitability

Run: python scripts/debug/analysis_ema_bounce_v3_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [5, 9, 12, 20, 50, 100, 200]
TOUCH_THRESHOLD_BPS = 15
APPROACH_BARS = 5
BOUNCE_CONFIRM_BPS = 10
HORIZONS = [5, 10, 15, 30, 60]
SAMPLE_SIZE = 200000

# Backtest parameters
TARGETS_BPS = [15, 20, 25, 30]
STOPS_BPS = [10, 15, 20, 25]
FEE_BPS = 8  # Round-trip fee

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS: EMA BOUNCE - PATH SEQUENCE & P&L BACKTEST")
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
# PART 1: PATH SEQUENCE ANALYSIS
# =============================================================================
def analyze_path_sequence(ema_period, H):
    """
    Analyze WHEN bounce and correction happen.

    Returns:
    - bounce_first_pct: % of trades where max bounce happens BEFORE max correction
    - avg_bounce_bar: Average bar when max bounce occurs
    - avg_correction_bar: Average bar when max correction occurs
    """
    ema = emas[ema_period]
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    confirm_pct = BOUNCE_CONFIRM_BPS / 10000

    bounce_first_count = 0
    correction_first_count = 0
    bounce_bars = []
    correction_bars = []

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

        # Check if touching EMA
        distance_pct = (current_close - current_ema) / current_ema
        if abs(distance_pct) > touch_pct:
            continue

        # Check approach direction
        approach_positions = []
        for j in range(1, APPROACH_BARS + 1):
            if i - j >= 0:
                rel_pos = (close[i - j] - ema[i - j]) / ema[i - j]
                approach_positions.append(rel_pos)

        if not approach_positions:
            continue

        avg_approach = np.mean(approach_positions)

        # Only analyze bounces (price came from above)
        if avg_approach <= touch_pct:
            continue

        # Check if bounce confirmed
        bounce_confirmed = False
        for j in range(1, 11):
            if i + j >= n:
                break
            future_rel = (close[i + j] - ema[i + j]) / ema[i + j]
            if future_rel > confirm_pct:
                bounce_confirmed = True
                break
            elif future_rel < -confirm_pct:
                break

        if not bounce_confirmed:
            continue

        # Track path: when does max bounce and max correction occur?
        entry = current_close
        max_bounce = 0
        max_bounce_bar = 0
        max_correction = 0
        max_correction_bar = 0

        for j in range(1, H + 1):
            if i + j >= n:
                break

            up_move = (high[i + j] - entry) / entry * 10000
            down_move = (entry - low[i + j]) / entry * 10000

            if up_move > max_bounce:
                max_bounce = up_move
                max_bounce_bar = j

            if down_move > max_correction:
                max_correction = down_move
                max_correction_bar = j

        if max_bounce_bar > 0 and max_correction_bar > 0:
            bounce_bars.append(max_bounce_bar)
            correction_bars.append(max_correction_bar)

            if max_bounce_bar < max_correction_bar:
                bounce_first_count += 1
            else:
                correction_first_count += 1

    total = bounce_first_count + correction_first_count
    if total == 0:
        return None

    return {
        'total': total,
        'bounce_first_pct': bounce_first_count / total * 100,
        'correction_first_pct': correction_first_count / total * 100,
        'avg_bounce_bar': np.mean(bounce_bars) if bounce_bars else 0,
        'avg_correction_bar': np.mean(correction_bars) if correction_bars else 0
    }


print("\n" + "=" * 80)
print("PART 1: PATH SEQUENCE ANALYSIS")
print("(When does max bounce happen vs max correction?)")
print("=" * 80)

print("\n### Bounce Timing Analysis")
print("\n| EMA | H | Count | Bounce First | Correction First | Avg Bounce Bar | Avg Correction Bar |")
print("|-----|---|-------|--------------|------------------|----------------|-------------------|")

for ema_period in [20, 50, 100]:
    for H in [10, 30, 60]:
        result = analyze_path_sequence(ema_period, H)
        if result:
            print(f"| EMA{ema_period} | {H} | {result['total']:,} | "
                  f"{result['bounce_first_pct']:.1f}% | {result['correction_first_pct']:.1f}% | "
                  f"{result['avg_bounce_bar']:.1f} | {result['avg_correction_bar']:.1f} |")


# =============================================================================
# PART 2: P&L BACKTEST
# =============================================================================
def backtest_ema_bounce(ema_period, H, target_bps, stop_bps):
    """
    Backtest EMA bounce strategy with specific target and stop.

    Entry: At EMA touch (confirmed bounce)
    Exit: Target hit, Stop hit, or Timeout (H bars)
    """
    ema = emas[ema_period]
    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    confirm_pct = BOUNCE_CONFIRM_BPS / 10000
    target_pct = target_bps / 10000
    stop_pct = stop_bps / 10000

    trades = 0
    wins = 0
    losses = 0
    timeouts = 0
    total_pnl_bps = 0

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

        # Check if touching EMA
        distance_pct = (current_close - current_ema) / current_ema
        if abs(distance_pct) > touch_pct:
            continue

        # Check approach direction (from above = potential bounce)
        approach_positions = []
        for j in range(1, APPROACH_BARS + 1):
            if i - j >= 0:
                rel_pos = (close[i - j] - ema[i - j]) / ema[i - j]
                approach_positions.append(rel_pos)

        if not approach_positions:
            continue

        avg_approach = np.mean(approach_positions)

        if avg_approach <= touch_pct:
            continue  # Not coming from above

        # Check if bounce confirmed
        bounce_confirmed = False
        for j in range(1, 11):
            if i + j >= n:
                break
            future_rel = (close[i + j] - ema[i + j]) / ema[i + j]
            if future_rel > confirm_pct:
                bounce_confirmed = True
                break
            elif future_rel < -confirm_pct:
                break

        if not bounce_confirmed:
            continue

        # TRADE: Enter LONG at current close
        trades += 1
        entry = current_close
        target_price = entry * (1 + target_pct)
        stop_price = entry * (1 - stop_pct)

        # Simulate trade
        outcome = 'timeout'
        exit_pnl = 0

        for j in range(1, H + 1):
            if i + j >= n:
                break

            # Check stop first (conservative)
            if low[i + j] <= stop_price:
                outcome = 'stop'
                exit_pnl = -stop_bps - FEE_BPS
                break

            # Check target
            if high[i + j] >= target_price:
                outcome = 'target'
                exit_pnl = target_bps - FEE_BPS
                break

        if outcome == 'timeout':
            # Exit at close of last bar
            exit_price = close[i + H] if i + H < n else close[-1]
            exit_pnl = (exit_price - entry) / entry * 10000 - FEE_BPS

        total_pnl_bps += exit_pnl

        if outcome == 'target':
            wins += 1
        elif outcome == 'stop':
            losses += 1
        else:
            timeouts += 1

    if trades == 0:
        return None

    return {
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'timeouts': timeouts,
        'win_rate': wins / trades * 100,
        'total_pnl_bps': total_pnl_bps,
        'avg_pnl_bps': total_pnl_bps / trades,
        'pnl_per_1000_trades': total_pnl_bps / trades * 1000
    }


print("\n" + "=" * 80)
print("PART 2: P&L BACKTEST")
print("(Actual profitability with target/stop/fees)")
print("=" * 80)

print(f"\nFees: {FEE_BPS}bp round-trip")

# Test key combinations
print("\n### EMA50 Backtest Results")
print("\n| Target | Stop | H | Trades | Wins | Losses | Win% | Total P&L | Avg P&L |")
print("|--------|------|---|--------|------|--------|------|-----------|---------|")

for target in [15, 20, 25]:
    for stop in [10, 15, 20]:
        for H in [30, 60]:
            result = backtest_ema_bounce(50, H, target, stop)
            if result:
                status = "PROFIT" if result['total_pnl_bps'] > 0 else "LOSS"
                print(f"| {target}bp | {stop}bp | {H} | {result['trades']:,} | "
                      f"{result['wins']:,} | {result['losses']:,} | {result['win_rate']:.1f}% | "
                      f"{result['total_pnl_bps']:.0f}bp | {result['avg_pnl_bps']:.2f}bp | {status}")


print("\n### Comparison Across EMAs (Target=25bp, Stop=15bp, H=30)")
print("\n| EMA | Trades | Win% | Total P&L | Avg P&L | Status |")
print("|-----|--------|------|-----------|---------|--------|")

for ema_period in EMA_PERIODS:
    result = backtest_ema_bounce(ema_period, 30, 25, 15)
    if result:
        status = "PROFIT" if result['total_pnl_bps'] > 0 else "LOSS"
        print(f"| EMA{ema_period} | {result['trades']:,} | {result['win_rate']:.1f}% | "
              f"{result['total_pnl_bps']:.0f}bp | {result['avg_pnl_bps']:.2f}bp | {status} |")


# =============================================================================
# PART 3: FIND BEST PARAMETERS
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: PARAMETER OPTIMIZATION")
print("(Find best Target/Stop combinations)")
print("=" * 80)

best_results = []

for ema_period in [20, 50, 100]:
    for target in TARGETS_BPS:
        for stop in STOPS_BPS:
            for H in [30, 60]:
                result = backtest_ema_bounce(ema_period, H, target, stop)
                if result and result['trades'] >= 500:
                    best_results.append({
                        'ema': ema_period,
                        'target': target,
                        'stop': stop,
                        'H': H,
                        **result
                    })

# Sort by total P&L
best_results.sort(key=lambda x: x['total_pnl_bps'], reverse=True)

print("\n### Top 10 Most Profitable Combinations")
print("\n| Rank | EMA | Target | Stop | H | Trades | Win% | Total P&L | Avg P&L |")
print("|------|-----|--------|------|---|--------|------|-----------|---------|")

for i, r in enumerate(best_results[:10]):
    print(f"| {i+1} | EMA{r['ema']} | {r['target']}bp | {r['stop']}bp | {r['H']} | "
          f"{r['trades']:,} | {r['win_rate']:.1f}% | {r['total_pnl_bps']:.0f}bp | {r['avg_pnl_bps']:.2f}bp |")


print("\n### Bottom 5 (Worst Combinations)")
print("\n| Rank | EMA | Target | Stop | H | Trades | Win% | Total P&L | Avg P&L |")
print("|------|-----|--------|------|---|--------|------|-----------|---------|")

for i, r in enumerate(best_results[-5:]):
    print(f"| {len(best_results)-4+i} | EMA{r['ema']} | {r['target']}bp | {r['stop']}bp | {r['H']} | "
          f"{r['trades']:,} | {r['win_rate']:.1f}% | {r['total_pnl_bps']:.0f}bp | {r['avg_pnl_bps']:.2f}bp |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

profitable = [r for r in best_results if r['total_pnl_bps'] > 0]
unprofitable = [r for r in best_results if r['total_pnl_bps'] <= 0]

print(f"\n1. Profitable combinations: {len(profitable)} / {len(best_results)}")
print(f"2. Unprofitable combinations: {len(unprofitable)} / {len(best_results)}")

if profitable:
    print(f"\n3. Best combination:")
    best = profitable[0]
    print(f"   EMA{best['ema']}, Target={best['target']}bp, Stop={best['stop']}bp, H={best['H']}")
    print(f"   Win rate: {best['win_rate']:.1f}%")
    print(f"   Avg P&L per trade: {best['avg_pnl_bps']:.2f}bp")
    print(f"   Total P&L: {best['total_pnl_bps']:.0f}bp over {best['trades']:,} trades")

print("\n4. Path sequence summary:")
seq_result = analyze_path_sequence(50, 30)
if seq_result:
    print(f"   Bounce happens FIRST: {seq_result['bounce_first_pct']:.1f}% of trades")
    print(f"   Correction happens FIRST: {seq_result['correction_first_pct']:.1f}% of trades")
    print(f"   Avg bounce bar: {seq_result['avg_bounce_bar']:.1f}")
    print(f"   Avg correction bar: {seq_result['avg_correction_bar']:.1f}")
