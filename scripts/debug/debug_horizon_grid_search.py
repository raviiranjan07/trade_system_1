"""
Data-Driven Threshold Discovery: Multi-Horizon Grid Search

Run: .venv/Scripts/python.exe debug_horizon_grid_search.py

NO ARBITRARY THRESHOLDS - Everything comes from data.

Cost Reality:
- Fees only (limit orders): 8 bps round-trip
- MWNM = 8 bps (minimum target)

Questions to answer:
1. What moves does the market actually provide at each horizon?
2. Is there ANY profitable (H, Target, Stop) combination after 8 bps fees?
3. What is the optimal horizon for this cost structure?
4. Does the edge hold out-of-sample?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
FEES_BPS = 8  # Round-trip fees (limit orders)

# Grid search parameters
HORIZONS = [3, 5, 10, 15, 30, 60]
TARGETS = [8, 10, 12, 15, 20, 25, 30, 40, 50]  # In bps
STOPS = [5, 8, 10, 15, 20, 25, 30, 40]  # In bps

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

SAMPLE_SIZE = 200000  # Sample for speed

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("DATA-DRIVEN THRESHOLD DISCOVERY")
print("=" * 70)
print(f"\nFees: {FEES_BPS} bps (round-trip)")
print(f"Minimum Target: {FEES_BPS} bps (to cover fees)")

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Split data
train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
test_ohlcv = ohlcv[ohlcv.index >= TEST_START]
print(f"TRAIN: {len(train_ohlcv):,} | TEST: {len(test_ohlcv):,}")

# =============================================================================
# PART 1: MOVE DISTRIBUTION BY HORIZON
# =============================================================================
print("\n" + "=" * 70)
print("PART 1: MOVE DISTRIBUTION BY HORIZON")
print("=" * 70)

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
max_horizon = max(HORIZONS)
sample_idx = np.random.choice(n - max_horizon, size=min(SAMPLE_SIZE, n - max_horizon), replace=False)

print(f"\nSampling {len(sample_idx):,} bars for analysis...")

print(f"\n{'Horizon':<10} {'10th':>10} {'25th':>10} {'50th':>10} {'75th':>10} {'90th':>10} {'95th':>10}")
print("-" * 72)

horizon_stats = {}

for H in HORIZONS:
    max_moves = []

    for i in sample_idx:
        entry = close[i]
        future_highs = high[i+1:i+1+H]
        future_lows = low[i+1:i+1+H]

        max_up = (np.max(future_highs) - entry) / entry * 10000  # bps
        max_down = (entry - np.min(future_lows)) / entry * 10000  # bps
        max_move = max(max_up, max_down)
        max_moves.append(max_move)

    max_moves = np.array(max_moves)

    p10 = np.percentile(max_moves, 10)
    p25 = np.percentile(max_moves, 25)
    p50 = np.percentile(max_moves, 50)
    p75 = np.percentile(max_moves, 75)
    p90 = np.percentile(max_moves, 90)
    p95 = np.percentile(max_moves, 95)

    horizon_stats[H] = {
        'p10': p10, 'p25': p25, 'p50': p50,
        'p75': p75, 'p90': p90, 'p95': p95
    }

    print(f"H={H:<7} {p10:>9.1f} {p25:>10.1f} {p50:>10.1f} {p75:>10.1f} {p90:>10.1f} {p95:>10.1f}")

# Show % of bars that can cover fees at each horizon
print(f"\n{'Horizon':<10} {'% >= 8bps':>12} {'% >= 15bps':>12} {'% >= 25bps':>12}")
print("-" * 50)

for H in HORIZONS:
    max_moves = []
    for i in sample_idx:
        entry = close[i]
        future_highs = high[i+1:i+1+H]
        future_lows = low[i+1:i+1+H]
        max_up = (np.max(future_highs) - entry) / entry * 10000
        max_down = (entry - np.min(future_lows)) / entry * 10000
        max_moves.append(max(max_up, max_down))

    max_moves = np.array(max_moves)
    pct_8 = (max_moves >= 8).mean() * 100
    pct_15 = (max_moves >= 15).mean() * 100
    pct_25 = (max_moves >= 25).mean() * 100

    print(f"H={H:<7} {pct_8:>11.1f}% {pct_15:>11.1f}% {pct_25:>11.1f}%")

# =============================================================================
# PART 2: GRID SEARCH FOR PROFITABLE COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: GRID SEARCH (ALL COMBINATIONS)")
print("=" * 70)

def compute_outcomes(close, high, low, indices, horizon, target_bps, stop_bps):
    """Compute WIN/LOSS/TIMEOUT for each bar."""
    target_pct = target_bps / 10000
    stop_pct = stop_bps / 10000

    wins = 0
    losses = 0
    timeouts = 0

    for i in indices:
        entry = close[i]
        target_price = entry * (1 + target_pct)
        stop_price = entry * (1 - stop_pct)

        outcome = 'TIMEOUT'
        for j in range(i+1, min(i+1+horizon, len(close))):
            if low[j] <= stop_price:
                outcome = 'LOSS'
                break
            if high[j] >= target_price:
                outcome = 'WIN'
                break

        if outcome == 'WIN':
            wins += 1
        elif outcome == 'LOSS':
            losses += 1
        else:
            timeouts += 1

    return wins, losses, timeouts


print(f"\nSearching {len(HORIZONS)} horizons x {len(TARGETS)} targets x {len(STOPS)} stops = {len(HORIZONS)*len(TARGETS)*len(STOPS)} combinations...")

results = []

for H in HORIZONS:
    print(f"  Testing H={H}...")

    # Re-sample for this horizon
    sample_idx_h = np.random.choice(n - H, size=min(SAMPLE_SIZE, n - H), replace=False)

    for target in TARGETS:
        if target < FEES_BPS:
            continue  # Target must cover fees

        for stop in STOPS:
            wins, losses, timeouts = compute_outcomes(
                close, high, low, sample_idx_h, H, target, stop
            )

            total = wins + losses
            if total < 100:
                continue  # Not enough data

            win_rate = wins / total * 100
            breakeven = stop / (target + stop) * 100
            edge = win_rate - breakeven

            # Expected value per trade (in bps)
            # EV = WR * (target - fees) - (1-WR) * (stop + fees)
            # Simplified: EV = WR * target - (1-WR) * stop - fees
            ev_per_trade = (win_rate/100) * target - (1 - win_rate/100) * stop - FEES_BPS

            # Resolved rate (what % of trades don't timeout)
            resolved_rate = total / len(sample_idx_h) * 100

            results.append({
                'H': H,
                'target': target,
                'stop': stop,
                'wins': wins,
                'losses': losses,
                'timeouts': timeouts,
                'total': total,
                'win_rate': win_rate,
                'breakeven': breakeven,
                'edge': edge,
                'ev_per_trade': ev_per_trade,
                'resolved_rate': resolved_rate
            })

results_df = pd.DataFrame(results)

# Filter to positive EV after fees
positive_ev = results_df[results_df['ev_per_trade'] > 0].sort_values('ev_per_trade', ascending=False)

print(f"\n{'='*70}")
print(f"RESULTS: {len(positive_ev)} combinations with POSITIVE EV after {FEES_BPS}bps fees")
print(f"{'='*70}")

if len(positive_ev) > 0:
    print(f"\n{'H':>4} {'Tgt':>5} {'Stp':>5} {'WinRate':>9} {'B/E':>7} {'Edge':>8} {'EV/Trade':>10} {'Trades':>10}")
    print("-" * 75)

    for _, row in positive_ev.head(20).iterrows():
        print(f"{row['H']:>4} {row['target']:>5} {row['stop']:>5} {row['win_rate']:>8.1f}% {row['breakeven']:>6.1f}% {row['edge']:>+7.1f}pp {row['ev_per_trade']:>+9.2f}bp {row['total']:>10,}")
else:
    print("\n>>> NO COMBINATIONS have positive EV after fees <<<")

    # Show best combinations anyway
    print("\nBest combinations (still negative EV):")
    best = results_df.nlargest(10, 'ev_per_trade')
    print(f"\n{'H':>4} {'Tgt':>5} {'Stp':>5} {'WinRate':>9} {'B/E':>7} {'Edge':>8} {'EV/Trade':>10} {'Trades':>10}")
    print("-" * 75)
    for _, row in best.iterrows():
        print(f"{row['H']:>4} {row['target']:>5} {row['stop']:>5} {row['win_rate']:>8.1f}% {row['breakeven']:>6.1f}% {row['edge']:>+7.1f}pp {row['ev_per_trade']:>+9.2f}bp {row['total']:>10,}")

# =============================================================================
# PART 3: VALIDATE ON TEST DATA
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: VALIDATION ON TEST DATA")
print("=" * 70)

if len(positive_ev) > 0:
    # Take top 3 combinations
    top_combos = positive_ev.head(3)

    test_close = test_ohlcv['close'].values
    test_high = test_ohlcv['high'].values
    test_low = test_ohlcv['low'].values
    n_test = len(test_ohlcv)

    print(f"\nValidating top {len(top_combos)} combinations on TEST data ({len(test_ohlcv):,} bars)...")
    print(f"\n{'H':>4} {'Tgt':>5} {'Stp':>5} {'Train WR':>10} {'Test WR':>10} {'Train EV':>10} {'Test EV':>10} {'Status':>10}")
    print("-" * 80)

    for _, row in top_combos.iterrows():
        H = int(row['H'])
        target = int(row['target'])
        stop = int(row['stop'])

        # Test on all test data
        test_idx = np.arange(n_test - H)
        wins, losses, timeouts = compute_outcomes(
            test_close, test_high, test_low, test_idx, H, target, stop
        )

        total = wins + losses
        if total > 0:
            test_wr = wins / total * 100
            test_ev = (test_wr/100) * target - (1 - test_wr/100) * stop - FEES_BPS
        else:
            test_wr = 0
            test_ev = 0

        status = "HOLDS" if test_ev > 0 else "FAILS"

        print(f"{H:>4} {target:>5} {stop:>5} {row['win_rate']:>9.1f}% {test_wr:>9.1f}% {row['ev_per_trade']:>+9.2f}bp {test_ev:>+9.2f}bp {status:>10}")

else:
    print("\nNo positive EV combinations to validate.")

    # Test best negative EV combination anyway
    best = results_df.nlargest(1, 'ev_per_trade').iloc[0]
    H = int(best['H'])
    target = int(best['target'])
    stop = int(best['stop'])

    test_close = test_ohlcv['close'].values
    test_high = test_ohlcv['high'].values
    test_low = test_ohlcv['low'].values
    n_test = len(test_ohlcv)

    test_idx = np.arange(n_test - H)
    wins, losses, timeouts = compute_outcomes(
        test_close, test_high, test_low, test_idx, H, target, stop
    )

    total = wins + losses
    test_wr = wins / total * 100 if total > 0 else 0
    test_ev = (test_wr/100) * target - (1 - test_wr/100) * stop - FEES_BPS

    print(f"\nBest combination (negative EV):")
    print(f"  H={H}, Target={target}bps, Stop={stop}bps")
    print(f"  Train WR: {best['win_rate']:.1f}%, Test WR: {test_wr:.1f}%")
    print(f"  Train EV: {best['ev_per_trade']:+.2f}bp, Test EV: {test_ev:+.2f}bp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
ANALYSIS PARAMETERS:
- Fees: {FEES_BPS} bps (round-trip, limit orders)
- Horizons tested: {HORIZONS}
- Targets tested: {TARGETS}
- Stops tested: {STOPS}
- Total combinations: {len(results)}

KEY FINDINGS:
""")

if len(positive_ev) > 0:
    best = positive_ev.iloc[0]
    print(f"  POSITIVE EV combinations found: {len(positive_ev)}")
    print(f"  Best combination:")
    print(f"    H = {int(best['H'])} bars")
    print(f"    Target = {int(best['target'])} bps")
    print(f"    Stop = {int(best['stop'])} bps")
    print(f"    Win Rate = {best['win_rate']:.1f}%")
    print(f"    EV per trade = {best['ev_per_trade']:+.2f} bps")
else:
    print(f"  NO positive EV combinations after {FEES_BPS}bps fees")
    print(f"  Random entry is NOT profitable at any (H, Target, Stop)")
    print(f"\n  IMPLICATION:")
    print(f"  You need SELECTIVE entry (not random) to be profitable.")
    print(f"  The state vector / similarity search should provide this selection.")
