"""
Raw Price Path Analysis - No Pre-Set Thresholds

Run: .venv/Scripts/python.exe debug_raw_price_paths.py

APPROACH:
1. Look at what price ACTUALLY does after each bar (no targets/stops)
2. Find natural movement patterns in the data
3. Let the data reveal profitable thresholds (if any exist)

NO ARBITRARY CHOICES - everything comes from data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION - Only horizon (time window to observe)
# =============================================================================
HORIZON = 60  # Observe price for 60 bars after entry
MWNM_BPS = 15  # Minimum worthwhile move (this IS grounded in real costs)

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 100000  # Sample for speed

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("RAW PRICE PATH ANALYSIS")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Filter to TRAIN
train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"TRAIN: {len(train_ohlcv):,} candles")

# =============================================================================
# EXTRACT RAW PRICE PATHS
# =============================================================================
print(f"\nExtracting raw price paths (H={HORIZON})...")

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

# Sample indices for speed
np.random.seed(42)
sample_indices = np.random.choice(n - HORIZON, size=min(SAMPLE_SIZE, n - HORIZON), replace=False)

# For each sampled bar, compute raw metrics
results = []

for idx, i in enumerate(sample_indices):
    if idx % 10000 == 0:
        print(f"  Progress: {idx:,}/{len(sample_indices):,}")

    entry = close[i]
    future_highs = high[i+1:i+1+HORIZON]
    future_lows = low[i+1:i+1+HORIZON]
    future_closes = close[i+1:i+1+HORIZON]

    # Raw metrics - what price actually did
    max_up_abs = np.max(future_highs) - entry  # Max upside in absolute
    max_down_abs = entry - np.min(future_lows)  # Max downside in absolute

    max_up_bps = (max_up_abs / entry) * 10000  # In basis points
    max_down_bps = (max_down_abs / entry) * 10000

    # When did max up/down occur?
    time_to_max_up = np.argmax(future_highs) + 1
    time_to_max_down = np.argmin(future_lows) + 1

    # Which came first - significant up or down move?
    first_up_bar = None
    first_down_bar = None

    for j in range(HORIZON):
        up_move = (future_highs[j] - entry) / entry * 10000
        down_move = (entry - future_lows[j]) / entry * 10000

        if first_up_bar is None and up_move >= MWNM_BPS:
            first_up_bar = j + 1
        if first_down_bar is None and down_move >= MWNM_BPS:
            first_down_bar = j + 1

        if first_up_bar and first_down_bar:
            break

    # First direction (which hit MWNM first)
    if first_up_bar is None and first_down_bar is None:
        first_direction = "NONE"  # Didn't move enough either way
    elif first_up_bar is None:
        first_direction = "DOWN"
    elif first_down_bar is None:
        first_direction = "UP"
    elif first_up_bar < first_down_bar:
        first_direction = "UP"
    elif first_down_bar < first_up_bar:
        first_direction = "DOWN"
    else:
        first_direction = "TIE"

    # Net move at end of horizon
    net_move_bps = (future_closes[-1] - entry) / entry * 10000

    results.append({
        'max_up_bps': max_up_bps,
        'max_down_bps': max_down_bps,
        'time_to_max_up': time_to_max_up,
        'time_to_max_down': time_to_max_down,
        'first_direction': first_direction,
        'first_up_bar': first_up_bar,
        'first_down_bar': first_down_bar,
        'net_move_bps': net_move_bps,
    })

df = pd.DataFrame(results)
print(f"\nAnalyzed {len(df):,} price paths")

# =============================================================================
# PART 1: RAW DISTRIBUTIONS
# =============================================================================
print("\n" + "=" * 70)
print("PART 1: RAW PRICE PATH DISTRIBUTIONS")
print("=" * 70)

print("\n--- MAX UPSIDE (bps) ---")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(df['max_up_bps'], p)
    print(f"  {p}th percentile: {val:.1f} bps")

print("\n--- MAX DOWNSIDE (bps) ---")
for p in percentiles:
    val = np.percentile(df['max_down_bps'], p)
    print(f"  {p}th percentile: {val:.1f} bps")

print("\n--- FIRST DIRECTION (which hit MWNM first) ---")
direction_counts = df['first_direction'].value_counts()
for dir, count in direction_counts.items():
    pct = count / len(df) * 100
    print(f"  {dir}: {count:,} ({pct:.1f}%)")

print("\n--- TIME TO MAX (bars) ---")
print(f"  Time to max UP:   median = {df['time_to_max_up'].median():.0f}, mean = {df['time_to_max_up'].mean():.1f}")
print(f"  Time to max DOWN: median = {df['time_to_max_down'].median():.0f}, mean = {df['time_to_max_down'].mean():.1f}")

# =============================================================================
# PART 2: NATURAL THRESHOLD DISCOVERY
# =============================================================================
print("\n" + "=" * 70)
print("PART 2: GRID SEARCH FOR PROFITABLE TARGET/STOP COMBINATIONS")
print("=" * 70)

print(f"\nTesting all target/stop combinations...")
print(f"MWNM = {MWNM_BPS} bps (minimum target must exceed this)")

# Grid of possible targets and stops
targets = [15, 20, 25, 30, 40, 50, 60, 75, 100]  # In bps
stops = [10, 15, 20, 25, 30, 40, 50]  # In bps

# For each combination, compute win rate
def compute_win_rate(df, target_bps, stop_bps):
    """
    WIN = max_up >= target AND (max_down < stop OR up came first)
    LOSS = max_down >= stop AND down came before up
    """
    wins = 0
    losses = 0

    for _, row in df.iterrows():
        can_hit_target = row['max_up_bps'] >= target_bps
        can_hit_stop = row['max_down_bps'] >= stop_bps

        if not can_hit_target and not can_hit_stop:
            continue  # Timeout - exclude

        if can_hit_target and not can_hit_stop:
            wins += 1
        elif can_hit_stop and not can_hit_target:
            losses += 1
        else:
            # Both can be hit - which comes first?
            # Approximate by first_up_bar vs first_down_bar at their respective thresholds
            # This is a simplification - need to recompute for exact thresholds
            # For now, use the ratio heuristic
            if row['first_direction'] == 'UP':
                wins += 1
            elif row['first_direction'] == 'DOWN':
                losses += 1
            else:
                # Need more precise calculation
                # Assume proportional to which is closer
                if row['max_up_bps'] / target_bps > row['max_down_bps'] / stop_bps:
                    wins += 1
                else:
                    losses += 1

    total = wins + losses
    if total == 0:
        return 0, 0, 0

    win_rate = wins / total * 100
    return win_rate, wins, losses


# More accurate path-based win rate
def compute_win_rate_accurate(close, high, low, indices, target_bps, stop_bps, horizon):
    """Accurately compute win rate by checking actual price paths."""
    wins = 0
    losses = 0
    timeouts = 0

    target_pct = target_bps / 10000
    stop_pct = stop_bps / 10000

    for i in indices:
        entry = close[i]
        target_price = entry * (1 + target_pct)
        stop_price = entry * (1 - stop_pct)

        outcome = 'TIMEOUT'
        for j in range(i+1, i+1+horizon):
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

    total = wins + losses
    if total == 0:
        return 0, 0, 0, timeouts

    win_rate = wins / total * 100
    return win_rate, wins, losses, timeouts


print(f"\n{'Target':>8} {'Stop':>8} {'R:R':>8} {'Win Rate':>10} {'Break-even':>12} {'Edge':>10} {'Trades':>10}")
print("-" * 78)

best_edge = -999
best_combo = None

for target in targets:
    for stop in stops:
        if target < MWNM_BPS:
            continue  # Target must exceed costs

        # Compute accurate win rate
        wr, wins, losses, timeouts = compute_win_rate_accurate(
            close, high, low, sample_indices, target, stop, HORIZON
        )

        if wins + losses < 100:
            continue  # Not enough data

        # R:R ratio
        rr = target / stop

        # Break-even win rate
        breakeven = stop / (target + stop) * 100

        # Edge = actual WR - breakeven
        edge = wr - breakeven

        if edge > best_edge:
            best_edge = edge
            best_combo = (target, stop, wr, breakeven, wins + losses)

        # Only print interesting rows
        if edge > -5 or (target == 50 and stop == 25):
            print(f"{target:>8} {stop:>8} {rr:>8.1f} {wr:>9.1f}% {breakeven:>11.1f}% {edge:>+9.1f}pp {wins+losses:>10,}")

# =============================================================================
# PART 3: BEST COMBINATION FOUND
# =============================================================================
print("\n" + "=" * 70)
print("PART 3: BEST TARGET/STOP COMBINATION FROM DATA")
print("=" * 70)

if best_combo:
    target, stop, wr, be, n_trades = best_combo
    print(f"\nBest combination found:")
    print(f"  Target: {target} bps")
    print(f"  Stop:   {stop} bps")
    print(f"  R:R:    {target/stop:.1f}:1")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Break-even: {be:.1f}%")
    print(f"  Edge: {best_edge:+.1f}pp")
    print(f"  Trades: {n_trades:,}")

    if best_edge > 0:
        print(f"\n  >>> POSITIVE EDGE FOUND! <<<")
    else:
        print(f"\n  >>> NO POSITIVE EDGE at any target/stop combination <<<")
else:
    print("\nNo valid combination found.")

# =============================================================================
# PART 4: WHAT THE DATA TELLS US
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: WHAT THE DATA REVEALS")
print("=" * 70)

# Directional bias
up_first = (df['first_direction'] == 'UP').sum()
down_first = (df['first_direction'] == 'DOWN').sum()
none_dir = (df['first_direction'] == 'NONE').sum()

print(f"\nDirectional Analysis (which direction moves first by {MWNM_BPS} bps):")
print(f"  UP first:   {up_first:,} ({up_first/len(df)*100:.1f}%)")
print(f"  DOWN first: {down_first:,} ({down_first/len(df)*100:.1f}%)")
print(f"  Neither:    {none_dir:,} ({none_dir/len(df)*100:.1f}%)")

if up_first > 0 and down_first > 0:
    ratio = up_first / down_first
    print(f"  UP/DOWN ratio: {ratio:.2f}")

    if ratio > 1.1:
        print(f"  >>> Market has UPWARD bias (go LONG)")
    elif ratio < 0.9:
        print(f"  >>> Market has DOWNWARD bias (go SHORT)")
    else:
        print(f"  >>> Market is roughly BALANCED (no directional edge)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
RAW DATA FINDINGS:
- Max up (median):   {df['max_up_bps'].median():.1f} bps
- Max down (median): {df['max_down_bps'].median():.1f} bps
- First direction:   {df['first_direction'].mode()[0]} (most common)

PROFITABLE THRESHOLDS:
- Best edge found: {best_edge:+.1f}pp
- {'POSITIVE EDGE EXISTS' if best_edge > 0 else 'NO POSITIVE EDGE at any combination'}

IMPLICATION:
""")

if best_edge > 0:
    print(f"  The data reveals a profitable threshold at Target={best_combo[0]}bps, Stop={best_combo[1]}bps")
    print(f"  This is a data-driven discovery, not an arbitrary choice.")
else:
    print(f"  NO target/stop combination produces positive edge on random entry.")
    print(f"  This means: random entry cannot be profitable, regardless of thresholds.")
    print(f"  You need DIRECTIONAL PREDICTION or TIMING to have edge.")
