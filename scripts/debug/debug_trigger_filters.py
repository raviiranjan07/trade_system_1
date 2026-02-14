"""
Test micro-triggers with advanced filters.

Run: .venv\Scripts\python.exe debug_trigger_filters.py

This tests the 4 filters suggested to improve win rate:
1. Expansion Acceleration - only trade when expansion is increasing
2. Location Filter - inside bar must be at range extremes
3. Direction Agreement - trade only in direction of prior impulse
4. Cooldown - only 1 trade per expansion burst
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from trade_system.triggers import MicroTriggerDetector, TriggerConfig
from trade_system.expansion import ExpansionLabeler, ExpansionConfig

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZON = 30  # Use H=30 (profitable horizon)
INVALIDATION_RATIO = 0.5
ROUND_TRIP_FEE_BPS = 8

# Filter parameters
RANGE_LOOKBACK = 20  # Bars to compute range for location filter
LOCATION_THRESHOLD = 0.25  # Top/bottom 25% of range
IMPULSE_LOOKBACK = 5  # Bars to determine prior impulse direction
COOLDOWN_BARS = 5  # Minimum bars between trades in same direction

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("ADVANCED FILTER TESTING")
print("=" * 70)
print("\nLoading data...")

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"OHLCV: {len(ohlcv):,} candles")

# =============================================================================
# STEP 1: DETECT TRIGGERS (Inside Bar only - best performer)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: DETECTING INSIDE BAR TRIGGERS")
print("=" * 70)

config = TriggerConfig(min_candle_range_bps=5.0)
detector = MicroTriggerDetector(ohlcv, config)
triggers_df = detector.detect_all(show_progress=False)

print(f"Inside Bar LONG:  {triggers_df['trigger_insidebar_long'].sum():,}")
print(f"Inside Bar SHORT: {triggers_df['trigger_insidebar_short'].sum():,}")

# =============================================================================
# STEP 2: COMPUTE EXPANSION LABELS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: COMPUTING EXPANSION LABELS")
print("=" * 70)

close = ohlcv['close'].values
high = ohlcv['high'].values
low = ohlcv['low'].values
n = len(ohlcv)

print(f"Computing move distribution for H={HORIZON}...")
moves = []
for i in range(n - HORIZON):
    entry = close[i]
    future_high = np.max(high[i+1:i+1+HORIZON])
    future_low = np.min(low[i+1:i+1+HORIZON])
    moves.append((future_high - entry) / entry)
    moves.append((entry - future_low) / entry)
moves = np.array(moves)

median_move = np.percentile(moves, 50)
target_pct = median_move
invalidation_pct = median_move * INVALIDATION_RATIO

print(f"Target (median): {target_pct*10000:.1f} bps")
print(f"Invalidation: {invalidation_pct*10000:.1f} bps")

exp_config = ExpansionConfig(
    horizon=HORIZON,
    target_pct=target_pct,
    invalidation_pct=invalidation_pct,
)
labeler = ExpansionLabeler(ohlcv)
expansion_df = labeler.label(exp_config, show_progress=True)

long_exp_col = f'long_expansion_{HORIZON}m'
short_exp_col = f'short_expansion_{HORIZON}m'

# =============================================================================
# STEP 3: COMPUTE ADVANCED FILTERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: COMPUTING ADVANCED FILTERS")
print("=" * 70)

# --- Filter 1: Expansion Acceleration ---
print("Computing expansion acceleration...")
# Expansion is "accelerating" if current expansion > previous
# We use a simple proxy: check if expansion changed from 0 to 1 recently
expansion_df['long_exp_prev'] = expansion_df[long_exp_col].shift(1)
expansion_df['short_exp_prev'] = expansion_df[short_exp_col].shift(1)

# Acceleration = expansion started (was 0, now 1) or continuing (was 1, now 1)
# We want "fresh" expansion, not stale
expansion_df['long_exp_fresh'] = (
    (expansion_df[long_exp_col] == 1) &
    (expansion_df['long_exp_prev'] == 0)
).astype(int)
expansion_df['short_exp_fresh'] = (
    (expansion_df[short_exp_col] == 1) &
    (expansion_df['short_exp_prev'] == 0)
).astype(int)

# Alternative: rolling sum of expansion over last N bars (lower = fresher)
expansion_df['long_exp_age'] = expansion_df[long_exp_col].rolling(5).sum().shift(1).fillna(5)
expansion_df['short_exp_age'] = expansion_df[short_exp_col].rolling(5).sum().shift(1).fillna(5)

# Fresh = age < 2 (expansion just started)
expansion_df['long_exp_accel'] = (expansion_df['long_exp_age'] < 2).astype(int)
expansion_df['short_exp_accel'] = (expansion_df['short_exp_age'] < 2).astype(int)

print(f"  Long acceleration signals: {expansion_df['long_exp_accel'].sum():,}")
print(f"  Short acceleration signals: {expansion_df['short_exp_accel'].sum():,}")

# --- Filter 2: Location Filter ---
print("Computing location filter...")
# Inside bar must be in top 25% (for long) or bottom 25% (for short) of recent range
ohlcv['rolling_high'] = ohlcv['high'].rolling(RANGE_LOOKBACK).max()
ohlcv['rolling_low'] = ohlcv['low'].rolling(RANGE_LOOKBACK).min()
ohlcv['rolling_range'] = ohlcv['rolling_high'] - ohlcv['rolling_low']

# Position in range (0 = at bottom, 1 = at top)
ohlcv['range_position'] = (ohlcv['close'] - ohlcv['rolling_low']) / ohlcv['rolling_range'].replace(0, np.nan)

# Long: price at bottom of range (good for bounce)
# Short: price at top of range (good for reversal)
ohlcv['location_long'] = (ohlcv['range_position'] <= LOCATION_THRESHOLD).astype(int)
ohlcv['location_short'] = (ohlcv['range_position'] >= (1 - LOCATION_THRESHOLD)).astype(int)

print(f"  Long location signals (bottom 25%): {ohlcv['location_long'].sum():,}")
print(f"  Short location signals (top 25%): {ohlcv['location_short'].sum():,}")

# --- Filter 3: Direction Agreement ---
print("Computing direction agreement...")
# Prior impulse = direction of last N bars
ohlcv['impulse'] = ohlcv['close'] - ohlcv['close'].shift(IMPULSE_LOOKBACK)
ohlcv['direction_long'] = (ohlcv['impulse'] > 0).astype(int)  # Recent move was up
ohlcv['direction_short'] = (ohlcv['impulse'] < 0).astype(int)  # Recent move was down

print(f"  Long direction (bullish impulse): {ohlcv['direction_long'].sum():,}")
print(f"  Short direction (bearish impulse): {ohlcv['direction_short'].sum():,}")

# --- Filter 4: Cooldown ---
print("Computing cooldown filter...")
# Only trade if no trade in last N bars
# We'll implement this as: expansion must have been 0 at some point in last N bars
expansion_df['long_had_reset'] = (
    expansion_df[long_exp_col].rolling(COOLDOWN_BARS).min() == 0
).astype(int)
expansion_df['short_had_reset'] = (
    expansion_df[short_exp_col].rolling(COOLDOWN_BARS).min() == 0
).astype(int)

print(f"  Long cooldown OK: {expansion_df['long_had_reset'].sum():,}")
print(f"  Short cooldown OK: {expansion_df['short_had_reset'].sum():,}")

# =============================================================================
# STEP 4: MERGE ALL DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: MERGING DATA")
print("=" * 70)

# Forward returns
ohlcv['fwd_return'] = (ohlcv['close'].shift(-HORIZON) - ohlcv['close']) / ohlcv['close']

# Merge everything
common_idx = triggers_df.index.intersection(expansion_df.index).intersection(ohlcv.index)
common_idx = common_idx[RANGE_LOOKBACK:-HORIZON]  # Skip edges

merged = pd.DataFrame(index=common_idx)
merged['fwd_return'] = ohlcv.loc[common_idx, 'fwd_return']

# Triggers
merged['trigger_long'] = triggers_df.loc[common_idx, 'trigger_insidebar_long']
merged['trigger_short'] = triggers_df.loc[common_idx, 'trigger_insidebar_short']

# Expansion
merged['expansion_long'] = expansion_df.loc[common_idx, long_exp_col]
merged['expansion_short'] = expansion_df.loc[common_idx, short_exp_col]

# Filters
merged['accel_long'] = expansion_df.loc[common_idx, 'long_exp_accel']
merged['accel_short'] = expansion_df.loc[common_idx, 'short_exp_accel']
merged['location_long'] = ohlcv.loc[common_idx, 'location_long']
merged['location_short'] = ohlcv.loc[common_idx, 'location_short']
merged['direction_long'] = ohlcv.loc[common_idx, 'direction_long']
merged['direction_short'] = ohlcv.loc[common_idx, 'direction_short']
merged['cooldown_long'] = expansion_df.loc[common_idx, 'long_had_reset']
merged['cooldown_short'] = expansion_df.loc[common_idx, 'short_had_reset']

print(f"Merged data: {len(merged):,} rows")

# =============================================================================
# STEP 5: TEST EACH FILTER INCREMENTALLY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: INCREMENTAL FILTER TESTING (Inside Bar)")
print("=" * 70)

fee_pct = ROUND_TRIP_FEE_BPS / 10000

def compute_stats(df, mask_long, mask_short, name):
    """Compute win rate for given filter masks."""
    long_signals = df[mask_long]
    short_signals = df[mask_short]

    long_wins = (long_signals['fwd_return'] > fee_pct).sum() if len(long_signals) > 0 else 0
    short_wins = (short_signals['fwd_return'] < -fee_pct).sum() if len(short_signals) > 0 else 0

    long_rate = long_wins / len(long_signals) if len(long_signals) > 0 else 0
    short_rate = short_wins / len(short_signals) if len(short_signals) > 0 else 0

    total_signals = len(long_signals) + len(short_signals)
    total_wins = long_wins + short_wins
    combined_rate = total_wins / total_signals if total_signals > 0 else 0

    return {
        'name': name,
        'long_signals': len(long_signals),
        'short_signals': len(short_signals),
        'long_rate': long_rate,
        'short_rate': short_rate,
        'combined_rate': combined_rate,
        'total_signals': total_signals,
    }

results = []

# Baseline: Trigger only
mask_long = merged['trigger_long'] == 1
mask_short = merged['trigger_short'] == 1
results.append(compute_stats(merged, mask_long, mask_short, "1. Trigger Only"))

# + Expansion
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1)
results.append(compute_stats(merged, mask_long, mask_short, "2. + Expansion"))

# + Acceleration
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['accel_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['accel_short'] == 1)
results.append(compute_stats(merged, mask_long, mask_short, "3. + Acceleration"))

# + Location
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['accel_long'] == 1) & (merged['location_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['accel_short'] == 1) & (merged['location_short'] == 1)
results.append(compute_stats(merged, mask_long, mask_short, "4. + Location"))

# + Direction
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['accel_long'] == 1) & (merged['location_long'] == 1) & (merged['direction_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['accel_short'] == 1) & (merged['location_short'] == 1) & (merged['direction_short'] == 1)
results.append(compute_stats(merged, mask_long, mask_short, "5. + Direction"))

# + Cooldown
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['accel_long'] == 1) & (merged['location_long'] == 1) & (merged['direction_long'] == 1) & (merged['cooldown_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['accel_short'] == 1) & (merged['location_short'] == 1) & (merged['direction_short'] == 1) & (merged['cooldown_short'] == 1)
results.append(compute_stats(merged, mask_long, mask_short, "6. + Cooldown (ALL)"))

# Print results
print(f"\n{'Filter Stack':<25} {'Long':>8} {'Short':>8} {'Total':>10} {'Long WR':>10} {'Short WR':>10} {'Combined':>10}")
print("-" * 85)

for r in results:
    print(f"{r['name']:<25} {r['long_signals']:>8,} {r['short_signals']:>8,} {r['total_signals']:>10,} {r['long_rate']*100:>9.1f}% {r['short_rate']*100:>9.1f}% {r['combined_rate']*100:>9.1f}%")

# =============================================================================
# STEP 6: PROFITABILITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: PROFITABILITY ANALYSIS")
print("=" * 70)

target_bps = target_pct * 10000
stop_bps = target_bps / 2
fee_bps = ROUND_TRIP_FEE_BPS

print(f"\nAssumptions:")
print(f"  Target: {target_bps:.1f} bps")
print(f"  Stop: {stop_bps:.1f} bps")
print(f"  Fees: {fee_bps} bps")

print(f"\n{'Filter Stack':<25} {'Win Rate':>10} {'EV/Trade':>12} {'Profitable?':>12}")
print("-" * 62)

for r in results:
    win_rate = r['combined_rate']
    win_payout = target_bps - fee_bps
    lose_payout = -stop_bps - fee_bps
    ev = win_rate * win_payout + (1 - win_rate) * lose_payout

    profitable = "YES" if ev > 0 else "NO"
    print(f"{r['name']:<25} {win_rate*100:>9.1f}% {ev:>+11.1f} bps {profitable:>12}")

# =============================================================================
# STEP 7: ALTERNATIVE FILTER COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: ALTERNATIVE FILTER COMBINATIONS")
print("=" * 70)

alt_results = []

# Expansion + Direction only (skip location/accel)
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['direction_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['direction_short'] == 1)
alt_results.append(compute_stats(merged, mask_long, mask_short, "Exp + Direction"))

# Expansion + Location only
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['location_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['location_short'] == 1)
alt_results.append(compute_stats(merged, mask_long, mask_short, "Exp + Location"))

# Expansion + Cooldown only
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['cooldown_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['cooldown_short'] == 1)
alt_results.append(compute_stats(merged, mask_long, mask_short, "Exp + Cooldown"))

# Expansion + Acceleration only
mask_long = (merged['trigger_long'] == 1) & (merged['expansion_long'] == 1) & (merged['accel_long'] == 1)
mask_short = (merged['trigger_short'] == 1) & (merged['expansion_short'] == 1) & (merged['accel_short'] == 1)
alt_results.append(compute_stats(merged, mask_long, mask_short, "Exp + Acceleration"))

print(f"\n{'Combination':<25} {'Signals':>10} {'Win Rate':>10} {'EV/Trade':>12}")
print("-" * 60)

for r in alt_results:
    win_rate = r['combined_rate']
    win_payout = target_bps - fee_bps
    lose_payout = -stop_bps - fee_bps
    ev = win_rate * win_payout + (1 - win_rate) * lose_payout

    print(f"{r['name']:<25} {r['total_signals']:>10,} {win_rate*100:>9.1f}% {ev:>+11.1f} bps")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

baseline = results[0]['combined_rate']
with_expansion = results[1]['combined_rate']
all_filters = results[-1]['combined_rate']

print(f"""
INSIDE BAR TRIGGER RESULTS:

  Trigger Only:           {baseline*100:.1f}% win rate
  + Expansion Filter:     {with_expansion*100:.1f}% win rate (+{(with_expansion-baseline)*100:.1f}pp)
  + All 4 Filters:        {all_filters*100:.1f}% win rate (+{(all_filters-baseline)*100:.1f}pp from baseline)

FILTER IMPACT:
  Expansion alone adds:   +{(with_expansion-baseline)*100:.1f} percentage points
  Additional filters add: +{(all_filters-with_expansion)*100:.1f} percentage points

SIGNALS REMAINING:
  After all filters: {results[-1]['total_signals']:,} signals (from {results[0]['total_signals']:,} original)
  Reduction: {(1 - results[-1]['total_signals']/results[0]['total_signals'])*100:.1f}%
""")
