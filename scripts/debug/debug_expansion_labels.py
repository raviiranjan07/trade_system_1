"""
Debug script to see expansion rates by state/regime.
Run: .venv\Scripts\python.exe debug_expansion_labels.py

This answers: "Which states historically reach ≥X bps in Y bars?"
"""

import pandas as pd
import numpy as np
from pathlib import Path

from trade_system.expansion import ExpansionLabeler, ExpansionConfig

# =============================================================================
# CONFIGURATION - CHANGE THESE TO TEST DIFFERENT SETTINGS
# =============================================================================
HORIZON = 30  # Using 30m (longest available in outcome data)
INVALIDATION_RATIO = 0.5  # 50% of median for invalidation

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")

# Load OHLCV
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    exit(1)
ohlcv = pd.read_parquet(ohlcv_path)

# Load regimes
regime_path = Path("data/regimes/BTCUSDT_1m_regimes.parquet")
if not regime_path.exists():
    print(f"ERROR: Regime file not found: {regime_path}")
    print("Run pipeline with regime_labeling stage first.")
    exit(1)
regime_df = pd.read_parquet(regime_path)

print(f"OHLCV: {len(ohlcv):,} candles")
print(f"Regimes: {len(regime_df):,} rows")
print(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")

# =============================================================================
# STEP 1: COMPUTE THRESHOLD VALUES (like debug_avg_vs_percentile.py)
# =============================================================================
print("\n" + "=" * 65)
print(f"THRESHOLD ANALYSIS FOR H={HORIZON}")
print("=" * 65)

close = ohlcv['close'].values
high = ohlcv['high'].values
low = ohlcv['low'].values
n = len(ohlcv)

# Compute all moves
print(f"\nComputing moves for H={HORIZON}...")
moves = []
for i in range(n - HORIZON):
    entry = close[i]
    future_high = np.max(high[i+1:i+1+HORIZON])
    future_low = np.min(low[i+1:i+1+HORIZON])
    moves.append((future_high - entry) / entry)
    moves.append((entry - future_low) / entry)

moves = np.array(moves)
print(f"Total moves: {len(moves):,}")

# Calculate threshold values
median_move = np.percentile(moves, 50)
avg_move = np.mean(moves)
pct_75 = np.percentile(moves, 75)
pct_90 = np.percentile(moves, 90)

print("\nTHRESHOLD VALUES:")
print(f"  Median (50th pct):  {median_move*10000:.1f} bps")
print(f"  Average (mean):     {avg_move*10000:.1f} bps")
print(f"  75th percentile:    {pct_75*10000:.1f} bps")
print(f"  90th percentile:    {pct_90*10000:.1f} bps")

# Invalidation threshold (same for all - 50% of median)
invalidation = median_move * INVALIDATION_RATIO
print(f"\nINVALIDATION (50% of median): {invalidation*10000:.1f} bps")

# =============================================================================
# STEP 2: EXPANSION LABELING WITH EACH THRESHOLD
# =============================================================================
print("\n" + "=" * 65)
print("EXPANSION RATES (Path-Dependent: Target BEFORE Invalidation)")
print("=" * 65)

labeler = ExpansionLabeler(ohlcv)

# Define thresholds to test
thresholds_to_test = [
    ("Median", median_move),
    ("Average", avg_move),
    ("75th pct", pct_75),
    ("90th pct", pct_90),
]

print(f"\n{'Threshold':<12} {'Target':>10} {'Invalid':>10} {'Long Rate':>12} {'Short Rate':>12}")
print("-" * 60)

results = {}
for name, target in thresholds_to_test:
    config = ExpansionConfig(
        horizon=HORIZON,
        target_pct=target,
        invalidation_pct=invalidation,
    )

    labels = labeler.label(config, show_progress=False)
    long_col = f'long_expansion_{HORIZON}m'
    short_col = f'short_expansion_{HORIZON}m'

    long_rate = labels[long_col].mean() * 100
    short_rate = labels[short_col].mean() * 100

    results[name] = {
        'target': target,
        'long_rate': long_rate,
        'short_rate': short_rate,
        'labels': labels,
    }

    print(f"{name:<12} {target*10000:>8.1f}bp {invalidation*10000:>8.1f}bp {long_rate:>11.1f}% {short_rate:>11.1f}%")

print("-" * 60)

# =============================================================================
# STEP 3: COMPARE RAW MOVES VS PATH-DEPENDENT EXPANSION
# =============================================================================
print("\n" + "=" * 65)
print("RAW MOVES vs PATH-DEPENDENT EXPANSION")
print("=" * 65)

print(f"\n{'Threshold':<12} {'Raw Moves >= Target':>20} {'Path-Dependent Exp':>20}")
print("-" * 55)

for name, target in thresholds_to_test:
    raw_pct = np.sum(moves >= target) / len(moves) * 100
    path_pct = (results[name]['long_rate'] + results[name]['short_rate']) / 2

    print(f"{name:<12} {raw_pct:>18.1f}% {path_pct:>18.1f}%")

print("-" * 55)
print("\nPath-dependent is LOWER because some moves hit invalidation FIRST!")

# =============================================================================
# STEP 4: EXPANSION RATES BY REGIME (for both 75th percentile AND Average)
# =============================================================================

MIN_SAMPLES = 1000
long_col = f'long_expansion_{HORIZON}m'
short_col = f'short_expansion_{HORIZON}m'

for threshold_name in ['75th pct', 'Average']:
    target_val = results[threshold_name]['target']
    labels_df = results[threshold_name]['labels']

    print("\n" + "=" * 65)
    print(f"EXPANSION RATES BY REGIME (H={HORIZON}, {threshold_name} = {target_val*10000:.1f} bps)")
    print("=" * 65)

    # Merge with regimes
    common_idx = labels_df.index.intersection(regime_df.index)
    merged = labels_df.loc[common_idx].copy()
    merged['regime'] = regime_df.loc[common_idx, 'regime']

    # Group by regime
    stats = merged.groupby('regime').agg(
        samples=(long_col, 'count'),
        long_rate=(long_col, 'mean'),
        short_rate=(short_col, 'mean'),
    ).round(4)

    stats = stats.sort_values('long_rate', ascending=False)

    print(f"\n{'Regime':<20} {'Samples':>10} {'Long Rate':>12} {'Short Rate':>12}")
    print("-" * 56)
    for regime, row in stats.iterrows():
        print(f"{regime:<20} {row['samples']:>10,} {row['long_rate']*100:>11.1f}% {row['short_rate']*100:>11.1f}%")
    print("-" * 56)

    # Best states
    significant = stats[stats['samples'] >= MIN_SAMPLES]

    print(f"\nBEST STATES (n >= {MIN_SAMPLES:,}):")
    print("  LONG:  ", end="")
    for regime, row in significant.nlargest(2, 'long_rate').iterrows():
        print(f"{regime} ({row['long_rate']*100:.1f}%)  ", end="")
    print()
    print("  SHORT: ", end="")
    for regime, row in significant.nlargest(2, 'short_rate').iterrows():
        print(f"{regime} ({row['short_rate']*100:.1f}%)  ", end="")
    print()

    # Overall stats
    total_bars = len(merged)
    long_expansions = merged[long_col].sum()
    short_expansions = merged[short_col].sum()
    print(f"\n  Overall: LONG={long_expansions/total_bars*100:.1f}%, SHORT={short_expansions/total_bars*100:.1f}%")

    # Trades per day
    date_range = (merged.index.max() - merged.index.min()).days
    if date_range > 0:
        long_per_day = long_expansions / date_range
        short_per_day = short_expansions / date_range
        total_per_day = (long_expansions + short_expansions) / date_range
        print(f"\n  TRADES PER DAY (if traded every expansion):")
        print(f"    LONG:  {long_per_day:.1f} trades/day")
        print(f"    SHORT: {short_per_day:.1f} trades/day")
        print(f"    TOTAL: {total_per_day:.1f} trades/day")
        print(f"    (over {date_range:,} days)")

# =============================================================================
# INTERPRETATION
# =============================================================================
print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)
print(f"""
For H={HORIZON}:
  - Median = {median_move*10000:.1f} bps -> {results['Median']['long_rate']:.1f}% expansion rate
  - Average = {avg_move*10000:.1f} bps -> {results['Average']['long_rate']:.1f}% expansion rate
  - 75th pct = {pct_75*10000:.1f} bps -> {results['75th pct']['long_rate']:.1f}% expansion rate
  - 90th pct = {pct_90*10000:.1f} bps -> {results['90th pct']['long_rate']:.1f}% expansion rate

KEY INSIGHT:
  Raw moves >= threshold is NOT the same as path-dependent expansion!
  Path-dependent is lower because invalidation can be hit FIRST.

TRADING IMPLICATION:
  - Need expansion_rate > 50% for positive expectancy
  - Current best: {max(results.values(), key=lambda x: x['long_rate'])['long_rate']:.1f}% with Median threshold
  - Even Median threshold doesn't give >50% expansion rate!
  - This suggests: REGIME ALONE is not enough - need SIMILARITY SEARCH
""")
