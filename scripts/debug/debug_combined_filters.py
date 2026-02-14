"""
COMBINED FILTER ANALYSIS
========================
Test if stacking multiple filters can boost expansion rate to profitable levels.

Key insight from bottleneck analysis:
- Individual features correlate weakly with expansion (<0.07)
- Best subgroups: high volume (+4.7pp), low RSI (+4.1pp), high ATR (+2.7pp)
- Need ~58% expansion rate for profitability (vs base ~38.5%)

Goal: Find if combining filters can reach 58%+ expansion rate.

Run: .venv\Scripts\python.exe debug_combined_filters.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZON = 30
INVALIDATION_RATIO = 0.5
MIN_SAMPLES = 100  # Minimum samples for statistical validity

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("COMBINED FILTER ANALYSIS")
print("=" * 70)
print("\nLoading data...")

ohlcv = pd.read_parquet("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
outcome_df = pd.read_parquet("data/outcomes/BTCUSDT_1m_outcomes.parquet")
regime_df = pd.read_parquet("data/regimes/BTCUSDT_1m_regimes.parquet")

print(f"Data: {len(ohlcv):,} candles")

# =============================================================================
# CREATE EXPANSION LABELS
# =============================================================================
print("\nComputing expansion labels...")

from trade_system.expansion import ExpansionLabeler, ExpansionConfig

close = ohlcv['close'].values
high = ohlcv['high'].values
low = ohlcv['low'].values
n = len(ohlcv)

# Compute target from data
moves = []
for i in range(n - HORIZON):
    entry = close[i]
    future_high = np.max(high[i+1:i+1+HORIZON])
    future_low = np.min(low[i+1:i+1+HORIZON])
    moves.append((future_high - entry) / entry)
    moves.append((entry - future_low) / entry)
moves = np.array(moves)

target_pct = np.percentile(moves, 50)
invalidation_pct = target_pct * INVALIDATION_RATIO

print(f"Target: {target_pct*10000:.1f} bps, Invalidation: {invalidation_pct*10000:.1f} bps")

exp_config = ExpansionConfig(
    horizon=HORIZON,
    target_pct=target_pct,
    invalidation_pct=invalidation_pct,
)

labeler = ExpansionLabeler(ohlcv)
expansion_df = labeler.label(exp_config, show_progress=False)

long_col = f'long_expansion_{HORIZON}m'
short_col = f'short_expansion_{HORIZON}m'

# =============================================================================
# MERGE DATA
# =============================================================================
common_idx = outcome_df.index.intersection(regime_df.index).intersection(expansion_df.index)
df = outcome_df.loc[common_idx].copy()
df['regime'] = regime_df.loc[common_idx, 'regime']
df[long_col] = expansion_df.loc[common_idx, long_col]
df[short_col] = expansion_df.loc[common_idx, short_col]

base_long_rate = df[long_col].mean()
base_short_rate = df[short_col].mean()

print(f"\nBase expansion rate: LONG={base_long_rate*100:.1f}%, SHORT={base_short_rate*100:.1f}%")
print(f"Target for profitability: ~58%")

# =============================================================================
# DEFINE FILTERS
# =============================================================================
FILTERS = {
    "high_volume": df['volume_z'] > 1.0,
    "low_rsi": df['rsi_z'] < -1.0,
    "high_rsi": df['rsi_z'] > 1.0,
    "high_atr": df['atr_percentile'] > 0.7,
    "trend_up": df['trend_alignment'] > 0.5,
    "trend_down": df['trend_alignment'] < -0.5,
    "low_range_pos": df['range_position'] < 0.3,
    "high_range_pos": df['range_position'] > 0.7,
    "recent_up": df['return_5m_z'] > 1.0,
    "recent_down": df['return_5m_z'] < -1.0,
    "very_high_volume": df['volume_z'] > 2.0,
    "vwap_below": df['vwap_distance_z'] < -1.0,
    "vwap_above": df['vwap_distance_z'] > 1.0,
}

# =============================================================================
# SINGLE FILTER ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SINGLE FILTER ANALYSIS")
print("=" * 70)
print(f"\n{'Filter':<25} {'Samples':>10} {'LONG Rate':>12} {'SHORT Rate':>12} {'LONG Lift':>10} {'SHORT Lift':>10}")
print("-" * 80)

single_results = []
for name, mask in FILTERS.items():
    n_samples = mask.sum()
    if n_samples >= MIN_SAMPLES:
        long_rate = df.loc[mask, long_col].mean()
        short_rate = df.loc[mask, short_col].mean()
        long_lift = long_rate - base_long_rate
        short_lift = short_rate - base_short_rate

        single_results.append({
            'filter': name,
            'samples': n_samples,
            'long_rate': long_rate,
            'short_rate': short_rate,
            'long_lift': long_lift,
            'short_lift': short_lift,
        })

        print(f"{name:<25} {n_samples:>10,} {long_rate*100:>11.1f}% {short_rate*100:>11.1f}% {long_lift*100:>+9.1f}pp {short_lift*100:>+9.1f}pp")

# =============================================================================
# TWO-FILTER COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("TWO-FILTER COMBINATIONS (sorted by LONG expansion rate)")
print("=" * 70)

two_filter_results = []
filter_names = list(FILTERS.keys())

for f1, f2 in combinations(filter_names, 2):
    combined_mask = FILTERS[f1] & FILTERS[f2]
    n_samples = combined_mask.sum()

    if n_samples >= MIN_SAMPLES:
        long_rate = df.loc[combined_mask, long_col].mean()
        short_rate = df.loc[combined_mask, short_col].mean()

        two_filter_results.append({
            'filters': f"{f1} + {f2}",
            'samples': n_samples,
            'long_rate': long_rate,
            'short_rate': short_rate,
        })

# Sort by long rate descending
two_filter_results = sorted(two_filter_results, key=lambda x: x['long_rate'], reverse=True)

print(f"\n{'Filter Combination':<45} {'Samples':>10} {'LONG Rate':>12} {'SHORT Rate':>12}")
print("-" * 85)

# Show top 15
for r in two_filter_results[:15]:
    print(f"{r['filters']:<45} {r['samples']:>10,} {r['long_rate']*100:>11.1f}% {r['short_rate']*100:>11.1f}%")

# =============================================================================
# THREE-FILTER COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("THREE-FILTER COMBINATIONS (sorted by LONG expansion rate)")
print("=" * 70)

three_filter_results = []

for f1, f2, f3 in combinations(filter_names, 3):
    combined_mask = FILTERS[f1] & FILTERS[f2] & FILTERS[f3]
    n_samples = combined_mask.sum()

    if n_samples >= MIN_SAMPLES:
        long_rate = df.loc[combined_mask, long_col].mean()
        short_rate = df.loc[combined_mask, short_col].mean()

        three_filter_results.append({
            'filters': f"{f1} + {f2} + {f3}",
            'samples': n_samples,
            'long_rate': long_rate,
            'short_rate': short_rate,
        })

# Sort by long rate descending
three_filter_results = sorted(three_filter_results, key=lambda x: x['long_rate'], reverse=True)

print(f"\n{'Filter Combination':<60} {'Samples':>10} {'LONG Rate':>12} {'SHORT Rate':>12}")
print("-" * 100)

# Show top 15
for r in three_filter_results[:15]:
    print(f"{r['filters']:<60} {r['samples']:>10,} {r['long_rate']*100:>11.1f}% {r['short_rate']*100:>11.1f}%")

# =============================================================================
# BEST LONG EXPANSION COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("BEST COMBINATIONS FOR LONG EXPANSION (>50% rate)")
print("=" * 70)

all_results = []
all_results.extend([(r['filter'], r['samples'], r['long_rate'], 'single') for r in single_results])
all_results.extend([(r['filters'], r['samples'], r['long_rate'], 'two') for r in two_filter_results])
all_results.extend([(r['filters'], r['samples'], r['long_rate'], 'three') for r in three_filter_results])

# Filter for >50% and sort
best_long = [r for r in all_results if r[2] >= 0.50]
best_long = sorted(best_long, key=lambda x: x[2], reverse=True)

print(f"\n{'Filter Combination':<65} {'Samples':>10} {'LONG Rate':>12}")
print("-" * 90)

for filters, samples, rate, ftype in best_long[:20]:
    profitable = "* PROFITABLE" if rate >= 0.58 else ""
    print(f"{filters:<65} {samples:>10,} {rate*100:>11.1f}% {profitable}")

# =============================================================================
# BEST SHORT EXPANSION COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("BEST COMBINATIONS FOR SHORT EXPANSION (>50% rate)")
print("=" * 70)

all_short = []
all_short.extend([(r['filter'], r['samples'], r['short_rate']) for r in single_results])
all_short.extend([(r['filters'], r['samples'], r['short_rate']) for r in two_filter_results])
all_short.extend([(r['filters'], r['samples'], r['short_rate']) for r in three_filter_results])

# Filter for >50% and sort
best_short = [r for r in all_short if r[2] >= 0.50]
best_short = sorted(best_short, key=lambda x: x[2], reverse=True)

print(f"\n{'Filter Combination':<65} {'Samples':>10} {'SHORT Rate':>12}")
print("-" * 90)

for filters, samples, rate in best_short[:20]:
    profitable = "* PROFITABLE" if rate >= 0.58 else ""
    print(f"{filters:<65} {samples:>10,} {rate*100:>11.1f}% {profitable}")

# =============================================================================
# REGIME-BASED ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("EXPANSION RATES BY REGIME")
print("=" * 70)

print(f"\n{'Regime':<25} {'Samples':>12} {'LONG Rate':>12} {'SHORT Rate':>12}")
print("-" * 65)

for regime in df['regime'].unique():
    mask = df['regime'] == regime
    n = mask.sum()
    long_rate = df.loc[mask, long_col].mean()
    short_rate = df.loc[mask, short_col].mean()
    print(f"{regime:<25} {n:>12,} {long_rate*100:>11.1f}% {short_rate*100:>11.1f}%")

# =============================================================================
# REGIME + FILTER COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("REGIME + FILTER COMBINATIONS (LONG expansion)")
print("=" * 70)

regime_filter_results = []
for regime in df['regime'].unique():
    regime_mask = df['regime'] == regime

    for fname, fmask in FILTERS.items():
        combined = regime_mask & fmask
        n = combined.sum()

        if n >= MIN_SAMPLES:
            long_rate = df.loc[combined, long_col].mean()
            short_rate = df.loc[combined, short_col].mean()

            regime_filter_results.append({
                'combo': f"{regime} + {fname}",
                'samples': n,
                'long_rate': long_rate,
                'short_rate': short_rate,
            })

# Sort by long rate
regime_filter_results = sorted(regime_filter_results, key=lambda x: x['long_rate'], reverse=True)

print(f"\n{'Regime + Filter':<50} {'Samples':>10} {'LONG Rate':>12} {'SHORT Rate':>12}")
print("-" * 88)

for r in regime_filter_results[:20]:
    profitable = "* PROF" if r['long_rate'] >= 0.58 else ""
    print(f"{r['combo']:<50} {r['samples']:>10,} {r['long_rate']*100:>11.1f}% {r['short_rate']*100:>11.1f}% {profitable}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Check if any combination reaches profitability threshold
profitable_threshold = 0.58

profitable_combos = [r for r in all_results if r[2] >= profitable_threshold]
profitable_short = [r for r in all_short if r[2] >= profitable_threshold]

print(f"\nBase LONG rate: {base_long_rate*100:.1f}%")
print(f"Base SHORT rate: {base_short_rate*100:.1f}%")
print(f"Target for profitability: {profitable_threshold*100:.0f}%")

if profitable_combos:
    print(f"\n** FOUND {len(profitable_combos)} PROFITABLE LONG COMBINATIONS! **")
    for filters, samples, rate, _ in profitable_combos[:5]:
        print(f"  - {filters}: {rate*100:.1f}% ({samples:,} samples)")
else:
    print(f"\nNo LONG combinations reach {profitable_threshold*100:.0f}% threshold.")
    if all_results:
        best = max(all_results, key=lambda x: x[2])
        print(f"Best LONG: {best[0]} at {best[2]*100:.1f}% ({best[1]:,} samples)")

if profitable_short:
    print(f"\n** FOUND {len(profitable_short)} PROFITABLE SHORT COMBINATIONS! **")
    for filters, samples, rate in profitable_short[:5]:
        print(f"  - {filters}: {rate*100:.1f}% ({samples:,} samples)")
else:
    print(f"\nNo SHORT combinations reach {profitable_threshold*100:.0f}% threshold.")
    if all_short:
        best = max(all_short, key=lambda x: x[2])
        print(f"Best SHORT: {best[0]} at {best[2]*100:.1f}% ({best[1]:,} samples)")

print("\n" + "=" * 70)
