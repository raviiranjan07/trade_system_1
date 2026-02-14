"""
PROPER TRAIN/TEST VALIDATION
=============================
1. TRAIN (2020-2023): Compute thresholds, find best filters/regimes
2. TEST (2024-2025): Apply SAME thresholds, validate if patterns hold

This is the REAL test - no look-ahead bias.

Run: .venv/Scripts/python.exe debug_train_test_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZON = 60  # 1 hour
INVALIDATION_RATIO = 0.5
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
MIN_SAMPLES = 500  # Minimum samples for valid analysis

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("TRAIN / TEST VALIDATION")
print("=" * 70)
print(f"\nHorizon: {HORIZON} minutes")
print(f"Train: up to {TRAIN_END}")
print(f"Test: from {TEST_START}")

print("\nLoading data...")
ohlcv = pd.read_parquet("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
outcome_df = pd.read_parquet("data/outcomes/BTCUSDT_1m_outcomes.parquet")
regime_df = pd.read_parquet("data/regimes/BTCUSDT_1m_regimes.parquet")

print(f"Total data: {len(ohlcv):,} candles")
print(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")

# =============================================================================
# SPLIT DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: SPLIT DATA")
print("=" * 70)

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
test_ohlcv = ohlcv[ohlcv.index >= TEST_START]

train_outcomes = outcome_df[outcome_df.index <= TRAIN_END]
test_outcomes = outcome_df[outcome_df.index >= TEST_START]

train_regimes = regime_df[regime_df.index <= TRAIN_END]
test_regimes = regime_df[regime_df.index >= TEST_START]

train_days = (train_ohlcv.index.max() - train_ohlcv.index.min()).days
test_days = (test_ohlcv.index.max() - test_ohlcv.index.min()).days

print(f"\nTRAIN: {len(train_ohlcv):,} candles ({train_days:,} days)")
print(f"  {train_ohlcv.index.min()} to {train_ohlcv.index.max()}")
print(f"\nTEST:  {len(test_ohlcv):,} candles ({test_days:,} days)")
print(f"  {test_ohlcv.index.min()} to {test_ohlcv.index.max()}")

# =============================================================================
# STEP 2: COMPUTE THRESHOLDS FROM TRAIN DATA ONLY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: COMPUTE THRESHOLDS (TRAIN DATA ONLY)")
print("=" * 70)

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

print(f"\nComputing moves for H={HORIZON} on TRAIN data...")
moves = []
for i in range(0, n - HORIZON, 10):  # Sample every 10 for speed
    entry = close[i]
    future_high = np.max(high[i+1:i+1+HORIZON])
    future_low = np.min(low[i+1:i+1+HORIZON])
    moves.append((future_high - entry) / entry)
    moves.append((entry - future_low) / entry)

moves = np.array(moves)
median_move = np.percentile(moves, 50)
p75_move = np.percentile(moves, 75)
target_pct = median_move  # Use 50th percentile (median) as target
stop_pct = target_pct * INVALIDATION_RATIO  # Stop = 50% of target (2:1 R:R)

print(f"\nTRAIN thresholds:")
print(f"  Median move (50th): {median_move*10000:.1f} bps")
print(f"  75th percentile: {p75_move*10000:.1f} bps")
print(f"  TARGET: {target_pct*10000:.1f} bps")
print(f"  STOP: {stop_pct*10000:.1f} bps")
print(f"  R:R ratio: {target_pct/stop_pct:.1f}:1")

# =============================================================================
# STEP 3: COMPUTE EXPANSION LABELS (BOTH TRAIN & TEST)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: COMPUTE EXPANSION LABELS")
print("=" * 70)

def compute_expansion(ohlcv_df, target_pct, stop_pct, horizon):
    """Compute expansion labels for a dataset."""
    close = ohlcv_df['close'].values
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    n = len(ohlcv_df)

    long_exp = np.zeros(n, dtype=np.int8)
    short_exp = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        entry = close[i]

        # LONG barriers
        long_target = entry * (1 + target_pct)
        long_stop = entry * (1 - stop_pct)

        # SHORT barriers
        short_target = entry * (1 - target_pct)
        short_stop = entry * (1 + stop_pct)

        long_done = False
        short_done = False

        for j in range(1, horizon + 1):
            idx = i + j
            if idx >= n:
                break

            h = high[idx]
            l = low[idx]

            # Check LONG
            if not long_done:
                if h >= long_target:
                    long_exp[i] = 1
                    long_done = True
                elif l <= long_stop:
                    long_done = True

            # Check SHORT
            if not short_done:
                if l <= short_target:
                    short_exp[i] = 1
                    short_done = True
                elif h >= short_stop:
                    short_done = True

            if long_done and short_done:
                break

    return pd.DataFrame({
        'long_exp': long_exp,
        'short_exp': short_exp,
    }, index=ohlcv_df.index)

print("\nComputing TRAIN expansion labels...")
train_exp = compute_expansion(train_ohlcv, target_pct, stop_pct, HORIZON)
train_long_rate = train_exp['long_exp'].mean()
train_short_rate = train_exp['short_exp'].mean()
print(f"  TRAIN expansion: LONG={train_long_rate*100:.1f}%, SHORT={train_short_rate*100:.1f}%")

print("\nComputing TEST expansion labels (using TRAIN thresholds)...")
test_exp = compute_expansion(test_ohlcv, target_pct, stop_pct, HORIZON)
test_long_rate = test_exp['long_exp'].mean()
test_short_rate = test_exp['short_exp'].mean()
print(f"  TEST expansion: LONG={test_long_rate*100:.1f}%, SHORT={test_short_rate*100:.1f}%")

# =============================================================================
# STEP 4: MERGE WITH FEATURES & REGIMES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: MERGE DATA")
print("=" * 70)

# TRAIN merge
train_common = train_outcomes.index.intersection(train_regimes.index).intersection(train_exp.index)
train_df = train_outcomes.loc[train_common].copy()
train_df['regime'] = train_regimes.loc[train_common, 'regime']
train_df['long_exp'] = train_exp.loc[train_common, 'long_exp']
train_df['short_exp'] = train_exp.loc[train_common, 'short_exp']
print(f"TRAIN merged: {len(train_df):,} rows")

# TEST merge
test_common = test_outcomes.index.intersection(test_regimes.index).intersection(test_exp.index)
test_df = test_outcomes.loc[test_common].copy()
test_df['regime'] = test_regimes.loc[test_common, 'regime']
test_df['long_exp'] = test_exp.loc[test_common, 'long_exp']
test_df['short_exp'] = test_exp.loc[test_common, 'short_exp']
print(f"TEST merged: {len(test_df):,} rows")

# =============================================================================
# STEP 5: EXPANSION RATES BY REGIME
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: EXPANSION RATES BY REGIME")
print("=" * 70)

print(f"\n{'Regime':<20} {'TRAIN Long':>12} {'TEST Long':>12} {'TRAIN Short':>12} {'TEST Short':>12}")
print("-" * 72)

for regime in train_df['regime'].unique():
    train_mask = train_df['regime'] == regime
    test_mask = test_df['regime'] == regime

    train_n = train_mask.sum()
    test_n = test_mask.sum()

    if train_n >= MIN_SAMPLES and test_n >= MIN_SAMPLES:
        train_long = train_df.loc[train_mask, 'long_exp'].mean() * 100
        test_long = test_df.loc[test_mask, 'long_exp'].mean() * 100
        train_short = train_df.loc[train_mask, 'short_exp'].mean() * 100
        test_short = test_df.loc[test_mask, 'short_exp'].mean() * 100

        print(f"{regime:<20} {train_long:>11.1f}% {test_long:>11.1f}% {train_short:>11.1f}% {test_short:>11.1f}%")

# =============================================================================
# STEP 6: FILTER ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: FILTER ANALYSIS (TRAIN vs TEST)")
print("=" * 70)

# Define filters
def create_filters(df):
    return {
        "high_volume": df['volume_z'] > 1.0,
        "very_high_volume": df['volume_z'] > 2.0,
        "low_rsi": df['rsi_z'] < -1.0,
        "high_rsi": df['rsi_z'] > 1.0,
        "high_atr": df['atr_percentile'] > 0.7,
        "low_range_pos": df['range_position'] < 0.3,
        "high_range_pos": df['range_position'] > 0.7,
        "recent_down": df['return_5m_z'] < -1.0,
        "recent_up": df['return_5m_z'] > 1.0,
        "vwap_below": df['vwap_distance_z'] < -1.0,
        "vwap_above": df['vwap_distance_z'] > 1.0,
    }

train_filters = create_filters(train_df)
test_filters = create_filters(test_df)

print("\nSINGLE FILTER RESULTS:")
print(f"\n{'Filter':<25} {'TRAIN LONG':>12} {'TEST LONG':>12} {'Diff':>8} {'TRAIN SHORT':>12} {'TEST SHORT':>12}")
print("-" * 90)

single_results = []
for name in train_filters.keys():
    train_mask = train_filters[name]
    test_mask = test_filters[name]

    train_n = train_mask.sum()
    test_n = test_mask.sum()

    if train_n >= MIN_SAMPLES and test_n >= MIN_SAMPLES:
        train_long = train_df.loc[train_mask, 'long_exp'].mean() * 100
        test_long = test_df.loc[test_mask, 'long_exp'].mean() * 100
        train_short = train_df.loc[train_mask, 'short_exp'].mean() * 100
        test_short = test_df.loc[test_mask, 'short_exp'].mean() * 100
        diff = test_long - train_long

        single_results.append({
            'filter': name,
            'train_long': train_long,
            'test_long': test_long,
            'train_short': train_short,
            'test_short': test_short,
            'diff': diff,
            'train_n': train_n,
            'test_n': test_n,
        })

        print(f"{name:<25} {train_long:>11.1f}% {test_long:>11.1f}% {diff:>+7.1f}pp {train_short:>11.1f}% {test_short:>11.1f}%")

# =============================================================================
# STEP 7: TWO-FILTER COMBINATIONS
# =============================================================================
print("\n" + "-" * 90)
print("BEST TWO-FILTER COMBINATIONS:")
print("-" * 90)

two_filter_results = []
filter_names = list(train_filters.keys())

for f1, f2 in combinations(filter_names, 2):
    train_mask = train_filters[f1] & train_filters[f2]
    test_mask = test_filters[f1] & test_filters[f2]

    train_n = train_mask.sum()
    test_n = test_mask.sum()

    if train_n >= MIN_SAMPLES and test_n >= MIN_SAMPLES:
        train_long = train_df.loc[train_mask, 'long_exp'].mean() * 100
        test_long = test_df.loc[test_mask, 'long_exp'].mean() * 100
        train_short = train_df.loc[train_mask, 'short_exp'].mean() * 100
        test_short = test_df.loc[test_mask, 'short_exp'].mean() * 100

        two_filter_results.append({
            'filters': f"{f1} + {f2}",
            'train_long': train_long,
            'test_long': test_long,
            'train_short': train_short,
            'test_short': test_short,
            'train_n': train_n,
            'test_n': test_n,
        })

# Sort by TRAIN long rate
two_filter_results = sorted(two_filter_results, key=lambda x: x['train_long'], reverse=True)

print(f"\n{'Filters':<45} {'TRAIN LONG':>12} {'TEST LONG':>12} {'Diff':>8}")
print("-" * 80)

for r in two_filter_results[:15]:
    diff = r['test_long'] - r['train_long']
    print(f"{r['filters']:<45} {r['train_long']:>11.1f}% {r['test_long']:>11.1f}% {diff:>+7.1f}pp")

# =============================================================================
# STEP 8: PROFITABILITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: PROFITABILITY ANALYSIS")
print("=" * 70)

FEE_BPS = 8
win_payout = target_pct * 10000 - FEE_BPS
lose_payout = stop_pct * 10000 + FEE_BPS

# Break-even win rate
breakeven = lose_payout / (win_payout + lose_payout) * 100

print(f"\nTarget: {target_pct*10000:.1f} bps, Stop: {stop_pct*10000:.1f} bps")
print(f"Win payout: +{win_payout:.1f} bps, Lose payout: -{lose_payout:.1f} bps")
print(f"Break-even win rate: {breakeven:.1f}%")

print(f"\n{'Strategy':<45} {'TRAIN WR':>10} {'TEST WR':>10} {'TRAIN EV':>10} {'TEST EV':>10}")
print("-" * 90)

# Base rates
train_base_ev = train_long_rate * win_payout - (1 - train_long_rate) * lose_payout
test_base_ev = test_long_rate * win_payout - (1 - test_long_rate) * lose_payout
print(f"{'Base (no filter)':<45} {train_long_rate*100:>9.1f}% {test_long_rate*100:>9.1f}% {train_base_ev:>+9.1f} {test_base_ev:>+9.1f}")

# Best single filters
for r in sorted(single_results, key=lambda x: x['train_long'], reverse=True)[:5]:
    train_ev = r['train_long']/100 * win_payout - (1 - r['train_long']/100) * lose_payout
    test_ev = r['test_long']/100 * win_payout - (1 - r['test_long']/100) * lose_payout
    print(f"{r['filter']:<45} {r['train_long']:>9.1f}% {r['test_long']:>9.1f}% {train_ev:>+9.1f} {test_ev:>+9.1f}")

# Best two-filter combos
for r in two_filter_results[:5]:
    train_ev = r['train_long']/100 * win_payout - (1 - r['train_long']/100) * lose_payout
    test_ev = r['test_long']/100 * win_payout - (1 - r['test_long']/100) * lose_payout
    print(f"{r['filters']:<45} {r['train_long']:>9.1f}% {r['test_long']:>9.1f}% {train_ev:>+9.1f} {test_ev:>+9.1f}")

# =============================================================================
# STEP 9: FIND STRATEGIES THAT WORK ON BOTH
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: STRATEGIES PROFITABLE ON BOTH TRAIN AND TEST")
print("=" * 70)

profitable_both = []

# Check single filters
for r in single_results:
    train_ev = r['train_long']/100 * win_payout - (1 - r['train_long']/100) * lose_payout
    test_ev = r['test_long']/100 * win_payout - (1 - r['test_long']/100) * lose_payout

    if train_ev > 0 and test_ev > 0:
        profitable_both.append({
            'name': r['filter'],
            'train_wr': r['train_long'],
            'test_wr': r['test_long'],
            'train_ev': train_ev,
            'test_ev': test_ev,
            'train_n': r['train_n'],
            'test_n': r['test_n'],
        })

# Check two-filter combos
for r in two_filter_results:
    train_ev = r['train_long']/100 * win_payout - (1 - r['train_long']/100) * lose_payout
    test_ev = r['test_long']/100 * win_payout - (1 - r['test_long']/100) * lose_payout

    if train_ev > 0 and test_ev > 0:
        profitable_both.append({
            'name': r['filters'],
            'train_wr': r['train_long'],
            'test_wr': r['test_long'],
            'train_ev': train_ev,
            'test_ev': test_ev,
            'train_n': r['train_n'],
            'test_n': r['test_n'],
        })

if profitable_both:
    print(f"\n*** FOUND {len(profitable_both)} PROFITABLE STRATEGIES ***")
    profitable_both = sorted(profitable_both, key=lambda x: x['test_ev'], reverse=True)

    print(f"\n{'Strategy':<50} {'TRAIN':>8} {'TEST':>8} {'TRAIN EV':>10} {'TEST EV':>10}")
    print("-" * 90)

    for s in profitable_both[:10]:
        print(f"{s['name']:<50} {s['train_wr']:>7.1f}% {s['test_wr']:>7.1f}% {s['train_ev']:>+9.1f} {s['test_ev']:>+9.1f}")

    # Estimate returns for best strategy
    best = profitable_both[0]
    trades_per_day_train = best['train_n'] / train_days
    trades_per_day_test = best['test_n'] / test_days
    daily_return = trades_per_day_test * best['test_ev'] / 10000 * 100

    print(f"\n--- BEST STRATEGY: {best['name']} ---")
    print(f"  TRAIN: {best['train_wr']:.1f}% WR, {best['train_ev']:+.1f} bps/trade")
    print(f"  TEST:  {best['test_wr']:.1f}% WR, {best['test_ev']:+.1f} bps/trade")
    print(f"  Trades per day (test): {trades_per_day_test:.1f}")
    print(f"  Est. daily return: {daily_return:+.4f}%")
    print(f"  Est. monthly return: {daily_return * 30:+.2f}%")
else:
    print("\nNo strategies profitable on BOTH train and test.")
    print("\nBest strategies that were profitable on TRAIN but not TEST:")

    train_profitable = []
    for r in single_results:
        train_ev = r['train_long']/100 * win_payout - (1 - r['train_long']/100) * lose_payout
        test_ev = r['test_long']/100 * win_payout - (1 - r['test_long']/100) * lose_payout
        if train_ev > 0:
            train_profitable.append({
                'name': r['filter'],
                'train_wr': r['train_long'],
                'test_wr': r['test_long'],
                'train_ev': train_ev,
                'test_ev': test_ev,
            })

    for r in two_filter_results:
        train_ev = r['train_long']/100 * win_payout - (1 - r['train_long']/100) * lose_payout
        test_ev = r['test_long']/100 * win_payout - (1 - r['test_long']/100) * lose_payout
        if train_ev > 0:
            train_profitable.append({
                'name': r['filters'],
                'train_wr': r['train_long'],
                'test_wr': r['test_long'],
                'train_ev': train_ev,
                'test_ev': test_ev,
            })

    if train_profitable:
        train_profitable = sorted(train_profitable, key=lambda x: x['train_ev'], reverse=True)
        print(f"\n{'Strategy':<50} {'TRAIN WR':>10} {'TEST WR':>10} {'Difference':>12}")
        print("-" * 85)
        for s in train_profitable[:10]:
            diff = s['test_wr'] - s['train_wr']
            print(f"{s['name']:<50} {s['train_wr']:>9.1f}% {s['test_wr']:>9.1f}% {diff:>+11.1f}pp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
TRAIN/TEST VALIDATION RESULTS:

  Train period: {train_ohlcv.index.min().date()} to {train_ohlcv.index.max().date()} ({train_days:,} days)
  Test period:  {test_ohlcv.index.min().date()} to {test_ohlcv.index.max().date()} ({test_days:,} days)

  Thresholds (from TRAIN):
    Target: {target_pct*10000:.1f} bps
    Stop: {stop_pct*10000:.1f} bps
    Break-even WR: {breakeven:.1f}%

  Base expansion rates:
    TRAIN: LONG={train_long_rate*100:.1f}%, SHORT={train_short_rate*100:.1f}%
    TEST:  LONG={test_long_rate*100:.1f}%, SHORT={test_short_rate*100:.1f}%
""")

if profitable_both:
    print(f"  RESULT: {len(profitable_both)} strategies profitable on BOTH train and test!")
else:
    print(f"  RESULT: No strategies generalize from train to test.")
    print(f"  This suggests: Patterns found in training are likely overfitting.")

print("\n" + "=" * 70)
