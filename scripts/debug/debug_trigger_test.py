"""
Test micro-triggers alone vs with expansion filter.

Run: .venv\Scripts\python.exe debug_trigger_test.py

This answers: "Does expansion filter improve trigger performance?"
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
HORIZON = 120  # Forward-looking horizon for measuring outcomes
INVALIDATION_RATIO = 0.5

# Fees
ROUND_TRIP_FEE_BPS = 8  # 0.08% round trip

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("MICRO-TRIGGER TESTING")
print("=" * 70)
print("\nLoading data...")

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    exit(1)
ohlcv = pd.read_parquet(ohlcv_path)

regime_path = Path("data/regimes/BTCUSDT_1m_regimes.parquet")
if not regime_path.exists():
    print(f"ERROR: Regime file not found: {regime_path}")
    exit(1)
regime_df = pd.read_parquet(regime_path)

print(f"OHLCV: {len(ohlcv):,} candles")
print(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")

# =============================================================================
# STEP 1: DETECT ALL TRIGGERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: DETECTING MICRO-TRIGGERS")
print("=" * 70)

config = TriggerConfig(
    engulfing_min_body_ratio=0.6,
    pin_bar_wick_ratio=2.0,
    volume_spike_multiplier=2.0,
    min_candle_range_bps=5.0,
)

detector = MicroTriggerDetector(ohlcv, config)
triggers_df = detector.detect_all(show_progress=True)

# Trigger statistics
print("\nTrigger counts by type:")
trigger_types = ['engulfing', 'pinbar', 'insidebar', 'volumespike', 'retest']
for ttype in trigger_types:
    long_count = triggers_df[f'trigger_{ttype}_long'].sum()
    short_count = triggers_df[f'trigger_{ttype}_short'].sum()
    total = long_count + short_count
    print(f"  {ttype.upper():<15}: {total:>10,} ({long_count:,} long, {short_count:,} short)")

any_long = triggers_df['any_trigger_long'].sum()
any_short = triggers_df['any_trigger_short'].sum()
print(f"\n  ANY TRIGGER:     {any_long + any_short:>10,} ({any_long:,} long, {any_short:,} short)")

# =============================================================================
# STEP 2: COMPUTE EXPANSION LABELS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: COMPUTING EXPANSION LABELS")
print("=" * 70)

# Compute thresholds
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
target_pct = median_move  # Using median for higher base rate
invalidation_pct = median_move * INVALIDATION_RATIO

print(f"Target (median): {target_pct*10000:.1f} bps")
print(f"Invalidation: {invalidation_pct*10000:.1f} bps")

# Create expansion labels
exp_config = ExpansionConfig(
    horizon=HORIZON,
    target_pct=target_pct,
    invalidation_pct=invalidation_pct,
)
labeler = ExpansionLabeler(ohlcv)
expansion_df = labeler.label(exp_config, show_progress=True)

long_exp_col = f'long_expansion_{HORIZON}m'
short_exp_col = f'short_expansion_{HORIZON}m'

overall_long_rate = expansion_df[long_exp_col].mean()
overall_short_rate = expansion_df[short_exp_col].mean()
print(f"\nOverall expansion rate: LONG={overall_long_rate*100:.1f}%, SHORT={overall_short_rate*100:.1f}%")

# =============================================================================
# STEP 3: COMPUTE FORWARD RETURNS FOR ALL TRIGGERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: COMPUTING TRIGGER OUTCOMES")
print("=" * 70)

# Create forward returns
print("Computing forward returns...")
ohlcv['fwd_return'] = (ohlcv['close'].shift(-HORIZON) - ohlcv['close']) / ohlcv['close']

# For long triggers: positive return = win
# For short triggers: negative return = win

# Merge all data
common_idx = triggers_df.index.intersection(expansion_df.index).intersection(ohlcv.index)
common_idx = common_idx[:-HORIZON]  # Exclude last HORIZON bars (no forward data)

merged = pd.DataFrame(index=common_idx)
merged['fwd_return'] = ohlcv.loc[common_idx, 'fwd_return']
merged['long_expansion'] = expansion_df.loc[common_idx, long_exp_col]
merged['short_expansion'] = expansion_df.loc[common_idx, short_exp_col]

# Add all trigger columns
for col in triggers_df.columns:
    merged[col] = triggers_df.loc[common_idx, col]

print(f"Merged data: {len(merged):,} rows")

# =============================================================================
# STEP 4: MEASURE WIN RATES - TRIGGERS ALONE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: TRIGGER PERFORMANCE (WITHOUT EXPANSION FILTER)")
print("=" * 70)

fee_pct = ROUND_TRIP_FEE_BPS / 10000  # Convert bps to decimal

def compute_win_rate(df, trigger_col, direction):
    """Compute win rate for a trigger."""
    triggered = df[df[trigger_col] == 1]
    if len(triggered) == 0:
        return None, 0

    if direction == "LONG":
        # Win if forward return > fees
        wins = (triggered['fwd_return'] > fee_pct).sum()
    else:
        # Win if forward return < -fees (price went down)
        wins = (triggered['fwd_return'] < -fee_pct).sum()

    return wins / len(triggered), len(triggered)

print(f"\n{'Trigger':<20} {'Direction':<8} {'Signals':>10} {'Win Rate':>12} {'vs Random':>12}")
print("-" * 65)

# Random baseline (what's the win rate if we just guess?)
all_long_wins = (merged['fwd_return'] > fee_pct).sum() / len(merged)
all_short_wins = (merged['fwd_return'] < -fee_pct).sum() / len(merged)
print(f"{'RANDOM':<20} {'LONG':<8} {len(merged):>10,} {all_long_wins*100:>11.1f}% {'baseline':>12}")
print(f"{'RANDOM':<20} {'SHORT':<8} {len(merged):>10,} {all_short_wins*100:>11.1f}% {'baseline':>12}")
print("-" * 65)

trigger_results = {}
for ttype in trigger_types:
    for direction in ['LONG', 'SHORT']:
        col = f'trigger_{ttype}_{direction.lower()}'
        win_rate, count = compute_win_rate(merged, col, direction)

        if win_rate is not None:
            baseline = all_long_wins if direction == "LONG" else all_short_wins
            edge = (win_rate - baseline) * 100
            edge_str = f"{edge:+.1f}pp"

            trigger_results[f"{ttype}_{direction}"] = {
                'win_rate': win_rate,
                'count': count,
                'edge': edge,
            }

            print(f"{ttype.upper():<20} {direction:<8} {count:>10,} {win_rate*100:>11.1f}% {edge_str:>12}")

# Combined (any trigger)
for direction in ['LONG', 'SHORT']:
    col = f'any_trigger_{direction.lower()}'
    win_rate, count = compute_win_rate(merged, col, direction)

    if win_rate is not None:
        baseline = all_long_wins if direction == "LONG" else all_short_wins
        edge = (win_rate - baseline) * 100
        edge_str = f"{edge:+.1f}pp"
        print(f"{'ANY TRIGGER':<20} {direction:<8} {count:>10,} {win_rate*100:>11.1f}% {edge_str:>12}")

# =============================================================================
# STEP 5: MEASURE WIN RATES - TRIGGERS + EXPANSION FILTER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: TRIGGER PERFORMANCE (WITH EXPANSION FILTER)")
print("=" * 70)
print("Filter: Only trade when expansion_rate = 1 (historically expanded)")
print()

def compute_filtered_win_rate(df, trigger_col, direction, expansion_col):
    """Compute win rate for trigger + expansion filter."""
    # Both trigger and expansion must be 1
    filtered = df[(df[trigger_col] == 1) & (df[expansion_col] == 1)]
    if len(filtered) == 0:
        return None, 0

    if direction == "LONG":
        wins = (filtered['fwd_return'] > fee_pct).sum()
    else:
        wins = (filtered['fwd_return'] < -fee_pct).sum()

    return wins / len(filtered), len(filtered)

print(f"{'Trigger':<20} {'Direction':<8} {'Signals':>10} {'Win Rate':>12} {'vs Alone':>12}")
print("-" * 65)

filtered_results = {}
for ttype in trigger_types:
    for direction in ['LONG', 'SHORT']:
        trigger_col = f'trigger_{ttype}_{direction.lower()}'
        exp_col = 'long_expansion' if direction == "LONG" else 'short_expansion'

        win_rate, count = compute_filtered_win_rate(merged, trigger_col, direction, exp_col)

        if win_rate is not None and count >= 100:  # Minimum sample size
            alone_key = f"{ttype}_{direction}"
            if alone_key in trigger_results:
                alone_rate = trigger_results[alone_key]['win_rate']
                improvement = (win_rate - alone_rate) * 100
                improvement_str = f"{improvement:+.1f}pp"
            else:
                improvement_str = "N/A"

            filtered_results[f"{ttype}_{direction}"] = {
                'win_rate': win_rate,
                'count': count,
            }

            print(f"{ttype.upper():<20} {direction:<8} {count:>10,} {win_rate*100:>11.1f}% {improvement_str:>12}")

# Combined (any trigger + expansion)
print("-" * 65)
for direction in ['LONG', 'SHORT']:
    trigger_col = f'any_trigger_{direction.lower()}'
    exp_col = 'long_expansion' if direction == "LONG" else 'short_expansion'

    win_rate, count = compute_filtered_win_rate(merged, trigger_col, direction, exp_col)

    if win_rate is not None:
        # Get alone rate
        alone = merged[merged[trigger_col] == 1]
        if direction == "LONG":
            alone_rate = (alone['fwd_return'] > fee_pct).sum() / len(alone) if len(alone) > 0 else 0
        else:
            alone_rate = (alone['fwd_return'] < -fee_pct).sum() / len(alone) if len(alone) > 0 else 0

        improvement = (win_rate - alone_rate) * 100
        improvement_str = f"{improvement:+.1f}pp"

        print(f"{'ANY + EXPANSION':<20} {direction:<8} {count:>10,} {win_rate*100:>11.1f}% {improvement_str:>12}")

# =============================================================================
# STEP 6: BEST COMBINATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: BEST TRIGGER + EXPANSION COMBINATIONS")
print("=" * 70)

# Find combinations with >50% win rate
good_combos = []

for ttype in trigger_types:
    for direction in ['LONG', 'SHORT']:
        trigger_col = f'trigger_{ttype}_{direction.lower()}'
        exp_col = 'long_expansion' if direction == "LONG" else 'short_expansion'

        win_rate, count = compute_filtered_win_rate(merged, trigger_col, direction, exp_col)

        if win_rate is not None and count >= 50:
            good_combos.append({
                'trigger': ttype.upper(),
                'direction': direction,
                'win_rate': win_rate,
                'count': count,
            })

# Sort by win rate
good_combos.sort(key=lambda x: x['win_rate'], reverse=True)

if len(good_combos) > 0:
    print("\nTop combinations (sorted by win rate):")
    print(f"{'Trigger':<15} {'Direction':<8} {'Win Rate':>10} {'Signals':>10} {'Profitable?':>12}")
    print("-" * 58)

    for combo in good_combos[:10]:
        profitable = "YES" if combo['win_rate'] > 0.50 else "NO"
        print(f"{combo['trigger']:<15} {combo['direction']:<8} {combo['win_rate']*100:>9.1f}% {combo['count']:>10,} {profitable:>12}")
else:
    print("\nNo combinations with sufficient samples found.")

# =============================================================================
# STEP 7: PROFITABILITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: PROFITABILITY ANALYSIS")
print("=" * 70)

# For profitable combos, estimate P&L
target_bps = target_pct * 10000
fee_bps = ROUND_TRIP_FEE_BPS

print(f"\nAssumptions:")
print(f"  Target (median move): {target_bps:.1f} bps")
print(f"  Stop loss (50% of target): {target_bps/2:.1f} bps")
print(f"  Fees: {fee_bps} bps round trip")
print(f"  R:R ratio: 2:1")

profitable_found = False
for combo in good_combos:
    if combo['win_rate'] > 0.50:
        profitable_found = True
        win_rate = combo['win_rate']

        # Expected value per trade (in bps)
        # Win: +target - fees
        # Lose: -stop - fees
        win_payout = target_bps - fee_bps
        lose_payout = -(target_bps / 2) - fee_bps

        ev_bps = win_rate * win_payout + (1 - win_rate) * lose_payout

        print(f"\n{combo['trigger']} {combo['direction']}:")
        print(f"  Win rate: {win_rate*100:.1f}%")
        print(f"  Signals: {combo['count']:,}")
        print(f"  EV per trade: {ev_bps:+.1f} bps")

        if ev_bps > 0:
            print(f"  STATUS: PROFITABLE!")
        else:
            print(f"  STATUS: Not profitable (need higher win rate)")

if not profitable_found:
    print("\nNo combinations with >50% win rate found.")
    print("The expansion filter did not improve triggers enough for profitability.")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Calculate overall improvement
trigger_alone_rates = [v['win_rate'] for v in trigger_results.values()]
filtered_rates = [v['win_rate'] for v in filtered_results.values() if v['count'] >= 100]

if len(trigger_alone_rates) > 0 and len(filtered_rates) > 0:
    avg_alone = np.mean(trigger_alone_rates) * 100
    avg_filtered = np.mean(filtered_rates) * 100
    improvement = avg_filtered - avg_alone

    print(f"""
TRIGGER PERFORMANCE:
  - Average win rate (triggers alone): {avg_alone:.1f}%
  - Average win rate (triggers + expansion): {avg_filtered:.1f}%
  - Improvement from expansion filter: {improvement:+.1f} percentage points

CONCLUSION:
  {'Expansion filter HELPS! Consider using for live trading.' if improvement > 2 else 'Expansion filter provides minimal improvement.'}
  {'Some combinations are profitable!' if profitable_found else 'No profitable combinations found yet.'}

NEXT STEPS:
  1. If profitable combinations exist: backtest with real execution
  2. If not: try different trigger parameters or longer horizons
  3. Consider combining multiple triggers for higher conviction
""")
else:
    print("\nInsufficient data for analysis.")
