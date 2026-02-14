"""
PROPER TRAIN/TEST PREDICTION TEST

This is the REAL test - no look-ahead bias.

1. TRAIN (2020-2023): Build similarity index + expansion labels
2. TEST (2024-2025): Make predictions using ONLY train data
3. MEASURE: Compare predictions to actual outcomes

Run: .venv\Scripts\python.exe debug_prediction_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
# Note: outcome_df only has horizons [2, 3, 5, 10, 15, 30]
# Using H=30 for this test (longest available in outcome data)
HORIZON = 30  # 30 minutes
INVALIDATION_RATIO = 0.5
K_NEIGHBORS = 200
ROUND_TRIP_FEE_BPS = 8

# Train/Test split
TRAIN_END = "2023-12-31"  # Train on 2020-2023
TEST_START = "2024-01-01"  # Test on 2024-2025

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("PROPER TRAIN/TEST PREDICTION TEST")
print("=" * 70)
print(f"\nHorizon: {HORIZON} minutes")
print(f"Train: up to {TRAIN_END}")
print(f"Test: from {TEST_START}")
print("\nLoading data...")

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    exit(1)
ohlcv = pd.read_parquet(ohlcv_path)

# Load state vectors (for similarity search)
state_path = Path("data/outcomes/BTCUSDT_1m_outcomes.parquet")
if not state_path.exists():
    print(f"ERROR: Outcomes file not found: {state_path}")
    exit(1)
outcome_df = pd.read_parquet(state_path)

# Load regimes
regime_path = Path("data/regimes/BTCUSDT_1m_regimes.parquet")
if not regime_path.exists():
    print(f"ERROR: Regime file not found: {regime_path}")
    exit(1)
regime_df = pd.read_parquet(regime_path)

print(f"OHLCV: {len(ohlcv):,} candles")
print(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")

# =============================================================================
# SPLIT DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: SPLIT DATA")
print("=" * 70)

# Split OHLCV
train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
test_ohlcv = ohlcv[ohlcv.index >= TEST_START]

# Split outcomes
train_outcomes = outcome_df[outcome_df.index <= TRAIN_END]
test_outcomes = outcome_df[outcome_df.index >= TEST_START]

# Split regimes
train_regimes = regime_df[regime_df.index <= TRAIN_END]
test_regimes = regime_df[regime_df.index >= TEST_START]

print(f"\nTRAIN: {len(train_ohlcv):,} candles ({train_ohlcv.index.min()} to {train_ohlcv.index.max()})")
print(f"TEST:  {len(test_ohlcv):,} candles ({test_ohlcv.index.min()} to {test_ohlcv.index.max()})")

# =============================================================================
# STEP 2: COMPUTE EXPANSION LABELS (TRAIN ONLY)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: COMPUTE EXPANSION LABELS (TRAIN DATA ONLY)")
print("=" * 70)

from trade_system.expansion import ExpansionLabeler, ExpansionConfig

# Compute target from TRAIN data only
close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

print(f"Computing move distribution for H={HORIZON} on TRAIN data...")
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

print(f"Target (median from TRAIN): {target_pct*10000:.1f} bps")
print(f"Invalidation: {invalidation_pct*10000:.1f} bps")

# Create expansion labels for TRAIN data
exp_config = ExpansionConfig(
    horizon=HORIZON,
    target_pct=target_pct,
    invalidation_pct=invalidation_pct,
)

print("\nLabeling TRAIN data...")
labeler = ExpansionLabeler(train_ohlcv)
train_expansion = labeler.label(exp_config, show_progress=True)

long_col = f'long_expansion_{HORIZON}m'
short_col = f'short_expansion_{HORIZON}m'

train_long_rate = train_expansion[long_col].mean()
train_short_rate = train_expansion[short_col].mean()
print(f"\nTRAIN expansion rate: LONG={train_long_rate*100:.1f}%, SHORT={train_short_rate*100:.1f}%")

# =============================================================================
# STEP 3: BUILD SIMILARITY INDEX (TRAIN ONLY)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: BUILD SIMILARITY INDEX (TRAIN DATA ONLY)")
print("=" * 70)

from trade_system.similarity import SimilarityEngine

# Merge train data
train_common = train_outcomes.index.intersection(train_regimes.index).intersection(train_expansion.index)
train_merged = train_outcomes.loc[train_common].copy()
train_merged['regime'] = train_regimes.loc[train_common, 'regime']
train_merged[long_col] = train_expansion.loc[train_common, long_col]
train_merged[short_col] = train_expansion.loc[train_common, short_col]

print(f"Train data for index: {len(train_merged):,} rows")

# Build similarity engine on TRAIN data only
# Create regime df without overlap
train_regime_only = train_regimes.loc[train_common, ['regime']].copy()

print("\nBuilding FAISS index on TRAIN data...")
sim_engine = SimilarityEngine(
    outcome_df=train_merged.drop(columns=['regime']),  # Remove regime to avoid overlap
    regime_df=train_regime_only,
    k=K_NEIGHBORS,
    backend="faiss",
)

# =============================================================================
# STEP 4: DETECT TRIGGERS (TEST DATA)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: DETECT TRIGGERS (TEST DATA)")
print("=" * 70)

from trade_system.triggers import MicroTriggerDetector, TriggerConfig

trigger_config = TriggerConfig(
    engulfing_min_body_ratio=0.6,
    pin_bar_wick_ratio=2.0,
    volume_spike_multiplier=2.0,
    min_candle_range_bps=5.0,
)

detector = MicroTriggerDetector(test_ohlcv, trigger_config)
test_triggers = detector.detect_all(show_progress=True)

# Focus on Inside Bar (best performer)
insidebar_long = test_triggers[test_triggers['trigger_insidebar_long'] == 1].index
insidebar_short = test_triggers[test_triggers['trigger_insidebar_short'] == 1].index

print(f"\nInside Bar LONG signals in TEST: {len(insidebar_long):,}")
print(f"Inside Bar SHORT signals in TEST: {len(insidebar_short):,}")

# =============================================================================
# STEP 5: MAKE PREDICTIONS (Using TRAIN index)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: MAKE PREDICTIONS ON TEST DATA")
print("=" * 70)
print("For each trigger, query similarity engine (TRAIN only) to predict expansion")

STATE_COLUMNS = [
    "ema50_slope_z", "ema200_slope_z", "trend_alignment",
    "return_5m_z", "return_15m_z", "rsi_z",
    "atr_percentile", "volume_z", "vwap_distance_z", "range_position",
]

# We need test outcomes for state vectors
test_common = test_outcomes.index.intersection(test_regimes.index)
test_merged = test_outcomes.loc[test_common].copy()
test_merged['regime'] = test_regimes.loc[test_common, 'regime']

# Filter to signals that have state data
insidebar_long = [t for t in insidebar_long if t in test_merged.index]
insidebar_short = [t for t in insidebar_short if t in test_merged.index]

print(f"Signals with state data: LONG={len(insidebar_long):,}, SHORT={len(insidebar_short):,}")

# Sample if too many (for speed)
MAX_SAMPLES = 5000
if len(insidebar_long) > MAX_SAMPLES:
    insidebar_long = list(np.random.choice(insidebar_long, MAX_SAMPLES, replace=False))
if len(insidebar_short) > MAX_SAMPLES:
    insidebar_short = list(np.random.choice(insidebar_short, MAX_SAMPLES, replace=False))

print(f"Testing: LONG={len(insidebar_long):,}, SHORT={len(insidebar_short):,}")

# Make predictions
predictions = []
errors = []

print("\nPredicting LONG signals...")
for ts in tqdm(insidebar_long):
    try:
        state = test_merged.loc[ts]
        regime = state['regime']

        # Query similarity engine (built on TRAIN only)
        result = sim_engine.query(
            current_state=state,
            regime=regime,
            horizon=HORIZON,
        )

        if result.get('status') == 'OK':
            # Get neighbor indices (these are timestamps from train data)
            neighbor_indices = result.get('_neighbor_indices', [])

            if len(neighbor_indices) > 0:
                # Convert to pandas timestamps if needed and filter valid ones
                valid_indices = [idx for idx in neighbor_indices if idx in train_merged.index]

                if len(valid_indices) > 0:
                    neighbor_expansions = train_merged.loc[valid_indices, long_col]
                    predicted_expansion = neighbor_expansions.mean()
                else:
                    # Use long stats from result as fallback
                    long_stats = result.get('long', {})
                    win_rate = long_stats.get('win_rate', train_long_rate)
                    predicted_expansion = win_rate
            else:
                predicted_expansion = train_long_rate  # fallback to base rate

            predictions.append({
                'timestamp': ts,
                'direction': 'LONG',
                'predicted_expansion': predicted_expansion,
                'regime': regime,
                'distance_mean': result.get('distance_mean', 0),
            })
    except Exception as e:
        errors.append(str(e)[:50])

print(f"LONG errors: {len(errors)}, unique: {len(set(errors))}")
if errors:
    print(f"Sample errors: {list(set(errors))[:3]}")

print("\nPredicting SHORT signals...")
for ts in tqdm(insidebar_short):
    try:
        state = test_merged.loc[ts]
        regime = state['regime']

        result = sim_engine.query(
            current_state=state,
            regime=regime,
            horizon=HORIZON,
        )

        if result.get('status') == 'OK':
            neighbor_indices = result.get('_neighbor_indices', [])

            if len(neighbor_indices) > 0:
                valid_indices = [idx for idx in neighbor_indices if idx in train_merged.index]

                if len(valid_indices) > 0:
                    neighbor_expansions = train_merged.loc[valid_indices, short_col]
                    predicted_expansion = neighbor_expansions.mean()
                else:
                    short_stats = result.get('short', {})
                    win_rate = short_stats.get('win_rate', train_short_rate)
                    predicted_expansion = win_rate
            else:
                predicted_expansion = train_short_rate

            predictions.append({
                'timestamp': ts,
                'direction': 'SHORT',
                'predicted_expansion': predicted_expansion,
                'regime': regime,
                'distance_mean': result.get('distance_mean', 0),
            })
    except Exception as e:
        errors.append(str(e)[:50])

print(f"SHORT errors: {len(errors)}, unique: {len(set(errors))}")
if errors:
    print(f"Sample errors: {list(set(errors))[:3]}")

predictions_df = pd.DataFrame(predictions)
print(f"\nTotal predictions: {len(predictions_df):,}")

if len(predictions_df) == 0:
    print("\nNo predictions made! Debugging info:")
    # Test one sample
    ts = insidebar_long[0] if insidebar_long else insidebar_short[0]
    print(f"  Sample timestamp: {ts}")
    print(f"  In test_merged: {ts in test_merged.index}")
    if ts in test_merged.index:
        state = test_merged.loc[ts]
        regime = state['regime']
        print(f"  Regime: {regime}")
        result = sim_engine.query(current_state=state, regime=regime, horizon=HORIZON)
        print(f"  Result status: {result.get('status')}")
        print(f"  Neighbor indices count: {len(result.get('_neighbor_indices', []))}")
        if result.get('_neighbor_indices'):
            print(f"  First few indices: {result.get('_neighbor_indices')[:3]}")
    exit(0)

# =============================================================================
# STEP 6: COMPUTE ACTUAL OUTCOMES (TEST DATA)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: COMPUTE ACTUAL OUTCOMES")
print("=" * 70)

# Create expansion labels for TEST data to get actual outcomes
print("Labeling TEST data for actual outcomes...")
test_labeler = ExpansionLabeler(test_ohlcv)
test_expansion = test_labeler.label(exp_config, show_progress=True)

# Merge actual outcomes
predictions_df['actual_expansion'] = predictions_df.apply(
    lambda row: test_expansion.loc[row['timestamp'], long_col if row['direction'] == 'LONG' else short_col]
    if row['timestamp'] in test_expansion.index else np.nan,
    axis=1
)

# Drop rows without actual outcome
predictions_df = predictions_df.dropna(subset=['actual_expansion'])
print(f"Predictions with actual outcomes: {len(predictions_df):,}")

# =============================================================================
# STEP 7: MEASURE PREDICTION ACCURACY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: PREDICTION ACCURACY")
print("=" * 70)

# Test different thresholds
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

print(f"\n{'Threshold':<12} {'Signals':>10} {'Predicted':>12} {'Actual':>12} {'Accuracy':>12}")
print("-" * 60)

for thresh in thresholds:
    # Filter predictions above threshold
    filtered = predictions_df[predictions_df['predicted_expansion'] >= thresh]

    if len(filtered) > 0:
        predicted_rate = filtered['predicted_expansion'].mean()
        actual_rate = filtered['actual_expansion'].mean()
        accuracy = (filtered['actual_expansion'] == 1).mean()

        print(f">= {thresh*100:.0f}%{' ':7} {len(filtered):>10,} {predicted_rate*100:>11.1f}% {actual_rate*100:>11.1f}% {accuracy*100:>11.1f}%")

# =============================================================================
# STEP 8: PROFITABILITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: PROFITABILITY ANALYSIS")
print("=" * 70)

target_bps = target_pct * 10000
fee_bps = ROUND_TRIP_FEE_BPS

print(f"\nTarget: {target_bps:.1f} bps")
print(f"Stop: {target_bps/2:.1f} bps")
print(f"Fees: {fee_bps} bps")

# Best threshold (>50% predicted)
best_filter = predictions_df[predictions_df['predicted_expansion'] >= 0.50]

if len(best_filter) > 0:
    actual_win_rate = best_filter['actual_expansion'].mean()

    # EV calculation
    win_payout = target_bps - fee_bps
    lose_payout = -(target_bps / 2) - fee_bps
    ev_bps = actual_win_rate * win_payout + (1 - actual_win_rate) * lose_payout

    print(f"\nWhen predicted >= 50%:")
    print(f"  Signals: {len(best_filter):,}")
    print(f"  Actual win rate: {actual_win_rate*100:.1f}%")
    print(f"  EV per trade: {ev_bps:+.1f} bps")

    if ev_bps > 0:
        print(f"\n  STATUS: PROFITABLE!")

        # Estimate daily trades and returns
        test_days = (test_ohlcv.index.max() - test_ohlcv.index.min()).days
        trades_per_day = len(best_filter) / test_days
        daily_ev = trades_per_day * ev_bps / 10000 * 100  # as percentage

        print(f"  Trades per day: {trades_per_day:.1f}")
        print(f"  Daily return: +{daily_ev:.2f}%")
    else:
        print(f"\n  STATUS: NOT PROFITABLE")
        print(f"  Need higher win rate or lower fees")
else:
    print("\nNo signals with predicted expansion >= 50%")

# =============================================================================
# STEP 9: BY DIRECTION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: RESULTS BY DIRECTION")
print("=" * 70)

for direction in ['LONG', 'SHORT']:
    dir_df = predictions_df[predictions_df['direction'] == direction]
    dir_filtered = dir_df[dir_df['predicted_expansion'] >= 0.50]

    if len(dir_filtered) > 0:
        actual_rate = dir_filtered['actual_expansion'].mean()
        ev_bps = actual_rate * win_payout + (1 - actual_rate) * lose_payout

        print(f"\n{direction}:")
        print(f"  Signals (predicted >= 50%): {len(dir_filtered):,}")
        print(f"  Actual win rate: {actual_rate*100:.1f}%")
        print(f"  EV per trade: {ev_bps:+.1f} bps")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
TRAIN/TEST SPLIT RESULTS:
  Train period: up to {TRAIN_END}
  Test period: from {TEST_START}

PREDICTION PERFORMANCE:
  - Used similarity engine trained on PAST data only
  - Predicted expansion rate for each trigger
  - Compared to ACTUAL outcomes in test period

KEY FINDING:
""")

if len(best_filter) > 0 and ev_bps > 0:
    print(f"  The system IS PREDICTIVE and PROFITABLE!")
    print(f"  Predicted win rate: {best_filter['predicted_expansion'].mean()*100:.1f}%")
    print(f"  Actual win rate: {actual_win_rate*100:.1f}%")
    print(f"  EV per trade: {ev_bps:+.1f} bps")
elif len(best_filter) > 0:
    print(f"  The system is PREDICTIVE but NOT profitable enough.")
    print(f"  Actual win rate: {actual_win_rate*100:.1f}% (need >~58% for profit)")
else:
    print(f"  Insufficient high-confidence predictions.")
