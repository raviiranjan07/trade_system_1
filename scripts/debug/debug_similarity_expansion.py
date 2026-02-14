"""
Debug script to test if SIMILARITY SEARCH can improve expansion rates.

Run: .venv/Scripts/python.exe debug_similarity_expansion.py

This answers: "Can similarity search find states with >55% expansion rate?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from trade_system.expansion import ExpansionLabeler, ExpansionConfig, ExpansionQueryEngine
from trade_system.similarity import SimilarityEngine

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZON = 30  # 30 minutes (longest available in outcome data)
NUM_SAMPLES = 500  # Number of random states to test
K_NEIGHBORS = 200  # Number of neighbors to find
MIN_NEIGHBORS = 50  # Minimum neighbors required

# Threshold settings (using Average for better target size)
EXPANSION_PERCENTILE = 0.50  # Use median for higher base rate
INVALIDATION_RATIO = 0.5

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

# Load outcomes
outcome_path = Path("data/outcomes/BTCUSDT_1m_outcomes.parquet")
if not outcome_path.exists():
    print(f"ERROR: Outcome file not found: {outcome_path}")
    exit(1)
outcome_df = pd.read_parquet(outcome_path)

# Load regimes
regime_path = Path("data/regimes/BTCUSDT_1m_regimes.parquet")
if not regime_path.exists():
    print(f"ERROR: Regime file not found: {regime_path}")
    exit(1)
regime_df = pd.read_parquet(regime_path)

print(f"OHLCV: {len(ohlcv):,} candles")
print(f"Outcomes: {len(outcome_df):,} rows")
print(f"Regimes: {len(regime_df):,} rows")

# =============================================================================
# COMPUTE THRESHOLDS
# =============================================================================
print(f"\nComputing thresholds for H={HORIZON}...")

close = ohlcv['close'].values
high = ohlcv['high'].values
low = ohlcv['low'].values
n = len(ohlcv)

moves = []
for i in range(n - HORIZON):
    entry = close[i]
    future_high = np.max(high[i+1:i+1+HORIZON])
    future_low = np.min(low[i+1:i+1+HORIZON])
    moves.append((future_high - entry) / entry)
    moves.append((entry - future_low) / entry)
moves = np.array(moves)

median_move = np.percentile(moves, 50)
target_pct = np.percentile(moves, EXPANSION_PERCENTILE * 100)
invalidation_pct = median_move * INVALIDATION_RATIO

print(f"Target ({EXPANSION_PERCENTILE*100:.0f}th pct): {target_pct*10000:.1f} bps")
print(f"Invalidation (50% of median): {invalidation_pct*10000:.1f} bps")

# =============================================================================
# CREATE EXPANSION LABELS
# =============================================================================
print(f"\nCreating expansion labels...")

config = ExpansionConfig(
    horizon=HORIZON,
    target_pct=target_pct,
    invalidation_pct=invalidation_pct,
)
labeler = ExpansionLabeler(ohlcv)
expansion_df = labeler.label(config, show_progress=True)

long_col = f'long_expansion_{HORIZON}m'
short_col = f'short_expansion_{HORIZON}m'

overall_long_rate = expansion_df[long_col].mean()
overall_short_rate = expansion_df[short_col].mean()
print(f"\nOverall expansion rate: LONG={overall_long_rate*100:.1f}%, SHORT={overall_short_rate*100:.1f}%")

# =============================================================================
# INITIALIZE EXPANSION QUERY ENGINE
# =============================================================================
print("\nInitializing ExpansionQueryEngine...")

engine = ExpansionQueryEngine(
    expansion_df=expansion_df,
    outcome_df=outcome_df,
    regime_df=regime_df,
    k=K_NEIGHBORS,
    backend="faiss",
    min_neighbors=MIN_NEIGHBORS,
)

# =============================================================================
# SAMPLE RANDOM STATES AND QUERY SIMILARITY
# =============================================================================
print(f"\nTesting {NUM_SAMPLES} random states with similarity search...")

# Get common index
common_idx = outcome_df.index.intersection(regime_df.index).intersection(expansion_df.index)

# Sample random timestamps (skip first/last 1000 to avoid edge effects)
valid_idx = common_idx[1000:-1000]
sample_idx = np.random.choice(len(valid_idx), size=min(NUM_SAMPLES, len(valid_idx)), replace=False)
sample_timestamps = valid_idx[sample_idx]

results = []
errors = {}
for ts in tqdm(sample_timestamps, desc="Querying"):
    try:
        # Get current state
        state = outcome_df.loc[ts]
        regime = regime_df.loc[ts, 'regime']

        # Query similar states (without temporal filter for now to test mechanism)
        result = engine.query(
            current_state=state,
            regime=regime,
            horizon=HORIZON,
            # max_timestamp=ts,  # Disabled to test basic mechanism
        )

        if result.status == "OK":
            results.append({
                'timestamp': ts,
                'regime': regime,
                'neighbors': result.neighbors,
                'distance_mean': result.distance_mean,
                'long_expansion_rate': result.long_expansion_rate,
                'short_expansion_rate': result.short_expansion_rate,
                'expansion_rate': result.expansion_rate,
                'direction': result.direction,
            })
        else:
            errors[result.status] = errors.get(result.status, 0) + 1
    except Exception as e:
        errors[str(e)[:50]] = errors.get(str(e)[:50], 0) + 1

print(f"\nQuery errors: {errors}")

results_df = pd.DataFrame(results)
print(f"\nSuccessful queries: {len(results_df)}")

if len(results_df) == 0:
    print("\nNo successful queries! Check error counts above.")
    print("This typically happens when:")
    print("  - max_timestamp filtering removes all neighbors")
    print("  - FAISS index has issues")
    print("  - expansion_df columns don't match expected format")
    exit(0)

# =============================================================================
# ANALYZE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SIMILARITY SEARCH RESULTS")
print("=" * 70)

print(f"\nExpansion Rate Distribution (from {len(results_df)} queries):")
print(f"  Min:    {results_df['expansion_rate'].min()*100:.1f}%")
print(f"  25th:   {results_df['expansion_rate'].quantile(0.25)*100:.1f}%")
print(f"  Median: {results_df['expansion_rate'].median()*100:.1f}%")
print(f"  75th:   {results_df['expansion_rate'].quantile(0.75)*100:.1f}%")
print(f"  Max:    {results_df['expansion_rate'].max()*100:.1f}%")

# Count high expansion rate states
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
print(f"\nStates with expansion_rate >= threshold:")
for thresh in thresholds:
    count = (results_df['expansion_rate'] >= thresh).sum()
    pct = count / len(results_df) * 100
    print(f"  >= {thresh*100:.0f}%: {count:>4} states ({pct:>5.1f}% of queries)")

# By regime
print(f"\nExpansion Rate by Regime (similarity-filtered):")
print("-" * 60)
regime_stats = results_df.groupby('regime').agg(
    count=('expansion_rate', 'count'),
    mean_rate=('expansion_rate', 'mean'),
    max_rate=('expansion_rate', 'max'),
    pct_above_55=('expansion_rate', lambda x: (x >= 0.55).mean() * 100),
).round(3)
regime_stats = regime_stats.sort_values('mean_rate', ascending=False)

print(f"{'Regime':<20} {'Count':>8} {'Mean Rate':>12} {'Max Rate':>10} {'% >= 55%':>10}")
print("-" * 60)
for regime, row in regime_stats.iterrows():
    print(f"{regime:<20} {row['count']:>8} {row['mean_rate']*100:>11.1f}% {row['max_rate']*100:>9.1f}% {row['pct_above_55']:>9.1f}%")

# =============================================================================
# COMPARE: REGIME ONLY vs SIMILARITY SEARCH
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: REGIME ONLY vs SIMILARITY SEARCH")
print("=" * 70)

# Regime-only rates
print("\nRegime-only expansion rates (baseline):")
regime_baseline = expansion_df.copy()
regime_baseline['regime'] = regime_df.loc[expansion_df.index.intersection(regime_df.index), 'regime']
baseline_stats = regime_baseline.groupby('regime').agg(
    long_rate=(long_col, 'mean'),
    short_rate=(short_col, 'mean'),
)

print(f"{'Regime':<20} {'Regime-Only Rate':>18} {'Similarity Rate':>18} {'Improvement':>12}")
print("-" * 70)
for regime in regime_stats.index:
    if regime in baseline_stats.index:
        baseline_rate = max(baseline_stats.loc[regime, 'long_rate'], baseline_stats.loc[regime, 'short_rate'])
        sim_rate = regime_stats.loc[regime, 'mean_rate']
        improvement = (sim_rate - baseline_rate) * 100
        print(f"{regime:<20} {baseline_rate*100:>17.1f}% {sim_rate*100:>17.1f}% {improvement:>+11.1f}pp")

# =============================================================================
# TRADEABLE SIGNALS
# =============================================================================
print("\n" + "=" * 70)
print("TRADEABLE SIGNALS (expansion_rate >= 55%)")
print("=" * 70)

tradeable = results_df[results_df['expansion_rate'] >= 0.55]
print(f"\nTotal tradeable signals: {len(tradeable)} out of {len(results_df)} ({len(tradeable)/len(results_df)*100:.1f}%)")

if len(tradeable) > 0:
    print(f"\nTradeable signals by regime:")
    trade_by_regime = tradeable.groupby('regime').size()
    for regime, count in trade_by_regime.items():
        print(f"  {regime}: {count} signals")

    print(f"\nTradeable signals by direction:")
    print(f"  LONG:  {(tradeable['direction'] == 'LONG').sum()}")
    print(f"  SHORT: {(tradeable['direction'] == 'SHORT').sum()}")

# =============================================================================
# INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

avg_sim_rate = results_df['expansion_rate'].mean()
pct_tradeable = len(tradeable) / len(results_df) * 100 if len(results_df) > 0 else 0

print(f"""
BASELINE (regime only):
  - Overall expansion rate: ~30%
  - Best regime (HIGH_VOL): ~25%
  - Result: NOT TRADEABLE (need >55%)

WITH SIMILARITY SEARCH:
  - Average expansion rate: {avg_sim_rate*100:.1f}%
  - States with rate >= 55%: {pct_tradeable:.1f}%
  - Max expansion rate found: {results_df['expansion_rate'].max()*100:.1f}%

CONCLUSION:
  {'Similarity search CAN find tradeable setups!' if pct_tradeable > 5 else 'Need more tuning to find tradeable setups.'}

NEXT STEPS:
  1. If tradeable signals exist: integrate into backtester
  2. If not: try different K values, state features, or horizons
""")
