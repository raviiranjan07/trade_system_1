"""
Debug script to check what similarity engine returns with FAISS backend.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from trade_system.similarity import SimilarityEngine
from trade_system.config import get_config

# Load data (same as grid search)
config = get_config()
data_dir = Path(config.get("paths.data_dir", "data"))
pair = "BTCUSDT"

outcome_path = sorted((data_dir / "outcomes").glob(f"{pair}_*.parquet"))[-1]
regime_path = sorted((data_dir / "regimes").glob(f"{pair}_*.parquet"))[-1]

print(f"Loading from: {outcome_path}")

outcome_df = pd.read_parquet(outcome_path)
outcome_df.index = pd.to_datetime(outcome_df.index)

# Use same sample as grid search
SAMPLE_SIZE = 500_000
if len(outcome_df) > SAMPLE_SIZE:
    outcome_df = outcome_df.iloc[-SAMPLE_SIZE:]

regime_df = pd.read_parquet(regime_path)
regime_df.index = pd.to_datetime(regime_df.index)
regime_df = regime_df.loc[regime_df.index.isin(outcome_df.index)]

# Split like grid search
train_ratio = 0.7
split_idx = int(len(outcome_df) * train_ratio)
train_outcomes = outcome_df.iloc[:split_idx]
test_outcomes = outcome_df.iloc[split_idx:]

print(f"Train: {len(train_outcomes):,} rows")
print(f"Test: {len(test_outcomes):,} rows")

# Check if short columns exist
print(f"\nColumns with 'short': {[c for c in outcome_df.columns if 'short' in c]}")
print(f"Columns with 'long': {[c for c in outcome_df.columns if 'long' in c]}")

# Build similarity engine with FAISS (like grid search)
print("\nBuilding FAISS similarity engine...")
similarity_engine = SimilarityEngine(
    outcome_df=train_outcomes,
    regime_df=regime_df,
    k=100,
    backend="faiss",
    faiss_nlist=50,
    faiss_nprobe=5,
    use_gpu=False
)

# Test query
test_idx = len(test_outcomes) // 2  # Middle of test set
test_state = test_outcomes.iloc[test_idx]
test_ts = test_outcomes.index[test_idx]
test_regime = regime_df.loc[test_ts, "regime"]

print(f"\nQuery at {test_ts}, regime={test_regime}")

result = similarity_engine.query(
    current_state=test_state,
    regime=test_regime,
    horizon=3,
    max_timestamp=test_ts
)

print(f"\n=== SIMILARITY RESULT ===")
print(f"Status: {result.get('status')}")
print(f"Neighbors: {result.get('neighbors')}")
print(f"Distance mean: {result.get('distance_mean'):.4f}")

print(f"\n--- TOP LEVEL (backward compat) ---")
print(f"mean_mfe: {result.get('mean_mfe')}")
print(f"expectancy: {result.get('expectancy')}")

print(f"\n--- LONG STATS ---")
long_stats = result.get('long')
if long_stats:
    print(f"  mean_mfe: {long_stats.get('mean_mfe')}")
    print(f"  expectancy: {long_stats.get('expectancy')}")
else:
    print("  NONE!")

print(f"\n--- SHORT STATS ---")
short_stats = result.get('short')
if short_stats:
    print(f"  mean_mfe: {short_stats.get('mean_mfe')}")
    print(f"  expectancy: {short_stats.get('expectancy')}")
else:
    print("  NONE!")

# Test multiple random states
print(f"\n=== TESTING 20 RANDOM STATES ===")
import random
random.seed(42)
test_indices = random.sample(range(len(test_outcomes)), 20)

long_wins = 0
short_wins = 0
ties = 0

for idx in test_indices:
    state = test_outcomes.iloc[idx]
    ts = test_outcomes.index[idx]
    if ts not in regime_df.index:
        continue
    regime = regime_df.loc[ts, "regime"]

    result = similarity_engine.query(
        current_state=state,
        regime=regime,
        horizon=3,
        max_timestamp=ts
    )

    if result.get('status') != 'OK':
        continue

    long_stats = result.get('long')
    short_stats = result.get('short')

    if long_stats and short_stats:
        long_mfe = long_stats.get('mean_mfe', 0)
        short_mfe = short_stats.get('mean_mfe', 0)

        if long_mfe > short_mfe:
            long_wins += 1
            winner = "LONG"
        elif short_mfe > long_mfe:
            short_wins += 1
            winner = "SHORT"
        else:
            ties += 1
            winner = "TIE"

        print(f"  [{idx}] L_mfe={long_mfe:.5f} S_mfe={short_mfe:.5f} → {winner}")
    else:
        print(f"  [{idx}] Missing stats! long={long_stats is not None}, short={short_stats is not None}")

print(f"\n=== SUMMARY ===")
print(f"LONG wins: {long_wins}")
print(f"SHORT wins: {short_wins}")
print(f"Ties: {ties}")

if short_wins == 0:
    print("\n*** WARNING: SHORT never won! This explains 0 short trades. ***")
