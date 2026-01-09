"""
Debug script to trace decision engine behavior with FAISS data.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from trade_system.similarity import SimilarityEngine
from trade_system.decision import DecisionEngine
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

SAMPLE_SIZE = 500_000
if len(outcome_df) > SAMPLE_SIZE:
    outcome_df = outcome_df.iloc[-SAMPLE_SIZE:]

regime_df = pd.read_parquet(regime_path)
regime_df.index = pd.to_datetime(regime_df.index)
regime_df = regime_df.loc[regime_df.index.isin(outcome_df.index)]

train_ratio = 0.7
split_idx = int(len(outcome_df) * train_ratio)
train_outcomes = outcome_df.iloc[:split_idx]
test_outcomes = outcome_df.iloc[split_idx:]

print(f"Train: {len(train_outcomes):,}, Test: {len(test_outcomes):,}")

# Build similarity engine with FAISS
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

# Create decision engine with LOWER min_mfe to see more trades
decision_engine = DecisionEngine(
    capital=100,
    risk_per_trade=0.005,
    min_expectancy=0.0,
    max_distance=5.0,  # More permissive
    blocked_regimes=[],
    min_mfe=0.0003,  # Much lower to see what passes
    max_leverage=1.0,
    stop_floor=1e-4,
    use_stop_loss=True,
)

print(f"\nDecision Engine config:")
print(f"  min_mfe: {decision_engine.min_mfe}")
print(f"  use_stop_loss: {decision_engine.use_stop_loss}")

# Test 100 states for better statistics
import random
random.seed(42)
test_indices = random.sample(range(len(test_outcomes)), 100)

long_decisions = 0
short_decisions = 0
no_trade = 0

print(f"\n=== TESTING DECISION ENGINE ===")
for idx in test_indices:
    state = test_outcomes.iloc[idx]
    ts = test_outcomes.index[idx]
    if ts not in regime_df.index:
        continue
    regime = regime_df.loc[ts, "regime"]

    # Get similarity result
    sim_result = similarity_engine.query(
        current_state=state,
        regime=regime,
        horizon=3,
        max_timestamp=ts
    )

    if sim_result.get('status') != 'OK':
        print(f"  [{idx}] Status: {sim_result.get('status')}")
        continue

    long_stats = sim_result.get('long')
    short_stats = sim_result.get('short')

    # Call decision engine
    decision = decision_engine.decide(sim_result, regime)

    action = decision.get('action')
    direction = decision.get('direction', '-')
    reason = decision.get('reason', '-')

    if action == 'TRADE':
        if direction == 'LONG':
            long_decisions += 1
        else:
            short_decisions += 1

        long_mfe = long_stats.get('mean_mfe', 0) if long_stats else 0
        short_mfe = short_stats.get('mean_mfe', 0) if short_stats else 0
        winner_by_mfe = "LONG" if long_mfe > short_mfe else "SHORT"

        print(f"  [{idx}] TRADE {direction} | L_mfe={long_mfe:.5f} S_mfe={short_mfe:.5f} | MFE_winner={winner_by_mfe}")

        if direction != winner_by_mfe:
            print(f"       *** MISMATCH! Direction should be {winner_by_mfe} ***")
    else:
        no_trade += 1
        long_mfe = long_stats.get('mean_mfe', 0) if long_stats else 0
        short_mfe = short_stats.get('mean_mfe', 0) if short_stats else 0
        print(f"  [{idx}] NO_TRADE ({reason}) | L_mfe={long_mfe:.5f} S_mfe={short_mfe:.5f}")

print(f"\n=== DECISION SUMMARY ===")
print(f"LONG decisions: {long_decisions}")
print(f"SHORT decisions: {short_decisions}")
print(f"NO_TRADE: {no_trade}")

if short_decisions == 0 and long_decisions > 0:
    print("\n*** BUG CONFIRMED: Decision engine never selects SHORT! ***")
elif short_decisions > 0:
    print("\n*** Decision engine IS selecting SHORT - bug is elsewhere ***")
