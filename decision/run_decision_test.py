"""
Decision Engine Test

Generates trading decisions based on similarity analysis.
Can be run standalone or via the unified orchestrator.

Usage:
    python -m decision.run_decision_test
    python -m decision.run_decision_test --pair ETHUSDT
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from similarity.similarity_engine import SimilarityEngine
from decision.decision_engine import DecisionEngine


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test decision engine with trading signal generation")
    parser.add_argument("-c", "--config", type=str, default=None, help="Config file path")
    parser.add_argument("-p", "--pair", type=str, default=None, help="Trading pair")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config
    config = get_config(args.config)

    # Get parameters
    pair = args.pair or config.get("data.pair", "BTCUSDT")
    timeframe = config.get("data.timeframe", "1m")
    k = config.get("similarity.k", 200)
    horizon = config.get("similarity.default_horizon", 30)
    capital = config.get("decision.capital", 10000)
    risk_per_trade = config.get("decision.risk_per_trade", 0.005)

    # Paths
    base_dir = Path(config.get("paths.data_dir", "data"))
    outcome_path = base_dir / config.get("paths.outcomes_dir", "outcomes") / f"{pair}_{timeframe}_outcomes.parquet"
    regime_path = base_dir / config.get("paths.regimes_dir", "regimes") / f"{pair}_{timeframe}_regimes.parquet"

    print(f"Running decision test for {pair} ({timeframe})")
    print(f"Capital: ${capital:,} | Risk: {risk_per_trade * 100:.1f}%")

    # Load data
    outcome_df = pd.read_parquet(outcome_path)
    regime_df = pd.read_parquet(regime_path)

    # Init engines
    similarity_engine = SimilarityEngine(
        outcome_df=outcome_df,
        regime_df=regime_df,
        k=k
    )

    decision_engine = DecisionEngine(
        capital=capital,
        risk_per_trade=risk_per_trade
    )

    # Current context
    current_state = outcome_df.iloc[-1]
    current_regime = regime_df.iloc[-1]["regime"]

    sim_result = similarity_engine.query(
        current_state=current_state,
        regime=current_regime,
        horizon=horizon
    )

    decision = decision_engine.decide(
        similarity_result=sim_result,
        regime=current_regime
    )

    print(f"\nCurrent Regime: {current_regime}")
    print("\nSimilarity Result:")
    for key, value in sim_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\nDecision:")
    for key, value in decision.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
