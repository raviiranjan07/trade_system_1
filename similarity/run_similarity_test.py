"""
Similarity Engine Test

Finds similar historical states using KNN.
Can be run standalone or via the unified orchestrator.

Usage:
    python -m similarity.run_similarity_test
    python -m similarity.run_similarity_test --pair ETHUSDT
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from similarity.similarity_engine import SimilarityEngine


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test similarity engine with KNN search")
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

    # Paths
    base_dir = Path(config.get("paths.data_dir", "data"))
    outcome_path = base_dir / config.get("paths.outcomes_dir", "outcomes") / f"{pair}_{timeframe}_outcomes.parquet"
    regime_path = base_dir / config.get("paths.regimes_dir", "regimes") / f"{pair}_{timeframe}_regimes.parquet"

    print(f"Running similarity test for {pair} ({timeframe})")

    # Load data
    outcome_df = pd.read_parquet(outcome_path)
    regime_df = pd.read_parquet(regime_path)

    # Init engine
    engine = SimilarityEngine(
        outcome_df=outcome_df,
        regime_df=regime_df,
        k=k
    )

    # Pick latest state
    current_state = outcome_df.iloc[-1]
    current_regime = regime_df.iloc[-1]["regime"]

    result = engine.query(
        current_state=current_state,
        regime=current_regime,
        horizon=horizon
    )

    print(f"Current Regime: {current_regime}")
    print("Similarity Result:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
