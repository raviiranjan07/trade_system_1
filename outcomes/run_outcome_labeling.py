"""
Outcome Labeling Pipeline

Computes MFE/MAE (Maximum Favorable/Adverse Excursion) for state vectors.
Can be run standalone or via the unified orchestrator.

Usage:
    python -m outcomes.run_outcome_labeling
    python -m outcomes.run_outcome_labeling --pair ETHUSDT
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.raw.ohlcv_loader import OHLCVLoader
from outcomes.outcome_labeler import label_outcomes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute outcome labels (MFE/MAE)")
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

    # Paths
    base_dir = Path(config.get("paths.data_dir", "data"))
    state_path = base_dir / config.get("paths.state_vectors_dir", "state_vectors") / f"{pair}_{timeframe}_state.parquet"

    print(f"Computing outcomes for {pair} ({timeframe})")

    # 1. Load state vectors
    state_df = pd.read_parquet(state_path)

    # 2. Load close prices
    loader = OHLCVLoader()
    ohlcv = loader.fetch_ohlcv(pair=pair)
    close_prices = ohlcv["close"].loc[state_df.index]

    # 3. Label outcomes
    outcome_df = label_outcomes(
        state_df=state_df,
        price_series=close_prices,
        pair=pair,
        timeframe=timeframe
    )

    print("Outcome labeling completed")
    print(outcome_df.head())
    print(outcome_df.describe())
