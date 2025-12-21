"""
Regime Labeling Pipeline

Classifies market regimes from state vectors.
Can be run standalone or via the unified orchestrator.

Usage:
    python -m regime.run_regime_labeling
    python -m regime.run_regime_labeling --pair ETHUSDT
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from regime.regime_labeler import label_regime_row, smooth_regime


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Label market regimes from state vectors")
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
    smoothing_window = config.get("regime.smoothing_window", 30)

    # Paths
    base_dir = Path(config.get("paths.data_dir", "data"))
    state_path = base_dir / config.get("paths.state_vectors_dir", "state_vectors") / f"{pair}_{timeframe}_state.parquet"
    output_dir = base_dir / config.get("paths.regimes_dir", "regimes")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pair}_{timeframe}_regimes.parquet"

    print(f"Labeling regimes for {pair} ({timeframe})")

    # 1. Load state vectors
    df = pd.read_parquet(state_path)

    # 2. Label regimes (row-wise, online-safe)
    df["regime_raw"] = df.apply(label_regime_row, axis=1)

    # 3. Smooth regimes
    df["regime"] = smooth_regime(df["regime_raw"], window=smoothing_window)

    # 4. Save
    df[["regime"]].to_parquet(output_path, engine="pyarrow")

    print("Regime labeling completed")
    print(df["regime"].value_counts())
