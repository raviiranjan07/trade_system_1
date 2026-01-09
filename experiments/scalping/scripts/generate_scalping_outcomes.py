#!/usr/bin/env python3
"""
Generate H=1 and H=2 outcomes from local OHLCV parquet files.
No database connection needed.

Usage:
    python -m experiments.scalping.generate_scalping_outcomes
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trade_system.config import get_config
from trade_system.outcomes.outcome_labeler import label_outcomes


def main():
    print("=" * 60)
    print("   GENERATING H=1, H=2 OUTCOMES FROM LOCAL DATA")
    print("=" * 60)

    config = get_config()
    pair = config.get("data.pair", "BTCUSDT")
    timeframe = config.get("data.timeframe", "1m")

    base_dir = Path(config.get("paths.data_dir", "data"))

    # Load state vectors (local)
    state_path = base_dir / "state_vectors" / f"{pair}_{timeframe}_state.parquet"
    print(f"\nLoading state vectors from: {state_path}")
    state_df = pd.read_parquet(state_path)
    print(f"  Loaded {len(state_df):,} rows")

    # Load OHLCV from local parquet (NOT database)
    ohlcv_files = sorted((base_dir / "ohlcv").glob(f"{pair}_*.parquet"))
    if not ohlcv_files:
        print("ERROR: No OHLCV parquet files found!")
        return

    ohlcv_path = ohlcv_files[-1]
    print(f"\nLoading OHLCV from: {ohlcv_path}")
    ohlcv_df = pd.read_parquet(ohlcv_path)
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
    print(f"  Loaded {len(ohlcv_df):,} bars")

    # Get close prices aligned with state vectors
    state_df.index = pd.to_datetime(state_df.index)
    common_idx = state_df.index.intersection(ohlcv_df.index)
    close_prices = ohlcv_df.loc[common_idx, "close"]
    state_df = state_df.loc[common_idx]

    print(f"\nAligned data: {len(common_idx):,} rows")

    # Label outcomes with H=1, H=2 (and others from config)
    print("\nComputing outcomes for horizons:", config.get("outcomes.horizons"))
    start_time = time.time()

    outcome_df = label_outcomes(
        state_df=state_df,
        price_series=close_prices,
        pair=pair,
        timeframe=timeframe
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Outcome shape: {outcome_df.shape}")
    print("\nColumns:", list(outcome_df.columns))

    # Show sample
    print("\nSample outcomes:")
    print(outcome_df.head())


if __name__ == "__main__":
    main()
