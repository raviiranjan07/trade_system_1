"""
State Vector Pipeline

Builds state vectors from raw OHLCV data.
Can be run standalone or via the unified orchestrator.

Usage:
    python -m state.run_state_pipeline
    python -m state.run_state_pipeline --pair ETHUSDT --start 2023-01-01
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.raw.ohlcv_loader import OHLCVLoader
from data.validators.data_integrity import validate_ohlcv
from features.trend import compute_trend_features
from features.momentum import compute_momentum_features
from features.volatility import compute_volatility_features
from features.volume import compute_volume_features
from features.location import compute_location_features
from state.state_store import save_state_vectors_parquet
from state.normalizer import RollingNormalizer
from state.state_builder import build_state


def build_state_vectors_from_db(
    pair: str,
    start_time: str = None,
    end_time: str = None,
    normalization_window: int = 2000
) -> pd.DataFrame:
    """
    Build state vectors from database OHLCV data.

    Args:
        pair: Trading pair (e.g., "BTCUSDT")
        start_time: Start date (YYYY-MM-DD)
        end_time: End date (YYYY-MM-DD)
        normalization_window: Rolling window for z-score normalization

    Returns:
        DataFrame with state vectors
    """
    # 1. Fetch from DB
    loader = OHLCVLoader()
    df = loader.fetch_ohlcv(
        pair=pair,
        start_time=start_time,
        end_time=end_time
    )

    # 2. Validate raw data
    df = validate_ohlcv(df)

    # 3. Feature computation
    df = compute_trend_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_location_features(df)

    # 4. Normalization
    norm = RollingNormalizer(normalization_window)

    df["ema50_slope_z"] = norm.zscore(df["ema50_slope"])
    df["ema200_slope_z"] = norm.zscore(df["ema200_slope"])
    df["return_5m_z"] = norm.zscore(df["return_5m"])
    df["return_15m_z"] = norm.zscore(df["return_15m"])
    df["rsi_z"] = norm.zscore(df["rsi_14"])
    df["volume_z"] = norm.zscore(df["volume_raw"])
    df["vwap_distance_z"] = norm.zscore(df["vwap_distance"])
    df["atr_percentile"] = norm.percentile(df["atr_14"])

    # 5. Build states
    state_df = df.dropna().copy()
    state_df["state"] = state_df.apply(build_state, axis=1)

    return state_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build state vectors from OHLCV data")
    parser.add_argument("-c", "--config", type=str, default=None, help="Config file path")
    parser.add_argument("-p", "--pair", type=str, default=None, help="Trading pair")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config
    config = get_config(args.config)

    # Get parameters (CLI overrides config)
    pair = args.pair or config.get("data.pair", "BTCUSDT")
    start_time = args.start or config.get("data.start_date", "2023-01-01")
    end_time = args.end or config.get("data.end_date", "2023-03-01")
    timeframe = config.get("data.timeframe", "1m")
    norm_window = config.get("normalization.window", 2000)

    print(f"Building state vectors for {pair} ({start_time} to {end_time})")

    df_states = build_state_vectors_from_db(
        pair=pair,
        start_time=start_time,
        end_time=end_time,
        normalization_window=norm_window
    )

    print(df_states.head())
    print(df_states.describe())

    save_state_vectors_parquet(
        df=df_states,
        pair=pair,
        timeframe=timeframe
    )

    print(f"State vectors saved: {len(df_states)} rows")

