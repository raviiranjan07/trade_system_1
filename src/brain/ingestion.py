"""Brain Pipeline — Step 1: Data Ingestion

Load raw OHLCV data, validate, and apply warm-up period.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_ohlcv(config: dict) -> pd.DataFrame:
    """Load OHLCV data based on config.

    Returns validated dataframe with columns: open, high, low, close, volume
    """
    ing = config["ingestion"]
    symbol = ing["symbol"]
    timeframe = ing["timeframe"]
    source = Path(ing["source"])

    # Build file path
    filename = f"{symbol}_{timeframe}_ohlcv.parquet"
    filepath = source / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading {filepath}...")
    df = pd.read_parquet(filepath)

    # Ensure timezone-naive index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Filter date range
    start = ing["date_range"]["start"]
    end = ing["date_range"]["end"]
    df = df.loc[start:end]

    print(f"  Loaded: {len(df)} bars, {df.index.min()} to {df.index.max()}")

    # Validate
    report = validate_ohlcv(df, config)

    # Apply warm-up
    warm_up = ing.get("warm_up", {}).get("min_bars", 50)
    if warm_up > 0:
        df = df.iloc[warm_up:]
        report["warm_up_skipped"] = warm_up
        print(f"  Warm-up: skipped first {warm_up} bars")

    print(f"  Final: {len(df)} bars, {df.index.min()} to {df.index.max()}")

    return df, report


def validate_ohlcv(df: pd.DataFrame, config: dict) -> dict:
    """Validate OHLCV data and return report."""
    report = {
        "total_bars": len(df),
        "date_start": str(df.index.min()),
        "date_end": str(df.index.max()),
        "gaps_found": 0,
        "gaps_filled": 0,
        "null_bars": 0,
    }

    val = config["ingestion"].get("validation", {})

    # Check for nulls
    null_count = df.isnull().any(axis=1).sum()
    report["null_bars"] = int(null_count)
    if null_count > 0:
        print(f"  WARNING: {null_count} bars with null values")

    # Check for gaps
    if val.get("check_gaps", True):
        timeframe = config["ingestion"]["timeframe"]
        expected_freq = _timeframe_to_freq(timeframe)
        if expected_freq:
            expected_index = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
            missing = expected_index.difference(df.index)
            report["gaps_found"] = len(missing)

            if len(missing) > 0:
                max_gap = val.get("max_gap_bars", 3)
                # Find consecutive gaps
                if len(missing) > 0:
                    print(f"  Gaps: {len(missing)} missing bars")

                    # Fill small gaps
                    fill_method = val.get("fill_method", "forward")
                    if fill_method == "forward" and len(missing) > 0:
                        df = df.reindex(expected_index, method="ffill")
                        report["gaps_filled"] = len(missing)
                        print(f"  Filled {len(missing)} gaps (forward fill)")

    # Basic sanity checks
    if (df["high"] < df["low"]).any():
        bad = (df["high"] < df["low"]).sum()
        print(f"  WARNING: {bad} bars where high < low")

    if (df["close"] <= 0).any():
        bad = (df["close"] <= 0).sum()
        print(f"  WARNING: {bad} bars with close <= 0")

    return report


def _timeframe_to_freq(timeframe: str) -> str:
    """Convert timeframe string to pandas frequency."""
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
    }
    return mapping.get(timeframe, None)
