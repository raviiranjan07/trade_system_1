"""Build feature_cache.parquet from raw 15m OHLCV.

Computes 23 columns per bar matching the spec in
configs/data_cards/direction_feature_cache.yaml.

Run: python -m training.build_features  (from repo root, after pip install -e .)
Or:  python src/training/build_features.py

This is a "v2" build — formulas are documented here and will not match
the existing feature_cache.parquet bit-for-bit. The old model (ML_V1
v1) stays in production until v2 is validated and promoted.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from engine.signals.feature_lib import (
    range_position_rolling as range_position,
    rsi_rolling as rsi,
    sma_dist_pct,
)

RAW_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
OUT_PATH = Path("data/features/direction_prediction/feature_cache.parquet")


# rsi / range_position now come from engine.signals.feature_lib (imported
# above with local aliases) — the single source of truth for formulas.
# NOTE: range_position here is the ROLLING (training-cache) variant; live
# inference uses the INCLUSIVE variant. See feature_lib docstring.


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 23 feature columns from a raw OHLCV dataframe.

    Input must have columns: open, high, low, close, volume
    Index must be a DatetimeIndex.
    Output has 23 columns (see data card for full list).
    """
    out = pd.DataFrame(index=df.index)

    # --- OHLCV passthrough (5) ---
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    out["volume"] = df["volume"]

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # --- Direction features (6) ---
    out["rsi7"] = rsi(close, 7)
    out["rsi"] = rsi(close, 14)
    out["range_position"] = range_position(high, low, close, 20)
    sma200 = close.rolling(200).mean()
    out["sma200_dist_pct"] = sma_dist_pct(close.to_numpy(), 200)
    out["roc5"] = (close - close.shift(5)) / close.shift(5) * 10000
    out["momentum10"] = close - close.shift(10)

    # --- Magnitude features (8) ---
    tr = true_range(high, low, close)
    atr = tr.rolling(14).mean()
    out["atr_pct"] = atr / close * 100
    atr_bps = atr / close * 10000
    out["atr_percentile"] = atr_bps.rolling(500).rank(pct=True) * 100

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["ema_separation"] = (ema9 - ema21).abs() / close * 100

    out["range_bps"] = (high - low) / close * 10000
    out["body_bps"] = (close - open_).abs() / close * 10000

    high20 = high.rolling(20).max()
    out["dist_from_high20_pct"] = (close - high20) / high20 * 100

    out["hour_utc"] = df.index.hour.astype("int32")

    vol_ma20 = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_ma20.where(vol_ma20 > 0, np.nan)

    # --- Analysis features (4) ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["ema20_slope"] = ema20 - ema20.shift(5)
    out["ema50_dist_pct"] = (close - ema50) / ema50 * 100
    out["sma200_slope"] = sma200 - sma200.shift(5)
    out["range_position_50"] = range_position(high, low, close, 50)

    return out


def main() -> None:
    print(f"Loading {RAW_PATH}...")
    df = pd.read_parquet(RAW_PATH)

    # Normalize index timezone (strip if present) to match existing parquet
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"  Shape: {df.shape}, range: {df.index.min()} -> {df.index.max()}")

    print("Computing features...")
    out = build_features(df)

    print(f"  Shape: {out.shape}, columns: {len(out.columns)}")
    expected = 23
    if len(out.columns) != expected:
        raise RuntimeError(f"Expected {expected} columns, got {len(out.columns)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")

    # Sanity stats
    print("\nColumn summary:")
    print(f"  {'column':<25} {'nan_pct':>10} {'mean':>14} {'std':>14}")
    for c in out.columns:
        s = out[c]
        nan_pct = s.isna().mean() * 100
        try:
            mean = float(s.mean()); std = float(s.std())
            print(f"  {c:<25} {nan_pct:>9.1f}% {mean:>14.4f} {std:>14.4f}")
        except Exception:
            print(f"  {c:<25} {nan_pct:>9.1f}% (non-numeric)")


if __name__ == "__main__":
    main()
