"""
One-time: Create 15-min OHLCV parquet file for fast loading
"""

import pandas as pd
from pathlib import Path

print("Creating 15-min OHLCV parquet file (one-time process)...")

# Load 1-min data
print("Loading 1-min data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"Loaded: {len(ohlcv_1m):,} rows")

# Resample to 15-min
print("Resampling to 15-min...")
ohlcv_15m = ohlcv_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"15-min data: {len(ohlcv_15m):,} rows")

# Save
output_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
ohlcv_15m.to_parquet(output_path)

print(f"\n[OK] Saved: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
print("\nNow you can use this file for fast visualization!")
