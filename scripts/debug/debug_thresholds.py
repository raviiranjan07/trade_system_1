"""
Debug script to test ThresholdAnalyzer manually.
Run: .venv/Scripts/python.exe debug_thresholds.py
"""

import pandas as pd
from pathlib import Path

from trade_system.expansion import ThresholdAnalyzer

# Load OHLCV data
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    print("Run the pipeline first to generate OHLCV data.")
    exit(1)

print(f"Loading OHLCV from {ohlcv_path}...")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"Loaded {len(ohlcv):,} candles")
print(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")
print()

# Initialize analyzer
analyzer = ThresholdAnalyzer(ohlcv)

# Analyze multiple horizons
horizons = [3, 5, 10, 15]
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

for h in horizons:
    result = analyzer.analyze(
        horizon=h,
        expansion_percentile=0.75,  # Top 25% of moves
        invalidation_ratio=0.5,     # Half of median
    )

    print(f"\nHorizon = {h} bars")
    print("-" * 40)
    print(f"  Sample size:      {result.sample_size:,}")
    print(f"  Median move:      {result.median_move * 10000:.1f} bps ({result.median_move:.4%})")
    print(f"  75th percentile:  {result.p75_move * 10000:.1f} bps")
    print(f"  90th percentile:  {result.p90_move * 10000:.1f} bps")
    print()
    print(f"  >>> EXPANSION threshold:    {result.expansion_bps} bps ({result.expansion_pct:.4%})")
    print(f"  >>> INVALIDATION threshold: {result.invalidation_bps} bps ({result.invalidation_pct:.4%})")

# Show detailed distribution for H=5
print("\n" + "=" * 60)
print("DETAILED DISTRIBUTION (H=5)")
print("=" * 60)
stats = analyzer.get_distribution_stats(horizon=5)

print("\nUP moves (max high - close):")
for k, v in stats["up_moves"].items():
    if k != "count":
        print(f"  {k:>5}: {v * 10000:>6.1f} bps")

print("\nDOWN moves (close - min low):")
for k, v in stats["down_moves"].items():
    if k != "count":
        print(f"  {k:>5}: {v * 10000:>6.1f} bps")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
- Expansion threshold (75th pct): Only the top 25% of moves count as "real" expansion
- Invalidation threshold (50% of median): Conservative stop - if price goes against
  you by half a typical move, the expansion thesis is invalidated

Example for H=5:
  If expansion=25 bps, invalidation=10 bps, then:
  - LONG expansion = 1 if price hits +0.25% BEFORE hitting -0.10%
  - SHORT expansion = 1 if price hits -0.25% BEFORE hitting +0.10%
""")
