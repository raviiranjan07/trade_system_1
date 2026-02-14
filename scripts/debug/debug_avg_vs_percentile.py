"""
Compare AVERAGE vs 75th PERCENTILE thresholds.
Run: .venv/Scripts/python.exe debug_avg_vs_percentile.py

Modify the 'horizon' variable below to test different horizons.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
# CHANGE THIS TO TEST DIFFERENT HORIZONS
# =============================================================
horizon = 5  # Try: 3, 5, 10, 15, 20
# =============================================================

# Load OHLCV data
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    exit(1)

print(f"Loading OHLCV from {ohlcv_path}...")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"Loaded {len(ohlcv):,} candles")
print()

close = ohlcv['close'].values
high = ohlcv['high'].values
low = ohlcv['low'].values
n = len(ohlcv)

# Compute max moves for the horizon
print(f"Computing moves for H={horizon}...")
moves = []
for i in range(n - horizon):
    entry = close[i]
    future_high = np.max(high[i+1:i+1+horizon])
    future_low = np.min(low[i+1:i+1+horizon])
    up_move = (future_high - entry) / entry
    down_move = (entry - future_low) / entry
    moves.append(up_move)
    moves.append(down_move)

moves = np.array(moves)
print(f"Total moves: {len(moves):,}")
print()

# Calculate different thresholds
average = np.mean(moves)
median = np.percentile(moves, 50)
pct_75 = np.percentile(moves, 75)
pct_90 = np.percentile(moves, 90)

print("=" * 60)
print(f"THRESHOLD COMPARISON FOR H={horizon}")
print("=" * 60)
print()
print("THRESHOLD VALUES:")
print(f"  Median (50th pct):  {median*10000:.1f} bps")
print(f"  Average (mean):     {average*10000:.1f} bps")
print(f"  75th percentile:    {pct_75*10000:.1f} bps")
print(f"  90th percentile:    {pct_90*10000:.1f} bps")
print()

# Count expansions with each threshold
print("IF WE USE EACH AS EXPANSION THRESHOLD:")
print("-" * 60)

for name, threshold in [("Median", median), ("Average", average), ("75th pct", pct_75), ("90th pct", pct_90)]:
    count = np.sum(moves >= threshold)
    pct = count / len(moves) * 100
    print(f"  {name:<12} ({threshold*10000:>5.1f} bps):  {count:>10,} moves = {pct:>5.1f}% labeled as expansion")

print()
print("=" * 60)
print("SAMPLE MOVES FROM YOUR DATA")
print("=" * 60)

sorted_moves_bps = np.sort(moves) * 10000

print()
print(f"Smallest moves:     {sorted_moves_bps[:5].round(1)}")
print(f"Around median:      {sorted_moves_bps[len(moves)//2-2 : len(moves)//2+3].round(1)}")
print(f"Around 75th pct:    {sorted_moves_bps[int(len(moves)*0.75)-2 : int(len(moves)*0.75)+3].round(1)}")
print(f"Around 90th pct:    {sorted_moves_bps[int(len(moves)*0.90)-2 : int(len(moves)*0.90)+3].round(1)}")
print(f"Largest moves:      {sorted_moves_bps[-5:].round(1)}")

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)
print(f"""
For H={horizon}:
- Median = {median*10000:.1f} bps means HALF of all moves are smaller than this
- Average = {average*10000:.1f} bps is HIGHER than median because big outliers pull it up
- 75th pct = {pct_75*10000:.1f} bps means only TOP 25% of moves are bigger than this

Using AVERAGE: {np.sum(moves >= average)/len(moves)*100:.1f}% of moves labeled as expansion
Using 75th PCT: {np.sum(moves >= pct_75)/len(moves)*100:.1f}% of moves labeled as expansion

The 75th percentile is more STABLE and gives you only the EXCEPTIONAL moves.
""")
