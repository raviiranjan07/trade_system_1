"""S/R Data Exploration - Understanding 15min candle level behavior within 8 bars

No assumed parameters. Pure data discovery.

Questions:
1. How big are 15min candles? (high-low range)
2. How far apart are consecutive candle highs? Consecutive candle lows?
3. Within 8 bars, how often does price revisit a previous candle's high or low?
4. When price makes a new high, does the previous high act as support?
5. What's the natural clustering of price levels in an 8-bar window?

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_data_exploration.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

print("=" * 70)
print("S/R DATA EXPLORATION - 15min candles")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index
print(f"Loaded: {len(ohlcv)} bars")

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
open_ = ohlcv["open"].values
N = len(ohlcv)

# =========================================================================
print(f"\n{'='*70}")
print("Q1: How big are 15min candles? (high - low in bps)")
print(f"{'='*70}")

candle_range_bps = (high - low) / low * 10000
print(f"  Mean:   {np.mean(candle_range_bps):.1f} bps")
print(f"  Median: {np.median(candle_range_bps):.1f} bps")
print(f"  P10:    {np.percentile(candle_range_bps, 10):.1f} bps")
print(f"  P25:    {np.percentile(candle_range_bps, 25):.1f} bps")
print(f"  P75:    {np.percentile(candle_range_bps, 75):.1f} bps")
print(f"  P90:    {np.percentile(candle_range_bps, 90):.1f} bps")

# =========================================================================
print(f"\n{'='*70}")
print("Q2: Distance between consecutive candle highs and lows (in bps)")
print(f"{'='*70}")

high_diff_bps = np.abs(np.diff(high)) / high[:-1] * 10000
low_diff_bps = np.abs(np.diff(low)) / low[:-1] * 10000

print(f"\n  Consecutive HIGH difference:")
print(f"    Mean:   {np.mean(high_diff_bps):.1f} bps")
print(f"    Median: {np.median(high_diff_bps):.1f} bps")
print(f"    P10:    {np.percentile(high_diff_bps, 10):.1f} bps")
print(f"    P25:    {np.percentile(high_diff_bps, 25):.1f} bps")
print(f"    P75:    {np.percentile(high_diff_bps, 75):.1f} bps")
print(f"    P90:    {np.percentile(high_diff_bps, 90):.1f} bps")

print(f"\n  Consecutive LOW difference:")
print(f"    Mean:   {np.mean(low_diff_bps):.1f} bps")
print(f"    Median: {np.median(low_diff_bps):.1f} bps")
print(f"    P10:    {np.percentile(low_diff_bps, 10):.1f} bps")
print(f"    P25:    {np.percentile(low_diff_bps, 25):.1f} bps")
print(f"    P75:    {np.percentile(low_diff_bps, 75):.1f} bps")
print(f"    P90:    {np.percentile(low_diff_bps, 90):.1f} bps")

# =========================================================================
print(f"\n{'='*70}")
print("Q3: Within 8 bars, how often does price revisit a previous high or low?")
print(f"{'='*70}")
print("""
For each bar t, look at bars t-7 to t-1 (7 previous bars).
Check: does bar t's price range (low to high) overlap with any previous bar's high or low?
""")

revisit_high_count = 0  # bar t touches a previous bar's high
revisit_low_count = 0   # bar t touches a previous bar's low
total_checked = 0

for i in range(8, N):
    bar_high = high[i]
    bar_low = low[i]
    total_checked += 1

    # Check if current bar's range touches any of the 7 previous bars' highs
    for j in range(i-7, i):
        prev_high = high[j]
        # Current bar's range [bar_low, bar_high] passes through prev_high
        if bar_low <= prev_high <= bar_high:
            revisit_high_count += 1
            break  # count once per bar

    # Check if current bar's range touches any of the 7 previous bars' lows
    for j in range(i-7, i):
        prev_low = low[j]
        if bar_low <= prev_low <= bar_high:
            revisit_low_count += 1
            break

print(f"  Bars where price passes through a prev-7 high: {revisit_high_count}/{total_checked} ({revisit_high_count/total_checked*100:.1f}%)")
print(f"  Bars where price passes through a prev-7 low:  {revisit_low_count}/{total_checked} ({revisit_low_count/total_checked*100:.1f}%)")

# =========================================================================
print(f"\n{'='*70}")
print("Q4: New high then revisit previous high (your S/R pattern)")
print(f"{'='*70}")
print("""
Pattern: bar B makes a higher high than bar A.
Later bar C comes back down and touches bar A's high.
How often does this happen within 8 bars?

Also: when it happens, what does bar C do next?
  - Does price bounce UP from bar A's high (support held)?
  - Does price break DOWN through bar A's high (support failed)?
""")

# Pattern detection
pattern_support = []  # new high, then retest previous high as support
pattern_resist = []   # new low, then retest previous low as resistance

for i in range(8, N - 1):
    # SUPPORT PATTERN: new high breaks old high, then retest old high
    for j in range(i-7, i):
        prev_high = high[j]
        # Was prev_high broken by any bar between j+1 and i-1?
        broken = False
        for k in range(j+1, i):
            if high[k] > prev_high:
                broken = True
                break
        if not broken:
            continue
        # Is bar i retesting prev_high? (bar i's low touches prev_high)
        # Bar i's low should be near prev_high (touching from above = support test)
        if low[i] <= prev_high <= high[i] and close[i-1] > prev_high:
            # Price was above prev_high recently and now touching it
            pattern_support.append({
                "bar_idx": i,
                "level": prev_high,
                "level_bar": j,
                "bars_gap": i - j,
                "close_at_retest": close[i],
                "next_close": close[i+1] if i+1 < N else np.nan,
                "next_high": high[i+1] if i+1 < N else np.nan,
                "next_low": low[i+1] if i+1 < N else np.nan,
            })
            break  # one pattern per bar

    # RESISTANCE PATTERN: new low breaks old low, then retest old low
    for j in range(i-7, i):
        prev_low = low[j]
        broken = False
        for k in range(j+1, i):
            if low[k] < prev_low:
                broken = True
                break
        if not broken:
            continue
        if low[i] <= prev_low <= high[i] and close[i-1] < prev_low:
            pattern_resist.append({
                "bar_idx": i,
                "level": prev_low,
                "level_bar": j,
                "bars_gap": i - j,
                "close_at_retest": close[i],
                "next_close": close[i+1] if i+1 < N else np.nan,
                "next_high": high[i+1] if i+1 < N else np.nan,
                "next_low": low[i+1] if i+1 < N else np.nan,
            })
            break

    if i % 50000 == 0:
        print(f"  {i}/{N} scanned... ({len(pattern_support)} sup, {len(pattern_resist)} res)")

print(f"\n  SUPPORT pattern (old high becomes support): {len(pattern_support)} occurrences")
print(f"  RESISTANCE pattern (old low becomes resistance): {len(pattern_resist)} occurrences")
print(f"  Rate: support={len(pattern_support)/total_checked*100:.1f}%, resistance={len(pattern_resist)/total_checked*100:.1f}%")

# Analyze outcomes
for name, patterns in [("SUPPORT (old high)", pattern_support), ("RESISTANCE (old low)", pattern_resist)]:
    if len(patterns) == 0:
        continue
    df = pd.DataFrame(patterns)
    print(f"\n  --- {name} ---")
    print(f"  Count: {len(df)}")
    print(f"  Bars gap (level to retest): mean={df['bars_gap'].mean():.1f}, median={df['bars_gap'].median():.0f}")

    # Next bar behavior
    if name.startswith("SUPPORT"):
        # Support held = next bar goes up (next_close > close_at_retest)
        held = (df["next_close"] > df["close_at_retest"]).sum()
        failed = (df["next_close"] <= df["close_at_retest"]).sum()
        print(f"  Next bar UP (support held):   {held} ({held/len(df)*100:.1f}%)")
        print(f"  Next bar DOWN (support fail):  {failed} ({failed/len(df)*100:.1f}%)")
    else:
        # Resistance held = next bar goes down
        held = (df["next_close"] < df["close_at_retest"]).sum()
        failed = (df["next_close"] >= df["close_at_retest"]).sum()
        print(f"  Next bar DOWN (resist held):  {held} ({held/len(df)*100:.1f}%)")
        print(f"  Next bar UP (resist fail):     {failed} ({failed/len(df)*100:.1f}%)")


# =========================================================================
print(f"\n{'='*70}")
print("Q5: How many unique levels exist in a typical 8-bar window?")
print(f"{'='*70}")
print("""
For 1000 random 8-bar windows, collect all highs and lows (16 values).
Cluster them and count how many distinct price levels exist.
""")

np.random.seed(42)
sample_indices = np.random.randint(100, N - 10, size=1000)

cluster_counts = []
cluster_widths = []

for idx in sample_indices:
    # 8 bars: highs and lows
    prices = np.concatenate([high[idx:idx+8], low[idx:idx+8]])
    prices = np.sort(prices)
    ref_price = np.mean(prices)

    # Cluster: prices within X bps of each other are same level
    # Use 5bps as initial clustering (half a median candle)
    tol = ref_price * 5 / 10000
    clusters = []
    current_cluster = [prices[0]]
    for p in prices[1:]:
        if p - current_cluster[-1] <= tol:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)
    cluster_counts.append(len(clusters))

    # Width of clusters in bps
    for c in clusters:
        if len(c) > 1:
            width = (max(c) - min(c)) / min(c) * 10000
            cluster_widths.append(width)

print(f"  Distinct levels per 8-bar window (5bps clustering):")
print(f"    Mean:   {np.mean(cluster_counts):.1f}")
print(f"    Median: {np.median(cluster_counts):.0f}")
print(f"    Min:    {np.min(cluster_counts)}")
print(f"    Max:    {np.max(cluster_counts)}")

if cluster_widths:
    print(f"\n  Cluster width (when multiple prices at same level):")
    print(f"    Mean:   {np.mean(cluster_widths):.1f} bps")
    print(f"    Median: {np.median(cluster_widths):.1f} bps")

# Repeat with different tolerances
for tol_bps in [3, 5, 8, 10, 15, 20]:
    counts = []
    for idx in sample_indices:
        prices = np.concatenate([high[idx:idx+8], low[idx:idx+8]])
        prices = np.sort(prices)
        ref_price = np.mean(prices)
        tol = ref_price * tol_bps / 10000
        clusters = []
        current = [prices[0]]
        for p in prices[1:]:
            if p - current[-1] <= tol:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)
        counts.append(len(clusters))
    print(f"  {tol_bps:2d} bps tolerance: {np.mean(counts):.1f} levels (median {np.median(counts):.0f})")


print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
