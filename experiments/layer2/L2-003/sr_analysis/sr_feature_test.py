"""S/R Feature Test - Compute all 9 features and test on historical data

Zone detection logic:
  Support = price area REVISITED after price moved UP away from it
  Resistance = price area REVISITED after price moved DOWN away from it

  For each pair of bars (i, j) with gap between them:
    - Find overlapping price range
    - If bars between went HIGHER than overlap = overlap is SUPPORT
    - If bars between went LOWER than overlap = overlap is RESISTANCE

9 Features:
  1. zone_width (bps)
  2. support_range_low, support_range_high (price range)
  3. resistance_range_low, resistance_range_high (price range)
  4. support_retest (count)
  5. resistance_retest (count)
  6. distance_to_support (bps)
  7. distance_to_resistance (bps)
  8. recovery_up (bps per bar)
  9. recovery_down (bps per bar)

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_feature_test.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7  # 8 bars: t-7 to t
MIN_GAP = 1   # at least 1 bar between pair

print("=" * 70)
print("S/R FEATURE TEST - All 9 Features")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
open_ = ohlcv["open"].values
N = len(ohlcv)

print(f"Loaded: {N} bars")


def find_zones(highs, lows, closes):
    """Find support and resistance zones from bar data.

    Returns: (support_low, support_high, resistance_low, resistance_high) or Nones
    """
    n = len(highs)
    support_overlaps = []
    resistance_overlaps = []

    # Check all pairs with at least 1 bar gap
    for i in range(n):
        for j in range(i + 2, n):
            # Overlap of bar i and bar j ranges
            overlap_low = max(lows[i], lows[j])
            overlap_high = min(highs[i], highs[j])

            if overlap_low > overlap_high:
                continue  # no overlap

            # Check bars between i and j
            between_highs = highs[i+1:j]
            between_lows = lows[i+1:j]

            if len(between_highs) == 0:
                continue

            max_between = np.max(between_highs)
            min_between = np.min(between_lows)

            # If bars between went ABOVE overlap = support
            if max_between > overlap_high:
                support_overlaps.append((overlap_low, overlap_high))

            # If bars between went BELOW overlap = resistance
            if min_between < overlap_low:
                resistance_overlaps.append((overlap_low, overlap_high))

    # Merge overlaps into ranges
    sup_low, sup_high = None, None
    res_low, res_high = None, None

    if support_overlaps:
        # Take the lowest support area (strongest floor)
        sup_low = min(s[0] for s in support_overlaps)
        sup_high = max(s[1] for s in support_overlaps if s[0] == sup_low)
        # Actually merge all nearby support overlaps
        # Sort by low price
        sorted_sup = sorted(support_overlaps, key=lambda x: x[0])
        # Take the cluster at the bottom
        sup_low = sorted_sup[0][0]
        sup_high = sorted_sup[0][1]
        for s in sorted_sup[1:]:
            if s[0] <= sup_high:  # overlapping with current cluster
                sup_high = max(sup_high, s[1])
            else:
                break  # gap, stop at lowest cluster

    if resistance_overlaps:
        # Take the highest resistance area (strongest ceiling)
        sorted_res = sorted(resistance_overlaps, key=lambda x: x[1], reverse=True)
        res_high = sorted_res[0][1]
        res_low = sorted_res[0][0]
        for r in sorted_res[1:]:
            if r[1] >= res_low:  # overlapping with current cluster
                res_low = min(res_low, r[0])
            else:
                break

    return sup_low, sup_high, res_low, res_high


def count_touches(highs, lows, range_low, range_high):
    """Count how many bars touch a price range."""
    count = 0
    for i in range(len(highs)):
        if lows[i] <= range_high and highs[i] >= range_low:
            count += 1
    return count


def compute_recovery_speeds(highs, lows, closes, sup_low, sup_high, res_low, res_high):
    """Compute recovery_up and recovery_down speeds."""
    n = len(highs)
    recovery_up = 0.0
    recovery_down = 0.0

    # recovery_up: when price was at support and moved up, how fast?
    # Find bars at support, then measure speed of move up
    if sup_low is not None and sup_high is not None:
        for i in range(n - 1):
            # Bar i touches support
            if lows[i] <= sup_high and lows[i] >= sup_low:
                # Find how far up it went in next bars
                for j in range(i + 1, n):
                    if closes[j] > sup_high:
                        bars = j - i
                        move = (closes[j] - sup_high) / sup_high * 10000
                        speed = move / bars
                        if speed > recovery_up:
                            recovery_up = speed
                        break

    # recovery_down: when price was at resistance and moved down, how fast?
    if res_low is not None and res_high is not None:
        for i in range(n - 1):
            if highs[i] >= res_low and highs[i] <= res_high:
                for j in range(i + 1, n):
                    if closes[j] < res_low:
                        bars = j - i
                        move = (res_low - closes[j]) / res_low * 10000
                        speed = move / bars
                        if speed > recovery_down:
                            recovery_down = speed
                        break

    return recovery_up, recovery_down


# =========================================================================
# Compute features for every bar
# =========================================================================
print("\nComputing S/R features for every bar...")

results = []

for t in range(LOOKBACK + 1, N - 3):
    # 8-bar window: t-7 to t
    w_high = high[t-LOOKBACK:t+1]
    w_low = low[t-LOOKBACK:t+1]
    w_close = close[t-LOOKBACK:t+1]

    cur_price = close[t]

    # Find zones
    sup_low, sup_high, res_low, res_high = find_zones(w_high, w_low, w_close)

    has_support = sup_low is not None
    has_resistance = res_low is not None
    has_zone = has_support and has_resistance and res_low > sup_high

    # Compute features
    row = {
        "date": ohlcv.index[t],
        "current_price": cur_price,
        "has_support": has_support,
        "has_resistance": has_resistance,
        "has_zone": has_zone,
    }

    if has_zone:
        row["zone_width"] = (res_low - sup_high) / sup_high * 10000
        row["support_range_low"] = sup_low
        row["support_range_high"] = sup_high
        row["resistance_range_low"] = res_low
        row["resistance_range_high"] = res_high
        row["support_retest"] = count_touches(w_high, w_low, sup_low, sup_high)
        row["resistance_retest"] = count_touches(w_high, w_low, res_low, res_high)
        row["distance_to_support"] = (cur_price - sup_high) / sup_high * 10000
        row["distance_to_resistance"] = (res_low - cur_price) / res_low * 10000

        rec_up, rec_down = compute_recovery_speeds(w_high, w_low, w_close, sup_low, sup_high, res_low, res_high)
        row["recovery_up"] = rec_up
        row["recovery_down"] = rec_down
    else:
        row["zone_width"] = np.nan
        row["support_range_low"] = sup_low
        row["support_range_high"] = sup_high
        row["resistance_range_low"] = res_low
        row["resistance_range_high"] = res_high
        row["support_retest"] = count_touches(w_high, w_low, sup_low, sup_high) if has_support else 0
        row["resistance_retest"] = count_touches(w_high, w_low, res_low, res_high) if has_resistance else 0
        row["distance_to_support"] = (cur_price - sup_high) / sup_high * 10000 if has_support else np.nan
        row["distance_to_resistance"] = (res_low - cur_price) / res_low * 10000 if has_resistance else np.nan
        row["recovery_up"] = 0.0
        row["recovery_down"] = 0.0

    # Outcomes: next 1, 2, 3 bars
    row["next1_close"] = close[t+1]
    row["next2_close"] = close[t+2]
    row["next3_close"] = close[t+3]
    row["next1_high"] = high[t+1]
    row["next2_high"] = high[t+2]
    row["next3_high"] = high[t+3]
    row["next1_low"] = low[t+1]
    row["next2_low"] = low[t+2]
    row["next3_low"] = low[t+3]

    results.append(row)

    if t % 50000 == 0:
        zones = sum(1 for r in results if r["has_zone"])
        print(f"  {t}/{N}... {len(results)} bars, {zones} with zones ({zones/len(results)*100:.1f}%)")

df = pd.DataFrame(results)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

print(f"\nTotal bars: {len(df)}")
print(f"  Has support: {df['has_support'].sum()} ({df['has_support'].mean()*100:.1f}%)")
print(f"  Has resistance: {df['has_resistance'].sum()} ({df['has_resistance'].mean()*100:.1f}%)")
print(f"  Has full zone: {df['has_zone'].sum()} ({df['has_zone'].mean()*100:.1f}%)")

# =========================================================================
# Feature distributions
# =========================================================================
print(f"\n{'='*70}")
print("FEATURE DISTRIBUTIONS (bars with full zone only)")
print(f"{'='*70}")

zdf = df[df["has_zone"]].copy()
print(f"\nBars with full zone: {len(zdf)}")

for feat in ["zone_width", "support_retest", "resistance_retest",
             "distance_to_support", "distance_to_resistance",
             "recovery_up", "recovery_down"]:
    vals = zdf[feat].dropna()
    if len(vals) == 0:
        continue
    print(f"\n  {feat}:")
    print(f"    Mean:   {vals.mean():.1f}")
    print(f"    Median: {vals.median():.1f}")
    print(f"    P25:    {vals.quantile(0.25):.1f}")
    print(f"    P75:    {vals.quantile(0.75):.1f}")

# =========================================================================
# OUTCOME ANALYSIS: Does price respect the zone?
# =========================================================================
print(f"\n{'='*70}")
print("OUTCOME: Does price respect support and resistance?")
print(f"{'='*70}")

if len(zdf) > 0:
    sup_high = zdf["support_range_high"]
    sup_low = zdf["support_range_low"]
    res_high = zdf["resistance_range_high"]
    res_low = zdf["resistance_range_low"]

    # Where is price relative to zone?
    at_support = zdf["distance_to_support"] <= 5  # within 5 bps of support
    at_resistance = zdf["distance_to_resistance"] <= 5  # within 5 bps of resistance
    in_middle = (~at_support) & (~at_resistance) & (zdf["distance_to_support"] > 0) & (zdf["distance_to_resistance"] > 0)

    print(f"\n  Price position:")
    print(f"    At support (within 5 bps):    {at_support.sum()} ({at_support.mean()*100:.1f}%)")
    print(f"    At resistance (within 5 bps): {at_resistance.sum()} ({at_resistance.mean()*100:.1f}%)")
    print(f"    In middle:                    {in_middle.sum()} ({in_middle.mean()*100:.1f}%)")

    # At support: does price bounce UP?
    sup_bars = zdf[at_support]
    if len(sup_bars) > 0:
        bounce_up_1 = (sup_bars["next1_close"] > sup_bars["current_price"]).mean() * 100
        bounce_up_3 = (sup_bars["next3_close"] > sup_bars["current_price"]).mean() * 100
        stayed_above = (sup_bars["next1_low"] >= sup_bars["support_range_low"]).mean() * 100
        broke_below = (sup_bars["next1_low"] < sup_bars["support_range_low"]).mean() * 100

        print(f"\n  AT SUPPORT ({len(sup_bars)} bars):")
        print(f"    Next bar close UP:        {bounce_up_1:.1f}%")
        print(f"    3 bars later close UP:    {bounce_up_3:.1f}%")
        print(f"    Stayed above support low: {stayed_above:.1f}%")
        print(f"    Broke below support low:  {broke_below:.1f}%")

    # At resistance: does price bounce DOWN?
    res_bars = zdf[at_resistance]
    if len(res_bars) > 0:
        bounce_dn_1 = (res_bars["next1_close"] < res_bars["current_price"]).mean() * 100
        bounce_dn_3 = (res_bars["next3_close"] < res_bars["current_price"]).mean() * 100
        stayed_below = (res_bars["next1_high"] <= res_bars["resistance_range_high"]).mean() * 100
        broke_above = (res_bars["next1_high"] > res_bars["resistance_range_high"]).mean() * 100

        print(f"\n  AT RESISTANCE ({len(res_bars)} bars):")
        print(f"    Next bar close DOWN:        {bounce_dn_1:.1f}%")
        print(f"    3 bars later close DOWN:    {bounce_dn_3:.1f}%")
        print(f"    Stayed below resistance hi: {stayed_below:.1f}%")
        print(f"    Broke above resistance hi:  {broke_above:.1f}%")

    # In middle: what happens?
    mid_bars = zdf[in_middle]
    if len(mid_bars) > 0:
        up_1 = (mid_bars["next1_close"] > mid_bars["current_price"]).mean() * 100
        print(f"\n  IN MIDDLE ({len(mid_bars)} bars):")
        print(f"    Next bar close UP: {up_1:.1f}%")

# =========================================================================
# BY FEATURE VALUE: Does each feature predict outcome?
# =========================================================================
print(f"\n{'='*70}")
print("BY FEATURE: How does each feature affect bounce rate?")
print(f"{'='*70}")

# By zone_width
if len(zdf) > 0:
    print(f"\n--- By zone_width ---")
    for q_low, q_high, label in [(0, 0.25, "Q1 narrow"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.01, "Q4 wide")]:
        low_val = zdf["zone_width"].quantile(q_low)
        high_val = zdf["zone_width"].quantile(min(q_high, 1.0))
        mask = (zdf["zone_width"] >= low_val) & (zdf["zone_width"] < high_val)
        sub = zdf[mask]
        if len(sub) < 50:
            continue

        # At support within this zone width
        at_sup = sub[sub["distance_to_support"] <= 5]
        if len(at_sup) > 20:
            bounce = (at_sup["next1_close"] > at_sup["current_price"]).mean() * 100
            print(f"  {label} ({low_val:.0f}-{high_val:.0f} bps) | at support: {len(at_sup)} bars, bounce UP = {bounce:.1f}%")

    # By support_retest count
    print(f"\n--- By support_retest count ---")
    sup_bars = zdf[zdf["distance_to_support"] <= 5]
    for tc in [1, 2, 3, 4, 5]:
        mask = sup_bars["support_retest"] == tc
        sub = sup_bars[mask]
        if len(sub) < 20:
            continue
        bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
        print(f"  {tc} touches: {len(sub)} bars, bounce UP = {bounce:.1f}%")

    mask = sup_bars["support_retest"] >= 6
    sub = sup_bars[mask]
    if len(sub) >= 20:
        bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
        print(f"  6+ touches: {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # By resistance_retest count
    print(f"\n--- By resistance_retest count ---")
    res_bars = zdf[zdf["distance_to_resistance"] <= 5]
    for tc in [1, 2, 3, 4, 5]:
        mask = res_bars["resistance_retest"] == tc
        sub = res_bars[mask]
        if len(sub) < 20:
            continue
        bounce = (sub["next1_close"] < sub["current_price"]).mean() * 100
        print(f"  {tc} touches: {len(sub)} bars, bounce DOWN = {bounce:.1f}%")

    mask = res_bars["resistance_retest"] >= 6
    sub = res_bars[mask]
    if len(sub) >= 20:
        bounce = (sub["next1_close"] < sub["current_price"]).mean() * 100
        print(f"  6+ touches: {len(sub)} bars, bounce DOWN = {bounce:.1f}%")

    # By recovery_up
    print(f"\n--- By recovery_up speed ---")
    sup_bars = zdf[zdf["distance_to_support"] <= 5]
    if len(sup_bars) > 0:
        for q_low, q_high, label in [(0, 0.33, "Slow"), (0.33, 0.66, "Medium"), (0.66, 1.01, "Fast")]:
            low_val = sup_bars["recovery_up"].quantile(q_low)
            high_val = sup_bars["recovery_up"].quantile(min(q_high, 1.0))
            mask = (sup_bars["recovery_up"] >= low_val) & (sup_bars["recovery_up"] < high_val)
            sub = sup_bars[mask]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
            print(f"  {label} ({low_val:.1f}-{high_val:.1f} bps/bar): {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # By recovery_down
    print(f"\n--- By recovery_down speed ---")
    res_bars = zdf[zdf["distance_to_resistance"] <= 5]
    if len(res_bars) > 0:
        for q_low, q_high, label in [(0, 0.33, "Slow"), (0.33, 0.66, "Medium"), (0.66, 1.01, "Fast")]:
            low_val = res_bars["recovery_down"].quantile(q_low)
            high_val = res_bars["recovery_down"].quantile(min(q_high, 1.0))
            mask = (res_bars["recovery_down"] >= low_val) & (res_bars["recovery_down"] < high_val)
            sub = res_bars[mask]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] < sub["current_price"]).mean() * 100
            print(f"  {label} ({low_val:.1f}-{high_val:.1f} bps/bar): {len(sub)} bars, bounce DOWN = {bounce:.1f}%")

# =========================================================================
# TRAIN vs TEST
# =========================================================================
print(f"\n{'='*70}")
print("TRAIN vs TEST")
print(f"{'='*70}")

for period, start, end in [("TRAIN 2020-2023", "2020-01-01", "2023-12-31"),
                            ("TEST 2024-2025", "2024-01-01", "2025-12-31")]:
    p_mask = (zdf.index >= start) & (zdf.index <= end)
    sub = zdf[p_mask]
    if len(sub) == 0:
        continue

    at_sup = sub[sub["distance_to_support"] <= 5]
    at_res = sub[sub["distance_to_resistance"] <= 5]

    sup_bounce = (at_sup["next1_close"] > at_sup["current_price"]).mean() * 100 if len(at_sup) > 0 else 0
    res_bounce = (at_res["next1_close"] < at_res["current_price"]).mean() * 100 if len(at_res) > 0 else 0

    print(f"\n  {period}: {len(sub)} bars with zone")
    print(f"    At support: {len(at_sup)} bars, bounce UP = {sup_bounce:.1f}%")
    print(f"    At resistance: {len(at_res)} bars, bounce DOWN = {res_bounce:.1f}%")

# =========================================================================
# BASELINE
# =========================================================================
print(f"\n{'='*70}")
print("BASELINE: Bars with NO zone")
print(f"{'='*70}")

no_zone = df[~df["has_zone"]]
if len(no_zone) > 0:
    up = (no_zone["next1_close"] > no_zone["current_price"]).mean() * 100
    print(f"\n  No zone bars: {len(no_zone)}, next bar UP = {up:.1f}%")

all_bars_up = (df["next1_close"] > df["current_price"]).mean() * 100
print(f"  All bars: {len(df)}, next bar UP = {all_bars_up:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
