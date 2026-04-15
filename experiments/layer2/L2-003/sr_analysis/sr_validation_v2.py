"""S/R Validation V2 - Zone-based support/resistance

Support ZONE = full candle range (low to high) of a bar whose high was broken upward.
Resistance ZONE = full candle range (low to high) of a bar whose low was broken downward.

Test: when price comes back into the zone, does it bounce toward recent highs/lows?

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_validation_v2.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7  # bars t-7 to t-1

print("=" * 70)
print("S/R VALIDATION V2 - Zone-Based")
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
# STEP 1: Detect S/R zones and retests
# =========================================================================
print(f"\n{'='*70}")
print("STEP 1: Detecting S/R zone retests")
print(f"{'='*70}")

results = []

for t in range(LOOKBACK + 1, N - 3):
    bar_high = high[t]
    bar_low = low[t]
    bar_close = close[t]
    prev_close = close[t - 1]

    best_support = None
    best_resistance = None

    for a in range(t - LOOKBACK, t):

        zone_high = high[a]
        zone_low = low[a]

        # --- SUPPORT ZONE ---
        # Bar A's high was broken upward by some later bar
        # Price comes back into bar A's range [zone_low, zone_high]
        broken_up = False
        highest_after = zone_high
        for k in range(a + 1, t):
            if high[k] > zone_high:
                broken_up = True
                if high[k] > highest_after:
                    highest_after = high[k]

        if broken_up:
            # Is bar t entering the zone? (bar t's low dips into [zone_low, zone_high])
            # Bar t touches or enters the zone from above
            if bar_low <= zone_high and bar_high >= zone_low and prev_close > zone_high:
                # Count touches: how many bars between break and now entered this zone
                touches = 0
                for k in range(a + 1, t):
                    if low[k] <= zone_high and high[k] >= zone_low:
                        touches += 1

                zone_width = (zone_high - zone_low) / zone_low * 10000
                dist_to_zone_top = (bar_close - zone_high) / zone_high * 10000
                recent_high = highest_after

                if best_support is None or abs(dist_to_zone_top) < abs(best_support["dist_to_zone_top"]):
                    best_support = {
                        "zone_high": zone_high,
                        "zone_low": zone_low,
                        "zone_width": zone_width,
                        "touches": touches,
                        "dist_to_zone_top": dist_to_zone_top,
                        "recent_high": recent_high,
                        "bars_since": t - a,
                    }

        # --- RESISTANCE ZONE ---
        # Bar A's low was broken downward by some later bar
        # Price comes back into bar A's range [zone_low, zone_high]
        broken_down = False
        lowest_after = zone_low
        for k in range(a + 1, t):
            if low[k] < zone_low:
                broken_down = True
                if low[k] < lowest_after:
                    lowest_after = low[k]

        if broken_down:
            # Bar t enters the zone from below
            if bar_high >= zone_low and bar_low <= zone_high and prev_close < zone_low:
                touches = 0
                for k in range(a + 1, t):
                    if low[k] <= zone_high and high[k] >= zone_low:
                        touches += 1

                zone_width = (zone_high - zone_low) / zone_low * 10000
                dist_to_zone_bottom = (zone_low - bar_close) / zone_low * 10000
                recent_low = lowest_after

                if best_resistance is None or abs(dist_to_zone_bottom) < abs(best_resistance["dist_to_zone_bottom"]):
                    best_resistance = {
                        "zone_high": zone_high,
                        "zone_low": zone_low,
                        "zone_width": zone_width,
                        "touches": touches,
                        "dist_to_zone_bottom": dist_to_zone_bottom,
                        "recent_low": recent_low,
                        "bars_since": t - a,
                    }

    # Record result
    row = {
        "date": ohlcv.index[t],
        "bar_idx": t,
        "current_price": bar_close,
        "bar_high": bar_high,
        "bar_low": bar_low,
        # Next bars for outcome measurement
        "next1_close": close[t + 1],
        "next2_close": close[t + 2],
        "next3_close": close[t + 3],
        "next1_high": high[t + 1],
        "next2_high": high[t + 2],
        "next3_high": high[t + 3],
        "next1_low": low[t + 1],
        "next2_low": low[t + 2],
        "next3_low": low[t + 3],
    }

    if best_support:
        row["support_retest"] = 1
        row["sup_zone_high"] = best_support["zone_high"]
        row["sup_zone_low"] = best_support["zone_low"]
        row["sup_zone_width"] = best_support["zone_width"]
        row["sup_touches"] = best_support["touches"]
        row["sup_dist_to_zone_top"] = best_support["dist_to_zone_top"]
        row["sup_recent_high"] = best_support["recent_high"]
        row["sup_bars_since"] = best_support["bars_since"]
    else:
        row["support_retest"] = 0

    if best_resistance:
        row["resistance_retest"] = 1
        row["res_zone_high"] = best_resistance["zone_high"]
        row["res_zone_low"] = best_resistance["zone_low"]
        row["res_zone_width"] = best_resistance["zone_width"]
        row["res_touches"] = best_resistance["touches"]
        row["res_dist_to_zone_bottom"] = best_resistance["dist_to_zone_bottom"]
        row["res_recent_low"] = best_resistance["recent_low"]
        row["res_bars_since"] = best_resistance["bars_since"]
    else:
        row["resistance_retest"] = 0

    results.append(row)

    if t % 50000 == 0:
        sup = sum(1 for r in results if r["support_retest"] == 1)
        res = sum(1 for r in results if r["resistance_retest"] == 1)
        print(f"  {t}/{N} bars... support={sup}, resistance={res}")

df = pd.DataFrame(results)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

sup_count = (df["support_retest"] == 1).sum()
res_count = (df["resistance_retest"] == 1).sum()
no_level = ((df["support_retest"] == 0) & (df["resistance_retest"] == 0)).sum()
print(f"\nTotal bars: {len(df)}")
print(f"  Support retests:    {sup_count} ({sup_count/len(df)*100:.1f}%)")
print(f"  Resistance retests: {res_count} ({res_count/len(df)*100:.1f}%)")
print(f"  No level:           {no_level} ({no_level/len(df)*100:.1f}%)")

# =========================================================================
# STEP 2: Bounce behavior - does price return toward recent highs/lows?
# =========================================================================
print(f"\n{'='*70}")
print("STEP 2: Bounce behavior - does price return toward recent extremes?")
print(f"{'='*70}")

# --- SUPPORT ---
sup = df[df["support_retest"] == 1].copy()
if len(sup) > 0:
    print(f"\n--- SUPPORT RETEST ({len(sup)} bars) ---")
    print(f"  Zone width: mean={sup['sup_zone_width'].mean():.1f} bps, median={sup['sup_zone_width'].median():.1f} bps")

    recent_high = sup["sup_recent_high"]
    zone_high = sup["sup_zone_high"]
    zone_low = sup["sup_zone_low"]

    # Bounce = price moves back toward recent high
    # Measure: did price reach at least halfway back to recent high within 3 bars?
    halfway = (zone_high + recent_high) / 2
    max_high_3 = np.maximum(sup["next1_high"], np.maximum(sup["next2_high"], sup["next3_high"]))

    reached_halfway = (max_high_3 >= halfway).mean() * 100
    reached_recent_high = (max_high_3 >= recent_high).mean() * 100
    reached_zone_top = (sup["next1_high"] >= zone_high).mean() * 100

    print(f"\n  After entering support zone, within 3 bars:")
    print(f"    Price reaches back above zone top:     {reached_zone_top:.1f}%")
    print(f"    Price reaches halfway to recent high:  {reached_halfway:.1f}%")
    print(f"    Price reaches back to recent high:     {reached_recent_high:.1f}%")

    # Break = price falls below zone bottom
    broke_zone_1 = (sup["next1_low"] < zone_low).mean() * 100
    broke_zone_3 = (np.minimum(sup["next1_low"], np.minimum(sup["next2_low"], sup["next3_low"])) < zone_low).mean() * 100

    print(f"\n  Support breaks (price falls below zone bottom):")
    print(f"    Within 1 bar: {broke_zone_1:.1f}%")
    print(f"    Within 3 bars: {broke_zone_3:.1f}%")

    # Close within zone = support playing its role
    closed_in_zone_1 = ((sup["next1_close"] >= zone_low) & (sup["next1_close"] <= zone_high)).mean() * 100
    closed_above_zone_1 = (sup["next1_close"] > zone_high).mean() * 100
    closed_below_zone_1 = (sup["next1_close"] < zone_low).mean() * 100

    print(f"\n  Next bar close position:")
    print(f"    Above zone:  {closed_above_zone_1:.1f}%")
    print(f"    In zone:     {closed_in_zone_1:.1f}%")
    print(f"    Below zone:  {closed_below_zone_1:.1f}%")

# --- RESISTANCE ---
res = df[df["resistance_retest"] == 1].copy()
if len(res) > 0:
    print(f"\n--- RESISTANCE RETEST ({len(res)} bars) ---")
    print(f"  Zone width: mean={res['res_zone_width'].mean():.1f} bps, median={res['res_zone_width'].median():.1f} bps")

    recent_low = res["res_recent_low"]
    zone_high = res["res_zone_high"]
    zone_low = res["res_zone_low"]

    halfway = (zone_low + recent_low) / 2
    min_low_3 = np.minimum(res["next1_low"], np.minimum(res["next2_low"], res["next3_low"]))

    reached_halfway = (min_low_3 <= halfway).mean() * 100
    reached_recent_low = (min_low_3 <= recent_low).mean() * 100
    reached_zone_bottom = (res["next1_low"] <= zone_low).mean() * 100

    print(f"\n  After entering resistance zone, within 3 bars:")
    print(f"    Price reaches back below zone bottom:  {reached_zone_bottom:.1f}%")
    print(f"    Price reaches halfway to recent low:   {reached_halfway:.1f}%")
    print(f"    Price reaches back to recent low:      {reached_recent_low:.1f}%")

    broke_zone_1 = (res["next1_high"] > zone_high).mean() * 100
    broke_zone_3 = (np.maximum(res["next1_high"], np.maximum(res["next2_high"], res["next3_high"])) > zone_high).mean() * 100

    print(f"\n  Resistance breaks (price rises above zone top):")
    print(f"    Within 1 bar: {broke_zone_1:.1f}%")
    print(f"    Within 3 bars: {broke_zone_3:.1f}%")

    closed_in_zone_1 = ((res["next1_close"] >= zone_low) & (res["next1_close"] <= zone_high)).mean() * 100
    closed_above_zone_1 = (res["next1_close"] > zone_high).mean() * 100
    closed_below_zone_1 = (res["next1_close"] < zone_low).mean() * 100

    print(f"\n  Next bar close position:")
    print(f"    Above zone:  {closed_above_zone_1:.1f}%")
    print(f"    In zone:     {closed_in_zone_1:.1f}%")
    print(f"    Below zone:  {closed_below_zone_1:.1f}%")

# =========================================================================
# STEP 3: By touch count
# =========================================================================
print(f"\n{'='*70}")
print("STEP 3: By touch count")
print(f"{'='*70}")

for name, subset, zone_top_col, zone_bot_col, direction in [
    ("SUPPORT", df[df["support_retest"] == 1], "sup_zone_high", "sup_zone_low", "up"),
    ("RESISTANCE", df[df["resistance_retest"] == 1], "res_zone_high", "res_zone_low", "down")
]:
    if len(subset) == 0:
        continue
    touch_col = "sup_touches" if name == "SUPPORT" else "res_touches"
    print(f"\n--- {name} ---")

    for tc_min, tc_max, label in [(0, 0, "0 touches"), (1, 1, "1 touch"), (2, 2, "2 touches"), (3, 99, "3+ touches")]:
        tc_mask = (subset[touch_col] >= tc_min) & (subset[touch_col] <= tc_max)
        sub = subset[tc_mask]
        if len(sub) < 50:
            continue

        zone_top = sub[zone_top_col]
        zone_bot = sub[zone_bot_col]

        if direction == "up":
            bounce_above = (sub["next1_close"] > zone_top).mean() * 100
            broke_below = (sub["next1_low"] < zone_bot).mean() * 100
        else:
            bounce_below = (sub["next1_close"] < zone_bot).mean() * 100
            broke_above = (sub["next1_high"] > zone_top).mean() * 100

        if direction == "up":
            print(f"  {label}: {len(sub)} bars | next close above zone: {bounce_above:.1f}% | broke below zone: {broke_below:.1f}%")
        else:
            print(f"  {label}: {len(sub)} bars | next close below zone: {bounce_below:.1f}% | broke above zone: {broke_above:.1f}%")

# =========================================================================
# STEP 4: By zone width
# =========================================================================
print(f"\n{'='*70}")
print("STEP 4: By zone width")
print(f"{'='*70}")

for name, subset, width_col, zone_top_col, zone_bot_col, direction in [
    ("SUPPORT", df[df["support_retest"] == 1], "sup_zone_width", "sup_zone_high", "sup_zone_low", "up"),
    ("RESISTANCE", df[df["resistance_retest"] == 1], "res_zone_width", "res_zone_high", "res_zone_low", "down")
]:
    if len(subset) == 0:
        continue
    print(f"\n--- {name} ---")

    quartiles = subset[width_col].quantile([0, 0.25, 0.5, 0.75, 1.0])
    print(f"  Zone width quartiles: {quartiles.values}")

    for q_low, q_high, label in [
        (0, 0.25, "Q1 (narrowest)"),
        (0.25, 0.5, "Q2"),
        (0.5, 0.75, "Q3"),
        (0.75, 1.01, "Q4 (widest)")
    ]:
        low_val = subset[width_col].quantile(q_low)
        high_val = subset[width_col].quantile(min(q_high, 1.0))
        q_mask = (subset[width_col] >= low_val) & (subset[width_col] < high_val)
        sub = subset[q_mask]
        if len(sub) < 50:
            continue

        zone_top = sub[zone_top_col]
        zone_bot = sub[zone_bot_col]

        if direction == "up":
            bounce = (sub["next1_close"] > zone_top).mean() * 100
            broke = (sub["next1_low"] < zone_bot).mean() * 100
            print(f"  {label} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars | above zone: {bounce:.1f}% | broke: {broke:.1f}%")
        else:
            bounce = (sub["next1_close"] < zone_bot).mean() * 100
            broke = (sub["next1_high"] > zone_top).mean() * 100
            print(f"  {label} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars | below zone: {bounce:.1f}% | broke: {broke:.1f}%")

# =========================================================================
# STEP 5: Train vs Test
# =========================================================================
print(f"\n{'='*70}")
print("STEP 5: Train vs Test")
print(f"{'='*70}")

for name, subset, zone_top_col, zone_bot_col, direction in [
    ("SUPPORT", df[df["support_retest"] == 1], "sup_zone_high", "sup_zone_low", "up"),
    ("RESISTANCE", df[df["resistance_retest"] == 1], "res_zone_high", "res_zone_low", "down")
]:
    if len(subset) == 0:
        continue
    print(f"\n--- {name} ---")

    for period, start, end in [("TRAIN 2020-2023", "2020-01-01", "2023-12-31"),
                                ("TEST 2024-2025", "2024-01-01", "2025-12-31")]:
        p_mask = (subset.index >= start) & (subset.index <= end)
        sub = subset[p_mask]
        if len(sub) == 0:
            continue

        zone_top = sub[zone_top_col]
        zone_bot = sub[zone_bot_col]

        if direction == "up":
            bounce = (sub["next1_close"] > zone_top).mean() * 100
            broke = (sub["next1_low"] < zone_bot).mean() * 100
            in_zone = ((sub["next1_close"] >= zone_bot) & (sub["next1_close"] <= zone_top)).mean() * 100
            print(f"  {period}: {len(sub)} bars | above: {bounce:.1f}% | in zone: {in_zone:.1f}% | broke: {broke:.1f}%")
        else:
            bounce = (sub["next1_close"] < zone_bot).mean() * 100
            broke = (sub["next1_high"] > zone_top).mean() * 100
            in_zone = ((sub["next1_close"] >= zone_bot) & (sub["next1_close"] <= zone_top)).mean() * 100
            print(f"  {period}: {len(sub)} bars | below: {bounce:.1f}% | in zone: {in_zone:.1f}% | broke: {broke:.1f}%")

# =========================================================================
# STEP 6: Baseline
# =========================================================================
print(f"\n{'='*70}")
print("STEP 6: Baseline comparison")
print(f"{'='*70}")

no_sr = df[(df["support_retest"] == 0) & (df["resistance_retest"] == 0)]
print(f"\nBars with NO S/R retest: {len(no_sr)}")
up = (no_sr["next1_close"] > no_sr["current_price"]).mean() * 100
down = (no_sr["next1_close"] < no_sr["current_price"]).mean() * 100
print(f"  Next bar UP: {up:.1f}%, DOWN: {down:.1f}%")

sup = df[df["support_retest"] == 1]
if len(sup) > 0:
    up = (sup["next1_close"] > sup["current_price"]).mean() * 100
    print(f"\nSUPPORT bars ({len(sup)}): next bar UP = {up:.1f}%")

res = df[df["resistance_retest"] == 1]
if len(res) > 0:
    down = (res["next1_close"] < res["current_price"]).mean() * 100
    print(f"RESISTANCE bars ({len(res)}): next bar DOWN = {down:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
