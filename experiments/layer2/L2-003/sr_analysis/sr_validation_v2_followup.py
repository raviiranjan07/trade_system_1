"""S/R V2 Follow-up Analysis

Q1: When support breaks (9.8%), how far does price fall? Temporary or real break?
Q2: When price bounces above zone top (84.6%), how far does it go?
Q3: When resistance close below zone (52%), does price reach next support?

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_validation_v2_followup.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7

print("=" * 70)
print("S/R V2 FOLLOW-UP ANALYSIS")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
N = len(ohlcv)

# =========================================================================
# Recompute S/R zones (same logic as V2) but store more info
# =========================================================================
print("\nComputing S/R zones...")

support_rows = []
resistance_rows = []

for t in range(LOOKBACK + 1, N - 8):  # need 8 bars ahead
    bar_high = high[t]
    bar_low = low[t]
    bar_close = close[t]
    prev_close = close[t - 1]

    best_support = None
    best_resistance = None

    # Also find ALL zones in the window (for "next support/resistance" questions)
    all_support_zones = []
    all_resistance_zones = []

    for a in range(t - LOOKBACK, t):
        zone_high = high[a]
        zone_low = low[a]

        # --- SUPPORT ZONE ---
        broken_up = False
        highest_after = zone_high
        for k in range(a + 1, t):
            if high[k] > zone_high:
                broken_up = True
                if high[k] > highest_after:
                    highest_after = high[k]

        if broken_up:
            if bar_low <= zone_high and bar_high >= zone_low and prev_close > zone_high:
                touches = 0
                for k in range(a + 1, t):
                    if low[k] <= zone_high and high[k] >= zone_low:
                        touches += 1
                dist = abs(bar_close - zone_high) / zone_high * 10000
                info = {
                    "zone_high": zone_high, "zone_low": zone_low,
                    "touches": touches, "dist": dist,
                    "recent_high": highest_after, "bar_a": a,
                }
                if best_support is None or dist < best_support["dist"]:
                    best_support = info

            # Also track as potential "next support" for resistance bars
            all_support_zones.append({"zone_high": zone_high, "zone_low": zone_low})

        # --- RESISTANCE ZONE ---
        broken_down = False
        lowest_after = zone_low
        for k in range(a + 1, t):
            if low[k] < zone_low:
                broken_down = True
                if low[k] < lowest_after:
                    lowest_after = low[k]

        if broken_down:
            if bar_high >= zone_low and bar_low <= zone_high and prev_close < zone_low:
                touches = 0
                for k in range(a + 1, t):
                    if low[k] <= zone_high and high[k] >= zone_low:
                        touches += 1
                dist = abs(bar_close - zone_low) / zone_low * 10000
                info = {
                    "zone_high": zone_high, "zone_low": zone_low,
                    "touches": touches, "dist": dist,
                    "recent_low": lowest_after, "bar_a": a,
                }
                if best_resistance is None or dist < best_resistance["dist"]:
                    best_resistance = info

            all_resistance_zones.append({"zone_high": zone_high, "zone_low": zone_low})

    # Collect next 8 bars for outcome
    next_highs = high[t+1:t+9]
    next_lows = low[t+1:t+9]
    next_closes = close[t+1:t+9]

    if best_support:
        zh = best_support["zone_high"]
        zl = best_support["zone_low"]
        rh = best_support["recent_high"]

        # Find next support zone below current one
        lower_zones = [z for z in all_support_zones if z["zone_high"] < zl]
        next_sup_top = max([z["zone_high"] for z in lower_zones]) if lower_zones else None
        next_sup_bot = None
        if lower_zones:
            nearest_lower = max(lower_zones, key=lambda z: z["zone_high"])
            next_sup_top = nearest_lower["zone_high"]
            next_sup_bot = nearest_lower["zone_low"]

        support_rows.append({
            "date": ohlcv.index[t],
            "zone_high": zh,
            "zone_low": zl,
            "zone_width": (zh - zl) / zl * 10000,
            "recent_high": rh,
            "touches": best_support["touches"],
            "current_close": bar_close,
            "next_sup_top": next_sup_top,
            "next_sup_bot": next_sup_bot,
            # Next bar outcomes
            "next1_close": next_closes[0],
            "next1_high": next_highs[0],
            "next1_low": next_lows[0],
            # Max move within 3 bars
            "max_high_3": np.max(next_highs[:3]),
            "min_low_3": np.min(next_lows[:3]),
            # Max move within 8 bars
            "max_high_8": np.max(next_highs),
            "min_low_8": np.min(next_lows),
        })

    if best_resistance:
        zh = best_resistance["zone_high"]
        zl = best_resistance["zone_low"]
        rl = best_resistance["recent_low"]

        # Find next resistance zone above current one
        upper_zones = [z for z in all_resistance_zones if z["zone_low"] > zh]
        next_res_top = None
        next_res_bot = None
        if upper_zones:
            nearest_upper = min(upper_zones, key=lambda z: z["zone_low"])
            next_res_top = nearest_upper["zone_high"]
            next_res_bot = nearest_upper["zone_low"]

        # Find next support below for Q3
        lower_sup_zones = [z for z in all_support_zones if z["zone_high"] < zl]
        next_sup_for_res_top = None
        next_sup_for_res_bot = None
        if lower_sup_zones:
            nearest_lower = max(lower_sup_zones, key=lambda z: z["zone_high"])
            next_sup_for_res_top = nearest_lower["zone_high"]
            next_sup_for_res_bot = nearest_lower["zone_low"]

        resistance_rows.append({
            "date": ohlcv.index[t],
            "zone_high": zh,
            "zone_low": zl,
            "zone_width": (zh - zl) / zl * 10000,
            "recent_low": rl,
            "touches": best_resistance["touches"],
            "current_close": bar_close,
            "next_res_top": next_res_top,
            "next_res_bot": next_res_bot,
            "next_sup_top": next_sup_for_res_top,
            "next_sup_bot": next_sup_for_res_bot,
            "next1_close": next_closes[0],
            "next1_high": next_highs[0],
            "next1_low": next_lows[0],
            "max_high_3": np.max(next_highs[:3]),
            "min_low_3": np.min(next_lows[:3]),
            "max_high_8": np.max(next_highs),
            "min_low_8": np.min(next_lows),
        })

    if t % 50000 == 0:
        print(f"  {t}/{N}... sup={len(support_rows)}, res={len(resistance_rows)}")

sup_df = pd.DataFrame(support_rows).set_index("date")
res_df = pd.DataFrame(resistance_rows).set_index("date")
print(f"\nSupport retests: {len(sup_df)}")
print(f"Resistance retests: {len(res_df)}")

# =========================================================================
# Q1: When support breaks, how far does price fall?
# =========================================================================
print(f"\n{'='*70}")
print("Q1: When support breaks (next bar closes below zone), how far does it fall?")
print(f"{'='*70}")

broke_sup = sup_df[sup_df["next1_close"] < sup_df["zone_low"]]
print(f"\nSupport breaks: {len(broke_sup)} bars ({len(broke_sup)/len(sup_df)*100:.1f}% of support retests)")

# How far below zone bottom did it close?
drop_from_zone = (broke_sup["zone_low"] - broke_sup["next1_close"]) / broke_sup["zone_low"] * 10000
print(f"\n  Drop below zone bottom (next bar close):")
print(f"    Mean:   {drop_from_zone.mean():.1f} bps")
print(f"    Median: {drop_from_zone.median():.1f} bps")
print(f"    P25:    {drop_from_zone.quantile(0.25):.1f} bps")
print(f"    P75:    {drop_from_zone.quantile(0.75):.1f} bps")
print(f"    P90:    {drop_from_zone.quantile(0.90):.1f} bps")

# Max drop within 3 bars
max_drop_3 = (broke_sup["zone_low"] - broke_sup["min_low_3"]) / broke_sup["zone_low"] * 10000
print(f"\n  Max drop below zone (within 3 bars):")
print(f"    Mean:   {max_drop_3.mean():.1f} bps")
print(f"    Median: {max_drop_3.median():.1f} bps")
print(f"    P75:    {max_drop_3.quantile(0.75):.1f} bps")
print(f"    P90:    {max_drop_3.quantile(0.90):.1f} bps")

# Max drop within 8 bars
max_drop_8 = (broke_sup["zone_low"] - broke_sup["min_low_8"]) / broke_sup["zone_low"] * 10000
print(f"\n  Max drop below zone (within 8 bars):")
print(f"    Mean:   {max_drop_8.mean():.1f} bps")
print(f"    Median: {max_drop_8.median():.1f} bps")
print(f"    P75:    {max_drop_8.quantile(0.75):.1f} bps")
print(f"    P90:    {max_drop_8.quantile(0.90):.1f} bps")

# Did price recover back into zone within 3 bars? (temporary break / bluff)
recovered_3 = (broke_sup["max_high_3"] >= broke_sup["zone_low"]).mean() * 100
recovered_above_3 = (broke_sup["max_high_3"] >= broke_sup["zone_high"]).mean() * 100
print(f"\n  Recovery after break:")
print(f"    Price returns to zone bottom within 3 bars: {recovered_3:.1f}%")
print(f"    Price returns above zone top within 3 bars: {recovered_above_3:.1f}%")

# Did it reach next support zone?
has_next_sup = broke_sup["next_sup_top"].notna()
broke_with_next = broke_sup[has_next_sup]
if len(broke_with_next) > 0:
    reached_next = (broke_with_next["min_low_3"] <= broke_with_next["next_sup_top"]).mean() * 100
    reached_next_8 = (broke_with_next["min_low_8"] <= broke_with_next["next_sup_top"]).mean() * 100
    dist_to_next = ((broke_with_next["zone_low"] - broke_with_next["next_sup_top"]) / broke_with_next["zone_low"] * 10000)
    print(f"\n  Next support zone ({len(broke_with_next)} bars have a lower zone):")
    print(f"    Distance to next support: mean={dist_to_next.mean():.1f} bps, median={dist_to_next.median():.1f} bps")
    print(f"    Reached next support within 3 bars: {reached_next:.1f}%")
    print(f"    Reached next support within 8 bars: {reached_next_8:.1f}%")

# =========================================================================
# Q2: When price bounces above zone top, how far does it go?
# =========================================================================
print(f"\n{'='*70}")
print("Q2: When price bounces above zone top, how far does it go?")
print(f"{'='*70}")

# All support retests where price reaches above zone top within 3 bars (84.6%)
bounced = sup_df[sup_df["max_high_3"] >= sup_df["zone_high"]]
print(f"\nBounced above zone top within 3 bars: {len(bounced)} bars")

# How far above zone top?
above_zone_3 = (bounced["max_high_3"] - bounced["zone_high"]) / bounced["zone_high"] * 10000
above_zone_8 = (bounced["max_high_8"] - bounced["zone_high"]) / bounced["zone_high"] * 10000
print(f"\n  Max move above zone top (within 3 bars):")
print(f"    Mean:   {above_zone_3.mean():.1f} bps")
print(f"    Median: {above_zone_3.median():.1f} bps")
print(f"    P25:    {above_zone_3.quantile(0.25):.1f} bps")
print(f"    P75:    {above_zone_3.quantile(0.75):.1f} bps")
print(f"    P90:    {above_zone_3.quantile(0.90):.1f} bps")

print(f"\n  Max move above zone top (within 8 bars):")
print(f"    Mean:   {above_zone_8.mean():.1f} bps")
print(f"    Median: {above_zone_8.median():.1f} bps")
print(f"    P25:    {above_zone_8.quantile(0.25):.1f} bps")
print(f"    P75:    {above_zone_8.quantile(0.75):.1f} bps")
print(f"    P90:    {above_zone_8.quantile(0.90):.1f} bps")

# Did it reach back to recent high?
dist_to_recent = (bounced["recent_high"] - bounced["zone_high"]) / bounced["zone_high"] * 10000
print(f"\n  Distance from zone top to recent high:")
print(f"    Mean:   {dist_to_recent.mean():.1f} bps")
print(f"    Median: {dist_to_recent.median():.1f} bps")

reached_recent = (bounced["max_high_3"] >= bounced["recent_high"]).mean() * 100
reached_recent_8 = (bounced["max_high_8"] >= bounced["recent_high"]).mean() * 100
exceeded_recent = (bounced["max_high_8"] > bounced["recent_high"]).mean() * 100
print(f"\n  Reached recent high within 3 bars: {reached_recent:.1f}%")
print(f"  Reached recent high within 8 bars: {reached_recent_8:.1f}%")
print(f"  Exceeded recent high within 8 bars: {exceeded_recent:.1f}%")

# Move above zone in bps buckets
print(f"\n  Move above zone top (3 bars) - distribution:")
for threshold in [5, 10, 15, 20, 30, 50]:
    pct = (above_zone_3 >= threshold).mean() * 100
    print(f"    >= {threshold} bps: {pct:.1f}%")

# =========================================================================
# Q3: Resistance close below zone - does price reach next support?
# =========================================================================
print(f"\n{'='*70}")
print("Q3: When resistance retest closes below zone (52%), does it reach next support?")
print(f"{'='*70}")

closed_below = res_df[res_df["next1_close"] < res_df["zone_low"]]
print(f"\nResistance: next bar closed below zone: {len(closed_below)} bars")

# How far below zone?
drop_below = (res_df[res_df["next1_close"] < res_df["zone_low"]]["zone_low"] - closed_below["next1_close"]) / closed_below["zone_low"] * 10000
print(f"\n  Drop below zone bottom:")
print(f"    Mean:   {drop_below.mean():.1f} bps")
print(f"    Median: {drop_below.median():.1f} bps")
print(f"    P75:    {drop_below.quantile(0.75):.1f} bps")

# Does it reach next support?
has_next = closed_below["next_sup_top"].notna()
with_next = closed_below[has_next]
print(f"\n  Bars with a lower support zone: {len(with_next)}/{len(closed_below)}")

if len(with_next) > 0:
    dist_to_next = (with_next["zone_low"] - with_next["next_sup_top"]) / with_next["zone_low"] * 10000
    print(f"  Distance to next support: mean={dist_to_next.mean():.1f} bps, median={dist_to_next.median():.1f} bps")

    reached_3 = (with_next["min_low_3"] <= with_next["next_sup_top"]).mean() * 100
    reached_8 = (with_next["min_low_8"] <= with_next["next_sup_top"]).mean() * 100
    print(f"  Reached next support within 3 bars: {reached_3:.1f}%")
    print(f"  Reached next support within 8 bars: {reached_8:.1f}%")

    # Of those that reached next support, did they bounce?
    reached_mask = with_next["min_low_8"] <= with_next["next_sup_top"]
    reached = with_next[reached_mask]
    if len(reached) > 0:
        # After reaching next support, did price go back up?
        bounced_back = (reached["max_high_8"] >= reached["zone_low"]).mean() * 100
        print(f"  Of those that reached next support, bounced back to resistance zone: {bounced_back:.1f}%")

# Did price that closed below resistance zone come BACK to the zone? (rejected then returned)
came_back_3 = (closed_below["max_high_3"] >= closed_below["zone_low"]).mean() * 100
came_back_8 = (closed_below["max_high_8"] >= closed_below["zone_low"]).mean() * 100
print(f"\n  Price returned back to resistance zone:")
print(f"    Within 3 bars: {came_back_3:.1f}%")
print(f"    Within 8 bars: {came_back_8:.1f}%")

# Also check: the 10.5% that closed ABOVE resistance zone (broke through)
broke_res = res_df[res_df["next1_close"] > res_df["zone_high"]]
print(f"\n  Resistance broke (closed above zone): {len(broke_res)} bars ({len(broke_res)/len(res_df)*100:.1f}%)")
if len(broke_res) > 0:
    above = (broke_res["next1_close"] - broke_res["zone_high"]) / broke_res["zone_high"] * 10000
    print(f"  How far above: mean={above.mean():.1f} bps, median={above.median():.1f} bps")
    # Did it come back down?
    came_down_3 = (broke_res["min_low_3"] <= broke_res["zone_high"]).mean() * 100
    print(f"  Came back to zone within 3 bars: {came_down_3:.1f}%")

# In-zone closes (37.5%)
in_zone = res_df[(res_df["next1_close"] >= res_df["zone_low"]) & (res_df["next1_close"] <= res_df["zone_high"])]
print(f"\n  Closed IN zone: {len(in_zone)} bars ({len(in_zone)/len(res_df)*100:.1f}%)")
if len(in_zone) > 0:
    # What happens after? Does it break up or down?
    broke_up_3 = (in_zone["max_high_3"] > in_zone["zone_high"]).mean() * 100
    broke_down_3 = (in_zone["min_low_3"] < in_zone["zone_low"]).mean() * 100
    print(f"  From in-zone, within 3 bars:")
    print(f"    Broke above zone: {broke_up_3:.1f}%")
    print(f"    Broke below zone: {broke_down_3:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
