"""S/R V2 Deep Analysis - What happens after support break + recovery

When support breaks and price recovers back to the zone:
1. Does it break THROUGH the zone (upward) or get rejected again?
2. If breaks through: how far up? Does it reach next resistance?
3. If rejected: how far down? Does it test the support (zone bottom) again?

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_validation_v2_deep.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7

print("=" * 70)
print("S/R V2 DEEP ANALYSIS - Recovery behavior after support break")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
N = len(ohlcv)

# =========================================================================
# Find support zones, track breaks and what happens after recovery
# =========================================================================
print("\nComputing S/R zones and tracking break + recovery...")

support_break_rows = []
resistance_break_rows = []

for t in range(LOOKBACK + 1, N - 12):  # need 12 bars ahead for tracking
    bar_high = high[t]
    bar_low = low[t]
    bar_close = close[t]
    prev_close = close[t - 1]

    # Find all zones
    all_zones = []

    for a in range(t - LOOKBACK, t):
        zone_high = high[a]
        zone_low = low[a]

        # SUPPORT ZONE
        broken_up = False
        highest_after = zone_high
        for k in range(a + 1, t):
            if high[k] > zone_high:
                broken_up = True
                if high[k] > highest_after:
                    highest_after = high[k]

        if broken_up and bar_low <= zone_high and bar_high >= zone_low and prev_close > zone_high:
            touches = 0
            for k in range(a + 1, t):
                if low[k] <= zone_high and high[k] >= zone_low:
                    touches += 1
            dist = abs(bar_close - zone_high) / zone_high * 10000
            all_zones.append({
                "zone_high": zone_high, "zone_low": zone_low,
                "role": "support", "touches": touches, "dist": dist,
                "recent_high": highest_after, "bar_a": a,
            })

        # RESISTANCE ZONE
        broken_down = False
        lowest_after = zone_low
        for k in range(a + 1, t):
            if low[k] < zone_low:
                broken_down = True
                if low[k] < lowest_after:
                    lowest_after = low[k]

        if broken_down and bar_high >= zone_low and bar_low <= zone_high and prev_close < zone_low:
            touches = 0
            for k in range(a + 1, t):
                if low[k] <= zone_high and high[k] >= zone_low:
                    touches += 1
            dist = abs(bar_close - zone_low) / zone_low * 10000
            all_zones.append({
                "zone_high": zone_high, "zone_low": zone_low,
                "role": "resistance", "touches": touches, "dist": dist,
                "recent_low": lowest_after, "bar_a": a,
            })

    # Pick nearest support and resistance
    support_zones = [z for z in all_zones if z["role"] == "support"]
    resistance_zones = [z for z in all_zones if z["role"] == "resistance"]

    best_sup = min(support_zones, key=lambda x: x["dist"]) if support_zones else None
    best_res = min(resistance_zones, key=lambda x: x["dist"]) if resistance_zones else None

    # Find other zones for "next resistance" / "next support" lookups
    # All resistance zones above current support
    upper_res_zones = [z for z in all_zones if z["role"] == "resistance" and z["zone_low"] > (best_sup["zone_high"] if best_sup else 0)]
    # All support zones below current resistance
    lower_sup_zones = [z for z in all_zones if z["role"] == "support" and z["zone_high"] < (best_res["zone_low"] if best_res else 999999999)]

    # ---- SUPPORT BREAK TRACKING ----
    if best_sup:
        zh = best_sup["zone_high"]
        zl = best_sup["zone_low"]

        # Did next bar close below zone? (support break)
        if close[t + 1] < zl:
            # Track what happens in bars t+2 to t+10
            # Does price come back to zone?
            recovery_bar = None
            for b in range(t + 2, t + 11):
                if high[b] >= zl:  # price reaches zone bottom
                    recovery_bar = b
                    break

            if recovery_bar is not None:
                # Price recovered to zone. Now what?
                # Check bars AFTER recovery (start from recovery_bar + 1)
                rb1 = recovery_bar + 1
                bars_left = min(t + 14, N) - rb1
                if bars_left >= 3:
                    post_highs = high[rb1:rb1 + 3]
                    post_lows = low[rb1:rb1 + 3]
                    post_closes = close[rb1:rb1 + 3]

                    # Did price break THROUGH zone upward? (high > zone_high)
                    broke_through_up = np.max(post_highs) > zh
                    # Did price get rejected? (drops below zone_low again)
                    rejected = np.min(post_lows) < zl

                    # How far up after recovery?
                    max_up = (np.max(post_highs) - zh) / zh * 10000
                    # How far down after recovery?
                    max_down = (zl - np.min(post_lows)) / zl * 10000
                    max_down = max(max_down, 0)

                    # Did it reach next resistance?
                    next_res_top = None
                    if upper_res_zones:
                        nearest_upper = min(upper_res_zones, key=lambda z: z["zone_low"])
                        next_res_top = nearest_upper["zone_high"]
                        next_res_bot = nearest_upper["zone_low"]

                    # How deep was the initial break?
                    break_depth = (zl - close[t + 1]) / zl * 10000
                    min_before_recovery = np.min(low[t + 1:recovery_bar + 1])
                    max_break_depth = (zl - min_before_recovery) / zl * 10000

                    support_break_rows.append({
                        "date": ohlcv.index[t],
                        "zone_high": zh,
                        "zone_low": zl,
                        "zone_width": (zh - zl) / zl * 10000,
                        "touches": best_sup["touches"],
                        "break_depth_close": break_depth,
                        "max_break_depth": max_break_depth,
                        "recovery_bars": recovery_bar - t - 1,
                        "broke_through_up": broke_through_up,
                        "rejected_down": rejected,
                        "max_up_from_zone": max_up,
                        "max_down_from_zone": max_down,
                        "next_res_top": next_res_top,
                        "reached_next_res": np.max(post_highs) >= next_res_top if next_res_top else None,
                        "post_close_3": post_closes[3] if len(post_closes) > 3 else post_closes[-1],
                        "close_at_break": close[t + 1],
                    })

    # ---- RESISTANCE BREAK TRACKING ----
    if best_res:
        zh = best_res["zone_high"]
        zl = best_res["zone_low"]

        # Did next bar close above zone? (resistance break)
        if close[t + 1] > zh:
            recovery_bar = None
            for b in range(t + 2, t + 11):
                if low[b] <= zh:  # price reaches zone top
                    recovery_bar = b
                    break

            if recovery_bar is not None:
                rb1 = recovery_bar + 1
                bars_left = min(t + 14, N) - rb1
                if bars_left >= 3:
                    post_highs = high[rb1:rb1 + 3]
                    post_lows = low[rb1:rb1 + 3]
                    post_closes = close[rb1:rb1 + 3]

                    broke_through_down = np.min(post_lows) < zl
                    rejected = np.max(post_highs) > zh

                    max_down = (zl - np.min(post_lows)) / zl * 10000
                    max_down = max(max_down, 0)
                    max_up = (np.max(post_highs) - zh) / zh * 10000
                    max_up = max(max_up, 0)

                    next_sup_bot = None
                    if lower_sup_zones:
                        nearest_lower = max(lower_sup_zones, key=lambda z: z["zone_high"])
                        next_sup_bot = nearest_lower["zone_low"]
                        next_sup_top = nearest_lower["zone_high"]

                    break_height = (close[t + 1] - zh) / zh * 10000
                    max_before_recovery = np.max(high[t + 1:recovery_bar + 1])
                    max_break_height = (max_before_recovery - zh) / zh * 10000

                    resistance_break_rows.append({
                        "date": ohlcv.index[t],
                        "zone_high": zh,
                        "zone_low": zl,
                        "zone_width": (zh - zl) / zl * 10000,
                        "touches": best_res["touches"],
                        "break_height_close": break_height,
                        "max_break_height": max_break_height,
                        "recovery_bars": recovery_bar - t - 1,
                        "broke_through_down": broke_through_down,
                        "rejected_up": rejected,
                        "max_down_from_zone": max_down,
                        "max_up_from_zone": max_up,
                        "next_sup_bot": next_sup_bot,
                        "reached_next_sup": np.min(post_lows) <= next_sup_bot if next_sup_bot else None,
                        "post_close_3": post_closes[3] if len(post_closes) > 3 else post_closes[-1],
                        "close_at_break": close[t + 1],
                    })

    if t % 50000 == 0:
        print(f"  {t}/{N}... sup_breaks={len(support_break_rows)}, res_breaks={len(resistance_break_rows)}")

sup_brk = pd.DataFrame(support_break_rows)
res_brk = pd.DataFrame(resistance_break_rows)
print(f"\nSupport breaks with recovery: {len(sup_brk)}")
print(f"Resistance breaks with recovery: {len(res_brk)}")

# =========================================================================
# SUPPORT BREAK + RECOVERY ANALYSIS
# =========================================================================
print(f"\n{'='*70}")
print("SUPPORT: Break -> Recovery -> What next?")
print(f"{'='*70}")

if len(sup_brk) > 0:
    print(f"\nTotal support breaks that recovered to zone: {len(sup_brk)}")

    # Initial break depth
    print(f"\n  Initial break depth (close below zone bottom):")
    print(f"    Mean:   {sup_brk['break_depth_close'].mean():.1f} bps")
    print(f"    Median: {sup_brk['break_depth_close'].median():.1f} bps")

    print(f"\n  Max depth before recovery:")
    print(f"    Mean:   {sup_brk['max_break_depth'].mean():.1f} bps")
    print(f"    Median: {sup_brk['max_break_depth'].median():.1f} bps")

    print(f"\n  Recovery time: mean={sup_brk['recovery_bars'].mean():.1f} bars, median={sup_brk['recovery_bars'].median():.0f} bars")

    # After recovery: break through or rejected?
    through = sup_brk["broke_through_up"].sum()
    rejected = sup_brk["rejected_down"].sum()
    both = ((sup_brk["broke_through_up"]) & (sup_brk["rejected_down"])).sum()
    neither = ((~sup_brk["broke_through_up"]) & (~sup_brk["rejected_down"])).sum()

    print(f"\n  After price recovers to support zone (within 3 bars):")
    print(f"    Broke THROUGH upward (above zone top):  {through} ({through/len(sup_brk)*100:.1f}%)")
    print(f"    Rejected DOWN (fell below zone bottom):  {rejected} ({rejected/len(sup_brk)*100:.1f}%)")
    print(f"    Both (volatile - went both ways):        {both} ({both/len(sup_brk)*100:.1f}%)")
    print(f"    Neither (stayed in zone):                {neither} ({neither/len(sup_brk)*100:.1f}%)")

    # Only broke through (not both)
    only_through = ((sup_brk["broke_through_up"]) & (~sup_brk["rejected_down"]))
    only_rejected = ((~sup_brk["broke_through_up"]) & (sup_brk["rejected_down"]))

    print(f"\n    ONLY broke through (clean break up):   {only_through.sum()} ({only_through.mean()*100:.1f}%)")
    print(f"    ONLY rejected (clean rejection down):  {only_rejected.sum()} ({only_rejected.mean()*100:.1f}%)")

    # How far up if broke through?
    through_rows = sup_brk[sup_brk["broke_through_up"]]
    if len(through_rows) > 0:
        print(f"\n  When broke through upward ({len(through_rows)} bars):")
        print(f"    Max up from zone top: mean={through_rows['max_up_from_zone'].mean():.1f} bps, median={through_rows['max_up_from_zone'].median():.1f} bps")

        # Did it reach next resistance?
        has_next = through_rows["reached_next_res"].notna()
        with_next = through_rows[has_next]
        if len(with_next) > 0:
            reached = with_next["reached_next_res"].sum()
            print(f"    Reached next resistance zone: {reached}/{len(with_next)} ({reached/len(with_next)*100:.1f}%)")

    # How far down if rejected?
    rejected_rows = sup_brk[sup_brk["rejected_down"]]
    if len(rejected_rows) > 0:
        print(f"\n  When rejected downward ({len(rejected_rows)} bars):")
        print(f"    Max drop from zone bottom: mean={rejected_rows['max_down_from_zone'].mean():.1f} bps, median={rejected_rows['max_down_from_zone'].median():.1f} bps")

    # By touch count
    print(f"\n  By touch count:")
    for tc_min, tc_max, label in [(0, 1, "0-1 touches"), (2, 2, "2 touches"), (3, 99, "3+ touches")]:
        tc_mask = (sup_brk["touches"] >= tc_min) & (sup_brk["touches"] <= tc_max)
        sub = sup_brk[tc_mask]
        if len(sub) < 20:
            continue
        through_pct = sub["broke_through_up"].mean() * 100
        rejected_pct = sub["rejected_down"].mean() * 100
        print(f"    {label}: {len(sub)} bars | broke through: {through_pct:.1f}% | rejected: {rejected_pct:.1f}%")

    # By zone width
    print(f"\n  By zone width:")
    for q_low, q_high, label in [(0, 0.25, "Q1 narrow"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.01, "Q4 wide")]:
        low_val = sup_brk["zone_width"].quantile(q_low)
        high_val = sup_brk["zone_width"].quantile(min(q_high, 1.0))
        q_mask = (sup_brk["zone_width"] >= low_val) & (sup_brk["zone_width"] < high_val)
        sub = sup_brk[q_mask]
        if len(sub) < 20:
            continue
        through_pct = sub["broke_through_up"].mean() * 100
        rejected_pct = sub["rejected_down"].mean() * 100
        print(f"    {label} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars | through: {through_pct:.1f}% | rejected: {rejected_pct:.1f}%")

# =========================================================================
# RESISTANCE BREAK + RECOVERY ANALYSIS
# =========================================================================
print(f"\n{'='*70}")
print("RESISTANCE: Break -> Recovery -> What next?")
print(f"{'='*70}")

if len(res_brk) > 0:
    print(f"\nTotal resistance breaks that recovered to zone: {len(res_brk)}")

    print(f"\n  Initial break height (close above zone top):")
    print(f"    Mean:   {res_brk['break_height_close'].mean():.1f} bps")
    print(f"    Median: {res_brk['break_height_close'].median():.1f} bps")

    print(f"\n  Max height before recovery:")
    print(f"    Mean:   {res_brk['max_break_height'].mean():.1f} bps")
    print(f"    Median: {res_brk['max_break_height'].median():.1f} bps")

    print(f"\n  Recovery time: mean={res_brk['recovery_bars'].mean():.1f} bars, median={res_brk['recovery_bars'].median():.0f} bars")

    through = res_brk["broke_through_down"].sum()
    rejected = res_brk["rejected_up"].sum()
    both = ((res_brk["broke_through_down"]) & (res_brk["rejected_up"])).sum()
    neither = ((~res_brk["broke_through_down"]) & (~res_brk["rejected_up"])).sum()

    print(f"\n  After price recovers to resistance zone (within 3 bars):")
    print(f"    Broke THROUGH downward (below zone bottom):  {through} ({through/len(res_brk)*100:.1f}%)")
    print(f"    Rejected UP (rose above zone top):            {rejected} ({rejected/len(res_brk)*100:.1f}%)")
    print(f"    Both (volatile):                              {both} ({both/len(res_brk)*100:.1f}%)")
    print(f"    Neither (stayed in zone):                     {neither} ({neither/len(res_brk)*100:.1f}%)")

    only_through = ((res_brk["broke_through_down"]) & (~res_brk["rejected_up"]))
    only_rejected = ((~res_brk["broke_through_down"]) & (res_brk["rejected_up"]))

    print(f"\n    ONLY broke through (clean break down):  {only_through.sum()} ({only_through.mean()*100:.1f}%)")
    print(f"    ONLY rejected (clean rejection up):     {only_rejected.sum()} ({only_rejected.mean()*100:.1f}%)")

    through_rows = res_brk[res_brk["broke_through_down"]]
    if len(through_rows) > 0:
        print(f"\n  When broke through downward ({len(through_rows)} bars):")
        print(f"    Max drop from zone bottom: mean={through_rows['max_down_from_zone'].mean():.1f} bps, median={through_rows['max_down_from_zone'].median():.1f} bps")

        has_next = through_rows["reached_next_sup"].notna()
        with_next = through_rows[has_next]
        if len(with_next) > 0:
            reached = with_next["reached_next_sup"].sum()
            print(f"    Reached next support zone: {reached}/{len(with_next)} ({reached/len(with_next)*100:.1f}%)")

    rejected_rows = res_brk[res_brk["rejected_up"]]
    if len(rejected_rows) > 0:
        print(f"\n  When rejected upward ({len(rejected_rows)} bars):")
        print(f"    Max rise from zone top: mean={rejected_rows['max_up_from_zone'].mean():.1f} bps, median={rejected_rows['max_up_from_zone'].median():.1f} bps")

    print(f"\n  By touch count:")
    for tc_min, tc_max, label in [(0, 1, "0-1 touches"), (2, 2, "2 touches"), (3, 99, "3+ touches")]:
        tc_mask = (res_brk["touches"] >= tc_min) & (res_brk["touches"] <= tc_max)
        sub = res_brk[tc_mask]
        if len(sub) < 20:
            continue
        through_pct = sub["broke_through_down"].mean() * 100
        rejected_pct = sub["rejected_up"].mean() * 100
        print(f"    {label}: {len(sub)} bars | broke through: {through_pct:.1f}% | rejected: {rejected_pct:.1f}%")

    print(f"\n  By zone width:")
    for q_low, q_high, label in [(0, 0.25, "Q1 narrow"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.01, "Q4 wide")]:
        low_val = res_brk["zone_width"].quantile(q_low)
        high_val = res_brk["zone_width"].quantile(min(q_high, 1.0))
        q_mask = (res_brk["zone_width"] >= low_val) & (res_brk["zone_width"] < high_val)
        sub = res_brk[q_mask]
        if len(sub) < 20:
            continue
        through_pct = sub["broke_through_down"].mean() * 100
        rejected_pct = sub["rejected_up"].mean() * 100
        print(f"    {label} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars | through: {through_pct:.1f}% | rejected: {rejected_pct:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
