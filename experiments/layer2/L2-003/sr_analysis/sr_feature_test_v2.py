"""S/R Feature Test V2 - With carry-forward logic

Same zone detection as V1, but when S/R is not found in current 8-bar window,
carry forward from previous snapshot.

S/R levels persist until new ones replace them.

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_feature_test_v2.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7  # 8 bars: t-7 to t

print("=" * 70)
print("S/R FEATURE TEST V2 - With carry-forward")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
N = len(ohlcv)

print(f"Loaded: {N} bars")


def find_zones(highs, lows):
    """Find support and resistance zones from bar data.

    Support = overlapping price area of non-adjacent bars where bars between went HIGHER
    Resistance = overlapping price area of non-adjacent bars where bars between went LOWER
    """
    n = len(highs)
    support_overlaps = []
    resistance_overlaps = []

    for i in range(n):
        for j in range(i + 2, n):
            overlap_low = max(lows[i], lows[j])
            overlap_high = min(highs[i], highs[j])

            if overlap_low > overlap_high:
                continue

            between_highs = highs[i+1:j]
            between_lows = lows[i+1:j]

            if len(between_highs) == 0:
                continue

            max_between = np.max(between_highs)
            min_between = np.min(between_lows)

            if max_between > overlap_high:
                support_overlaps.append((overlap_low, overlap_high))

            if min_between < overlap_low:
                resistance_overlaps.append((overlap_low, overlap_high))

    sup_low, sup_high = None, None
    res_low, res_high = None, None

    if support_overlaps:
        sorted_sup = sorted(support_overlaps, key=lambda x: x[0])
        sup_low = sorted_sup[0][0]
        sup_high = sorted_sup[0][1]
        for s in sorted_sup[1:]:
            if s[0] <= sup_high:
                sup_high = max(sup_high, s[1])
            else:
                break

    if resistance_overlaps:
        sorted_res = sorted(resistance_overlaps, key=lambda x: x[1], reverse=True)
        res_high = sorted_res[0][1]
        res_low = sorted_res[0][0]
        for r in sorted_res[1:]:
            if r[1] >= res_low:
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

    if sup_low is not None and sup_high is not None:
        for i in range(n - 1):
            if lows[i] <= sup_high and lows[i] >= sup_low:
                for j in range(i + 1, n):
                    if closes[j] > sup_high:
                        bars = j - i
                        move = (closes[j] - sup_high) / sup_high * 10000
                        speed = move / bars
                        if speed > recovery_up:
                            recovery_up = speed
                        break

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
# Compute features with carry-forward
# =========================================================================
print("\nComputing S/R features with carry-forward...")

results = []

# Carry-forward state
prev_sup_low = None
prev_sup_high = None
prev_res_low = None
prev_res_high = None

# Stats
found_both = 0
found_sup_only = 0
found_res_only = 0
found_none = 0
carried_sup = 0
carried_res = 0
carried_both = 0

for t in range(LOOKBACK + 1, N - 3):
    w_high = high[t-LOOKBACK:t+1]
    w_low = low[t-LOOKBACK:t+1]
    w_close = close[t-LOOKBACK:t+1]
    cur_price = close[t]

    # Find zones in current window
    sup_low, sup_high, res_low, res_high = find_zones(w_high, w_low)

    has_new_sup = sup_low is not None
    has_new_res = res_low is not None

    # Track what we found vs carried
    source_sup = "new"
    source_res = "new"

    # Carry-forward logic
    if has_new_sup and has_new_res:
        found_both += 1
    elif has_new_sup and not has_new_res:
        found_sup_only += 1
        if prev_res_low is not None:
            res_low = prev_res_low
            res_high = prev_res_high
            source_res = "carried"
            carried_res += 1
    elif not has_new_sup and has_new_res:
        found_res_only += 1
        if prev_sup_low is not None:
            sup_low = prev_sup_low
            sup_high = prev_sup_high
            source_sup = "carried"
            carried_sup += 1
    else:
        found_none += 1
        if prev_sup_low is not None:
            sup_low = prev_sup_low
            sup_high = prev_sup_high
            source_sup = "carried"
        if prev_res_low is not None:
            res_low = prev_res_low
            res_high = prev_res_high
            source_res = "carried"
        if prev_sup_low is not None or prev_res_low is not None:
            carried_both += 1

    # Update carry-forward state (only update with new values, keep old if nothing new)
    if has_new_sup:
        prev_sup_low = sup_low
        prev_sup_high = sup_high
    if has_new_res:
        prev_res_low = res_low
        prev_res_high = res_high

    # Now compute features
    has_support = sup_low is not None
    has_resistance = res_low is not None
    has_zone = has_support and has_resistance and res_low > sup_high

    row = {
        "date": ohlcv.index[t],
        "current_price": cur_price,
        "has_support": has_support,
        "has_resistance": has_resistance,
        "has_zone": has_zone,
        "source_sup": source_sup if has_support else "none",
        "source_res": source_res if has_resistance else "none",
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

    # Outcomes
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

total = len(df)
print(f"\nTotal bars: {total}")
print(f"  Has support: {df['has_support'].sum()} ({df['has_support'].mean()*100:.1f}%)")
print(f"  Has resistance: {df['has_resistance'].sum()} ({df['has_resistance'].mean()*100:.1f}%)")
print(f"  Has full zone: {df['has_zone'].sum()} ({df['has_zone'].mean()*100:.1f}%)")

print(f"\nZone detection source:")
print(f"  Found both in window:       {found_both} ({found_both/total*100:.1f}%)")
print(f"  Found support only:         {found_sup_only} ({found_sup_only/total*100:.1f}%)")
print(f"  Found resistance only:      {found_res_only} ({found_res_only/total*100:.1f}%)")
print(f"  Found neither:              {found_none} ({found_none/total*100:.1f}%)")
print(f"  Carried support forward:    {carried_sup} ({carried_sup/total*100:.1f}%)")
print(f"  Carried resistance forward: {carried_res} ({carried_res/total*100:.1f}%)")
print(f"  Carried both forward:       {carried_both} ({carried_both/total*100:.1f}%)")

# Compare: new vs carried zones
print(f"\n{'='*70}")
print("NEW vs CARRIED: Does carry-forward maintain the edge?")
print(f"{'='*70}")

zdf = df[df["has_zone"]].copy()
print(f"\nBars with full zone: {len(zdf)} ({len(zdf)/total*100:.1f}%)")

# Split by source
new_both = zdf[(zdf["source_sup"] == "new") & (zdf["source_res"] == "new")]
any_carried = zdf[(zdf["source_sup"] == "carried") | (zdf["source_res"] == "carried")]

print(f"  New S/R (both from current window): {len(new_both)}")
print(f"  Carried (at least one from previous): {len(any_carried)}")

for label, subset in [("NEW S/R", new_both), ("CARRIED S/R", any_carried)]:
    if len(subset) == 0:
        continue

    at_sup = subset[subset["distance_to_support"] <= 5]
    at_res = subset[subset["distance_to_resistance"] <= 5]

    print(f"\n  --- {label} ---")
    if len(at_sup) > 20:
        bounce = (at_sup["next1_close"] > at_sup["current_price"]).mean() * 100
        print(f"    At support: {len(at_sup)} bars, bounce UP = {bounce:.1f}%")
    if len(at_res) > 20:
        bounce = (at_res["next1_close"] < at_res["current_price"]).mean() * 100
        print(f"    At resistance: {len(at_res)} bars, bounce DOWN = {bounce:.1f}%")

# =========================================================================
# Full outcome analysis (same as V1 but with carry-forward)
# =========================================================================
print(f"\n{'='*70}")
print("OUTCOME: Does price respect support and resistance?")
print(f"{'='*70}")

if len(zdf) > 0:
    at_support = zdf["distance_to_support"] <= 5
    at_resistance = zdf["distance_to_resistance"] <= 5
    in_middle = (~at_support) & (~at_resistance) & (zdf["distance_to_support"] > 0) & (zdf["distance_to_resistance"] > 0)

    print(f"\n  Price position:")
    print(f"    At support (within 5 bps):    {at_support.sum()} ({at_support.mean()*100:.1f}%)")
    print(f"    At resistance (within 5 bps): {at_resistance.sum()} ({at_resistance.mean()*100:.1f}%)")
    print(f"    In middle:                    {in_middle.sum()} ({in_middle.mean()*100:.1f}%)")

    sup_bars = zdf[at_support]
    if len(sup_bars) > 0:
        bounce_up_1 = (sup_bars["next1_close"] > sup_bars["current_price"]).mean() * 100
        bounce_up_3 = (sup_bars["next3_close"] > sup_bars["current_price"]).mean() * 100
        stayed_above = (sup_bars["next1_low"] >= sup_bars["support_range_low"]).mean() * 100

        print(f"\n  AT SUPPORT ({len(sup_bars)} bars):")
        print(f"    Next bar close UP:        {bounce_up_1:.1f}%")
        print(f"    3 bars later close UP:    {bounce_up_3:.1f}%")
        print(f"    Stayed above support low: {stayed_above:.1f}%")

    res_bars = zdf[at_resistance]
    if len(res_bars) > 0:
        bounce_dn_1 = (res_bars["next1_close"] < res_bars["current_price"]).mean() * 100
        bounce_dn_3 = (res_bars["next3_close"] < res_bars["current_price"]).mean() * 100
        stayed_below = (res_bars["next1_high"] <= res_bars["resistance_range_high"]).mean() * 100

        print(f"\n  AT RESISTANCE ({len(res_bars)} bars):")
        print(f"    Next bar close DOWN:        {bounce_dn_1:.1f}%")
        print(f"    3 bars later close DOWN:    {bounce_dn_3:.1f}%")
        print(f"    Stayed below resistance hi: {stayed_below:.1f}%")

    mid_bars = zdf[in_middle]
    if len(mid_bars) > 0:
        up_1 = (mid_bars["next1_close"] > mid_bars["current_price"]).mean() * 100
        print(f"\n  IN MIDDLE ({len(mid_bars)} bars):")
        print(f"    Next bar close UP: {up_1:.1f}%")

# =========================================================================
# By feature value
# =========================================================================
print(f"\n{'='*70}")
print("BY FEATURE: How does each feature affect bounce rate?")
print(f"{'='*70}")

if len(zdf) > 0:
    # By zone_width
    print(f"\n--- By zone_width (at support) ---")
    sup_bars = zdf[zdf["distance_to_support"] <= 5]
    for q_low, q_high, label in [(0, 0.25, "Q1 narrow"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.01, "Q4 wide")]:
        low_val = sup_bars["zone_width"].quantile(q_low)
        high_val = sup_bars["zone_width"].quantile(min(q_high, 1.0))
        mask = (sup_bars["zone_width"] >= low_val) & (sup_bars["zone_width"] < high_val)
        sub = sup_bars[mask]
        if len(sub) < 20:
            continue
        bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
        print(f"  {label} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # By support_retest count
    print(f"\n--- By support_retest count (at support) ---")
    for tc in [1, 2, 3, 4, 5]:
        sub = sup_bars[sup_bars["support_retest"] == tc]
        if len(sub) < 20:
            continue
        bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
        print(f"  {tc} touches: {len(sub)} bars, bounce UP = {bounce:.1f}%")
    sub = sup_bars[sup_bars["support_retest"] >= 6]
    if len(sub) >= 20:
        bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
        print(f"  6+ touches: {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # By recovery_up speed
    print(f"\n--- By recovery_up speed (at support) ---")
    no_recovery = sup_bars[sup_bars["recovery_up"] == 0]
    has_recovery = sup_bars[sup_bars["recovery_up"] > 0]
    if len(no_recovery) >= 20:
        bounce = (no_recovery["next1_close"] > no_recovery["current_price"]).mean() * 100
        print(f"  No recovery (0): {len(no_recovery)} bars, bounce UP = {bounce:.1f}%")
    if len(has_recovery) >= 20:
        median_speed = has_recovery["recovery_up"].median()
        slow = has_recovery[has_recovery["recovery_up"] <= median_speed]
        fast = has_recovery[has_recovery["recovery_up"] > median_speed]
        if len(slow) >= 20:
            bounce = (slow["next1_close"] > slow["current_price"]).mean() * 100
            print(f"  Slow (<={median_speed:.1f} bps/bar): {len(slow)} bars, bounce UP = {bounce:.1f}%")
        if len(fast) >= 20:
            bounce = (fast["next1_close"] > fast["current_price"]).mean() * 100
            print(f"  Fast (>{median_speed:.1f} bps/bar): {len(fast)} bars, bounce UP = {bounce:.1f}%")

    # By recovery_down speed (at resistance)
    print(f"\n--- By recovery_down speed (at resistance) ---")
    res_bars = zdf[zdf["distance_to_resistance"] <= 5]
    no_recovery = res_bars[res_bars["recovery_down"] == 0]
    has_recovery = res_bars[res_bars["recovery_down"] > 0]
    if len(no_recovery) >= 20:
        bounce = (no_recovery["next1_close"] < no_recovery["current_price"]).mean() * 100
        print(f"  No recovery (0): {len(no_recovery)} bars, bounce DOWN = {bounce:.1f}%")
    if len(has_recovery) >= 20:
        median_speed = has_recovery["recovery_down"].median()
        slow = has_recovery[has_recovery["recovery_down"] <= median_speed]
        fast = has_recovery[has_recovery["recovery_down"] > median_speed]
        if len(slow) >= 20:
            bounce = (slow["next1_close"] < slow["current_price"]).mean() * 100
            print(f"  Slow (<={median_speed:.1f} bps/bar): {len(slow)} bars, bounce DOWN = {bounce:.1f}%")
        if len(fast) >= 20:
            bounce = (fast["next1_close"] < fast["current_price"]).mean() * 100
            print(f"  Fast (>{median_speed:.1f} bps/bar): {len(fast)} bars, bounce DOWN = {bounce:.1f}%")

# =========================================================================
# Train vs Test
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
# Baseline
# =========================================================================
print(f"\n{'='*70}")
print("BASELINE")
print(f"{'='*70}")

no_zone = df[~df["has_zone"]]
if len(no_zone) > 0:
    up = (no_zone["next1_close"] > no_zone["current_price"]).mean() * 100
    print(f"  No zone bars: {len(no_zone)}, next bar UP = {up:.1f}%")

all_up = (df["next1_close"] > df["current_price"]).mean() * 100
print(f"  All bars: {len(df)}, next bar UP = {all_up:.1f}%")

# =========================================================================
# V1 vs V2 comparison
# =========================================================================
print(f"\n{'='*70}")
print("V1 (no carry-forward) vs V2 (with carry-forward)")
print(f"{'='*70}")
print(f"  V1: 43,713 bars with zone (20.8%)")
print(f"  V2: {df['has_zone'].sum()} bars with zone ({df['has_zone'].mean()*100:.1f}%)")
print(f"  Carry-forward added {df['has_zone'].sum() - 43713} more zone bars")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
