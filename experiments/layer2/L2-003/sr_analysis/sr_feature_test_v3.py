"""S/R Feature Test V3 - Correct zone detection using clustering

Support = where multiple bars have their LOWS (price floor)
Resistance = where multiple bars have their HIGHS (price ceiling)

Approach:
  1. Collect all lows from 8 bars -> cluster nearby lows -> support range
  2. Collect all highs from 8 bars -> cluster nearby highs -> resistance range
  3. Zone = between support range and resistance range

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_feature_test_v3.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7  # 8 bars: t-7 to t

print("=" * 70)
print("S/R FEATURE TEST V3 - Clustering approach")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
N = len(ohlcv)

print(f"Loaded: {N} bars")


def find_zones_v3(highs, lows):
    """Find support and resistance using natural gaps in the data.

    Support = cluster of lows below the biggest gap (the floor)
    Resistance = cluster of highs above the biggest gap (the ceiling)

    No fixed tolerance — each window's own gaps define the clusters.
    """
    # --- SUPPORT: cluster of lowest lows ---
    sorted_lows = np.sort(lows)
    low_gaps = np.diff(sorted_lows)

    if len(low_gaps) == 0:
        return None, None, None, None

    # Find the biggest gap — everything below it is support
    biggest_gap_idx = np.argmax(low_gaps)
    sup_low = sorted_lows[0]
    sup_high = sorted_lows[biggest_gap_idx]
    sup_count = biggest_gap_idx + 1

    # --- RESISTANCE: cluster of highest highs ---
    sorted_highs = np.sort(highs)
    high_gaps = np.diff(sorted_highs)

    # Find the biggest gap — everything above it is resistance
    biggest_gap_idx = np.argmax(high_gaps)
    res_low = sorted_highs[biggest_gap_idx + 1]
    res_high = sorted_highs[-1]
    res_count = len(sorted_highs) - biggest_gap_idx - 1

    # Zone only valid if resistance is above support
    if res_low <= sup_high:
        return None, None, None, None

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
# Compute features
# =========================================================================
print("\nComputing S/R features (V3 clustering)...")

results = []

# Carry-forward state
prev_sup_low = None
prev_sup_high = None
prev_res_low = None
prev_res_high = None

for t in range(LOOKBACK + 1, N - 3):
    w_high = high[t-LOOKBACK:t+1]
    w_low = low[t-LOOKBACK:t+1]
    w_close = close[t-LOOKBACK:t+1]
    cur_price = close[t]

    # Find zones
    sup_low, sup_high, res_low, res_high = find_zones_v3(w_high, w_low)

    has_new_sup = sup_low is not None
    has_new_res = res_low is not None

    # Carry-forward if missing
    if not has_new_sup and prev_sup_low is not None:
        sup_low = prev_sup_low
        sup_high = prev_sup_high
    if not has_new_res and prev_res_low is not None:
        res_low = prev_res_low
        res_high = prev_res_high

    # Update carry-forward
    if has_new_sup:
        prev_sup_low = sup_low
        prev_sup_high = sup_high
    if has_new_res:
        prev_res_low = res_low
        prev_res_high = res_high

    has_support = sup_low is not None
    has_resistance = res_low is not None
    has_zone = has_support and has_resistance and res_low > sup_high

    row = {
        "date": ohlcv.index[t],
        "current_price": cur_price,
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
        row["support_retest"] = 0
        row["resistance_retest"] = 0
        row["distance_to_support"] = np.nan
        row["distance_to_resistance"] = np.nan
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
print(f"  Has full zone: {df['has_zone'].sum()} ({df['has_zone'].mean()*100:.1f}%)")

# =========================================================================
# Verify with example
# =========================================================================
print(f"\n{'='*70}")
print("VERIFICATION: Example zone detection")
print(f"{'='*70}")

# Show a few examples of detected zones
zone_bars = df[df["has_zone"]]
if len(zone_bars) > 0:
    sample = zone_bars.iloc[100]
    t_idx = ohlcv.index.get_loc(sample.name)
    print(f"\n  Example at {sample.name}:")
    print(f"  8-bar window:")
    for i in range(t_idx - LOOKBACK, t_idx + 1):
        print(f"    Bar {i-t_idx}: high={high[i]:.0f}  low={low[i]:.0f}  close={close[i]:.0f}")
    print(f"  Support range:  {sample['support_range_low']:.0f} - {sample['support_range_high']:.0f}")
    print(f"  Resistance range: {sample['resistance_range_low']:.0f} - {sample['resistance_range_high']:.0f}")
    print(f"  Zone width: {sample['zone_width']:.1f} bps")
    print(f"  Support retest: {sample['support_retest']:.0f}")
    print(f"  Resistance retest: {sample['resistance_retest']:.0f}")
    print(f"  Distance to support: {sample['distance_to_support']:.1f} bps")
    print(f"  Distance to resistance: {sample['distance_to_resistance']:.1f} bps")

# =========================================================================
# Outcome analysis
# =========================================================================
print(f"\n{'='*70}")
print("OUTCOME: Does price respect support and resistance?")
print(f"{'='*70}")

zdf = df[df["has_zone"]].copy()
print(f"\nBars with full zone: {len(zdf)}")

if len(zdf) > 0:
    at_support = zdf["distance_to_support"] <= 5
    at_resistance = zdf["distance_to_resistance"] <= 5
    in_middle = (~at_support) & (~at_resistance) & (zdf["distance_to_support"] > 0) & (zdf["distance_to_resistance"] > 0)
    below_support = zdf["distance_to_support"] < 0
    above_resistance = zdf["distance_to_resistance"] < 0

    print(f"\n  Price position:")
    print(f"    At support (within 5 bps):    {at_support.sum()} ({at_support.mean()*100:.1f}%)")
    print(f"    At resistance (within 5 bps): {at_resistance.sum()} ({at_resistance.mean()*100:.1f}%)")
    print(f"    In middle:                    {in_middle.sum()} ({in_middle.mean()*100:.1f}%)")
    print(f"    Below support:                {below_support.sum()} ({below_support.mean()*100:.1f}%)")
    print(f"    Above resistance:             {above_resistance.sum()} ({above_resistance.mean()*100:.1f}%)")

    # At support
    sup_bars = zdf[at_support]
    if len(sup_bars) > 0:
        bounce_up_1 = (sup_bars["next1_close"] > sup_bars["current_price"]).mean() * 100
        bounce_up_3 = (sup_bars["next3_close"] > sup_bars["current_price"]).mean() * 100
        stayed = (sup_bars["next1_low"] >= sup_bars["support_range_low"]).mean() * 100
        broke = (sup_bars["next1_low"] < sup_bars["support_range_low"]).mean() * 100

        print(f"\n  AT SUPPORT ({len(sup_bars)} bars):")
        print(f"    Next bar close UP:        {bounce_up_1:.1f}%")
        print(f"    3 bars later close UP:    {bounce_up_3:.1f}%")
        print(f"    Stayed above support low: {stayed:.1f}%")
        print(f"    Broke below support low:  {broke:.1f}%")

    # At resistance
    res_bars = zdf[at_resistance]
    if len(res_bars) > 0:
        bounce_dn_1 = (res_bars["next1_close"] < res_bars["current_price"]).mean() * 100
        bounce_dn_3 = (res_bars["next3_close"] < res_bars["current_price"]).mean() * 100
        stayed = (res_bars["next1_high"] <= res_bars["resistance_range_high"]).mean() * 100
        broke = (res_bars["next1_high"] > res_bars["resistance_range_high"]).mean() * 100

        print(f"\n  AT RESISTANCE ({len(res_bars)} bars):")
        print(f"    Next bar close DOWN:        {bounce_dn_1:.1f}%")
        print(f"    3 bars later close DOWN:    {bounce_dn_3:.1f}%")
        print(f"    Stayed below resistance hi: {stayed:.1f}%")
        print(f"    Broke above resistance hi:  {broke:.1f}%")

    # In middle
    mid_bars = zdf[in_middle]
    if len(mid_bars) > 0:
        up_1 = (mid_bars["next1_close"] > mid_bars["current_price"]).mean() * 100
        print(f"\n  IN MIDDLE ({len(mid_bars)} bars):")
        print(f"    Next bar close UP: {up_1:.1f}%")

    # Below support
    bel_bars = zdf[below_support]
    if len(bel_bars) > 0:
        up_1 = (bel_bars["next1_close"] > bel_bars["current_price"]).mean() * 100
        print(f"\n  BELOW SUPPORT ({len(bel_bars)} bars):")
        print(f"    Next bar close UP: {up_1:.1f}%")

    # Above resistance
    abv_bars = zdf[above_resistance]
    if len(abv_bars) > 0:
        up_1 = (abv_bars["next1_close"] > abv_bars["current_price"]).mean() * 100
        print(f"\n  ABOVE RESISTANCE ({len(abv_bars)} bars):")
        print(f"    Next bar close UP: {up_1:.1f}%")

# =========================================================================
# By feature value
# =========================================================================
print(f"\n{'='*70}")
print("BY FEATURE: How does each feature affect bounce rate?")
print(f"{'='*70}")

if len(zdf) > 0:
    # Zone width
    print(f"\n--- By zone_width (at support) ---")
    sup_bars = zdf[zdf["distance_to_support"] <= 5]
    if len(sup_bars) > 0:
        for q_low, q_high, label in [(0, 0.25, "Q1 narrow"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.01, "Q4 wide")]:
            low_val = sup_bars["zone_width"].quantile(q_low)
            high_val = sup_bars["zone_width"].quantile(min(q_high, 1.0))
            mask = (sup_bars["zone_width"] >= low_val) & (sup_bars["zone_width"] < high_val)
            sub = sup_bars[mask]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
            print(f"  {label} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # Support retest count
    print(f"\n--- By support_retest count (at support) ---")
    if len(sup_bars) > 0:
        for tc in sorted(sup_bars["support_retest"].unique()):
            sub = sup_bars[sup_bars["support_retest"] == tc]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
            print(f"  {int(tc)} touches: {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # Resistance retest count
    print(f"\n--- By resistance_retest count (at resistance) ---")
    res_bars = zdf[zdf["distance_to_resistance"] <= 5]
    if len(res_bars) > 0:
        for tc in sorted(res_bars["resistance_retest"].unique()):
            sub = res_bars[res_bars["resistance_retest"] == tc]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] < sub["current_price"]).mean() * 100
            print(f"  {int(tc)} touches: {len(sub)} bars, bounce DOWN = {bounce:.1f}%")

    # Recovery up
    print(f"\n--- By recovery_up speed (at support) ---")
    if len(sup_bars) > 0:
        no_rec = sup_bars[sup_bars["recovery_up"] == 0]
        has_rec = sup_bars[sup_bars["recovery_up"] > 0]
        if len(no_rec) >= 20:
            bounce = (no_rec["next1_close"] > no_rec["current_price"]).mean() * 100
            print(f"  No recovery: {len(no_rec)} bars, bounce UP = {bounce:.1f}%")
        if len(has_rec) >= 40:
            med = has_rec["recovery_up"].median()
            slow = has_rec[has_rec["recovery_up"] <= med]
            fast = has_rec[has_rec["recovery_up"] > med]
            if len(slow) >= 20:
                bounce = (slow["next1_close"] > slow["current_price"]).mean() * 100
                print(f"  Slow (<={med:.1f}): {len(slow)} bars, bounce UP = {bounce:.1f}%")
            if len(fast) >= 20:
                bounce = (fast["next1_close"] > fast["current_price"]).mean() * 100
                print(f"  Fast (>{med:.1f}): {len(fast)} bars, bounce UP = {bounce:.1f}%")

    # Recovery down
    print(f"\n--- By recovery_down speed (at resistance) ---")
    if len(res_bars) > 0:
        no_rec = res_bars[res_bars["recovery_down"] == 0]
        has_rec = res_bars[res_bars["recovery_down"] > 0]
        if len(no_rec) >= 20:
            bounce = (no_rec["next1_close"] < no_rec["current_price"]).mean() * 100
            print(f"  No recovery: {len(no_rec)} bars, bounce DOWN = {bounce:.1f}%")
        if len(has_rec) >= 40:
            med = has_rec["recovery_down"].median()
            slow = has_rec[has_rec["recovery_down"] <= med]
            fast = has_rec[has_rec["recovery_down"] > med]
            if len(slow) >= 20:
                bounce = (slow["next1_close"] < slow["current_price"]).mean() * 100
                print(f"  Slow (<={med:.1f}): {len(slow)} bars, bounce DOWN = {bounce:.1f}%")
            if len(fast) >= 20:
                bounce = (fast["next1_close"] < fast["current_price"]).mean() * 100
                print(f"  Fast (>{med:.1f}): {len(fast)} bars, bounce DOWN = {bounce:.1f}%")

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
up = (no_zone["next1_close"] > no_zone["current_price"]).mean() * 100
print(f"  No zone: {len(no_zone)} bars, next bar UP = {up:.1f}%")

all_up = (df["next1_close"] > df["current_price"]).mean() * 100
print(f"  All bars: {len(df)}, next bar UP = {all_up:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
