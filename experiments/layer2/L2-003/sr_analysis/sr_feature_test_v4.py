"""S/R Feature Test V4 - Compare Option A (extremes) vs Option B (most touches)

Option A: Support = lowest cluster, Resistance = highest cluster (always extremes)
Option B: Support/Resistance = cluster with most touches (even if not extreme)

Zone detection uses natural gaps (biggest gap in sorted lows/highs).
After finding clusters, count touches for each cluster.

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_feature_test_v4.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7

print("=" * 70)
print("S/R FEATURE TEST V4 - Option A (extremes) vs Option B (most touches)")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
N = len(ohlcv)

print(f"Loaded: {N} bars")


def find_all_clusters(sorted_prices):
    """Split sorted prices into clusters using natural gaps.

    Find ALL gaps, split at the biggest one first, then recursively
    split each sub-group at its biggest gap. Stop when cluster has <= 2 members.

    Returns list of (cluster_low, cluster_high, member_count).
    """
    if len(sorted_prices) <= 1:
        return [(sorted_prices[0], sorted_prices[0], 1)] if len(sorted_prices) == 1 else []

    gaps = np.diff(sorted_prices)
    if len(gaps) == 0:
        return [(sorted_prices[0], sorted_prices[-1], len(sorted_prices))]

    # Find biggest gap
    biggest_idx = np.argmax(gaps)
    biggest_gap = gaps[biggest_idx]

    # If biggest gap is very small relative to the range, don't split further
    total_range = sorted_prices[-1] - sorted_prices[0]
    if total_range == 0 or biggest_gap < total_range * 0.15:
        return [(sorted_prices[0], sorted_prices[-1], len(sorted_prices))]

    # Split into two groups
    left = sorted_prices[:biggest_idx + 1]
    right = sorted_prices[biggest_idx + 1:]

    # Recursively split each group
    left_clusters = find_all_clusters(left)
    right_clusters = find_all_clusters(right)

    return left_clusters + right_clusters


def count_bar_touches(highs, lows, range_low, range_high):
    """Count how many bars have their high or low within the range."""
    count = 0
    for i in range(len(highs)):
        # Bar touches range if any part of bar overlaps with range
        if lows[i] <= range_high and highs[i] >= range_low:
            count += 1
    return count


def find_zones_option_a(highs, lows):
    """Option A: Always use extremes.

    Support = lowest cluster of lows.
    Resistance = highest cluster of highs.
    """
    sorted_lows = np.sort(lows)
    low_gaps = np.diff(sorted_lows)
    if len(low_gaps) == 0:
        return None, None, None, None, 0, 0

    biggest_low_gap = np.argmax(low_gaps)
    sup_low = sorted_lows[0]
    sup_high = sorted_lows[biggest_low_gap]

    sorted_highs = np.sort(highs)
    high_gaps = np.diff(sorted_highs)
    biggest_high_gap = np.argmax(high_gaps)
    res_low = sorted_highs[biggest_high_gap + 1]
    res_high = sorted_highs[-1]

    if res_low <= sup_high:
        return None, None, None, None, 0, 0

    sup_touches = count_bar_touches(highs, lows, sup_low, sup_high)
    res_touches = count_bar_touches(highs, lows, res_low, res_high)

    return sup_low, sup_high, res_low, res_high, sup_touches, res_touches


def find_zones_option_b(highs, lows):
    """Option B: Use cluster with most touches.

    1. Find all clusters of lows and highs using natural gaps
    2. For each cluster, count how many bars touch it
    3. Support = cluster of lows with most touches (if tie, take lowest)
    4. Resistance = cluster of highs with most touches (if tie, take highest)
    """
    sorted_lows = np.sort(lows)
    sorted_highs = np.sort(highs)

    low_clusters = find_all_clusters(sorted_lows)
    high_clusters = find_all_clusters(sorted_highs)

    if not low_clusters or not high_clusters:
        return None, None, None, None, 0, 0

    # Count touches for each low cluster
    best_sup = None
    best_sup_touches = -1
    for cl, ch, count in low_clusters:
        touches = count_bar_touches(highs, lows, cl, ch)
        # More touches wins. If tie, lower cluster wins (stronger floor)
        if touches > best_sup_touches or (touches == best_sup_touches and (best_sup is None or cl < best_sup[0])):
            best_sup = (cl, ch)
            best_sup_touches = touches

    # Count touches for each high cluster
    best_res = None
    best_res_touches = -1
    for cl, ch, count in high_clusters:
        touches = count_bar_touches(highs, lows, cl, ch)
        # More touches wins. If tie, higher cluster wins (stronger ceiling)
        if touches > best_res_touches or (touches == best_res_touches and (best_res is None or ch > best_res[1])):
            best_res = (cl, ch)
            best_res_touches = touches

    if best_sup is None or best_res is None:
        return None, None, None, None, 0, 0

    sup_low, sup_high = best_sup
    res_low, res_high = best_res

    if res_low <= sup_high:
        return None, None, None, None, 0, 0

    return sup_low, sup_high, res_low, res_high, best_sup_touches, best_res_touches


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
# Compute features for both options
# =========================================================================
print("\nComputing S/R features for Option A and Option B...")

results_a = []
results_b = []

for t in range(LOOKBACK + 1, N - 3):
    w_high = high[t-LOOKBACK:t+1]
    w_low = low[t-LOOKBACK:t+1]
    w_close = close[t-LOOKBACK:t+1]
    cur_price = close[t]

    # Option A
    sl_a, sh_a, rl_a, rh_a, st_a, rt_a = find_zones_option_a(w_high, w_low)
    # Option B
    sl_b, sh_b, rl_b, rh_b, st_b, rt_b = find_zones_option_b(w_high, w_low)

    for option, sl, sh, rl, rh, st, rt, results in [
        ("A", sl_a, sh_a, rl_a, rh_a, st_a, rt_a, results_a),
        ("B", sl_b, sh_b, rl_b, rh_b, st_b, rt_b, results_b),
    ]:
        has_zone = sl is not None and rl is not None

        row = {
            "date": ohlcv.index[t],
            "current_price": cur_price,
            "has_zone": has_zone,
        }

        if has_zone:
            row["zone_width"] = (rl - sh) / sh * 10000
            row["support_range_low"] = sl
            row["support_range_high"] = sh
            row["resistance_range_low"] = rl
            row["resistance_range_high"] = rh
            row["support_retest"] = st
            row["resistance_retest"] = rt
            row["distance_to_support"] = (cur_price - sh) / sh * 10000
            row["distance_to_resistance"] = (rl - cur_price) / rl * 10000

            rec_up, rec_down = compute_recovery_speeds(w_high, w_low, w_close, sl, sh, rl, rh)
            row["recovery_up"] = rec_up
            row["recovery_down"] = rec_down

        row["next1_close"] = close[t+1]
        row["next3_close"] = close[t+3]
        row["next1_high"] = high[t+1]
        row["next1_low"] = low[t+1]

        results.append(row)

    if t % 50000 == 0:
        za = sum(1 for r in results_a if r["has_zone"])
        zb = sum(1 for r in results_b if r["has_zone"])
        print(f"  {t}/{N}... A zones={za}, B zones={zb}")

df_a = pd.DataFrame(results_a).set_index(pd.to_datetime([r["date"] for r in results_a]))
df_b = pd.DataFrame(results_b).set_index(pd.to_datetime([r["date"] for r in results_b]))

total = len(df_a)
print(f"\nTotal bars: {total}")
print(f"  Option A zones: {df_a['has_zone'].sum()} ({df_a['has_zone'].mean()*100:.1f}%)")
print(f"  Option B zones: {df_b['has_zone'].sum()} ({df_b['has_zone'].mean()*100:.1f}%)")

# =========================================================================
# Compare both options
# =========================================================================
print(f"\n{'='*70}")
print("COMPARISON: Option A (extremes) vs Option B (most touches)")
print(f"{'='*70}")

for label, df in [("OPTION A (extremes)", df_a), ("OPTION B (most touches)", df_b)]:
    zdf = df[df["has_zone"]].copy()
    if len(zdf) == 0:
        print(f"\n{label}: no zones found")
        continue

    at_sup = zdf[zdf["distance_to_support"] <= 5]
    at_res = zdf[zdf["distance_to_resistance"] <= 5]
    in_mid = zdf[(zdf["distance_to_support"] > 5) & (zdf["distance_to_resistance"] > 5) &
                 (zdf["distance_to_support"] > 0) & (zdf["distance_to_resistance"] > 0)]

    print(f"\n--- {label} ---")
    print(f"  Zones: {len(zdf)} ({len(zdf)/total*100:.1f}%)")
    print(f"  Zone width: mean={zdf['zone_width'].mean():.1f}, median={zdf['zone_width'].median():.1f} bps")

    if len(at_sup) > 0:
        bounce = (at_sup["next1_close"] > at_sup["current_price"]).mean() * 100
        bounce3 = (at_sup["next3_close"] > at_sup["current_price"]).mean() * 100
        stayed = (at_sup["next1_low"] >= at_sup["support_range_low"]).mean() * 100
        print(f"  At support: {len(at_sup)} bars ({len(at_sup)/len(zdf)*100:.1f}%)")
        print(f"    Bounce UP 1bar: {bounce:.1f}%")
        print(f"    Bounce UP 3bar: {bounce3:.1f}%")
        print(f"    Stayed above:   {stayed:.1f}%")

    if len(at_res) > 0:
        bounce = (at_res["next1_close"] < at_res["current_price"]).mean() * 100
        bounce3 = (at_res["next3_close"] < at_res["current_price"]).mean() * 100
        stayed = (at_res["next1_high"] <= at_res["resistance_range_high"]).mean() * 100
        print(f"  At resistance: {len(at_res)} bars ({len(at_res)/len(zdf)*100:.1f}%)")
        print(f"    Bounce DOWN 1bar: {bounce:.1f}%")
        print(f"    Bounce DOWN 3bar: {bounce3:.1f}%")
        print(f"    Stayed below:     {stayed:.1f}%")

    if len(in_mid) > 0:
        up = (in_mid["next1_close"] > in_mid["current_price"]).mean() * 100
        print(f"  In middle: {len(in_mid)} bars, next UP: {up:.1f}%")

    # By support retest count
    print(f"\n  Support retest count (at support):")
    if len(at_sup) > 0:
        for tc in sorted(at_sup["support_retest"].unique()):
            sub = at_sup[at_sup["support_retest"] == tc]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
            print(f"    {int(tc)} touches: {len(sub)} bars, bounce UP = {bounce:.1f}%")

    print(f"\n  Resistance retest count (at resistance):")
    if len(at_res) > 0:
        for tc in sorted(at_res["resistance_retest"].unique()):
            sub = at_res[at_res["resistance_retest"] == tc]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] < sub["current_price"]).mean() * 100
            print(f"    {int(tc)} touches: {len(sub)} bars, bounce DOWN = {bounce:.1f}%")

    # By zone width
    print(f"\n  Zone width (at support):")
    if len(at_sup) > 0:
        for q_low, q_high, qlabel in [(0, 0.25, "Q1 narrow"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.01, "Q4 wide")]:
            low_val = at_sup["zone_width"].quantile(q_low)
            high_val = at_sup["zone_width"].quantile(min(q_high, 1.0))
            mask = (at_sup["zone_width"] >= low_val) & (at_sup["zone_width"] < high_val)
            sub = at_sup[mask]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
            print(f"    {qlabel} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # Train vs Test
    print(f"\n  Train vs Test:")
    for period, start, end in [("TRAIN 2020-2023", "2020-01-01", "2023-12-31"),
                                ("TEST 2024-2025", "2024-01-01", "2025-12-31")]:
        p_mask = (zdf.index >= start) & (zdf.index <= end)
        sub = zdf[p_mask]
        if len(sub) == 0:
            continue
        p_sup = sub[sub["distance_to_support"] <= 5]
        p_res = sub[sub["distance_to_resistance"] <= 5]
        sb = (p_sup["next1_close"] > p_sup["current_price"]).mean() * 100 if len(p_sup) > 0 else 0
        rb = (p_res["next1_close"] < p_res["current_price"]).mean() * 100 if len(p_res) > 0 else 0
        print(f"    {period}: sup {len(p_sup)} bars bounce {sb:.1f}% | res {len(p_res)} bars bounce {rb:.1f}%")

# Baseline
print(f"\n{'='*70}")
print("BASELINE")
print(f"{'='*70}")
all_up = (df_a["next1_close"] > df_a["current_price"]).mean() * 100
print(f"  All bars: next bar UP = {all_up:.1f}%")

# =========================================================================
# Show example
# =========================================================================
print(f"\n{'='*70}")
print("EXAMPLE: Real bar comparison")
print(f"{'='*70}")

# Use t=1000 example
t = 1000
w_high = high[t-7:t+1]
w_low = low[t-7:t+1]
w_close = close[t-7:t+1]

print(f"\nBar t={t}, date={ohlcv.index[t]}")
print(f"8-bar window:")
for i in range(8):
    print(f"  Bar {i-7}: high={w_high[i]:.2f}  low={w_low[i]:.2f}  close={w_close[i]:.2f}")

sl_a, sh_a, rl_a, rh_a, st_a, rt_a = find_zones_option_a(w_high, w_low)
sl_b, sh_b, rl_b, rh_b, st_b, rt_b = find_zones_option_b(w_high, w_low)

print(f"\nOption A (extremes):")
if sl_a is not None:
    print(f"  Support:    {sl_a:.2f} - {sh_a:.2f} ({st_a} touches)")
    print(f"  Resistance: {rl_a:.2f} - {rh_a:.2f} ({rt_a} touches)")
    print(f"  Zone width: {(rl_a - sh_a) / sh_a * 10000:.1f} bps")
else:
    print(f"  No zone")

print(f"\nOption B (most touches):")
if sl_b is not None:
    print(f"  Support:    {sl_b:.2f} - {sh_b:.2f} ({st_b} touches)")
    print(f"  Resistance: {rl_b:.2f} - {rh_b:.2f} ({rt_b} touches)")
    print(f"  Zone width: {(rl_b - sh_b) / sh_b * 10000:.1f} bps")
else:
    print(f"  No zone")

# Show all clusters for this window
print(f"\nAll low clusters:")
sorted_lows = np.sort(w_low)
low_clusters = find_all_clusters(sorted_lows)
for cl, ch, cnt in low_clusters:
    touches = count_bar_touches(w_high, w_low, cl, ch)
    print(f"  {cl:.2f} - {ch:.2f} ({cnt} members, {touches} bar touches)")

print(f"\nAll high clusters:")
sorted_highs = np.sort(w_high)
high_clusters = find_all_clusters(sorted_highs)
for cl, ch, cnt in high_clusters:
    touches = count_bar_touches(w_high, w_low, cl, ch)
    print(f"  {cl:.2f} - {ch:.2f} ({cnt} members, {touches} bar touches)")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
