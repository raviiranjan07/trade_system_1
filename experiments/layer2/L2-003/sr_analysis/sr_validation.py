"""S/R Feature Validation - Test if price respects S/R levels

Following SR_PLAN.md steps 1-6.

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_validation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7  # bars t-7 to t-1

print("=" * 70)
print("S/R FEATURE VALIDATION")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index
print(f"Loaded: {len(ohlcv)} bars, {ohlcv.index.min()} to {ohlcv.index.max()}")

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
open_ = ohlcv["open"].values
N = len(ohlcv)

# =========================================================================
# STEP 1: Compute S/R features for every bar
# =========================================================================
print(f"\n{'='*70}")
print("STEP 1: Computing S/R features for every bar")
print(f"{'='*70}")

results = []

for t in range(LOOKBACK + 1, N - 3):  # need 3 bars ahead for outcome
    bar_high = high[t]
    bar_low = low[t]
    bar_close = close[t]
    prev_close = close[t - 1]

    # Find all levels in lookback window
    levels = []

    for a in range(t - LOOKBACK, t):
        # --- Check high[a] as a level ---
        h_level = high[a]
        # Was it broken? (any bar between a+1 and t-1 went above it)
        broken_up = False
        for k in range(a + 1, t):
            if high[k] > h_level:
                broken_up = True
                break

        # Is bar t passing through this level?
        if broken_up and bar_low <= h_level <= bar_high:
            # Count touches: how many bars between a+1 and t-1 passed through this level
            touches = 0
            for k in range(a + 1, t):
                if low[k] <= h_level <= high[k]:
                    touches += 1

            # Is it support or resistance?
            if prev_close > h_level:
                role = "support"
            elif prev_close < h_level:
                role = "resistance"
            else:
                role = "neutral"

            dist = abs(bar_close - h_level) / h_level * 10000
            levels.append({
                "level": h_level,
                "source": "high",
                "bar_a": a,
                "role": role,
                "touches": touches,
                "dist_bps": dist,
            })

        # --- Check low[a] as a level ---
        l_level = low[a]
        # Was it broken? (any bar between a+1 and t-1 went below it)
        broken_down = False
        for k in range(a + 1, t):
            if low[k] < l_level:
                broken_down = True
                break

        if broken_down and bar_low <= l_level <= bar_high:
            touches = 0
            for k in range(a + 1, t):
                if low[k] <= l_level <= high[k]:
                    touches += 1

            if prev_close > l_level:
                role = "support"
            elif prev_close < l_level:
                role = "resistance"
            else:
                role = "neutral"

            dist = abs(bar_close - l_level) / l_level * 10000
            levels.append({
                "level": l_level,
                "source": "low",
                "bar_a": a,
                "role": role,
                "touches": touches,
                "dist_bps": dist,
            })

    # Pick nearest level to current price
    if not levels:
        results.append({
            "date": ohlcv.index[t],
            "bar_idx": t,
            "support_retest": 0,
            "resistance_retest": 0,
            "distance_to_level": np.nan,
            "touch_count": 0,
            "level_price": np.nan,
            "current_price": bar_close,
            # Outcomes (next 1,2,3 bars)
            "next1_close": close[t + 1],
            "next2_close": close[t + 2],
            "next3_close": close[t + 3],
            "next1_high": high[t + 1],
            "next2_high": high[t + 2],
            "next3_high": high[t + 3],
            "next1_low": low[t + 1],
            "next2_low": low[t + 2],
            "next3_low": low[t + 3],
        })
        continue

    # Nearest level
    nearest = min(levels, key=lambda x: x["dist_bps"])

    results.append({
        "date": ohlcv.index[t],
        "bar_idx": t,
        "support_retest": 1 if nearest["role"] == "support" else 0,
        "resistance_retest": 1 if nearest["role"] == "resistance" else 0,
        "distance_to_level": (bar_close - nearest["level"]) / nearest["level"] * 10000,
        "touch_count": nearest["touches"],
        "level_price": nearest["level"],
        "current_price": bar_close,
        "next1_close": close[t + 1],
        "next2_close": close[t + 2],
        "next3_close": close[t + 3],
        "next1_high": high[t + 1],
        "next2_high": high[t + 2],
        "next3_high": high[t + 3],
        "next1_low": low[t + 1],
        "next2_low": low[t + 2],
        "next3_low": low[t + 3],
    })

    if t % 50000 == 0:
        sup = sum(1 for r in results if r["support_retest"] == 1)
        res = sum(1 for r in results if r["resistance_retest"] == 1)
        none = sum(1 for r in results if r["support_retest"] == 0 and r["resistance_retest"] == 0)
        print(f"  {t}/{N} bars... support={sup}, resistance={res}, no_level={none}")

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
# STEP 2: Measure bounce behavior
# =========================================================================
print(f"\n{'='*70}")
print("STEP 2: Bounce behavior")
print(f"{'='*70}")

for name, mask in [("SUPPORT RETEST", df["support_retest"] == 1),
                    ("RESISTANCE RETEST", df["resistance_retest"] == 1)]:
    subset = df[mask]
    if len(subset) == 0:
        print(f"\n{name}: no data")
        continue

    print(f"\n--- {name} ({len(subset)} bars) ---")

    level = subset["level_price"]

    if name == "SUPPORT RETEST":
        # Support: expect price to stay ABOVE level (bounce up)
        # Bounce = next bar's low stays above level
        bounce_1 = (subset["next1_low"] >= level).mean() * 100
        bounce_2 = ((subset["next1_low"] >= level) & (subset["next2_low"] >= level)).mean() * 100
        bounce_3 = ((subset["next1_low"] >= level) & (subset["next2_low"] >= level) & (subset["next3_low"] >= level)).mean() * 100

        print(f"  Price stays above level (support holds):")
        print(f"    After 1 bar: {bounce_1:.1f}%")
        print(f"    After 2 bars: {bounce_2:.1f}%")
        print(f"    After 3 bars: {bounce_3:.1f}%")

        # Bounce size: how far up from level
        bounce_up_1 = ((subset["next1_high"] - level) / level * 10000)
        bounce_up_2 = ((np.maximum(subset["next1_high"], subset["next2_high"]) - level) / level * 10000)
        bounce_up_3 = ((np.maximum(subset["next1_high"], np.maximum(subset["next2_high"], subset["next3_high"])) - level) / level * 10000)

        print(f"  Max move UP from level (bps):")
        print(f"    1 bar:  mean={bounce_up_1.mean():.1f}, median={bounce_up_1.median():.1f}")
        print(f"    2 bars: mean={bounce_up_2.mean():.1f}, median={bounce_up_2.median():.1f}")
        print(f"    3 bars: mean={bounce_up_3.mean():.1f}, median={bounce_up_3.median():.1f}")

        # Break: how far below level
        break_down_1 = ((level - subset["next1_low"]) / level * 10000).clip(lower=0)
        break_down_3 = ((level - np.minimum(subset["next1_low"], np.minimum(subset["next2_low"], subset["next3_low"]))) / level * 10000).clip(lower=0)

        print(f"  Max move DOWN through level - break (bps):")
        print(f"    1 bar:  mean={break_down_1.mean():.1f}, median={break_down_1.median():.1f}")
        print(f"    3 bars: mean={break_down_3.mean():.1f}, median={break_down_3.median():.1f}")

    else:
        # Resistance: expect price to stay BELOW level (bounce down)
        bounce_1 = (subset["next1_high"] <= level).mean() * 100
        bounce_2 = ((subset["next1_high"] <= level) & (subset["next2_high"] <= level)).mean() * 100
        bounce_3 = ((subset["next1_high"] <= level) & (subset["next2_high"] <= level) & (subset["next3_high"] <= level)).mean() * 100

        print(f"  Price stays below level (resistance holds):")
        print(f"    After 1 bar: {bounce_1:.1f}%")
        print(f"    After 2 bars: {bounce_2:.1f}%")
        print(f"    After 3 bars: {bounce_3:.1f}%")

        # Bounce size: how far down from level
        bounce_down_1 = ((level - subset["next1_low"]) / level * 10000)
        bounce_down_2 = ((level - np.minimum(subset["next1_low"], subset["next2_low"])) / level * 10000)
        bounce_down_3 = ((level - np.minimum(subset["next1_low"], np.minimum(subset["next2_low"], subset["next3_low"]))) / level * 10000)

        print(f"  Max move DOWN from level (bps):")
        print(f"    1 bar:  mean={bounce_down_1.mean():.1f}, median={bounce_down_1.median():.1f}")
        print(f"    2 bars: mean={bounce_down_2.mean():.1f}, median={bounce_down_2.median():.1f}")
        print(f"    3 bars: mean={bounce_down_3.mean():.1f}, median={bounce_down_3.median():.1f}")

        # Break: how far above level
        break_up_1 = ((subset["next1_high"] - level) / level * 10000).clip(lower=0)
        break_up_3 = ((np.maximum(subset["next1_high"], np.maximum(subset["next2_high"], subset["next3_high"])) - level) / level * 10000).clip(lower=0)

        print(f"  Max move UP through level - break (bps):")
        print(f"    1 bar:  mean={break_up_1.mean():.1f}, median={break_up_1.median():.1f}")
        print(f"    3 bars: mean={break_up_3.mean():.1f}, median={break_up_3.median():.1f}")

# =========================================================================
# STEP 3: Measure by touch_count
# =========================================================================
print(f"\n{'='*70}")
print("STEP 3: Bounce rate by touch_count")
print(f"{'='*70}")

for name, mask, expected_dir in [
    ("SUPPORT", df["support_retest"] == 1, "above"),
    ("RESISTANCE", df["resistance_retest"] == 1, "below")
]:
    subset = df[mask]
    if len(subset) == 0:
        continue

    print(f"\n--- {name} ---")
    level = subset["level_price"]

    for tc_min, tc_max, label in [(1, 1, "1 touch"), (2, 2, "2 touches"), (3, 99, "3+ touches")]:
        tc_mask = (subset["touch_count"] >= tc_min) & (subset["touch_count"] <= tc_max)
        sub = subset[tc_mask]
        lvl = level[tc_mask]
        if len(sub) < 10:
            continue

        if expected_dir == "above":
            bounce = (sub["next1_low"] >= lvl).mean() * 100
            bounce3 = ((sub["next1_low"] >= lvl) & (sub["next2_low"] >= lvl) & (sub["next3_low"] >= lvl)).mean() * 100
        else:
            bounce = (sub["next1_high"] <= lvl).mean() * 100
            bounce3 = ((sub["next1_high"] <= lvl) & (sub["next2_high"] <= lvl) & (sub["next3_high"] <= lvl)).mean() * 100

        print(f"  {label}: {len(sub)} bars, bounce 1bar={bounce:.1f}%, bounce 3bars={bounce3:.1f}%")

# =========================================================================
# STEP 4: Measure by distance_to_level
# =========================================================================
print(f"\n{'='*70}")
print("STEP 4: Bounce rate by distance_to_level")
print(f"{'='*70}")

for name, mask, expected_dir in [
    ("SUPPORT", df["support_retest"] == 1, "above"),
    ("RESISTANCE", df["resistance_retest"] == 1, "below")
]:
    subset = df[mask]
    if len(subset) == 0:
        continue

    print(f"\n--- {name} ---")
    level = subset["level_price"]
    abs_dist = subset["distance_to_level"].abs()

    for d_max, label in [(3, "<3 bps"), (5, "<5 bps"), (10, "<10 bps"), (20, "<20 bps"), (9999, "all")]:
        d_mask = abs_dist <= d_max
        sub = subset[d_mask]
        lvl = level[d_mask]
        if len(sub) < 10:
            continue

        if expected_dir == "above":
            bounce = (sub["next1_low"] >= lvl).mean() * 100
        else:
            bounce = (sub["next1_high"] <= lvl).mean() * 100

        print(f"  {label}: {len(sub)} bars, bounce 1bar={bounce:.1f}%")

# =========================================================================
# STEP 5: Train vs Test
# =========================================================================
print(f"\n{'='*70}")
print("STEP 5: Train vs Test consistency")
print(f"{'='*70}")

for name, mask, expected_dir in [
    ("SUPPORT", df["support_retest"] == 1, "above"),
    ("RESISTANCE", df["resistance_retest"] == 1, "below")
]:
    subset = df[mask]
    if len(subset) == 0:
        continue

    print(f"\n--- {name} ---")
    level = subset["level_price"]

    for period, start, end in [("TRAIN 2020-2023", "2020-01-01", "2023-12-31"),
                                ("TEST 2024-2025", "2024-01-01", "2025-12-31")]:
        p_mask = (subset.index >= start) & (subset.index <= end)
        sub = subset[p_mask]
        lvl = level[p_mask]
        if len(sub) == 0:
            continue

        if expected_dir == "above":
            bounce1 = (sub["next1_low"] >= lvl).mean() * 100
            bounce3 = ((sub["next1_low"] >= lvl) & (sub["next2_low"] >= lvl) & (sub["next3_low"] >= lvl)).mean() * 100
        else:
            bounce1 = (sub["next1_high"] <= lvl).mean() * 100
            bounce3 = ((sub["next1_high"] <= lvl) & (sub["next2_high"] <= lvl) & (sub["next3_high"] <= lvl)).mean() * 100

        print(f"  {period}: {len(sub)} bars, bounce 1bar={bounce1:.1f}%, bounce 3bars={bounce3:.1f}%")

# =========================================================================
# STEP 6: Baseline comparison
# =========================================================================
print(f"\n{'='*70}")
print("STEP 6: Baseline - bars with NO S/R retest")
print(f"{'='*70}")

no_sr = df[(df["support_retest"] == 0) & (df["resistance_retest"] == 0)]
print(f"\nBars with no S/R level: {len(no_sr)}")

if len(no_sr) > 0:
    # For no-SR bars, check generic "does next bar go up or down"
    up_1 = (no_sr["next1_close"] > no_sr["current_price"]).mean() * 100
    down_1 = (no_sr["next1_close"] < no_sr["current_price"]).mean() * 100
    print(f"  Next bar UP:   {up_1:.1f}%")
    print(f"  Next bar DOWN: {down_1:.1f}%")

# Also compare: on S/R bars, generic up/down rate
sr_bars = df[(df["support_retest"] == 1) | (df["resistance_retest"] == 1)]
if len(sr_bars) > 0:
    up_1 = (sr_bars["next1_close"] > sr_bars["current_price"]).mean() * 100
    down_1 = (sr_bars["next1_close"] < sr_bars["current_price"]).mean() * 100
    print(f"\nBars WITH S/R level: {len(sr_bars)}")
    print(f"  Next bar UP:   {up_1:.1f}%")
    print(f"  Next bar DOWN: {down_1:.1f}%")

    # Support should predict UP, resistance should predict DOWN
    sup = df[df["support_retest"] == 1]
    res = df[df["resistance_retest"] == 1]
    if len(sup) > 0:
        sup_up = (sup["next1_close"] > sup["current_price"]).mean() * 100
        print(f"\n  SUPPORT bars: {len(sup)}, next bar UP = {sup_up:.1f}%")
    if len(res) > 0:
        res_down = (res["next1_close"] < res["current_price"]).mean() * 100
        print(f"  RESISTANCE bars: {len(res)}, next bar DOWN = {res_down:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
