"""Stage 5: Label Validation — test different label parameters.

Uses saved bar numbers from every_bar dataset.
Recomputes labels with different horizon + dominance.
Checks feature separation for each combination.

Run: PYTHONPATH=src python experiments/brain/SR/label_validation.py
"""

import numpy as np
import json
from brain.config import load_config
from brain.ingestion import load_ohlcv

# Load OHLCV
base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
close = df["close"].values
open_arr = df["open"].values
N = len(df)

# Load saved dataset (has bar numbers)
data = np.load("data/features/sr_bounce_break/every_bar/train.npz")
Xd = data["X_dynamic"][:, -1, :]  # last snapshot
Xs = data["X_static"]
bars = data["bars"]

print("Loaded %d events with bar numbers" % len(bars))
print()

dyn_names = ["dist_to_zone_pct", "support_width_pct", "res_width_pct",
             "support_retest", "resistance_retest", "zone_width",
             "recovery_up_pct", "recovery_down_pct",
             "speed_short", "speed_mid", "speed_long"]
sta_names = ["bounce_ratio", "recent_bounce_ratio", "pressure",
             "bars_since_touch", "touch_count_scaled", "level_type_binary"]

all_X = np.hstack([Xd, Xs])
all_names = dyn_names + sta_names

# Test parameters
horizons = [5, 10, 15, 25]
dominances = [0.60, 0.70, 0.80]
min_move = 15  # fixed

print("=" * 80)
print("LABEL VALIDATION: %d horizons x %d dominances = %d tests" % (len(horizons), len(dominances), len(horizons) * len(dominances)))
print("=" * 80)

results = []

for horizon in horizons:
    for dominance in dominances:
        # Recompute labels
        labels = np.full(len(bars), 2, dtype=np.int64)  # default chop

        for idx in range(len(bars)):
            t = bars[idx]
            if t + horizon + 1 >= N:
                continue

            entry = open_arr[t + 1]
            if entry <= 0:
                continue

            wh = high[t + 1:t + horizon + 1]
            wl = low[t + 1:t + horizon + 1]

            mfe_up = max((np.max(wh) - entry) / entry * 10000, 0)
            mae_down = max((entry - np.min(wl)) / entry * 10000, 0)

            # Determine role from level_type_binary
            is_support = Xs[idx, 5] > 0.5

            if is_support:
                fav, adv = mfe_up, mae_down
            else:
                fav, adv = mae_down, mfe_up

            total = fav + adv
            if total == 0:
                continue

            fav_pct = fav / total
            if fav_pct > dominance and fav >= min_move:
                labels[idx] = 1  # bounce
            elif (1 - fav_pct) > dominance and adv >= min_move:
                labels[idx] = 0  # break

        bounce = (labels == 1).sum()
        brk = (labels == 0).sum()
        chop = (labels == 2).sum()
        total_labeled = bounce + brk
        bounce_rate = bounce / max(total_labeled, 1) * 100

        # Feature separation
        max_sep = 0
        avg_sep = 0
        sep_count = 0
        best_feat = ""

        if bounce > 50 and brk > 50:
            for i in range(all_X.shape[1]):
                b_mean = all_X[labels == 1, i].mean()
                k_mean = all_X[labels == 0, i].mean()
                mx = max(abs(b_mean), abs(k_mean), 0.001)
                sep = abs(b_mean - k_mean) / mx * 100
                avg_sep += sep
                sep_count += 1
                if sep > max_sep:
                    max_sep = sep
                    best_feat = all_names[i]
            avg_sep = avg_sep / max(sep_count, 1)

        result = {
            "horizon": horizon, "dominance": dominance,
            "bounce": bounce, "break": brk, "chop": chop,
            "bounce_rate": bounce_rate, "max_sep": max_sep,
            "avg_sep": avg_sep, "best_feat": best_feat,
        }
        results.append(result)

        print("H=%2d dom=%.2f | bounce=%5d break=%5d chop=%5d | rate=%.1f%% | max_sep=%.1f%% avg_sep=%.1f%% best=%s" % (
            horizon, dominance, bounce, brk, chop, bounce_rate, max_sep, avg_sep, best_feat))

# Summary
print()
print("=" * 80)
print("BEST CONFIGURATIONS (by max feature separation):")
print("=" * 80)
sorted_results = sorted(results, key=lambda x: x["max_sep"], reverse=True)
for r in sorted_results[:5]:
    print("H=%2d dom=%.2f | max_sep=%.1f%% avg=%.1f%% | bounce=%d break=%d rate=%.1f%% | best=%s" % (
        r["horizon"], r["dominance"], r["max_sep"], r["avg_sep"],
        r["bounce"], r["break"], r["bounce_rate"], r["best_feat"]))

print()
print("BEST CONFIGURATIONS (by bounce rate away from 50%):")
sorted_by_rate = sorted(results, key=lambda x: abs(x["bounce_rate"] - 50), reverse=True)
for r in sorted_by_rate[:5]:
    print("H=%2d dom=%.2f | rate=%.1f%% | max_sep=%.1f%% | bounce=%d break=%d" % (
        r["horizon"], r["dominance"], r["bounce_rate"], r["max_sep"],
        r["bounce"], r["break"]))
