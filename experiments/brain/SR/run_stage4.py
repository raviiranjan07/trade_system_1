"""Stage 4: Rebuild with hybrid KDE + check feature separation.

Run: PYTHONPATH=src python experiments/brain/SR/run_stage4.py
"""

import numpy as np
import yaml
import json
import pandas as pd
from pathlib import Path
from brain.config import load_config
from brain.ingestion import load_ohlcv
from brain.processing import find_sr_zones, count_bar_touches, compute_recovery_speed
from brain.zone_registry import ZoneRegistry

sr_cfg = yaml.safe_load(open('experiments/brain/SR/config.yaml'))
base_cfg = load_config('configs/base.yaml')

df, _ = load_ohlcv(base_cfg)
high = df['high'].values
low = df['low'].values
close = df['close'].values
open_arr = df['open'].values
N = len(df)
dates = df.index

lookback = 25
sr_method = 'hybrid_kde'
sr_bw = 0.03
sr_sw = 'reaction'
sr_rw = 'recency'
sr_pt = 0.2
horizon = 25

print("=" * 60)
print("STAGE 4: Rebuild with Hybrid KDE")
print("=" * 60)

# Step 1: Compute S/R zones per bar
print("\nStep 1: Computing hybrid KDE zones (25-bar)...")
sr_data = []
for t in range(N):
    if t < lookback:
        sr_data.append(None)
        continue
    w_high = high[t - lookback + 1:t + 1]
    w_low = low[t - lookback + 1:t + 1]
    w_close = close[t - lookback + 1:t + 1]
    sl, sh, rl, rh = find_sr_zones(w_high, w_low, 2, method=sr_method,
                                     bandwidth=sr_bw, support_weight=sr_sw,
                                     resistance_weight=sr_rw, peak_threshold=sr_pt)
    if sl and rl and rl > sh:
        zw_bps = (rl - sh) / sh * 10000
        sr_data.append({
            "sl": sl, "sh": sh, "rl": rl, "rh": rh, "zw_bps": zw_bps,
            "dist_sup": (close[t] - sh) / sh * 10000,
            "dist_res": (rl - close[t]) / rl * 10000,
            "sup_retest": count_bar_touches(w_high, w_low, sl, sh),
            "res_retest": count_bar_touches(w_high, w_low, rl, rh),
            "rec_up": compute_recovery_speed(w_high, w_low, w_close, sl, sh, "up"),
            "rec_down": compute_recovery_speed(w_high, w_low, w_close, rl, rh, "down"),
        })
    else:
        sr_data.append(None)
    if t % 50000 == 0:
        print(f"  {t}/{N} bars...")

# Pre-compute dist_to_zone for speed features
dist_to_zone = np.zeros(N, dtype=np.float32)
for t in range(N):
    sr = sr_data[t]
    if sr is None:
        continue
    if sr["dist_sup"] >= 0 and sr["dist_res"] >= 0:
        dist_to_zone[t] = min(sr["dist_sup"], sr["dist_res"])
    elif sr["dist_sup"] >= 0:
        dist_to_zone[t] = sr["dist_sup"]
    elif sr["dist_res"] >= 0:
        dist_to_zone[t] = sr["dist_res"]

# Step 2: Zone registry + events
print("\nStep 2: Zone registry + event detection...")
registry = ZoneRegistry(sr_cfg)
zone_inside = {}
pending = []
events = []

for t in range(lookback, N):
    sr = sr_data[t]
    if sr is None:
        continue

    price = close[t]
    zw = sr["zw_bps"]
    safe_zw = max(zw, 1.0)

    sup_zone = registry.process_zone(sr["sl"], sr["sh"], "support", t, price)
    res_zone = registry.process_zone(sr["rl"], sr["rh"], "resistance", t, price)

    registry.check_role_flip(sup_zone, price, sr["rl"] - sr["sh"], price)
    registry.check_role_flip(res_zone, price, sr["rl"] - sr["sh"], price)

    # Resolve pending events
    new_pending = []
    for ev in pending:
        if t >= ev["bar"] + horizon:
            eb = ev["bar"]
            entry = open_arr[eb + 1]
            if entry > 0 and eb + horizon + 1 <= N:
                wh = high[eb + 1:eb + horizon + 1]
                wl = low[eb + 1:eb + horizon + 1]
                mfe = max((np.max(wh) - entry) / entry * 10000, 0)
                mae = max((entry - np.min(wl)) / entry * 10000, 0)
                if ev["role"] == "support":
                    fav, adv = mfe, mae
                else:
                    fav, adv = mae, mfe
                total = fav + adv
                if total > 0:
                    fav_pct = fav / total
                    if fav_pct > 0.70 and fav >= 15:
                        outcome, label = "bounce", 1
                    elif (1 - fav_pct) > 0.70 and adv >= 15:
                        outcome, label = "break", 0
                    else:
                        outcome, label = "chop", 2
                else:
                    outcome, label = "chop", 2

                zone = ev["zone"]
                if zone.active:
                    zone.update_memory(outcome, ev["role"])

                ev["label"] = label
                ev["outcome"] = outcome
                events.append(ev)
        else:
            new_pending.append(ev)
    pending = new_pending

    # Detect touches (entry-only)
    for zone, dist, role in [(sup_zone, sr["dist_sup"], sup_zone.role),
                              (res_zone, sr["dist_res"], res_zone.role)]:
        if not zone.active or dist < 0:
            continue
        inside_key = (zone.id, role)
        is_at = registry.is_touch(dist, zw)
        was_in = zone_inside.get(inside_key, False)

        if is_at and not was_in:
            zone_inside[inside_key] = True

            speed_s = (dist_to_zone[max(t - 3, 0)] - dist_to_zone[t]) / safe_zw if t >= 3 else 0
            speed_m = (dist_to_zone[max(t - 10, 0)] - dist_to_zone[t]) / safe_zw if t >= 10 else 0
            speed_l = (dist_to_zone[max(t - 25, 0)] - dist_to_zone[t]) / safe_zw if t >= 25 else 0

            dynamic = [
                dist / safe_zw,
                sr["sup_retest"],
                sr["res_retest"],
                np.log1p(zw),
                sr["rec_up"] / safe_zw,
                sr["rec_down"] / safe_zw,
                np.clip(speed_s, -3, 3),
                np.clip(speed_m, -3, 3),
                np.clip(speed_l, -3, 3),
            ]

            mem = registry.get_memory_features(zone, t)
            static = [
                mem["bounce_ratio"],
                mem["recent_bounce_ratio"],
                mem["pressure"],
                np.log1p(mem["bars_since_touch"]),
                mem["touch_count_scaled"],
                mem["level_type_binary"],
            ]

            pending.append({
                "bar": t, "zone": zone, "zone_id": zone.id, "role": role,
                "dynamic": dynamic, "static": static, "date": dates[t],
            })
            zone.last_touch_bar = t
        elif not is_at:
            zone_inside[inside_key] = False

    if t % 50000 == 0:
        print(f"  {t}/{N}, zones={len(registry.zones)}, events={len(events)}")

print(f"  Total events: {len(events)}")

# Step 3: Build arrays
print("\nStep 3: Building dataset...")
X_dyn = np.array([e["dynamic"] for e in events], dtype=np.float32)
X_sta = np.array([e["static"] for e in events], dtype=np.float32)
Y = np.array([e["label"] for e in events], dtype=np.int64)
event_dates = np.array([e["date"] for e in events])

print(f"  Samples: {len(Y)}")
for label, name in [(0, "BREAK"), (1, "BOUNCE"), (2, "CHOP")]:
    count = (Y == label).sum()
    print(f"  {name}: {count} ({count / len(Y) * 100:.1f}%)")

# Step 4: Feature separation
print("\n" + "=" * 60)
print("FEATURE SEPARATION")
print("=" * 60)

dyn_names = ["dist_to_zone_pct", "support_retest", "resistance_retest", "zone_width",
             "recovery_up_pct", "recovery_down_pct", "speed_short", "speed_mid", "speed_long"]
sta_names = ["bounce_ratio", "recent_bounce_ratio", "pressure",
             "bars_since_touch", "touch_count_scaled", "level_type_binary"]

print("\nDYNAMIC:")
print(f"{'Feature':25s} | {'Bounce':>10s} {'Break':>10s} {'Chop':>10s} | {'MaxDiff%':>8s}")
for i, name in enumerate(dyn_names):
    b = X_dyn[Y == 1, i].mean()
    k = X_dyn[Y == 0, i].mean()
    c = X_dyn[Y == 2, i].mean()
    mx = max(abs(b), abs(k), abs(c), 0.001)
    diff = max(abs(b - k), abs(b - c), abs(k - c)) / mx * 100
    print(f"{name:25s} | {b:10.3f} {k:10.3f} {c:10.3f} | {diff:7.1f}%")

print("\nSTATIC:")
for i, name in enumerate(sta_names):
    b = X_sta[Y == 1, i].mean()
    k = X_sta[Y == 0, i].mean()
    c = X_sta[Y == 2, i].mean()
    mx = max(abs(b), abs(k), abs(c), 0.001)
    diff = max(abs(b - k), abs(b - c), abs(k - c)) / mx * 100
    print(f"{name:25s} | {b:10.3f} {k:10.3f} {c:10.3f} | {diff:7.1f}%")

# Save
out = Path("experiments/brain/SR/datasets_v2")
out.mkdir(parents=True, exist_ok=True)

train_mask = event_dates < pd.Timestamp("2023-01-01")
val_mask = (event_dates >= pd.Timestamp("2023-01-01")) & (event_dates < pd.Timestamp("2024-01-01"))
test_mask = event_dates >= pd.Timestamp("2024-01-01")

print(f"\nTrain: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
    np.savez_compressed(out / f"{name}.npz", X_dynamic=X_dyn[mask], X_static=X_sta[mask], Y=Y[mask])

meta = {"dynamic_features": dyn_names, "static_features": sta_names,
        "total_events": len(events), "train": int(train_mask.sum()),
        "val": int(val_mask.sum()), "test": int(test_mask.sum())}
json.dump(meta, open(out / "metadata.json", "w"), indent=2)

# Save zone registry
zones_data = []
for z in registry.zones:
    zones_data.append({
        "id": z.id, "center": z.center, "width": z.width,
        "low": z.low, "high": z.high, "role": z.role,
        "touches": z.touches, "bounces": z.bounces, "breaks": z.breaks,
        "history": list(z.history), "last_touch_bar": z.last_touch_bar,
        "created_bar": z.created_bar, "active": z.active,
    })
json.dump(zones_data, open(out / "zone_registry.json", "w"), indent=2)

print(f"\nSaved to {out}/")
print(f"  Zones: {len(zones_data)}")
print("DONE")
