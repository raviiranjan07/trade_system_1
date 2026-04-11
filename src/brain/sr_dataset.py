"""S/R Level Memory Dataset Pipeline — Stage 2

Processes OHLCV bars sequentially:
1. Detect S/R zones per bar (25-bar lookback)
2. Match/register zones in registry
3. Detect touches, create pending events
4. Resolve events after label horizon, update memory
5. Build dataset: dynamic features (LSTM) + static features (Dense)

Usage:
  PYTHONPATH=src python -m brain.sr_dataset --config experiments/brain/SR/config.yaml
"""

import argparse
import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from collections import deque

from brain.ingestion import load_ohlcv
from brain.processing import compute_sr_features, find_sr_zones, count_bar_touches, compute_recovery_speed
from brain.zone_registry import ZoneRegistry


def load_sr_config(config_path: str) -> dict:
    """Load SR experiment config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataset(config_path: str):
    """Build the full Stage 2 dataset."""
    sr_cfg = load_sr_config(config_path)

    # Load base config for OHLCV ingestion
    from brain.config import load_config
    base_cfg = load_config("configs/base.yaml")

    print("=" * 70)
    print("STAGE 2: S/R Level Memory Dataset")
    print("=" * 70)

    # --- Step 1: Load OHLCV ---
    df, _ = load_ohlcv(base_cfg)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_arr = df["open"].values
    N = len(df)
    dates = df.index

    # --- Step 2: Compute raw S/R zones per bar ---
    lookback = sr_cfg["sr_detection"]["lookback"]
    min_touches = sr_cfg["sr_detection"].get("min_touches", 2)
    print(f"\nComputing S/R zones ({lookback}-bar lookback)...")

    # Pre-compute S/R for every bar
    # KDE config
    sr_method = sr_cfg["sr_detection"].get("method", "biggest_gap")
    sr_bandwidth = sr_cfg["sr_detection"].get("bandwidth", 0.03)
    sr_sup_weight = sr_cfg["sr_detection"].get("support_weight", "reaction")
    sr_res_weight = sr_cfg["sr_detection"].get("resistance_weight", "recency")
    sr_peak_thresh = sr_cfg["sr_detection"].get("peak_threshold", 0.2)

    sr_data = []  # list of dicts per bar
    for t in range(N):
        if t < lookback:
            sr_data.append(None)
            continue

        w_high = high[t - lookback + 1:t + 1]
        w_low = low[t - lookback + 1:t + 1]
        w_close = close[t - lookback + 1:t + 1]

        sl, sh, rl, rh = find_sr_zones(w_high, w_low, min_touches,
                                         method=sr_method, bandwidth=sr_bandwidth,
                                         support_weight=sr_sup_weight,
                                         resistance_weight=sr_res_weight,
                                         peak_threshold=sr_peak_thresh)

        if sl is not None and rl is not None and rl > sh:
            zone_width_bps = (rl - sh) / sh * 10000
            zone_width_price = rl - sh
            dist_sup = (close[t] - sh) / sh * 10000
            dist_res = (rl - close[t]) / rl * 10000
            sup_retest = count_bar_touches(w_high, w_low, sl, sh)
            res_retest = count_bar_touches(w_high, w_low, rl, rh)
            rec_up = compute_recovery_speed(w_high, w_low, w_close, sl, sh, "up")
            rec_down = compute_recovery_speed(w_high, w_low, w_close, rl, rh, "down")
            window_low = float(np.min(w_low))
            window_high = float(np.max(w_high))

            sr_data.append({
                "sl": sl, "sh": sh, "rl": rl, "rh": rh,
                "zone_width_bps": zone_width_bps,
                "zone_width_price": zone_width_price,
                "dist_sup": dist_sup, "dist_res": dist_res,
                "sup_retest": sup_retest, "res_retest": res_retest,
                "rec_up": rec_up, "rec_down": rec_down,
                "window_low": window_low, "window_high": window_high,
                "sup_width": sh - sl, "res_width": rh - rl,
            })
        else:
            sr_data.append(None)

        if t % 50000 == 0:
            print(f"  S/R: {t}/{N} bars...")

    # --- Step 3: Sequential processing with zone registry ---
    print("\nRunning zone registry...")
    registry = ZoneRegistry(sr_cfg)
    label_cfg = sr_cfg["label"]
    horizon = label_cfg["horizon"]
    bounce_thresh = label_cfg["bounce_threshold"]
    clip_range = sr_cfg["features"]["clip_range"]

    # Pending events: list of dicts
    pending = []
    resolved_events = []

    # Dynamic feature names
    dynamic_names = [
        "dist_to_zone_pct", "support_width_pct", "res_width_pct",
        "support_retest", "resistance_retest", "zone_width",
        "recovery_up_pct", "recovery_down_pct",
        "speed_short", "speed_mid", "speed_long",
    ]

    static_names = [
        "bounce_ratio",
        "recent_bounce_ratio",
        "pressure", "bars_since_touch", "touch_count_scaled",
        "level_type_binary",
        "avg_bounce_mfe", "avg_break_mfe",
        "max_bounce_mfe", "max_break_mfe",
        "last_outcome", "bounce_streak",
        "bounce_mfe_trend", "chop_ratio",
    ]

    # Pre-compute dynamic features per bar (for snapshots)
    print("Computing dynamic features per bar...")
    bar_features = np.zeros((N, len(dynamic_names)), dtype=np.float32)

    for t in range(N):
        sr = sr_data[t]
        if sr is None:
            continue

        zw_bps = sr["zone_width_bps"]
        zw_price = sr["zone_width_price"]
        safe_zw_bps = max(zw_bps, 1.0)
        safe_zw_price = max(zw_price, 1.0)

        # dist_to_zone_pct: use whichever zone is closer
        if sr["dist_sup"] >= 0 and sr["dist_res"] >= 0:
            if sr["dist_sup"] <= sr["dist_res"]:
                dist_to_zone = sr["dist_sup"]
            else:
                dist_to_zone = sr["dist_res"]
        elif sr["dist_sup"] >= 0:
            dist_to_zone = sr["dist_sup"]
        elif sr["dist_res"] >= 0:
            dist_to_zone = sr["dist_res"]
        else:
            dist_to_zone = min(abs(sr["dist_sup"]), abs(sr["dist_res"]))

        bar_features[t, 0] = dist_to_zone / safe_zw_bps
        bar_features[t, 1] = sr["sup_width"] / safe_zw_price
        bar_features[t, 2] = sr["res_width"] / safe_zw_price
        bar_features[t, 3] = sr["sup_retest"]
        bar_features[t, 4] = sr["res_retest"]
        bar_features[t, 5] = np.log1p(zw_bps)
        bar_features[t, 6] = sr["rec_up"] / safe_zw_bps
        bar_features[t, 7] = sr["rec_down"] / safe_zw_bps

    # Speed features (computed after all bars have dist_to_zone)
    print("Computing speed features...")
    for t in range(3, N):
        bar_features[t, 8] = bar_features[t - 3, 0] - bar_features[t, 0]  # speed_short
    for t in range(10, N):
        bar_features[t, 9] = bar_features[t - 10, 0] - bar_features[t, 0]  # speed_mid
    for t in range(lookback, N):
        bar_features[t, 10] = bar_features[t - lookback, 0] - bar_features[t, 0]  # speed_long

    # Clip dynamic features
    clip_idx = [0, 1, 2, 6, 7, 8, 9, 10]  # zone-relative features
    bar_features[:, clip_idx] = np.clip(bar_features[:, clip_idx], -clip_range, clip_range)

    # --- Main sequential loop ---
    print("Processing touches and events...")
    snapshot_count = sr_cfg["features"]["snapshot_count"]
    touch_mode = sr_cfg.get("zone_filter", {}).get("touch_mode", "entry_only")
    print(f"  Touch mode: {touch_mode}")

    # Track which zones price is currently "inside" (for entry_only mode)
    zone_inside = {}  # (zone_id, role) -> True if price is currently at this zone

    for t in range(lookback, N):
        sr = sr_data[t]
        if sr is None:
            continue

        price = close[t]
        zw_bps = sr["zone_width_bps"]
        zw_price = sr["zone_width_price"]

        # Process support zone
        sup_zone = registry.process_zone(sr["sl"], sr["sh"], "support", t, price)
        # Process resistance zone
        res_zone = registry.process_zone(sr["rl"], sr["rh"], "resistance", t, price)

        # Check role flips
        registry.check_role_flip(sup_zone, price, zw_price, price)
        registry.check_role_flip(res_zone, price, zw_price, price)

        # Zone death removed — zones deactivate/reactivate naturally via process_zone

        # --- Resolve pending events (from horizon bars ago) ---
        new_pending = []
        for event in pending:
            resolve_bar = event["entry_bar"] + horizon
            if t >= resolve_bar:
                # Compute label
                entry = open_arr[event["entry_bar"] + 1]
                if entry <= 0:
                    continue
                eb = event["entry_bar"]
                window_high = high[eb + 1:eb + horizon + 1]
                window_low = low[eb + 1:eb + horizon + 1]
                mfe_up = max((np.max(window_high) - entry) / entry * 10000, 0)
                mae_down = max((entry - np.min(window_low)) / entry * 10000, 0)

                # Normalize direction
                if event["role"] == "support":
                    favorable = mfe_up
                    adverse = mae_down
                else:
                    favorable = mae_down
                    adverse = mfe_up

                total = favorable + adverse
                min_move = label_cfg.get("min_move_bps", 15)
                if total == 0:
                    outcome = "chop"
                    label = 2
                else:
                    fav_pct = favorable / total
                    adv_pct = adverse / total
                    if fav_pct > bounce_thresh and favorable >= min_move:
                        outcome = "bounce"
                        label = 1
                    elif adv_pct > bounce_thresh and adverse >= min_move:
                        outcome = "break"
                        label = 0
                    else:
                        outcome = "chop"
                        label = 2

                # Update zone memory (AFTER label)
                zone = event["zone"]
                if zone.active:
                    mfe_val = favorable if outcome == "bounce" else (adverse if outcome == "break" else None)
                    zone.update_memory(outcome, event["role"], mfe_value=mfe_val)

                # Store resolved event
                event["label"] = label
                event["outcome"] = outcome
                resolved_events.append(event)
            else:
                new_pending.append(event)
        pending = new_pending

        # --- Detect touches ---
        for zone, dist, role in [
            (sup_zone, sr["dist_sup"], sup_zone.role),
            (res_zone, sr["dist_res"], res_zone.role),
        ]:
            if not zone.active:
                continue

            inside_key = (zone.id, role)
            is_at_zone = dist >= 0 and registry.is_touch(dist, zw_bps)

            if touch_mode == "every_bar":
                # Record every bar at zone
                if is_at_zone:
                    if t < snapshot_count:
                        continue

                    memory_snapshot = registry.get_memory_features(zone, t)
                    memory_snapshot["bars_since_touch"] = np.log1p(memory_snapshot["bars_since_touch"])
                    dynamic_snapshot = bar_features[t - snapshot_count + 1:t + 1, :].copy()

                    event = {
                        "entry_bar": t,
                        "zone": zone,
                        "zone_id": zone.id,
                        "role": role,
                        "memory": memory_snapshot,
                        "dynamic": dynamic_snapshot,
                        "date": dates[t],
                    }
                    pending.append(event)
                    zone.last_touch_bar = t

            else:
                # entry_only: only record first bar of each visit
                was_inside = zone_inside.get(inside_key, False)

                if is_at_zone and not was_inside:
                    zone_inside[inside_key] = True

                    if t < snapshot_count:
                        continue

                    memory_snapshot = registry.get_memory_features(zone, t)
                    memory_snapshot["bars_since_touch"] = np.log1p(memory_snapshot["bars_since_touch"])
                    dynamic_snapshot = bar_features[t - snapshot_count + 1:t + 1, :].copy()

                    event = {
                        "entry_bar": t,
                        "zone": zone,
                        "zone_id": zone.id,
                        "role": role,
                        "memory": memory_snapshot,
                        "dynamic": dynamic_snapshot,
                        "date": dates[t],
                    }
                    pending.append(event)
                    zone.last_touch_bar = t

                elif not is_at_zone:
                    zone_inside[inside_key] = False

        if t % 50000 == 0:
            print(f"  Processing: {t}/{N}, zones={len(registry.zones)}, pending={len(pending)}, resolved={len(resolved_events)}")

    print(f"\nResolved events: {len(resolved_events)}")

    # --- Step 4: Build dataset ---
    print("\nBuilding dataset...")
    X_dynamic = []
    X_static = []
    Y = []
    event_dates = []
    event_bars = []

    for event in resolved_events:
        X_dynamic.append(event["dynamic"])
        X_static.append([event["memory"][k] for k in static_names])
        Y.append(event["label"])
        event_dates.append(event["date"])
        event_bars.append(event["entry_bar"])

    X_dynamic = np.array(X_dynamic, dtype=np.float32)
    X_static = np.array(X_static, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    event_dates = np.array(event_dates)

    print(f"  Samples: {len(Y)}")
    print(f"  Dynamic shape: {X_dynamic.shape}")
    print(f"  Static shape: {X_static.shape}")

    # Label distribution
    for label, name in [(0, "BREAK"), (1, "BOUNCE"), (2, "CHOP")]:
        count = (Y == label).sum()
        print(f"  {name}: {count} ({count / len(Y) * 100:.1f}%)")

    # --- Step 5: Split by date ---
    data_cfg = sr_cfg["data"]
    train_end = pd.Timestamp(data_cfg["train"][1])
    val_start = pd.Timestamp(data_cfg["val"][0])
    val_end = pd.Timestamp(data_cfg["val"][1])
    test_start = pd.Timestamp(data_cfg["test"][0])

    train_mask = event_dates <= train_end
    val_mask = (event_dates >= val_start) & (event_dates <= val_end)
    test_mask = event_dates >= test_start

    print(f"\n  Train: {train_mask.sum()}")
    print(f"  Val:   {val_mask.sum()}")
    print(f"  Test:  {test_mask.sum()}")

    # --- Step 6: Save ---
    out_dir = Path(config_path).parent / f"datasets_{touch_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    event_bars_arr = np.array(event_bars, dtype=np.int64)

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X_dynamic=X_dynamic[mask],
            X_static=X_static[mask],
            Y=Y[mask],
            bars=event_bars_arr[mask],
        )

    meta = {
        "dynamic_features": dynamic_names,
        "static_features": static_names,
        "snapshot_count": snapshot_count,
        "touch_mode": touch_mode,
        "total_samples": len(Y),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "label_map": {"0": "BREAK", "1": "BOUNCE", "2": "CHOP"},
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

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
    with open(out_dir / "zone_registry.json", "w") as f:
        json.dump(zones_data, f, indent=2)

    print(f"\n  Saved to {out_dir}/")
    print(f"  Zones: {len(zones_data)}")

    # --- Validation checks ---
    print(f"\n{'='*70}")
    print("VALIDATION CHECKS")
    print(f"{'='*70}")

    # Check 1: Label distribution
    print("\n1. Label distribution:")
    for label, name in [(0, "BREAK"), (1, "BOUNCE"), (2, "CHOP")]:
        for split, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
            count = (Y[mask] == label).sum()
            total = mask.sum()
            print(f"  {split} {name}: {count} ({count / total * 100:.1f}%)" if total > 0 else f"  {split} {name}: 0")

    # Check 2: Feature variance
    print("\n2. Feature variance (train):")
    for i, name in enumerate(dynamic_names):
        vals = X_dynamic[train_mask, :, i].flatten()
        print(f"  {name}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}")
    for i, name in enumerate(static_names):
        vals = X_static[train_mask, i]
        print(f"  {name}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}")

    # Check 3: Event gaps
    gaps = np.diff(np.where(train_mask)[0])
    if len(gaps) > 0:
        print(f"\n3. Event gaps (train): mean={gaps.mean():.1f} min={gaps.min()} median={np.median(gaps):.1f}")

    return resolved_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/brain/SR/config.yaml")
    args = parser.parse_args()
    build_dataset(args.config)
