"""Brain Pipeline -- Step 4: Dataset Creation

Create snapshots, split train/val/test, normalize.

Each sample = snapshot_count consecutive bars of features.
At each snapshot, S/R features are relative to that snapshot's own window.

Raw price features are converted to relative bps from current price.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Raw price columns that need relative conversion
RAW_PRICE_COLS = [
    "support_range_low", "support_range_high",
    "resistance_range_low", "resistance_range_high",
    "window_low", "window_high",
]


def create_dataset(features: pd.DataFrame, labels: pd.DataFrame,
                    config: dict) -> dict:
    """Create train/val/test datasets from features and labels.

    Returns dict with keys: train_X, train_Y, val_X, val_Y, test_X, test_Y,
                            feature_names, scaler (mean/std)
    """
    ds_cfg = config["dataset"]
    snapshot_count = ds_cfg["snapshot_count"]
    snapshot_gap = ds_cfg.get("snapshot_gap", 1)
    split = ds_cfg["split"]
    norm_cfg = ds_cfg.get("normalization", {})
    filter_cfg = ds_cfg.get("filtering", {})

    feature_names = features.columns.tolist()
    print(f"  Features: {len(feature_names)} columns")
    print(f"  Snapshot count: {snapshot_count}, gap: {snapshot_gap}")

    # --- Convert raw prices to relative bps ---
    raw_method = norm_cfg.get("raw_price_method", "relative_bps")
    if raw_method == "relative_bps":
        print("  Converting raw prices to relative bps...")
        features = _convert_raw_prices(features)

    # --- Align features and labels ---
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    print(f"  Aligned: {len(common_idx)} bars")

    # --- Handle NaN ---
    nan_method = norm_cfg.get("nan_handling", "zero")
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        print(f"  NaN values: {nan_count}")
        if nan_method == "zero":
            features = features.fillna(0)
        elif nan_method == "forward_fill":
            features = features.fillna(method="ffill").fillna(0)

    # --- Create samples ---
    feat_arr = features.values.astype(np.float32)
    dates = features.index
    label_cols = labels.columns.tolist()
    label_arr = labels.values

    if snapshot_count == 1:
        # Flat mode: 1 bar per sample (for tree-based models like XGBoost)
        print("  Creating flat samples (1 bar each)...")
        X = feat_arr
        Y = label_arr.astype(np.float32)
        date_arr = np.array(dates)
    else:
        # Snapshot mode: groups of bars (for sequence models like LSTM)
        print("  Creating snapshots...")
        total_lookback = (snapshot_count - 1) * snapshot_gap

        X_list = []
        Y_list = []
        date_list = []

        for i in range(total_lookback, len(feat_arr)):
            snap_indices = [i - (snapshot_count - 1 - s) * snapshot_gap for s in range(snapshot_count)]

            if snap_indices[0] < 0:
                continue

            snapshot = np.stack([feat_arr[idx] for idx in snap_indices])
            X_list.append(snapshot)
            Y_list.append(label_arr[i])
            date_list.append(dates[i])

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32)
        date_arr = np.array(date_list)

    print(f"  Samples created: {len(X)}, shape {X.shape}")

    # --- Filter ---
    if filter_cfg.get("remove_skip", True):
        # Direction is first label column
        if "direction" in label_cols:
            dir_idx = label_cols.index("direction")
            keep = (Y[:, dir_idx] == 0) | (Y[:, dir_idx] == 1)  # LONG or SHORT only
            X = X[keep]
            Y = Y[keep]
            date_arr = date_arr[keep]
            print(f"  Filtered SKIP/BOTH (direction): {len(X)} samples remaining")

        # Bounce/break: keep only labeled bars (0-3), remove BB_SKIP (4)
        if "bounce_break" in label_cols:
            bb_idx = label_cols.index("bounce_break")
            keep = Y[:, bb_idx] < 4  # 0,1,2,3 are valid labels
            X = X[keep]
            Y = Y[keep]
            date_arr = date_arr[keep]
            print(f"  Filtered SKIP (bounce_break): {len(X)} samples remaining")

    # --- Split by date ---
    train_mask = (date_arr >= pd.Timestamp(split["train"][0])) & (date_arr <= pd.Timestamp(split["train"][1]))
    val_mask = (date_arr >= pd.Timestamp(split["val"][0])) & (date_arr <= pd.Timestamp(split["val"][1]))
    test_mask = (date_arr >= pd.Timestamp(split["test"][0])) & (date_arr <= pd.Timestamp(split["test"][1]))

    train_X, train_Y = X[train_mask], Y[train_mask]
    val_X, val_Y = X[val_mask], Y[val_mask]
    test_X, test_Y = X[test_mask], Y[test_mask]

    print(f"  Train: {len(train_X)} samples ({split['train'][0]} to {split['train'][1]})")
    print(f"  Val:   {len(val_X)} samples ({split['val'][0]} to {split['val'][1]})")
    print(f"  Test:  {len(test_X)} samples ({split['test'][0]} to {split['test'][1]})")

    # --- Normalize ---
    scaler = None
    method = norm_cfg.get("method", "zscore")
    if method == "zscore":
        print("  Normalizing (z-score, fit on train)...")
        # Compute mean/std from train only
        # Flatten snapshots for stats: [N * snapshot_count, features]
        train_flat = train_X.reshape(-1, train_X.shape[-1])
        mean = np.nanmean(train_flat, axis=0)
        std = np.nanstd(train_flat, axis=0)
        std[std == 0] = 1.0  # avoid division by zero

        scaler = {"mean": mean, "std": std}

        # Apply to all splits
        train_X = (train_X - mean) / std
        val_X = (val_X - mean) / std
        test_X = (test_X - mean) / std

        # Replace any remaining NaN/inf
        train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
        val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
        test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Direction label distribution per split ---
    if "direction" in label_cols:
        dir_idx = label_cols.index("direction")
        for name, y in [("Train", train_Y), ("Val", val_Y), ("Test", test_Y)]:
            dirs = y[:, dir_idx]
            long_pct = (dirs == 0).mean() * 100
            short_pct = (dirs == 1).mean() * 100
            print(f"  {name}: LONG={long_pct:.1f}%, SHORT={short_pct:.1f}%")

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "val_X": val_X,
        "val_Y": val_Y,
        "test_X": test_X,
        "test_Y": test_Y,
        "feature_names": feature_names,
        "label_names": label_cols,
        "scaler": scaler,
        "train_dates": date_arr[train_mask],
        "val_dates": date_arr[val_mask],
        "test_dates": date_arr[test_mask],
    }


def _convert_raw_prices(features: pd.DataFrame) -> pd.DataFrame:
    """Convert raw price columns to relative bps from current close.

    support_range_low becomes: (close - support_range_low) / close * 10000
    This way the brain sees "support is 20 bps below current price"
    regardless of whether BTC is at 7k or 100k.
    """
    features = features.copy()

    # We need close price to compute relative. Use distance_to_support/resistance
    # which already capture some of this. For raw prices, convert relative to window midpoint.
    # Actually, the raw prices should be relative to the close at that bar.
    # But close isn't a feature column. We need to derive it.
    #
    # For S/R: distance_to_support and distance_to_resistance already capture
    # the relative position. The raw prices are supplementary.
    #
    # Convert: raw_price -> (raw_price - midpoint) / midpoint * 10000
    # where midpoint = (window_high + window_low) / 2

    # Compute midpoint ONCE before converting any columns
    if "window_low" in features.columns and "window_high" in features.columns:
        midpoint = (features["window_high"] + features["window_low"]) / 2
        mask = midpoint > 0

        for col in RAW_PRICE_COLS:
            if col not in features.columns:
                continue
            features.loc[mask, col] = (features.loc[mask, col] - midpoint[mask]) / midpoint[mask] * 10000
            features.loc[~mask, col] = 0

    return features


def save_dataset(data: dict, output_dir: str):
    """Save dataset as .npz files for Colab upload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split
    for split in ["train", "val", "test"]:
        np.savez_compressed(
            output_dir / f"{split}.npz",
            X=data[f"{split}_X"],
            Y=data[f"{split}_Y"],
        )

    # Save scaler
    if data["scaler"] is not None:
        np.savez(
            output_dir / "scaler.npz",
            mean=data["scaler"]["mean"],
            std=data["scaler"]["std"],
        )

    # Save metadata
    import json
    meta = {
        "feature_names": data["feature_names"],
        "label_names": data["label_names"],
        "train_samples": len(data["train_X"]),
        "val_samples": len(data["val_X"]),
        "test_samples": len(data["test_X"]),
        "snapshot_shape": list(data["train_X"].shape[1:]),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"    train.npz: {data['train_X'].shape}")
    print(f"    val.npz:   {data['val_X'].shape}")
    print(f"    test.npz:  {data['test_X'].shape}")
    print(f"    scaler.npz")
    print(f"    metadata.json")
