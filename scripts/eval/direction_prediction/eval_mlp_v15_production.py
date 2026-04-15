"""Evaluate V1.5 production model and compute all 26 protocol metrics.

Loads the saved model, computes features, runs predictions on test set,
and logs everything through run_experiment().

Usage:
  PYTHONPATH=src python scripts/eval_v15_production.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from mlops.runner import run_experiment
from mlops.evaluation import evaluate_direction_prediction


# Paths
CACHE_PATH = Path("data/features/direction_prediction/feature_cache.parquet")
LABELS_PATH = Path("data/features/direction_prediction/labels.parquet")
MODEL_PATH = Path("models/direction_v15/direction_model.pt")
SCALER_PATH = Path("models/direction_v15/scaler.npz")

# Same split as ml_train.py (but we only evaluate on 2025 test)
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# Confidence thresholds (same as production bot)
CONF_LONG = 0.60
CONF_SHORT = 0.35


# Model definition (same as src/engine/ml_train.py)
class MLPBinaryDir(nn.Module):
    def __init__(self, input_size=10, hidden=128, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.out(h).squeeze(-1)


def main():
    # --- Load data ---
    print("Loading data...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)
    print(f"  feature_cache: {fc.shape}")
    print(f"  labels: {lb.shape}")

    # --- Compute features (same as ml_train.py) ---
    print("Computing features...")
    close = fc["close"].values.astype(np.float64)
    high = fc["high"].values.astype(np.float64)
    low = fc["low"].values.astype(np.float64)

    # roc1-8
    for n in range(1, 9):
        roc = np.zeros(len(close), dtype=np.float32)
        roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
        fc[f"roc{n}"] = roc

    # rsi7
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / loss_s
    fc["rsi7_computed"] = (100 - (100 / (1 + rs))).values

    # range_position_50
    rp = np.zeros(len(close), dtype=np.float32)
    for i in range(50, len(close)):
        hh = np.max(high[i - 50:i + 1])
        ll = np.min(low[i - 50:i + 1])
        rng = hh - ll
        rp[i] = (close[i] - ll) / rng if rng > 0 else 0.5
    fc["range_position_50"] = rp

    feat_cols = [f"roc{n}" for n in range(1, 9)] + ["range_position_50", "rsi7_computed"]

    # --- Align features and labels ---
    common_idx = lb.index.intersection(fc.index)
    lb = lb.loc[common_idx]
    fc_aligned = fc.loc[common_idx]

    # --- Load scaler ---
    scaler = np.load(SCALER_PATH)
    scaler_mean = scaler["mean"]
    scaler_std = scaler["std"]

    # --- Prepare test set ---
    test_mask = (common_idx >= pd.Timestamp(TEST_START)) & (common_idx <= pd.Timestamp(TEST_END))
    direction = lb["direction"].values  # H96 (same as ml_train.py)
    valid_mask = (direction == 0) | (direction == 1)
    test_valid = test_mask & valid_mask

    feat_arr = fc_aligned[feat_cols].values.astype(np.float32)
    feat_arr = (feat_arr - scaler_mean) / scaler_std
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_test = feat_arr[test_valid]
    y_test = np.zeros(test_valid.sum(), dtype=np.float32)
    y_test[direction[test_valid] == 0] = 1.0  # LONG=1, SHORT=0 (same encoding as ml_train.py)

    # MFE for magnitude metrics (H96)
    mfe_up_test = lb["mfe_up_96"].values[test_valid]
    mfe_down_test = lb["mfe_down_96"].values[test_valid]

    print(f"  Test bars: {len(X_test)} (LONG={int(y_test.sum())}, SHORT={int(len(y_test) - y_test.sum())})")

    # --- Load model and predict ---
    print("Loading model...")
    model = MLPBinaryDir(input_size=10, hidden=128)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Running predictions...")
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        probs = torch.sigmoid(logits).numpy()

    print(f"  Predictions: min={probs.min():.3f}, max={probs.max():.3f}, mean={probs.mean():.3f}")
    print(f"  Confident LONG (>{CONF_LONG}): {(probs >= CONF_LONG).sum()}")
    print(f"  Confident SHORT (<{CONF_SHORT}): {(probs <= CONF_SHORT).sum()}")

    # --- Compute all 26 metrics ---
    print("Computing metrics...")
    metrics = evaluate_direction_prediction(
        probs=probs,
        labels=y_test,
        mfe_up_bps=mfe_up_test,
        mfe_down_bps=mfe_down_test,
        conf_threshold_long=CONF_LONG,
        conf_threshold_short=CONF_SHORT,
        split_name="test",
    )

    # --- Run through MLOps system ---
    with run_experiment(
        experiment_name="direction_prediction",
        protocol_name="direction_prediction_v1",
        config_path="src/engine/ml_train.py",
        params={
            "model_type": "MLP",
            "architecture": "10-128-128-1",
            "features": "roc1-8+range_position_50+rsi7",
            "label": "direction_h96",
            "horizon": 96,
            "conf_threshold_long": CONF_LONG,
            "conf_threshold_short": CONF_SHORT,
            "batch_size": 512,
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        model_type="MLP",
        dataset_version="feature_cache_cleaned_23col",
        primary_metric="test_confident_accuracy",
        notes="V1.5 production model — full 26 metric evaluation on 2025 test set",
    ) as run:
        run.log_metrics(metrics)

        # Save model as artifact
        run.log_artifact(str(MODEL_PATH))

        # Print all metrics
        print("\n=== ALL 26 METRICS ===")
        for k, v in sorted(metrics.items()):
            if "confusion" not in k:
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
