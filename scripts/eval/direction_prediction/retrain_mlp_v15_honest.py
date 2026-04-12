"""Retrain V1.5 MLP with proper date-based split for honest evaluation.

Same architecture, features, label as production V1.5.
Only difference: date-based split (train 2020-2023, val 2024, test 2025).
Scaler fit on train only. No data leakage.

The production model (models/direction_v15/) is NOT modified.
This is a separate training run purely for honest metrics.

Usage:
  PYTHONPATH=src python scripts/retrain_v15_honest_eval.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from mlops.runner import run_experiment
from mlops.evaluation import evaluate_direction_prediction


# Paths
CACHE_PATH = Path("data/features/direction_prediction/feature_cache.parquet")
LABELS_PATH = Path("data/features/direction_prediction/labels.parquet")

# Date-based split (honest evaluation)
TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2025-12-31"

# Same hyperparams as ml_train.py
LR = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 512
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN = 128
SEED = 42
CONF_LONG = 0.60
CONF_SHORT = 0.35


class MLPBinaryDir(nn.Module):
    def __init__(self, input_size=10, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.out(h).squeeze(-1)


class SimpleDS(Dataset):
    def __init__(self, features, labels):
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- Load data ---
    print("Loading data...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    # --- Compute features (same as ml_train.py) ---
    print("Computing features...")
    close = fc["close"].values.astype(np.float64)
    high = fc["high"].values.astype(np.float64)
    low = fc["low"].values.astype(np.float64)

    for n in range(1, 9):
        roc = np.zeros(len(close), dtype=np.float32)
        roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
        fc[f"roc{n}"] = roc

    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / loss_s
    fc["rsi7_computed"] = (100 - (100 / (1 + rs))).values

    rp = np.zeros(len(close), dtype=np.float32)
    for i in range(50, len(close)):
        hh = np.max(high[i - 50:i + 1])
        ll = np.min(low[i - 50:i + 1])
        rng = hh - ll
        rp[i] = (close[i] - ll) / rng if rng > 0 else 0.5
    fc["range_position_50"] = rp

    feat_cols = [f"roc{n}" for n in range(1, 9)] + ["range_position_50", "rsi7_computed"]

    # --- Align ---
    common_idx = lb.index.intersection(fc.index)
    lb = lb.loc[common_idx]
    fc_aligned = fc.loc[common_idx]
    dates = common_idx

    # --- Labels: direction H96 (same as ml_train.py) ---
    direction = lb["direction"].values
    valid_mask = (direction == 0) | (direction == 1)
    y_binary = np.zeros(len(direction), dtype=np.float32)
    y_binary[direction == 0] = 1.0  # LONG=1, SHORT=0

    # --- Date-based split ---
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END) & valid_mask
    val_mask = (dates >= VAL_START) & (dates <= VAL_END) & valid_mask
    test_mask = (dates >= TEST_START) & (dates <= TEST_END) & valid_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"  Train: {len(train_idx)} bars (2020-2023)")
    print(f"  Val:   {len(val_idx)} bars (2024)")
    print(f"  Test:  {len(test_idx)} bars (2025)")

    # --- Scaler: fit on TRAIN ONLY ---
    feat_arr_raw = fc_aligned[feat_cols].values.astype(np.float32)
    feat_arr_raw = np.nan_to_num(feat_arr_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_mean = feat_arr_raw[train_idx].mean(axis=0)
    scaler_std = feat_arr_raw[train_idx].std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0

    feat_arr = (feat_arr_raw - scaler_mean) / scaler_std
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, y_train = feat_arr[train_idx], y_binary[train_idx]
    X_val, y_val = feat_arr[val_idx], y_binary[val_idx]
    X_test, y_test = feat_arr[test_idx], y_binary[test_idx]

    # MFE for magnitude metrics
    mfe_up_test = lb["mfe_up_96"].values[test_idx]
    mfe_down_test = lb["mfe_down_96"].values[test_idx]

    print(f"  Test LONG: {int(y_test.sum())}, SHORT: {int(len(y_test) - y_test.sum())}")

    # --- Train ---
    print("\nTraining...")
    train_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = MLPBinaryDir(input_size=10, hidden=HIDDEN)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for X_batch, y_batch in train_loader:
            logits = model(X_batch)
            loss = bce(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                vl += bce(model(X_batch), y_batch).item()
        vl /= len(val_loader)
        scheduler.step(vl)

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch <= 3:
                print(f"  Epoch {epoch:3d}: train={tr_loss:.4f} val={vl:.4f} ***")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best epoch: {epoch - PATIENCE})")
                break

    model.load_state_dict(best_state)

    # --- Predict on test ---
    print("\nPredicting on test set...")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        probs = torch.sigmoid(logits).numpy()

    print(f"  Predictions: min={probs.min():.3f}, max={probs.max():.3f}, mean={probs.mean():.3f}")
    print(f"  Confident LONG (>{CONF_LONG}): {(probs >= CONF_LONG).sum()}")
    print(f"  Confident SHORT (<{CONF_SHORT}): {(probs <= CONF_SHORT).sum()}")

    # --- Compute all 26 metrics ---
    print("\nComputing metrics...")
    test_metrics = evaluate_direction_prediction(
        probs=probs, labels=y_test,
        mfe_up_bps=mfe_up_test, mfe_down_bps=mfe_down_test,
        conf_threshold_long=CONF_LONG, conf_threshold_short=CONF_SHORT,
        split_name="test",
    )

    # Also compute train and val metrics
    with torch.no_grad():
        train_probs = torch.sigmoid(model(torch.from_numpy(X_train))).numpy()
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val))).numpy()

    train_metrics = evaluate_direction_prediction(
        probs=train_probs, labels=y_train, split_name="train",
    )
    val_metrics = evaluate_direction_prediction(
        probs=val_probs, labels=y_val, split_name="val",
    )

    # Merge (only take accuracy/confident from train/val, all from test)
    all_metrics = {}
    all_metrics.update(test_metrics)
    all_metrics["train_accuracy"] = train_metrics["train_accuracy"]
    all_metrics["train_confident_accuracy"] = train_metrics["train_confident_accuracy"]
    all_metrics["val_accuracy"] = val_metrics["val_accuracy"]
    all_metrics["val_confident_accuracy"] = val_metrics["val_confident_accuracy"]

    # Save model weights temporarily for artifact logging
    tmp_model_path = Path("tmp_honest_eval_model.pt")
    torch.save({"model": model.state_dict(), "val_loss": best_val_loss}, tmp_model_path)

    # --- Log through MLOps ---
    with run_experiment(
        experiment_name="direction_prediction",
        protocol_name="direction_prediction_v1",
        params={
            "model_type": "MLP",
            "architecture": "10-128-128-1",
            "features": "roc1-8+range_position_50+rsi7",
            "label": "direction_h96",
            "horizon": 96,
            "split": "date_based",
            "train": "2020-2023",
            "val": "2024",
            "test": "2025",
            "scaler_fit": "train_only",
            "conf_threshold_long": CONF_LONG,
            "conf_threshold_short": CONF_SHORT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        model_type="MLP",
        dataset_version="feature_cache_cleaned_23col",
        primary_metric="test_confident_accuracy",
        notes="V1.5 architecture — HONEST eval with date-based split. Train 2020-2023, test 2025. No data leakage.",
    ) as run:
        run.log_metrics(all_metrics)
        run.log_artifact(str(tmp_model_path))

        print("\n=== ALL METRICS ===")
        print(f"  train_accuracy:            {all_metrics['train_accuracy']}")
        print(f"  train_confident_accuracy:  {all_metrics['train_confident_accuracy']}")
        print(f"  val_accuracy:              {all_metrics['val_accuracy']}")
        print(f"  val_confident_accuracy:    {all_metrics['val_confident_accuracy']}")
        print()
        for k, v in sorted(test_metrics.items()):
            if "confusion" not in k:
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")

    # Cleanup
    tmp_model_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
