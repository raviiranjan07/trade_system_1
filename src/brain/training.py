"""Brain Pipeline -- Step 5: Training

Train the brain model on GPU (designed for Google Colab T4).
Loads dataset from .npz files, trains, saves model + history.

Usage on Colab:
  Upload: train.npz, val.npz, scaler.npz, metadata.json, config.yaml
  Run: python training.py --data_dir ./datasets --config config.yaml --output_dir ./output

Usage locally:
  PYTHONPATH=src python -m brain.training --data_dir data/archive/brain_datasets --config configs/base.yaml --output_dir experiments/brain/EXP-B001
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


def load_datasets(data_dir):
    """Load train/val/test from .npz files."""
    data_dir = Path(data_dir)

    train = np.load(data_dir / "train.npz")
    val = np.load(data_dir / "val.npz")
    test = np.load(data_dir / "test.npz")

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    return {
        "train_X": train["X"], "train_Y": train["Y"],
        "val_X": val["X"], "val_Y": val["Y"],
        "test_X": test["X"], "test_Y": test["Y"],
        "meta": meta,
    }


def load_config_yaml(config_path):
    """Load YAML config. Works with or without brain.config module."""
    try:
        from brain.config import load_config
        return load_config(config_path)
    except ImportError:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)


def build_model_from_config(config, num_features, snapshot_count):
    """Build model. Works with or without brain.models module."""
    try:
        from brain.models.lstm import build_model
        return build_model(config, num_features, snapshot_count)
    except ImportError:
        # Inline model definitions for Colab
        arch = config["training"]["architecture"]
        model_cfg = config["training"]["model"]
        hidden = model_cfg.get("hidden_size", 128)
        layers = model_cfg.get("num_layers", 1)
        dropout = model_cfg.get("dropout", 0.2)
        mfe_horizons = len(config["labels"].get("mfe", {}).get("horizons", [1,2,3,4,5,6,7,8]))

        if arch == "lstm":
            return _InlineLSTM(num_features, hidden, layers, dropout, mfe_horizons)
        elif arch == "mlp":
            return _InlineMLP(num_features, snapshot_count, hidden, dropout, mfe_horizons)
        else:
            raise ValueError(f"Unknown architecture: {arch}")


class _InlineLSTM(nn.Module):
    """Standalone LSTM for Colab (no imports needed)."""
    def __init__(self, nf, hidden, layers, dropout, mfe_h):
        super().__init__()
        self.lstm = nn.LSTM(nf, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.dir_head = nn.Linear(hidden, 2)
        self.mfe_head = nn.Linear(hidden, mfe_h * 2)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = self.drop(h[-1])
        return self.dir_head(h), self.mfe_head(h)


class _InlineMLP(nn.Module):
    """Standalone MLP for Colab."""
    def __init__(self, nf, snap, hidden, dropout, mfe_h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nf * snap, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.dir_head = nn.Linear(hidden, 2)
        self.mfe_head = nn.Linear(hidden, mfe_h * 2)
    def forward(self, x):
        h = self.net(x.reshape(x.size(0), -1))
        return self.dir_head(h), self.mfe_head(h)


def train_model(config, data_dir, output_dir):
    """Full training loop."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup ---
    train_cfg = config["training"]
    seed = train_cfg.get("seed", config.get("reproducibility", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = train_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading datasets...")
    data = load_datasets(data_dir)
    meta = data["meta"]
    num_features = meta["snapshot_shape"][1]
    snapshot_count = meta["snapshot_shape"][0]
    print(f"  Features: {num_features}, Snapshots: {snapshot_count}")
    print(f"  Train: {len(data['train_X'])}, Val: {len(data['val_X'])}, Test: {len(data['test_X'])}")

    # --- Build model ---
    model = build_model_from_config(config, num_features, snapshot_count)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {train_cfg['architecture']}, params: {total_params:,}")

    # --- Data loaders ---
    batch_size = train_cfg["schedule"]["batch_size"]

    # Separate direction and MFE labels
    # Y columns: [direction, mfe_up_1, mfe_down_1, mfe_up_2, mfe_down_2, ...]
    train_dir = data["train_Y"][:, 0].astype(np.int64)
    train_mfe = data["train_Y"][:, 1:].astype(np.float32)
    val_dir = data["val_Y"][:, 0].astype(np.int64)
    val_mfe = data["val_Y"][:, 1:].astype(np.float32)
    test_dir = data["test_Y"][:, 0].astype(np.int64)
    test_mfe = data["test_Y"][:, 1:].astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(data["train_X"]),
            torch.tensor(train_dir),
            torch.tensor(train_mfe),
        ),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(data["val_X"]),
            torch.tensor(val_dir),
            torch.tensor(val_mfe),
        ),
        batch_size=batch_size, shuffle=False,
    )

    # --- Loss functions ---
    task_cfg = train_cfg.get("tasks", {})
    dir_weight = task_cfg.get("direction", {}).get("weight", 1.0)
    mfe_weight = task_cfg.get("mfe", {}).get("weight", 5.0)
    dir_enabled = task_cfg.get("direction", {}).get("enabled", True)
    mfe_enabled = task_cfg.get("mfe", {}).get("enabled", True)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # --- Optimizer ---
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0),
    )

    # --- Training loop ---
    max_epochs = train_cfg["schedule"]["max_epochs"]
    patience = train_cfg["schedule"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\nTraining ({max_epochs} max epochs, patience={patience})...")
    print(f"  Direction weight: {dir_weight}, MFE weight: {mfe_weight}")

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0
        train_correct = 0
        train_total = 0

        for X_batch, dir_batch, mfe_batch in train_loader:
            X_batch = X_batch.to(device)
            dir_batch = dir_batch.to(device)
            mfe_batch = mfe_batch.to(device)

            optimizer.zero_grad()

            dir_pred, mfe_pred = model(X_batch)

            loss = torch.tensor(0.0, device=device)
            if dir_enabled:
                loss = loss + dir_weight * ce_loss(dir_pred, dir_batch)
            if mfe_enabled:
                loss = loss + mfe_weight * mse_loss(mfe_pred, mfe_batch)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(X_batch)
            if dir_enabled:
                pred_class = dir_pred.argmax(dim=1)
                train_correct += (pred_class == dir_batch).sum().item()
            train_total += len(X_batch)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total * 100 if train_total > 0 else 0

        # --- Validate ---
        model.eval()
        val_loss_sum = 0
        val_correct = 0
        val_total = 0

        # Track confident predictions
        val_probs_all = []
        val_dirs_all = []

        with torch.no_grad():
            for X_batch, dir_batch, mfe_batch in val_loader:
                X_batch = X_batch.to(device)
                dir_batch = dir_batch.to(device)
                mfe_batch = mfe_batch.to(device)

                dir_pred, mfe_pred = model(X_batch)

                loss = torch.tensor(0.0, device=device)
                if dir_enabled:
                    loss = loss + dir_weight * ce_loss(dir_pred, dir_batch)
                if mfe_enabled:
                    loss = loss + mfe_weight * mse_loss(mfe_pred, mfe_batch)

                val_loss_sum += loss.item() * len(X_batch)

                if dir_enabled:
                    probs = torch.softmax(dir_pred, dim=1)
                    pred_class = dir_pred.argmax(dim=1)
                    val_correct += (pred_class == dir_batch).sum().item()
                    val_probs_all.append(probs.cpu().numpy())
                    val_dirs_all.append(dir_batch.cpu().numpy())

                val_total += len(X_batch)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0

        # Confident accuracy
        conf_acc = 0
        conf_count = 0
        if val_probs_all:
            all_probs = np.concatenate(val_probs_all)
            all_dirs = np.concatenate(val_dirs_all)
            max_prob = np.max(all_probs, axis=1)
            conf_mask = max_prob >= 0.60
            if conf_mask.sum() > 0:
                pred_conf = np.argmax(all_probs[conf_mask], axis=1)
                conf_acc = (pred_conf == all_dirs[conf_mask]).mean() * 100
                conf_count = conf_mask.sum()

        # Log
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "conf_acc": conf_acc,
            "conf_count": int(conf_count),
        })

        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.1f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% | "
              f"conf({conf_count})={conf_acc:.1f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # --- Final evaluation on test set ---
    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
    model.eval()

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(data["test_X"]),
            torch.tensor(test_dir),
            torch.tensor(test_mfe),
        ),
        batch_size=batch_size, shuffle=False,
    )

    test_correct = 0
    test_total = 0
    test_probs_all = []
    test_dirs_all = []
    test_mfe_pred_all = []
    test_mfe_true_all = []

    with torch.no_grad():
        for X_batch, dir_batch, mfe_batch in test_loader:
            X_batch = X_batch.to(device)
            dir_batch = dir_batch.to(device)

            dir_pred, mfe_pred = model(X_batch)

            probs = torch.softmax(dir_pred, dim=1)
            pred_class = dir_pred.argmax(dim=1)
            test_correct += (pred_class == dir_batch).sum().item()
            test_total += len(X_batch)

            test_probs_all.append(probs.cpu().numpy())
            test_dirs_all.append(dir_batch.cpu().numpy())
            test_mfe_pred_all.append(mfe_pred.cpu().numpy())
            test_mfe_true_all.append(mfe_batch.numpy())

    test_acc = test_correct / test_total * 100

    all_probs = np.concatenate(test_probs_all)
    all_dirs = np.concatenate(test_dirs_all)
    all_mfe_pred = np.concatenate(test_mfe_pred_all)
    all_mfe_true = np.concatenate(test_mfe_true_all)

    # Confident accuracy at various thresholds
    print(f"\n  Test overall accuracy: {test_acc:.1f}%")
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        max_prob = np.max(all_probs, axis=1)
        mask = max_prob >= thresh
        if mask.sum() > 0:
            pred = np.argmax(all_probs[mask], axis=1)
            acc = (pred == all_dirs[mask]).mean() * 100
            print(f"  Confident (>{thresh}): {acc:.1f}% ({mask.sum()} bars)")

    # LONG vs SHORT accuracy
    long_mask = all_dirs == 0
    short_mask = all_dirs == 1
    pred_all = np.argmax(all_probs, axis=1)
    if long_mask.sum() > 0:
        long_acc = (pred_all[long_mask] == 0).mean() * 100
        print(f"  LONG accuracy: {long_acc:.1f}% ({long_mask.sum()} bars)")
    if short_mask.sum() > 0:
        short_acc = (pred_all[short_mask] == 1).mean() * 100
        print(f"  SHORT accuracy: {short_acc:.1f}% ({short_mask.sum()} bars)")

    # MFE MAE
    mfe_mae = np.mean(np.abs(all_mfe_pred - all_mfe_true))
    print(f"  MFE MAE: {mfe_mae:.2f} bps")

    # --- Save results ---
    elapsed = time.time() - start

    metrics = {
        "architecture": train_cfg["architecture"],
        "num_features": num_features,
        "total_params": total_params,
        "epochs_trained": len(history),
        "best_val_loss": float(best_val_loss),
        "train_acc": float(history[-1]["train_acc"]),
        "val_acc": float(history[-1]["val_acc"]),
        "test_acc": float(test_acc),
        "test_long_acc": float(long_acc) if long_mask.sum() > 0 else None,
        "test_short_acc": float(short_acc) if short_mask.sum() > 0 else None,
        "mfe_mae": float(mfe_mae),
        "training_time_sec": elapsed,
    }

    # Add confident accuracies
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        max_prob = np.max(all_probs, axis=1)
        mask = max_prob >= thresh
        if mask.sum() > 0:
            pred = np.argmax(all_probs[mask], axis=1)
            acc = (pred == all_dirs[mask]).mean() * 100
            metrics[f"test_conf_{int(thresh*100)}"] = float(acc)
            metrics[f"test_conf_{int(thresh*100)}_count"] = int(mask.sum())

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save history
    import csv
    with open(output_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"\nSaved to {output_dir}/")
    print(f"  model.pt, metrics.json, training_history.csv")
    print(f"  Training time: {elapsed:.1f}s")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config_yaml(args.config)
    train_model(config, args.data_dir, args.output_dir)
