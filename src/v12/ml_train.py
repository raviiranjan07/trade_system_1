"""Train ML direction model and export to ONNX for production use.

Trains on ALL data (2020-2025) with random 10% validation split.
Saves ONNX model + scaler to src/v12/ml_model/

Requires torch (local training only — production uses onnxruntime).

Run: PYTHONPATH=src python -m v12.ml_train
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MODEL_DIR = Path("src/v12/ml_model")
MODEL_DIR.mkdir(exist_ok=True)

CACHE_PATH = Path("experiments/layer2/L2-003/feature_cache.parquet")
LABELS_PATH = Path("experiments/layer2/L2-003/labels.parquet")


# Model definition (training only — production uses ONNX)
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


def train_and_save():
    print("Loading data...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    close = fc["close"].values.astype(np.float64)
    for n in range(1, 9):
        roc = np.zeros(len(close), dtype=np.float32)
        roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
        fc[f"roc{n}"] = roc

    # RSI7
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(7).mean()
    rs = gain / loss_s
    fc["rsi7"] = (100 - (100 / (1 + rs))).values

    # Range position 50
    high = fc["high"].values.astype(np.float64)
    low = fc["low"].values.astype(np.float64)
    rp = np.zeros(len(close), dtype=np.float32)
    for i in range(50, len(close)):
        hh = np.max(high[i-50:i+1])
        ll = np.min(low[i-50:i+1])
        rng = hh - ll
        rp[i] = (close[i] - ll) / rng if rng > 0 else 0.5
    fc["range_position_50"] = rp

    feat_cols = [f"roc{n}" for n in range(1, 9)] + ["range_position_50", "rsi7"]

    common_idx = lb.index.intersection(fc.index)
    lb = lb.loc[common_idx]
    fc_aligned = fc.loc[common_idx]
    feat_arr_raw = fc_aligned[feat_cols].values.astype(np.float32)

    # Scaler from ALL data (2020-2025)
    scaler_mean = feat_arr_raw.mean(axis=0)
    scaler_std = feat_arr_raw.std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0

    feat_arr = (feat_arr_raw - scaler_mean) / scaler_std
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels
    direction = lb["direction"].values
    valid_mask = (direction == 0) | (direction == 1)
    y_binary = np.zeros(len(direction), dtype=np.float32)
    y_binary[direction == 0] = 1.0

    # Random 90/10 split
    valid_indices = np.where(valid_mask)[0]
    np.random.seed(42)
    np.random.shuffle(valid_indices)
    split_point = int(len(valid_indices) * 0.90)
    train_indices = valid_indices[:split_point]
    val_indices = valid_indices[split_point:]

    print(f"Total valid: {len(valid_indices)} | Train: {len(train_indices)} (90%) | Val: {len(val_indices)} (10%)")

    class SimpleDS(Dataset):
        def __init__(self, idx):
            self.idx = idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            n = self.idx[i]
            return torch.from_numpy(feat_arr[n]), torch.tensor(y_binary[n])

    model = MLPBinaryDir(input_size=10, hidden=128, dropout=0.5)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(SimpleDS(train_indices), batch_size=512, shuffle=True)
    val_loader = DataLoader(SimpleDS(val_indices), batch_size=512, shuffle=False)

    best_val_loss = float("inf")
    patience_ctr = 0

    print("\nTraining...")
    for epoch in range(1, 101):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            loss = bce(model(batch[0]), batch[1])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vl += bce(model(batch[0]), batch[1]).item()
        vl /= len(val_loader)
        scheduler.step(vl)

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch:3d}: train={tr_loss:.4f} val={vl:.4f} ***")
        else:
            patience_ctr += 1
            if patience_ctr >= 10:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # Quick accuracy check
    model.eval()
    all_pred, all_actual = [], []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch[0])
            all_pred.append(pred.numpy())
            all_actual.append(batch[1].numpy())
    logits = np.concatenate(all_pred)
    actuals = np.concatenate(all_actual)
    probs = 1 / (1 + np.exp(-logits))
    acc = ((probs > 0.5).astype(int) == actuals).mean() * 100
    conf_mask = (probs >= 0.60) | (probs <= 0.40)
    if conf_mask.sum() > 0:
        conf_acc = ((np.where(probs[conf_mask] > 0.5, 1, 0)) == actuals[conf_mask]).mean() * 100
        n_conf = conf_mask.sum()
    else:
        conf_acc = 0; n_conf = 0
    print(f"\n  Val accuracy: {acc:.1f}% | Confident: {n_conf} bars ({n_conf/len(probs)*100:.1f}%), acc={conf_acc:.1f}%")

    # Save .pt (for local use / future retraining)
    pt_path = MODEL_DIR / "direction_model.pt"
    torch.save({"model": model.state_dict(), "val_loss": best_val_loss}, pt_path)
    print(f"\n  PyTorch saved -> {pt_path}")

    # Export to ONNX (for production — no torch dependency needed)
    onnx_path = MODEL_DIR / "direction_model.onnx"
    dummy = torch.randn(1, 10)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["features"],
        output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=18,
    )
    print(f"  ONNX exported -> {onnx_path}")

    # Save scaler
    scaler_path = MODEL_DIR / "scaler.npz"
    np.savez(scaler_path, mean=scaler_mean, std=scaler_std)
    print(f"  Scaler saved -> {scaler_path}")

    print(f"\nDone. Model trained on 2020-2025 ({len(train_indices)} bars).")
    print("Production uses direction_model.onnx + scaler.npz (no torch needed).")


if __name__ == "__main__":
    train_and_save()
