"""Test base features (roc, rsi7, range_position, sma200_dist) with H8 and H25.

8 snapshots x 4 features. LSTM. Binary LONG/SHORT.

Run: PYTHONPATH=src python experiments/brain/SR/test_base_h25.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from brain.config import load_config
from brain.ingestion import load_ohlcv

# === Load data ===
print("=" * 60)
print("TEST: 4 base features x 8 snapshots, H8 vs H25")
print("=" * 60)

base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
close = df["close"].values
open_arr = df["open"].values
N = len(df)

# Load features
fc = pd.read_parquet("experiments/layer2/L2-003/feature_cache.parquet")
fc.index = fc.index.tz_localize(None) if fc.index.tz is not None else fc.index

# Compute roc1 if missing
if "roc1" not in fc.columns:
    fc["roc1"] = ((fc["close"] - fc["close"].shift(1)) / fc["close"].shift(1) * 10000).astype(np.float32)

# Find sma200 column
sma_col = "sma200_dist_pct"
if sma_col not in fc.columns:
    sma_cols = [c for c in fc.columns if "sma200" in c.lower()]
    print(f"  SMA200 columns available: {sma_cols}")
    if sma_cols:
        sma_col = sma_cols[0]

feat_cols = ["roc1", "rsi7", "range_position", sma_col]
features = fc[feat_cols].values.astype(np.float32)
print(f"Features: {feat_cols}")


def create_labels(horizon):
    labs = np.full(N, -1, dtype=np.int64)
    for i in range(N - horizon):
        entry = open_arr[i + 1]
        if entry <= 0:
            continue
        thu = entry * (1 + 15 / 10000)
        thd = entry * (1 - 15 / 10000)
        fu, fd = 0, 0
        for j in range(1, horizon + 1):
            if i + j >= N:
                break
            if fu == 0 and high[i + j] >= thu:
                fu = j
            if fd == 0 and low[i + j] <= thd:
                fd = j
        if fu > 0 and fd > 0:
            labs[i] = 0 if fu < fd else (1 if fd < fu else 2)
        elif fu > 0:
            labs[i] = 0
        elif fd > 0:
            labs[i] = 1
        else:
            labs[i] = 3
    return labs


def build_snapshots(labels):
    valid = (labels == 0) | (labels == 1)
    X_list, Y_list, D_list = [], [], []
    for i in range(8, N):
        if not valid[i]:
            continue
        snap = features[i - 7:i + 1, :]
        if np.any(np.isnan(snap)):
            continue
        X_list.append(snap)
        Y_list.append(labels[i])
        D_list.append(df.index[i])
    return np.array(X_list, np.float32), np.array(Y_list, np.int64), np.array(D_list)


def train_and_test(X, Y, dates, label_name):
    print(f"\n{'='*60}")
    print(f"TRAINING: {label_name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(Y)}, LONG: {(Y==0).sum()} ({(Y==0).mean()*100:.1f}%), SHORT: {(Y==1).sum()} ({(Y==1).mean()*100:.1f}%)")

    train_mask = dates < pd.Timestamp("2023-01-01")
    val_mask = (dates >= pd.Timestamp("2023-01-01")) & (dates < pd.Timestamp("2024-01-01"))
    test_mask = dates >= pd.Timestamp("2024-01-01")

    # Normalize
    tf = X[train_mask].reshape(-1, X.shape[-1])
    mean = np.nanmean(tf, axis=0)
    std = np.nanstd(tf, axis=0)
    std[std == 0] = 1.0
    Xn = np.nan_to_num((X - mean) / std, nan=0.0)

    tr_X = torch.FloatTensor(Xn[train_mask])
    va_X = torch.FloatTensor(Xn[val_mask])
    te_X = torch.FloatTensor(Xn[test_mask])
    tr_Y = torch.LongTensor(Y[train_mask])
    va_Y = torch.LongTensor(Y[val_mask])
    te_Y = torch.LongTensor(Y[test_mask])

    print(f"  Train: {len(tr_Y)} | Val: {len(va_Y)} | Test: {len(te_Y)}")

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(4, 128, num_layers=1, batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            self.fc1 = nn.Linear(128, 128)
            self.relu = nn.ReLU()
            self.dropout2 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(128, 2)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            h = self.dropout1(h[-1])
            h = self.relu(self.fc1(h))
            h = self.dropout2(h)
            return self.fc2(h)

    model = LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tr_X, tr_Y), batch_size=512, shuffle=True)

    best_vl = float("inf")
    pat = 0
    best_st = None

    print("\n  %-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s" %
          ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_LONG", "Va_SHORT"))

    for epoch in range(100):
        model.train()
        tl, co, to = 0, 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            p = model(xb)
            l = crit(p, yb)
            l.backward()
            opt.step()
            tl += l.item() * len(yb)
            co += (p.argmax(1) == yb).sum().item()
            to += len(yb)

        model.eval()
        with torch.no_grad():
            vp = model(va_X)
            vl = crit(vp, va_Y).item()
            vc = vp.argmax(1)
            va = (vc == va_Y).float().mean().item() * 100
            vl_acc = (vc[va_Y == 0] == 0).float().mean().item() * 100 if (va_Y == 0).sum() > 0 else 0
            vs_acc = (vc[va_Y == 1] == 1).float().mean().item() * 100 if (va_Y == 1).sum() > 0 else 0

        if epoch % 10 == 0 or epoch < 5:
            print("  %-6d | %-8.1f %-8.4f | %-8.1f %-8.4f | %-8.1f %-8.1f" %
                  (epoch, co/to*100, tl/to, va, vl, vl_acc, vs_acc))

        if vl < best_vl:
            best_vl = vl
            pat = 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= 15:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        tp = model(te_X)
        tc = tp.argmax(1)
        ta = (tc == te_Y).float().mean().item() * 100
        tl_acc = (tc[te_Y == 0] == 0).float().mean().item() * 100
        ts_acc = (tc[te_Y == 1] == 1).float().mean().item() * 100

        probs = torch.softmax(tp, dim=1).max(dim=1).values
        for ct in [0.55, 0.60, 0.65]:
            m = probs >= ct
            if m.sum() > 0:
                ca = (tc[m] == te_Y[m]).float().mean().item() * 100
                print(f"  Confident (>{ct}): {ca:.1f}% ({m.sum()} bars, {m.sum()/len(te_Y)*100:.1f}%)")

    print(f"\n  TEST {label_name}: {ta:.1f}% (LONG={tl_acc:.1f}%, SHORT={ts_acc:.1f}%)")
    return ta


# === Run both ===
print("\nCreating labels...")
lab_h8 = create_labels(8)
lab_h25 = create_labels(25)

print("\n--- H8 ---")
print(f"  LONG: {(lab_h8==0).sum()} SHORT: {(lab_h8==1).sum()} BOTH: {(lab_h8==2).sum()} SKIP: {(lab_h8==3).sum()}")

print("\n--- H25 ---")
print(f"  LONG: {(lab_h25==0).sum()} SHORT: {(lab_h25==1).sum()} BOTH: {(lab_h25==2).sum()} SKIP: {(lab_h25==3).sum()}")

X_h8, Y_h8, D_h8 = build_snapshots(lab_h8)
X_h25, Y_h25, D_h25 = build_snapshots(lab_h25)

acc_h8 = train_and_test(X_h8, Y_h8, D_h8, "H8 (8-bar)")
acc_h25 = train_and_test(X_h25, Y_h25, D_h25, "H25 (25-bar)")

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"  H8:  {acc_h8:.1f}%")
print(f"  H25: {acc_h25:.1f}%")
print(f"  V1.5 reference (H96, MLP): 57-58%")
print(f"  Baseline: 50.0%")
