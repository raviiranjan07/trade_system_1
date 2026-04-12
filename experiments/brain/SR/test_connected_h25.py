"""Connected LSTM: 25 snapshots x 4 features, H25, multi-task (direction + MFE).

Architecture:
  LSTM(4, 128) -> hidden(128)
    -> MFE head: Dense(128 -> 2) -> mfe_up, mfe_down
    -> Direction head: Dense(128+2 -> 128 -> ReLU -> 2) -> LONG/SHORT

Run: PYTHONPATH=src python experiments/brain/SR/test_connected_h25.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("CONNECTED LSTM: 25 snap x 4 feat, H25, multi-task")
print("=" * 60)

# Load features
fc = pd.read_parquet("data/features/direction_prediction/feature_cache.parquet")
fc.index = fc.index.tz_localize(None) if fc.index.tz is not None else fc.index

# Load OHLCV for label computation
from brain.config import load_config
from brain.ingestion import load_ohlcv
base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
open_arr = df["open"].values
close = df["close"].values
N_ohlcv = len(df)

# Align fc with df by date
common = fc.index.intersection(df.index)
fc = fc.loc[common]
# Map common dates to df indices
date_to_idx = {d: i for i, d in enumerate(df.index)}
common_idx = np.array([date_to_idx[d] for d in common])

# Compute roc1
if "roc1" not in fc.columns:
    fc["roc1"] = ((fc["close"] - fc["close"].shift(1)) / fc["close"].shift(1) * 10000).astype(np.float32)

# Features
feat_cols = ["roc1", "rsi7", "range_position", "sma200_dist_pct"]
features = fc[feat_cols].values.astype(np.float32)
print(f"Features: {feat_cols}")
print(f"Aligned bars: {len(common)}")

# Create H25 labels + MFE
print("\nCreating H25 labels + MFE...")
horizon = 25
threshold = 15

direction = np.full(len(common), -1, dtype=np.int64)
mfe_up = np.zeros(len(common), dtype=np.float32)
mfe_down = np.zeros(len(common), dtype=np.float32)

for ci in range(len(common)):
    i = common_idx[ci]
    if i + 1 >= N_ohlcv or i + horizon + 1 > N_ohlcv:
        continue
    entry = open_arr[i + 1]
    if entry <= 0:
        continue

    wh = high[i + 1:i + horizon + 1]
    wl = low[i + 1:i + horizon + 1]

    mu = max((np.max(wh) - entry) / entry * 10000, 0)
    md = max((entry - np.min(wl)) / entry * 10000, 0)
    mfe_up[ci] = mu
    mfe_down[ci] = md

    thu = entry * (1 + threshold / 10000)
    thd = entry * (1 - threshold / 10000)
    fu, fd = 0, 0
    for j in range(1, horizon + 1):
        if i + j >= N_ohlcv:
            break
        if fu == 0 and high[i + j] >= thu:
            fu = j
        if fd == 0 and low[i + j] <= thd:
            fd = j

    if fu > 0 and fd > 0:
        direction[ci] = 0 if fu < fd else (1 if fd < fu else 2)
    elif fu > 0:
        direction[ci] = 0
    elif fd > 0:
        direction[ci] = 1
    else:
        direction[ci] = 3

# Binary only
valid = (direction == 0) | (direction == 1)
print(f"  LONG: {(direction==0).sum()} SHORT: {(direction==1).sum()} BOTH: {(direction==2).sum()} SKIP: {(direction==3).sum()}")
print(f"  Valid: {valid.sum()}")

# Create 25-bar snapshots
print("\nCreating 25-bar snapshots...")
snapshot_count = 25
X_list, Y_dir_list, Y_mfe_list, D_list = [], [], [], []

for ci in range(snapshot_count, len(common)):
    if not valid[ci]:
        continue
    snap = features[ci - snapshot_count + 1:ci + 1, :]
    if np.any(np.isnan(snap)):
        continue
    X_list.append(snap)
    Y_dir_list.append(direction[ci])
    Y_mfe_list.append([mfe_up[ci], mfe_down[ci]])
    D_list.append(common[ci])

X = np.array(X_list, dtype=np.float32)
Y_dir = np.array(Y_dir_list, dtype=np.int64)
Y_mfe = np.array(Y_mfe_list, dtype=np.float32)
dates = np.array(D_list)

print(f"  Samples: {len(X)}, shape: {X.shape}")
print(f"  LONG: {(Y_dir==0).sum()} ({(Y_dir==0).mean()*100:.1f}%)")
print(f"  SHORT: {(Y_dir==1).sum()} ({(Y_dir==1).mean()*100:.1f}%)")
print(f"  MFE up mean: {Y_mfe[:,0].mean():.1f} bps, down mean: {Y_mfe[:,1].mean():.1f} bps")

# Split
train_mask = dates < pd.Timestamp("2023-01-01")
val_mask = (dates >= pd.Timestamp("2023-01-01")) & (dates < pd.Timestamp("2024-01-01"))
test_mask = dates >= pd.Timestamp("2024-01-01")

# Normalize features
tf = X[train_mask].reshape(-1, X.shape[-1])
mean = np.nanmean(tf, axis=0)
std = np.nanstd(tf, axis=0)
std[std == 0] = 1.0
X = np.nan_to_num((X - mean) / std, nan=0.0)

# Normalize MFE targets (z-score on train)
mfe_mean = Y_mfe[train_mask].mean(axis=0)
mfe_std = Y_mfe[train_mask].std(axis=0)
mfe_std[mfe_std == 0] = 1.0
Y_mfe_norm = (Y_mfe - mfe_mean) / mfe_std

tr_X = torch.FloatTensor(X[train_mask])
va_X = torch.FloatTensor(X[val_mask])
te_X = torch.FloatTensor(X[test_mask])
tr_dir = torch.LongTensor(Y_dir[train_mask])
va_dir = torch.LongTensor(Y_dir[val_mask])
te_dir = torch.LongTensor(Y_dir[test_mask])
tr_mfe = torch.FloatTensor(Y_mfe_norm[train_mask])
va_mfe = torch.FloatTensor(Y_mfe_norm[val_mask])
te_mfe = torch.FloatTensor(Y_mfe_norm[test_mask])

print(f"\n  Train: {len(tr_dir)} | Val: {len(va_dir)} | Test: {len(te_dir)}")

# Connected model
class ConnectedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        # MFE head
        self.mfe_head = nn.Linear(128, 2)

        # Direction head (sees hidden + mfe predictions)
        self.dir_fc1 = nn.Linear(128 + 2, 128)
        self.dir_relu = nn.ReLU()
        self.dir_dropout = nn.Dropout(0.2)
        self.dir_fc2 = nn.Linear(128, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = self.dropout(h[-1])

        mfe_pred = self.mfe_head(h)
        combined = torch.cat([h, mfe_pred], dim=1)
        dir_out = self.dir_relu(self.dir_fc1(combined))
        dir_out = self.dir_dropout(dir_out)
        dir_out = self.dir_fc2(dir_out)

        return dir_out, mfe_pred

model = ConnectedLSTM()
print(f"\nParams: {sum(p.numel() for p in model.parameters())}")

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
dir_weight = 1.0
mfe_weight = 5.0

loader = DataLoader(TensorDataset(tr_X, tr_dir, tr_mfe), batch_size=512, shuffle=True)

print(f"\nLoss = {dir_weight} x direction + {mfe_weight} x MFE")
print("\n%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s" %
      ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_LONG", "Va_SHORT"))

best_vl = float("inf")
pat = 0
best_st = None

for epoch in range(200):
    model.train()
    tl, co, to = 0, 0, 0
    for xb, db, mb in loader:
        opt.zero_grad()
        dp, mp = model(xb)
        loss = dir_weight * ce_loss(dp, db) + mfe_weight * mse_loss(mp, mb)
        loss.backward()
        opt.step()
        tl += loss.item() * len(db)
        co += (dp.argmax(1) == db).sum().item()
        to += len(db)

    model.eval()
    with torch.no_grad():
        vdp, vmp = model(va_X)
        vl = (dir_weight * ce_loss(vdp, va_dir) + mfe_weight * mse_loss(vmp, va_mfe)).item()
        vc = vdp.argmax(1)
        va = (vc == va_dir).float().mean().item() * 100
        vla = (vc[va_dir == 0] == 0).float().mean().item() * 100 if (va_dir == 0).sum() > 0 else 0
        vsa = (vc[va_dir == 1] == 1).float().mean().item() * 100 if (va_dir == 1).sum() > 0 else 0

    if epoch % 10 == 0 or epoch < 5:
        print("%-6d | %-8.1f %-8.4f | %-8.1f %-8.4f | %-8.1f %-8.1f" %
              (epoch, co/to*100, tl/to, va, vl, vla, vsa))

    if vl < best_vl:
        best_vl = vl
        pat = 0
        best_st = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        pat += 1
        if pat >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_st)
model.eval()
with torch.no_grad():
    tdp, tmp = model(te_X)
    tc = tdp.argmax(1)
    ta = (tc == te_dir).float().mean().item() * 100
    tla = (tc[te_dir == 0] == 0).float().mean().item() * 100
    tsa = (tc[te_dir == 1] == 1).float().mean().item() * 100

    probs = torch.softmax(tdp, dim=1).max(dim=1).values
    print()
    for ct in [0.55, 0.60, 0.65, 0.70]:
        m = probs >= ct
        if m.sum() > 0:
            ca = (tc[m] == te_dir[m]).float().mean().item() * 100
            print(f"  Confident (>{ct}): {ca:.1f}% ({m.sum()} bars, {m.sum()/len(te_dir)*100:.1f}%)")

print(f"\nTEST RESULTS (Connected LSTM, H25):")
print(f"  Overall: {ta:.1f}%")
print(f"  LONG: {tla:.1f}% ({(te_dir==0).sum()} samples)")
print(f"  SHORT: {tsa:.1f}% ({(te_dir==1).sum()} samples)")
print(f"  V1.5 MLP reference (H8): 52.1% overall, 58.2% confident")
print(f"  Baseline: 50.0%")
