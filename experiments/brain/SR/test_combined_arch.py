"""Combined Architecture: Base(LSTM) + S/R Dynamic(Dense) + S/R Static(direct) + MFE connection

3 separate paths combine at direction head.
H25 label, binary LONG/SHORT, multi-task (direction + MFE).

Run: PYTHONPATH=src python experiments/brain/SR/test_combined_arch.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from brain.config import load_config
from brain.ingestion import load_ohlcv

print("=" * 60)
print("COMBINED ARCHITECTURE TEST")
print("=" * 60)

# === Load data ===
base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
close = df["close"].values
open_arr = df["open"].values
N_ohlcv = len(df)

# Load base features
fc = pd.read_parquet("data/features/direction_prediction/feature_cache.parquet")
fc.index = fc.index.tz_localize(None) if fc.index.tz is not None else fc.index
common = fc.index.intersection(df.index)
fc = fc.loc[common]
date_to_idx = {d: i for i, d in enumerate(df.index)}
common_idx = np.array([date_to_idx[d] for d in common])

if "roc1" not in fc.columns:
    fc["roc1"] = ((fc["close"] - fc["close"].shift(1)) / fc["close"].shift(1) * 10000).astype(np.float32)

base_cols = ["roc1", "rsi7", "range_position", "sma200_dist_pct"]
base_features = fc[base_cols].values.astype(np.float32)
print(f"Base features: {base_cols}")

# Load S/R features from every_bar dataset
print("\nLoading S/R features...")
sr_train = np.load("data/features/sr_bounce_break/every_bar/train_stage6.npz")
sr_val = np.load("data/features/sr_bounce_break/every_bar/val_stage6.npz")
sr_test = np.load("data/features/sr_bounce_break/every_bar/test_stage6.npz")

# Get bar numbers for S/R events
sr_bars_train = sr_train["bars"]
sr_bars_val = sr_val["bars"]
sr_bars_test = sr_test["bars"]

# S/R dynamic and static
# stage6 data is 2D (already flattened to single bar)
sr_dyn_train = sr_train["X_sr_dynamic"] if sr_train["X_sr_dynamic"].ndim == 2 else sr_train["X_sr_dynamic"][:, -1, :]
sr_sta_train = sr_train["X_sr_static"]
sr_dyn_val = sr_val["X_sr_dynamic"] if sr_val["X_sr_dynamic"].ndim == 2 else sr_val["X_sr_dynamic"][:, -1, :]
sr_sta_val = sr_val["X_sr_static"]
sr_dyn_test = sr_test["X_sr_dynamic"] if sr_test["X_sr_dynamic"].ndim == 2 else sr_test["X_sr_dynamic"][:, -1, :]
sr_sta_test = sr_test["X_sr_static"]

sr_dyn_size = sr_dyn_train.shape[1]
sr_sta_size = sr_sta_train.shape[1]
print(f"S/R dynamic: {sr_dyn_size} features")
print(f"S/R static: {sr_sta_size} features")

# === Create H25 labels + MFE for S/R event bars ===
print("\nCreating H25 labels + MFE for S/R events...")
horizon = 25
threshold = 15

def create_labels_mfe(bars):
    direction = np.full(len(bars), -1, dtype=np.int64)
    mfe = np.zeros((len(bars), 2), dtype=np.float32)
    for ci in range(len(bars)):
        i = bars[ci]
        if i + 1 >= N_ohlcv or i + horizon + 1 > N_ohlcv:
            continue
        entry = open_arr[i + 1]
        if entry <= 0:
            continue
        wh = high[i + 1:i + horizon + 1]
        wl = low[i + 1:i + horizon + 1]
        mfe[ci, 0] = max((np.max(wh) - entry) / entry * 10000, 0)
        mfe[ci, 1] = max((entry - np.min(wl)) / entry * 10000, 0)
        thu = entry * (1 + threshold / 10000)
        thd = entry * (1 - threshold / 10000)
        fu, fd = 0, 0
        for j in range(1, horizon + 1):
            if i + j >= N_ohlcv: break
            if fu == 0 and high[i + j] >= thu: fu = j
            if fd == 0 and low[i + j] <= thd: fd = j
        if fu > 0 and fd > 0:
            direction[ci] = 0 if fu < fd else (1 if fd < fu else 2)
        elif fu > 0: direction[ci] = 0
        elif fd > 0: direction[ci] = 1
        else: direction[ci] = 3
    return direction, mfe

dir_train, mfe_train = create_labels_mfe(sr_bars_train)
dir_val, mfe_val = create_labels_mfe(sr_bars_val)
dir_test, mfe_test = create_labels_mfe(sr_bars_test)

# === Create 25-bar base feature snapshots for each S/R event ===
print("Creating base feature snapshots for S/R events...")
snapshot_count = 25

def create_snapshots(bars, direction, mfe, sr_dyn, sr_sta):
    valid = (direction == 0) | (direction == 1)
    X_base_list, X_dyn_list, X_sta_list = [], [], []
    Y_dir_list, Y_mfe_list = [], []

    for ci in range(len(bars)):
        if not valid[ci]:
            continue
        # Find bar index in fc
        bar = bars[ci]
        # Find position in common array
        pos = np.searchsorted(common_idx, bar)
        if pos >= len(common_idx) or common_idx[pos] != bar:
            continue
        if pos < snapshot_count:
            continue

        snap = base_features[pos - snapshot_count + 1:pos + 1, :]
        if np.any(np.isnan(snap)):
            continue

        X_base_list.append(snap)
        X_dyn_list.append(sr_dyn[ci])
        X_sta_list.append(sr_sta[ci])
        Y_dir_list.append(direction[ci])
        Y_mfe_list.append(mfe[ci])

    return (np.array(X_base_list, np.float32),
            np.array(X_dyn_list, np.float32),
            np.array(X_sta_list, np.float32),
            np.array(Y_dir_list, np.int64),
            np.array(Y_mfe_list, np.float32))

Xb_tr, Xd_tr, Xs_tr, Yd_tr, Ym_tr = create_snapshots(sr_bars_train, dir_train, mfe_train, sr_dyn_train, sr_sta_train)
Xb_va, Xd_va, Xs_va, Yd_va, Ym_va = create_snapshots(sr_bars_val, dir_val, mfe_val, sr_dyn_val, sr_sta_val)
Xb_te, Xd_te, Xs_te, Yd_te, Ym_te = create_snapshots(sr_bars_test, dir_test, mfe_test, sr_dyn_test, sr_sta_test)

print(f"\nTrain: {len(Yd_tr)} | Val: {len(Yd_va)} | Test: {len(Yd_te)}")
print(f"Base snapshots: {Xb_tr.shape}")
print(f"S/R dynamic: {Xd_tr.shape}")
print(f"S/R static: {Xs_tr.shape}")
print(f"LONG: {(Yd_tr==0).sum()} ({(Yd_tr==0).mean()*100:.1f}%)")
print(f"SHORT: {(Yd_tr==1).sum()} ({(Yd_tr==1).mean()*100:.1f}%)")

# === Normalize ===
print("\nNormalizing...")
# Base features
bf = Xb_tr.reshape(-1, Xb_tr.shape[-1])
b_mean, b_std = np.nanmean(bf, 0), np.nanstd(bf, 0)
b_std[b_std == 0] = 1.0
Xb_tr = np.nan_to_num((Xb_tr - b_mean) / b_std, nan=0.0)
Xb_va = np.nan_to_num((Xb_va - b_mean) / b_std, nan=0.0)
Xb_te = np.nan_to_num((Xb_te - b_mean) / b_std, nan=0.0)

# S/R dynamic
d_mean, d_std = np.nanmean(Xd_tr, 0), np.nanstd(Xd_tr, 0)
d_std[d_std == 0] = 1.0
Xd_tr = np.nan_to_num((Xd_tr - d_mean) / d_std, nan=0.0)
Xd_va = np.nan_to_num((Xd_va - d_mean) / d_std, nan=0.0)
Xd_te = np.nan_to_num((Xd_te - d_mean) / d_std, nan=0.0)

# S/R static
s_mean, s_std = np.nanmean(Xs_tr, 0), np.nanstd(Xs_tr, 0)
s_std[s_std == 0] = 1.0
Xs_tr = np.nan_to_num((Xs_tr - s_mean) / s_std, nan=0.0)
Xs_va = np.nan_to_num((Xs_va - s_mean) / s_std, nan=0.0)
Xs_te = np.nan_to_num((Xs_te - s_mean) / s_std, nan=0.0)

# MFE
m_mean, m_std = Ym_tr.mean(0), Ym_tr.std(0)
m_std[m_std == 0] = 1.0
Ym_tr_n = (Ym_tr - m_mean) / m_std
Ym_va_n = (Ym_va - m_mean) / m_std
Ym_te_n = (Ym_te - m_mean) / m_std

# Convert to tensors
tb = torch.FloatTensor(Xb_tr); vb = torch.FloatTensor(Xb_va); eb = torch.FloatTensor(Xb_te)
td = torch.FloatTensor(Xd_tr); vd = torch.FloatTensor(Xd_va); ed = torch.FloatTensor(Xd_te)
ts = torch.FloatTensor(Xs_tr); vs = torch.FloatTensor(Xs_va); es = torch.FloatTensor(Xs_te)
ty = torch.LongTensor(Yd_tr); vy = torch.LongTensor(Yd_va); ey = torch.LongTensor(Yd_te)
tm = torch.FloatTensor(Ym_tr_n); vm = torch.FloatTensor(Ym_va_n); em = torch.FloatTensor(Ym_te_n)

# === Model ===
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

class CombinedModel(nn.Module):
    def __init__(self, base_size=4, sr_dyn_size=11, sr_sta_size=6):
        super().__init__()
        # Path A: Base features through LSTM
        self.lstm = nn.LSTM(base_size, 128, num_layers=1, batch_first=True)
        self.lstm_drop = nn.Dropout(0.2)
        self.lstm_fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU())

        # Path B: S/R dynamic compression
        self.sr_dyn = nn.Sequential(nn.Linear(sr_dyn_size, 4), nn.ReLU())

        # MFE head (from base LSTM output)
        self.mfe_head = nn.Linear(32, 2)

        # Direction head: combines all paths
        # 32 (base) + 4 (sr_dyn) + 6 (sr_sta) + 2 (mfe) = 44
        self.dir_head = nn.Sequential(
            nn.Linear(32 + 4 + sr_sta_size + 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x_base, x_sr_dyn, x_sr_sta):
        # Path A: LSTM on base features
        _, (h, _) = self.lstm(x_base)
        h = self.lstm_drop(h[-1])
        base_out = self.lstm_fc(h)  # 32

        # Path B: compress S/R dynamic
        sr_dyn_out = self.sr_dyn(x_sr_dyn)  # 4

        # MFE prediction
        mfe_pred = self.mfe_head(base_out)  # 2

        # Combine all for direction
        combined = torch.cat([base_out, sr_dyn_out, x_sr_sta, mfe_pred], dim=1)  # 44
        dir_pred = self.dir_head(combined)

        return dir_pred, mfe_pred

model = CombinedModel(base_size=4, sr_dyn_size=sr_dyn_size, sr_sta_size=sr_sta_size)
print(f"Params: {sum(p.numel() for p in model.parameters())}")

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()

loader = DataLoader(TensorDataset(tb, td, ts, ty, tm), batch_size=256, shuffle=True)

print(f"\nLoss = 1.0 x direction + 5.0 x MFE")
print(f"\n%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s" %
      ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_LONG", "Va_SHORT"))

best_vl = float("inf")
pat = 0
best_st = None

for epoch in range(200):
    model.train()
    tl, co, to = 0, 0, 0
    for xb_b, xd_b, xs_b, y_b, m_b in loader:
        opt.zero_grad()
        dp, mp = model(xb_b, xd_b, xs_b)
        loss = 1.0 * ce_loss(dp, y_b) + 5.0 * mse_loss(mp, m_b)
        loss.backward()
        opt.step()
        tl += loss.item() * len(y_b)
        co += (dp.argmax(1) == y_b).sum().item()
        to += len(y_b)

    model.eval()
    with torch.no_grad():
        vdp, vmp = model(vb, vd, vs)
        vl = (1.0 * ce_loss(vdp, vy) + 5.0 * mse_loss(vmp, vm)).item()
        vc = vdp.argmax(1)
        va = (vc == vy).float().mean().item() * 100
        vla = (vc[vy == 0] == 0).float().mean().item() * 100 if (vy == 0).sum() > 0 else 0
        vsa = (vc[vy == 1] == 1).float().mean().item() * 100 if (vy == 1).sum() > 0 else 0

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

# === Test ===
model.load_state_dict(best_st)
model.eval()
with torch.no_grad():
    tdp, tmp = model(eb, ed, es)
    tc = tdp.argmax(1)
    ta = (tc == ey).float().mean().item() * 100
    tla = (tc[ey == 0] == 0).float().mean().item() * 100
    tsa = (tc[ey == 1] == 1).float().mean().item() * 100

    probs = torch.softmax(tdp, dim=1).max(dim=1).values
    print()
    for ct in [0.55, 0.60, 0.65, 0.70]:
        m = probs >= ct
        if m.sum() > 0:
            ca = (tc[m] == ey[m]).float().mean().item() * 100
            print(f"  Confident (>{ct}): {ca:.1f}% ({m.sum()} bars, {m.sum()/len(ey)*100:.1f}%)")

print(f"\n{'='*60}")
print("TEST RESULTS (Combined Architecture, H25)")
print(f"{'='*60}")
print(f"  Overall: {ta:.1f}%")
print(f"  LONG: {tla:.1f}% ({(ey==0).sum()} samples)")
print(f"  SHORT: {tsa:.1f}% ({(ey==1).sum()} samples)")
print(f"\nCOMPARISON:")
print(f"  Connected LSTM base-only (H25): 51.9% overall, 56.7% confident")
print(f"  V1.5 MLP (H8): 52.1% overall, 58.2% confident")
print(f"  Baseline: 50.0%")
