"""Stage 8: S/R Advisor Standalone Test

8a: Without MFE head
8b: With MFE head (connected)

Separate dynamic/static paths. Bounce/break prediction.

Run: PYTHONPATH=src python experiments/brain/SR/test_stage8.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from brain.config import load_config
from brain.ingestion import load_ohlcv

print("=" * 60)
print("STAGE 8: S/R Advisor Standalone Test")
print("=" * 60)

# === Load S/R features ===
print("\nLoading data...")
train = np.load("experiments/brain/SR/datasets_every_bar/train_stage6.npz")
val = np.load("experiments/brain/SR/datasets_every_bar/val_stage6.npz")
test = np.load("experiments/brain/SR/datasets_every_bar/test_stage6.npz")

sr_dyn_train = train["X_sr_dynamic"]  # (N, 11) or 2D
sr_sta_train = train["X_sr_static"]   # (N, 6)
sr_dyn_val = val["X_sr_dynamic"]
sr_sta_val = val["X_sr_static"]
sr_dyn_test = test["X_sr_dynamic"]
sr_sta_test = test["X_sr_static"]

# Handle 3D (take last snapshot) or 2D
if sr_dyn_train.ndim == 3:
    sr_dyn_train = sr_dyn_train[:, -1, :]
    sr_dyn_val = sr_dyn_val[:, -1, :]
    sr_dyn_test = sr_dyn_test[:, -1, :]

dyn_size = sr_dyn_train.shape[1]
sta_size = sr_sta_train.shape[1]
print(f"  S/R dynamic: {dyn_size}, static: {sta_size}")

# === Load OHLCV for labels ===
base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
open_arr = df["open"].values
N = len(df)

bars_train = train["bars"]
bars_val = val["bars"]
bars_test = test["bars"]

# === Create bounce/break labels (direction-normalized) ===
print("\nCreating bounce/break labels (H25, direction-normalized)...")
horizon = 25
threshold = 15

def create_bounce_break_labels(bars, sr_sta):
    labels = np.full(len(bars), -1, dtype=np.int64)
    mfe = np.zeros((len(bars), 2), dtype=np.float32)

    for ci in range(len(bars)):
        i = bars[ci]
        if i + 1 >= N or i + horizon + 1 > N:
            continue
        entry = open_arr[i + 1]
        if entry <= 0:
            continue

        wh = high[i + 1:i + horizon + 1]
        wl = low[i + 1:i + horizon + 1]
        mfe_up = max((np.max(wh) - entry) / entry * 10000, 0)
        mfe_down = max((entry - np.min(wl)) / entry * 10000, 0)

        mfe[ci, 0] = mfe_up
        mfe[ci, 1] = mfe_down

        # Direction normalize: support=bounce is UP, resistance=bounce is DOWN
        is_support = sr_sta[ci, -1] > 0.5  # level_type_binary is last static feature

        if is_support:
            fav, adv = mfe_up, mfe_down
        else:
            fav, adv = mfe_down, mfe_up

        total = fav + adv
        if total == 0:
            continue

        # First-hit direction
        thu = entry * (1 + threshold / 10000)
        thd = entry * (1 - threshold / 10000)
        fu, fd = 0, 0
        for j in range(1, horizon + 1):
            if i + j >= N: break
            if fu == 0 and high[i + j] >= thu: fu = j
            if fd == 0 and low[i + j] <= thd: fd = j

        if is_support:
            # Bounce = UP first, Break = DOWN first
            if fu > 0 and (fd == 0 or fu < fd):
                labels[ci] = 0  # bounce
            elif fd > 0 and (fu == 0 or fd < fu):
                labels[ci] = 1  # break
        else:
            # Bounce = DOWN first, Break = UP first
            if fd > 0 and (fu == 0 or fd < fu):
                labels[ci] = 0  # bounce
            elif fu > 0 and (fd == 0 or fu < fd):
                labels[ci] = 1  # break

    return labels, mfe

lab_tr, mfe_tr = create_bounce_break_labels(bars_train, sr_sta_train)
lab_va, mfe_va = create_bounce_break_labels(bars_val, sr_sta_val)
lab_te, mfe_te = create_bounce_break_labels(bars_test, sr_sta_test)

# Keep only bounce (0) and break (1)
for name, lab in [("Train", lab_tr), ("Val", lab_va), ("Test", lab_te)]:
    bounce = (lab == 0).sum()
    brk = (lab == 1).sum()
    skip = (lab == -1).sum()
    total = bounce + brk
    print(f"  {name}: bounce={bounce} ({bounce/max(total,1)*100:.1f}%) break={brk} ({brk/max(total,1)*100:.1f}%) skip={skip}")

# Filter to valid only
def filter_valid(dyn, sta, lab, mfe_arr):
    valid = (lab == 0) | (lab == 1)
    return dyn[valid], sta[valid], lab[valid], mfe_arr[valid]

d_tr, s_tr, y_tr, m_tr = filter_valid(sr_dyn_train, sr_sta_train, lab_tr, mfe_tr)
d_va, s_va, y_va, m_va = filter_valid(sr_dyn_val, sr_sta_val, lab_va, mfe_va)
d_te, s_te, y_te, m_te = filter_valid(sr_dyn_test, sr_sta_test, lab_te, mfe_te)

print(f"\n  After filter - Train: {len(y_tr)} | Val: {len(y_va)} | Test: {len(y_te)}")

# === Normalize ===
print("Normalizing...")
d_mean, d_std = d_tr.mean(0), d_tr.std(0)
d_std[d_std == 0] = 1.0
d_tr = np.nan_to_num((d_tr - d_mean) / d_std, nan=0.0)
d_va = np.nan_to_num((d_va - d_mean) / d_std, nan=0.0)
d_te = np.nan_to_num((d_te - d_mean) / d_std, nan=0.0)

s_mean, s_std = s_tr.mean(0), s_tr.std(0)
s_std[s_std == 0] = 1.0
s_tr = np.nan_to_num((s_tr - s_mean) / s_std, nan=0.0)
s_va = np.nan_to_num((s_va - s_mean) / s_std, nan=0.0)
s_te = np.nan_to_num((s_te - s_mean) / s_std, nan=0.0)

m_mean, m_std = m_tr.mean(0), m_tr.std(0)
m_std[m_std == 0] = 1.0
m_tr_n = (m_tr - m_mean) / m_std
m_va_n = (m_va - m_mean) / m_std
m_te_n = (m_te - m_mean) / m_std

# Tensors
td = torch.FloatTensor(d_tr); vd = torch.FloatTensor(d_va); ed = torch.FloatTensor(d_te)
ts = torch.FloatTensor(s_tr); vs = torch.FloatTensor(s_va); es = torch.FloatTensor(s_te)
ty = torch.LongTensor(y_tr); vy = torch.LongTensor(y_va); ey = torch.LongTensor(y_te)
tm = torch.FloatTensor(m_tr_n); vm = torch.FloatTensor(m_va_n); em = torch.FloatTensor(m_te_n)


def train_and_test(model, use_mfe, name):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    if use_mfe:
        loader = DataLoader(TensorDataset(td, ts, ty, tm), batch_size=256, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(td, ts, ty), batch_size=256, shuffle=True)

    best_vl = float("inf")
    pat = 0
    best_st = None

    print(f"\n%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s" %
          ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_Bnce", "Va_Brk"))

    for epoch in range(200):
        model.train()
        tl, co, to = 0, 0, 0

        if use_mfe:
            for xd, xs, yb, mb in loader:
                opt.zero_grad()
                dp, mp = model(xd, xs)
                loss = 1.0 * ce(dp, yb) + 5.0 * mse(mp, mb)
                loss.backward()
                opt.step()
                tl += loss.item() * len(yb)
                co += (dp.argmax(1) == yb).sum().item()
                to += len(yb)
        else:
            for xd, xs, yb in loader:
                opt.zero_grad()
                dp = model(xd, xs)
                loss = ce(dp, yb)
                loss.backward()
                opt.step()
                tl += loss.item() * len(yb)
                co += (dp.argmax(1) == yb).sum().item()
                to += len(yb)

        model.eval()
        with torch.no_grad():
            if use_mfe:
                vdp, vmp = model(vd, vs)
                vl = (1.0 * ce(vdp, vy) + 5.0 * mse(vmp, vm)).item()
            else:
                vdp = model(vd, vs)
                vl = ce(vdp, vy).item()
            vc = vdp.argmax(1)
            va = (vc == vy).float().mean().item() * 100
            vba = (vc[vy == 0] == 0).float().mean().item() * 100 if (vy == 0).sum() > 0 else 0
            vka = (vc[vy == 1] == 1).float().mean().item() * 100 if (vy == 1).sum() > 0 else 0

        if epoch % 10 == 0 or epoch < 5:
            print("%-6d | %-8.1f %-8.4f | %-8.1f %-8.4f | %-8.1f %-8.1f" %
                  (epoch, co/to*100, tl/to, va, vl, vba, vka))

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
        if use_mfe:
            tdp, _ = model(ed, es)
        else:
            tdp = model(ed, es)
        tc = tdp.argmax(1)
        ta = (tc == ey).float().mean().item() * 100
        tba = (tc[ey == 0] == 0).float().mean().item() * 100
        tka = (tc[ey == 1] == 1).float().mean().item() * 100

        probs = torch.softmax(tdp, dim=1).max(dim=1).values
        for ct in [0.55, 0.60, 0.65]:
            m = probs >= ct
            if m.sum() > 0:
                ca = (tc[m] == ey[m]).float().mean().item() * 100
                print(f"  Confident (>{ct}): {ca:.1f}% ({m.sum()} bars, {m.sum()/len(ey)*100:.1f}%)")

    print(f"\n  TEST: {ta:.1f}% (Bounce={tba:.1f}%, Break={tka:.1f}%)")
    return ta


# === Stage 8a: Without MFE ===
class SRAdvisor8a(nn.Module):
    def __init__(self, dyn_size, sta_size):
        super().__init__()
        self.dyn_path = nn.Sequential(nn.Linear(dyn_size, 4), nn.ReLU())
        self.sta_path = nn.Sequential(nn.Linear(sta_size, 4), nn.ReLU())
        self.head = nn.Linear(8, 2)

    def forward(self, x_dyn, x_sta):
        d = self.dyn_path(x_dyn)
        s = self.sta_path(x_sta)
        combined = torch.cat([d, s], dim=1)
        return self.head(combined)

model_8a = SRAdvisor8a(dyn_size, sta_size)
acc_8a = train_and_test(model_8a, use_mfe=False, name="STAGE 8a: Without MFE")


# === Stage 8b: With MFE (connected) ===
class SRAdvisor8b(nn.Module):
    def __init__(self, dyn_size, sta_size):
        super().__init__()
        self.dyn_path = nn.Sequential(nn.Linear(dyn_size, 4), nn.ReLU())
        self.sta_path = nn.Sequential(nn.Linear(sta_size, 4), nn.ReLU())
        self.mfe_head = nn.Linear(8, 2)
        self.bounce_head = nn.Linear(10, 2)  # 8 + 2 mfe

    def forward(self, x_dyn, x_sta):
        d = self.dyn_path(x_dyn)
        s = self.sta_path(x_sta)
        combined = torch.cat([d, s], dim=1)
        mfe_pred = self.mfe_head(combined)
        bounce_input = torch.cat([combined, mfe_pred], dim=1)
        bounce_pred = self.bounce_head(bounce_input)
        return bounce_pred, mfe_pred

model_8b = SRAdvisor8b(dyn_size, sta_size)
acc_8b = train_and_test(model_8b, use_mfe=True, name="STAGE 8b: With MFE (connected)")


# === Comparison ===
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"  8a (no MFE):       {acc_8a:.1f}%")
print(f"  8b (with MFE):     {acc_8b:.1f}%")
print(f"  Previous S/R best: ~50%")
print(f"  Baseline:          50.0%")
print(f"  Target:            >52%")
