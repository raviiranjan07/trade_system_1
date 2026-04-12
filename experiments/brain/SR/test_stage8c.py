"""Stage 8c: Enriched Static Memory (14 features)

8c-A: Without MFE head
8c-B: With MFE head (connected)

Run: PYTHONPATH=src python experiments/brain/SR/test_stage8c.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from brain.config import load_config
from brain.ingestion import load_ohlcv

print("=" * 60)
print("STAGE 8c: Enriched Static Memory (14 features)")
print("=" * 60)

# Load data
print("\nLoading data...")
train = np.load("data/features/sr_bounce_break/every_bar/train.npz")
val = np.load("data/features/sr_bounce_break/every_bar/val.npz")
test = np.load("data/features/sr_bounce_break/every_bar/test.npz")

# Get dynamic (last snapshot) and static
def get_split(data):
    dyn = data["X_dynamic"]
    if dyn.ndim == 3:
        dyn = dyn[:, -1, :]
    sta = data["X_static"]
    return dyn, sta

d_tr, s_tr = get_split(train)
d_va, s_va = get_split(val)
d_te, s_te = get_split(test)

dyn_size = d_tr.shape[1]
sta_size = s_tr.shape[1]
print(f"  Dynamic: {dyn_size}, Static: {sta_size}")
print(f"  Train: {len(d_tr)}, Val: {len(d_va)}, Test: {len(d_te)}")

# Load OHLCV for labels
base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
open_arr = df["open"].values
N = len(df)

# Create bounce/break labels (H25, direction-normalized)
print("\nCreating bounce/break labels...")
horizon = 25
threshold = 15

def create_labels(bars, sta):
    labels = np.full(len(bars), -1, dtype=np.int64)
    mfe_vals = np.zeros((len(bars), 2), dtype=np.float32)

    for ci in range(len(bars)):
        i = bars[ci]
        if i + 1 >= N or i + horizon + 1 > N: continue
        entry = open_arr[i + 1]
        if entry <= 0: continue

        wh = high[i + 1:i + horizon + 1]
        wl = low[i + 1:i + horizon + 1]
        mfe_up = max((np.max(wh) - entry) / entry * 10000, 0)
        mfe_down = max((entry - np.min(wl)) / entry * 10000, 0)
        mfe_vals[ci] = [mfe_up, mfe_down]

        is_support = sta[ci, 5] > 0.5  # level_type_binary at index 5
        thu = entry * (1 + threshold / 10000)
        thd = entry * (1 - threshold / 10000)
        fu, fd = 0, 0
        for j in range(1, horizon + 1):
            if i + j >= N: break
            if fu == 0 and high[i + j] >= thu: fu = j
            if fd == 0 and low[i + j] <= thd: fd = j

        if is_support:
            if fu > 0 and (fd == 0 or fu < fd): labels[ci] = 0
            elif fd > 0 and (fu == 0 or fd < fu): labels[ci] = 1
        else:
            if fd > 0 and (fu == 0 or fd < fu): labels[ci] = 0
            elif fu > 0 and (fd == 0 or fu < fd): labels[ci] = 1

    return labels, mfe_vals

bars_tr, bars_va, bars_te = train["bars"], val["bars"], test["bars"]
lab_tr, mfe_tr = create_labels(bars_tr, s_tr)
lab_va, mfe_va = create_labels(bars_va, s_va)
lab_te, mfe_te = create_labels(bars_te, s_te)

# Filter valid
def filt(d, s, lab, mfe):
    v = (lab == 0) | (lab == 1)
    return d[v], s[v], lab[v], mfe[v]

d_tr, s_tr, y_tr, m_tr = filt(d_tr, s_tr, lab_tr, mfe_tr)
d_va, s_va, y_va, m_va = filt(d_va, s_va, lab_va, mfe_va)
d_te, s_te, y_te, m_te = filt(d_te, s_te, lab_te, mfe_te)

for name, y in [("Train", y_tr), ("Val", y_va), ("Test", y_te)]:
    print(f"  {name}: {len(y)} (bounce={( y==0).sum()} break={(y==1).sum()})")

# Normalize
print("Normalizing...")
dm, ds = d_tr.mean(0), d_tr.std(0); ds[ds == 0] = 1.0
d_tr = np.nan_to_num((d_tr - dm) / ds); d_va = np.nan_to_num((d_va - dm) / ds); d_te = np.nan_to_num((d_te - dm) / ds)

sm, ss = s_tr.mean(0), s_tr.std(0); ss[ss == 0] = 1.0
s_tr = np.nan_to_num((s_tr - sm) / ss); s_va = np.nan_to_num((s_va - sm) / ss); s_te = np.nan_to_num((s_te - sm) / ss)

mm, ms = m_tr.mean(0), m_tr.std(0); ms[ms == 0] = 1.0
m_tr_n = (m_tr - mm) / ms; m_va_n = (m_va - mm) / ms; m_te_n = (m_te - mm) / ms

# Tensors
td = torch.FloatTensor(d_tr); vd = torch.FloatTensor(d_va); ed = torch.FloatTensor(d_te)
ts = torch.FloatTensor(s_tr); vs = torch.FloatTensor(s_va); es = torch.FloatTensor(s_te)
ty = torch.LongTensor(y_tr); vy = torch.LongTensor(y_va); ey = torch.LongTensor(y_te)
tm = torch.FloatTensor(m_tr_n); vm = torch.FloatTensor(m_va_n); em = torch.FloatTensor(m_te_n)


def train_test(model, use_mfe, name):
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

    best_vl = float("inf"); pat = 0; best_st = None

    print(f"\n%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s" %
          ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_Bnce", "Va_Brk"))

    for epoch in range(200):
        model.train()
        tl, co, to = 0, 0, 0
        if use_mfe:
            for xd, xs, yb, mb in loader:
                opt.zero_grad()
                dp, mp = model(xd, xs)
                loss = ce(dp, yb) + 5.0 * mse(mp, mb)
                loss.backward(); opt.step()
                tl += loss.item() * len(yb); co += (dp.argmax(1) == yb).sum().item(); to += len(yb)
        else:
            for xd, xs, yb in loader:
                opt.zero_grad()
                dp = model(xd, xs)
                loss = ce(dp, yb)
                loss.backward(); opt.step()
                tl += loss.item() * len(yb); co += (dp.argmax(1) == yb).sum().item(); to += len(yb)

        model.eval()
        with torch.no_grad():
            if use_mfe:
                vdp, _ = model(vd, vs)
                vl = (ce(vdp, vy) + 5.0 * mse(_, vm)).item()
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
            best_vl = vl; pat = 0
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


# === 8c-A: Without MFE head ===
class Advisor8cA(nn.Module):
    def __init__(self, dyn_size, sta_size):
        super().__init__()
        self.dyn_path = nn.Sequential(nn.Linear(dyn_size, 4), nn.ReLU())
        self.sta_path = nn.Sequential(nn.Linear(sta_size, 6), nn.ReLU())
        self.head = nn.Linear(10, 2)
    def forward(self, xd, xs):
        d = self.dyn_path(xd)
        s = self.sta_path(xs)
        return self.head(torch.cat([d, s], dim=1))

acc_a = train_test(Advisor8cA(dyn_size, sta_size), use_mfe=False, name="8c-A: No MFE head (14 static)")


# === 8c-B: With MFE head (connected) ===
class Advisor8cB(nn.Module):
    def __init__(self, dyn_size, sta_size):
        super().__init__()
        self.dyn_path = nn.Sequential(nn.Linear(dyn_size, 4), nn.ReLU())
        self.sta_path = nn.Sequential(nn.Linear(sta_size, 6), nn.ReLU())
        self.mfe_head = nn.Linear(10, 2)
        self.bounce_head = nn.Linear(12, 2)
    def forward(self, xd, xs):
        d = self.dyn_path(xd)
        s = self.sta_path(xs)
        combined = torch.cat([d, s], dim=1)
        mfe_pred = self.mfe_head(combined)
        bounce_input = torch.cat([combined, mfe_pred], dim=1)
        return self.bounce_head(bounce_input), mfe_pred

acc_b = train_test(Advisor8cB(dyn_size, sta_size), use_mfe=True, name="8c-B: With MFE head (14 static)")


# === Comparison ===
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"  8c-A (14 static, no MFE):     {acc_a:.1f}%")
print(f"  8c-B (14 static, with MFE):   {acc_b:.1f}%")
print(f"  8a (6 static, no MFE):         51.7%")
print(f"  8b (6 static, with MFE):       51.7%")
print(f"  Baseline:                      50.0%")
print(f"  Target:                        >52%")
