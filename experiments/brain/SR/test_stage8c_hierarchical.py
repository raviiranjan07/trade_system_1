"""Stage 8c: Hierarchical S/R Advisor with Gated Fusion

Step 1: Zone position (2 raw)
Step 2: History branch (8->4)
Step 3: Recent branch (6->4)
Gate: sigmoid decides trust history vs recent
Step 4: Approach context (15->4)
Step 5: Decision (4->2)

Run: PYTHONPATH=src python experiments/brain/SR/test_stage8c_hierarchical.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from brain.config import load_config
from brain.ingestion import load_ohlcv

print("=" * 60)
print("STAGE 8c: Hierarchical + Gated Fusion")
print("=" * 60)

# Load data
print("\nLoading data...")
train = np.load("data/features/sr_bounce_break/every_bar/train.npz")
val = np.load("data/features/sr_bounce_break/every_bar/val.npz")
test = np.load("data/features/sr_bounce_break/every_bar/test.npz")

def get_split(data):
    dyn = data["X_dynamic"]
    if dyn.ndim == 3:
        dyn = dyn[:, -1, :]
    sta = data["X_static"]
    return dyn, sta

d_tr, s_tr = get_split(train)
d_va, s_va = get_split(val)
d_te, s_te = get_split(test)

print(f"  Dynamic: {d_tr.shape[1]}, Static: {s_tr.shape[1]}")

# Static feature indices (from sr_dataset.py static_names order):
# 0: bounce_ratio, 1: recent_bounce_ratio, 2: pressure, 3: bars_since_touch,
# 4: touch_count_scaled, 5: level_type_binary,
# 6: avg_bounce_mfe, 7: avg_break_mfe, 8: max_bounce_mfe, 9: max_break_mfe,
# 10: last_outcome, 11: bounce_streak, 12: bounce_mfe_trend, 13: chop_ratio

# Load OHLCV for labels + zone_width for MFE normalization
base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
open_arr = df["open"].values
N = len(df)

# Create labels
print("Creating bounce/break labels (H25)...")
horizon = 25; threshold = 15

def create_labels(bars, sta):
    labels = np.full(len(bars), -1, dtype=np.int64)
    for ci in range(len(bars)):
        i = bars[ci]
        if i + 1 >= N or i + horizon + 1 > N: continue
        entry = open_arr[i + 1]
        if entry <= 0: continue
        thu = entry * (1 + threshold / 10000)
        thd = entry * (1 - threshold / 10000)
        fu, fd = 0, 0
        for j in range(1, horizon + 1):
            if i + j >= N: break
            if fu == 0 and high[i + j] >= thu: fu = j
            if fd == 0 and low[i + j] <= thd: fd = j
        is_sup = sta[ci, 5] > 0.5
        if is_sup:
            if fu > 0 and (fd == 0 or fu < fd): labels[ci] = 0
            elif fd > 0 and (fu == 0 or fd < fu): labels[ci] = 1
        else:
            if fd > 0 and (fu == 0 or fd < fu): labels[ci] = 0
            elif fu > 0 and (fd == 0 or fu < fd): labels[ci] = 1
    return labels

lab_tr = create_labels(train["bars"], s_tr)
lab_va = create_labels(val["bars"], s_va)
lab_te = create_labels(test["bars"], s_te)

def filt(d, s, lab):
    v = (lab == 0) | (lab == 1)
    return d[v], s[v], lab[v]

d_tr, s_tr, y_tr = filt(d_tr, s_tr, lab_tr)
d_va, s_va, y_va = filt(d_va, s_va, lab_va)
d_te, s_te, y_te = filt(d_te, s_te, lab_te)

for name, y in [("Train", y_tr), ("Val", y_va), ("Test", y_te)]:
    print(f"  {name}: {len(y)} (bounce={(y==0).sum()} break={(y==1).sum()})")

# Normalize MFE features by zone_width (dynamic feature index 5 = zone_width in log)
# zone_width is in log bps, MFE is in raw bps. Convert zone_width back: exp(zw) - 1
print("\nNormalizing MFE features by zone_width...")
for d_arr, s_arr in [(d_tr, s_tr), (d_va, s_va), (d_te, s_te)]:
    zw_bps = np.expm1(d_arr[:, 5])  # convert log back to bps
    zw_bps = np.maximum(zw_bps, 1.0)  # avoid division by zero
    for mfe_idx in [6, 7, 8, 9]:  # avg_bounce_mfe, avg_break_mfe, max_bounce_mfe, max_break_mfe
        s_arr[:, mfe_idx] = s_arr[:, mfe_idx] / zw_bps

# Split features into groups
def split_features(dyn, sta):
    # Step 1: zone position
    zone_pos = np.column_stack([sta[:, 5], dyn[:, 0]])  # level_type_binary, dist_to_zone_pct
    # (2 features)

    # Step 2: history
    history = np.column_stack([
        sta[:, 0],   # bounce_ratio
        sta[:, 4],   # touch_count_scaled
        sta[:, 6],   # avg_bounce_mfe_pct
        sta[:, 7],   # avg_break_mfe_pct
        sta[:, 8],   # max_bounce_mfe_pct
        sta[:, 9],   # max_break_mfe_pct
    ])  # (6 features)

    # Step 3: recent
    recent = np.column_stack([
        sta[:, 1],   # recent_bounce_ratio
        sta[:, 2],   # pressure
        sta[:, 10],  # last_outcome
        sta[:, 11],  # bounce_streak
        sta[:, 12],  # bounce_mfe_trend
        sta[:, 13],  # chop_ratio
    ])  # (6 features)

    # Step 4: approach (all dynamic features)
    approach = dyn  # (11 features)

    return zone_pos, history, recent, approach

zp_tr, h_tr, r_tr, a_tr = split_features(d_tr, s_tr)
zp_va, h_va, r_va, a_va = split_features(d_va, s_va)
zp_te, h_te, r_te, a_te = split_features(d_te, s_te)

# Z-score normalize each group (fit on train)
def normalize(tr, va, te):
    m, s = tr.mean(0), tr.std(0)
    s[s == 0] = 1.0
    return (np.nan_to_num((tr - m) / s),
            np.nan_to_num((va - m) / s),
            np.nan_to_num((te - m) / s))

zp_tr, zp_va, zp_te = normalize(zp_tr, zp_va, zp_te)
h_tr, h_va, h_te = normalize(h_tr, h_va, h_te)
r_tr, r_va, r_te = normalize(r_tr, r_va, r_te)
a_tr, a_va, a_te = normalize(a_tr, a_va, a_te)

# Also normalize bounce_mfe_trend separately (huge range)
# Already done in z-score above

print(f"\nFeature groups: zone_pos={zp_tr.shape[1]}, history={h_tr.shape[1]}, recent={r_tr.shape[1]}, approach={a_tr.shape[1]}")

# Tensors
tzp = torch.FloatTensor(zp_tr); vzp = torch.FloatTensor(zp_va); ezp = torch.FloatTensor(zp_te)
th = torch.FloatTensor(h_tr); vh = torch.FloatTensor(h_va); eh = torch.FloatTensor(h_te)
tr_ = torch.FloatTensor(r_tr); vr = torch.FloatTensor(r_va); er = torch.FloatTensor(r_te)
ta = torch.FloatTensor(a_tr); va_t = torch.FloatTensor(a_va); ea = torch.FloatTensor(a_te)
ty = torch.LongTensor(y_tr); vy = torch.LongTensor(y_va); ey = torch.LongTensor(y_te)

# Model
class HierarchicalGatedAdvisor(nn.Module):
    def __init__(self):
        super().__init__()
        # Step 2: History branch (zone_pos 2 + history 6 = 8 -> 4)
        self.history_branch = nn.Sequential(nn.Linear(8, 4), nn.ReLU())

        # Step 3: Recent branch (6 -> 4)
        self.recent_branch = nn.Sequential(nn.Linear(6, 4), nn.ReLU())

        # Gate (8 -> 4 sigmoid)
        self.gate = nn.Sequential(nn.Linear(8, 4), nn.Sigmoid())

        # Step 4: Approach (zone_state 4 + approach 11 = 15 -> 4)
        self.approach = nn.Sequential(nn.Linear(15, 4), nn.ReLU())

        # Step 5: Decision (4 -> 2)
        self.decision = nn.Linear(4, 2)

    def forward(self, zone_pos, history_feat, recent_feat, approach_feat):
        # Step 2: history
        hist_input = torch.cat([zone_pos, history_feat], dim=1)  # 2+6=8
        history_signal = self.history_branch(hist_input)  # 4

        # Step 3: recent
        recent_signal = self.recent_branch(recent_feat)  # 4

        # Gate
        gate_input = torch.cat([history_signal, recent_signal], dim=1)  # 8
        gate = self.gate(gate_input)  # 4 (each 0-1)
        zone_state = gate * recent_signal + (1 - gate) * history_signal  # 4

        # Step 4: approach
        approach_input = torch.cat([zone_state, approach_feat], dim=1)  # 4+11=15
        approach_context = self.approach(approach_input)  # 4

        # Step 5: decision
        return self.decision(approach_context)  # 2

model = HierarchicalGatedAdvisor()
print(f"\nParams: {sum(p.numel() for p in model.parameters())}")

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
ce = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(tzp, th, tr_, ta, ty), batch_size=256, shuffle=True)

print(f"\n%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s" %
      ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_Bnce", "Va_Brk"))

best_vl = float("inf"); pat = 0; best_st = None

for epoch in range(200):
    model.train()
    tl, co, to = 0, 0, 0
    for xzp, xh, xr, xa, yb in loader:
        opt.zero_grad()
        pred = model(xzp, xh, xr, xa)
        loss = ce(pred, yb)
        loss.backward(); opt.step()
        tl += loss.item() * len(yb)
        co += (pred.argmax(1) == yb).sum().item()
        to += len(yb)

    model.eval()
    with torch.no_grad():
        vpred = model(vzp, vh, vr, va_t)
        vl = ce(vpred, vy).item()
        vc = vpred.argmax(1)
        vacc = (vc == vy).float().mean().item() * 100
        vba = (vc[vy == 0] == 0).float().mean().item() * 100 if (vy == 0).sum() > 0 else 0
        vka = (vc[vy == 1] == 1).float().mean().item() * 100 if (vy == 1).sum() > 0 else 0

    if epoch % 10 == 0 or epoch < 5:
        print("%-6d | %-8.1f %-8.4f | %-8.1f %-8.4f | %-8.1f %-8.1f" %
              (epoch, co/to*100, tl/to, vacc, vl, vba, vka))

    if vl < best_vl:
        best_vl = vl; pat = 0
        best_st = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        pat += 1
        if pat >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

# Test
model.load_state_dict(best_st)
model.eval()
with torch.no_grad():
    tpred = model(ezp, eh, er, ea)
    tc = tpred.argmax(1)
    tacc = (tc == ey).float().mean().item() * 100
    tba = (tc[ey == 0] == 0).float().mean().item() * 100
    tka = (tc[ey == 1] == 1).float().mean().item() * 100

    probs = torch.softmax(tpred, dim=1).max(dim=1).values
    print()
    for ct in [0.55, 0.60, 0.65]:
        m = probs >= ct
        if m.sum() > 0:
            ca = (tc[m] == ey[m]).float().mean().item() * 100
            print(f"  Confident (>{ct}): {ca:.1f}% ({m.sum()} bars, {m.sum()/len(ey)*100:.1f}%)")

# Gate analysis
with torch.no_grad():
    hist_input = torch.cat([ezp, eh], dim=1)
    hist_sig = model.history_branch(hist_input)
    rec_sig = model.recent_branch(er)
    gate_input = torch.cat([hist_sig, rec_sig], dim=1)
    gate_vals = model.gate(gate_input).numpy()
    print(f"\n  Gate values (mean): {gate_vals.mean(0)}")
    print(f"  Gate values (std):  {gate_vals.std(0)}")
    print(f"  Gate > 0.5 (trust recent): {(gate_vals.mean(1) > 0.5).mean()*100:.1f}%")

print(f"\n{'='*60}")
print("TEST RESULTS (Hierarchical Gated)")
print(f"{'='*60}")
print(f"  Overall: {tacc:.1f}%")
print(f"  Bounce: {tba:.1f}% ({(ey==0).sum()} samples)")
print(f"  Break: {tka:.1f}% ({(ey==1).sum()} samples)")
print(f"\nCOMPARISON:")
print(f"  8c Hierarchical Gated:  {tacc:.1f}%")
print(f"  8c-B (simple + MFE):    52.0%")
print(f"  8a (6 static):          51.7%")
print(f"  Baseline:               50.0%")
