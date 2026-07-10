# NOTE: the feature formulas / model architecture below are a NECESSARY COPY
# (Colab cannot import this repo). Source of truth: src/engine/signals/feature_lib.py
# and src/training/train_attention.py — keep in sync MANUALLY when either changes.
"""
COLAB NOTEBOOK: LSTM+Attention evaluation with correct 0.40 SHORT threshold.
Paste this entire script into one Colab cell. Set runtime to T4 GPU.

Tests all 4 temperatures: 1.0, 0.5, 0.1, 0.05
Same setup as original L2_003_stage3_mlp.py:
  - 32 features: 4 diffs x 8 lookbacks as [8,4] sequence
  - Label: direction_h8 (binary LONG/SHORT)
  - Connected MFE heads
  - SHORT threshold: 0.40 (CORRECT — original used this)
  - Train: 2020-2023, Val: 2024-H1, Test: 2025
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# =============================================================================
# CONFIG (exact match to L2_003_stage3_mlp.py)
# =============================================================================
CACHE_PATH  = "/content/drive/MyDrive/L2-003/feature_cache.parquet"
LABELS_PATH = "/content/drive/MyDrive/L2-003/labels.parquet"

TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
VAL_START,   VAL_END   = "2024-01-01", "2024-06-30"
TEST_START,  TEST_END  = "2025-01-01", "2025-12-31"

LOOKBACKS = [1, 2, 3, 4, 5, 6, 7, 8]
MFE_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]

BATCH_SIZE = 2048
MAX_EPOCHS = 100
PATIENCE   = 10
LR         = 0.001
HIDDEN     = 128
DROPOUT    = 0.5

# CORRECT thresholds (matching original line 263)
CONF_LONG  = 0.60
CONF_SHORT = 0.40   # FIXED — was 0.35 in our local runs

TEMPERATURES = [1.0, 0.5, 0.1, 0.05]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =============================================================================
# LOAD + PREPARE DATA
# =============================================================================
print("\nLoading data...")
fc = pd.read_parquet(CACHE_PATH)
lb = pd.read_parquet(LABELS_PATH)
print(f"  feature_cache: {fc.shape}")
print(f"  labels: {lb.shape}")

close = fc["close"].values.astype(np.float64)
rsi7 = fc["rsi7"].values.astype(np.float64)
rp = fc["range_position"].values.astype(np.float64)
sma200 = fc["sma200_dist_pct"].values.astype(np.float64)

print("Computing diff features...")
diff_list = []
for n in LOOKBACKS:
    roc_d = np.zeros(len(close), dtype=np.float32)
    roc_d[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
    rsi_d = np.zeros(len(close), dtype=np.float32)
    rsi_d[n:] = (rsi7[n:] - rsi7[:-n]).astype(np.float32)
    rp_d = np.zeros(len(close), dtype=np.float32)
    rp_d[n:] = (rp[n:] - rp[:-n]).astype(np.float32)
    sma_d = np.zeros(len(close), dtype=np.float32)
    sma_d[n:] = (sma200[n:] - sma200[:-n]).astype(np.float32)
    diff_list.extend([roc_d, rsi_d, rp_d, sma_d])

diff_arr_raw = np.column_stack(diff_list).astype(np.float32)
F_INPUT = len(LOOKBACKS) * 4

common_idx = lb.index.intersection(fc.index)
lb = lb.loc[common_idx]
label_dates = np.array(common_idx)
label_pos = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)

train_fc_mask = (fc.index >= TRAIN_START) & (fc.index <= TRAIN_END)
diff_arr_raw = np.nan_to_num(diff_arr_raw, nan=0.0, posinf=0.0, neginf=0.0)
mean = diff_arr_raw[train_fc_mask].mean(axis=0)
std = diff_arr_raw[train_fc_mask].std(axis=0)
std[std < 1e-8] = 1.0
diff_arr = (diff_arr_raw - mean) / std
diff_aligned = diff_arr[label_pos]

y_mfe_up = lb[[f"mfe_up_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0
y_mfe_down = lb[[f"mfe_down_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0

direction_h8 = lb["direction_h8"].values
valid_mask = (direction_h8 == 0) | (direction_h8 == 1)
y_dir = np.zeros(len(direction_h8), dtype=np.float32)
y_dir[direction_h8 == 0] = 1.0

train_mask = (label_dates >= np.datetime64(TRAIN_START)) & (label_dates <= np.datetime64(TRAIN_END)) & valid_mask
val_mask = (label_dates >= np.datetime64(VAL_START)) & (label_dates <= np.datetime64(VAL_END)) & valid_mask
test_mask = (label_dates >= np.datetime64(TEST_START)) & (label_dates <= np.datetime64(TEST_END)) & valid_mask

train_idx_all = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"  Train: {len(train_idx_all)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
print(f"  H8: LONG={(direction_h8==0).sum()} SHORT={(direction_h8==1).sum()} SKIP={(direction_h8==3).sum()}")

# =============================================================================
# DATASETS
# =============================================================================
class SeqDS(Dataset):
    def __init__(self, idx_array):
        self.idx = idx_array
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        seq = diff_aligned[n].reshape(len(LOOKBACKS), 4)
        return (torch.from_numpy(seq),
                torch.from_numpy(y_mfe_up[n]),
                torch.from_numpy(y_mfe_down[n]),
                torch.tensor(y_dir[n]))

# =============================================================================
# MODELS (exact from original)
# =============================================================================
class LSTMConnected(nn.Module):
    def __init__(self, input_size=4, hidden=128, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.h_mfe_up = nn.Linear(hidden * 2, 8)
        self.h_mfe_down = nn.Linear(hidden * 2, 8)
        self.h_dir = nn.Linear(hidden * 2 + 8 + 8, 1)
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        hc = torch.cat([h.squeeze(0), c.squeeze(0)], dim=1)
        hc = self.dropout(hc)
        p_mu = self.h_mfe_up(hc)
        p_md = self.h_mfe_down(hc)
        dir_input = torch.cat([hc, p_mu, p_md], dim=1)
        p_dir = self.h_dir(dir_input).squeeze(-1)
        return p_mu, p_md, p_dir

class LSTMAttention(nn.Module):
    def __init__(self, input_size=4, hidden=128, dropout=0.5, temperature=1.0):
        super().__init__()
        self.hidden = hidden
        self.temperature = temperature
        self.lstm = nn.LSTM(input_size, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_score = nn.Linear(hidden, 1)
        self.h_mfe_up = nn.Linear(hidden, 8)
        self.h_mfe_down = nn.Linear(hidden, 8)
        self.h_dir = nn.Linear(hidden + 8 + 8, 1)

    def forward(self, x):
        all_h, (h_final, c_final) = self.lstm(x)
        scores = self.attn_score(all_h).squeeze(-1)
        attn_weights = torch.softmax(scores / self.temperature, dim=1)
        attended = torch.bmm(attn_weights.unsqueeze(1), all_h).squeeze(1)
        attended = self.dropout(attended)
        p_mu = self.h_mfe_up(attended)
        p_md = self.h_mfe_down(attended)
        dir_input = torch.cat([attended, p_mu, p_md], dim=1)
        p_dir = self.h_dir(dir_input).squeeze(-1)
        return p_mu, p_md, p_dir

mse_fn = nn.MSELoss()
bce_fn = nn.BCEWithLogitsLoss()

# =============================================================================
# TRAIN FUNCTION
# =============================================================================
def train_model(model, train_loader, val_loader, name):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*70}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_vl = float("inf"); best_ep = 0; pat = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, mu, md, d in train_loader:
            x, mu, md, d = x.to(DEVICE), mu.to(DEVICE), md.to(DEVICE), d.to(DEVICE)
            p_mu, p_md, p_dir = model(x)
            loss = mse_fn(p_mu, mu) + mse_fn(p_md, md) + 5.0 * bce_fn(p_dir, d)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for x, mu, md, d in val_loader:
                x, mu, md, d = x.to(DEVICE), mu.to(DEVICE), md.to(DEVICE), d.to(DEVICE)
                p_mu, p_md, p_dir = model(x)
                vl += (mse_fn(p_mu, mu) + mse_fn(p_md, md) + 5.0 * bce_fn(p_dir, d)).item()
        vl /= len(val_loader)
        scheduler.step(vl)

        if vl < best_vl - 1e-5:
            best_vl = vl; best_ep = epoch; pat = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"  Best epoch: {best_ep}, val_loss: {best_vl:.4f}")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model

# =============================================================================
# EVAL FUNCTION (with full 26 metrics)
# =============================================================================
def eval_full(model, loader, y_labels, mfe_up_96, mfe_down_96, name):
    model.eval()
    all_dir = []
    all_y = []
    with torch.no_grad():
        for x, mu, md, d in loader:
            x = x.to(DEVICE)
            p_mu, p_md, p_dir = model(x)
            all_dir.append(p_dir.cpu().numpy())
            all_y.append(d.numpy())

    p_dir = np.concatenate(all_dir)
    y = np.concatenate(all_y)
    probs = 1 / (1 + np.exp(-p_dir))

    # Overall
    preds = (probs > 0.5).astype(int)
    acc = (preds == y).mean()

    # Confidence (CORRECT: 0.60 / 0.40)
    conf_long = probs >= CONF_LONG
    conf_short = probs <= CONF_SHORT
    conf = conf_long | conf_short
    n_conf = conf.sum()
    n_conf_long = conf_long.sum()
    n_conf_short = conf_short.sum()

    conf_acc = (np.where(probs[conf] > 0.5, 1, 0) == y[conf]).mean() if n_conf > 0 else 0
    conf_acc_long = (y[conf_long] == 1).mean() if n_conf_long > 0 else 0
    conf_acc_short = (y[conf_short] == 0).mean() if n_conf_short > 0 else 0
    ls_ratio = n_conf_long / max(n_conf_short, 1)

    # Per-class
    long_mask = y == 1
    short_mask = y == 0
    pred_long = preds == 1
    pred_short = preds == 0
    tp_l = (pred_long & long_mask).sum()
    fp_l = (pred_long & short_mask).sum()
    fn_l = (~pred_long & long_mask).sum()
    tp_s = (pred_short & short_mask).sum()
    fp_s = (pred_short & long_mask).sum()
    prec_l = tp_l / max(tp_l + fp_l, 1)
    rec_l = tp_l / max(tp_l + fn_l, 1)
    prec_s = tp_s / max(tp_s + fp_s, 1)
    rec_s = tp_s / max(tp_s + fp_s + (short_mask & pred_long).sum(), 1)
    f1_l = 2*prec_l*rec_l/max(prec_l+rec_l, 0.001)
    f1_s = 2*prec_s*rec_s/max(prec_s+rec_s, 0.001)

    # MFE/MAE on confident bars
    avg_mfe_long = float(mfe_up_96[conf_long].mean()) if n_conf_long > 0 else 0
    avg_mae_long = float(mfe_down_96[conf_long].mean()) if n_conf_long > 0 else 0
    avg_mfe_short = float(mfe_down_96[conf_short].mean()) if n_conf_short > 0 else 0
    avg_mae_short = float(mfe_up_96[conf_short].mean()) if n_conf_short > 0 else 0

    print(f"\n  {name}:")
    print(f"    Overall acc: {acc*100:.1f}%")
    print(f"    Conf acc: {conf_acc*100:.1f}% ({n_conf} bars: {n_conf_long}L + {n_conf_short}S)")
    print(f"    LONG conf acc: {conf_acc_long*100:.1f}% ({n_conf_long} bars)")
    print(f"    SHORT conf acc: {conf_acc_short*100:.1f}% ({n_conf_short} bars)")
    print(f"    L/S ratio: {ls_ratio:.1f}")
    print(f"    Prec L/S: {prec_l*100:.1f}% / {prec_s*100:.1f}%")
    print(f"    Rec L/S: {rec_l*100:.1f}% / {rec_s*100:.1f}%")
    print(f"    F1 L/S: {f1_l:.3f} / {f1_s:.3f}")
    print(f"    MFE L/S: {avg_mfe_long:.1f} / {avg_mfe_short:.1f} bps")
    print(f"    MAE L/S: {avg_mae_long:.1f} / {avg_mae_short:.1f} bps")

    return {
        "acc": acc, "conf_acc": conf_acc,
        "n_conf": n_conf, "n_conf_long": n_conf_long, "n_conf_short": n_conf_short,
        "conf_acc_long": conf_acc_long, "conf_acc_short": conf_acc_short,
        "ls_ratio": ls_ratio,
        "prec_long": prec_l, "prec_short": prec_s,
        "rec_long": rec_l, "rec_short": rec_s,
        "f1_long": f1_l, "f1_short": f1_s,
        "mfe_long": avg_mfe_long, "mae_long": avg_mae_long,
        "mfe_short": avg_mfe_short, "mae_short": avg_mae_short,
    }

# =============================================================================
# RUN
# =============================================================================
dl = lambda ds: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

seq_val = dl(SeqDS(val_idx))
seq_test = dl(SeqDS(test_idx))
seq_train_all = DataLoader(SeqDS(train_idx_all), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

mfe_up_96_test = lb["mfe_up_96"].values[test_idx]
mfe_down_96_test = lb["mfe_down_96"].values[test_idx]

results = {}

# Config A: LSTM baseline (no attention)
m = train_model(LSTMConnected().to(DEVICE), seq_train_all, seq_val, "A: LSTM baseline (no attention)")
results['A'] = eval_full(m, seq_test, y_dir[test_idx], mfe_up_96_test, mfe_down_96_test, "A: LSTM baseline TEST")

# Configs B-E: Attention with temperatures
for config_name, temp_val in [('B', 1.0), ('C', 0.5), ('D', 0.1), ('E', 0.05)]:
    name = f"{config_name}: Attention temp={temp_val}"
    m = train_model(LSTMAttention(temperature=temp_val).to(DEVICE), seq_train_all, seq_val, name)
    results[config_name] = eval_full(m, seq_test, y_dir[test_idx], mfe_up_96_test, mfe_down_96_test, f"{name} TEST")

# =============================================================================
# COMPARISON TABLE
# =============================================================================
print(f"\n\n{'='*100}")
print(f"FINAL COMPARISON (SHORT threshold = {CONF_SHORT}, GPU = {DEVICE})")
print(f"{'='*100}")
print(f"  {'Config':<30s} | {'Acc':>5s} {'Conf':>6s} {'N':>6s} {'L':>5s} {'S':>5s} | {'L acc':>6s} {'S acc':>6s} {'L/S':>5s} | {'MFE_L':>6s} {'MFE_S':>6s}")
print(f"  {'-'*30}-+-{'-'*5}-{'-'*6}-{'-'*6}-{'-'*5}-{'-'*5}-+-{'-'*6}-{'-'*6}-{'-'*5}-+-{'-'*6}-{'-'*6}")

for key, name in [('A', 'LSTM baseline'), ('B', 'Attn temp=1.0'), ('C', 'Attn temp=0.5'), ('D', 'Attn temp=0.1'), ('E', 'Attn temp=0.05')]:
    r = results[key]
    print(f"  {name:<30s} | {r['acc']*100:4.1f}% {r['conf_acc']*100:5.1f}% {r['n_conf']:5d} {r['n_conf_long']:5d} {r['n_conf_short']:5d} | {r['conf_acc_long']*100:5.1f}% {r['conf_acc_short']*100:5.1f}% {r['ls_ratio']:4.1f} | {r['mfe_long']:5.0f} {r['mfe_short']:5.0f}")

print(f"\n  Previous reported:")
print(f"  {'Attn temp=0.5 (prev)':<30s} | {'—':>5s} {'57.8%':>6s} {'1814':>6s}")
print(f"  {'Attn temp=0.05 (prev)':<30s} | {'—':>5s} {'58.8%':>6s} {'816':>6s}")

print(f"\n  Thresholds: LONG > {CONF_LONG}, SHORT < {CONF_SHORT}")
print(f"  Label: direction_h8 (H8)")
print(f"  Features: 4 diffs x 8 lookbacks = 32")
print(f"  Train: 2020-2023, Val: 2024-H1, Test: 2025")
