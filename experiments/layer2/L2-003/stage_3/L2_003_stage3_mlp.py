"""L2-003 Stage 4B v2: Attention with temperature — sharper attention weights

Tests if lower temperature makes attention weights more varied (sharper focus).

Config A: LSTM h8 only (baseline)
Config B: LSTM attention temp=1.0 (default softmax)
Config C: LSTM attention temp=0.5 (sharper)
Config D: LSTM attention temp=0.1 (very sharp)
Config E: LSTM attention temp=0.05 (near hard attention)

4 diff features x 8 lookbacks = 32 inputs
Label: H8 direction + MFE at H1-H8
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
# CONFIG
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =============================================================================
# LOAD + PREPARE DATA
# =============================================================================
print("\nLoading data...")
fc = pd.read_parquet(CACHE_PATH)
lb = pd.read_parquet(LABELS_PATH)

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
mfe_diff_h8 = np.abs(y_mfe_up[:, 7] - y_mfe_down[:, 7]) * 100

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

print(f"\n  Curriculum phase sizes (train):")
for phase_name, threshold in PHASES:
    if threshold > 0:
        n = (mfe_diff_h8[train_idx_all] >= threshold).sum()
    else:
        n = len(train_idx_all)
    print(f"    {phase_name}: {n} bars ({n/len(train_idx_all)*100:.1f}%)")

print(f"\n  Val: {len(val_idx)} | Test: {len(test_idx)}")

# =============================================================================
# DATASETS
# =============================================================================
class FlatDS(Dataset):
    def __init__(self, idx_array):
        self.idx = idx_array
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        return (torch.from_numpy(diff_aligned[n]),
                torch.from_numpy(y_mfe_up[n]),
                torch.from_numpy(y_mfe_down[n]),
                torch.tensor(y_dir[n]))

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
# MODELS
# =============================================================================
class LSTMConnected(nn.Module):
    """Baseline: uses only final hidden state h8"""
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
    """LSTM + attention with temperature control"""
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
# TRAIN FUNCTIONS
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
                print(f"  Best epoch: {best_ep}")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model

# =============================================================================
# EVAL
# =============================================================================
def eval_model(model, loader, name):
    model.eval()
    all_mu, all_md, all_dir = [], [], []
    all_y_dir = []
    with torch.no_grad():
        for x, mu, md, d in loader:
            x = x.to(DEVICE)
            p_mu, p_md, p_dir = model(x)
            all_mu.append(p_mu.cpu().numpy())
            all_md.append(p_md.cpu().numpy())
            all_dir.append(p_dir.cpu().numpy())
            all_y_dir.append(d.numpy())

    p_mu = np.concatenate(all_mu)
    p_md = np.concatenate(all_md)
    p_dir = np.concatenate(all_dir)
    y_dir = np.concatenate(all_y_dir)

    probs = 1 / (1 + np.exp(-p_dir))
    acc = ((probs > 0.5).astype(int) == y_dir).mean() * 100
    conf = (probs >= 0.60) | (probs <= 0.40)
    n_conf = conf.sum()
    conf_acc = ((np.where(probs[conf] > 0.5, 1, 0)) == y_dir[conf]).mean() * 100 if n_conf > 0 else 0

    mfe_dir = np.where(p_mu[:, 7] > p_md[:, 7], 1, 0)
    mfe_dir_acc = (mfe_dir == y_dir).mean() * 100
    pred_gap_std = ((p_mu[:, 7] - p_md[:, 7]) * 100).std()

    print(f"  {name}: dir_acc={acc:.1f}% | conf={conf_acc:.1f}% ({n_conf}) | mfe_dir={mfe_dir_acc:.1f}% | gap_std={pred_gap_std:.1f}")

    return {'acc': acc, 'conf_acc': conf_acc, 'n_conf': n_conf,
            'mfe_dir': mfe_dir_acc, 'gap_std': pred_gap_std}

# =============================================================================
# RUN
# =============================================================================
dl_v = lambda ds: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

flat_val = dl_v(FlatDS(val_idx))
flat_test = dl_v(FlatDS(test_idx))
seq_val = dl_v(SeqDS(val_idx))
seq_test = dl_v(SeqDS(test_idx))
flat_train_all = DataLoader(FlatDS(train_idx_all), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
seq_train_all = DataLoader(SeqDS(train_idx_all), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

results = {}
configs = []

# A: LSTM h8 only (baseline)
m = train_model(LSTMConnected().to(DEVICE), seq_train_all, seq_val, "A: LSTM h8 only (baseline)")
print("  Results:")
results['A'] = {'train': eval_model(m, seq_train_all, "Train"), 'test': eval_model(m, seq_test, "Test")}
configs.append(('A', 'LSTM h8 only (baseline)', None))

# B-E: LSTM + attention with different temperatures
for temp_name, temp_val in [('B', 1.0), ('C', 0.5), ('D', 0.1), ('E', 0.05)]:
    name = f"{temp_name}: LSTM attention temp={temp_val}"
    m = train_model(LSTMAttention(temperature=temp_val).to(DEVICE), seq_train_all, seq_val, name)
    print("  Results:")
    results[temp_name] = {'train': eval_model(m, seq_train_all, "Train"), 'test': eval_model(m, seq_test, "Test")}
    configs.append((temp_name, f"LSTM attn temp={temp_val}", m))

# =============================================================================
# COMPARISON
# =============================================================================
print(f"\n\n{'='*70}")
print("FINAL COMPARISON - Attention Temperature")
print(f"{'='*70}")

print(f"\n  {'Config':<40s} | {'Dir conf':>8s} {'n_conf':>6s} {'MFE dir':>7s} {'Gap std':>7s}")
print(f"  {'-'*40}-+-{'-'*8}-{'-'*6}-{'-'*7}-{'-'*7}")
for key, name, _ in configs:
    r = results[key]['test']
    print(f"  {name:<40s} | {r['conf_acc']:7.1f}% {r['n_conf']:6d} {r['mfe_dir']:6.1f}% {r['gap_std']:6.1f}")

print(f"\n  Overfitting check:")
for key, name, _ in configs:
    tr = results[key]['train']
    te = results[key]['test']
    print(f"  {name:<40s}: train={tr['conf_acc']:.1f}% test={te['conf_acc']:.1f}% gap={tr['conf_acc']-te['conf_acc']:.1f}%")

# =============================================================================
# ATTENTION DIAGNOSTICS — what did attention learn?
# =============================================================================
print(f"\n\n{'='*70}")
print("ATTENTION DIAGNOSTICS")
print(f"{'='*70}")

# Use best attention model (highest test conf_acc among B-E)
best_key = max(['B', 'C', 'D', 'E'], key=lambda k: results[k]['test']['conf_acc'])
best_temp_model = [m for k, n, m in configs if k == best_key][0]
print(f"\n  Best attention config: {best_key} (conf_acc={results[best_key]['test']['conf_acc']:.1f}%)")
m_attn = best_temp_model
m_attn.eval()

all_weights = []
all_preds = []
all_actuals = []

seq_test_diag = dl_v(SeqDS(test_idx))
with torch.no_grad():
    for x, mu, md, d in seq_test_diag:
        x = x.to(DEVICE)
        # Get attention weights
        all_h, (h_final, c_final) = m_attn.lstm(x)
        scores = m_attn.attn_score(all_h).squeeze(-1)
        attn_w = torch.softmax(scores, dim=1)
        all_weights.append(attn_w.cpu().numpy())

        # Get predictions
        p_mu, p_md, p_dir = m_attn(x)
        probs = torch.sigmoid(p_dir)
        all_preds.append(probs.cpu().numpy())
        all_actuals.append(d.numpy())

weights = np.concatenate(all_weights)  # (N_test, 8)
preds = np.concatenate(all_preds)
actuals = np.concatenate(all_actuals)

# 1. Average attention weights per step
print(f"\n  Average attention weights per step:")
step_names = [f"{n}-bar" for n in LOOKBACKS]
avg_w = weights.mean(axis=0)
for i, name in enumerate(step_names):
    bar = "#" * int(avg_w[i] * 100)
    print(f"    Step {i+1} ({name:>5s}): {avg_w[i]:.3f} ({avg_w[i]*100:.1f}%) {bar}")

uniform = 1.0 / len(LOOKBACKS)
print(f"\n  Uniform would be: {uniform:.3f} ({uniform*100:.1f}%)")
max_dev = max(abs(avg_w - uniform))
print(f"  Max deviation from uniform: {max_dev:.3f}")
if max_dev < 0.02:
    print(f"  --> Attention weights are nearly UNIFORM -- not being used effectively")
elif max_dev < 0.05:
    print(f"  --> Attention weights show SOME variation")
else:
    print(f"  --> Attention weights show STRONG variation -- being used!")

# 2. Do weights vary between bars?
print(f"\n  Attention weight std per step (across bars):")
for i, name in enumerate(step_names):
    print(f"    Step {i+1} ({name:>5s}): std={weights[:, i].std():.4f}")
print(f"  If all stds are near 0: every bar gets same weights (not useful)")
print(f"  If stds are high: different bars get different attention (useful)")

# 3. Confident LONG vs SHORT: different attention?
conf_long = preds > 0.60
conf_short = preds < 0.40

if conf_long.sum() > 50 and conf_short.sum() > 50:
    print(f"\n  Attention weights: confident LONG vs SHORT")
    print(f"    Step     LONG_mean  SHORT_mean  diff")
    for i, name in enumerate(step_names):
        lw = weights[conf_long, i].mean()
        sw = weights[conf_short, i].mean()
        diff = abs(lw - sw)
        marker = " ***" if diff > 0.02 else ""
        print(f"    {name:>5s}:   {lw:.3f}      {sw:.3f}      {diff:.3f}{marker}")
else:
    print(f"\n  Not enough confident bars for LONG/SHORT attention comparison")
