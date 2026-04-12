"""
COLAB NOTEBOOK V2: LSTM+Attention evaluation with backtest.
Paste this entire script into one Colab cell. Set runtime to T4 GPU.

Same as V1 but ADDS V1.4 backtest for each model.
SHORT threshold: 0.40 (matching original research)

Usage: paste into Colab cell, run.
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

CONF_LONG  = 0.60
CONF_SHORT = 0.40

# V1.4 exit rules
TS_LONG = 20
TS_SHORT = 30
TIME_EXIT = 10
TIGHTEN_BAR = 5
TIGHT_BPS = 8
FEE_BPS = 8.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =============================================================================
# LOAD DATA
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

# For backtest — need all 2025 bars (not just valid)
test_mask_all = (label_dates >= np.datetime64(TEST_START)) & (label_dates <= np.datetime64(TEST_END))
all_test_indices = np.where(test_mask_all)[0]

# OHLCV for backtest
ohlcv_open = fc["open"].values
ohlcv_high = fc["high"].values
ohlcv_low = fc["low"].values
label_to_fc = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)

# MFE for metrics
mfe_up_96_test = lb["mfe_up_96"].values[test_idx]
mfe_down_96_test = lb["mfe_down_96"].values[test_idx]

print(f"  Train: {len(train_idx_all)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# =============================================================================
# DATASETS + MODELS
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
        all_h, _ = self.lstm(x)
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
# TRAIN
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
    return model.to(DEVICE)

# =============================================================================
# BACKTEST (V1.4 exit rules)
# =============================================================================
def run_backtest(model, name):
    """Score all 2025 bars, find confident signals, simulate trades."""
    model.eval()
    all_probs = np.zeros(len(label_dates), dtype=np.float32)
    with torch.no_grad():
        for i in all_test_indices:
            seq = torch.from_numpy(diff_aligned[i].reshape(1, len(LOOKBACKS), 4)).to(DEVICE)
            _, _, p_dir = model(seq)
            all_probs[i] = torch.sigmoid(p_dir).cpu().item()

    long_entries = all_test_indices[all_probs[all_test_indices] > CONF_LONG]
    short_entries = all_test_indices[all_probs[all_test_indices] <= CONF_SHORT]

    trades = []
    last_exit_bar = -999

    for label_idx in sorted(np.concatenate([long_entries, short_entries])):
        fc_idx = label_to_fc[label_idx]
        if fc_idx + 1 >= len(ohlcv_open): continue
        if fc_idx <= last_exit_bar: continue

        direction = "LONG" if all_probs[label_idx] > CONF_LONG else "SHORT"
        entry_price = ohlcv_open[fc_idx + 1]
        ts = TS_LONG if direction == "LONG" else TS_SHORT
        best = entry_price
        exit_price = None
        exit_bar = TIME_EXIT

        for bar in range(1, TIME_EXIT + 1):
            bi = fc_idx + 1 + bar
            if bi >= len(ohlcv_high): break
            h, l = ohlcv_high[bi], ohlcv_low[bi]
            cur_ts = TIGHT_BPS if bar >= TIGHTEN_BAR else ts
            if direction == "LONG":
                best = max(best, h)
                trail = best * (1 - cur_ts / 10000)
                if l <= trail:
                    exit_price = trail; exit_bar = bar; break
            else:
                best = min(best, l)
                trail = best * (1 + cur_ts / 10000)
                if h >= trail:
                    exit_price = trail; exit_bar = bar; break

        if exit_price is None:
            ei = fc_idx + 1 + TIME_EXIT + 1
            if ei < len(ohlcv_open): exit_price = ohlcv_open[ei]
            else: continue

        if direction == "LONG":
            gross = (exit_price - entry_price) / entry_price * 10000
        else:
            gross = (entry_price - exit_price) / entry_price * 10000

        trades.append({"direction": direction, "net_bps": gross - FEE_BPS})
        last_exit_bar = fc_idx + 1 + exit_bar

    if not trades:
        print(f"  {name} BACKTEST: 0 trades")
        return {"n": 0, "win": 0, "bps": 0, "pf": 0, "l_n": 0, "l_bps": 0, "s_n": 0, "s_bps": 0}

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    w = (tdf["net_bps"] > 0).sum()
    bps = tdf["net_bps"].sum()
    gw = tdf.loc[tdf["net_bps"] > 0, "net_bps"].sum()
    gl = abs(tdf.loc[tdf["net_bps"] <= 0, "net_bps"].sum())
    pf = gw / max(gl, 0.01)
    lt = tdf[tdf["direction"] == "LONG"]
    st = tdf[tdf["direction"] == "SHORT"]

    print(f"  {name} BACKTEST: {n} trades, {w/n*100:.1f}% win, {bps:+.0f} bps, PF {pf:.2f}")
    print(f"    LONG:  {len(lt)} trades, {lt['net_bps'].sum():+.0f} bps")
    print(f"    SHORT: {len(st)} trades, {st['net_bps'].sum():+.0f} bps")

    return {
        "n": n, "win": round(w/n*100, 1), "bps": round(float(bps)), "pf": round(pf, 2),
        "l_n": len(lt), "l_bps": round(float(lt["net_bps"].sum())),
        "s_n": len(st), "s_bps": round(float(st["net_bps"].sum())),
    }

# =============================================================================
# EVAL (accuracy metrics)
# =============================================================================
def eval_metrics(model, loader, y_labels, mfe_up, mfe_down, name):
    model.eval()
    all_dir, all_y = [], []
    with torch.no_grad():
        for x, mu, md, d in loader:
            x = x.to(DEVICE)
            _, _, p_dir = model(x)
            all_dir.append(p_dir.cpu().numpy())
            all_y.append(d.numpy())
    p_dir = np.concatenate(all_dir)
    y = np.concatenate(all_y)
    probs = 1 / (1 + np.exp(-p_dir))

    acc = ((probs > 0.5).astype(int) == y).mean()
    cl = probs >= CONF_LONG
    cs = probs <= CONF_SHORT
    conf = cl | cs
    nc = conf.sum(); ncl = cl.sum(); ncs = cs.sum()
    ca = (np.where(probs[conf] > 0.5, 1, 0) == y[conf]).mean() if nc > 0 else 0
    cal = (y[cl] == 1).mean() if ncl > 0 else 0
    cas = (y[cs] == 0).mean() if ncs > 0 else 0
    lsr = ncl / max(ncs, 1)

    # MFE/MAE
    mfe_l = float(mfe_up[cl].mean()) if ncl > 0 else 0
    mae_l = float(mfe_down[cl].mean()) if ncl > 0 else 0
    mfe_s = float(mfe_down[cs].mean()) if ncs > 0 else 0
    mae_s = float(mfe_up[cs].mean()) if ncs > 0 else 0

    return {
        "acc": acc, "ca": ca, "nc": nc, "ncl": ncl, "ncs": ncs,
        "cal": cal, "cas": cas, "lsr": lsr,
        "mfe_l": mfe_l, "mae_l": mae_l, "mfe_s": mfe_s, "mae_s": mae_s,
    }

# =============================================================================
# RUN ALL CONFIGS
# =============================================================================
dl = lambda ds: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
seq_val = dl(SeqDS(val_idx))
seq_test = dl(SeqDS(test_idx))
seq_train = DataLoader(SeqDS(train_idx_all), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

results = {}

# Config A: LSTM baseline
m = train_model(LSTMConnected().to(DEVICE), seq_train, seq_val, "A: LSTM baseline")
r = eval_metrics(m, seq_test, y_dir[test_idx], mfe_up_96_test, mfe_down_96_test, "A")
bt = run_backtest(m, "A: LSTM baseline")
results['A'] = {**r, **{f"bt_{k}": v for k, v in bt.items()}}

# Configs B-E: Attention
for cfg, temp in [('B', 1.0), ('C', 0.5), ('D', 0.1), ('E', 0.05)]:
    name = f"{cfg}: Attn temp={temp}"
    m = train_model(LSTMAttention(temperature=temp).to(DEVICE), seq_train, seq_val, name)
    r = eval_metrics(m, seq_test, y_dir[test_idx], mfe_up_96_test, mfe_down_96_test, cfg)
    bt = run_backtest(m, name)
    results[cfg] = {**r, **{f"bt_{k}": v for k, v in bt.items()}}

# =============================================================================
# FINAL COMPARISON
# =============================================================================
print(f"\n\n{'='*120}")
print(f"FINAL RESULTS (threshold: LONG>{CONF_LONG}, SHORT<{CONF_SHORT}, device={DEVICE})")
print(f"{'='*120}")
print(f"  {'Config':<22s} | {'Acc':>5s} {'Conf':>6s} {'N':>5s} {'L':>5s} {'S':>5s} | {'L_acc':>6s} {'S_acc':>6s} {'L/S':>4s} | {'MFE_L':>5s} {'MFE_S':>5s} | {'Trades':>6s} {'Win%':>5s} {'Bps':>7s} {'PF':>5s} {'L_bps':>6s} {'S_bps':>6s}")
print(f"  {'-'*22}-+-{'-'*33}-+-{'-'*20}-+-{'-'*12}-+-{'-'*44}")

for key, name in [('A','LSTM baseline'),('B','Attn t=1.0'),('C','Attn t=0.5'),('D','Attn t=0.1'),('E','Attn t=0.05')]:
    r = results[key]
    print(f"  {name:<22s} | {r['acc']*100:4.1f}% {r['ca']*100:5.1f}% {r['nc']:5d} {r['ncl']:5d} {r['ncs']:5d} | {r['cal']*100:5.1f}% {r['cas']*100:5.1f}% {r['lsr']:3.1f} | {r['mfe_l']:5.0f} {r['mfe_s']:5.0f} | {r['bt_n']:6d} {r['bt_win']:4.1f}% {r['bt_bps']:+7.0f} {r['bt_pf']:5.2f} {r['bt_l_bps']:+6.0f} {r['bt_s_bps']:+6.0f}")

print(f"\n  Previous reported:")
print(f"  {'Attn t=0.5 (prev)':<22s} | {'—':>5s} {'57.8%':>6s} {'1814':>5s}")
print(f"  {'Attn t=0.05 (prev)':<22s} | {'—':>5s} {'58.8%':>6s} {'816':>5s}")
print(f"  {'MLP H96 (production)':<22s} | {'—':>5s} {'55.8%':>6s} {'735':>5s} {'—':>5s} {'—':>5s} | {'—':>6s} {'—':>6s} {'—':>4s} | {'—':>5s} {'—':>5s} | {'785':>6s} {'43.3%':>5s} {'+4385':>7s} {'1.52':>5s} {'+3925':>6s} {'+461':>6s}")
