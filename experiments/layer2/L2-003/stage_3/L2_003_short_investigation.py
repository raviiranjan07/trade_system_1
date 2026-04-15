"""L2-003: Investigate why ML model's SHORT predictions fail

Run: PYTHONPATH=src python experiments/layer2/L2-003/L2_003_short_investigation.py
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.WARNING)

BASE = Path("experiments/layer2/L2-003")

# =============================================================================
# Train model (same as backtest)
# =============================================================================
print("Training model...")
fc = pd.read_parquet(BASE / "feature_cache.parquet")
lb = pd.read_parquet(BASE / "labels.parquet")

close = fc["close"].values.astype(np.float64)
for n in range(1, 9):
    roc = np.zeros(len(close), dtype=np.float32)
    roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
    fc[f"roc{n}"] = roc

feat_cols = [f"roc{n}" for n in range(1, 9)] + ["range_position", "rsi7"]
F = len(feat_cols)

common_idx = lb.index.intersection(fc.index)
lb = lb.loc[common_idx]
fc_aligned = fc.loc[common_idx]
feat_arr_raw = fc_aligned[feat_cols].values.astype(np.float32)
label_dates = np.array(common_idx)

train_fc_mask = (pd.DatetimeIndex(common_idx) >= "2020-01-01") & (pd.DatetimeIndex(common_idx) <= "2023-12-31")
scaler_mean = feat_arr_raw[train_fc_mask].mean(axis=0)
scaler_std = feat_arr_raw[train_fc_mask].std(axis=0)
scaler_std[scaler_std < 1e-8] = 1.0
feat_arr = (feat_arr_raw - scaler_mean) / scaler_std
feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

direction = lb["direction"].values
valid_mask = (direction == 0) | (direction == 1)
y_binary = np.zeros(len(direction), dtype=np.float32)
y_binary[direction == 0] = 1.0

train_mask = (label_dates >= np.datetime64("2020-01-01")) & (label_dates <= np.datetime64("2023-12-31")) & valid_mask
val_mask = (label_dates >= np.datetime64("2024-01-01")) & (label_dates <= np.datetime64("2024-06-30")) & valid_mask

class SimpleDS(Dataset):
    def __init__(self, idx):
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        return torch.from_numpy(feat_arr[n]), torch.tensor(y_binary[n])

class MLPBinaryDir(nn.Module):
    def __init__(self, input_size, hidden=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)

model = MLPBinaryDir(input_size=F)
bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
train_loader = DataLoader(SimpleDS(np.where(train_mask)[0]), batch_size=512, shuffle=True)
val_loader = DataLoader(SimpleDS(np.where(val_mask)[0]), batch_size=512, shuffle=False)

best_val_loss = float("inf")
patience_ctr = 0
for epoch in range(1, 101):
    model.train()
    for batch in train_loader:
        loss = bce(model(batch[0]), batch[1])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    vl = 0.0
    with torch.no_grad():
        for batch in val_loader:
            vl += bce(model(batch[0]), batch[1]).item()
    vl /= len(val_loader)
    scheduler.step(vl)
    if vl < best_val_loss - 1e-5:
        best_val_loss = vl; patience_ctr = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= 10: break

model.load_state_dict(best_state)
model.eval()
print(f"  Model ready (val_loss: {best_val_loss:.4f})")

# =============================================================================
# Score 2025 bars
# =============================================================================
test_mask = (label_dates >= np.datetime64("2025-01-01")) & (label_dates <= np.datetime64("2025-12-31"))
test_indices = np.where(test_mask)[0]

all_probs = np.zeros(len(label_dates))
with torch.no_grad():
    for i in test_indices:
        x = torch.from_numpy(feat_arr[i]).unsqueeze(0)
        all_probs[i] = 1 / (1 + np.exp(-model(x).item()))

# =============================================================================
# CHECK 1: Confidence distribution for LONG vs SHORT signals
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 1: Confidence distribution — LONG vs SHORT signals")
print("="*70)

long_probs = all_probs[test_indices][all_probs[test_indices] > 0.60]
short_probs = all_probs[test_indices][all_probs[test_indices] < 0.40]

print(f"\n  LONG signals (prob > 0.60): {len(long_probs)}")
print(f"    mean prob: {long_probs.mean():.4f}")
print(f"    0.60-0.65: {((long_probs >= 0.60) & (long_probs < 0.65)).sum()}")
print(f"    0.65-0.70: {((long_probs >= 0.65) & (long_probs < 0.70)).sum()}")
print(f"    0.70+:     {(long_probs >= 0.70).sum()}")

print(f"\n  SHORT signals (prob < 0.40): {len(short_probs)}")
short_conf = 1 - short_probs  # convert to confidence
print(f"    mean confidence: {short_conf.mean():.4f}")
print(f"    0.60-0.65: {((short_conf >= 0.60) & (short_conf < 0.65)).sum()}")
print(f"    0.65-0.70: {((short_conf >= 0.65) & (short_conf < 0.70)).sum()}")
print(f"    0.70+:     {(short_conf >= 0.70).sum()}")

# =============================================================================
# CHECK 2: Direction accuracy — LONG vs SHORT on 2025
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 2: Direction accuracy — LONG vs SHORT (2025)")
print("="*70)

test_valid = test_mask & valid_mask
test_dir = direction[test_valid]
test_probs_valid = all_probs[test_valid]

# LONG predictions accuracy
long_mask = test_probs_valid > 0.60
if long_mask.sum() > 0:
    long_correct = (test_dir[long_mask] == 0).mean() * 100  # 0=LONG
    print(f"\n  LONG predictions (prob>0.60): {long_mask.sum()} bars, {long_correct:.1f}% correct")

# SHORT predictions accuracy
short_mask = test_probs_valid < 0.40
if short_mask.sum() > 0:
    short_correct = (test_dir[short_mask] == 1).mean() * 100  # 1=SHORT
    print(f"  SHORT predictions (prob<0.40): {short_mask.sum()} bars, {short_correct:.1f}% correct")

# By confidence level
print(f"\n  LONG accuracy by confidence:")
for lo, hi in [(0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]:
    m = (test_probs_valid >= lo) & (test_probs_valid < hi)
    if m.sum() > 0:
        acc = (test_dir[m] == 0).mean() * 100
        print(f"    [{lo:.2f}-{hi:.2f}): {m.sum()} bars, {acc:.1f}% correct")

print(f"\n  SHORT accuracy by confidence:")
for lo, hi in [(0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]:
    m = (test_probs_valid <= (1-lo)) & (test_probs_valid > (1-hi))
    if m.sum() > 0:
        acc = (test_dir[m] == 1).mean() * 100
        print(f"    [{lo:.2f}-{hi:.2f}): {m.sum()} bars, {acc:.1f}% correct")

# =============================================================================
# CHECK 3: Feature values at SHORT signals — what market conditions?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 3: Feature values at SHORT signals vs LONG signals")
print("="*70)

# Raw (unnormalized) features
raw_feats = fc_aligned[feat_cols].values.astype(np.float32)
test_raw = raw_feats[test_indices]
test_probs_all = all_probs[test_indices]

long_sig = test_probs_all > 0.60
short_sig = test_probs_all < 0.40

print(f"\n  {'Feature':<20s} | {'LONG signal mean':>16s} {'SHORT signal mean':>17s} {'All bars mean':>14s}")
print(f"  {'-'*20}-+-{'-'*16}-{'-'*17}-{'-'*14}")

for f_idx, fname in enumerate(feat_cols):
    long_mean = test_raw[long_sig, f_idx].mean() if long_sig.sum() > 0 else 0
    short_mean = test_raw[short_sig, f_idx].mean() if short_sig.sum() > 0 else 0
    all_mean = test_raw[:, f_idx].mean()
    print(f"  {fname:<20s} | {long_mean:16.2f} {short_mean:17.2f} {all_mean:14.2f}")

# =============================================================================
# CHECK 4: What is the market doing when SHORT signals fire?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 4: Market context at SHORT signals")
print("="*70)

# Get additional features for context
fc_test = fc_aligned.iloc[test_indices]

short_bars = fc_test.iloc[np.where(short_sig)[0]]
long_bars = fc_test.iloc[np.where(long_sig)[0]]

context_feats = ["atr_percentile", "ema_separation", "price_above_sma200", "volume_ratio"]
existing_feats = [f for f in context_feats if f in fc_test.columns]

print(f"\n  {'Feature':<25s} | {'SHORT signals':>14s} {'LONG signals':>14s} {'All 2025':>14s}")
print(f"  {'-'*25}-+-{'-'*14}-{'-'*14}-{'-'*14}")

for fname in existing_feats:
    s_mean = short_bars[fname].mean()
    l_mean = long_bars[fname].mean()
    a_mean = fc_test[fname].mean()
    print(f"  {fname:<25s} | {s_mean:14.2f} {l_mean:14.2f} {a_mean:14.2f}")

# =============================================================================
# CHECK 5: BTC price trend in 2025 — was market bullish?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 5: BTC price trend in 2025")
print("="*70)

fc_2025 = fc[(fc.index >= "2025-01-01") & (fc.index <= "2025-12-31")]
monthly_close = fc_2025["close"].resample("ME").last()
print(f"\n  Monthly closing price:")
for dt, price in monthly_close.items():
    pct_above_sma = (price / fc_2025.loc[:dt, "close"].rolling(200*4).mean().iloc[-1] - 1) * 100 if len(fc_2025.loc[:dt]) > 200 else 0
    print(f"    {dt.strftime('%Y-%m')}: ${price:,.0f}")

# Bull/bear ratio
if "price_above_sma200" in fc_2025.columns:
    bull_pct = fc_2025["price_above_sma200"].mean() * 100
    print(f"\n  Bull market (price > SMA200): {bull_pct:.1f}% of 2025 bars")
    print(f"  Bear market: {100-bull_pct:.1f}% of 2025 bars")

# =============================================================================
# CHECK 6: SHORT accuracy by month
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 6: SHORT prediction accuracy by month (2025)")
print("="*70)

test_dates = pd.DatetimeIndex(common_idx[test_indices])
test_months = test_dates.to_period('M')

print(f"\n  {'Month':<10s} | {'SHORT signals':>13s} {'SHORT correct':>14s} {'LONG signals':>12s} {'LONG correct':>13s}")
print(f"  {'-'*10}-+-{'-'*13}-{'-'*14}-{'-'*12}-{'-'*13}")

for month in sorted(test_months.unique()):
    m_mask = test_months == month
    m_probs = test_probs_all[m_mask]
    m_dir = direction[test_indices[m_mask]]
    m_valid = (m_dir == 0) | (m_dir == 1)

    m_short = (m_probs < 0.40) & m_valid
    m_long = (m_probs > 0.60) & m_valid

    n_short = m_short.sum()
    n_long = m_long.sum()

    if n_short > 0:
        short_acc = (m_dir[m_short] == 1).mean() * 100
    else:
        short_acc = 0

    if n_long > 0:
        long_acc = (m_dir[m_long] == 0).mean() * 100
    else:
        long_acc = 0

    print(f"  {str(month):<10s} | {n_short:13d} {short_acc:13.1f}% {n_long:12d} {long_acc:12.1f}%")
