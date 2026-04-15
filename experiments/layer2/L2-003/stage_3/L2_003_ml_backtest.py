"""L2-003: Backtest MLP model — H8 label + momentum10

Compares current production (H96, 10 feat) vs new (H8, 11 feat with momentum10).
Uses V1.4 exit rules (trailing stop + time exit + tightening).

Run: PYTHONPATH=src python experiments/layer2/L2-003/L2_003_ml_backtest.py
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

# =============================================================================
# CONFIG
# =============================================================================
BASE = Path("experiments/layer2/L2-003")
CACHE_PATH = BASE / "feature_cache.parquet"
LABELS_PATH = BASE / "labels.parquet"
OHLCV_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")

CONF_LONG = 0.60   # prob > this → LONG
CONF_SHORT = 0.65  # prob < (1-this) = 0.35 → SHORT
FEE_BPS = 8.0          # round-trip fees

# V1.4 exit rules
TRAILING_STOP_LONG = 20   # bps
TRAILING_STOP_SHORT = 30  # bps
TIME_EXIT_BAR = 10
TIGHTEN_AFTER_BAR = 5
TIGHT_STOP_BPS = 8

TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# =============================================================================
# STEP 1: Train model (same as before)
# =============================================================================
print("="*70)
print("STEP 1: Training MLP model (2020-2023)")
print("="*70)

fc = pd.read_parquet(CACHE_PATH)
lb = pd.read_parquet(LABELS_PATH)

close = fc["close"].values.astype(np.float64)
high = fc["high"].values.astype(np.float64)
low = fc["low"].values.astype(np.float64)

for n in range(1, 9):
    roc = np.zeros(len(close), dtype=np.float32)
    roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
    fc[f"roc{n}"] = roc

# Compute momentum10
if "momentum10" not in fc.columns:
    mom = np.zeros(len(close), dtype=np.float32)
    mom[10:] = (close[10:] - close[:-10]).astype(np.float32)
    fc["momentum10"] = mom

feat_cols = [f"roc{n}" for n in range(1, 9)] + ["range_position", "rsi7", "momentum10"]
F = len(feat_cols)
print(f"  Features: {feat_cols} ({F} total)")

common_idx = lb.index.intersection(fc.index)
lb = lb.loc[common_idx]
fc_aligned = fc.loc[common_idx]
feat_arr_raw = fc_aligned[feat_cols].values.astype(np.float32)
label_dates = np.array(common_idx)

# Normalize on 2020-2023
train_fc_mask = (pd.DatetimeIndex(common_idx) >= "2020-01-01") & (pd.DatetimeIndex(common_idx) <= "2023-12-31")
scaler_mean = feat_arr_raw[train_fc_mask].mean(axis=0)
scaler_std = feat_arr_raw[train_fc_mask].std(axis=0)
scaler_std[scaler_std < 1e-8] = 1.0
feat_arr = (feat_arr_raw - scaler_mean) / scaler_std
feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

# Use H8 direction label
direction = lb["direction_h8"].values
valid_mask = (direction == 0) | (direction == 1)
y_binary = np.zeros(len(direction), dtype=np.float32)
y_binary[direction == 0] = 1.0

print(f"  H8 label: LONG={( direction==0).sum()} SHORT={(direction==1).sum()} SKIP={(direction==3).sum()}")

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

DEVICE = torch.device("cpu")
model = MLPBinaryDir(input_size=F).to(DEVICE)
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
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        loss = bce(model(x), y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    vl = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            vl += bce(model(x), y).item()
    vl /= len(val_loader)
    scheduler.step(vl)
    if vl < best_val_loss - 1e-5:
        best_val_loss = vl; patience_ctr = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= 10:
            print(f"  Trained {epoch} epochs, val_loss: {best_val_loss:.4f}")
            break

model.load_state_dict(best_state)
model.eval()
print(f"  Model ready")

# =============================================================================
# STEP 2: Score all 2025 bars
# =============================================================================
print(f"\n{'='*70}")
print("STEP 2: Scoring all 2025 bars")
print("="*70)

test_mask_all = (label_dates >= np.datetime64(TEST_START)) & (label_dates <= np.datetime64(TEST_END))
test_indices = np.where(test_mask_all)[0]

all_probs = np.zeros(len(label_dates), dtype=np.float32)
model.eval()
with torch.no_grad():
    for i in test_indices:
        x = torch.from_numpy(feat_arr[i]).unsqueeze(0)
        logit = model(x).item()
        all_probs[i] = 1 / (1 + np.exp(-logit))

# Find confident bars
long_entries = test_indices[all_probs[test_indices] > CONF_LONG]
short_entries = test_indices[all_probs[test_indices] < (1 - CONF_SHORT)]

print(f"  Total 2025 bars: {len(test_indices)}")
print(f"  Confident LONG entries:  {len(long_entries)} (prob > {CONF_LONG})")
print(f"  Confident SHORT entries: {len(short_entries)} (prob < {1-CONF_SHORT})")

# =============================================================================
# STEP 3: Load OHLCV for trade simulation
# =============================================================================
print(f"\n{'='*70}")
print("STEP 3: Simulating trades with V1.4 exit rules")
print("="*70)

# Use feature_cache OHLCV (already aligned)
ohlcv_open = fc["open"].values
ohlcv_high = fc["high"].values
ohlcv_low = fc["low"].values
ohlcv_close = fc["close"].values

# Map from label index to fc index
label_to_fc = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)

# =============================================================================
# Trade simulator with V1.4 exit rules
# =============================================================================
@dataclass
class Trade:
    entry_time: str
    direction: str
    entry_price: float
    exit_price: float
    exit_bar: int
    exit_reason: str
    gross_bps: float
    net_bps: float
    mfe_bps: float
    mae_bps: float
    model_prob: float

def simulate_trade(label_idx, direction, prob):
    """Simulate a single trade using V1.4 exit rules."""
    fc_idx = label_to_fc[label_idx]

    # Entry at next bar's open
    if fc_idx + 1 >= len(ohlcv_open):
        return None
    entry_price = ohlcv_open[fc_idx + 1]
    entry_time = str(common_idx[label_idx])

    if direction == "LONG":
        trailing_bps = TRAILING_STOP_LONG
    else:
        trailing_bps = TRAILING_STOP_SHORT

    # Walk through bars
    best_price = entry_price
    worst_price = entry_price
    exit_price = None
    exit_bar = 0
    exit_reason = None

    for bar in range(1, TIME_EXIT_BAR + 1):
        bar_fc_idx = fc_idx + 1 + bar  # +1 for entry bar, +bar for subsequent
        if bar_fc_idx >= len(ohlcv_high):
            break

        h = ohlcv_high[bar_fc_idx]
        l = ohlcv_low[bar_fc_idx]
        c = ohlcv_close[bar_fc_idx]

        # Determine current trailing stop
        if bar > TIGHTEN_AFTER_BAR:
            current_trail = TIGHT_STOP_BPS
        else:
            current_trail = trailing_bps

        if direction == "LONG":
            # Update best price
            if h > best_price:
                best_price = h
            if l < worst_price:
                worst_price = l

            # Check trailing stop
            trail_level = best_price * (1 - current_trail / 10000)
            if l <= trail_level:
                exit_price = trail_level
                exit_bar = bar
                exit_reason = "TIGHT_TS" if bar > TIGHTEN_AFTER_BAR else "TRAILING_STOP"
                break
        else:  # SHORT
            if l < best_price:
                best_price = l
            if h > worst_price:
                worst_price = h

            trail_level = best_price * (1 + current_trail / 10000)
            if h >= trail_level:
                exit_price = trail_level
                exit_bar = bar
                exit_reason = "TIGHT_TS" if bar > TIGHTEN_AFTER_BAR else "TRAILING_STOP"
                break

    # Time exit if no trailing stop hit
    if exit_price is None:
        bar_fc_idx = fc_idx + 1 + TIME_EXIT_BAR
        if bar_fc_idx < len(ohlcv_close):
            exit_price = ohlcv_close[bar_fc_idx]
            exit_bar = TIME_EXIT_BAR
            exit_reason = "TIME_EXIT"
        else:
            return None

    # Calculate PnL
    if direction == "LONG":
        gross_bps = (exit_price - entry_price) / entry_price * 10000
        mfe_bps = (best_price - entry_price) / entry_price * 10000
        mae_bps = (entry_price - worst_price) / entry_price * 10000
    else:
        gross_bps = (entry_price - exit_price) / entry_price * 10000
        mfe_bps = (entry_price - best_price) / entry_price * 10000
        mae_bps = (worst_price - entry_price) / entry_price * 10000

    net_bps = gross_bps - FEE_BPS

    return Trade(
        entry_time=entry_time,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_bar=exit_bar,
        exit_reason=exit_reason,
        gross_bps=gross_bps,
        net_bps=net_bps,
        mfe_bps=mfe_bps,
        mae_bps=mae_bps,
        model_prob=prob,
    )

# =============================================================================
# Run all trades (skip if already in position)
# =============================================================================
# Combine long and short entries, sort by time
all_entries = []
for idx in long_entries:
    all_entries.append((idx, "LONG", all_probs[idx]))
for idx in short_entries:
    all_entries.append((idx, "SHORT", all_probs[idx]))

all_entries.sort(key=lambda x: x[0])

trades = []
next_available_bar = 0  # can't enter if still in previous trade

for label_idx, direction, prob in all_entries:
    fc_idx = label_to_fc[label_idx]

    # Skip if overlapping with previous trade
    if fc_idx < next_available_bar:
        continue

    trade = simulate_trade(label_idx, direction, prob)
    if trade is not None:
        trades.append(trade)
        # Block next bars (entry bar + exit_bar bars)
        next_available_bar = fc_idx + 1 + trade.exit_bar + 1

print(f"  Total signals: {len(all_entries)}")
print(f"  Trades taken (no overlap): {len(trades)}")

# =============================================================================
# STEP 4: Results
# =============================================================================
print(f"\n{'='*70}")
print("STEP 4: RESULTS — ML Model Standalone Backtest (2025)")
print("="*70)

if len(trades) == 0:
    print("  No trades!")
else:
    tdf = pd.DataFrame([t.__dict__ for t in trades])

    winners = tdf[tdf['net_bps'] > 0]
    losers = tdf[tdf['net_bps'] <= 0]

    total_bps = tdf['net_bps'].sum()
    win_rate = len(winners) / len(tdf) * 100
    avg_win = winners['net_bps'].mean() if len(winners) > 0 else 0
    avg_loss = losers['net_bps'].mean() if len(losers) > 0 else 0
    gross_win = winners['net_bps'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['net_bps'].sum()) if len(losers) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n  Trades:       {len(tdf)}")
    print(f"  Winners:      {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers:       {len(losers)} ({100-win_rate:.1f}%)")
    print(f"  Total bps:    {total_bps:+.0f}")
    print(f"  Avg win:      {avg_win:+.1f} bps")
    print(f"  Avg loss:     {avg_loss:.1f} bps")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg bps/trade: {tdf['net_bps'].mean():.1f}")
    print(f"  Max drawdown:  {tdf['net_bps'].cumsum().min():.0f} bps")

    # By direction
    print(f"\n  By direction:")
    for d in ["LONG", "SHORT"]:
        d_df = tdf[tdf['direction'] == d]
        if len(d_df) > 0:
            d_win = (d_df['net_bps'] > 0).mean() * 100
            d_total = d_df['net_bps'].sum()
            print(f"    {d}: {len(d_df)} trades, {d_win:.1f}% win, {d_total:+.0f} bps")

    # By exit reason
    print(f"\n  By exit reason:")
    for er in tdf['exit_reason'].unique():
        er_df = tdf[tdf['exit_reason'] == er]
        er_win = (er_df['net_bps'] > 0).mean() * 100
        er_total = er_df['net_bps'].sum()
        print(f"    {er}: {len(er_df)} trades, {er_win:.1f}% win, {er_total:+.0f} bps")

    # By confidence bucket
    print(f"\n  By model confidence:")
    for lo, hi in [(0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.0)]:
        mask = (tdf['model_prob'] >= lo) & (tdf['model_prob'] < hi)
        mask_s = (tdf['model_prob'] <= (1-lo)) & (tdf['model_prob'] > (1-hi))
        combined = mask | mask_s
        n = combined.sum()
        if n > 0:
            c_win = (tdf.loc[combined, 'net_bps'] > 0).mean() * 100
            c_total = tdf.loc[combined, 'net_bps'].sum()
            print(f"    conf [{lo:.2f}-{hi:.2f}): {n} trades, {c_win:.1f}% win, {c_total:+.0f} bps")

    # Monthly breakdown
    print(f"\n  Monthly breakdown:")
    tdf['month'] = pd.to_datetime(tdf['entry_time']).dt.to_period('M')
    for month, mdf in tdf.groupby('month'):
        m_win = (mdf['net_bps'] > 0).mean() * 100
        m_total = mdf['net_bps'].sum()
        print(f"    {month}: {len(mdf)} trades, {m_win:.1f}% win, {m_total:+.0f} bps")

    # =============================================================================
    # COMPARISON WITH V1.4
    # =============================================================================
    print(f"\n{'='*70}")
    print("COMPARISON: ML Model vs V1.4 (2025)")
    print("="*70)
    print(f"\n  {'System':<30s} | {'Trades':>6s} {'Win%':>6s} {'Total bps':>10s} {'Avg/trade':>9s} {'PF':>6s}")
    print(f"  {'-'*30}-+-{'-'*6}-{'-'*6}-{'-'*10}-{'-'*9}-{'-'*6}")
    print(f"  {'V1.4 (2024-2025)':<30s} | {'220':>6s} {'60.0%':>6s} {'+5267':>10s} {'23.9':>9s} {'3.46':>6s}")
    print(f"  {'ML H96 10feat (prev)':<30s} | {'961':>6s} {'44.8%':>6s} {'+6096':>10s} {'6.3':>9s} {'1.62':>6s}")
    print(f"  {'ML H8 11feat (new)':<30s} | {len(tdf):6d} {win_rate:5.1f}% {total_bps:+10.0f} {tdf['net_bps'].mean():9.1f} {pf:6.2f}")
