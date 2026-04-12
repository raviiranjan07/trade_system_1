"""Final V1.5 evaluation: 26 protocol metrics + trading backtest.

Matches the original backtest script setup:
  - 10 features: roc1-8 + range_position (20-bar) + rsi7
  - Label: direction H96 (LONG/SHORT binary)
  - Architecture: MLP 10→128→128→1 with dropout 0.5 applied
  - Train: 2020-2023, Val: 2024-H1, Test: 2025
  - Scaler: fit on train only
  - V1.4 exit rules for trading simulation

Usage:
  PYTHONPATH=src python scripts/final_v15_eval.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from mlops.runner import run_experiment
from mlops.evaluation import evaluate_direction_prediction

# ============================================================
# CONFIG
# ============================================================
CACHE_PATH = Path("experiments/layer2/L2-003/feature_cache.parquet")
LABELS_PATH = Path("experiments/layer2/L2-003/labels.parquet")

TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-06-30"
TEST_START, TEST_END = "2025-01-01", "2025-12-31"

LR = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 512
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN = 128
SEED = 42
CONF_LONG = 0.60
CONF_SHORT = 0.35

# V1.4 exit rules
TRAILING_STOP_LONG = 20
TRAILING_STOP_SHORT = 30
TIME_EXIT_BAR = 10
TIGHTEN_AFTER_BAR = 5
TIGHT_STOP_BPS = 8
FEE_BPS = 8.0


# ============================================================
# MODEL
# ============================================================
class MLPBinaryDir(nn.Module):
    def __init__(self, input_size=10, hidden=128, dropout=0.5):
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


class SimpleDS(Dataset):
    def __init__(self, features, labels):
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================================================
# TRADE SIMULATOR (V1.4 exit rules)
# ============================================================
def simulate_trades(confident_indices, all_probs, label_to_fc, ohlcv_open, ohlcv_high, ohlcv_low):
    """Simulate trades with V1.4 exit rules. Returns list of trade dicts."""
    trades = []
    last_exit_bar = -999

    for label_idx in sorted(confident_indices):
        fc_idx = label_to_fc[label_idx]
        if fc_idx + 1 >= len(ohlcv_open):
            continue
        if fc_idx <= last_exit_bar:
            continue

        direction = "LONG" if all_probs[label_idx] > CONF_LONG else "SHORT"
        entry_price = ohlcv_open[fc_idx + 1]
        trailing_bps = TRAILING_STOP_LONG if direction == "LONG" else TRAILING_STOP_SHORT
        best_price = entry_price
        exit_price = None
        exit_bar = TIME_EXIT_BAR
        exit_reason = "TIME_EXIT"

        for bar in range(1, TIME_EXIT_BAR + 1):
            bar_fc_idx = fc_idx + 1 + bar
            if bar_fc_idx >= len(ohlcv_high):
                break
            h = ohlcv_high[bar_fc_idx]
            l = ohlcv_low[bar_fc_idx]

            current_ts = TIGHT_STOP_BPS if bar >= TIGHTEN_AFTER_BAR else trailing_bps

            if direction == "LONG":
                best_price = max(best_price, h)
                trail_price = best_price * (1 - current_ts / 10000)
                if l <= trail_price:
                    exit_price = trail_price
                    exit_bar = bar
                    exit_reason = "TIGHT_TS" if bar >= TIGHTEN_AFTER_BAR else "TRAILING_STOP"
                    break
            else:
                best_price = min(best_price, l)
                trail_price = best_price * (1 + current_ts / 10000)
                if h >= trail_price:
                    exit_price = trail_price
                    exit_bar = bar
                    exit_reason = "TIGHT_TS" if bar >= TIGHTEN_AFTER_BAR else "TRAILING_STOP"
                    break

        if exit_price is None:
            last_bar_idx = fc_idx + 1 + TIME_EXIT_BAR + 1
            if last_bar_idx < len(ohlcv_open):
                exit_price = ohlcv_open[last_bar_idx]
            else:
                continue

        if direction == "LONG":
            gross_bps = (exit_price - entry_price) / entry_price * 10000
            mfe = (best_price - entry_price) / entry_price * 10000
            mae = (entry_price - min(ohlcv_low[fc_idx + 1:fc_idx + 1 + exit_bar + 1])) / entry_price * 10000
        else:
            gross_bps = (entry_price - exit_price) / entry_price * 10000
            mfe = (entry_price - best_price) / entry_price * 10000
            mae = (max(ohlcv_high[fc_idx + 1:fc_idx + 1 + exit_bar + 1]) - entry_price) / entry_price * 10000

        net_bps = gross_bps - FEE_BPS

        trades.append({
            "direction": direction,
            "gross_bps": gross_bps,
            "net_bps": net_bps,
            "mfe": mfe,
            "mae": mae,
            "exit_bar": exit_bar,
            "exit_reason": exit_reason,
            "prob": all_probs[label_idx],
        })
        last_exit_bar = fc_idx + 1 + exit_bar

    return trades


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ============================================================
    # LOAD DATA
    # ============================================================
    print("Loading data...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    close = fc["close"].values.astype(np.float64)

    # Compute roc1-8 fresh
    for n in range(1, 9):
        roc = np.zeros(len(close), dtype=np.float32)
        roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
        fc[f"roc{n}"] = roc

    feat_cols = [f"roc{n}" for n in range(1, 9)] + ["range_position", "rsi7"]
    print(f"  Features: {feat_cols} ({len(feat_cols)} total)")

    # Align
    common_idx = lb.index.intersection(fc.index)
    lb = lb.loc[common_idx]
    fc_aligned = fc.loc[common_idx]
    label_dates = np.array(common_idx)

    # H96 label
    direction = lb["direction"].values
    valid_mask = (direction == 0) | (direction == 1)
    y_binary = np.zeros(len(direction), dtype=np.float32)
    y_binary[direction == 0] = 1.0

    print(f"  H96: LONG={( direction==0).sum()} SHORT={(direction==1).sum()} BOTH={(direction==2).sum()} SKIP={(direction==3).sum()}")

    # Split
    train_mask = (label_dates >= np.datetime64(TRAIN_START)) & (label_dates <= np.datetime64(TRAIN_END)) & valid_mask
    val_mask = (label_dates >= np.datetime64(VAL_START)) & (label_dates <= np.datetime64(VAL_END)) & valid_mask
    test_mask = (label_dates >= np.datetime64(TEST_START)) & (label_dates <= np.datetime64(TEST_END)) & valid_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Scaler on train only
    feat_arr_raw = fc_aligned[feat_cols].values.astype(np.float32)
    feat_arr_raw = np.nan_to_num(feat_arr_raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler_mean = feat_arr_raw[train_idx].mean(axis=0)
    scaler_std = feat_arr_raw[train_idx].std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0
    feat_arr = (feat_arr_raw - scaler_mean) / scaler_std
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, y_train = feat_arr[train_idx], y_binary[train_idx]
    X_val, y_val = feat_arr[val_idx], y_binary[val_idx]
    X_test, y_test = feat_arr[test_idx], y_binary[test_idx]

    mfe_up_test = lb["mfe_up_96"].values[test_idx]
    mfe_down_test = lb["mfe_down_96"].values[test_idx]

    print(f"  Test: LONG={int(y_test.sum())} SHORT={int(len(y_test)-y_test.sum())}")

    # ============================================================
    # TRAIN
    # ============================================================
    print("\nTraining...")
    train_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = MLPBinaryDir(input_size=10, hidden=HIDDEN, dropout=0.5)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for X_batch, y_batch in train_loader:
            loss = bce(model(X_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                vl += bce(model(X_batch), y_batch).item()
        vl /= len(val_loader)
        scheduler.step(vl)

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # ============================================================
    # PREDICT ON ALL TEST BARS (for 26 metrics)
    # ============================================================
    print("\nPredicting on test set...")
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test))
        test_probs = torch.sigmoid(test_logits).numpy()

    print(f"  Confident LONG (>{CONF_LONG}): {(test_probs >= CONF_LONG).sum()}")
    print(f"  Confident SHORT (<{CONF_SHORT}): {(test_probs <= CONF_SHORT).sum()}")

    # ============================================================
    # COMPUTE 26 PROTOCOL METRICS
    # ============================================================
    print("\nComputing 26 protocol metrics...")
    test_metrics = evaluate_direction_prediction(
        probs=test_probs, labels=y_test,
        mfe_up_bps=mfe_up_test, mfe_down_bps=mfe_down_test,
        conf_threshold_long=CONF_LONG, conf_threshold_short=CONF_SHORT,
        split_name="test",
    )

    # Train/val metrics
    with torch.no_grad():
        train_probs = torch.sigmoid(model(torch.from_numpy(X_train))).numpy()
        val_probs = torch.sigmoid(model(torch.from_numpy(X_val))).numpy()

    train_metrics = evaluate_direction_prediction(probs=train_probs, labels=y_train, split_name="train")
    val_metrics = evaluate_direction_prediction(probs=val_probs, labels=y_val, split_name="val")

    # ============================================================
    # TRADING BACKTEST (V1.4 exit rules)
    # ============================================================
    print("\nRunning trading backtest (V1.4 exit rules)...")

    # Score ALL 2025 bars (not just valid ones — need to find confident bars)
    test_mask_all = (label_dates >= np.datetime64(TEST_START)) & (label_dates <= np.datetime64(TEST_END))
    all_test_indices = np.where(test_mask_all)[0]

    all_probs = np.zeros(len(label_dates), dtype=np.float32)
    with torch.no_grad():
        for i in all_test_indices:
            x = torch.from_numpy(feat_arr[i]).unsqueeze(0)
            all_probs[i] = 1 / (1 + np.exp(-model(x).item()))

    long_entries = all_test_indices[all_probs[all_test_indices] > CONF_LONG]
    short_entries = all_test_indices[all_probs[all_test_indices] < CONF_SHORT]
    confident_indices = np.concatenate([long_entries, short_entries])

    label_to_fc = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)

    trades = simulate_trades(
        confident_indices, all_probs, label_to_fc,
        fc["open"].values, fc["high"].values, fc["low"].values,
    )

    tdf = pd.DataFrame(trades)
    n_trades = len(tdf)
    winners = (tdf["net_bps"] > 0).sum()
    total_bps = tdf["net_bps"].sum()
    win_rate = winners / max(n_trades, 1) * 100
    gross_win = tdf.loc[tdf["net_bps"] > 0, "net_bps"].sum()
    gross_loss = abs(tdf.loc[tdf["net_bps"] <= 0, "net_bps"].sum())
    pf = gross_win / max(gross_loss, 0.01)

    long_trades = tdf[tdf["direction"] == "LONG"]
    short_trades = tdf[tdf["direction"] == "SHORT"]

    # Add trading metrics
    backtest_metrics = {
        "backtest_n_trades": n_trades,
        "backtest_win_rate": round(win_rate, 1),
        "backtest_total_bps": round(float(total_bps), 0),
        "backtest_profit_factor": round(pf, 2),
        "backtest_avg_bps_per_trade": round(float(tdf["net_bps"].mean()), 1) if n_trades > 0 else 0,
        "backtest_long_trades": len(long_trades),
        "backtest_long_win_rate": round(float((long_trades["net_bps"] > 0).mean() * 100), 1) if len(long_trades) > 0 else 0,
        "backtest_long_bps": round(float(long_trades["net_bps"].sum()), 0) if len(long_trades) > 0 else 0,
        "backtest_short_trades": len(short_trades),
        "backtest_short_win_rate": round(float((short_trades["net_bps"] > 0).mean() * 100), 1) if len(short_trades) > 0 else 0,
        "backtest_short_bps": round(float(short_trades["net_bps"].sum()), 0) if len(short_trades) > 0 else 0,
    }

    # Merge all metrics
    all_metrics = {}
    all_metrics.update(test_metrics)
    all_metrics.update(backtest_metrics)
    all_metrics["train_accuracy"] = train_metrics["train_accuracy"]
    all_metrics["train_confident_accuracy"] = train_metrics["train_confident_accuracy"]
    all_metrics["val_accuracy"] = val_metrics["val_accuracy"]
    all_metrics["val_confident_accuracy"] = val_metrics["val_confident_accuracy"]

    # Save model temporarily
    tmp_model = Path("tmp_final_eval_model.pt")
    torch.save({"model": model.state_dict(), "val_loss": best_val_loss}, tmp_model)

    # ============================================================
    # LOG THROUGH MLOPS
    # ============================================================
    with run_experiment(
        experiment_name="direction_prediction",
        protocol_name="direction_prediction_v1",
        params={
            "model_type": "MLP",
            "architecture": "10-128-128-1",
            "dropout": 0.5,
            "dropout_applied": True,
            "features": "roc1-8+range_position(20bar)+rsi7",
            "label": "direction_h96",
            "horizon": 96,
            "split": "date_based",
            "train": "2020-2023",
            "val": "2024-H1",
            "test": "2025",
            "scaler_fit": "train_only",
            "conf_threshold_long": CONF_LONG,
            "conf_threshold_short": CONF_SHORT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "exit_rules": "V1.4 (TS 20/30, tighten bar5 to 8bps, time exit bar10)",
        },
        model_type="MLP",
        dataset_version="feature_cache_cleaned_23col+labels_h96",
        primary_metric="test_confident_accuracy",
        notes="FINAL V1.5 eval: honest date split + 26 metrics + V1.4 backtest. Matches original backtest script setup.",
    ) as run:
        run.log_metrics(all_metrics)
        run.log_artifact(str(tmp_model))

        # Print everything
        print("\n" + "=" * 70)
        print("26 PROTOCOL METRICS")
        print("=" * 70)
        print(f"  train_accuracy:              {all_metrics['train_accuracy']}")
        print(f"  train_confident_accuracy:    {all_metrics['train_confident_accuracy']}")
        print(f"  val_accuracy:                {all_metrics['val_accuracy']}")
        print(f"  val_confident_accuracy:      {all_metrics['val_confident_accuracy']}")
        print()
        for k in sorted(test_metrics.keys()):
            print(f"  {k}: {test_metrics[k]}")

        print("\n" + "=" * 70)
        print("TRADING BACKTEST (V1.4 exit rules, 2025)")
        print("=" * 70)
        for k in sorted(backtest_metrics.keys()):
            print(f"  {k}: {backtest_metrics[k]}")

        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"  V1.4 baseline (2024-2025):   220 trades, 60.0% win, +5267 bps, PF 3.46")
        print(f"  ML H96 10feat (prev report): 961 trades, 44.8% win, +6096 bps, PF 1.62")
        print(f"  ML H96 10feat (this run):    {n_trades} trades, {win_rate:.1f}% win, {total_bps:+.0f} bps, PF {pf:.2f}")

    tmp_model.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
