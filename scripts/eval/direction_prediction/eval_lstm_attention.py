"""Evaluate LSTM+Attention model — all 4 temperature variants.

Reproduces Configs B-E from L2_003_stage3_mlp.py:
  - LSTMAttention: LSTM(4,128) + attention + connected MFE heads
  - Same 32 features, H8 label, date-based split as LSTM/GRU
  - 4 temperatures: 1.0, 0.5, 0.1, 0.05

Usage:
  PYTHONPATH=src python scripts/eval_lstm_attention.py
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
# CONFIG (exact match to L2_003_stage3_mlp.py)
# ============================================================
CACHE_PATH = Path("data/features/direction_prediction/feature_cache.parquet")
LABELS_PATH = Path("data/features/direction_prediction/labels.parquet")

TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-06-30"
TEST_START, TEST_END = "2025-01-01", "2025-12-31"

LOOKBACKS = [1, 2, 3, 4, 5, 6, 7, 8]
MFE_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]

BATCH_SIZE = 2048
MAX_EPOCHS = 100
PATIENCE = 10
LR = 0.001
HIDDEN = 128
DROPOUT = 0.5
SEED = 42

CONF_LONG = 0.60
CONF_SHORT = 0.35

TEMPERATURES = [1.0, 0.5, 0.1, 0.05]

# V1.4 exit rules
TRAILING_STOP_LONG = 20
TRAILING_STOP_SHORT = 30
TIME_EXIT_BAR = 10
TIGHTEN_AFTER_BAR = 5
TIGHT_STOP_BPS = 8
FEE_BPS = 8.0


# ============================================================
# MODEL (exact from L2_003_stage3_mlp.py line 166-189)
# ============================================================
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


class SeqDS(Dataset):
    def __init__(self, features, mfe_up, mfe_down, labels, indices):
        self.features = features
        self.mfe_up = mfe_up
        self.mfe_down = mfe_down
        self.labels = labels
        self.idx = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n = self.idx[i]
        seq = self.features[n].reshape(len(LOOKBACKS), 4)
        return (torch.from_numpy(seq),
                torch.from_numpy(self.mfe_up[n]),
                torch.from_numpy(self.mfe_down[n]),
                torch.tensor(self.labels[n]))


# ============================================================
# TRADE SIMULATOR
# ============================================================
def simulate_trades(confident_indices, all_probs, label_to_fc, ohlcv_open, ohlcv_high, ohlcv_low):
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
                    break
            else:
                best_price = min(best_price, l)
                trail_price = best_price * (1 + current_ts / 10000)
                if h >= trail_price:
                    exit_price = trail_price
                    exit_bar = bar
                    break
        if exit_price is None:
            last_bar_idx = fc_idx + 1 + TIME_EXIT_BAR + 1
            if last_bar_idx < len(ohlcv_open):
                exit_price = ohlcv_open[last_bar_idx]
            else:
                continue
        if direction == "LONG":
            gross_bps = (exit_price - entry_price) / entry_price * 10000
        else:
            gross_bps = (entry_price - exit_price) / entry_price * 10000
        net_bps = gross_bps - FEE_BPS
        trades.append({"direction": direction, "net_bps": net_bps, "prob": all_probs[label_idx]})
        last_exit_bar = fc_idx + 1 + exit_bar
    return trades


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ============================================================
    # LOAD + PREPARE DATA (same as LSTM/GRU eval)
    # ============================================================
    print("Loading data...")
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

    common_idx = lb.index.intersection(fc.index)
    lb = lb.loc[common_idx]
    label_dates = np.array(common_idx)
    label_pos = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)

    train_fc_mask = (fc.index >= TRAIN_START) & (fc.index <= TRAIN_END)
    diff_arr_raw = np.nan_to_num(diff_arr_raw, nan=0.0, posinf=0.0, neginf=0.0)
    mean = diff_arr_raw[train_fc_mask].mean(axis=0)
    std = diff_arr_raw[train_fc_mask].std(axis=0)
    std[std < 1e-8] = 1.0
    feat_arr = (diff_arr_raw - mean) / std
    diff_aligned = feat_arr[label_pos]

    y_mfe_up = lb[[f"mfe_up_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0
    y_mfe_down = lb[[f"mfe_down_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0

    direction_h8 = lb["direction_h8"].values
    valid_mask = (direction_h8 == 0) | (direction_h8 == 1)
    y_dir = np.zeros(len(direction_h8), dtype=np.float32)
    y_dir[direction_h8 == 0] = 1.0

    train_mask = (label_dates >= np.datetime64(TRAIN_START)) & (label_dates <= np.datetime64(TRAIN_END)) & valid_mask
    val_mask = (label_dates >= np.datetime64(VAL_START)) & (label_dates <= np.datetime64(VAL_END)) & valid_mask
    test_mask = (label_dates >= np.datetime64(TEST_START)) & (label_dates <= np.datetime64(TEST_END)) & valid_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    mfe_up_96_test = lb["mfe_up_96"].values[test_idx]
    mfe_down_96_test = lb["mfe_down_96"].values[test_idx]

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_ds = SeqDS(diff_aligned, y_mfe_up, y_mfe_down, y_dir, train_idx)
    val_ds = SeqDS(diff_aligned, y_mfe_up, y_mfe_down, y_dir, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    label_to_fc = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)
    test_mask_all = (label_dates >= np.datetime64(TEST_START)) & (label_dates <= np.datetime64(TEST_END))
    all_test_indices = np.where(test_mask_all)[0]

    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    # ============================================================
    # RUN ALL 4 TEMPERATURES
    # ============================================================
    all_results = []

    for temp in TEMPERATURES:
        print(f"\n{'='*70}")
        print(f"TEMPERATURE = {temp}")
        print(f"{'='*70}")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = LSTMAttention(input_size=4, hidden=HIDDEN, dropout=DROPOUT, temperature=temp)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_vl = float("inf")
        best_state = None
        patience_ctr = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for x, mu, md, d in train_loader:
                p_mu, p_md, p_dir = model(x)
                loss = mse_fn(p_mu, mu) + mse_fn(p_md, md) + 5.0 * bce_fn(p_dir, d)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for x, mu, md, d in val_loader:
                    p_mu, p_md, p_dir = model(x)
                    vl += (mse_fn(p_mu, mu) + mse_fn(p_md, md) + 5.0 * bce_fn(p_dir, d)).item()
            vl /= len(val_loader)
            scheduler.step(vl)

            if vl < best_vl - 1e-5:
                best_vl = vl
                patience_ctr = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        model.eval()

        # Predict on test
        test_probs = np.zeros(len(test_idx), dtype=np.float32)
        with torch.no_grad():
            for i, idx in enumerate(test_idx):
                seq = torch.from_numpy(diff_aligned[idx].reshape(1, len(LOOKBACKS), 4))
                _, _, p_dir = model(seq)
                test_probs[i] = torch.sigmoid(p_dir).item()

        y_test = y_dir[test_idx]
        n_conf_long = (test_probs >= CONF_LONG).sum()
        n_conf_short = (test_probs <= CONF_SHORT).sum()
        print(f"  Confident LONG: {n_conf_long} | SHORT: {n_conf_short}")

        # 26 metrics
        test_metrics = evaluate_direction_prediction(
            probs=test_probs, labels=y_test,
            mfe_up_bps=mfe_up_96_test, mfe_down_bps=mfe_down_96_test,
            conf_threshold_long=CONF_LONG, conf_threshold_short=CONF_SHORT,
            split_name="test",
        )

        # Backtest
        all_probs = np.zeros(len(label_dates), dtype=np.float32)
        with torch.no_grad():
            for i in all_test_indices:
                seq = torch.from_numpy(feat_arr[i].reshape(1, len(LOOKBACKS), 4))
                _, _, p_dir = model(seq)
                all_probs[i] = torch.sigmoid(p_dir).item()

        long_entries = all_test_indices[all_probs[all_test_indices] > CONF_LONG]
        short_entries = all_test_indices[all_probs[all_test_indices] < CONF_SHORT]
        trades = simulate_trades(
            np.concatenate([long_entries, short_entries]), all_probs, label_to_fc,
            fc["open"].values, fc["high"].values, fc["low"].values,
        )

        tdf = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["direction", "net_bps", "prob"])
        n_trades = len(tdf)
        backtest_metrics = {}
        if n_trades > 0:
            winners = (tdf["net_bps"] > 0).sum()
            total_bps = tdf["net_bps"].sum()
            win_rate = winners / n_trades * 100
            gross_win = tdf.loc[tdf["net_bps"] > 0, "net_bps"].sum()
            gross_loss = abs(tdf.loc[tdf["net_bps"] <= 0, "net_bps"].sum())
            pf = gross_win / max(gross_loss, 0.01)
            long_t = tdf[tdf["direction"] == "LONG"]
            short_t = tdf[tdf["direction"] == "SHORT"]
            backtest_metrics = {
                "backtest_n_trades": n_trades,
                "backtest_win_rate": round(win_rate, 1),
                "backtest_total_bps": round(float(total_bps), 0),
                "backtest_profit_factor": round(pf, 2),
                "backtest_long_trades": len(long_t),
                "backtest_long_bps": round(float(long_t["net_bps"].sum()), 0) if len(long_t) > 0 else 0,
                "backtest_short_trades": len(short_t),
                "backtest_short_bps": round(float(short_t["net_bps"].sum()), 0) if len(short_t) > 0 else 0,
            }

        all_metrics = {}
        all_metrics.update(test_metrics)
        all_metrics.update(backtest_metrics)

        # Log through MLOps
        with run_experiment(
            experiment_name="direction_prediction",
            protocol_name="direction_prediction_v1",
            params={
                "model_type": "LSTMAttention",
                "architecture": "LSTM(4,128)+attention+MFE_heads+connected_direction",
                "temperature": temp,
                "features": "roc_diff+rsi_diff+rp_diff+sma200_diff x 8 = 32",
                "input_shape": "[8, 4] sequence",
                "label": "direction_h8",
                "horizon": 8,
                "loss": "MSE(mfe_up)+MSE(mfe_down)+5.0*BCE(dir)",
                "weight_decay": 0.0,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "dropout": DROPOUT,
                "seed": SEED,
            },
            model_type=f"LSTMAttention_temp{temp}",
            dataset_version="feature_cache_cleaned_23col+labels_h8",
            primary_metric="test_confident_accuracy",
            notes=f"LSTM+Attention temp={temp}. Connected MFE heads. Honest date split.",
        ) as run:
            run.log_metrics(all_metrics)

            ca = test_metrics.get("test_confident_accuracy", 0)
            nc = test_metrics.get("test_n_confident", 0)
            bt = backtest_metrics.get("backtest_total_bps", 0)
            bp = backtest_metrics.get("backtest_profit_factor", 0)
            print(f"  Result: {ca:.1%} conf acc, {nc} bars, {bt:+.0f} bps, PF {bp:.2f}")

        all_results.append({
            "temp": temp,
            "conf_acc": test_metrics.get("test_confident_accuracy", 0),
            "conf_acc_long": test_metrics.get("test_confident_accuracy_long", 0),
            "conf_acc_short": test_metrics.get("test_confident_accuracy_short", 0),
            "n_conf": test_metrics.get("test_n_confident", 0),
            "n_conf_long": test_metrics.get("test_n_confident_long", 0),
            "n_conf_short": test_metrics.get("test_n_confident_short", 0),
            "bt_bps": backtest_metrics.get("backtest_total_bps", 0),
            "bt_pf": backtest_metrics.get("backtest_profit_factor", 0),
            "bt_trades": backtest_metrics.get("backtest_n_trades", 0),
        })

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n\n{'='*70}")
    print("ATTENTION TEMPERATURE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Temp':<6s} | {'Conf acc':>8s} {'Long acc':>9s} {'Short acc':>10s} | {'N conf':>7s} {'L':>5s} {'S':>5s} | {'Trades':>7s} {'Bps':>8s} {'PF':>6s}")
    print(f"  {'-'*6}-+-{'-'*8}-{'-'*9}-{'-'*10}-+-{'-'*7}-{'-'*5}-{'-'*5}-+-{'-'*7}-{'-'*8}-{'-'*6}")
    for r in all_results:
        print(f"  {r['temp']:<6.2f} | {r['conf_acc']:7.1%} {r['conf_acc_long']:8.1%} {r['conf_acc_short']:9.1%} | {r['n_conf']:7d} {r['n_conf_long']:5d} {r['n_conf_short']:5d} | {r['bt_trades']:7d} {r['bt_bps']:+8.0f} {r['bt_pf']:6.2f}")

    print(f"\n  Previous models:")
    print(f"  {'MLP final (H96)':<25s} | 55.8% conf, 735 bars, +4385 bps, PF 1.52")
    print(f"  {'LSTM/GRU baseline':<25s} | 56.5% conf, 1046 bars, -2683 bps, PF 0.80")
    print(f"  {'Curriculum LOOSE':<25s} | 56.4% conf, 786 bars, -229 bps, PF 0.98")

    print(f"\n  Prev reported:")
    print(f"  {'Attention temp=0.5':<25s} | 57.8% conf, 1814 bars")
    print(f"  {'Attention temp=0.05':<25s} | 58.8% conf, 816 bars (BEST EVER)")


if __name__ == "__main__":
    main()
