"""Attention Architecture Ablation — WITH MFE vs NO MFE vs ASYMMETRY head.

Tests whether MFE auxiliary task helps or hurts direction prediction.
Each variant is trained 3× (different seeds), evaluated on test set, and
backtested with V2 exits. Results logged to MLflow + local JSON.

Variants:
  A) WITH_MFE:   LSTM → Attention → [mfe_up(8), mfe_down(8)] → concat → dir  (production)
  B) NO_MFE:     LSTM → Attention → dir  (simplest)
  C) ASYMMETRY:  LSTM → Attention → asymmetry(mfe_up-mfe_down, 8) → concat → dir

Run: PYTHONPATH=src python experiments/attention_ablation/run_ablation.py
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlflow
from mlops import tracking

# === DATA CONFIG ===
CACHE_PATH = REPO_ROOT / "data/features/direction_prediction/feature_cache.parquet"
LABELS_PATH = REPO_ROOT / "data/features/direction_prediction/labels.parquet"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_RANGE = ("2020-01-01", "2023-12-31")
VAL_RANGE = ("2024-01-01", "2024-12-31")
TEST_RANGE = ("2025-01-01", "2025-12-31")

# === HYPERPARAMS (same as production) ===
HIDDEN = 128
DROPOUT = 0.5
TEMPERATURE = 0.5
LR = 0.001
BATCH_SIZE = 2048
MAX_EPOCHS = 100
PATIENCE = 10
LOOKBACKS = [1, 2, 3, 4, 5, 6, 7, 8]
MFE_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]
CONF_LONG = 0.60
CONF_SHORT = 0.58
N_RUNS = 3


# =====================================================================
# MODEL VARIANTS
# =====================================================================

class VariantA_WithMFE(nn.Module):
    """Production architecture: MFE heads feed direction."""
    name = "WITH_MFE"

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.attn_score = nn.Linear(HIDDEN, 1)
        self.h_mfe_up = nn.Linear(HIDDEN, 8)
        self.h_mfe_down = nn.Linear(HIDDEN, 8)
        self.h_dir = nn.Linear(HIDDEN + 8 + 8, 1)

    def forward(self, x):
        all_h, _ = self.lstm(x)
        scores = self.attn_score(all_h).squeeze(-1)
        w = torch.softmax(scores / TEMPERATURE, dim=1)
        attended = torch.bmm(w.unsqueeze(1), all_h).squeeze(1)
        attended = self.dropout(attended)
        p_mu = self.h_mfe_up(attended)
        p_md = self.h_mfe_down(attended)
        p_dir = self.h_dir(torch.cat([attended, p_mu, p_md], dim=1)).squeeze(-1)
        return {"dir": p_dir, "mfe_up": p_mu, "mfe_down": p_md}


class VariantB_NoMFE(nn.Module):
    """No MFE — pure direction from attended vector."""
    name = "NO_MFE"

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.attn_score = nn.Linear(HIDDEN, 1)
        self.h_dir = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        all_h, _ = self.lstm(x)
        scores = self.attn_score(all_h).squeeze(-1)
        w = torch.softmax(scores / TEMPERATURE, dim=1)
        attended = torch.bmm(w.unsqueeze(1), all_h).squeeze(1)
        attended = self.dropout(attended)
        p_dir = self.h_dir(attended).squeeze(-1)
        return {"dir": p_dir}


class VariantC_Asymmetry(nn.Module):
    """Single asymmetry head: predict mfe_up - mfe_down (directional bias)."""
    name = "ASYMMETRY"

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, HIDDEN, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.attn_score = nn.Linear(HIDDEN, 1)
        self.h_asym = nn.Linear(HIDDEN, 8)
        self.h_dir = nn.Linear(HIDDEN + 8, 1)

    def forward(self, x):
        all_h, _ = self.lstm(x)
        scores = self.attn_score(all_h).squeeze(-1)
        w = torch.softmax(scores / TEMPERATURE, dim=1)
        attended = torch.bmm(w.unsqueeze(1), all_h).squeeze(1)
        attended = self.dropout(attended)
        p_asym = self.h_asym(attended)
        p_dir = self.h_dir(torch.cat([attended, p_asym], dim=1)).squeeze(-1)
        return {"dir": p_dir, "asym": p_asym}


# =====================================================================
# DATASET
# =====================================================================

class TrainDS(Dataset):
    def __init__(self, X, y_mu, y_md, y_dir, y_asym):
        self.X = torch.from_numpy(X)
        self.y_mu = torch.from_numpy(y_mu)
        self.y_md = torch.from_numpy(y_md)
        self.y_dir = torch.from_numpy(y_dir)
        self.y_asym = torch.from_numpy(y_asym)

    def __len__(self):
        return len(self.y_dir)

    def __getitem__(self, i):
        return self.X[i], self.y_mu[i], self.y_md[i], self.y_dir[i], self.y_asym[i]


# =====================================================================
# LOSS FUNCTIONS
# =====================================================================

def loss_with_mfe(out, y_mu, y_md, y_dir, y_asym, mse, bce):
    return mse(out["mfe_up"], y_mu) + mse(out["mfe_down"], y_md) + 0.5 * bce(out["dir"], y_dir)


def loss_no_mfe(out, y_mu, y_md, y_dir, y_asym, mse, bce):
    return bce(out["dir"], y_dir)


def loss_asymmetry(out, y_mu, y_md, y_dir, y_asym, mse, bce):
    return 0.5 * mse(out["asym"], y_asym) + bce(out["dir"], y_dir)


VARIANTS = [
    (VariantA_WithMFE, loss_with_mfe),
    (VariantB_NoMFE, loss_no_mfe),
    (VariantC_Asymmetry, loss_asymmetry),
]


# =====================================================================
# TRAINING
# =====================================================================

def train_model(model_cls, loss_fn, X_tr, y_mu_tr, y_md_tr, y_dir_tr, y_asym_tr,
                X_va, y_mu_va, y_md_va, y_dir_va, y_asym_va):
    model = model_cls()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    ds = TrainDS(X_tr, y_mu_tr, y_md_tr, y_dir_tr, y_asym_tr)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    best_vl, best_state, patience_ctr, epochs_used = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_used = epoch
        model.train()
        for xb, ymu, ymd, ydir, yasym in loader:
            out = model(xb)
            loss = loss_fn(out, ymu, ymd, ydir, yasym, mse, bce)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            out_v = model(torch.from_numpy(X_va))
            vl = loss_fn(out_v,
                         torch.from_numpy(y_mu_va), torch.from_numpy(y_md_va),
                         torch.from_numpy(y_dir_va), torch.from_numpy(y_asym_va),
                         mse, bce).item()
        sched.step(vl)

        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, epochs_used, best_vl


# =====================================================================
# EVALUATION
# =====================================================================

def evaluate_model(model, X_te, y_dir_te):
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X_te))
        probs = torch.sigmoid(out["dir"]).numpy()

    pred = (probs > 0.5).astype(int)
    y = y_dir_te.astype(int)
    acc = float((pred == y).mean())

    conf_mask = (probs >= CONF_LONG) | (probs <= (1 - CONF_SHORT))
    n_conf = int(conf_mask.sum())
    if n_conf > 0:
        conf_acc = float((pred[conf_mask] == y[conf_mask]).mean())
    else:
        conf_acc = 0.0

    # Signal breakdown
    long_mask = probs >= CONF_LONG
    short_mask = probs <= (1 - CONF_SHORT)
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())

    return {
        "overall_acc": round(acc * 100, 2),
        "confident_acc": round(conf_acc * 100, 2),
        "n_confident": n_conf,
        "n_long": n_long,
        "n_short": n_short,
        "prob_mean": round(float(probs.mean()), 4),
        "prob_std": round(float(probs.std()), 4),
    }


def backtest_model(model, diff_scaled, df_15m, scaler_mean, scaler_std):
    """Quick backtest using the trained model as signal generator."""
    from engine.config.loader import load_config
    from engine.position_manager import V12PositionManager
    from engine.strategy import V12Strategy, Direction, SignalType, Signal

    cfg = load_config()
    strategy = V12Strategy(cfg)
    df = strategy.compute_indicators(df_15m.copy())

    test = df["2024-01-01":"2025-12-31"]
    test_indices = df.index.get_indexer(test.index)

    # Generate signals from model
    model.eval()
    signals = {}
    for i, global_idx in enumerate(test_indices):
        if global_idx < 8 or global_idx >= len(diff_scaled):
            continue
        feats = diff_scaled[global_idx].reshape(1, 8, 4)
        with torch.no_grad():
            out = model(torch.from_numpy(feats))
            prob = torch.sigmoid(out["dir"]).item()

        if prob >= CONF_LONG:
            signals[i] = Signal(direction=Direction.LONG, signal_type=SignalType.ML_ATTN_LONG,
                                bar_index=i, timestamp=test.index[i], rsi=prob,
                                price=test.iloc[i]["close"])
        elif prob <= (1 - CONF_SHORT):
            signals[i] = Signal(direction=Direction.SHORT, signal_type=SignalType.ML_ATTN_SHORT,
                                bar_index=i, timestamp=test.index[i], rsi=prob,
                                price=test.iloc[i]["close"])

    # Run through position manager with V2 exits (no LOCKED_PROFIT)
    pm = V12PositionManager(cfg, exit_version="v2")

    # Load 1m ticks
    df_1m = pd.read_parquet(REPO_ROOT / "data/raw/BTCUSDT_1m_ohlcv.parquet")
    df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    df_1m = df_1m.sort_index()["2024-01-01":"2025-12-31"]
    idx_1m = df_1m.index.values
    prices_1m = df_1m["close"].values

    highs = test["high"].values
    lows = test["low"].values
    closes = test["close"].values
    opens = test["open"].values
    times = test.index
    n = len(test)

    i = 0
    while i < n:
        if pm.is_in_position:
            bar_start = np.datetime64(times[i])
            bar_end = bar_start + np.timedelta64(15, "m")
            s_idx = np.searchsorted(idx_1m, bar_start, side="left")
            e_idx = np.searchsorted(idx_1m, bar_end, side="left")
            trade = None
            for t_idx in range(s_idx, e_idx):
                trade = pm.on_tick(float(prices_1m[t_idx]), idx_1m[t_idx])
                if trade is not None:
                    break
            if trade is None:
                trade = pm.on_bar(highs[i], lows[i], closes[i], times[i], i)
            i += 1
            continue

        if i in signals:
            sig = signals[i]
            if i + 1 < n:
                pm.open_position(
                    direction=sig.direction, signal_type=sig.signal_type,
                    entry_price=opens[i + 1], entry_time=times[i + 1],
                    signal_time=sig.timestamp)
                i += 2
                continue
        i += 1

    # Compute metrics from ML_ATTN trades only
    all_trades = pm.trades
    ml_trades = [t for t in all_trades if t.signal_type.startswith("ML_ATTN")]
    if not ml_trades:
        return {"n_trades": 0, "total_bps": 0, "pf": 0, "stop_rate": 0, "avg_bps": 0, "max_dd": 0}

    bps = [t.net_profit_bps for t in ml_trades]
    wins = [b for b in bps if b > 0]
    losses = [b for b in bps if b <= 0]
    gw = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1
    stops = sum(1 for t in ml_trades if t.exit_reason == "STOP_LOSS")

    eq = np.cumsum(bps)
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) > 0 else 0

    return {
        "n_trades": len(ml_trades),
        "total_bps": round(sum(bps), 1),
        "pf": round(gw / gl, 2) if gl > 0 else 0,
        "stop_rate": round(stops / len(ml_trades) * 100, 1),
        "avg_bps": round(sum(bps) / len(ml_trades), 1),
        "max_dd": round(dd, 1),
        "win_rate": round(len(wins) / len(ml_trades) * 100, 1),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0 = time.time()

    print("Loading data...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    print("Computing features...")
    close = fc["close"].values.astype(np.float64)
    rsi7 = fc["rsi7"].values.astype(np.float64)
    rp = fc["range_position"].values.astype(np.float64)
    sma200 = fc["sma200_dist_pct"].values.astype(np.float64)

    diff_list = []
    for n_lb in LOOKBACKS:
        roc_d = np.zeros(len(close), dtype=np.float32)
        roc_d[n_lb:] = ((close[n_lb:] - close[:-n_lb]) / close[:-n_lb] * 10000).astype(np.float32)
        rsi_d = np.zeros(len(close), dtype=np.float32)
        rsi_d[n_lb:] = (rsi7[n_lb:] - rsi7[:-n_lb]).astype(np.float32)
        rp_d = np.zeros(len(close), dtype=np.float32)
        rp_d[n_lb:] = (rp[n_lb:] - rp[:-n_lb]).astype(np.float32)
        sma_d = np.zeros(len(close), dtype=np.float32)
        sma_d[n_lb:] = (sma200[n_lb:] - sma200[:-n_lb]).astype(np.float32)
        diff_list.extend([roc_d, rsi_d, rp_d, sma_d])
    diff_raw = np.column_stack(diff_list).astype(np.float32)
    diff_raw = np.nan_to_num(diff_raw, nan=0.0, posinf=0.0, neginf=0.0)

    common_idx = lb.index.intersection(fc.index)
    fc_pos = fc.index.get_indexer(common_idx)
    lb = lb.loc[common_idx]
    diff_raw = diff_raw[fc_pos]
    dates = common_idx

    direction_h8 = lb["direction_h8"].values
    valid_mask = (direction_h8 == 0) | (direction_h8 == 1)
    y_dir = np.zeros(len(direction_h8), dtype=np.float32)
    y_dir[direction_h8 == 0] = 1.0

    y_mfe_up = lb[[f"mfe_up_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0
    y_mfe_down = lb[[f"mfe_down_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0
    y_asym = (y_mfe_up - y_mfe_down).astype(np.float32)

    def in_range(lo, hi):
        return (dates >= lo) & (dates <= hi) & valid_mask
    splits = {
        "train": np.where(in_range(*TRAIN_RANGE))[0],
        "val": np.where(in_range(*VAL_RANGE))[0],
        "test": np.where(in_range(*TEST_RANGE))[0],
    }
    print(f"  train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")

    scaler_mean = diff_raw[splits["train"]].mean(axis=0)
    scaler_std = diff_raw[splits["train"]].std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0
    diff_scaled = (diff_raw - scaler_mean) / scaler_std
    diff_scaled = np.nan_to_num(diff_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def get_split(idx):
        return (diff_scaled[idx].reshape(-1, 8, 4),
                y_mfe_up[idx], y_mfe_down[idx], y_dir[idx], y_asym[idx])

    X_tr, ymu_tr, ymd_tr, ydir_tr, yasym_tr = get_split(splits["train"])
    X_va, ymu_va, ymd_va, ydir_va, yasym_va = get_split(splits["val"])
    X_te, ymu_te, ymd_te, ydir_te, yasym_te = get_split(splits["test"])

    # Load 15m data for backtest
    print("Loading 15m data for backtest...")
    df_15m = pd.read_parquet(REPO_ROOT / "data/raw/BTCUSDT_15m_ohlcv.parquet")
    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)

    # Init MLflow
    tracking.init()
    experiment_name = "attention_architecture_ablation"
    mlflow.set_experiment(experiment_name)

    all_results = []

    for variant_cls, loss_fn in VARIANTS:
        variant_name = variant_cls.name
        print(f"\n{'='*60}")
        print(f"VARIANT: {variant_name}")
        print(f"{'='*60}")

        run_results = []

        for run_i in range(N_RUNS):
            seed = 42 + run_i
            np.random.seed(seed)
            torch.manual_seed(seed)

            run_name = f"{variant_name}_seed{seed}"
            print(f"\n  Run {run_i+1}/{N_RUNS} (seed={seed})...")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "variant": variant_name,
                    "seed": seed,
                    "hidden": HIDDEN,
                    "dropout": DROPOUT,
                    "temperature": TEMPERATURE,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "conf_long": CONF_LONG,
                    "conf_short": CONF_SHORT,
                })

                model, epochs, val_loss = train_model(
                    variant_cls, loss_fn,
                    X_tr, ymu_tr, ymd_tr, ydir_tr, yasym_tr,
                    X_va, ymu_va, ymd_va, ydir_va, yasym_va,
                )

                metrics = evaluate_model(model, X_te, ydir_te)
                metrics["epochs"] = epochs
                metrics["val_loss"] = round(val_loss, 4)
                print(f"    acc={metrics['overall_acc']}% | conf_acc={metrics['confident_acc']}% "
                      f"({metrics['n_confident']} signals: {metrics['n_long']}L/{metrics['n_short']}S) | {epochs} epochs")

                # Backtest only the best seed (seed=42) to save time
                bt = {}
                if run_i == 0:
                    print(f"    Running backtest (2024-2025, V2 exits)...")
                    bt = backtest_model(model, diff_scaled, df_15m, scaler_mean, scaler_std)
                    print(f"    BT: {bt['n_trades']}t | {bt['total_bps']:+.0f} bps | "
                          f"PF {bt['pf']} | stop {bt['stop_rate']}% | DD {bt['max_dd']}")

                all_metrics = {**metrics, **{f"bt_{k}": v for k, v in bt.items()}}
                mlflow.log_metrics(all_metrics)

                run_results.append({"seed": seed, **metrics, "backtest": bt})

        all_results.append({"variant": variant_name, "runs": run_results})

    # === SUMMARY ===
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Variant':<14} {'Acc':>6} {'Conf Acc':>10} {'N Signals':>11} {'BT bps':>8} {'BT PF':>6} {'BT Stop%':>9}")
    print("-" * 70)

    for vr in all_results:
        name = vr["variant"]
        accs = [r["overall_acc"] for r in vr["runs"]]
        conf_accs = [r["confident_acc"] for r in vr["runs"]]
        n_confs = [r["n_confident"] for r in vr["runs"]]
        bt = vr["runs"][0].get("backtest", {})
        bt_bps = bt.get("total_bps", "---")
        bt_pf = bt.get("pf", "---")
        bt_stop = bt.get("stop_rate", "---")
        print(f"{name:<14} {np.mean(accs):>5.1f}% {np.mean(conf_accs):>9.1f}% "
              f"{int(np.mean(n_confs)):>10} {bt_bps:>8} {bt_pf:>6} {bt_stop:>8}%")

    # Save results
    results_path = RESULTS_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    print(f"MLflow experiment: {experiment_name}")


if __name__ == "__main__":
    main()
