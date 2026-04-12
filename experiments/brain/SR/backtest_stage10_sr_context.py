"""Stage 10 backtest: compare base-only vs base+SR-context decision variants.

This script trains base-only and base+SR models on the filtered subset and
backtests predictions on the test split with simple per-trade cost assumptions.

Usage:
  PYTHONPATH=src python experiments/brain/SR/backtest_stage10_sr_context.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data/archive/brain_datasets")
META_PATH = DATA_DIR / "metadata.json"
SCALER_PATH = DATA_DIR / "scaler.npz"


def load_split(name: str):
    d = np.load(DATA_DIR / f"{name}.npz")
    x = d["X"].astype(np.float32)
    y = d["Y"][:, 0].astype(np.int64)  # 0=LONG, 1=SHORT
    return x, y


def normalize_seq(train_x, val_x, test_x):
    m = train_x.mean(axis=(0, 1), keepdims=True)
    s = train_x.std(axis=(0, 1), keepdims=True)
    s[s == 0] = 1.0
    return (
        np.nan_to_num((train_x - m) / s),
        np.nan_to_num((val_x - m) / s),
        np.nan_to_num((test_x - m) / s),
    )


def parse_csv_cols(value: str):
    return [c.strip() for c in value.split(",") if c.strip()]


def filter_min_zone_width_bps(x, y, mean, std, sh_idx, rl_idx, min_bps):
    sh = x[:, -1, sh_idx] * std[sh_idx] + mean[sh_idx]
    rl = x[:, -1, rl_idx] * std[rl_idx] + mean[rl_idx]
    valid = (sh > 0) & (rl > sh)
    width_bps = np.zeros_like(sh)
    width_bps[valid] = (rl[valid] - sh[valid]) / sh[valid] * 10000.0
    keep = valid & (width_bps >= min_bps)
    return x[keep], y[keep]


class BaseOnlyLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2),
        )

    def forward(self, xb):
        _, (h, _) = self.lstm(xb)
        return self.head(h[-1])


class BaseSrFusion(nn.Module):
    def __init__(self, base_size, sr_size):
        super().__init__()
        self.base_lstm = nn.LSTM(input_size=base_size, hidden_size=32, batch_first=True)
        self.base_proj = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.sr_proj = nn.Sequential(nn.Linear(sr_size, 12), nn.ReLU(), nn.Linear(12, 8), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2),
        )

    def forward(self, xb, xsr):
        _, (h, _) = self.base_lstm(xb)
        b = self.base_proj(h[-1])
        s = self.sr_proj(xsr)
        return self.head(torch.cat([b, s], dim=1))


def train_base(xtr, ytr, xva, yva, epochs=120, patience=15):
    model = BaseOnlyLSTM(xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(xtr), torch.LongTensor(ytr)), batch_size=256, shuffle=True)
    va_x = torch.FloatTensor(xva)
    va_y = torch.LongTensor(yva)
    best = None
    best_loss = float("inf")
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = ce(model(va_x), va_y).item()
        if vl < best_loss:
            best_loss = vl
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return model


def train_fusion(xtr_b, xtr_sr, ytr, xva_b, xva_sr, yva, epochs=120, patience=15):
    model = BaseSrFusion(xtr_b.shape[2], xtr_sr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(xtr_b), torch.FloatTensor(xtr_sr), torch.LongTensor(ytr)),
        batch_size=256,
        shuffle=True,
    )
    va_b = torch.FloatTensor(xva_b)
    va_sr = torch.FloatTensor(xva_sr)
    va_y = torch.LongTensor(yva)
    best = None
    best_loss = float("inf")
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, xsr, yb in loader:
            opt.zero_grad()
            loss = ce(model(xb, xsr), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = ce(model(va_b, va_sr), va_y).item()
        if vl < best_loss:
            best_loss = vl
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    return model


def apply_second_touch_gate(logits, support_retest, resistance_retest, dist_support, dist_resistance, near_th, ret_th, strength):
    out = logits.clone()
    long_mask = (dist_support <= near_th) & (support_retest >= ret_th)
    short_mask = (dist_resistance <= near_th) & (resistance_retest >= ret_th)
    long_mask = torch.from_numpy(long_mask.astype(np.bool_))
    short_mask = torch.from_numpy(short_mask.astype(np.bool_))
    out[long_mask, 0] += strength
    out[long_mask, 1] -= strength
    out[short_mask, 1] += strength
    out[short_mask, 0] -= strength
    return out, int(long_mask.sum().item()), int(short_mask.sum().item())


def backtest_from_logits(logits, y_true, fee_bps=6.0, slippage_bps=2.0, conf_threshold=0.0):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    y = torch.LongTensor(y_true)

    take = conf >= conf_threshold
    pred = pred[take]
    y = y[take]
    if len(y) == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_bps": 0.0,
            "net_bps": 0.0,
            "coverage_pct": 0.0,
        }

    # +1 if correct direction, -1 if wrong direction.
    gross = torch.where(pred == y, torch.tensor(1.0), torch.tensor(-1.0)) * 100.0
    cost = fee_bps + slippage_bps
    net = gross - cost

    win_rate = float((net > 0).float().mean().item() * 100)
    avg_bps = float(net.mean().item())
    net_bps = float(net.sum().item())
    coverage = float(take.float().mean().item() * 100)
    return {
        "trades": int(len(net)),
        "win_rate": win_rate,
        "avg_bps": avg_bps,
        "net_bps": net_bps,
        "coverage_pct": coverage,
    }


def print_backtest(name, res):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Trades: {res['trades']}")
    print(f"Coverage: {res['coverage_pct']:.2f}%")
    print(f"Win rate: {res['win_rate']:.2f}%")
    print(f"Avg bps/trade: {res['avg_bps']:.2f}")
    print(f"Net bps: {res['net_bps']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-zone-bps", type=float, default=15.0)
    parser.add_argument("--fee-bps", type=float, default=6.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--second-touch-gate", action="store_true")
    parser.add_argument("--gate-near-threshold", type=float, default=0.0)
    parser.add_argument("--gate-retest-threshold", type=float, default=0.5)
    parser.add_argument("--gate-strength", type=float, default=0.35)
    parser.add_argument(
        "--base-cols",
        type=str,
        default="roc,rsi7,range_position,volume_ratio",
        help="Comma-separated base feature columns from data/archive/brain_datasets/metadata.json",
    )
    args = parser.parse_args()

    with open(META_PATH) as f:
        meta = json.load(f)
    names = meta["feature_names"]
    idx = {n: i for i, n in enumerate(names)}
    scaler = np.load(SCALER_PATH)
    mean = scaler["mean"]
    std = scaler["std"]

    base_cols = parse_csv_cols(args.base_cols)
    missing_base = [c for c in base_cols if c not in idx]
    if missing_base:
        raise ValueError(f"Missing base feature(s): {missing_base}")
    sr_cols = [
        "support_range_low",
        "support_range_high",
        "resistance_range_low",
        "resistance_range_high",
        "zone_width",
        "support_retest",
        "resistance_retest",
    ]
    gate_cols = ["support_retest", "resistance_retest", "distance_to_support", "distance_to_resistance"]
    base_idx = [idx[c] for c in base_cols]
    sr_idx = [idx[c] for c in sr_cols]
    gate_idx = [idx[c] for c in gate_cols]

    print("=" * 70)
    print("STAGE 10 BACKTEST: BASE VS BASE+SR")
    print("=" * 70)
    print("Base cols:", base_cols)
    print("SR cols:", sr_cols)

    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xte, yte = load_split("test")

    sh_idx = idx["support_range_high"]
    rl_idx = idx["resistance_range_low"]
    xtr, ytr = filter_min_zone_width_bps(xtr, ytr, mean, std, sh_idx, rl_idx, args.min_zone_bps)
    xva, yva = filter_min_zone_width_bps(xva, yva, mean, std, sh_idx, rl_idx, args.min_zone_bps)
    xte, yte = filter_min_zone_width_bps(xte, yte, mean, std, sh_idx, rl_idx, args.min_zone_bps)
    print(f"Filtered rows -> Train: {len(ytr)}, Val: {len(yva)}, Test: {len(yte)}")

    xtr_b = xtr[:, :, base_idx]
    xva_b = xva[:, :, base_idx]
    xte_b = xte[:, :, base_idx]
    xtr_sr = xtr[:, -1, sr_idx]
    xva_sr = xva[:, -1, sr_idx]
    xte_sr = xte[:, -1, sr_idx]
    xte_gate = xte[:, -1, gate_idx]

    xtr_b, xva_b, xte_b = normalize_seq(xtr_b, xva_b, xte_b)
    xtr_sr, xva_sr, xte_sr = normalize_seq(xtr_sr[:, None, :], xva_sr[:, None, :], xte_sr[:, None, :])
    xtr_sr = xtr_sr[:, 0, :]
    xva_sr = xva_sr[:, 0, :]
    xte_sr = xte_sr[:, 0, :]

    base_model = train_base(xtr_b, ytr, xva_b, yva)
    fuse_model = train_fusion(xtr_b, xtr_sr, ytr, xva_b, xva_sr, yva)

    base_model.eval()
    fuse_model.eval()
    with torch.no_grad():
        logits_base = base_model(torch.FloatTensor(xte_b))
        logits_fuse = fuse_model(torch.FloatTensor(xte_b), torch.FloatTensor(xte_sr))

    if args.second_touch_gate:
        logits_base, lg1, sg1 = apply_second_touch_gate(
            logits_base,
            xte_gate[:, 0],
            xte_gate[:, 1],
            xte_gate[:, 2],
            xte_gate[:, 3],
            args.gate_near_threshold,
            args.gate_retest_threshold,
            args.gate_strength,
        )
        logits_fuse, lg2, sg2 = apply_second_touch_gate(
            logits_fuse,
            xte_gate[:, 0],
            xte_gate[:, 1],
            xte_gate[:, 2],
            xte_gate[:, 3],
            args.gate_near_threshold,
            args.gate_retest_threshold,
            args.gate_strength,
        )
        print(f"Gate counts (base): long={lg1}, short={sg1}")
        print(f"Gate counts (fuse): long={lg2}, short={sg2}")

    res_base = backtest_from_logits(logits_base, yte, args.fee_bps, args.slippage_bps, args.conf_threshold)
    res_fuse = backtest_from_logits(logits_fuse, yte, args.fee_bps, args.slippage_bps, args.conf_threshold)
    print_backtest("BASE ONLY BACKTEST", res_base)
    print_backtest("BASE + SR BACKTEST", res_fuse)

    delta = res_fuse["net_bps"] - res_base["net_bps"]
    print("\nDelta net bps (Base+SR - Base): %.2f" % delta)


if __name__ == "__main__":
    main()
