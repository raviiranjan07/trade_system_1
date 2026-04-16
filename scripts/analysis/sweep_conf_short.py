"""Sweep conf_short threshold for any model. Uses current window from params.yaml.

Usage:
    python scripts/analysis/sweep_conf_short.py --model ml_v1
    python scripts/analysis/sweep_conf_short.py --model ml_v2_attention
    python scripts/analysis/sweep_conf_short.py --model ml_v1 --sweep 0.60,0.62,0.65,0.68,0.70
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine.backtest import run_backtest
from engine.config.loader import load_config
from engine.signals.ml_v1 import MLV1
from engine.signals.direction_attention import DirectionAttention

PARAMS_PATH = REPO_ROOT / "configs/params.yaml"

MODELS = {
    "ml_v1": {
        "registry_name": "ML_V1",
        "staging_dir": REPO_ROOT / "models/ML_V1_staging",
        "generator_class": MLV1,
        "onnx_filename": "direction_model.onnx",
        "scaler_filename": "scaler.npz",
        "short_signal": "ML_SHORT",
        # Default sweep range tuned to MLP's wider distribution
        "default_sweep": [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75],
    },
    "ml_v2_attention": {
        "registry_name": "ML_V2_ATTENTION",
        "staging_dir": REPO_ROOT / "models/ML_V2_ATTENTION_staging",
        "generator_class": DirectionAttention,
        "onnx_filename": "attention_model.onnx",
        "scaler_filename": "scaler.npz",
        "short_signal": "ML_ATTN_SHORT",
        "default_sweep": [0.55, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.65],
    },
}


def set_threshold(model: str, conf_short: float) -> None:
    p = yaml.safe_load(PARAMS_PATH.read_text())
    p[model]["inference"]["conf_short"] = conf_short
    PARAMS_PATH.write_text(yaml.safe_dump(p, sort_keys=False))


def run_one(model_name: str, conf_short: float, short_signal: str) -> dict:
    mc = MODELS[model_name]
    set_threshold(model_name, conf_short)

    p = yaml.safe_load(PARAMS_PATH.read_text())
    cfg = load_config()

    trades = run_backtest(
        cfg,
        start=p["backtest"]["start"],
        end=p["backtest"]["end"],
        ml_model_dir=mc["staging_dir"],
        ml_generator_class=mc["generator_class"],
        ml_onnx_filename=mc["onnx_filename"],
        ml_scaler_filename=mc["scaler_filename"],
    )
    tdf = pd.DataFrame([asdict(t) for t in trades])
    if tdf.empty:
        return {"conf_short": conf_short, "n_trades": 0}

    net = tdf["net_profit_bps"].astype(float)
    wins = net[net > 0]
    losses = net[net <= 0]
    pf = float(wins.sum() / (abs(losses.sum()) or 1e-9))
    equity = net.cumsum()
    max_dd = float((equity - equity.cummax()).min())

    n_short = int((tdf["signal_type"] == short_signal).sum())
    short_net = float(tdf[tdf["signal_type"] == short_signal]["net_profit_bps"].sum())

    return {
        "conf_short": conf_short,
        "n_trades": int(len(tdf)),
        "win_pct": round(float((net > 0).mean()) * 100, 1),
        "total_bps": round(float(net.sum()), 0),
        "pf": round(pf, 2),
        "max_dd_bps": round(max_dd, 0),
        "avg_bps": round(float(net.mean()), 1),
        "n_model_shorts": n_short,
        "model_short_bps": round(short_net, 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--sweep", default=None,
                        help="Comma-separated thresholds (default: model-specific range)")
    args = parser.parse_args()

    mc = MODELS[args.model]
    sweep = mc["default_sweep"] if args.sweep is None else [float(x) for x in args.sweep.split(",")]

    original = yaml.safe_load(PARAMS_PATH.read_text())[args.model]["inference"]["conf_short"]
    print(f"Model: {args.model} (SHORT signal: {mc['short_signal']})")
    print(f"Original conf_short: {original}")
    p = yaml.safe_load(PARAMS_PATH.read_text())
    print(f"Window: {p['backtest']['start']} .. {p['backtest']['end']}")

    results = []
    for cs in sweep:
        print(f"\nRunning conf_short={cs}...")
        r = run_one(args.model, cs, mc["short_signal"])
        results.append(r)
        print(f"  {r['n_trades']} trades, {r['win_pct']}% win, {r['total_bps']:+.0f} bps, "
              f"PF {r['pf']}, DD {r['max_dd_bps']:+.0f}")

    set_threshold(args.model, original)
    print(f"\nRestored conf_short = {original}")

    out = REPO_ROOT / f"data/reports/{args.model}_conf_short_sweep.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'conf':>6} {'trades':>7} {'win%':>6} {'total':>9} {'PF':>6} {'DD':>8} "
          f"{'shorts':>8} {'short_bps':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['conf_short']:>6.2f} {r['n_trades']:>7} {r['win_pct']:>5.1f}% "
              f"{r['total_bps']:>+9.0f} {r['pf']:>6.2f} {r['max_dd_bps']:>+8.0f} "
              f"{r['n_model_shorts']:>8} {r['model_short_bps']:>+10.0f}")
    print(f"\nSaved: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
