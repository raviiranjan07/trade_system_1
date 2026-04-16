"""Sweep conf_short threshold for ML_V2_ATTENTION, current window from params.yaml.

Saves results to data/reports/attention_conf_short_sweep.json and prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from engine.backtest import run_backtest
from engine.config.loader import load_config
from engine.signals.direction_attention import DirectionAttention

PARAMS_PATH = REPO_ROOT / "configs/params.yaml"
STAGING_DIR = REPO_ROOT / "models/ML_V2_ATTENTION_staging"


def set_threshold(conf_short: float) -> None:
    p = yaml.safe_load(PARAMS_PATH.read_text())
    p["ml_v2_attention"]["inference"]["conf_short"] = conf_short
    PARAMS_PATH.write_text(yaml.safe_dump(p, sort_keys=False))


def run_one(conf_short: float) -> dict:
    from dataclasses import asdict
    import pandas as pd

    set_threshold(conf_short)
    cfg = load_config()
    trades = run_backtest(
        cfg,
        start=yaml.safe_load(PARAMS_PATH.read_text())["backtest"]["start"],
        end=yaml.safe_load(PARAMS_PATH.read_text())["backtest"]["end"],
        ml_model_dir=STAGING_DIR,
        ml_generator_class=DirectionAttention,
        ml_onnx_filename="attention_model.onnx",
        ml_scaler_filename="scaler.npz",
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

    n_short = int((tdf["signal_type"] == "ML_ATTN_SHORT").sum())
    short_net = float(tdf[tdf["signal_type"] == "ML_ATTN_SHORT"]["net_profit_bps"].sum())

    return {
        "conf_short": conf_short,
        "n_trades": int(len(tdf)),
        "win_pct": round(float((net > 0).mean()) * 100, 1),
        "total_bps": round(float(net.sum()), 0),
        "pf": round(pf, 2),
        "max_dd_bps": round(max_dd, 0),
        "avg_bps": round(float(net.mean()), 1),
        "n_attn_short": n_short,
        "attn_short_bps": round(short_net, 0),
    }


def main() -> None:
    original = yaml.safe_load(PARAMS_PATH.read_text())["ml_v2_attention"]["inference"]["conf_short"]
    print(f"Original conf_short: {original}")

    sweep = [0.55, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.65]
    results = []
    for cs in sweep:
        print(f"\nRunning conf_short={cs}...")
        r = run_one(cs)
        results.append(r)
        print(f"  {r['n_trades']} trades, {r['win_pct']}% win, {r['total_bps']:+.0f} bps, PF {r['pf']}, DD {r['max_dd_bps']:+.0f}")

    # Restore original
    set_threshold(original)
    print(f"\nRestored conf_short = {original}")

    out = REPO_ROOT / "data/reports/attention_conf_short_sweep.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'conf':>6} {'trades':>7} {'win%':>6} {'total':>9} {'PF':>6} {'DD':>8} {'attn_S':>8} {'attn_S_bps':>11}")
    print("-" * 70)
    for r in results:
        print(f"{r['conf_short']:>6.2f} {r['n_trades']:>7} {r['win_pct']:>5.1f}% "
              f"{r['total_bps']:>+9.0f} {r['pf']:>6.2f} {r['max_dd_bps']:>+8.0f} "
              f"{r['n_attn_short']:>8} {r['attn_short_bps']:>+11.0f}")

    print(f"\nSaved: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
