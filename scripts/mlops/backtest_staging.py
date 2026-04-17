"""Backtest the @staging model on 2025 OOS and write a metrics report.

Does not compare against any other model. Just a clean, honest backtest
of whatever ML_V1 @staging currently points to. Human reads the numbers
and decides whether to promote.

Output: data/reports/backtest_staging.json

Run:
    python scripts/mlops/backtest_staging.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

import mlflow
import yaml

from engine.backtest import run_backtest
from engine.config.loader import load_config
from engine.signals.ml_v1 import MLV1
from engine.signals.direction_attention import DirectionAttention

with open(REPO_ROOT / "configs/params.yaml") as _f:
    _p = yaml.safe_load(_f)
BACKTEST_START = _p["backtest"]["start"]
BACKTEST_END = _p["backtest"]["end"]

# Per-model config: staging dir + signal generator class + onnx filename
MODELS = {
    "ml_v1": {
        "registry_name": "ML_V1",
        "staging_dir": REPO_ROOT / "models/ML_V1_staging",
        "generator_class": MLV1,
        "onnx_filename": "direction_model.onnx",
        "scaler_filename": "scaler.npz",
    },
    "ml_v2_attention": {
        "registry_name": "ML_V2_ATTENTION",
        "staging_dir": REPO_ROOT / "models/ML_V2_ATTENTION_staging",
        "generator_class": DirectionAttention,
        "onnx_filename": "attention_model.onnx",
        "scaler_filename": "scaler.npz",
    },
}


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0, "total_bps": 0.0, "win_pct": 0.0, "pf": 0.0, "max_dd_bps": 0.0}

    tdf = pd.DataFrame([asdict(t) for t in trades])
    net = tdf["net_profit_bps"].astype(float)

    wins = net[net > 0]
    losses = net[net <= 0]
    gross_loss = abs(losses.sum()) or 1e-9
    pf = float(wins.sum() / gross_loss)

    equity = net.cumsum()
    max_dd = float((equity - equity.cummax()).min())

    # Breakdown by signal type
    by_signal = {}
    for st in sorted(tdf["signal_type"].unique()):
        s = tdf[tdf["signal_type"] == st]
        by_signal[str(st)] = {
            "n": int(len(s)),
            "win_pct": round(float((s["net_profit_bps"] > 0).mean()) * 100, 1),
            "total_bps": round(float(s["net_profit_bps"].sum()), 1),
        }

    # Breakdown by exit reason (PT_TARGET, STOP_LOSS, etc.) — tells us WHICH
    # exit rules make or lose money.
    by_exit = {}
    if "exit_reason" in tdf.columns:
        for er in sorted(tdf["exit_reason"].unique()):
            s = tdf[tdf["exit_reason"] == er]
            by_exit[str(er)] = {
                "n": int(len(s)),
                "win_pct": round(float((s["net_profit_bps"] > 0).mean()) * 100, 1),
                "total_bps": round(float(s["net_profit_bps"].sum()), 1),
                "avg_bps": round(float(s["net_profit_bps"].mean()), 2),
            }

    return {
        "n_trades": int(len(tdf)),
        "total_bps": round(float(net.sum()), 1),
        "win_pct": round(float((net > 0).mean()) * 100, 1),
        "pf": round(pf, 2),
        "max_dd_bps": round(max_dd, 1),
        "avg_trade_bps": round(float(net.mean()), 2),
        "by_signal_type": by_signal,
        "by_exit_reason": by_exit,
    }


def staging_version(registry_name: str) -> int | None:
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    try:
        mv = mlflow.MlflowClient().get_model_version_by_alias(registry_name, "staging")
        return int(mv.version)
    except mlflow.exceptions.MlflowException:
        return None


def _run_one(model_name: str, exit_version: str) -> dict:
    """Run a single backtest for (model, exit) combo. Returns summary dict."""
    mc = MODELS[model_name]
    staging_dir = mc["staging_dir"]
    suffix = f"{model_name}_exit{exit_version}"
    report_path = REPO_ROOT / f"data/reports/backtest_staging_{suffix}.json"
    trades_path = REPO_ROOT / f"data/reports/backtest_staging_trades_{suffix}.parquet"

    # Temporarily override exit version via env var read by position_manager,
    # OR rewrite params.yaml momentarily. We take the clean path: pass via
    # config object since backtest ultimately creates V12PositionManager from cfg.
    cfg = load_config()
    version = staging_version(mc["registry_name"])

    print(f"\n=== {mc['registry_name']} v{version} @staging | exit={exit_version} | {BACKTEST_START}..{BACKTEST_END} ===")

    if not (staging_dir / mc["onnx_filename"]).exists():
        raise RuntimeError(f"Staging model not found at {staging_dir}.")

    trades = run_backtest(
        cfg,
        start=BACKTEST_START,
        end=BACKTEST_END,
        ml_model_dir=staging_dir,
        ml_generator_class=mc["generator_class"],
        ml_onnx_filename=mc["onnx_filename"],
        ml_scaler_filename=mc["scaler_filename"],
        exit_version=exit_version,
    )

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(trades_path)

    metrics = compute_metrics(trades)

    print(
        f"{metrics['n_trades']} trades | {metrics['win_pct']}% win | "
        f"{metrics['total_bps']:+.0f} bps | PF {metrics['pf']} | DD {metrics['max_dd_bps']:+.0f}"
    )

    report = {
        "window": {"start": BACKTEST_START, "end": BACKTEST_END},
        "model": {"name": mc["registry_name"], "version": version, "alias": "staging"},
        "exit_version": exit_version,
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return {"model": model_name, "exit": exit_version, "report": str(report_path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ml_v1",
                        help="Model name from MODELS registry, or 'all' for every model, "
                             "or comma-separated list (e.g. 'ml_v1,ml_v2_attention')")
    parser.add_argument("--exit", default=None,
                        help="Exit version to use (e.g. v1, v2). Default: uses exit.version from params.yaml. "
                             "Pass 'all' for every version or comma-separated list (e.g. 'v1,v2').")
    args = parser.parse_args()

    # Resolve models list
    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [m.strip() for m in args.model.split(",")]
        for m in models:
            if m not in MODELS:
                raise ValueError(f"Unknown model: {m}. Available: {list(MODELS.keys())}")

    # Resolve exit versions list (V2 removed — V1 is the only exit plan)
    if args.exit is None:
        exits = ["v1"]
    elif args.exit == "all":
        exits = ["v1", "v2"]
    else:
        exits = [e.strip() for e in args.exit.split(",")]

    print(f"Backtest window: {BACKTEST_START} .. {BACKTEST_END}")
    print(f"Grid: models={models} x exits={exits}  ({len(models) * len(exits)} run(s))")

    results = []
    for m in models:
        for e in exits:
            results.append(_run_one(m, e))

    # Summary (always, even for single runs)
    print("\n" + "=" * 70)
    print(f"{'model':<20} {'exit':<6} {'trades':>7} {'win%':>6} {'total_bps':>10} {'PF':>6} {'DD':>8}")
    print("-" * 70)
    for r in results:
        m = r["metrics"]
        print(f"{r['model']:<20} {r['exit']:<6} {m['n_trades']:>7} {m['win_pct']:>5.1f}% "
              f"{m['total_bps']:>+10.0f} {m['pf']:>6.2f} {m['max_dd_bps']:>+8.0f}")

    # Write summary report
    summary_path = REPO_ROOT / "data/reports/backtest_matrix_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSummary: {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
