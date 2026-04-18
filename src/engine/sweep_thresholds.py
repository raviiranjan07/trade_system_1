"""Level 1: Threshold Sweep — find optimal conf_long / conf_short.

Model-agnostic: reads sweep ranges from configs/params.yaml → {model}.sweep.
No retraining needed — loads existing model and re-runs backtest at each threshold.

Run: PYTHONPATH=src python -m engine.sweep_thresholds --model ml_v3
"""

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtest import run_backtest, ML_GENERATORS
from .config.loader import load_config

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "configs/params.yaml"


def backtest_at_threshold(config, model_name, conf_long, conf_short, start, end):
    """Run backtest with specific thresholds by temporarily patching params."""
    gen_class, model_dir = ML_GENERATORS[model_name]

    onnx_map = {
        "ml_v3": "v3_model.onnx",
        "ml_v2_attention": "attention_model.onnx",
        "ml_v1": "direction_model.onnx",
    }
    onnx_name = onnx_map.get(model_name, "direction_model.onnx")

    # Patch the params.yaml temporarily for this threshold
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)
    original_long = params[model_name]["inference"]["conf_long"]
    original_short = params[model_name]["inference"]["conf_short"]

    try:
        params[model_name]["inference"]["conf_long"] = conf_long
        params[model_name]["inference"]["conf_short"] = conf_short
        with open(PARAMS_PATH, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        trades = run_backtest(
            config, start=start, end=end,
            ml_model_dir=model_dir, ml_generator_class=gen_class,
            ml_onnx_filename=onnx_name,
            ml_only=True,
        )
    finally:
        # Restore original thresholds
        params[model_name]["inference"]["conf_long"] = original_long
        params[model_name]["inference"]["conf_short"] = original_short
        with open(PARAMS_PATH, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    return trades


def _signal_prefix(model_name: str) -> str:
    """Map model name to signal type prefix."""
    return {"ml_v3": "ML_V3_", "ml_v2_attention": "ML_ATTN_", "ml_v1": "ML_"}.get(model_name, "ML_")


def _filter_model_signals(tdf, model_name):
    """Filter trades to only this model's signals."""
    prefix = _signal_prefix(model_name)
    if model_name == "ml_v1":
        return tdf[tdf["signal_type"].str.startswith(prefix) &
                    ~tdf["signal_type"].str.startswith("ML_ATTN_") &
                    ~tdf["signal_type"].str.startswith("ML_V3_")]
    return tdf[tdf["signal_type"].str.startswith(prefix)]


def _compute_metrics(df):
    """Compute trading metrics for a dataframe of trades."""
    if len(df) == 0:
        return {"n": 0, "bps": 0, "pf": 0, "win_pct": 0, "stop_pct": 0, "dd": 0, "avg": 0}
    wins = df[df["net_profit_bps"] > 0]
    losses = df[df["net_profit_bps"] <= 0]
    gw = wins["net_profit_bps"].sum() if len(wins) else 0
    gl = abs(losses["net_profit_bps"].sum()) if len(losses) else 1
    stops = df[df["exit_reason"] == "STOP_LOSS"]
    eq = df["net_profit_bps"].cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0
    return {
        "n": len(df),
        "bps": round(float(df["net_profit_bps"].sum()), 1),
        "pf": round(gw / gl, 3) if gl > 0 else 0,
        "win_pct": round(len(wins) / len(df) * 100, 1),
        "stop_pct": round(len(stops) / len(df) * 100, 1),
        "dd": round(dd, 1),
        "avg": round(float(df["net_profit_bps"].mean()), 1),
    }


def evaluate_trades(trades, model_name):
    """Compute metrics filtered to model's signals, with LONG/SHORT split."""
    if not trades:
        return {"all": _compute_metrics(pd.DataFrame()), "long": _compute_metrics(pd.DataFrame()), "short": _compute_metrics(pd.DataFrame())}

    tdf = pd.DataFrame([asdict(t) for t in trades])
    ml = _filter_model_signals(tdf, model_name)
    ml_long = ml[ml["direction"] == "LONG"]
    ml_short = ml[ml["direction"] == "SHORT"]

    return {
        "all": _compute_metrics(ml),
        "long": _compute_metrics(ml_long),
        "short": _compute_metrics(ml_short),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ML_GENERATORS.keys()))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    t0 = time.time()

    model_name = args.model
    logger.info("Threshold sweep for: %s", model_name)

    # Load sweep config
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    model_cfg = params[model_name]
    sweep_cfg = model_cfg.get("sweep", {})
    split_cfg = model_cfg.get("split", {})

    long_values = sweep_cfg.get("conf_long", [0.40])
    short_values = sweep_cfg.get("conf_short", [0.40])
    strategy = sweep_cfg.get("strategy", "independent")

    val_start = split_cfg.get("val_start", "2024-01-01")
    val_end = split_cfg.get("val_end", "2024-12-31")
    test_start = split_cfg.get("test_start", "2025-01-01")
    test_end = split_cfg.get("test_end", "2025-12-31")

    current_long = model_cfg["inference"]["conf_long"]
    current_short = model_cfg["inference"]["conf_short"]

    logger.info("Sweep values — conf_long: %s | conf_short: %s | strategy: %s",
                long_values, short_values, strategy)
    logger.info("Val: %s to %s | Test: %s to %s", val_start, val_end, test_start, test_end)

    config = load_config()
    results = []

    # Init MLflow for sweep tracking
    try:
        import mlflow
        from mlops import tracking as _tracking
        _tracking.init()
        mlflow_experiment = f"{model_name}_threshold_sweep"
        mlflow.set_experiment(mlflow_experiment)
        _use_mlflow = True
        logger.info("MLflow experiment: %s", mlflow_experiment)
    except Exception as e:
        logger.warning("MLflow not available: %s", e)
        _use_mlflow = False

    def _log_to_mlflow(cl, cs, metrics, period):
        if not _use_mlflow:
            return
        try:
            run_name = f"{period}_cl{cl:.2f}_cs{cs:.2f}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({"model": model_name, "period": period,
                                  "conf_long": str(cl), "conf_short": str(cs)})
                for split in ["all", "long", "short"]:
                    for k, v in metrics[split].items():
                        mlflow.log_metric(f"{period}_{split}_{k}", v)
        except Exception:
            pass

    if strategy == "grid":
        combos = [(l, s) for l in long_values for s in short_values]
    else:
        # Independent: sweep long first, then short with best long
        combos = [(l, current_short) for l in long_values]

    def _log_metrics(cl, cs, m):
        a, l, s = m["all"], m["long"], m["short"]
        logger.info("    ALL:   %4d trades | %+7.0f bps | PF %5.2f | DD %+6.0f | stop %5.1f%%",
                    a["n"], a["bps"], a["pf"], a["dd"], a["stop_pct"])
        logger.info("    LONG:  %4d trades | %+7.0f bps | PF %5.2f | DD %+6.0f",
                    l["n"], l["bps"], l["pf"], l["dd"])
        logger.info("    SHORT: %4d trades | %+7.0f bps | PF %5.2f | DD %+6.0f",
                    s["n"], s["bps"], s["pf"], s["dd"])

    # Phase 1: Sweep on VAL
    logger.info("\n=== PHASE 1: Sweep on validation (%s to %s) ===", val_start, val_end)

    for i, (cl, cs) in enumerate(combos):
        logger.info("  [%d/%d] conf_long=%.2f conf_short=%.2f", i + 1, len(combos), cl, cs)
        trades = backtest_at_threshold(config, model_name, cl, cs, val_start, val_end)
        metrics = evaluate_trades(trades, model_name)
        result = {"conf_long": cl, "conf_short": cs, "period": "val", **metrics}
        results.append(result)
        _log_metrics(cl, cs, metrics)
        _log_to_mlflow(cl, cs, metrics, "val")

    # If independent: find best long, then sweep short
    if strategy == "independent":
        val_results = [r for r in results if r["all"]["n"] >= 100]
        if val_results:
            best_long = max(val_results, key=lambda r: r["all"]["pf"])["conf_long"]
        else:
            best_long = current_long
            logger.warning("No config with >= 100 trades on val. Using current conf_long=%.2f", best_long)

        logger.info("\n  Best conf_long from Phase 1: %.2f", best_long)
        logger.info("  Now sweeping conf_short with conf_long=%.2f ...", best_long)

        for cs in short_values:
            if cs == current_short and best_long == current_long:
                continue  # already tested
            logger.info("  conf_long=%.2f conf_short=%.2f", best_long, cs)
            trades = backtest_at_threshold(config, model_name, best_long, cs, val_start, val_end)
            metrics = evaluate_trades(trades, model_name)
            result = {"conf_long": best_long, "conf_short": cs, "period": "val", **metrics}
            results.append(result)
            _log_metrics(best_long, cs, metrics)
            _log_to_mlflow(best_long, cs, metrics, "val")

    # Rank by PF (min 100 trades)
    valid = [r for r in results if r["all"]["n"] >= 100]
    valid.sort(key=lambda r: r["all"]["pf"], reverse=True)

    logger.info("\n=== VAL RESULTS (ranked by PF, min 100 trades) ===")
    logger.info("  %-10s %-10s %5s %5s %8s %8s %6s %6s %7s %7s", "conf_long", "conf_short",
                "L_n", "S_n", "L_bps", "S_bps", "L_PF", "S_PF", "All_PF", "All_DD")
    logger.info("  " + "-" * 80)
    for r in valid[:10]:
        a, l, s = r["all"], r["long"], r["short"]
        logger.info("  %-10.2f %-10.2f %5d %5d %+8.0f %+8.0f %6.2f %6.2f %7.2f %+7.0f",
                    r["conf_long"], r["conf_short"],
                    l["n"], s["n"], l["bps"], s["bps"], l["pf"], s["pf"], a["pf"], a["dd"])

    # Phase 2: Confirm top 3 on TEST
    top3 = valid[:3]
    if not top3:
        logger.error("No valid configs found on val!")
        return

    logger.info("\n=== PHASE 2: Confirm top 3 on test (%s to %s) ===", test_start, test_end)

    test_results = []
    for r in top3:
        cl, cs = r["conf_long"], r["conf_short"]
        logger.info("  conf_long=%.2f conf_short=%.2f", cl, cs)
        trades = backtest_at_threshold(config, model_name, cl, cs, test_start, test_end)
        metrics = evaluate_trades(trades, model_name)
        test_result = {"conf_long": cl, "conf_short": cs, "period": "test",
                       "val_pf": r["all"]["pf"], "val_bps": r["all"]["bps"], **metrics}
        test_results.append(test_result)
        _log_metrics(cl, cs, metrics)
        _log_to_mlflow(cl, cs, metrics, "test")

    # Pick best on test (that also had good val)
    test_valid = [r for r in test_results if r["all"]["n"] >= 50]
    if test_valid:
        best = max(test_valid, key=lambda r: r["all"]["pf"])
    else:
        best = test_results[0] if test_results else top3[0]

    a, l, s = best["all"], best["long"], best["short"]
    logger.info("\n=== WINNER ===")
    logger.info("  conf_long=%.2f  conf_short=%.2f", best["conf_long"], best["conf_short"])
    logger.info("  ALL:   %d trades | %+.0f bps | PF %.2f | DD %+.0f", a["n"], a["bps"], a["pf"], a["dd"])
    logger.info("  LONG:  %d trades | %+.0f bps | PF %.2f | DD %+.0f", l["n"], l["bps"], l["pf"], l["dd"])
    logger.info("  SHORT: %d trades | %+.0f bps | PF %.2f | DD %+.0f", s["n"], s["bps"], s["pf"], s["dd"])

    # Save report
    report = {
        "model": model_name,
        "strategy": strategy,
        "val_period": f"{val_start} to {val_end}",
        "test_period": f"{test_start} to {test_end}",
        "winner": {
            "conf_long": best["conf_long"],
            "conf_short": best["conf_short"],
            "test_all": a, "test_long": l, "test_short": s,
            "val_pf": best.get("val_pf", 0), "val_bps": best.get("val_bps", 0),
        },
        "all_val_results": [r for r in results if r["period"] == "val"],
        "top3_test_results": test_results,
        "runtime_sec": round(time.time() - t0, 1),
    }

    output_path = REPO_ROOT / "data" / "reports" / model_name / "threshold_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\nSaved to %s", output_path)
    logger.info("Total time: %.1f min", (time.time() - t0) / 60)
    logger.info("\nTo apply: update configs/params.yaml → %s.inference.conf_long=%.2f conf_short=%.2f",
                model_name, best["conf_long"], best["conf_short"])


if __name__ == "__main__":
    main()
