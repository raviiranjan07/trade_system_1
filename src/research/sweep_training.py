"""Level 2/3: Training Hyperparameter Sweep — grid search with retrain.

Model-agnostic: reads sweep ranges from configs/params.yaml → {model}.sweep.
For each combo, patches params.yaml → retrains → backtests on val → logs to MLflow.
Top configs confirmed on test.

Unlike sweep_thresholds.py (Level 1, no retrain), this script retrains the model
for each combination — much slower but covers training params and architecture.

Levels:
  Level 2: training params (lr, batch_size, loss_weight_pnl, loss_weight_dir)
  Level 3: architecture (hidden_size, dropout, temperature)

Run:
  PYTHONPATH=src python -m research.sweep_training --model ml_v3 --level 2
  PYTHONPATH=src python -m research.sweep_training --model ml_v3 --level 3
  PYTHONPATH=src python -m research.sweep_training --model ml_v3 --level 2 --level 3
"""

import argparse
import itertools
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .backtest import run_backtest, ML_GENERATORS
from engine.config.loader import load_config
from .sweep_thresholds import evaluate_trades

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "configs/params.yaml"

# Which params belong to which level
LEVEL_2_PARAMS = ["learning_rate", "batch_size", "loss_weight_pnl", "loss_weight_dir"]
LEVEL_3_PARAMS = ["hidden_size", "dropout", "temperature"]

# Map sweep param names to params.yaml training section keys
PARAM_KEY_MAP = {
    "learning_rate": "lr",
    "batch_size": "batch_size",
    "loss_weight_pnl": "loss_weight_pnl",
    "loss_weight_dir": "loss_weight_dir",
    "hidden_size": "hidden",
    "dropout": "dropout",
    "temperature": "temperature",
}

# Training commands per model (subprocess)
TRAIN_COMMANDS = {
    "ml_v1": [sys.executable, "src/training/ml_train.py"],
    "ml_v3": [sys.executable, "-m", "training.train_v3"],
}


def _load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def _save_params(params):
    with open(PARAMS_PATH, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)


def _build_grid(sweep_cfg, param_names):
    """Build grid of param combos from sweep config."""
    axes = []
    names = []
    for name in param_names:
        values = sweep_cfg.get(name)
        if values and isinstance(values, list) and len(values) > 1:
            axes.append(values)
            names.append(name)

    if not axes:
        return [], []

    combos = list(itertools.product(*axes))
    return names, combos


def _patch_training_params(params, model_name, param_names, combo):
    """Patch params.yaml training section with sweep values."""
    training = params[model_name]["training"]
    for name, value in zip(param_names, combo):
        key = PARAM_KEY_MAP.get(name, name)
        training[key] = value


def _train_model(model_name, env_vars=None):
    """Run training script as subprocess. Returns (success, elapsed_sec)."""
    cmd = TRAIN_COMMANDS.get(model_name)
    if not cmd:
        logger.error("No training command for model: %s", model_name)
        return False, 0

    env = None
    if env_vars:
        import os
        env = os.environ.copy()
        env.update(env_vars)

    # Always set PYTHONIOENCODING for Windows compatibility
    if env is None:
        import os
        env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # For module-based commands, set PYTHONPATH
    if "-m" in cmd:
        env["PYTHONPATH"] = str(REPO_ROOT / "src")

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error("Training failed:\n%s\n%s", result.stdout[-500:], result.stderr[-500:])
        return False, elapsed

    return True, elapsed


def _backtest_model(config, model_name, start, end, exit_version):
    """Run backtest for model, return trades."""
    gen_class, model_dir = ML_GENERATORS[model_name]
    onnx_map = {
        "ml_v3": "v3_model.onnx",
        "ml_v2_attention": "attention_model.onnx",
        "ml_v1": "direction_model.onnx",
    }
    onnx_name = onnx_map.get(model_name, "direction_model.onnx")

    trades = run_backtest(
        config, start=start, end=end,
        ml_model_dir=model_dir, ml_generator_class=gen_class,
        ml_onnx_filename=onnx_name,
        ml_only=True,
        exit_version=exit_version,
    )
    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(TRAIN_COMMANDS.keys()))
    parser.add_argument("--level", type=int, action="append", required=True,
                        choices=[2, 3], help="Which levels to sweep (can repeat)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    t0_total = time.time()

    model_name = args.model
    levels = sorted(set(args.level))
    logger.info("Training sweep for: %s | Levels: %s", model_name, levels)

    # Load config
    params = _load_params()
    model_cfg = params[model_name]
    sweep_cfg = model_cfg.get("sweep", {})
    split_cfg = model_cfg.get("split", {})
    exit_version = model_cfg.get("inference", {}).get("exit_version", "v1")

    val_start = split_cfg.get("val_start", "2024-01-01")
    val_end = split_cfg.get("val_end", "2024-12-31")
    test_start = split_cfg.get("test_start", "2025-01-01")
    test_end = split_cfg.get("test_end", "2025-12-31")

    # Build param grid from selected levels
    all_param_names = []
    for level in levels:
        if level == 2:
            all_param_names.extend(LEVEL_2_PARAMS)
        elif level == 3:
            all_param_names.extend(LEVEL_3_PARAMS)

    param_names, combos = _build_grid(sweep_cfg, all_param_names)

    if not combos:
        logger.error("No sweep values found in params.yaml for %s.sweep", model_name)
        return

    logger.info("Grid: %d combos | Params: %s", len(combos), param_names)
    logger.info("Val: %s to %s | Test: %s to %s | Exit: %s",
                val_start, val_end, test_start, test_end, exit_version)

    # Save original training params for restore
    original_training = json.loads(json.dumps(model_cfg["training"]))

    config = load_config()

    # Init MLflow
    try:
        import mlflow
        from mlops import tracking
        tracking.init()
        experiment_name = f"{model_name}_training_sweep"
        mlflow.set_experiment(experiment_name)
        _use_mlflow = True
        logger.info("MLflow experiment: %s", experiment_name)
    except Exception as e:
        logger.warning("MLflow not available: %s", e)
        _use_mlflow = False

    # Phase 1: Train + backtest on VAL for each combo
    logger.info("\n=== PHASE 1: Train + backtest on val (%s to %s) ===", val_start, val_end)
    results = []

    for i, combo in enumerate(combos):
        combo_dict = dict(zip(param_names, combo))
        logger.info("\n  [%d/%d] %s", i + 1, len(combos), combo_dict)

        # Patch params
        params = _load_params()
        _patch_training_params(params, model_name, param_names, combo)
        _save_params(params)

        # Train
        success, train_time = _train_model(model_name)
        if not success:
            logger.error("    TRAINING FAILED — skipping")
            results.append({"params": combo_dict, "status": "train_failed"})
            continue
        logger.info("    Trained in %.1f min", train_time / 60)

        # Backtest on val
        trades = _backtest_model(config, model_name, val_start, val_end, exit_version)
        metrics = evaluate_trades(trades, model_name)
        a = metrics["all"]
        logger.info("    VAL: %d trades | %+.0f bps | PF %.2f | DD %+.0f | stop %.1f%%",
                    a["n"], a["bps"], a["pf"], a["dd"], a["stop_pct"])

        result = {
            "params": combo_dict,
            "status": "ok",
            "period": "val",
            "train_time_sec": round(train_time, 1),
            **metrics,
        }
        results.append(result)

        # Log to MLflow
        if _use_mlflow:
            try:
                run_name = f"L{''.join(str(l) for l in levels)}_{'_'.join(f'{k}{v}' for k,v in combo_dict.items())}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags({"model": model_name, "sweep_type": "training",
                                      "levels": str(levels)})
                    mlflow.log_params(combo_dict)
                    for split in ["all", "long", "short"]:
                        for k, v in metrics[split].items():
                            mlflow.log_metric(f"val_{split}_{k}", v)
                    mlflow.log_metric("train_time_sec", train_time)
            except Exception:
                pass

    # Restore original params
    params = _load_params()
    params[model_name]["training"] = original_training
    _save_params(params)
    logger.info("\nRestored original training params")

    # Rank by PF (min 50 trades on val)
    valid = [r for r in results if r.get("status") == "ok" and r["all"]["n"] >= 50]
    valid.sort(key=lambda r: r["all"]["pf"], reverse=True)

    logger.info("\n=== VAL RESULTS (ranked by PF, min 50 trades) ===")
    header = "  "
    for name in param_names:
        header += f"{name:<14}"
    header += f"{'Trades':>7} {'Bps':>8} {'PF':>6} {'DD':>7} {'Time':>6}"
    logger.info(header)
    logger.info("  " + "-" * (14 * len(param_names) + 40))
    for r in valid:
        row = "  "
        for name in param_names:
            row += f"{r['params'][name]:<14}"
        a = r["all"]
        row += f"{a['n']:>7} {a['bps']:>+8.0f} {a['pf']:>6.2f} {a['dd']:>+7.0f} {r['train_time_sec']/60:>5.1f}m"
        logger.info(row)

    # Phase 2: Test ALL valid configs on TEST
    if not valid:
        logger.error("No valid configs found on val!")
        return

    logger.info("\n=== PHASE 2: Retrain + backtest all %d valid configs on test (%s to %s) ===",
                len(valid), test_start, test_end)

    test_results = []
    for i, r in enumerate(valid):
        combo_dict = r["params"]
        combo = [combo_dict[name] for name in param_names]
        logger.info("\n  [%d/%d] %s (val PF=%.2f)", i + 1, len(valid), combo_dict, r["all"]["pf"])

        # Patch + retrain (model was overwritten by subsequent val trials)
        params = _load_params()
        _patch_training_params(params, model_name, param_names, combo)
        _save_params(params)

        success, train_time = _train_model(model_name)
        if not success:
            logger.error("    RETRAIN FAILED — skipping")
            continue

        # Backtest on test
        trades = _backtest_model(config, model_name, test_start, test_end, exit_version)
        metrics = evaluate_trades(trades, model_name)
        a = metrics["all"]
        logger.info("    TEST: %d trades | %+.0f bps | PF %.2f | DD %+.0f",
                    a["n"], a["bps"], a["pf"], a["dd"])

        test_result = {
            "params": combo_dict,
            "period": "test",
            "val_pf": r["all"]["pf"],
            "val_bps": r["all"]["bps"],
            **metrics,
        }
        test_results.append(test_result)

        # Log to MLflow
        if _use_mlflow:
            try:
                run_name = f"TEST_L{''.join(str(l) for l in levels)}_{'_'.join(f'{k}{v}' for k,v in combo_dict.items())}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags({"model": model_name, "sweep_type": "training",
                                      "levels": str(levels), "period": "test"})
                    mlflow.log_params(combo_dict)
                    for split in ["all", "long", "short"]:
                        for k, v in metrics[split].items():
                            mlflow.log_metric(f"test_{split}_{k}", v)
                    mlflow.log_metric("val_pf", r["all"]["pf"])
            except Exception:
                pass

    # Restore original params
    params = _load_params()
    params[model_name]["training"] = original_training
    _save_params(params)

    # Rank test results
    test_valid = [r for r in test_results if r["all"]["n"] >= 50]
    test_valid.sort(key=lambda r: r["all"]["pf"], reverse=True)

    logger.info("\n=== TEST RESULTS (ranked by PF, min 50 trades) ===")
    header = "  "
    for name in param_names:
        header += f"{name:<14}"
    header += f"{'Trades':>7} {'Bps':>8} {'PF':>6} {'DD':>7} {'Val_PF':>7}"
    logger.info(header)
    logger.info("  " + "-" * (14 * len(param_names) + 46))
    for r in test_valid:
        row = "  "
        for name in param_names:
            row += f"{r['params'][name]:<14}"
        a = r["all"]
        row += f"{a['n']:>7} {a['bps']:>+8.0f} {a['pf']:>6.2f} {a['dd']:>+7.0f} {r.get('val_pf', 0):>7.2f}"
        logger.info(row)

    # Winner
    if test_valid:
        best = test_valid[0]
    else:
        best = valid[0] if valid else None

    if best:
        a = best["all"]
        logger.info("\n=== WINNER ===")
        logger.info("  Params: %s", best["params"])
        logger.info("  ALL:   %d trades | %+.0f bps | PF %.2f | DD %+.0f",
                    a["n"], a["bps"], a["pf"], a["dd"])

    # Save report
    report = {
        "model": model_name,
        "levels": levels,
        "exit_version": exit_version,
        "val_period": f"{val_start} to {val_end}",
        "test_period": f"{test_start} to {test_end}",
        "param_names": param_names,
        "n_combos": len(combos),
        "winner": {
            "params": best["params"],
            "test_all": best["all"],
            "test_long": best["long"],
            "test_short": best["short"],
            "val_pf": best.get("val_pf", 0),
        } if best else None,
        "all_val_results": [r for r in results if r.get("status") == "ok"],
        "all_test_results": test_results,
        "runtime_sec": round(time.time() - t0_total, 1),
    }

    output_path = REPO_ROOT / "data/reports" / model_name / "training_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\nSaved to %s", output_path)
    logger.info("Total time: %.1f min", (time.time() - t0_total) / 60)
    logger.info("\nTo apply: update configs/params.yaml → %s.training with winner params",
                model_name)


if __name__ == "__main__":
    main()
