"""Stage 6: Compare V3 vs V2 backtest results side-by-side.

Produces a comparison table and saves to JSON.

Run: PYTHONPATH=src python -m engine.compare_models
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "data/reports"
OUTPUT_PATH = REPORT_DIR / "model_comparison.json"

MODELS = {
    "V1.4": REPORT_DIR / "backtest_v14.json",
    "ML_V1": REPORT_DIR / "backtest_ml_v1.json",
    "ML_V2": REPORT_DIR / "backtest_ml_v2_attention.json",
    "ML_V3": REPORT_DIR / "backtest_ml_v3.json",
}


def load_metrics(path, fallback=None):
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("metrics", data)
    if fallback:
        return fallback
    return {}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Load all model results
    all_data = {}
    for name, path in MODELS.items():
        all_data[name] = load_metrics(path)

    metric_keys = [
        ("Trades", "n_trades", "{:.0f}"),
        ("Total bps", "total_bps", "{:+.0f}"),
        ("Profit Factor", "pf", "{:.2f}"),
        ("Win Rate %", "win_pct", "{:.1f}"),
        ("Stop Rate %", "stop_rate", "{:.1f}"),
        ("Max DD bps", "max_dd_bps", "{:.0f}"),
        ("Avg bps/trade", "avg_trade_bps", "{:+.1f}"),
    ]

    print()
    print("=" * 80)
    print("ALL MODELS COMPARISON — Independent Backtests")
    print("=" * 80)

    header = f"  {'Metric':<16}"
    for name in MODELS:
        header += f" {name:>12}"
    print(header)
    print(f"  {'-'*72}")

    comparison = {}
    for label, key, fmt in metric_keys:
        row = f"  {label:<16}"
        vals = {}
        for name in MODELS:
            v = all_data[name].get(key, 0)
            if v is None:
                v = 0
            row += f" {fmt.format(v):>12}"
            vals[name] = v
        print(row)
        comparison[key] = vals

    # Best model by PF
    pf_vals = comparison.get("pf", {})
    if pf_vals:
        best = max(pf_vals, key=pf_vals.get)
        print(f"\n  Best PF: {best} ({pf_vals[best]:.2f})")

    bps_vals = comparison.get("total_bps", {})
    if bps_vals:
        best_bps = max(bps_vals, key=bps_vals.get)
        print(f"  Best bps: {best_bps} ({bps_vals[best_bps]:+.0f})")

    print()

    result = {
        "comparison": comparison,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
