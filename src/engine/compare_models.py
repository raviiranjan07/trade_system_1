"""Stage 6: Compare V3 vs V2 backtest results side-by-side.

Produces a comparison table and saves to JSON.

Run: PYTHONPATH=src python -m engine.compare_models
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
V3_BACKTEST = REPO_ROOT / "data/reports/backtest_staging_ml_v3.json"
V2_BASELINE = REPO_ROOT / "data/reports/backtest_staging_ml_v2_attention_exitv2.json"
OUTPUT_PATH = REPO_ROOT / "data/reports/v3_vs_v2_comparison.json"


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

    v3 = load_metrics(V3_BACKTEST)
    v2 = load_metrics(V2_BASELINE, fallback={
        "n_trades": 804, "total_bps": 10129, "pf": 2.28,
        "win_pct": 49.6, "max_dd_bps": -367, "avg_trade_bps": 12.6,
    })

    metrics = [
        ("Trades", "n_trades", "{:.0f}"),
        ("Total bps", "total_bps", "{:+.0f}"),
        ("Profit Factor", "pf", "{:.2f}"),
        ("Win Rate %", "win_pct", "{:.1f}"),
        ("Stop Rate %", "stop_rate", "{:.1f}"),
        ("Max DD bps", "max_dd_bps", "{:.0f}"),
        ("Avg bps/trade", "avg_trade_bps", "{:+.1f}"),
    ]

    print()
    print("=" * 65)
    print("V3 vs V2 COMPARISON")
    print("=" * 65)
    print(f"  {'Metric':<20} {'V2 (current)':>14} {'V3 (new)':>14} {'Delta':>12}")
    print(f"  {'-'*60}")

    comparison = {}
    for label, key, fmt in metrics:
        v2_val = v2.get(key, 0)
        v3_val = v3.get(key, 0)
        if v2_val is None:
            v2_val = 0
        if v3_val is None:
            v3_val = 0
        delta = v3_val - v2_val

        v2_str = fmt.format(v2_val)
        v3_str = fmt.format(v3_val)
        delta_str = f"{delta:+.1f}"

        print(f"  {label:<20} {v2_str:>14} {v3_str:>14} {delta_str:>12}")
        comparison[key] = {"v2": v2_val, "v3": v3_val, "delta": round(delta, 2)}

    # Per signal type if available
    v3_by_sig = v3.get("by_signal_type", {})
    if v3_by_sig:
        print(f"\n  V3 by signal type:")
        for sig, data in sorted(v3_by_sig.items()):
            if isinstance(data, dict):
                n = data.get("n", 0)
                bps = data.get("total_bps", 0)
                wp = data.get("win_pct", 0)
                print(f"    {sig:<16} {n:>4}t  win {wp:>5.1f}%  bps {bps:>+8.1f}")

    # Per year if available
    v3_by_year = v3.get("per_year", {})
    if v3_by_year:
        print(f"\n  V3 by year:")
        for year, data in sorted(v3_by_year.items()):
            if isinstance(data, dict):
                n = data.get("n", 0)
                bps = data.get("total_bps", 0)
                pf = data.get("pf", 0)
                print(f"    {year}: {n}t  bps {bps:>+8.1f}  PF {pf:.2f}")

    # Verdict
    bps_better = comparison.get("total_bps", {}).get("delta", 0) > 0
    stop_better = comparison.get("stop_rate", {}).get("delta", 0) < 0

    print()
    if bps_better and stop_better:
        verdict = "V3 WINS (better bps AND lower stop rate)"
    elif bps_better:
        verdict = "V3 MIXED (better bps but higher stop rate)"
    elif stop_better:
        verdict = "V3 MIXED (lower stop rate but less bps)"
    else:
        verdict = "V2 WINS (V3 did not improve)"
    print(f"  VERDICT: {verdict}")
    print()

    result = {
        "comparison": comparison,
        "verdict": verdict,
        "v3_by_signal_type": v3_by_sig,
        "v3_by_year": v3_by_year,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
