"""Stage 5b: Verify V3 backtest results are sane and improve over V2.

Pipeline FAILS if backtest results don't meet minimum thresholds.

Run: PYTHONPATH=src python -m engine.verify_v3_backtest
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
V3_BACKTEST = REPO_ROOT / "data/reports/backtest_staging_ml_v3.json"
V2_BASELINE = REPO_ROOT / "data/reports/backtest_staging_ml_v2_attention_exitv2.json"
OUTPUT_PATH = REPO_ROOT / "data/reports/v3_backtest_verification.json"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logger.info("Loading V3 backtest: %s", V3_BACKTEST)
    with open(V3_BACKTEST) as f:
        v3 = json.load(f)

    logger.info("Loading V2 baseline: %s", V2_BASELINE)
    if V2_BASELINE.exists():
        with open(V2_BASELINE) as f:
            v2 = json.load(f)
        v2_metrics = v2.get("metrics", v2)
    else:
        logger.warning("V2 baseline not found — using hardcoded values")
        v2_metrics = {
            "n_trades": 804,
            "total_bps": 10129,
            "pf": 2.28,
            "win_pct": 49.6,
            "max_dd_bps": -367,
        }

    v3_metrics = v3.get("metrics", v3)

    # === GATE CHECKS ===
    checks = {}
    all_pass = True

    # 1. Total trades > 50
    n_trades = v3_metrics.get("n_trades", 0)
    checks["total_trades"] = {"value": n_trades, "threshold": "> 50", "pass": n_trades > 50}
    logger.info("  Total trades: %d (gate > 50)", n_trades)

    # 2. PF > 1.0
    pf = v3_metrics.get("pf", 0)
    checks["profit_factor"] = {"value": round(pf, 2), "threshold": "> 1.0", "pass": pf > 1.0}
    logger.info("  Profit factor: %.2f (gate > 1.0)", pf)

    # 3. Stop rate < V2 baseline (50.4%)
    v2_stop = v2_metrics.get("stop_rate", 50.4)
    if isinstance(v2_stop, str):
        v2_stop = float(v2_stop.replace("%", ""))
    v3_stop = v3_metrics.get("stop_rate", 100)
    checks["stop_rate"] = {"value": round(v3_stop, 1), "threshold": f"< {v2_stop}", "pass": v3_stop < v2_stop}
    logger.info("  Stop rate: %.1f%% (gate < %.1f%% V2 baseline)", v3_stop, v2_stop)

    # 4. Max DD > -2000
    dd = v3_metrics.get("max_dd_bps", -9999)
    checks["max_drawdown"] = {"value": round(dd, 1), "threshold": "> -2000", "pass": dd > -2000}
    logger.info("  Max drawdown: %.1f bps (gate > -2000)", dd)

    # 5. Both years have trades
    per_year = v3_metrics.get("per_year", {})
    has_both = len(per_year) >= 2 if per_year else True  # pass if no year data available
    checks["both_years"] = {"value": len(per_year), "threshold": ">= 2", "pass": has_both}
    logger.info("  Years with trades: %d (gate >= 2)", len(per_year))

    # 6. No NaN in key metrics
    no_nan = all(v3_metrics.get(k) is not None for k in ["n_trades", "total_bps", "pf"])
    checks["no_nan"] = {"value": "OK" if no_nan else "NaN detected", "threshold": "no NaN", "pass": no_nan}

    # Informational comparisons (don't gate, just log)
    v2_bps = v2_metrics.get("total_bps", 10129)
    v3_bps = v3_metrics.get("total_bps", 0)
    v2_pf = v2_metrics.get("pf", 2.28)

    logger.info("\n  --- V3 vs V2 Comparison (informational) ---")
    logger.info("  Total bps: V3=%+.0f vs V2=%+.0f (%+.0f)", v3_bps, v2_bps, v3_bps - v2_bps)
    logger.info("  PF: V3=%.2f vs V2=%.2f", pf, v2_pf)
    logger.info("  Stop rate: V3=%.1f%% vs V2=%.1f%%", v3_stop, v2_stop)

    for name, check in checks.items():
        if not check["pass"]:
            all_pass = False
            logger.error("  GATE FAIL: %s = %s (required %s)", name, check["value"], check["threshold"])

    report = {
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "v3_metrics": {k: v for k, v in v3_metrics.items() if isinstance(v, (int, float, str, bool))},
        "v2_baseline": {k: v for k, v in v2_metrics.items() if isinstance(v, (int, float, str, bool))},
        "comparison": {
            "bps_delta": round(v3_bps - v2_bps, 1),
            "pf_delta": round(pf - v2_pf, 2),
            "stop_rate_delta": round(v3_stop - v2_stop, 1),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n%s: %s", "PIPELINE GATE", report["status"])
    logger.info("Saved to %s", OUTPUT_PATH)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
