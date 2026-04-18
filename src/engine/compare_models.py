"""Stage 6: Compare all models × exit strategies side-by-side.

Produces two comparison tables (V1 exits, V2 exits) + a delta table.

Run: PYTHONPATH=src python -m engine.compare_models
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "data/reports"
OUTPUT_PATH = REPORT_DIR / "comparison.json"

# V1 exits
MODELS_V1 = {
    "V1.4": REPORT_DIR / "v14" / "backtest.json",
    "ML_V1": REPORT_DIR / "ml_v1" / "backtest.json",
    "ML_V2": REPORT_DIR / "ml_v2_attention" / "backtest.json",
    "ML_V3": REPORT_DIR / "ml_v3" / "backtest.json",
}

# V2 exits (V1 minus LOCKED_PROFIT)
MODELS_V2 = {
    "V1.4": REPORT_DIR / "v14_v2" / "backtest.json",
    "ML_V1": REPORT_DIR / "ml_v1_v2" / "backtest.json",
    "ML_V2": REPORT_DIR / "ml_v2_attention_v2" / "backtest.json",
    "ML_V3": REPORT_DIR / "ml_v3_v2" / "backtest.json",
}

METRIC_KEYS = [
    ("Trades", "n", "{:.0f}"),
    ("Total bps", "bps", "{:+.0f}"),
    ("Profit Factor", "pf", "{:.2f}"),
    ("Win Rate %", "win_pct", "{:.1f}"),
    ("Stop Rate %", "stop_pct", "{:.1f}"),
    ("Max DD bps", "dd", "{:.0f}"),
    ("Avg bps/trade", "avg", "{:+.1f}"),
]


def load_group(models_dict, validate_report):
    """Load and validate a group of reports. Returns (reports, data) dicts."""
    reports = {}
    data = {}
    for name, path in models_dict.items():
        if not path.exists():
            logger.warning("Missing: %s — skipping %s", path, name)
            continue
        with open(path) as f:
            report = json.load(f)
        errors = validate_report(report)
        if errors:
            logger.error("INVALID REPORT %s: %s", name, errors)
            continue
        reports[name] = report
        data[name] = report.get("metrics", {}).get("all", {})
    return reports, data


def print_table(title, models_dict, reports, data):
    """Print a comparison table for one exit version."""
    if not data:
        print(f"\n  {title}: No valid reports")
        return {}

    # Show scope per model
    print()
    for name, report in reports.items():
        print(f"  {name}: mode={report['mode']} | exit={report['exit_version']} | {report['start']} to {report['end']}")

    print()
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)

    names = [n for n in models_dict if n in data]
    header = f"  {'Metric':<16}"
    for name in names:
        header += f" {name:>12}"
    print(header)
    print(f"  {'-' * (16 + 13 * len(names))}")

    comparison = {}
    for label, key, fmt in METRIC_KEYS:
        row = f"  {label:<16}"
        vals = {}
        for name in names:
            v = data[name].get(key, 0)
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

    return comparison


def print_delta(v1_data, v2_data):
    """Print V2-V1 delta for each model."""
    common = set(v1_data.keys()) & set(v2_data.keys())
    if not common:
        return

    print()
    print("=" * 80)
    print("V2 vs V1 EXIT DELTA (V2 - V1)")
    print("=" * 80)

    names = sorted(common)
    header = f"  {'Metric':<16}"
    for name in names:
        header += f" {name:>12}"
    print(header)
    print(f"  {'-' * (16 + 13 * len(names))}")

    for label, key, fmt in METRIC_KEYS:
        row = f"  {label:<16}"
        for name in names:
            v1 = v1_data[name].get(key, 0) or 0
            v2 = v2_data[name].get(key, 0) or 0
            delta = v2 - v1
            row += f" {delta:>+12.1f}"
        print(row)

    print()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mlops.backtest_report import validate_report, validate_comparable

    # Load V1 exits
    v1_reports, v1_data = load_group(MODELS_V1, validate_report)

    # Validate V1 comparability
    if len(v1_reports) >= 2:
        comp_errors = validate_comparable(list(v1_reports.values()))
        if comp_errors:
            for err in comp_errors:
                logger.error("V1 COMPARISON ERROR: %s", err)
            logger.error("REFUSING to compare incompatible V1 reports.")
            sys.exit(1)

    # Load V2 exits
    v2_reports, v2_data = load_group(MODELS_V2, validate_report)

    # Validate V2 comparability
    if len(v2_reports) >= 2:
        comp_errors = validate_comparable(list(v2_reports.values()))
        if comp_errors:
            for err in comp_errors:
                logger.error("V2 COMPARISON ERROR: %s", err)
            logger.error("REFUSING to compare incompatible V2 reports.")
            sys.exit(1)

    if not v1_data and not v2_data:
        logger.error("No valid reports to compare")
        sys.exit(1)

    # Print tables
    v1_comp = print_table("V1 EXITS — Independent Backtests", MODELS_V1, v1_reports, v1_data)
    v2_comp = print_table("V2 EXITS (no LOCKED_PROFIT) — Independent Backtests", MODELS_V2, v2_reports, v2_data)

    # Print delta
    if v1_data and v2_data:
        print_delta(v1_data, v2_data)

    print()

    # Save combined result
    result = {
        "v1_exits": v1_comp,
        "v2_exits": v2_comp,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
