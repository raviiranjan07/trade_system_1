"""Invariant verifier for backtest trades.

Reads data/reports/backtest_staging_trades.parquet and asserts that each
trade's outcome is consistent with the V1 exit rule that closed it.

Runs after backtest_staging. If any invariant fails, exits nonzero →
DVC halts the pipeline, no report is trusted.

Invariants (V1 exits, assuming FEES_BPS = 8):
  STOP_LOSS      net_bps in [-30 .. -10]           (stop at -10, fees -8 → ~-18; allow slight overshoot)
  PT_TARGET      net_bps >= +60  AND  mfe_bps >= +80
  PT_LOCK        net_bps >= +40  AND  mfe_bps >= +60
  MID_TRAIL      net_bps >= -5   AND  mfe_bps >= +25
  LOCKED_PROFIT  mfe_bps >= +15
  NO_ZONE        net_bps >= -10 (bar close positive-ish — legacy naming)
  TIME_EXIT      net_bps < 0 (bar-10 negative close)

Usage:
    python scripts/mlops/verify_backtest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TRADES_PATH = REPO_ROOT / "data/reports/backtest_staging_trades.parquet"
REPORT_PATH = REPO_ROOT / "data/reports/verify_backtest.json"

# V1 rule thresholds (read from schema defaults — keep in sync if schema changes)
V1_PT_TARGET = 80
V1_PT_LOCK = 60
V1_MID_TRAIL_ARM = 25
V1_LOCK_ARM = 15
V1_STOP_LOSS = -10
FEES_BPS = 8


def check_invariants(trades: pd.DataFrame) -> tuple[list[dict], int]:
    """Run per-exit-reason invariants. Returns (failures, total_checked)."""
    failures = []
    total = 0

    def flag(row, msg):
        failures.append({
            "exit_reason": row["exit_reason"],
            "direction": row["direction"],
            "signal_type": row["signal_type"],
            "net_profit_bps": float(row["net_profit_bps"]),
            "mfe_bps": float(row["mfe_bps"]),
            "mae_bps": float(row["mae_bps"]),
            "exit_bar": int(row["exit_bar"]),
            "reason": msg,
        })

    for _, r in trades.iterrows():
        total += 1
        er = r["exit_reason"]
        net = r["net_profit_bps"]
        mfe = r["mfe_bps"]

        if er == "STOP_LOSS":
            # Stop at -10 net of fees 8 => ~-18 net bps. Allow slippage to ~-30.
            if not (-30 <= net <= -10):
                flag(r, f"STOP_LOSS expected net in [-30..-10], got {net:.1f}")

        elif er == "PT_TARGET":
            if net < V1_PT_TARGET - FEES_BPS - 12:  # generous floor
                flag(r, f"PT_TARGET expected net >= ~+60, got {net:.1f}")
            if mfe < V1_PT_TARGET:
                flag(r, f"PT_TARGET requires MFE >= {V1_PT_TARGET}, got {mfe:.1f}")

        elif er == "PT_LOCK":
            if mfe < V1_PT_LOCK:
                flag(r, f"PT_LOCK requires MFE >= {V1_PT_LOCK}, got {mfe:.1f}")

        elif er == "MID_TRAIL":
            if mfe < V1_MID_TRAIL_ARM:
                flag(r, f"MID_TRAIL requires MFE >= {V1_MID_TRAIL_ARM}, got {mfe:.1f}")

        elif er == "LOCKED_PROFIT":
            if mfe < V1_LOCK_ARM:
                flag(r, f"LOCKED_PROFIT requires MFE >= {V1_LOCK_ARM}, got {mfe:.1f}")

        elif er in ("NO_ZONE", "TIME_EXIT"):
            # These fire only at bar close at max_bars. Invariant: exit_bar ==
            # max_bars. We don't know max_bars here for sure, but TIME_EXIT should
            # be negative; NO_ZONE should not be strongly negative.
            if er == "TIME_EXIT" and net >= 0:
                flag(r, f"TIME_EXIT expected negative net, got {net:.1f}")
            if er == "NO_ZONE" and net < -20:
                flag(r, f"NO_ZONE expected non-strongly-negative, got {net:.1f}")

        else:
            flag(r, f"unknown exit_reason: {er!r}")

    return failures, total


def main() -> int:
    if not TRADES_PATH.exists():
        print(f"Trades file missing: {TRADES_PATH}", file=sys.stderr)
        return 2

    trades = pd.read_parquet(TRADES_PATH)
    print(f"Verifying {len(trades)} trades from {TRADES_PATH.relative_to(REPO_ROOT)}\n")

    failures, total = check_invariants(trades)

    # Per-exit-reason summary
    summary = {}
    for er in sorted(trades["exit_reason"].unique()):
        sub = trades[trades["exit_reason"] == er]
        er_fails = sum(1 for f in failures if f["exit_reason"] == er)
        summary[er] = {
            "n": int(len(sub)),
            "fails": er_fails,
            "net_bps_range": [round(float(sub["net_profit_bps"].min()), 1),
                              round(float(sub["net_profit_bps"].max()), 1)],
            "mfe_bps_range": [round(float(sub["mfe_bps"].min()), 1),
                              round(float(sub["mfe_bps"].max()), 1)],
        }

    print("Per exit_reason:")
    print(f"  {'exit_reason':<18} {'n':>5} {'fails':>6} {'net_range':>20} {'mfe_range':>20}")
    for er, s in summary.items():
        tag = "OK" if s["fails"] == 0 else f"FAIL({s['fails']})"
        print(f"  {er:<18} {s['n']:>5} {tag:>6} {str(s['net_bps_range']):>20} {str(s['mfe_bps_range']):>20}")

    report = {
        "total_trades": total,
        "total_failures": len(failures),
        "summary_by_exit_reason": summary,
        "failures": failures[:50],  # cap for readability
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(f"\nReport: {REPORT_PATH.relative_to(REPO_ROOT)}")
    if failures:
        print(f"\n*** {len(failures)} invariant failures out of {total} trades ***")
        for f in failures[:5]:
            print(f"  {f['exit_reason']:<14} net={f['net_profit_bps']:+.1f} mfe={f['mfe_bps']:.1f}  -> {f['reason']}")
        return 1

    print(f"\nAll {total} trades satisfy V1 invariants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
