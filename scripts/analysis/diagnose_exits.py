"""Diagnose V1 exit bleeds on 2024 data (val period).

Reads data/reports/backtest_staging_trades.parquet (currently 2024 window)
and answers methodologically-clean questions about exit behavior WITHOUT
doing grid search on performance. Hypothesis generation, not optimization.

Key questions:
  1. STOP_LOSS trades: what was the MFE before the stop triggered?
     - MFE < 5: trade was wrong from bar 1 → stop is correct
     - MFE > 20: trade ran positive, then reversed → stop too tight
  2. For each signal_type: what's the loss rate? Avg MFE/MAE?
  3. Are any signal types structurally broken (loss rate > 60%)?
  4. Could a wider stop (-15, -20) reduce bleed without killing winners?
     (Check: how many current winners had MAE in [-10, -20] range?)

Usage:
    python scripts/analysis/diagnose_exits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TRADES_PATH = REPO_ROOT / "data/reports/backtest_staging_trades.parquet"
REPORT_PATH = REPO_ROOT / "data/reports/diagnose_exits.json"


def bucket(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def main() -> None:
    t = pd.read_parquet(TRADES_PATH)
    print(f"Analyzing {len(t)} trades from {TRADES_PATH.relative_to(REPO_ROOT)}")
    print(f"Direction: {t['direction'].value_counts().to_dict()}")
    print(f"Exit reasons: {t['exit_reason'].value_counts().to_dict()}\n")

    report = {"n_trades": int(len(t)), "total_bps": round(float(t["net_profit_bps"].sum()), 1)}

    # ---- Q1: STOP_LOSS diagnosis — were these wrong-direction or just tight stops? ----
    stops = t[t["exit_reason"] == "STOP_LOSS"].copy()
    print("=" * 70)
    print(f"Q1. STOP_LOSS ANALYSIS  ({len(stops)} trades = {len(stops)/len(t)*100:.1f}% of all)")
    print("=" * 70)
    print(f"Total bleed: {stops['net_profit_bps'].sum():+.0f} bps")
    print()

    # MFE buckets
    bins = [-1, 5, 10, 20, 50, 99999]
    labels = ["MFE<5 (wrong-dir)", "5-10", "10-20", "20-50", "50+ (recoverable)"]
    stops["mfe_bucket"] = bucket(stops["mfe_bps"], bins, labels)
    mfe_dist = stops["mfe_bucket"].value_counts().sort_index()
    print(f"  {'MFE bucket':<24} {'trades':>8} {'share':>8}")
    for b, n in mfe_dist.items():
        print(f"  {str(b):<24} {n:>8} {n/len(stops)*100:>7.1f}%")
    print()

    # The critical number
    wrong_dir = (stops["mfe_bps"] < 5).sum()
    maybe_recoverable = (stops["mfe_bps"] >= 20).sum()
    print(f"  -> Wrong-direction (MFE<5):  {wrong_dir} trades ({wrong_dir/len(stops)*100:.1f}%)")
    print(f"     Stop was correct — trade never had a chance.")
    print(f"  -> Went positive first (MFE>=20): {maybe_recoverable} trades ({maybe_recoverable/len(stops)*100:.1f}%)")
    print(f"     Potentially recoverable with wider stop or break-even rule.")

    report["stop_loss"] = {
        "total_trades": int(len(stops)),
        "total_bps": round(float(stops["net_profit_bps"].sum()), 1),
        "wrong_direction_mfe_lt_5": int(wrong_dir),
        "recoverable_mfe_gte_20": int(maybe_recoverable),
        "mfe_distribution": {str(k): int(v) for k, v in mfe_dist.items()},
    }

    # ---- Q2: Per signal_type profitability ----
    print()
    print("=" * 70)
    print("Q2. SIGNAL TYPE PROFITABILITY")
    print("=" * 70)
    by_sig = t.groupby("signal_type").agg(
        n=("net_profit_bps", "size"),
        win_pct=("net_profit_bps", lambda x: (x > 0).mean() * 100),
        total_bps=("net_profit_bps", "sum"),
        avg_bps=("net_profit_bps", "mean"),
        stop_rate=("exit_reason", lambda x: (x == "STOP_LOSS").mean() * 100),
    ).round(1).sort_values("total_bps", ascending=False)
    print(by_sig.to_string())

    broken = by_sig[by_sig["total_bps"] < 0]
    if len(broken) > 0:
        print(f"\n  -> Loss-making signals: {list(broken.index)}")
        print(f"     Combined bleed: {broken['total_bps'].sum():+.0f} bps")

    report["by_signal_type"] = by_sig.to_dict(orient="index")

    # ---- Q3: MAE of current WINNERS — how close did they get to the stop? ----
    print()
    print("=" * 70)
    print("Q3. WINNER RESILIENCE — MAE distribution of profitable trades")
    print("=" * 70)
    winners = t[t["net_profit_bps"] > 0].copy()
    # MAE is already negative (deepest drawdown during position)
    mae_bins = [-999, -20, -15, -10, -5, 0.1]
    mae_labels = ["<-20", "[-20,-15)", "[-15,-10)", "[-10,-5)", "[-5, 0]"]
    winners["mae_bucket"] = bucket(winners["mae_bps"], mae_bins, mae_labels)
    mae_dist = winners["mae_bucket"].value_counts().sort_index()
    print(f"  {'MAE bucket (bps)':<22} {'winners':>8} {'share':>8}")
    for b, n in mae_dist.items():
        print(f"  {str(b):<22} {n:>8} {n/len(winners)*100:>7.1f}%")
    print()
    deep = (winners["mae_bps"] <= -10).sum()
    print(f"  -> Winners that DIPPED below -10 bps: {deep} ({deep/len(winners)*100:.1f}%)")
    print(f"     These survived the stop being at -10. A wider stop wouldn't add winners here.")
    print(f"     A TIGHTER stop (e.g. -5) would kill {deep} existing winners.")

    report["winner_mae"] = {str(k): int(v) for k, v in mae_dist.items()}
    report["winners_that_dipped_past_neg10"] = int(deep)

    # ---- Q4: Loser MAE — wider stop impact simulation ----
    print()
    print("=" * 70)
    print("Q4. LOSER MAE — simulate wider stops")
    print("=" * 70)
    losers = t[t["net_profit_bps"] <= 0].copy()
    print(f"Current losers: {len(losers)} trades, {losers['net_profit_bps'].sum():+.0f} bps total")
    print()
    print("If stop widened to -X, the trade would still stop at -X (net ~-X-8 bps).")
    print("But SOME losers would recover before hitting the wider stop:")
    print()
    print(f"  {'Wider stop':<12} {'losers still stopping':<25} {'losers now winning':<22}")
    for new_stop in [-12, -15, -20, -25]:
        would_still_stop = (losers["mae_bps"] <= new_stop).sum()
        would_now_win = len(losers) - would_still_stop
        # But we have to assume MFE > 0 means these could profit
        would_now_win_actually = losers[(losers["mae_bps"] > new_stop) & (losers["mfe_bps"] > 10)].shape[0]
        print(f"  -{abs(new_stop)} bps{'':<5} {would_still_stop:<25} {would_now_win_actually:<22}")
    print()
    print("NOTE: 'would now win' assumes MFE>10 means it could have taken a winner path.")
    print("      Real outcome depends on exit rule once stop is widened. This is directional only.")

    report["stop_widening_sim"] = {}
    for new_stop in [-12, -15, -20, -25]:
        would_stop = int((losers["mae_bps"] <= new_stop).sum())
        potential_winners = int(losers[(losers["mae_bps"] > new_stop) & (losers["mfe_bps"] > 10)].shape[0])
        report["stop_widening_sim"][f"stop_at_{new_stop}"] = {
            "losers_still_stopping": would_stop,
            "potential_saved_trades": potential_winners,
        }

    # ---- Q5: Money left on table ----
    print()
    print("=" * 70)
    print("Q5. MONEY LEFT ON TABLE (winners — MFE vs net)")
    print("=" * 70)
    # For each winning trade, compare net to MFE
    w = winners.copy()
    w["gap_to_peak_bps"] = w["mfe_bps"] - w["net_profit_bps"]
    gap_bins = [0, 10, 25, 50, 100, 999]
    gap_labels = ["<10 bps left", "10-25", "25-50", "50-100", "100+ left"]
    w["gap_bucket"] = bucket(w["gap_to_peak_bps"], gap_bins, gap_labels)
    gap_dist = w["gap_bucket"].value_counts().sort_index()
    print(f"  {'gap to peak':<20} {'winners':>8} {'share':>8}")
    for b, n in gap_dist.items():
        print(f"  {str(b):<20} {n:>8} {n/len(w)*100:>7.1f}%")
    total_left = float(w["gap_to_peak_bps"].sum())
    print(f"\n  -> Total money left on table: {total_left:.0f} bps ({total_left/len(w):.1f} bps avg per winner)")

    report["money_left_on_table_bps"] = round(total_left, 1)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
