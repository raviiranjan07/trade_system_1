"""V1 + PT + LP + bar 6 min-profit rule + TIGHT_TS only if > 15.

Rules:
  1. PROFIT_TARGET: tick_pnl >= 60 AND bars_held <= 5 -> exit
  2. LOCKED_PROFIT: once peak >= 20, exit when tick_pnl <= 15 -> exit
  3. BAR6_MIN: at bar 6+, if tick_pnl < 15, exit immediately
  4. TIGHT_TS: bar 6+, only if tick_pnl >= 15, drawdown >= 8 -> exit
  5. TIME_EXIT: bar 10

Run: PYTHONPATH=src python experiments/exit_strategy/v1_bar6_min15/run_experiment.py
"""

import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from engine.config.loader import load_config
from engine.signals.ml_v1 import MLV1
from engine.strategy import V12Strategy, Direction
from engine.config.constants import FEES_BPS

logger = logging.getLogger(__name__)

DATA_15M = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
DATA_1M = Path("data/raw/BTCUSDT_1m_ohlcv.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_DIR.mkdir(exist_ok=True)

LOCK_ARM_BPS = 20
LOCK_TRIGGER_BPS = 15
TIGHT_BPS = 8
TIGHTEN_AFTER_BAR = 5
MAX_BARS = 10
PROFIT_TARGET_BPS = 60
PROFIT_TARGET_MAX_BAR = 5
BAR6_MIN_BPS = 15


@dataclass
class Trade:
    signal_time: object
    entry_time: object
    exit_time: object
    direction: str
    signal_type: str
    entry_price: float
    exit_price: float
    gross_profit_bps: float
    net_profit_bps: float
    mfe_bps: float
    mae_bps: float
    exit_bar: int
    exit_reason: str


def run_backtest(signals, df_15m, df_1m):
    times_15m = df_15m.index
    opens_15m = df_15m["open"].values
    closes_15m = df_15m["close"].values
    n_15m = len(df_15m)
    idx_1m = df_1m.index.values
    closes_1m = df_1m["close"].values
    signal_map = {s.bar_index: s for s in signals}

    trades = []
    pos = None

    i = 0
    while i < n_15m:
        if pos is not None:
            bar_start = np.datetime64(times_15m[i])
            bar_end = bar_start + np.timedelta64(15, "m")
            s_idx = np.searchsorted(idx_1m, bar_start, side="left")
            e_idx = np.searchsorted(idx_1m, bar_end, side="left")

            exit_reason = None
            exit_price = None
            exit_time = None

            for t in range(s_idx, e_idx):
                tick_price = closes_1m[t]
                if pos["direction"] == "LONG":
                    tick_pnl = (tick_price - pos["entry_price"]) / pos["entry_price"] * 10000
                else:
                    tick_pnl = (pos["entry_price"] - tick_price) / pos["entry_price"] * 10000

                if tick_pnl > pos["mfe_bps"]: pos["mfe_bps"] = tick_pnl
                if tick_pnl > pos["peak"]: pos["peak"] = tick_pnl
                if tick_pnl < pos["mae_bps"]: pos["mae_bps"] = tick_pnl

                if pos["peak"] >= LOCK_ARM_BPS:
                    pos["armed"] = True

                # 1. PROFIT_TARGET
                if tick_pnl >= PROFIT_TARGET_BPS and pos["bars_held"] <= PROFIT_TARGET_MAX_BAR:
                    exit_reason = "PROFIT_TARGET"
                    exit_price = tick_price
                    exit_time = idx_1m[t]
                    break

                # 2. LOCKED_PROFIT
                if pos["armed"] and tick_pnl <= LOCK_TRIGGER_BPS:
                    exit_reason = "LOCKED_PROFIT"
                    exit_price = tick_price
                    exit_time = idx_1m[t]
                    break

                # === BAR 6+ TIGHT_TS (only if pnl >= 15) ===
                # pnl < 15: "no zone" — keep holding
                # pnl >= 15: apply tight trailing stop
                if pos["bars_held"] > TIGHTEN_AFTER_BAR and tick_pnl >= BAR6_MIN_BPS:
                    drawdown = pos["peak"] - tick_pnl
                    if drawdown >= TIGHT_BPS:
                        exit_reason = "TIGHT_TS"
                        exit_price = tick_price
                        exit_time = idx_1m[t]
                        break

            if exit_reason is None:
                pos["bars_held"] += 1

                # Bar 10: split into NO_ZONE vs TIME_EXIT based on price
                if pos["bars_held"] >= MAX_BARS:
                    bar_close = closes_15m[i]
                    if pos["direction"] == "LONG":
                        bar_close_pnl = (bar_close - pos["entry_price"]) / pos["entry_price"] * 10000
                    else:
                        bar_close_pnl = (pos["entry_price"] - bar_close) / pos["entry_price"] * 10000
                    if bar_close_pnl < 0:
                        exit_reason = "TIME_EXIT"   # below entry
                    else:
                        exit_reason = "NO_ZONE"     # above entry but in no-zone
                    exit_price = bar_close
                    exit_time = times_15m[i]

            if exit_reason is not None:
                if pos["direction"] == "LONG":
                    gross = (exit_price - pos["entry_price"]) / pos["entry_price"] * 10000
                else:
                    gross = (pos["entry_price"] - exit_price) / pos["entry_price"] * 10000
                trades.append(Trade(
                    signal_time=pos["signal_time"],
                    entry_time=pos["entry_time"],
                    exit_time=exit_time,
                    direction=pos["direction"],
                    signal_type=pos["signal_type"],
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    gross_profit_bps=gross,
                    net_profit_bps=gross - FEES_BPS,
                    mfe_bps=pos["mfe_bps"],
                    mae_bps=pos["mae_bps"],
                    exit_bar=pos["bars_held"],
                    exit_reason=exit_reason,
                ))
                pos = None
                i += 1
                continue

            i += 1
            continue

        if i in signal_map and i + 1 < n_15m:
            sig = signal_map[i]
            pos = {
                "direction": sig.direction.value,
                "signal_type": sig.signal_type.value,
                "entry_price": opens_15m[i + 1],
                "entry_time": times_15m[i + 1],
                "signal_time": sig.timestamp,
                "bars_held": 0,
                "peak": 0.0,
                "mfe_bps": 0.0,
                "mae_bps": 0.0,
                "armed": False,
            }
            i += 2
            continue

        i += 1

    return trades


def summarize(trades, name):
    if not trades:
        return {"config": name, "status": "NO_TRADES"}
    tdf = pd.DataFrame([asdict(t) for t in trades])
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["month"] = tdf["entry_time"].dt.to_period("M")
    winners = tdf[tdf["net_profit_bps"] > 0]
    losers = tdf[tdf["net_profit_bps"] <= 0]
    gw = winners["net_profit_bps"].sum() if len(winners) > 0 else 0
    gl = abs(losers["net_profit_bps"].sum()) if len(losers) > 0 else 1
    pf = gw / gl
    equity = tdf["net_profit_bps"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    n_months = tdf["month"].nunique()
    by_reason = {}
    for r in sorted(tdf["exit_reason"].unique()):
        rdf = tdf[tdf["exit_reason"] == r]
        by_reason[r] = {
            "n": len(rdf),
            "net_bps": round(rdf["net_profit_bps"].sum(), 1),
            "win_rate": round((rdf["net_profit_bps"] > 0).sum() / len(rdf) * 100, 1),
            "avg_bps": round(rdf["net_profit_bps"].mean(), 1),
        }
    return {
        "config": name,
        "n_trades": len(tdf),
        "win_rate_pct": round(len(winners) / len(tdf) * 100, 1),
        "total_bps": round(tdf["net_profit_bps"].sum(), 1),
        "profit_factor": round(pf, 2),
        "avg_bps_per_month": round(tdf["net_profit_bps"].sum() / n_months, 1),
        "max_drawdown_bps": round(max_dd, 1),
        "by_exit_reason": by_reason,
    }


def print_summary(s):
    print(f"\n{'='*72}")
    print(f"CONFIG: {s['config']}")
    print('='*72)
    if s.get("status") == "NO_TRADES":
        return
    print(f"  Trades: {s['n_trades']} | Win: {s['win_rate_pct']}%")
    print(f"  Net PnL: {s['total_bps']:+.0f} bps | PF: {s['profit_factor']} | Avg/Mo: {s['avg_bps_per_month']:+.0f}")
    print(f"  Max DD: {s['max_drawdown_bps']:+.0f}")
    print(f"  By exit reason:")
    for r, d in s["by_exit_reason"].items():
        print(f"    {r:<18} {d['n']:>4}t | Win: {d['win_rate']:>5.1f}% | Net: {d['net_bps']:>+8.1f} | Avg: {d['avg_bps']:>+6.1f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config()
    df_15m = pd.read_parquet(DATA_15M)
    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
    strategy = V12Strategy(cfg)
    df_15m = strategy.compute_indicators(df_15m)
    df_1m = pd.read_parquet(DATA_1M)
    df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    test_15m = df_15m["2024-01-01":"2025-12-31"]
    test_1m = df_1m["2024-01-01":"2025-12-31"]

    v15_gen = MLV1()
    test_v15 = v15_gen.compute_features_from_df(test_15m.copy())
    signals = v15_gen.generate_signals(test_v15)
    logger.info("V1.5 signals: %d", len(signals))

    trades = run_backtest(signals, test_v15, test_1m)
    s = summarize(trades, "v15_bar6_min15")
    print_summary(s)

    import json
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(s, f, indent=2, default=str)
    tdf = pd.DataFrame([asdict(t) for t in trades])
    tdf.to_csv(RESULTS_DIR / "trades.csv", index=False)
    logger.info("Saved: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
