"""PT with trailing stop + hard cap at 100.

Replaces fixed PT_TARGET/PT_LOCK with:
  - Arm: peak >= 60
  - Trail width: 5 bps (floor at 60)
  - Hard target: 100 bps (exit immediately)

Everything else same as V3:
  LOCK 15, MID_TRAIL 25/10, PT_ARM 60, MAX_BARS 6.

Run: PYTHONPATH=src python experiments/exit_strategy/pt_trail/run_experiment.py
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

PT_ARM_BPS = 60
PT_FLOOR_BPS = 60
PT_HARD_TARGET = 100
PT_TRAIL_WIDTH = 10
PROFIT_TARGET_MAX_BAR = 5
LOCK_ARM_BPS = 15
LOCK_TRIGGER_BPS = 15
MID_TRAIL_ARM = 25
MID_TRAIL_WIDTH = 10
MAX_BARS = 6


@dataclass
class Trade:
    signal_time: object; entry_time: object; exit_time: object
    direction: str; signal_type: str
    entry_price: float; exit_price: float
    gross_profit_bps: float; net_profit_bps: float
    mfe_bps: float; mae_bps: float
    exit_bar: int; exit_reason: str


def ideal_exit_price(pos, exit_bps):
    if pos["direction"] == "LONG":
        return pos["entry_price"] * (1 + exit_bps / 10000)
    return pos["entry_price"] * (1 - exit_bps / 10000)


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

            exit_reason = None; exit_price = None; exit_time = None

            for t in range(s_idx, e_idx):
                tick_price = closes_1m[t]
                if pos["direction"] == "LONG":
                    tick_pnl = (tick_price - pos["entry_price"]) / pos["entry_price"] * 10000
                else:
                    tick_pnl = (pos["entry_price"] - tick_price) / pos["entry_price"] * 10000

                if tick_pnl > pos["mfe_bps"]: pos["mfe_bps"] = tick_pnl
                if tick_pnl > pos["peak"]: pos["peak"] = tick_pnl
                if tick_pnl < pos["mae_bps"]: pos["mae_bps"] = tick_pnl

                if pos["peak"] >= LOCK_ARM_BPS: pos["armed"] = True
                if pos["peak"] >= PT_ARM_BPS and pos["bars_held"] <= PROFIT_TARGET_MAX_BAR:
                    pos["pt_armed"] = True

                # PT_TARGET: hard cap at 100
                if pos.get("pt_armed") and tick_pnl >= PT_HARD_TARGET:
                    exit_reason = "PT_TARGET"
                    exit_price = ideal_exit_price(pos, PT_HARD_TARGET)
                    exit_time = idx_1m[t]
                    break

                # PT_TRAIL: once armed, trail at 5 bps (floor at 60)
                if pos.get("pt_armed"):
                    trail_exit = max(pos["peak"] - PT_TRAIL_WIDTH, PT_FLOOR_BPS)
                    if tick_pnl <= trail_exit:
                        exit_reason = "PT_TRAIL"
                        exit_price = ideal_exit_price(pos, trail_exit)
                        exit_time = idx_1m[t]
                        break

                # MID_TRAIL
                if pos["peak"] >= MID_TRAIL_ARM and not pos.get("pt_armed"):
                    drawdown = pos["peak"] - tick_pnl
                    if drawdown >= MID_TRAIL_WIDTH:
                        exit_at_bps = pos["peak"] - MID_TRAIL_WIDTH
                        exit_reason = "MID_TRAIL"
                        exit_price = ideal_exit_price(pos, exit_at_bps)
                        exit_time = idx_1m[t]
                        break

                # LOCKED_PROFIT
                if pos["armed"] and tick_pnl <= LOCK_TRIGGER_BPS:
                    exit_reason = "LOCKED_PROFIT"; exit_price = tick_price; exit_time = idx_1m[t]; break

            if exit_reason is None:
                pos["bars_held"] += 1
                if pos["bars_held"] >= MAX_BARS:
                    bar_close = closes_15m[i]
                    if pos["direction"] == "LONG":
                        bc_pnl = (bar_close - pos["entry_price"]) / pos["entry_price"] * 10000
                    else:
                        bc_pnl = (pos["entry_price"] - bar_close) / pos["entry_price"] * 10000
                    exit_reason = "NO_ZONE" if bc_pnl >= 0 else "TIME_EXIT"
                    exit_price = bar_close; exit_time = times_15m[i]

            if exit_reason is not None:
                if pos["direction"] == "LONG":
                    gross = (exit_price - pos["entry_price"]) / pos["entry_price"] * 10000
                else:
                    gross = (pos["entry_price"] - exit_price) / pos["entry_price"] * 10000
                trades.append(Trade(
                    signal_time=pos["signal_time"], entry_time=pos["entry_time"], exit_time=exit_time,
                    direction=pos["direction"], signal_type=pos["signal_type"],
                    entry_price=pos["entry_price"], exit_price=exit_price,
                    gross_profit_bps=gross, net_profit_bps=gross - FEES_BPS,
                    mfe_bps=pos["mfe_bps"], mae_bps=pos["mae_bps"],
                    exit_bar=pos["bars_held"], exit_reason=exit_reason,
                ))
                pos = None; i += 1; continue
            i += 1; continue

        if i in signal_map and i + 1 < n_15m:
            sig = signal_map[i]
            pos = {
                "direction": sig.direction.value, "signal_type": sig.signal_type.value,
                "entry_price": opens_15m[i + 1], "entry_time": times_15m[i + 1],
                "signal_time": sig.timestamp,
                "bars_held": 0, "peak": 0.0, "mfe_bps": 0.0, "mae_bps": 0.0,
                "armed": False, "pt_armed": False,
            }
            i += 2; continue
        i += 1
    return trades


def summarize(trades, name):
    tdf = pd.DataFrame([asdict(t) for t in trades])
    winners = tdf[tdf["net_profit_bps"] > 0]
    gw = winners["net_profit_bps"].sum()
    gl = abs(tdf[tdf["net_profit_bps"] <= 0]["net_profit_bps"].sum()) or 1
    pf = gw / gl
    equity = tdf["net_profit_bps"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    by_reason = {}
    for r in sorted(tdf["exit_reason"].unique()):
        rdf = tdf[tdf["exit_reason"] == r]
        by_reason[r] = {"n": len(rdf), "net_bps": round(rdf["net_profit_bps"].sum(), 1),
                        "avg": round(rdf["net_profit_bps"].mean(), 1)}
    return {"name": name, "n_trades": len(tdf), "win_pct": round(len(winners)/len(tdf)*100, 1),
            "total": round(tdf["net_profit_bps"].sum(), 0), "pf": round(pf, 2),
            "max_dd": round(max_dd, 0), "by_reason": by_reason}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config()
    df_15m = pd.read_parquet(DATA_15M); df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
    strategy = V12Strategy(cfg); df_15m = strategy.compute_indicators(df_15m)
    df_1m = pd.read_parquet(DATA_1M); df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    test_15m = df_15m["2024-01-01":"2025-12-31"]; test_1m = df_1m["2024-01-01":"2025-12-31"]
    v15 = MLV1(); test_v15 = v15.compute_features_from_df(test_15m.copy())
    signals = v15.generate_signals(test_v15)
    logger.info("V1.5 signals: %d", len(signals))

    trades = run_backtest(signals, test_v15, test_1m)
    s = summarize(trades, "PT_TRAIL5_HARD100")
    logger.info("  %d trades | Win %.1f%% | Net %+.0f | PF %.2f | DD %.0f",
                s["n_trades"], s["win_pct"], s["total"], s["pf"], s["max_dd"])

    print(f"\n{'='*72}")
    print(f"CONFIG: PT_TRAIL (5 bps, floor 60, hard target 100)")
    print('='*72)
    print(f"  Trades: {s['n_trades']} | Win: {s['win_pct']}%")
    print(f"  Net PnL: {s['total']:+.0f} bps | PF: {s['pf']} | DD {s['max_dd']:+.0f}")
    print(f"  By exit reason:")
    for r, d in s["by_reason"].items():
        print(f"    {r:<15} {d['n']:>4}t  Net {d['net_bps']:>+8.1f} bps  Avg {d['avg']:>+6.1f}")


if __name__ == "__main__":
    main()
