"""Sweep PT_TARGET values: 80, 90, 100, 120.

Everything else kept at V3 production values:
  LOCK_ARM=15, LOCK_TRIGGER=15
  MID_TRAIL arm=25, width=10
  PT_ARM=60, PT_LOCK=60, PT_MAX_BAR=5
  MAX_BARS=6

Only PT_TARGET varies.

Run: PYTHONPATH=src python experiments/exit_strategy/pt_target_sweep/run_experiment.py
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
PT_LOCK_BPS = 60
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


def run_backtest(signals, df_15m, df_1m, pt_target_bps: float):
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

                # PT_TARGET (variable)
                if pos.get("pt_armed") and tick_pnl >= pt_target_bps:
                    exit_reason = "PT_TARGET"; exit_price = tick_price; exit_time = idx_1m[t]; break

                # PT_LOCK
                if pos.get("pt_armed") and tick_pnl <= PT_LOCK_BPS:
                    if pos["direction"] == "LONG":
                        exit_price_ideal = pos["entry_price"] * (1 + PT_LOCK_BPS / 10000)
                    else:
                        exit_price_ideal = pos["entry_price"] * (1 - PT_LOCK_BPS / 10000)
                    exit_reason = "PT_LOCK"; exit_price = exit_price_ideal; exit_time = idx_1m[t]; break

                # MID_TRAIL
                if pos["peak"] >= MID_TRAIL_ARM and not pos.get("pt_armed"):
                    drawdown = pos["peak"] - tick_pnl
                    if drawdown >= MID_TRAIL_WIDTH:
                        exit_at_bps = pos["peak"] - MID_TRAIL_WIDTH
                        if pos["direction"] == "LONG":
                            exit_price_ideal = pos["entry_price"] * (1 + exit_at_bps / 10000)
                        else:
                            exit_price_ideal = pos["entry_price"] * (1 - exit_at_bps / 10000)
                        exit_reason = "MID_TRAIL"; exit_price = exit_price_ideal; exit_time = idx_1m[t]; break

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


def summarize(trades, pt_target):
    tdf = pd.DataFrame([asdict(t) for t in trades])
    winners = tdf[tdf["net_profit_bps"] > 0]
    gw = winners["net_profit_bps"].sum()
    gl = abs(tdf[tdf["net_profit_bps"] <= 0]["net_profit_bps"].sum()) or 1
    pf = gw / gl
    equity = tdf["net_profit_bps"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    pt_df = tdf[tdf["exit_reason"] == "PT_TARGET"]
    return {
        "pt_target": pt_target,
        "n_trades": len(tdf),
        "win_pct": round(len(winners)/len(tdf)*100, 1),
        "total": round(tdf["net_profit_bps"].sum(), 0),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 0),
        "pt_count": len(pt_df),
        "pt_total": round(pt_df["net_profit_bps"].sum(), 0),
        "pt_avg": round(pt_df["net_profit_bps"].mean(), 1) if len(pt_df) else 0,
    }


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

    results = []
    for pt in [80, 90, 100, 120, 150]:
        logger.info("=== PT_TARGET = %d ===", pt)
        trades = run_backtest(signals, test_v15, test_1m, pt)
        s = summarize(trades, pt)
        results.append(s)

    print(f"\n{'='*90}")
    print("PT_TARGET SWEEP")
    print('='*90)
    print(f"{'PT_TARGET':<12}{'Trades':>8}{'Win%':>7}{'Net bps':>12}{'PF':>7}{'MaxDD':>10}{'PT cnt':>8}{'PT tot':>10}{'PT avg':>9}")
    print('-'*90)
    for r in results:
        print(f"{r['pt_target']:<12}{r['n_trades']:>8}{r['win_pct']:>7}{r['total']:>+12.0f}{r['pf']:>7.2f}"
              f"{r['max_dd']:>+10.0f}{r['pt_count']:>8}{r['pt_total']:>+10.0f}{r['pt_avg']:>+9.1f}")


if __name__ == "__main__":
    main()
