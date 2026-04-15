"""Sweep TIME_EXIT bar from 1-9 (BE_LOCK disabled).

Rules:
  1. PROFIT_TARGET (60 bps, bars <= 5)
  2. LOCKED_PROFIT (arm 20, trigger 15)
  3. TIGHT_TS (bar 6+, pnl >= 15, drawdown 8)
  4. TIME_EXIT at MAX_BARS (varies 1-9)

BE_LOCK is DISABLED.

Run: PYTHONPATH=src python experiments/exit_strategy/bar_sweep/run_experiment.py
"""

import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from engine.config.loader import load_config
from engine.signals.direction_v15 import DirectionV15
from engine.strategy import V12Strategy, Direction
from engine.config.constants import FEES_BPS

logger = logging.getLogger(__name__)

DATA_15M = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
DATA_1M = Path("data/raw/BTCUSDT_1m_ohlcv.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_DIR.mkdir(exist_ok=True)

LOCK_ARM_BPS = 15
LOCK_TRIGGER_BPS = 15
TIGHT_BPS = 8
TIGHTEN_AFTER_BAR = 5
PT_ARM_BPS = 60     # arm PROFIT_TARGET once peak hits 60
PT_TARGET_BPS = 80  # take profit at 80
PT_LOCK_BPS = 60    # lock profit at 60 if pulls back
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


def run_backtest(signals, df_15m, df_1m, max_bars):
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
                # PT arm once peak touches 60 within 5 bars
                if pos["peak"] >= PT_ARM_BPS and pos["bars_held"] <= PROFIT_TARGET_MAX_BAR:
                    pos["pt_armed"] = True

                # Trailing stop: peak-activated at 20 bps, then trails at 30 bps
                ts_width = 30
                if pos["peak"] >= 20:
                    pos["ts_armed"] = True

                # 1a. PT_TARGET: take profit at 80 (use tick price — can gap above 80)
                if pos.get("pt_armed") and tick_pnl >= PT_TARGET_BPS:
                    exit_reason = "PT_TARGET"; exit_price = tick_price; exit_time = idx_1m[t]; break

                # 1b. PT_LOCK: simulate stop order at 60 bps — fires AT the 60 level (idealized)
                if pos.get("pt_armed") and tick_pnl <= PT_LOCK_BPS:
                    if pos["direction"] == "LONG":
                        exit_price_at_lock = pos["entry_price"] * (1 + PT_LOCK_BPS / 10000)
                    else:
                        exit_price_at_lock = pos["entry_price"] * (1 - PT_LOCK_BPS / 10000)
                    exit_reason = "PT_LOCK"; exit_price = exit_price_at_lock; exit_time = idx_1m[t]; break

                # 2. TRAILING_STOP: armed once peak >= 20, then trail at 30 bps
                drawdown = pos["peak"] - tick_pnl
                if pos.get("ts_armed") and drawdown >= ts_width:
                    exit_reason = "TRAILING_STOP"; exit_price = tick_price; exit_time = idx_1m[t]; break

                # 3. LOCKED_PROFIT (secondary — fires if price holds 15 zone)
                if pos["armed"] and tick_pnl <= LOCK_TRIGGER_BPS:
                    exit_reason = "LOCKED_PROFIT"; exit_price = tick_price; exit_time = idx_1m[t]; break

                # 4. TIGHT_TS (only pnl >= 15 after bar 5)
                if pos["bars_held"] > TIGHTEN_AFTER_BAR and tick_pnl >= BAR6_MIN_BPS:
                    if drawdown >= TIGHT_BPS:
                        exit_reason = "TIGHT_TS"; exit_price = tick_price; exit_time = idx_1m[t]; break

            if exit_reason is None:
                pos["bars_held"] += 1
                if pos["bars_held"] >= max_bars:
                    bar_close = closes_15m[i]
                    if pos["direction"] == "LONG":
                        bc_pnl = (bar_close - pos["entry_price"]) / pos["entry_price"] * 10000
                    else:
                        bc_pnl = (pos["entry_price"] - bar_close) / pos["entry_price"] * 10000
                    exit_reason = "NO_ZONE" if bc_pnl >= 0 else "TIME_EXIT"
                    exit_price = bar_close
                    exit_time = times_15m[i]

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
                pos = None
                i += 1
                continue

            i += 1
            continue

        if i in signal_map and i + 1 < n_15m:
            sig = signal_map[i]
            pos = {
                "direction": sig.direction.value, "signal_type": sig.signal_type.value,
                "entry_price": opens_15m[i + 1], "entry_time": times_15m[i + 1],
                "signal_time": sig.timestamp,
                "bars_held": 0, "peak": 0.0, "mfe_bps": 0.0, "mae_bps": 0.0,
                "armed": False,
                "pt_armed": False,
                "ts_armed": False,
            }
            i += 2
            continue
        i += 1
    return trades


def summarize(trades, max_bars):
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
        by_reason[r] = {
            "n": int(len(rdf)),
            "net_bps": round(rdf["net_profit_bps"].sum(), 1),
        }
    return {
        "max_bars": max_bars,
        "n_trades": len(tdf),
        "win_rate_pct": round(len(winners) / len(tdf) * 100, 1),
        "total_bps": round(tdf["net_profit_bps"].sum(), 1),
        "profit_factor": round(pf, 2),
        "max_drawdown_bps": round(max_dd, 1),
        "by_reason": by_reason,
    }


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

    v15_gen = DirectionV15()
    test_v15 = v15_gen.compute_features_from_df(test_15m.copy())
    signals = v15_gen.generate_signals(test_v15)
    logger.info("V1.5 signals: %d", len(signals))

    results = []
    for max_bars in range(1, 10):
        logger.info("=== MAX_BARS = %d ===", max_bars)
        trades = run_backtest(signals, test_v15, test_1m, max_bars)
        s = summarize(trades, max_bars)
        results.append(s)
        logger.info("  %d trades, %+.0f bps, PF %.2f, DD %.0f",
                    s["n_trades"], s["total_bps"], s["profit_factor"], s["max_drawdown_bps"])

    # Comparison table
    print(f"\n{'='*90}")
    print("TIME_EXIT BAR SWEEP (BE_LOCK disabled)")
    print('='*90)
    all_reasons = sorted({k for r in results for k in r["by_reason"].keys()})
    header = f"{'Bar':<4}{'Trades':>7}{'Win%':>6}{'Net':>9}{'PF':>6}"
    for rn in all_reasons:
        header += f"{rn[:10]:>12}"
    print(header)
    print('-'*len(header))
    for r in results:
        row = f"{r['max_bars']:<4}{r['n_trades']:>7}{r['win_rate_pct']:>6}"
        row += f"{r['total_bps']:>+9.0f}{r['profit_factor']:>6.2f}"
        for rn in all_reasons:
            d = r["by_reason"].get(rn)
            if d:
                row += f"{d['n']:>4}/{d['net_bps']:>+6.0f}"
            else:
                row += f"{'':>12}"
        print(row)

    import json
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
