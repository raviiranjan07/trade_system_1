"""Test Option A (medium trailing) and Option B (dynamic LP trigger).

Baseline: +2,235 bps, PF 1.08 at bar 6

Option A: add MID_TRAIL rule between LOCKED_PROFIT and PT_ARM
    - arm at peak >= 25
    - trail at 10 bps (exit when drawdown >= 10)

Option B: dynamic LOCKED_PROFIT
    - Replace static trigger at 15 with dynamic (peak - 10)
    - Once peak >= 15, exit when peak - tick_pnl >= 10

Run: PYTHONPATH=src python experiments/exit_strategy/dyn_trail/run_experiment.py
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
from engine.signals.direction_attention import DirectionAttention
from engine.strategy import V12Strategy, Direction
from engine.config.constants import FEES_BPS

logger = logging.getLogger(__name__)

DATA_15M = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
DATA_1M = Path("data/raw/BTCUSDT_1m_ohlcv.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_DIR.mkdir(exist_ok=True)

PT_ARM_BPS = 60
PT_TARGET_BPS = 80
PT_LOCK_BPS = 60
PROFIT_TARGET_MAX_BAR = 5
LOCK_ARM_BPS = 15
LOCK_TRIGGER_BPS = 15
MAX_BARS = 6


@dataclass
class Trade:
    signal_time: object; entry_time: object; exit_time: object
    direction: str; signal_type: str
    entry_price: float; exit_price: float
    gross_profit_bps: float; net_profit_bps: float
    mfe_bps: float; mae_bps: float
    exit_bar: int; exit_reason: str


def run_backtest(signals, df_15m, df_1m, mode: str):
    """mode: 'baseline', 'optionA', 'optionB'"""
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

                # 1a. PT_TARGET at 80
                if pos.get("pt_armed") and tick_pnl >= PT_TARGET_BPS:
                    exit_reason = "PT_TARGET"; exit_price = tick_price; exit_time = idx_1m[t]; break

                # 1b. PT_LOCK at 60 (stop order)
                if pos.get("pt_armed") and tick_pnl <= PT_LOCK_BPS:
                    if pos["direction"] == "LONG":
                        exit_price_at_lock = pos["entry_price"] * (1 + PT_LOCK_BPS / 10000)
                    else:
                        exit_price_at_lock = pos["entry_price"] * (1 - PT_LOCK_BPS / 10000)
                    exit_reason = "PT_LOCK"; exit_price = exit_price_at_lock; exit_time = idx_1m[t]; break

                # Option A variants: MID_TRAIL (arm 25, trail X) — IDEALIZED EXIT at peak-X
                if mode.startswith("optionA"):
                    mt_trail = int(mode.split("_")[-1])
                    if pos["peak"] >= 25 and not pos.get("pt_armed"):
                        pos["mt_armed"] = True
                    if pos.get("mt_armed"):
                        drawdown = pos["peak"] - tick_pnl
                        if drawdown >= mt_trail:
                            # Idealized exit at peak - trail level
                            exit_at_bps = pos["peak"] - mt_trail
                            if pos["direction"] == "LONG":
                                exit_price_ideal = pos["entry_price"] * (1 + exit_at_bps / 10000)
                            else:
                                exit_price_ideal = pos["entry_price"] * (1 - exit_at_bps / 10000)
                            exit_reason = "MID_TRAIL"; exit_price = exit_price_ideal; exit_time = idx_1m[t]; break

                if mode == "optionB":
                    if pos["armed"]:
                        drawdown = pos["peak"] - tick_pnl
                        if drawdown >= 10:
                            exit_reason = "LOCKED_PROFIT"; exit_price = tick_price; exit_time = idx_1m[t]; break
                else:
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
                "armed": False, "pt_armed": False, "mt_armed": False,
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
    # ML V1.5
    v15 = MLV1(); test_v15 = v15.compute_features_from_df(test_15m.copy())
    signals_v15 = v15.generate_signals(test_v15)
    # ML V2 attention (SHORT 0.40 + 0.35)
    attn_040 = DirectionAttention(); attn_040.conf_short = 0.60
    test_attn_040 = test_15m.copy(); attn_040.compute_features(test_attn_040)
    signals_attn_040 = [s for i in range(len(test_attn_040)) if (s := attn_040.predict_bar(test_attn_040, i)) is not None]
    attn_035 = DirectionAttention(); attn_035.conf_short = 0.65
    test_attn_035 = test_15m.copy(); attn_035.compute_features(test_attn_035)
    signals_attn_035 = [s for i in range(len(test_attn_035)) if (s := attn_035.predict_bar(test_attn_035, i)) is not None]
    logger.info("V1.5: %d | Attn 0.40: %d | Attn 0.35: %d",
                len(signals_v15), len(signals_attn_040), len(signals_attn_035))

    all_results = []
    for model_name, sigs, df in [
        ("V1.5", signals_v15, test_v15),
        ("Attn_0.40", signals_attn_040, test_attn_040),
        ("Attn_0.35", signals_attn_035, test_attn_035),
    ]:
        mode = "optionA_10"
        logger.info("=== %s + %s ===", model_name, mode)
        trades = run_backtest(sigs, df, test_1m, mode)
        s = summarize(trades, f"{model_name}_{mode}")
        all_results.append(s)
        logger.info("  %d trades | Win %.1f%% | Net %+.0f | PF %.2f | DD %.0f",
                    s["n_trades"], s["win_pct"], s["total"], s["pf"], s["max_dd"])
        tdf = pd.DataFrame([asdict(t) for t in trades])
        tdf.to_csv(RESULTS_DIR / f"trades_{model_name}.csv", index=False)

        for r, d in s["by_reason"].items():
            logger.info("    %-15s %4dt  %+.1f bps (avg %+.1f)", r, d["n"], d["net_bps"], d["avg"])

    # Final comparison
    print(f"\n{'='*80}")
    print("BEST EXIT CONFIG (optionA_10) APPLIED TO ALL 3 MODELS")
    print('='*80)
    print(f"{'Model':<14}{'Trades':>8}{'Win%':>8}{'Net bps':>12}{'PF':>8}{'MaxDD':>10}")
    print('-'*80)
    for s in all_results:
        print(f"{s['name']:<14}{s['n_trades']:>8}{s['win_pct']:>8}{s['total']:>+12.0f}{s['pf']:>8.2f}{s['max_dd']:>+10.0f}")


if __name__ == "__main__":
    main()
