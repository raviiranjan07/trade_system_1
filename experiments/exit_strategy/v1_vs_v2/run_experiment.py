"""Compare V1 vs V2 exits on all 3 model configs with tick monitoring (honest).

V1 exits: 20/30 TS, tighten bar 5 to 8bps, bar 10 time exit. NO early cut, NO BE lock.
V2 exits: 20/30 TS, tighten bar 4 to 6bps, early cut bar 3/4, BE lock at MFE>=15.

Test period: 2024-2025 OOS, BTCUSDT 15m signals + 1m ticks.

Run: PYTHONPATH=src python experiments/exit_strategy/v1_vs_v2/run_experiment.py
"""

import copy
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from engine.config.loader import load_config
from engine.signals.ml_v1 import MLV1
from engine.signals.direction_attention import DirectionAttention
from engine.position_manager import V12PositionManager
from engine.strategy import V12Strategy, Direction

logger = logging.getLogger(__name__)

DATA_15M = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
DATA_1M = Path("data/raw/BTCUSDT_1m_ohlcv.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_DIR.mkdir(exist_ok=True)


def make_v1_config(base_cfg):
    """Clone config and set V1 exit rules (no early cut, no BE, bar 5 tighten to 8)."""
    v1_exit = base_cfg.exit.model_copy(update={
        "tighten_after_bar": 5,
        "tight_trailing_stop_bps": 8,
        "early_cut_bar3_mfe": 0,            # disabled (peak MFE >= 0 always)
        "early_cut_bar4_mfe": 0,            # disabled
        "breakeven_activation_mfe": 99999,  # disabled (never activates)
        "breakeven_floor_gross_bps": 0,
    })
    return base_cfg.model_copy(update={"exit": v1_exit})


def run_tick_backtest(signals, df_15m, df_1m, cfg):
    """Tick-monitoring backtest."""
    pm = V12PositionManager(cfg)
    times_15m = df_15m.index
    opens_15m = df_15m["open"].values
    highs_15m = df_15m["high"].values
    lows_15m = df_15m["low"].values
    closes_15m = df_15m["close"].values
    n_15m = len(df_15m)
    idx_1m = df_1m.index.values
    closes_1m = df_1m["close"].values
    signal_map = {s.bar_index: s for s in signals}

    i = 0
    while i < n_15m:
        if pm.is_in_position:
            bar_start = np.datetime64(times_15m[i])
            bar_end = bar_start + np.timedelta64(15, "m")
            s_idx = np.searchsorted(idx_1m, bar_start, side="left")
            e_idx = np.searchsorted(idx_1m, bar_end, side="left")
            exited = False
            for t in range(s_idx, e_idx):
                trade = pm.on_tick(closes_1m[t], idx_1m[t])
                if trade is not None:
                    exited = True
                    break
            if exited:
                i += 1
                continue
            pm.on_bar(highs_15m[i], lows_15m[i], closes_15m[i], times_15m[i], i)
            i += 1
            continue

        if pm.reentry_signal_type is not None and pm.can_reenter(i, True) and i + 1 < n_15m:
            pm.open_position(
                direction=pm.reentry_direction,
                signal_type=pm.reentry_signal_type,
                entry_price=opens_15m[i + 1],
                entry_time=times_15m[i + 1],
                signal_time=times_15m[i],
                is_reentry=True,
            )
            i += 2
            continue

        if i in signal_map and i + 1 < n_15m:
            sig = signal_map[i]
            pm.reset_reentry()
            pm.open_position(
                direction=sig.direction,
                signal_type=sig.signal_type,
                entry_price=opens_15m[i + 1],
                entry_time=times_15m[i + 1],
                signal_time=sig.timestamp,
            )
            i += 2
            continue

        i += 1

    return pm.trades


def summarize(trades, config_name):
    if not trades:
        return {"config": config_name, "status": "NO_TRADES"}
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
        by_reason[r] = {"n": len(rdf), "net_bps": round(rdf["net_profit_bps"].sum(), 1)}
    return {
        "config": config_name,
        "n_trades": len(tdf),
        "win_rate_pct": round(len(winners) / len(tdf) * 100, 1),
        "total_bps": round(tdf["net_profit_bps"].sum(), 1),
        "profit_factor": round(pf, 2),
        "avg_bps_per_month": round(tdf["net_profit_bps"].sum() / n_months, 1),
        "max_drawdown_bps": round(max_dd, 1),
        "by_exit_reason": by_reason,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    v2_cfg = load_config()
    v1_cfg = make_v1_config(v2_cfg)

    logger.info("Loading 15m + 1m data...")
    df_15m = pd.read_parquet(DATA_15M)
    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
    strategy = V12Strategy(v2_cfg)
    df_15m = strategy.compute_indicators(df_15m)
    df_1m = pd.read_parquet(DATA_1M)
    df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    test_15m = df_15m["2024-01-01":"2025-12-31"]
    test_1m = df_1m["2024-01-01":"2025-12-31"]
    logger.info("15m bars: %d | 1m bars: %d", len(test_15m), len(test_1m))

    # Precompute signals once
    logger.info("Computing signals...")
    v15_gen = MLV1()
    test_v15 = v15_gen.compute_features_from_df(test_15m.copy())
    v15_signals = v15_gen.generate_signals(test_v15)

    attn_gen_040 = DirectionAttention()
    attn_gen_040.conf_short = 0.60  # SHORT < 0.40
    test_attn_040 = test_15m.copy()
    attn_gen_040.compute_features(test_attn_040)
    attn_signals_040 = [s for i in range(len(test_attn_040)) if (s := attn_gen_040.predict_bar(test_attn_040, i)) is not None]

    attn_gen_035 = DirectionAttention()
    attn_gen_035.conf_short = 0.65  # SHORT < 0.35
    test_attn_035 = test_15m.copy()
    attn_gen_035.compute_features(test_attn_035)
    attn_signals_035 = [s for i in range(len(test_attn_035)) if (s := attn_gen_035.predict_bar(test_attn_035, i)) is not None]

    logger.info("V1.5 signals: %d | Attn 0.40: %d | Attn 0.35: %d",
                len(v15_signals), len(attn_signals_040), len(attn_signals_035))

    model_configs = [
        ("v15", v15_signals, test_v15),
        ("attn_040", attn_signals_040, test_attn_040),
        ("attn_035", attn_signals_035, test_attn_035),
    ]
    exit_configs = [("V1", v1_cfg), ("V2", v2_cfg)]

    results = []
    for model_name, signals, df in model_configs:
        for exit_name, exit_cfg in exit_configs:
            cfg_name = f"{model_name}_{exit_name}"
            logger.info("=== %s ===", cfg_name)
            trades = run_tick_backtest(signals, df, test_1m, exit_cfg)
            s = summarize(trades, cfg_name)
            results.append(s)
            if s.get("status") == "NO_TRADES":
                continue
            logger.info("  %s: %d trades, %+.0f bps, PF %.2f",
                        cfg_name, s["n_trades"], s["total_bps"], s["profit_factor"])

    # Final comparison table
    print(f"\n{'='*105}")
    print("V1 vs V2 EXITS on 3 MODELS (tick-monitoring)")
    print('='*105)
    print(f"{'Config':<20}{'Trades':>8}{'Win%':>8}{'Net bps':>12}{'PF':>8}{'Avg/Mo':>10}{'MaxDD':>10}  Exit reasons")
    print('-'*105)
    for r in results:
        if r.get("status") == "NO_TRADES":
            continue
        reasons = " | ".join(f"{k}:{v['n']}({v['net_bps']:+.0f})" for k, v in r["by_exit_reason"].items())
        print(f"{r['config']:<20}{r['n_trades']:>8}{r['win_rate_pct']:>8}"
              f"{r['total_bps']:>+12.0f}{r['profit_factor']:>8.2f}"
              f"{r['avg_bps_per_month']:>+10.0f}{r['max_drawdown_bps']:>+10.0f}  {reasons}")

    import json
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s", RESULTS_DIR / "summary.json")


if __name__ == "__main__":
    main()
