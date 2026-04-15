"""V2 exits WITHOUT early cut — tick monitoring (honest).

Keeps: trailing stop 20/30, tighten bar4 to 6bps, BE lock MFE>=15.
Removes: early cut at bar 3/4.
Keeps: time exit at bar 10.

Test period: 2024-2025 OOS, 3 models.

Run: PYTHONPATH=src python experiments/exit_strategy/no_early_cut/run_experiment.py
"""

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from engine.config.loader import load_config
from engine.signals.direction_v15 import DirectionV15
from engine.signals.direction_attention import DirectionAttention
from engine.position_manager import V12PositionManager
from engine.strategy import V12Strategy

logger = logging.getLogger(__name__)

DATA_15M = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
DATA_1M = Path("data/raw/BTCUSDT_1m_ohlcv.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_DIR.mkdir(exist_ok=True)


def make_no_early_cut_config(base_cfg):
    """V2 config without early cut rules."""
    new_exit = base_cfg.exit.model_copy(update={
        "early_cut_bar3_mfe": 0,    # disabled
        "early_cut_bar4_mfe": 0,    # disabled
    })
    return base_cfg.model_copy(update={"exit": new_exit})


def run_tick_backtest(signals, df_15m, df_1m, cfg):
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
        by_reason[r] = {
            "n": len(rdf),
            "net_bps": round(rdf["net_profit_bps"].sum(), 1),
            "win_rate": round((rdf["net_profit_bps"] > 0).sum() / len(rdf) * 100, 1),
            "avg_bps": round(rdf["net_profit_bps"].mean(), 1),
        }
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


def print_summary(s):
    print(f"\n{'='*70}")
    print(f"CONFIG: {s['config']}")
    print('='*70)
    if s.get("status") == "NO_TRADES":
        print("NO TRADES")
        return
    print(f"  Trades: {s['n_trades']} | Win: {s['win_rate_pct']}%")
    print(f"  Net PnL: {s['total_bps']:+.0f} bps | PF: {s['profit_factor']} | Avg/Mo: {s['avg_bps_per_month']:+.0f}")
    print(f"  Max DD: {s['max_drawdown_bps']:+.0f}")
    print(f"  By exit reason:")
    for r, d in s["by_exit_reason"].items():
        print(f"    {r:<18} {d['n']:>4}t | Win: {d['win_rate']:>5.1f}% | Net: {d['net_bps']:>+8.1f} | Avg: {d['avg_bps']:>+6.1f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    base_cfg = load_config()
    cfg = make_no_early_cut_config(base_cfg)
    logger.info("Config: V2 - early_cut (keeping BE lock, tighten bar4=6, time exit bar10)")

    df_15m = pd.read_parquet(DATA_15M)
    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
    strategy = V12Strategy(base_cfg)
    df_15m = strategy.compute_indicators(df_15m)
    df_1m = pd.read_parquet(DATA_1M)
    df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    test_15m = df_15m["2024-01-01":"2025-12-31"]
    test_1m = df_1m["2024-01-01":"2025-12-31"]

    results = []

    # V1.5
    logger.info("=== V1.5 + V2-no-early-cut ===")
    v15_gen = DirectionV15()
    test_v15 = v15_gen.compute_features_from_df(test_15m.copy())
    v15_signals = v15_gen.generate_signals(test_v15)
    logger.info("V1.5 signals: %d", len(v15_signals))
    trades = run_tick_backtest(v15_signals, test_v15, test_1m, cfg)
    s = summarize(trades, "v15_no_early_cut")
    print_summary(s); results.append(s)

    # Attention 0.40
    logger.info("=== Attention 0.40 + V2-no-early-cut ===")
    attn_gen = DirectionAttention()
    attn_gen.conf_short = 0.60
    test_attn = test_15m.copy()
    attn_gen.compute_features(test_attn)
    attn_signals = [s for i in range(len(test_attn)) if (s := attn_gen.predict_bar(test_attn, i)) is not None]
    logger.info("Attention 0.40 signals: %d", len(attn_signals))
    trades = run_tick_backtest(attn_signals, test_attn, test_1m, cfg)
    s = summarize(trades, "attn_040_no_early_cut")
    print_summary(s); results.append(s)

    # Attention 0.35
    logger.info("=== Attention 0.35 + V2-no-early-cut ===")
    attn_gen2 = DirectionAttention()
    attn_gen2.conf_short = 0.65
    test_attn2 = test_15m.copy()
    attn_gen2.compute_features(test_attn2)
    attn_signals2 = [s for i in range(len(test_attn2)) if (s := attn_gen2.predict_bar(test_attn2, i)) is not None]
    logger.info("Attention 0.35 signals: %d", len(attn_signals2))
    trades = run_tick_backtest(attn_signals2, test_attn2, test_1m, cfg)
    s = summarize(trades, "attn_035_no_early_cut")
    print_summary(s); results.append(s)

    # Final comparison
    print(f"\n{'='*100}")
    print("V2 EXITS WITHOUT EARLY CUT — tick monitoring")
    print('='*100)
    print(f"{'Config':<28}{'Trades':>8}{'Win%':>8}{'Net bps':>12}{'PF':>8}{'Avg/Mo':>10}{'MaxDD':>10}")
    print('-'*100)
    for r in results:
        if r.get("status") == "NO_TRADES":
            continue
        print(f"{r['config']:<28}{r['n_trades']:>8}{r['win_rate_pct']:>8}"
              f"{r['total_bps']:>+12.0f}{r['profit_factor']:>8.2f}"
              f"{r['avg_bps_per_month']:>+10.0f}{r['max_drawdown_bps']:>+10.0f}")

    import json
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s", RESULTS_DIR / "summary.json")


if __name__ == "__main__":
    main()
