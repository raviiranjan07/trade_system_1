"""HONEST Backtest: Uses bar_close_pnl for trailing stop exits (not theoretical).

Fixes the bug in position_manager.py where trailing stop exit price is back-calculated
from peak-trailing math instead of actual bar close. This matches what live trading
on a real exchange would actually achieve.

Tests 3 configurations separately:
  1. ML V1.5 only + V2 exits (honest)
  2. ML V2 Attention only + V2 exits, SHORT 0.40 (honest)
  3. ML V2 Attention only + V2 exits, SHORT 0.35 (honest)

Run: PYTHONPATH=src python experiments/exit_strategy/honest_exits/run_experiment.py
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from engine.config.loader import load_config
from engine.signals.direction_v15 import DirectionV15
from engine.signals.direction_attention import DirectionAttention
from engine.strategy import V12Strategy, Direction, SignalType
from engine.config.constants import FEES_BPS

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class HonestTrade:
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


def run_honest_backtest(
    signals: list,
    df: pd.DataFrame,
    cfg,
) -> list:
    """Walk bars, apply exits, use BAR CLOSE PRICE as exit price (honest)."""
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    times = df.index

    signal_map = {s.bar_index: s for s in signals}
    n = len(df)
    trades = []
    pos = None
    i = 0

    while i < n:
        if pos is not None:
            # Bar update
            pos["bars_held"] += 1
            h, l, c = highs[i], lows[i], closes[i]

            if pos["direction"] == "LONG":
                bar_mfe = (h - pos["entry_price"]) / pos["entry_price"] * 10000
                bar_mae = (l - pos["entry_price"]) / pos["entry_price"] * 10000
                close_pnl = (c - pos["entry_price"]) / pos["entry_price"] * 10000
            else:
                bar_mfe = (pos["entry_price"] - l) / pos["entry_price"] * 10000
                bar_mae = (pos["entry_price"] - h) / pos["entry_price"] * 10000
                close_pnl = (pos["entry_price"] - c) / pos["entry_price"] * 10000

            if bar_mfe > pos["mfe_bps"]:
                pos["mfe_bps"] = bar_mfe
            if bar_mae < pos["mae_bps"]:
                pos["mae_bps"] = bar_mae
            if bar_mfe > pos["peak"]:
                pos["peak"] = bar_mfe

            # V2 exit rules
            be_active = pos["peak"] >= cfg.exit.breakeven_activation_mfe
            be_floor = cfg.exit.breakeven_floor_gross_bps

            # Trailing stop
            if pos["bars_held"] > cfg.exit.tighten_after_bar:
                active_ts = cfg.exit.tight_trailing_stop_bps
            else:
                active_ts = pos["trailing_stop"]

            drawdown = pos["peak"] - close_pnl
            exit_reason = None
            exit_profit = None

            # Trailing stop triggered — USE BAR CLOSE (honest)
            if drawdown >= active_ts and pos["peak"] > 0:
                exit_reason = "TIGHT_TS" if pos["bars_held"] > cfg.exit.tighten_after_bar else "TRAILING_STOP"
                exit_profit = close_pnl  # HONEST: use bar close
                # If BE was active and close is above be_floor, honor the floor
                if be_active and close_pnl < be_floor and close_pnl > 0:
                    # Close is above 0 but below floor — in honest mode, use actual close
                    pass  # leave as close_pnl (could be worse than floor in reality)

            # BE lock (separate check)
            elif be_active and close_pnl <= be_floor:
                exit_reason = "BE_LOCK"
                exit_profit = close_pnl  # HONEST: use actual close, not floor

            # Early cut (already honest — uses close_pnl)
            elif pos["bars_held"] == 3 and pos["peak"] < cfg.exit.early_cut_bar3_mfe:
                exit_reason = "EARLY_CUT"
                exit_profit = close_pnl

            elif pos["bars_held"] == 4 and pos["peak"] < cfg.exit.early_cut_bar4_mfe:
                exit_reason = "EARLY_CUT"
                exit_profit = close_pnl

            # Time exit
            elif pos["bars_held"] >= cfg.exit.max_bars:
                exit_reason = "TIME_EXIT"
                exit_profit = close_pnl

            if exit_reason:
                exit_price = c  # actual bar close price
                trades.append(HonestTrade(
                    signal_time=pos["signal_time"],
                    entry_time=pos["entry_time"],
                    exit_time=times[i],
                    direction=pos["direction"],
                    signal_type=pos["signal_type"],
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    gross_profit_bps=exit_profit,
                    net_profit_bps=exit_profit - FEES_BPS,
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

        # Not in position — check signal
        if i in signal_map and i + 1 < n:
            sig = signal_map[i]
            entry_price = opens[i + 1]
            ts = cfg.exit.long_trailing_stop_bps if sig.direction == Direction.LONG else cfg.exit.short_trailing_stop_bps
            pos = {
                "direction": sig.direction.value,
                "signal_type": sig.signal_type.value,
                "entry_price": entry_price,
                "entry_time": times[i + 1],
                "signal_time": sig.timestamp,
                "trailing_stop": ts,
                "bars_held": 0,
                "peak": 0.0,
                "mfe_bps": 0.0,
                "mae_bps": 0.0,
            }
            i += 2
            continue

        i += 1

    return trades


def summarize(trades, config_name):
    if not trades:
        return {"config": config_name, "status": "NO_TRADES"}
    import pandas as pd
    from dataclasses import asdict
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

    # By exit reason
    by_reason = {}
    for r in sorted(tdf["exit_reason"].unique()):
        rdf = tdf[tdf["exit_reason"] == r]
        by_reason[r] = {
            "n": len(rdf),
            "net_bps": round(rdf["net_profit_bps"].sum(), 1),
            "win_rate": round((rdf["net_profit_bps"] > 0).sum() / len(rdf) * 100, 1),
        }

    return {
        "config": config_name,
        "n_trades": len(tdf),
        "win_rate_pct": round(len(winners) / len(tdf) * 100, 1),
        "total_bps": round(tdf["net_profit_bps"].sum(), 1),
        "profit_factor": round(pf, 2),
        "avg_bps_per_trade": round(tdf["net_profit_bps"].mean(), 2),
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
    print(f"  Trades: {s['n_trades']} | Win Rate: {s['win_rate_pct']}%")
    print(f"  Net PnL: {s['total_bps']:+.0f} bps | PF: {s['profit_factor']}")
    print(f"  Avg/Trade: {s['avg_bps_per_trade']:+.1f} bps | Avg/Month: {s['avg_bps_per_month']:+.0f} bps")
    print(f"  Max DD: {s['max_drawdown_bps']:+.0f} bps")
    print(f"  By exit reason:")
    for r, d in s["by_exit_reason"].items():
        print(f"    {r:<18} {d['n']:>4}t | Win: {d['win_rate']:>5.1f}% | Net: {d['net_bps']:>+8.1f} bps")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()

    # Load data
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    strategy = V12Strategy(config)
    df = strategy.compute_indicators(df)
    test = df["2024-01-01":"2025-12-31"]
    logger.info("Test period: %s to %s (%d bars)", test.index[0], test.index[-1], len(test))

    results = []

    # === Config 1: ML V1.5 only + V2 exits (honest) ===
    logger.info("=== Config 1: ML V1.5 only + V2 exits (HONEST) ===")
    v15_gen = DirectionV15()
    test_v15 = v15_gen.compute_features_from_df(test.copy())
    v15_signals = v15_gen.generate_signals(test_v15)
    logger.info("V1.5 signals: %d", len(v15_signals))
    trades = run_honest_backtest(v15_signals, test_v15, config)
    s = summarize(trades, "ml_v15_only_v2exit_honest")
    print_summary(s)
    results.append(s)

    # === Config 2: ML V2 Attention only + V2 exits, SHORT 0.40 (honest) ===
    logger.info("=== Config 2: ML V2 (Attention) only + V2 exits, SHORT=0.40 (HONEST) ===")
    attn_gen_040 = DirectionAttention()
    attn_gen_040.conf_short = 0.60  # SHORT triggers when prob < 0.40
    test_attn = test.copy()
    attn_gen_040.compute_features(test_attn)
    attn_signals_040 = []
    for idx in range(len(test_attn)):
        sig = attn_gen_040.predict_bar(test_attn, idx)
        if sig is not None:
            attn_signals_040.append(sig)
    logger.info("Attention signals (SHORT=0.40): %d", len(attn_signals_040))
    trades = run_honest_backtest(attn_signals_040, test_attn, config)
    s = summarize(trades, "ml_v2_attn_only_v2exit_short040_honest")
    print_summary(s)
    results.append(s)

    # === Config 3: ML V2 Attention only + V2 exits, SHORT 0.35 (honest) ===
    logger.info("=== Config 3: ML V2 (Attention) only + V2 exits, SHORT=0.35 (HONEST) ===")
    attn_gen_035 = DirectionAttention()
    attn_gen_035.conf_short = 0.65  # SHORT triggers when prob < 0.35
    test_attn2 = test.copy()
    attn_gen_035.compute_features(test_attn2)
    attn_signals_035 = []
    for idx in range(len(test_attn2)):
        sig = attn_gen_035.predict_bar(test_attn2, idx)
        if sig is not None:
            attn_signals_035.append(sig)
    logger.info("Attention signals (SHORT=0.35): %d", len(attn_signals_035))
    trades = run_honest_backtest(attn_signals_035, test_attn2, config)
    s = summarize(trades, "ml_v2_attn_only_v2exit_short035_honest")
    print_summary(s)
    results.append(s)

    # Comparison
    print(f"\n{'='*95}")
    print("HONEST RESULTS COMPARISON (uses bar_close for trailing stop exits)")
    print('='*95)
    print(f"{'Config':<42}{'Trades':>8}{'Win%':>8}{'Net bps':>11}{'PF':>8}{'Avg/Mo':>10}{'MaxDD':>10}")
    print('-'*95)
    for r in results:
        if r.get("status") == "NO_TRADES":
            print(f"{r['config']:<42}  NO TRADES")
            continue
        print(f"{r['config']:<42}{r['n_trades']:>8}{r['win_rate_pct']:>8}"
              f"{r['total_bps']:>+11.0f}{r['profit_factor']:>8.2f}"
              f"{r['avg_bps_per_month']:>+10.0f}{r['max_drawdown_bps']:>+10.0f}")

    import json
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s", RESULTS_DIR / "summary.json")


if __name__ == "__main__":
    main()
