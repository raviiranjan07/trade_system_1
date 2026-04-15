"""Backtest: ML V2 (Attention) with V2 exit rules.

Tests 4 configurations:
  1. Attention only, SHORT threshold 0.40
  2. Attention only, SHORT threshold 0.35
  3. Combined (V1.4 + V1.5 + Attention), SHORT 0.40
  4. Combined, SHORT 0.35

All with V2 exit rules (from settings.yaml: early cut, BE lock, tighten bar4+ 6bps).
Test period: 2024-2025 OOS.

Run: PYTHONPATH=src python experiments/exit_strategy/ml_v2_v2exits/run_experiment.py
"""

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from engine.config.loader import load_config
from engine.config.schema import AppConfig
from engine.signals.direction_v15 import DirectionV15
from engine.signals.direction_attention import DirectionAttention
from engine.position_manager import V12PositionManager
from engine.strategy import V12Strategy, Direction, SignalType

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
RESULTS_DIR = Path(__file__).resolve().parent / "runs"
RESULTS_DIR.mkdir(exist_ok=True)


def _regime_valid(signal_type: SignalType, bull, bear, idx: int) -> bool:
    if signal_type in (SignalType.ML_LONG, SignalType.ML_SHORT,
                       SignalType.ML_ATTN_LONG, SignalType.ML_ATTN_SHORT):
        return True
    if signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
        return bool(bull[idx])
    return bool(bear[idx])


def run_backtest(
    config: AppConfig,
    include_v14: bool = True,
    include_v15: bool = True,
    include_attention: bool = True,
    attention_short_threshold: float = 0.40,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
):
    """Run backtest with configurable signal sources and thresholds."""
    # Load data
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Compute indicators
    strategy = V12Strategy(config)
    df = strategy.compute_indicators(df)
    test = df[start:end]

    # Pre-extract arrays
    highs = test["high"].values
    lows = test["low"].values
    closes = test["close"].values
    opens = test["open"].values
    times = test.index
    bull = test["bull_market"].values
    bear = test["bear_market"].values

    signals = []

    # V1.4 signals
    if include_v14:
        v14_signals = strategy.generate_signals(test)
        logger.info("V1.4 signals: %d", len(v14_signals))
        signals.extend(v14_signals)

    # ML V1.5 signals
    if include_v15:
        v15_gen = DirectionV15()
        if v15_gen.loaded:
            test_v15 = v15_gen.compute_features_from_df(test.copy())
            v15_signals = v15_gen.generate_signals(test_v15)
            logger.info("ML V1.5 signals: %d", len(v15_signals))
            signals.extend(v15_signals)

    # Attention signals with custom threshold
    if include_attention:
        attn_gen = DirectionAttention()
        # Override SHORT threshold: conf_short in base means (1 - conf_short) is the prob cutoff
        # We want SHORT when prob < threshold, so conf_short = 1 - threshold
        attn_gen.conf_short = 1.0 - attention_short_threshold
        if attn_gen.loaded:
            test_attn = test.copy()
            attn_gen.compute_features(test_attn)
            n_bars = len(test_attn)
            attn_signals = []
            for i in range(n_bars):
                sig = attn_gen.predict_bar(test_attn, i)
                if sig is not None:
                    attn_signals.append(sig)
            logger.info("ML V2 Attention signals (short_thresh=%.2f): %d",
                        attention_short_threshold, len(attn_signals))
            signals.extend(attn_signals)

    # Build signal map - priority: V1.4 > V1.5 > Attention (if conflict on same bar)
    priority = {
        SignalType.V12_LONG: 3, SignalType.V12_SHORT: 3,
        SignalType.BEAR_LONG: 3, SignalType.BULL_SHORT: 3,
        SignalType.ML_LONG: 2, SignalType.ML_SHORT: 2,
        SignalType.ML_ATTN_LONG: 1, SignalType.ML_ATTN_SHORT: 1,
    }
    signal_map = {}
    for s in signals:
        existing = signal_map.get(s.bar_index)
        if existing is None or priority.get(s.signal_type, 0) > priority.get(existing.signal_type, 0):
            signal_map[s.bar_index] = s

    # Walk bars
    pm = V12PositionManager(config)
    n = len(test)
    i = 0
    while i < n:
        if pm.is_in_position:
            trade = pm.on_bar(highs[i], lows[i], closes[i], times[i], i)
            i += 1
            continue
        # Re-entry check
        if pm.reentry_signal_type is not None:
            regime_ok = _regime_valid(pm.reentry_signal_type, bull, bear, i)
            if pm.can_reenter(i, regime_ok):
                if i + 1 < n:
                    pm.open_position(
                        direction=pm.reentry_direction,
                        signal_type=pm.reentry_signal_type,
                        entry_price=opens[i + 1],
                        entry_time=times[i + 1],
                        signal_time=times[i],
                        is_reentry=True,
                    )
                    i += 2
                    continue
        # New signal
        if i in signal_map:
            sig = signal_map[i]
            pm.reset_reentry()
            if i + 1 < n:
                pm.open_position(
                    direction=sig.direction,
                    signal_type=sig.signal_type,
                    entry_price=opens[i + 1],
                    entry_time=times[i + 1],
                    signal_time=sig.timestamp,
                )
                i += 2
                continue
        i += 1

    return pm.trades


def summarize(trades, config_name: str) -> dict:
    """Compute summary metrics + monthly breakdown."""
    if not trades:
        return {"config": config_name, "n_trades": 0, "status": "NO_TRADES"}

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

    # Monthly
    monthly = tdf.groupby("month")["net_profit_bps"].agg(['sum', 'count']).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly.columns = ["month", "net_bps", "n_trades"]

    # Per signal type
    by_sig = {}
    for st_name in sorted(tdf["signal_type"].unique()):
        st_df = tdf[tdf["signal_type"] == st_name]
        st_w = st_df[st_df["net_profit_bps"] > 0]
        st_l = st_df[st_df["net_profit_bps"] <= 0]
        st_gw = st_w["net_profit_bps"].sum() if len(st_w) > 0 else 0
        st_gl = abs(st_l["net_profit_bps"].sum()) if len(st_l) > 0 else 1
        by_sig[st_name] = {
            "n": len(st_df),
            "win_rate": round(len(st_w) / len(st_df) * 100, 1),
            "net_bps": round(st_df["net_profit_bps"].sum(), 1),
            "pf": round(st_gw / st_gl, 2),
        }

    return {
        "config": config_name,
        "n_trades": len(tdf),
        "win_rate_pct": round(len(winners) / len(tdf) * 100, 1),
        "total_bps": round(tdf["net_profit_bps"].sum(), 1),
        "profit_factor": round(pf, 2),
        "avg_bps_per_trade": round(tdf["net_profit_bps"].mean(), 2),
        "max_drawdown_bps": round(max_dd, 1),
        "avg_bps_per_month": round(tdf["net_profit_bps"].sum() / len(monthly), 1),
        "monthly_breakdown": monthly.to_dict("records"),
        "by_signal": by_sig,
    }


def print_summary(summary: dict):
    print(f"\n{'='*70}")
    print(f"CONFIG: {summary['config']}")
    print('='*70)
    if summary.get("status") == "NO_TRADES":
        print("NO TRADES")
        return
    print(f"  Trades: {summary['n_trades']} | Win Rate: {summary['win_rate_pct']}%")
    print(f"  Net PnL: {summary['total_bps']:+.0f} bps | PF: {summary['profit_factor']}")
    print(f"  Avg/Trade: {summary['avg_bps_per_trade']:+.1f} bps | Avg/Month: {summary['avg_bps_per_month']:+.0f} bps")
    print(f"  Max Drawdown: {summary['max_drawdown_bps']:+.0f} bps")
    print("  By signal type:")
    for st_name, d in summary["by_signal"].items():
        print(f"    {st_name:<18} {d['n']:>4}t | Win: {d['win_rate']:>5.1f}% | Net: {d['net_bps']:>+8.1f} bps | PF: {d['pf']:.2f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()

    configs_to_test = [
        {"name": "attention_only_short040", "v14": False, "v15": False, "attn": True, "short_thresh": 0.40},
        {"name": "attention_only_short035", "v14": False, "v15": False, "attn": True, "short_thresh": 0.35},
        {"name": "combined_short040", "v14": True, "v15": True, "attn": True, "short_thresh": 0.40},
        {"name": "combined_short035", "v14": True, "v15": True, "attn": True, "short_thresh": 0.35},
    ]

    all_summaries = []
    for cfg in configs_to_test:
        logger.info("=== Running: %s ===", cfg["name"])
        trades = run_backtest(
            config,
            include_v14=cfg["v14"],
            include_v15=cfg["v15"],
            include_attention=cfg["attn"],
            attention_short_threshold=cfg["short_thresh"],
        )
        summary = summarize(trades, cfg["name"])
        all_summaries.append(summary)
        print_summary(summary)

        # Save trades CSV
        if trades:
            tdf = pd.DataFrame([asdict(t) for t in trades])
            tdf.to_csv(RESULTS_DIR / f"trades_{cfg['name']}.csv", index=False)

    # Final comparison table
    print(f"\n{'='*90}")
    print("FINAL COMPARISON")
    print('='*90)
    print(f"{'Config':<28}{'Trades':>8}{'Win%':>8}{'Net bps':>12}{'PF':>8}{'Avg/Mo':>10}{'Max DD':>10}")
    print('-'*90)
    for s in all_summaries:
        if s.get("status") == "NO_TRADES":
            print(f"{s['config']:<28}  NO TRADES")
            continue
        print(f"{s['config']:<28}{s['n_trades']:>8}{s['win_rate_pct']:>8}"
              f"{s['total_bps']:>+12.0f}{s['profit_factor']:>8.2f}"
              f"{s['avg_bps_per_month']:>+10.0f}{s['max_drawdown_bps']:>+10.0f}")

    # Save combined JSON
    import json
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    logger.info("Saved: %s", RESULTS_DIR / "summary.json")


if __name__ == "__main__":
    main()
