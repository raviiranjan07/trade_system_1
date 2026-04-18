"""V1.5 Backtest — Uses V1.4 strategy + ML signal generator + position_manager.

V1.4 signals: V12_LONG, V12_SHORT, BEAR_LONG, BULL_SHORT (rule-based)
ML signals: ML_LONG (prob>0.60), ML_SHORT (prob<0.35) (model-based)

Run: python -m engine.backtest
"""

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config.constants import SYMBOL, TIMEFRAME
from .config.loader import load_config
from .config.schema import AppConfig
from .signals.ml_v1 import MLV1
from .signals.direction_attention import DirectionAttention
from .signals.ml_v3 import MLV3
from .signals.base import BaseSignalGenerator
from .position_manager import V12PositionManager, TradeRecord
from .strategy import V12Strategy, Direction, SignalType


# Registry of available ML signal generators, indexed by model name.
# Backtest resolves which class to instantiate based on the `model` arg.
ML_GENERATORS: dict[str, tuple[type[BaseSignalGenerator], Path]] = {
    "ml_v1": (MLV1, Path("models/ML_V1")),
    "ml_v2_attention": (DirectionAttention, Path("models/ML_V2_ATTENTION_staging")),
    "ml_v3": (MLV3, Path("models/ML_V3_staging")),
}

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
DATA_PATH_1M = Path("data/raw/BTCUSDT_1m_ohlcv.parquet")


def run_backtest(
    config: AppConfig,
    data_path: Path = DATA_PATH,
    data_path_1m: Path = DATA_PATH_1M,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    ml_model_dir: Path = Path("models/ML_V1"),
    ml_generator_class: type[BaseSignalGenerator] = MLV1,
    ml_onnx_filename: str = "direction_model.onnx",
    ml_scaler_filename: str = "scaler.npz",
    v14_only: bool = False,
    exit_version: str = "v1",
) -> list[TradeRecord]:
    """Run backtest using the same modules as the live bot.

    When V3 exits are active, 1-minute ticks within each 15-minute bar
    are fed through pm.on_tick() so tick-level exits (PT_TARGET, PT_LOCK,
    MID_TRAIL, LOCKED_PROFIT, STOP_LOSS) can fire — matching how the live
    bot receives WebSocket ticks.
    """
    # Load 15m bars (strategy signals + bar-close time exits)
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Compute indicators on FULL data (need 200+ bars warm-up)
    strategy = V12Strategy(config)
    df = strategy.compute_indicators(df)

    # Slice to test period
    test = df[start:end]
    logger.info(
        "Backtest %s to %s | %d bars | config_hash=%s",
        start, end, len(test), config.config_hash(),
    )

    # Load 1m ticks for V3 intrabar exits (sliced + sorted once)
    import numpy as np
    df_1m = pd.read_parquet(data_path_1m)
    df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    df_1m = df_1m.sort_index()
    df_1m = df_1m[start:end]
    idx_1m = df_1m.index.values
    prices_1m = df_1m["close"].values
    logger.info("1m ticks loaded: %d", len(df_1m))

    # Pre-extract 15m arrays for position manager
    highs = test["high"].values
    lows = test["low"].values
    closes = test["close"].values
    opens = test["open"].values
    times = test.index
    bull = test["bull_market"].values
    bear = test["bear_market"].values

    # Generate V1.4 signals
    signals = strategy.generate_signals(test)
    logger.info("V1.4 signals: %d", len(signals))

    # Generate ML signals — unless V1.4-only mode is requested.
    if v14_only:
        logger.info("V1.4-only mode: skipping ML signal generation")
    else:
        ml_model_path = ml_model_dir / ml_onnx_filename
        ml_scaler_path = ml_model_dir / ml_scaler_filename
        ml_gen = ml_generator_class(model_path=ml_model_path, scaler_path=ml_scaler_path)

        if ml_gen.loaded:
            test_with_ml = ml_gen.compute_features_from_df(test.copy())
            ml_signals = ml_gen.generate_signals(test_with_ml)
            logger.info("ML signals: %d", len(ml_signals))
            signals.extend(ml_signals)
        else:
            logger.warning("ML model not found — running V1.4 only")

    # Build signal lookup: bar_index -> Signal (V1.4 takes priority over ML)
    signal_map = {}
    for s in signals:
        if s.bar_index not in signal_map:
            signal_map[s.bar_index] = s
        else:
            # V1.4 signals take priority over ML signals
            existing = signal_map[s.bar_index]
            if existing.signal_type.value.startswith("ML") and not s.signal_type.value.startswith("ML"):
                signal_map[s.bar_index] = s  # replace ML with V1.4

    # Walk through bars, managing positions.
    pm = V12PositionManager(config, exit_version=exit_version)
    n = len(test)

    i = 0
    while i < n:
        # If in position, feed ticks first (V3 intrabar exits), then bar close
        if pm.is_in_position:
            # Walk through 1m ticks inside this 15m bar — tick-level exits
            # (PT_TARGET/PT_LOCK/MID_TRAIL/LOCKED_PROFIT/STOP_LOSS) may fire.
            bar_start = np.datetime64(times[i])
            bar_end = bar_start + np.timedelta64(15, "m")
            s_idx = np.searchsorted(idx_1m, bar_start, side="left")
            e_idx = np.searchsorted(idx_1m, bar_end, side="left")

            trade = None
            for t_idx in range(s_idx, e_idx):
                trade = pm.on_tick(float(prices_1m[t_idx]), idx_1m[t_idx])
                if trade is not None:
                    break

            # If no tick exit fired, check bar-level (handles bars_held increment
            # and NO_ZONE/TIME_EXIT at max_bars).
            if trade is None:
                trade = pm.on_bar(highs[i], lows[i], closes[i], times[i], i)

            if trade is not None:
                i += 1
                continue
            i += 1
            continue

        # Not in position — check re-entry first (signal-type-based regime)
        if pm.reentry_signal_type is not None:
            regime_ok = _regime_valid(pm.reentry_signal_type, bull, bear, i)
            if pm.can_reenter(i, regime_ok):
                # Re-enter at next bar's open
                if i + 1 < n:
                    entry_price = opens[i + 1]
                    pm.open_position(
                        direction=pm.reentry_direction,
                        signal_type=pm.reentry_signal_type,
                        entry_price=entry_price,
                        entry_time=times[i + 1],
                        signal_time=times[i],
                        is_reentry=True,
                    )
                    i += 2  # skip entry bar, start feeding from bar after entry
                    continue

        # Check for new signal on this bar
        if i in signal_map:
            sig = signal_map[i]
            pm.reset_reentry()

            # Enter at next bar's open
            if i + 1 < n:
                entry_price = opens[i + 1]
                pm.open_position(
                    direction=sig.direction,
                    signal_type=sig.signal_type,
                    entry_price=entry_price,
                    entry_time=times[i + 1],
                    signal_time=sig.timestamp,
                )
                i += 2  # skip entry bar
                continue

        i += 1

    return pm.trades


def _regime_valid(signal_type: SignalType, bull, bear, idx: int) -> bool:
    """Check if market regime is still valid for re-entry signal type.

    V1.3: regime depends on SIGNAL TYPE, not just direction:
      V12_LONG / BULL_SHORT -> needs bull (price > SMA200)
      V12_SHORT / BEAR_LONG -> needs bear (price < SMA200)
      ML_LONG / ML_SHORT -> no regime requirement (always valid)
    """
    if signal_type in (SignalType.ML_LONG, SignalType.ML_SHORT,
                       SignalType.ML_ATTN_LONG, SignalType.ML_ATTN_SHORT,
                       SignalType.ML_V3_LONG, SignalType.ML_V3_SHORT):
        return True  # ML signals don't require regime validation
    if signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
        return bool(bull[idx])
    else:  # V12_SHORT, BEAR_LONG
        return bool(bear[idx])


def print_results(trades: list[TradeRecord], config: AppConfig) -> None:
    """Print backtest results summary."""
    if not trades:
        print("NO TRADES")
        return

    tdf = pd.DataFrame([asdict(t) for t in trades])

    winners = tdf[tdf["net_profit_bps"] > 0]
    losers = tdf[tdf["net_profit_bps"] <= 0]
    gw = winners["net_profit_bps"].sum() if len(winners) > 0 else 0
    gl = abs(losers["net_profit_bps"].sum()) if len(losers) > 0 else 1
    pf = gw / gl

    lt = tdf[tdf["direction"] == "LONG"]
    st = tdf[tdf["direction"] == "SHORT"]
    orig = tdf[~tdf["is_reentry"]]
    re_trades = tdf[tdf["is_reentry"]]

    equity = tdf["net_profit_bps"].cumsum()
    max_dd = (equity - equity.cummax()).min()

    print("=" * 70)
    print(f"V1.3 BACKTEST RESULTS | config_hash={config.config_hash()}")
    print("=" * 70)
    print(f"Trades: {len(tdf)} ({len(orig)} orig + {len(re_trades)} RE)")
    print(f"Win Rate: {len(winners)/len(tdf)*100:.1f}%")
    print(f"Net Profit: {tdf['net_profit_bps'].sum():+.0f} bps")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Avg/Trade: {tdf['net_profit_bps'].mean():+.1f} bps")
    print(f"Max Drawdown: {max_dd:+.0f} bps")
    print(f"LONG:  {len(lt)}t, {lt['net_profit_bps'].sum():+.0f} bps")
    print(f"SHORT: {len(st)}t, {st['net_profit_bps'].sum():+.0f} bps")

    # Signal type breakdown
    print()
    print("By Signal Type:")
    for st_name in sorted(tdf["signal_type"].unique()):
        st_df = tdf[tdf["signal_type"] == st_name]
        st_w = st_df[st_df["net_profit_bps"] > 0]
        st_l = st_df[st_df["net_profit_bps"] <= 0]
        st_gw = st_w["net_profit_bps"].sum() if len(st_w) > 0 else 0
        st_gl = abs(st_l["net_profit_bps"].sum()) if len(st_l) > 0 else 1
        print(f"  {st_name:<12} {len(st_df):>4}t | Win: {len(st_w)/len(st_df)*100:>5.1f}% | "
              f"Net: {st_df['net_profit_bps'].sum():>+7.0f} bps | PF: {st_gw/st_gl:.2f}")

    if len(re_trades) > 0:
        re_w = re_trades[re_trades["net_profit_bps"] > 0]
        print(f"\nRE:    {len(re_trades)}t, win {len(re_w)/len(re_trades)*100:.1f}%, "
              f"{re_trades['net_profit_bps'].sum():+.0f} bps")

    # Year split
    tdf["year"] = pd.to_datetime(tdf["entry_time"]).dt.year
    print()
    for y in sorted(tdf["year"].unique()):
        yt = tdf[tdf["year"] == y]
        yw = yt[yt["net_profit_bps"] > 0]
        yl = yt[yt["net_profit_bps"] <= 0]
        ygw = yw["net_profit_bps"].sum() if len(yw) > 0 else 0
        ygl = abs(yl["net_profit_bps"].sum()) if len(yl) > 0 else 1
        print(f"  {y}: {len(yt)}t | Win: {len(yw)/len(yt)*100:.1f}% | "
              f"Net: {yt['net_profit_bps'].sum():+.0f} bps | PF: {ygw/ygl:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-only", action="store_true",
                        help="Skip ML signals — backtest V1.4 strategy alone")
    parser.add_argument("--exit-version", choices=["v1", "v2"], default="v1",
                        help="v1 = full V1 (default). v2 = V1 minus LOCKED_PROFIT.")
    parser.add_argument("--model", choices=list(ML_GENERATORS.keys()), default="ml_v1",
                        help="Which ML model to use for signals (default: ml_v1)")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()

    gen_class, model_dir = ML_GENERATORS[args.model]
    onnx_name = "v3_model.onnx" if args.model == "ml_v3" else (
        "attention_model.onnx" if args.model == "ml_v2_attention" else "direction_model.onnx")

    trades = run_backtest(
        config,
        start=args.start,
        end=args.end,
        v14_only=args.v14_only,
        exit_version=args.exit_version,
        ml_model_dir=model_dir,
        ml_generator_class=gen_class,
        ml_onnx_filename=onnx_name,
    )
    print_results(trades, config)

    # Save JSON + parquet for DVC pipeline stages
    if trades:
        import json
        from pathlib import Path
        tdf = pd.DataFrame([asdict(t) for t in trades])
        report_dir = Path("data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Compute metrics
        wins = tdf[tdf["net_profit_bps"] > 0]
        losses = tdf[tdf["net_profit_bps"] <= 0]
        gw = wins["net_profit_bps"].sum() if len(wins) else 0
        gl = abs(losses["net_profit_bps"].sum()) if len(losses) else 1
        stops = tdf[tdf["exit_reason"] == "STOP_LOSS"]
        equity = tdf["net_profit_bps"].cumsum()
        max_dd = float((equity - equity.cummax()).min())

        by_sig = {}
        for st, g in tdf.groupby("signal_type"):
            sw = g[g["net_profit_bps"] > 0]
            sl = g[g["net_profit_bps"] <= 0]
            sgw = sw["net_profit_bps"].sum() if len(sw) else 0
            sgl = abs(sl["net_profit_bps"].sum()) if len(sl) else 1
            by_sig[st] = {
                "n": len(g), "win_pct": round(len(sw)/len(g)*100, 1),
                "total_bps": round(float(g["net_profit_bps"].sum()), 1),
                "pf": round(sgw/sgl, 2),
            }

        tdf["year"] = pd.to_datetime(tdf["entry_time"]).dt.year
        per_year = {}
        for y, yg in tdf.groupby("year"):
            yw = yg[yg["net_profit_bps"] > 0]
            yl = yg[yg["net_profit_bps"] <= 0]
            ygw = yw["net_profit_bps"].sum() if len(yw) else 0
            ygl = abs(yl["net_profit_bps"].sum()) if len(yl) else 1
            per_year[str(y)] = {
                "n": len(yg), "total_bps": round(float(yg["net_profit_bps"].sum()), 1),
                "pf": round(ygw/ygl, 2),
            }

        report = {
            "model": args.model,
            "exit_version": args.exit_version,
            "window": {"start": args.start, "end": args.end},
            "metrics": {
                "n_trades": len(tdf),
                "total_bps": round(float(tdf["net_profit_bps"].sum()), 1),
                "pf": round(gw/gl, 2),
                "win_pct": round(len(wins)/len(tdf)*100, 1),
                "stop_rate": round(len(stops)/len(tdf)*100, 1),
                "max_dd_bps": round(max_dd, 1),
                "avg_trade_bps": round(float(tdf["net_profit_bps"].mean()), 1),
                "by_signal_type": by_sig,
                "per_year": per_year,
            },
        }

        suffix = f"_{args.model}" if args.model != "ml_v1" else ""
        json_path = report_dir / f"backtest_staging{suffix}.json"
        parquet_path = report_dir / f"backtest_staging_trades{suffix}.parquet"

        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        tdf.drop(columns=["year"], errors="ignore").to_parquet(parquet_path)
        logger.info("Saved: %s + %s", json_path, parquet_path)


if __name__ == "__main__":
    main()
