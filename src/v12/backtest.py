"""V1.5 Backtest — Uses V1.4 strategy + ML signal generator + position_manager.

V1.4 signals: V12_LONG, V12_SHORT, BEAR_LONG, BULL_SHORT (rule-based)
ML signals: ML_LONG (prob>0.60), ML_SHORT (prob<0.35) (model-based)

Run: python -m v12.backtest
"""

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config.constants import SYMBOL, TIMEFRAME
from .config.loader import load_config
from .config.schema import AppConfig
from .signals.direction_v15 import DirectionV15 as MLSignalGenerator
from .position_manager import V12PositionManager, TradeRecord
from .strategy import V12Strategy, Direction, SignalType

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")


def run_backtest(
    config: AppConfig,
    data_path: Path = DATA_PATH,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
) -> list[TradeRecord]:
    """Run V1.3 backtest using the same modules as the live bot.

    This is the ground truth. If live bot matches this, we're good.
    """
    # Load data
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

    # Pre-extract arrays for position manager
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

    # Generate ML signals
    ml_model_path = Path("models/direction_v15/direction_model.onnx")
    ml_scaler_path = Path("models/direction_v15/scaler.npz")
    ml_gen = MLSignalGenerator(model_path=ml_model_path, scaler_path=ml_scaler_path)

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

    # Walk through bars, managing positions
    pm = V12PositionManager(config)
    n = len(test)

    i = 0
    while i < n:
        # If in position, feed bars until exit
        if pm.is_in_position:
            trade = pm.on_bar(highs[i], lows[i], closes[i], times[i], i)
            if trade is not None:
                # Position closed — check re-entry on next iteration
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
    if signal_type in (SignalType.ML_LONG, SignalType.ML_SHORT):
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()
    trades = run_backtest(config)
    print_results(trades, config)


if __name__ == "__main__":
    main()
