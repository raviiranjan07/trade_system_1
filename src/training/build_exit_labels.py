"""Stage 2: Build exit-aware labels for ML V3.

For every 15m bar, simulate LONG and SHORT trades using 1m tick data
and the actual V2 exit rules. Records the net P&L for each direction.

Labels:
  long_net_bps:      actual P&L if entered LONG (after fees)
  short_net_bps:     actual P&L if entered SHORT (after fees)
  long_exit_reason:  which exit rule fired (STOP_LOSS, MID_TRAIL, etc.)
  short_exit_reason: same
  long_exit_bar:     which bar the exit fired (0-6)
  short_exit_bar:    same
  direction:         0=LONG, 1=SHORT, 2=SKIP

Uses the SAME V12PositionManager as backtest and live bot — labels are
guaranteed to match real trading outcomes.

Run: PYTHONPATH=src python -m training.build_exit_labels
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config.loader import load_config
from engine.config.constants import FEES_BPS
from engine.position_manager import V12PositionManager
from engine.strategy import Direction, SignalType

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_15M = REPO_ROOT / "data/raw/BTCUSDT_15m_ohlcv.parquet"
DATA_1M = REPO_ROOT / "data/raw/BTCUSDT_1m_ohlcv.parquet"
OUTPUT_PATH = REPO_ROOT / "data/features/direction_prediction/exit_aware_labels.parquet"

LONG = 0
SHORT = 1
SKIP = 2


def simulate_trade(config, direction: Direction, entry_price: float,
                   entry_time, signal_time,
                   ticks_prices: np.ndarray, ticks_times: np.ndarray,
                   bar_highs: np.ndarray, bar_lows: np.ndarray,
                   bar_closes: np.ndarray, bar_times, n_bars: int):
    """Simulate a single trade using real exit rules.

    Returns (net_profit_bps, exit_reason, exit_bar) or None if no data.
    """
    pm = V12PositionManager(config, exit_version="v2")
    pm.open_position(
        direction=direction,
        signal_type=SignalType.ML_LONG if direction == Direction.LONG else SignalType.ML_SHORT,
        entry_price=entry_price,
        entry_time=entry_time,
        signal_time=signal_time,
    )

    # Walk through bars (up to max_bars = 6)
    for bar_i in range(min(n_bars, 6)):
        # First: feed 1m ticks within this bar
        trade = None
        bar_start = bar_times[bar_i]
        bar_end_np = np.datetime64(bar_start) + np.timedelta64(15, "m")

        s_idx = np.searchsorted(ticks_times, np.datetime64(bar_start), side="left")
        e_idx = np.searchsorted(ticks_times, bar_end_np, side="left")

        for t_idx in range(s_idx, e_idx):
            trade = pm.on_tick(float(ticks_prices[t_idx]), ticks_times[t_idx])
            if trade is not None:
                return trade.net_profit_bps, trade.exit_reason, trade.exit_bar

        # Then: bar close (handles TIME_EXIT / NO_ZONE)
        trade = pm.on_bar(
            float(bar_highs[bar_i]), float(bar_lows[bar_i]),
            float(bar_closes[bar_i]), bar_times[bar_i], bar_i,
        )
        if trade is not None:
            return trade.net_profit_bps, trade.exit_reason, trade.exit_bar

    # Should not reach here (TIME_EXIT fires at bar 6), but safety net
    if pm.position is not None:
        close_pnl = 0.0
        if pm.position.direction == Direction.LONG:
            close_pnl = (bar_closes[-1] - entry_price) / entry_price * 10000
        else:
            close_pnl = (entry_price - bar_closes[-1]) / entry_price * 10000
        return close_pnl - FEES_BPS, "TIMEOUT", n_bars

    return 0.0, "NO_TRADE", 0


def main():
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logger.info("Loading config...")
    config = load_config()

    logger.info("Loading 15m data from %s", DATA_15M)
    df_15m = pd.read_parquet(DATA_15M)
    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
    df_15m = df_15m.sort_index()

    logger.info("Loading 1m data from %s", DATA_1M)
    df_1m = pd.read_parquet(DATA_1M)
    df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    df_1m = df_1m.sort_index()
    ticks_times = df_1m.index.values
    ticks_prices = df_1m["close"].values.astype(float)

    N = len(df_15m)
    logger.info("15m bars: %d | 1m ticks: %d", N, len(df_1m))

    # Pre-extract arrays
    opens = df_15m["open"].values.astype(float)
    highs = df_15m["high"].values.astype(float)
    lows = df_15m["low"].values.astype(float)
    closes = df_15m["close"].values.astype(float)
    times_15m = df_15m.index

    # Output arrays
    long_pnl = np.full(N, np.nan, dtype=np.float32)
    short_pnl = np.full(N, np.nan, dtype=np.float32)
    long_reason = np.full(N, "", dtype=object)
    short_reason = np.full(N, "", dtype=object)
    long_bar = np.full(N, -1, dtype=np.int8)
    short_bar = np.full(N, -1, dtype=np.int8)
    direction = np.full(N, SKIP, dtype=np.int8)

    # Max bars for the trade (from exit config)
    max_trade_bars = config.exit.v1_max_bars  # 6

    logger.info("Simulating trades for %d bars (max %d bars per trade)...", N, max_trade_bars)

    skipped_no_data = 0
    for i in range(N - max_trade_bars - 1):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (N - max_trade_bars - 1 - i) / rate
            logger.info("  Bar %d/%d (%.0f bars/sec, ETA %.0f sec)", i, N, rate, eta)

        # Entry at next bar's open (same as backtest)
        entry_price = opens[i + 1]
        if entry_price <= 0:
            skipped_no_data += 1
            continue

        entry_time = times_15m[i + 1]
        signal_time = times_15m[i]

        # Bars for this trade: i+1 through i+1+max_trade_bars
        trade_end = min(i + 1 + max_trade_bars, N)
        n_trade_bars = trade_end - (i + 1)
        if n_trade_bars < 1:
            skipped_no_data += 1
            continue

        bar_h = highs[i + 1:trade_end]
        bar_l = lows[i + 1:trade_end]
        bar_c = closes[i + 1:trade_end]
        bar_t = times_15m[i + 1:trade_end]

        # Simulate LONG
        l_pnl, l_reason, l_bar = simulate_trade(
            config, Direction.LONG, entry_price, entry_time, signal_time,
            ticks_prices, ticks_times, bar_h, bar_l, bar_c, bar_t, n_trade_bars,
        )
        long_pnl[i] = l_pnl
        long_reason[i] = l_reason
        long_bar[i] = l_bar

        # Simulate SHORT
        s_pnl, s_reason, s_bar = simulate_trade(
            config, Direction.SHORT, entry_price, entry_time, signal_time,
            ticks_prices, ticks_times, bar_h, bar_l, bar_c, bar_t, n_trade_bars,
        )
        short_pnl[i] = s_pnl
        short_reason[i] = s_reason
        short_bar[i] = s_bar

        # Direction label
        if l_pnl > 0 and s_pnl > 0:
            direction[i] = LONG if l_pnl >= s_pnl else SHORT
        elif l_pnl > 0:
            direction[i] = LONG
        elif s_pnl > 0:
            direction[i] = SHORT
        else:
            direction[i] = SKIP

    elapsed = time.time() - t0
    logger.info("Simulation complete: %.1f min (%.0f bars/sec)", elapsed / 60, N / elapsed)
    logger.info("Skipped (no data): %d", skipped_no_data)

    # Build output dataframe
    labels = pd.DataFrame({
        "long_net_bps": long_pnl,
        "short_net_bps": short_pnl,
        "long_exit_reason": long_reason,
        "short_exit_reason": short_reason,
        "long_exit_bar": long_bar,
        "short_exit_bar": short_bar,
        "direction": direction,
    }, index=df_15m.index)

    # Drop rows with NaN (edges of dataset)
    valid = labels["long_net_bps"].notna()
    labels = labels[valid]

    # Print distribution
    dir_counts = labels["direction"].value_counts().sort_index()
    dir_map = {0: "LONG", 1: "SHORT", 2: "SKIP"}
    logger.info("\nLabel distribution:")
    for val, count in dir_counts.items():
        logger.info("  %s: %d (%.1f%%)", dir_map.get(val, val), count, count / len(labels) * 100)

    # Per-exit-reason breakdown
    logger.info("\nLONG exit reasons:")
    for reason, count in labels["long_exit_reason"].value_counts().head(10).items():
        avg = labels[labels["long_exit_reason"] == reason]["long_net_bps"].mean()
        logger.info("  %s: %d (avg %+.1f bps)", reason, count, avg)

    logger.info("\nSHORT exit reasons:")
    for reason, count in labels["short_exit_reason"].value_counts().head(10).items():
        avg = labels[labels["short_exit_reason"] == reason]["short_net_bps"].mean()
        logger.info("  %s: %d (avg %+.1f bps)", reason, count, avg)

    # Per-year
    labels["year"] = labels.index.year
    logger.info("\nPer-year distribution:")
    for year in sorted(labels["year"].unique()):
        yr = labels[labels["year"] == year]
        n_long = (yr["direction"] == LONG).sum()
        n_short = (yr["direction"] == SHORT).sum()
        n_skip = (yr["direction"] == SKIP).sum()
        logger.info("  %d: LONG=%d SHORT=%d SKIP=%d (total %d)", year, n_long, n_short, n_skip, len(yr))
    labels = labels.drop(columns=["year"])

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(OUTPUT_PATH)
    logger.info("\nSaved %d labels to %s", len(labels), OUTPUT_PATH)
    logger.info("Total time: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
