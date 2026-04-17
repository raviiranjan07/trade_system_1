"""Stage 3: Verify exit-aware labels match real backtest.

Samples 1000 random bars, runs REAL trade simulation for LONG and SHORT,
compares P&L to stored labels. Pipeline FAILS if any mismatch.

Run: PYTHONPATH=src python -m engine.verify_exit_labels
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config.loader import load_config
from .position_manager import V12PositionManager
from .strategy import Direction, SignalType

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "data/features/direction_prediction/exit_aware_labels.parquet"
DATA_15M = REPO_ROOT / "data/raw/BTCUSDT_15m_ohlcv.parquet"
DATA_1M = REPO_ROOT / "data/raw/BTCUSDT_1m_ohlcv.parquet"
OUTPUT_PATH = REPO_ROOT / "data/reports/label_verification.json"

N_SAMPLES = 1000
TOLERANCE_BPS = 0.01


def simulate_one_trade(config, direction, entry_price, entry_time, signal_time,
                       ticks_prices, ticks_times, bar_highs, bar_lows,
                       bar_closes, bar_times, n_bars):
    """Run one trade through the real position manager. Returns net_profit_bps."""
    pm = V12PositionManager(config, exit_version="v2")
    sig_type = SignalType.ML_ATTN_LONG if direction == Direction.LONG else SignalType.ML_ATTN_SHORT
    pm.open_position(
        direction=direction, signal_type=sig_type,
        entry_price=entry_price, entry_time=entry_time, signal_time=signal_time,
    )

    for bar_i in range(min(n_bars, 6)):
        bar_start = bar_times[bar_i]
        bar_end_np = np.datetime64(bar_start) + np.timedelta64(15, "m")
        s_idx = np.searchsorted(ticks_times, np.datetime64(bar_start), side="left")
        e_idx = np.searchsorted(ticks_times, bar_end_np, side="left")

        for t_idx in range(s_idx, e_idx):
            trade = pm.on_tick(float(ticks_prices[t_idx]), ticks_times[t_idx])
            if trade is not None:
                return trade.net_profit_bps

        trade = pm.on_bar(
            float(bar_highs[bar_i]), float(bar_lows[bar_i]),
            float(bar_closes[bar_i]), bar_times[bar_i], bar_i,
        )
        if trade is not None:
            return trade.net_profit_bps

    return 0.0


def main():
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logger.info("Loading labels from %s", LABELS_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    logger.info("Labels: %d rows", len(labels))

    logger.info("Loading 15m data...")
    df_15m = pd.read_parquet(DATA_15M)
    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
    df_15m = df_15m.sort_index()

    logger.info("Loading 1m data...")
    df_1m = pd.read_parquet(DATA_1M)
    df_1m.index = pd.to_datetime(df_1m.index).tz_localize(None)
    df_1m = df_1m.sort_index()
    ticks_times = df_1m.index.values
    ticks_prices = df_1m["close"].values.astype(float)

    config = load_config()
    max_bars = config.exit.v1_max_bars

    opens = df_15m["open"].values.astype(float)
    highs = df_15m["high"].values.astype(float)
    lows = df_15m["low"].values.astype(float)
    closes = df_15m["close"].values.astype(float)
    times_15m = df_15m.index

    # Sample bars that have valid labels and enough future bars
    valid_labels = labels[labels["long_net_bps"].notna()].copy()
    valid_idx = valid_labels.index
    # Map label timestamps to 15m bar positions
    bar_positions = df_15m.index.get_indexer(valid_idx)
    valid_mask = (bar_positions >= 0) & (bar_positions < len(df_15m) - max_bars - 1)
    valid_labels = valid_labels[valid_mask]
    bar_positions = bar_positions[valid_mask]

    n_samples = min(N_SAMPLES, len(valid_labels))
    logger.info("Sampling %d bars for verification...", n_samples)

    np.random.seed(42)
    sample_indices = np.random.choice(len(valid_labels), size=n_samples, replace=False)

    mismatches = 0
    max_error = 0.0
    errors = []

    for count, si in enumerate(sample_indices):
        label_row = valid_labels.iloc[si]
        bar_pos = bar_positions[si]

        entry_price = opens[bar_pos + 1]
        if entry_price <= 0:
            continue

        entry_time = times_15m[bar_pos + 1]
        signal_time = times_15m[bar_pos]
        trade_end = min(bar_pos + 1 + max_bars, len(df_15m))
        n_trade_bars = trade_end - (bar_pos + 1)

        bar_h = highs[bar_pos + 1:trade_end]
        bar_l = lows[bar_pos + 1:trade_end]
        bar_c = closes[bar_pos + 1:trade_end]
        bar_t = times_15m[bar_pos + 1:trade_end]

        # Verify LONG
        actual_long = simulate_one_trade(
            config, Direction.LONG, entry_price, entry_time, signal_time,
            ticks_prices, ticks_times, bar_h, bar_l, bar_c, bar_t, n_trade_bars,
        )
        label_long = label_row["long_net_bps"]
        long_err = abs(actual_long - label_long)

        # Verify SHORT
        actual_short = simulate_one_trade(
            config, Direction.SHORT, entry_price, entry_time, signal_time,
            ticks_prices, ticks_times, bar_h, bar_l, bar_c, bar_t, n_trade_bars,
        )
        label_short = label_row["short_net_bps"]
        short_err = abs(actual_short - label_short)

        err = max(long_err, short_err)
        max_error = max(max_error, err)

        if err > TOLERANCE_BPS:
            mismatches += 1
            errors.append({
                "bar": str(valid_labels.index[si]),
                "long_label": round(float(label_long), 2),
                "long_actual": round(float(actual_long), 2),
                "long_err": round(float(long_err), 2),
                "short_label": round(float(label_short), 2),
                "short_actual": round(float(actual_short), 2),
                "short_err": round(float(short_err), 2),
            })
            if mismatches <= 5:
                logger.warning(
                    "MISMATCH bar %s: long=%.2f vs %.2f (err=%.2f), short=%.2f vs %.2f (err=%.2f)",
                    valid_labels.index[si], label_long, actual_long, long_err,
                    label_short, actual_short, short_err,
                )

        if (count + 1) % 200 == 0:
            logger.info("  Verified %d/%d (mismatches so far: %d)", count + 1, n_samples, mismatches)

    # Label distribution stats
    dir_map = {0: "LONG", 1: "SHORT", 2: "SKIP"}
    dir_counts = labels["direction"].value_counts().sort_index()
    distribution = {dir_map.get(k, str(k)): int(v) for k, v in dir_counts.items()}
    distribution_pct = {k: round(v / len(labels) * 100, 1) for k, v in distribution.items()}

    # Per-exit-reason stats
    long_reasons = labels["long_exit_reason"].value_counts().to_dict()
    short_reasons = labels["short_exit_reason"].value_counts().to_dict()

    # Per-year
    labels_copy = labels.copy()
    labels_copy["year"] = labels_copy.index.year
    per_year = {}
    for year in sorted(labels_copy["year"].unique()):
        yr = labels_copy[labels_copy["year"] == year]
        per_year[str(year)] = {
            "total": len(yr),
            "LONG": int((yr["direction"] == 0).sum()),
            "SHORT": int((yr["direction"] == 1).sum()),
            "SKIP": int((yr["direction"] == 2).sum()),
        }

    # Save report
    report = {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "samples_checked": n_samples,
        "mismatches": mismatches,
        "max_error_bps": round(float(max_error), 4),
        "tolerance_bps": TOLERANCE_BPS,
        "total_labels": len(labels),
        "distribution": distribution,
        "distribution_pct": distribution_pct,
        "long_exit_reasons": {k: int(v) for k, v in long_reasons.items()},
        "short_exit_reasons": {k: int(v) for k, v in short_reasons.items()},
        "per_year": per_year,
        "avg_long_pnl": round(float(labels["long_net_bps"].mean()), 2),
        "avg_short_pnl": round(float(labels["short_net_bps"].mean()), 2),
        "mismatch_details": errors[:10],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t0
    logger.info("\n%s", "=" * 60)
    logger.info("VERIFICATION: %s", report["status"])
    logger.info("  Samples: %d | Mismatches: %d | Max error: %.4f bps",
                n_samples, mismatches, max_error)
    logger.info("  Distribution: %s", distribution_pct)
    logger.info("  Avg LONG pnl: %+.1f bps | Avg SHORT pnl: %+.1f bps",
                report["avg_long_pnl"], report["avg_short_pnl"])
    logger.info("  Time: %.1f sec", elapsed)
    logger.info("  Saved to %s", OUTPUT_PATH)

    if mismatches > 0:
        logger.error("PIPELINE GATE FAILED: %d label mismatches detected!", mismatches)
        sys.exit(1)

    logger.info("PIPELINE GATE PASSED: all %d samples match within %.2f bps", n_samples, TOLERANCE_BPS)


if __name__ == "__main__":
    main()
