"""Exit Strategy Experiments — Test new exit mechanisms using 1-minute data.

Uses 15-minute bars for signal generation (same as live bot).
Uses 1-minute bars for exit monitoring (closer to real-time).

EXP-1 (Baseline with 1-min monitoring):
  - Same trailing stops: 20 bps LONG, 30 bps SHORT
  - Same tightening: 8 bps after bar 5
  - But exit checks every 1 minute instead of every 15 minutes

EXP-2 (Price-based tightening):
  - Same trailing stops: 20 bps LONG, 30 bps SHORT
  - Tighten to 8 bps when MFE crosses 25 bps (0.25%)
  - Exit checks every 1 minute

BASELINE (current V1.3.2 — bar-close only on 15-min):
  - For comparison

Run: PYTHONPATH=src python experiments/exit_strategy/run_experiments.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from engine.config.loader import load_config
from engine.config.constants import FEES_BPS
from engine.strategy import V12Strategy, Direction, SignalType

DATA_15M = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
DATA_1M = Path("data/raw/BTCUSDT_1m_ohlcv.parquet")


@dataclass
class Trade:
    direction: str
    signal_type: str
    entry_price: float
    entry_time: object
    exit_price: float
    exit_time: object
    gross_bps: float
    net_bps: float
    mfe_bps: float
    mae_bps: float
    exit_bar: int  # 15-min bars held
    exit_reason: str


def compute_pnl_bps(direction: str, entry_price: float, price: float) -> float:
    if direction == "LONG":
        return (price - entry_price) / entry_price * 10000
    else:
        return (entry_price - price) / entry_price * 10000


def run_baseline(signals_15m, df_15m):
    """BASELINE: Current V1.3.2 logic — check at 15-min bar close only."""
    trades = []
    i = 0
    n = len(df_15m)
    closes = df_15m["close"].values
    highs = df_15m["high"].values
    lows = df_15m["low"].values
    opens = df_15m["open"].values
    times = df_15m.index
    bull = df_15m["bull_market"].values
    bear = df_15m["bear_market"].values

    signal_map = {s.bar_index: s for s in signals_15m}

    pos = None  # dict with trade state

    while i < n:
        if pos is not None:
            pos["bars_held"] += 1
            h, l, c = highs[i], lows[i], closes[i]

            if pos["direction"] == "LONG":
                bar_mfe = (h - pos["entry_price"]) / pos["entry_price"] * 10000
                bar_mae = (l - pos["entry_price"]) / pos["entry_price"] * 10000
                bar_pnl = (c - pos["entry_price"]) / pos["entry_price"] * 10000
            else:
                bar_mfe = (pos["entry_price"] - l) / pos["entry_price"] * 10000
                bar_mae = (pos["entry_price"] - h) / pos["entry_price"] * 10000
                bar_pnl = (pos["entry_price"] - c) / pos["entry_price"] * 10000

            if bar_mfe > pos["mfe"]: pos["mfe"] = bar_mfe
            if bar_mae < pos["mae"]: pos["mae"] = bar_mae
            if bar_mfe > pos["peak"]: pos["peak"] = bar_mfe

            # Trailing stop (tighten after bar 5)
            active_ts = 8 if pos["bars_held"] > 5 else pos["trailing_stop"]
            drawdown = pos["peak"] - bar_pnl
            if drawdown >= active_ts and pos["peak"] > 0:
                exit_profit = pos["peak"] - active_ts
                reason = "TIGHT_TS" if pos["bars_held"] > 5 else "TRAILING_STOP"
                trades.append(Trade(
                    direction=pos["direction"], signal_type=pos["signal_type"],
                    entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                    exit_price=c, exit_time=times[i],
                    gross_bps=exit_profit, net_bps=exit_profit - FEES_BPS,
                    mfe_bps=pos["mfe"], mae_bps=pos["mae"],
                    exit_bar=pos["bars_held"], exit_reason=reason,
                ))
                pos = None
                i += 1
                continue

            # Time exit
            if pos["bars_held"] >= 10:
                trades.append(Trade(
                    direction=pos["direction"], signal_type=pos["signal_type"],
                    entry_price=pos["entry_price"], entry_time=pos["entry_time"],
                    exit_price=c, exit_time=times[i],
                    gross_bps=bar_pnl, net_bps=bar_pnl - FEES_BPS,
                    mfe_bps=pos["mfe"], mae_bps=pos["mae"],
                    exit_bar=pos["bars_held"], exit_reason="TIME_EXIT",
                ))
                pos = None
                i += 1
                continue

            i += 1
            continue

        # Check for signal
        if i in signal_map:
            sig = signal_map[i]
            # Regime check
            if sig.signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
                if not bull[i]:
                    i += 1
                    continue
            elif sig.signal_type in (SignalType.V12_SHORT, SignalType.BEAR_LONG):
                if not bear[i]:
                    i += 1
                    continue

            if i + 1 < n:
                entry_price = opens[i + 1]
                ts = 20 if sig.direction == Direction.LONG else 30
                pos = {
                    "direction": sig.direction.value,
                    "signal_type": sig.signal_type.value,
                    "entry_price": entry_price,
                    "entry_time": times[i + 1],
                    "trailing_stop": ts,
                    "bars_held": 0,
                    "peak": 0.0,
                    "mfe": 0.0,
                    "mae": 0.0,
                }
                i += 2
                continue

        i += 1

    return trades


def run_experiment(signals_15m, df_15m, df_1m, mode="exp1"):
    """
    EXP-1: 1-min monitoring always, bar-count tightening
    EXP-2: 1-min monitoring always, MFE-based tightening (>= 25 bps)
    EXP-3: 1-min monitoring ONLY when profitable, else 15-min baseline. Bar-count tightening.
    EXP-4: 1-min monitoring ONLY when profitable, else 15-min baseline. MFE-based tightening.
    """
    trades = []
    n = len(df_15m)
    opens = df_15m["open"].values
    times_15m = df_15m.index
    bull = df_15m["bull_market"].values
    bear = df_15m["bear_market"].values

    # Pre-convert 1-min index to numpy for fast slicing
    idx_1m = df_1m.index.values
    highs_1m = df_1m["high"].values
    lows_1m = df_1m["low"].values
    closes_1m = df_1m["close"].values

    signal_map = {s.bar_index: s for s in signals_15m}

    i = 0
    while i < n:
        if i not in signal_map:
            i += 1
            continue

        sig = signal_map[i]

        # Regime check
        if sig.signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
            if not bull[i]:
                i += 1
                continue
        elif sig.signal_type in (SignalType.V12_SHORT, SignalType.BEAR_LONG):
            if not bear[i]:
                i += 1
                continue

        if i + 1 >= n:
            i += 1
            continue

        # Entry at next 15-min bar open
        entry_time = times_15m[i + 1]
        entry_price = opens[i + 1]
        direction = sig.direction.value
        ts_initial = 20 if direction == "LONG" else 30
        max_time = entry_time + pd.Timedelta(minutes=150)  # 10 bars × 15 min

        # Fast numpy slice for 1-min bars
        entry_ts = np.datetime64(entry_time)
        max_ts = np.datetime64(max_time)
        start_idx = np.searchsorted(idx_1m, entry_ts, side='left')
        end_idx = np.searchsorted(idx_1m, max_ts, side='right')

        slice_times = idx_1m[start_idx:end_idx]
        slice_highs = highs_1m[start_idx:end_idx]
        slice_lows = lows_1m[start_idx:end_idx]
        slice_closes = closes_1m[start_idx:end_idx]

        if len(slice_times) == 0:
            i += 2
            continue

        peak = 0.0
        mfe = 0.0
        mae = 0.0
        exit_price = None
        exit_time = None
        exit_reason = None
        gross_bps = 0.0

        # For EXP-3/4: track whether trade has been profitable on a 15-min bar close
        use_hybrid = mode in ("exp3", "exp4")
        trade_was_profitable = False

        for j in range(len(slice_times)):
            t = slice_times[j]
            h, l, c = slice_highs[j], slice_lows[j], slice_closes[j]

            # Calculate MFE/MAE from high/low
            if direction == "LONG":
                bar_best = (h - entry_price) / entry_price * 10000
                bar_worst = (l - entry_price) / entry_price * 10000
                close_pnl = (c - entry_price) / entry_price * 10000
            else:
                bar_best = (entry_price - l) / entry_price * 10000
                bar_worst = (entry_price - h) / entry_price * 10000
                close_pnl = (entry_price - c) / entry_price * 10000

            if bar_best > mfe: mfe = bar_best
            if bar_worst < mae: mae = bar_worst
            if bar_best > peak: peak = bar_best

            minutes_held = (t - np.datetime64(entry_time)) / np.timedelta64(1, 'm')
            bars_15m_held = minutes_held / 15

            # For hybrid modes: check if we're on a 15-min boundary
            is_15m_boundary = (minutes_held > 0 and minutes_held % 15 < 1.5)

            if use_hybrid and not trade_was_profitable:
                # Trade not yet profitable — only check at 15-min bar close (baseline behavior)
                if is_15m_boundary:
                    if close_pnl > 0:
                        trade_was_profitable = True
                    # Apply baseline trailing stop at bar close
                    active_ts = 8 if bars_15m_held > 5 else ts_initial
                    drawdown = peak - close_pnl
                    if drawdown >= active_ts and peak > 0:
                        gross_bps = peak - active_ts
                        exit_time = t
                        exit_reason = "TIGHT_TS" if active_ts == 8 else "TRAILING_STOP"
                        if direction == "LONG":
                            exit_price = entry_price * (1 + gross_bps / 10000)
                        else:
                            exit_price = entry_price * (1 - gross_bps / 10000)
                        break
                    # Time exit
                    if bars_15m_held >= 10:
                        gross_bps = close_pnl
                        exit_price = c
                        exit_time = t
                        exit_reason = "TIME_EXIT"
                        break
                continue  # skip 1-min check while not profitable

            # Trade is profitable (or EXP-1/2 which always uses 1-min) — check every minute
            if mode in ("exp1", "exp3"):
                active_ts = 8 if bars_15m_held > 5 else ts_initial
            elif mode in ("exp2", "exp4"):
                active_ts = 8 if peak >= 25 else ts_initial

            # Check trailing stop against 1-min bar high/low (worst price in the bar)
            if direction == "LONG":
                worst_pnl = (l - entry_price) / entry_price * 10000
            else:
                worst_pnl = (entry_price - h) / entry_price * 10000

            drawdown = peak - worst_pnl
            if drawdown >= active_ts and peak > 0:
                gross_bps = peak - active_ts
                exit_time = t
                exit_reason = "TIGHT_TS" if active_ts == 8 else "TRAILING_STOP"
                if direction == "LONG":
                    exit_price = entry_price * (1 + gross_bps / 10000)
                else:
                    exit_price = entry_price * (1 - gross_bps / 10000)
                break

        # Time exit if no trailing stop triggered
        if exit_reason is None:
            pnl = compute_pnl_bps(direction, entry_price, slice_closes[-1])
            gross_bps = pnl
            exit_price = slice_closes[-1]
            exit_time = slice_times[-1]
            exit_reason = "TIME_EXIT"

        minutes_held = (exit_time - np.datetime64(entry_time)) / np.timedelta64(1, 'm')
        exit_bar_15m = int(minutes_held / 15) + 1

        trades.append(Trade(
            direction=direction, signal_type=sig.signal_type.value,
            entry_price=entry_price, entry_time=entry_time,
            exit_price=exit_price, exit_time=exit_time,
            gross_bps=gross_bps, net_bps=gross_bps - FEES_BPS,
            mfe_bps=mfe, mae_bps=mae,
            exit_bar=exit_bar_15m, exit_reason=exit_reason,
        ))

        # Skip past exit bar (block signals while in trade, same as baseline)
        i = i + 1 + exit_bar_15m

    return trades


def print_results(name, trades):
    if not trades:
        print(f"\n{name}: No trades")
        return

    wins = [t for t in trades if t.net_bps > 0]
    losses = [t for t in trades if t.net_bps <= 0]
    total_bps = sum(t.net_bps for t in trades)
    gross_win = sum(t.net_bps for t in wins) if wins else 0
    gross_loss = abs(sum(t.net_bps for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    cum = 0
    peak_cum = 0
    max_dd = 0
    for t in trades:
        cum += t.net_bps
        if cum > peak_cum: peak_cum = cum
        dd = peak_cum - cum
        if dd > max_dd: max_dd = dd

    avg_mfe = np.mean([t.mfe_bps for t in trades])
    avg_mae = np.mean([t.mae_bps for t in trades])
    avg_bps = total_bps / len(trades)

    # By exit reason
    reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in reasons:
            reasons[r] = {"count": 0, "bps": 0, "wins": 0}
        reasons[r]["count"] += 1
        reasons[r]["bps"] += t.net_bps
        if t.net_bps > 0:
            reasons[r]["wins"] += 1

    # By signal type
    by_signal = {}
    for t in trades:
        s = t.signal_type
        if s not in by_signal:
            by_signal[s] = {"count": 0, "bps": 0, "wins": 0}
        by_signal[s]["count"] += 1
        by_signal[s]["bps"] += t.net_bps
        if t.net_bps > 0:
            by_signal[s]["wins"] += 1

    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")
    print(f" Trades: {len(trades)}  |  Win: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)  |  Loss: {len(losses)}")
    print(f" Net P&L: {total_bps:+.1f} bps  |  Avg: {avg_bps:+.1f} bps/trade  |  PF: {pf:.2f}")
    print(f" Max DD: -{max_dd:.1f} bps  |  Avg MFE: +{avg_mfe:.1f}  |  Avg MAE: {avg_mae:.1f}")
    print()
    print(f" By Exit Reason:")
    for r, d in sorted(reasons.items(), key=lambda x: -x[1]["bps"]):
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        print(f"   {r:20s}  {d['count']:4d} trades  |  {d['bps']:+8.1f} bps  |  WR: {wr:.0f}%")
    print()
    print(f" By Signal Type:")
    for s, d in sorted(by_signal.items(), key=lambda x: -x[1]["bps"]):
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        print(f"   {s:20s}  {d['count']:4d} trades  |  {d['bps']:+8.1f} bps  |  WR: {wr:.0f}%")

    # Yearly breakdown
    by_year = {}
    for t in trades:
        y = str(t.exit_time)[:4]
        if y not in by_year:
            by_year[y] = {"count": 0, "bps": 0, "wins": 0}
        by_year[y]["count"] += 1
        by_year[y]["bps"] += t.net_bps
        if t.net_bps > 0:
            by_year[y]["wins"] += 1
    print()
    print(f" By Year:")
    for y, d in sorted(by_year.items()):
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        print(f"   {y}  {d['count']:4d} trades  |  {d['bps']:+8.1f} bps  |  WR: {wr:.0f}%")


def main():
    print("Loading data...")
    config = load_config()
    df_15m = pd.read_parquet(DATA_15M)
    if df_15m.index.tz is not None:
        df_15m.index = df_15m.index.tz_convert(None)
    else:
        df_15m.index = pd.to_datetime(df_15m.index)
    # Load 1-min data — read in batches, filter to 2024+
    import pyarrow.parquet as pq
    import gc
    gc.collect()
    pf = pq.ParquetFile(DATA_1M)
    chunks = []
    for batch in pf.iter_batches(batch_size=100_000):
        chunk = batch.to_pandas()
        if chunk.index.tz is not None:
            chunk.index = chunk.index.tz_convert(None)
        chunk = chunk[chunk.index >= "2024-01-01"]
        if len(chunk) > 0:
            chunks.append(chunk[["high", "low", "close"]])
    df_1m = pd.concat(chunks) if chunks else pd.DataFrame()
    del chunks
    gc.collect()
    print(f"1-min bars loaded: {len(df_1m):,}")

    # Compute indicators on FULL 15-min data (need 200+ bars warm-up)
    strategy = V12Strategy(config)
    df_15m = strategy.compute_indicators(df_15m)

    # Filter to OOS period (2024-2025)
    test_start = "2024-01-01"
    test_end = "2025-12-31"
    df_15m_test = df_15m[test_start:test_end]
    df_1m_test = df_1m[test_start:test_end]

    print(f"15-min bars: {len(df_15m_test):,} ({df_15m_test.index.min()} to {df_15m_test.index.max()})")
    print(f"1-min bars:  {len(df_1m_test):,} ({df_1m_test.index.min()} to {df_1m_test.index.max()})")

    # Generate signals on 15-min data (V1.4 only, no ML for clean comparison)
    signals = strategy.generate_signals(df_15m_test)
    print(f"V1.4 signals: {len(signals)}")

    # BASELINE: Current V1.3.2 logic
    print("\nRunning BASELINE (15-min bar close)...")
    baseline_trades = run_baseline(signals, df_15m_test)

    # EXP-1: 1-min monitoring, same tightening
    print("Running EXP-1 (1-min monitoring, bar-count tightening)...")
    exp1_trades = run_experiment(signals, df_15m_test, df_1m_test, mode="exp1")

    # EXP-2: 1-min monitoring, price-based tightening
    print("Running EXP-2 (1-min monitoring, MFE-based tightening at 25 bps)...")
    exp2_trades = run_experiment(signals, df_15m_test, df_1m_test, mode="exp2")

    # EXP-3: hybrid — baseline when losing, 1-min bar-count tightening when profitable
    print("Running EXP-3 (hybrid: baseline when losing, 1-min when profitable, bar-count)...")
    exp3_trades = run_experiment(signals, df_15m_test, df_1m_test, mode="exp3")

    # EXP-4: hybrid — baseline when losing, 1-min MFE tightening when profitable
    print("Running EXP-4 (hybrid: baseline when losing, 1-min when profitable, MFE >= 25)...")
    exp4_trades = run_experiment(signals, df_15m_test, df_1m_test, mode="exp4")

    # Print all results
    print_results("BASELINE (V1.3.2 — 15-min bar close)", baseline_trades)
    print_results("EXP-1 (1-min always, bar-count tighten)", exp1_trades)
    print_results("EXP-2 (1-min always, MFE tighten)", exp2_trades)
    print_results("EXP-3 (hybrid: losing=baseline, profit=1min bar-count)", exp3_trades)
    print_results("EXP-4 (hybrid: losing=baseline, profit=1min MFE>=25)", exp4_trades)

    # Comparison table
    print(f"\n{'='*70}")
    print(f" COMPARISON")
    print(f"{'='*70}")
    print(f" {'':35s}  {'Trades':>6s}  {'Win%':>6s}  {'Net bps':>9s}  {'PF':>6s}  {'Max DD':>8s}  {'Avg/T':>7s}")
    for name, t_list in [("BASELINE (15-min close)", baseline_trades),
                          ("EXP-1 (1-min, bar tighten)", exp1_trades),
                          ("EXP-2 (1-min, MFE tighten)", exp2_trades),
                          ("EXP-3 (hybrid, bar tighten)", exp3_trades),
                          ("EXP-4 (hybrid, MFE tighten)", exp4_trades)]:
        if not t_list:
            continue
        wins = len([t for t in t_list if t.net_bps > 0])
        total = sum(t.net_bps for t in t_list)
        gw = sum(t.net_bps for t in t_list if t.net_bps > 0)
        gl = abs(sum(t.net_bps for t in t_list if t.net_bps <= 0)) or 1
        cum = 0; pk = 0; dd = 0
        for t in t_list:
            cum += t.net_bps
            if cum > pk: pk = cum
            if pk - cum > dd: dd = pk - cum
        print(f" {name:35s}  {len(t_list):6d}  {wins/len(t_list)*100:5.1f}%  {total:+9.1f}  {gw/gl:6.2f}  {-dd:+8.1f}  {total/len(t_list):+7.1f}")


if __name__ == "__main__":
    main()
