"""EXP-014: Investigate missed entries after trailing stop exit.

Problem: BEAR_LONG fires on RSI cross below 10. After trade closes via
trailing stop, RSI may still be deep below 10. No new cross fires because
RSI never went back above 10. Second opportunity missed.

Options tested:
  Baseline: V1.3.1 as-is
  A: Multi-level RSI crosses (BEAR_LONG at 10, 8, 6; BULL_SHORT at 90, 92, 94)
  B: Level-based entry (fire BEAR_LONG on ANY bar with RSI<10, no cross required)
  C: Same-bar entry after exit (check signals on the bar where position closes)
  D: Count missed trades (how many exits happen on bars with signals)
  A+C: Multi-level + same-bar fix
  B+C: Level-based + same-bar fix

Run: PYTHONPATH=src python experiments/rsi/EXP-014/test_missed_entries.py
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from engine.config.constants import FEES_BPS
from engine.config.loader import load_config
from engine.config.schema import AppConfig
from engine.position_manager import V12PositionManager, TradeRecord
from engine.strategy import V12Strategy, Direction, Signal, SignalType

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"
START = "2024-01-01"
END = "2025-12-31"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config):
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    strategy = V12Strategy(config)
    df = strategy.compute_indicators(df)
    test = df[START:END]
    return test


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def generate_baseline_signals(config, test):
    """Standard V1.3.1 signals (cross-based)."""
    strategy = V12Strategy(config)
    return strategy.generate_signals(test)


def generate_option_a_signals(config, test):
    """Option A: Multi-level RSI crosses.

    BEAR_LONG: cross below 10, 8, 6  (first match per bar wins)
    BULL_SHORT: cross above 90, 92, 94
    V12_LONG and V12_SHORT: unchanged (standard crosses)
    """
    c = config
    signals = []

    rsi_vals = test["rsi"].values
    atr_pctl = test["atr_percentile"].values
    ema_sep = test["ema_separation"].values
    bull = test["bull_market"].values
    bear = test["bear_market"].values
    oversold_cross = test["rsi_oversold_cross"].values
    overbought_cross = test["rsi_overbought_cross"].values
    prices = test["close"].values
    times = test.index

    # Multi-level crosses for BEAR_LONG (10, 8, 6)
    bear_long_levels = [10, 8, 6]
    bear_long_crosses = {}
    for level in bear_long_levels:
        below = rsi_vals < level
        prev_below = np.roll(below, 1)
        prev_below[0] = False
        bear_long_crosses[level] = below & ~prev_below

    # Multi-level crosses for BULL_SHORT (90, 92, 94)
    bull_short_levels = [90, 92, 94]
    bull_short_crosses = {}
    for level in bull_short_levels:
        above = rsi_vals > level
        prev_above = np.roll(above, 1)
        prev_above[0] = False
        bull_short_crosses[level] = above & ~prev_above

    for i in range(len(test)):
        # V12_LONG (unchanged)
        if oversold_cross[i] and bull[i]:
            ap = atr_pctl[i]
            es = ema_sep[i]
            if np.isnan(ap) or np.isnan(es):
                continue
            if ap < c.long_filters.atr_percentile_min:
                continue
            if es < c.long_filters.ema_separation_min:
                continue
            signals.append(Signal(
                direction=Direction.LONG, signal_type=SignalType.V12_LONG,
                bar_index=i, timestamp=times[i], rsi=rsi_vals[i],
                price=prices[i], atr_percentile=ap, ema_separation=es,
            ))

        # V12_SHORT (unchanged)
        elif overbought_cross[i] and bear[i]:
            signals.append(Signal(
                direction=Direction.SHORT, signal_type=SignalType.V12_SHORT,
                bar_index=i, timestamp=times[i], rsi=rsi_vals[i], price=prices[i],
            ))

        # BEAR_LONG: multi-level crosses (10, 8, 6)
        elif bear[i]:
            for level in bear_long_levels:
                if bear_long_crosses[level][i]:
                    es = ema_sep[i]
                    if np.isnan(es):
                        break  # EMA is NaN for all levels on this bar
                    if es < c.bear_long_filters.ema_separation_min:
                        break  # EMA filter same for all levels
                    signals.append(Signal(
                        direction=Direction.LONG, signal_type=SignalType.BEAR_LONG,
                        bar_index=i, timestamp=times[i], rsi=rsi_vals[i],
                        price=prices[i], ema_separation=es,
                    ))
                    break  # One signal per bar

        # BULL_SHORT: multi-level crosses (90, 92, 94)
        elif bull[i]:
            for level in bull_short_levels:
                if bull_short_crosses[level][i]:
                    ap = atr_pctl[i]
                    es = ema_sep[i]
                    if np.isnan(ap) or np.isnan(es):
                        break
                    if ap < c.bull_short_filters.atr_percentile_min:
                        break
                    if es < c.bull_short_filters.ema_separation_min:
                        break
                    signals.append(Signal(
                        direction=Direction.SHORT, signal_type=SignalType.BULL_SHORT,
                        bar_index=i, timestamp=times[i], rsi=rsi_vals[i],
                        price=prices[i], atr_percentile=ap, ema_separation=es,
                    ))
                    break

    return signals


def generate_option_b_signals(config, test):
    """Option B: Level-based entry (no cross requirement for BEAR_LONG/BULL_SHORT).

    BEAR_LONG fires on ANY bar where RSI < 10 + bear + EMA >= 1.0%
    BULL_SHORT fires on ANY bar where RSI > 90 + bull + ATR >= 60 + EMA >= 1.0%
    V12_LONG and V12_SHORT: unchanged (still use crosses)
    """
    c = config
    signals = []

    rsi_vals = test["rsi"].values
    atr_pctl = test["atr_percentile"].values
    ema_sep = test["ema_separation"].values
    bull = test["bull_market"].values
    bear = test["bear_market"].values
    oversold_cross = test["rsi_oversold_cross"].values
    overbought_cross = test["rsi_overbought_cross"].values
    prices = test["close"].values
    times = test.index

    for i in range(len(test)):
        # V12_LONG (unchanged — uses cross)
        if oversold_cross[i] and bull[i]:
            ap = atr_pctl[i]
            es = ema_sep[i]
            if np.isnan(ap) or np.isnan(es):
                continue
            if ap < c.long_filters.atr_percentile_min:
                continue
            if es < c.long_filters.ema_separation_min:
                continue
            signals.append(Signal(
                direction=Direction.LONG, signal_type=SignalType.V12_LONG,
                bar_index=i, timestamp=times[i], rsi=rsi_vals[i],
                price=prices[i], atr_percentile=ap, ema_separation=es,
            ))

        # V12_SHORT (unchanged — uses cross)
        elif overbought_cross[i] and bear[i]:
            signals.append(Signal(
                direction=Direction.SHORT, signal_type=SignalType.V12_SHORT,
                bar_index=i, timestamp=times[i], rsi=rsi_vals[i], price=prices[i],
            ))

        # BEAR_LONG: LEVEL-BASED (fire on ANY bar where RSI < 10)
        elif rsi_vals[i] < c.bear_long_filters.rsi_threshold and bear[i]:
            es = ema_sep[i]
            if np.isnan(es):
                continue
            if es < c.bear_long_filters.ema_separation_min:
                continue
            signals.append(Signal(
                direction=Direction.LONG, signal_type=SignalType.BEAR_LONG,
                bar_index=i, timestamp=times[i], rsi=rsi_vals[i],
                price=prices[i], ema_separation=es,
            ))

        # BULL_SHORT: LEVEL-BASED (fire on ANY bar where RSI > 90)
        elif rsi_vals[i] > c.bull_short_filters.rsi_threshold and bull[i]:
            ap = atr_pctl[i]
            es = ema_sep[i]
            if np.isnan(ap) or np.isnan(es):
                continue
            if ap < c.bull_short_filters.atr_percentile_min:
                continue
            if es < c.bull_short_filters.ema_separation_min:
                continue
            signals.append(Signal(
                direction=Direction.SHORT, signal_type=SignalType.BULL_SHORT,
                bar_index=i, timestamp=times[i], rsi=rsi_vals[i],
                price=prices[i], atr_percentile=ap, ema_separation=es,
            ))

    return signals


# ---------------------------------------------------------------------------
# Backtest runners
# ---------------------------------------------------------------------------

def _regime_valid(signal_type, bull, bear, idx):
    if signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
        return bool(bull[idx])
    return bool(bear[idx])


def run_backtest_standard(config, test, signals):
    """Standard backtest loop (used for Baseline, Option A, Option B)."""
    highs = test["high"].values
    lows = test["low"].values
    closes = test["close"].values
    opens = test["open"].values
    times = test.index
    bull = test["bull_market"].values
    bear = test["bear_market"].values

    signal_map = {s.bar_index: s for s in signals}
    pm = V12PositionManager(config)
    n = len(test)

    i = 0
    while i < n:
        if pm.is_in_position:
            trade = pm.on_bar(highs[i], lows[i], closes[i], times[i], i)
            if trade is not None:
                i += 1
                continue
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

        # Signal check
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


def run_backtest_option_c(config, test, signals):
    """Option C: Same-bar entry after exit.

    When position closes on bar X, fall through to signal check on bar X
    instead of skipping to bar X+1.
    """
    highs = test["high"].values
    lows = test["low"].values
    closes = test["close"].values
    opens = test["open"].values
    times = test.index
    bull = test["bull_market"].values
    bear = test["bear_market"].values

    signal_map = {s.bar_index: s for s in signals}
    pm = V12PositionManager(config)
    n = len(test)

    i = 0
    while i < n:
        if pm.is_in_position:
            trade = pm.on_bar(highs[i], lows[i], closes[i], times[i], i)
            if trade is not None:
                # FALL THROUGH — don't skip, check for signals on this bar
                pass
            else:
                i += 1
                continue

        # Only reach here if: (a) not in position, or (b) just exited on this bar
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

        # Signal check
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


def count_missed_signals(config, test, signals):
    """Option D: Count bars where position exits AND a signal exists."""
    highs = test["high"].values
    lows = test["low"].values
    closes = test["close"].values
    opens = test["open"].values
    times = test.index
    bull = test["bull_market"].values
    bear = test["bear_market"].values

    signal_map = {s.bar_index: s for s in signals}
    pm = V12PositionManager(config)
    n = len(test)

    missed = []

    i = 0
    while i < n:
        if pm.is_in_position:
            trade = pm.on_bar(highs[i], lows[i], closes[i], times[i], i)
            if trade is not None:
                if i in signal_map:
                    sig = signal_map[i]
                    missed.append({
                        "exit_bar": i,
                        "exit_time": str(times[i]),
                        "exit_reason": trade.exit_reason,
                        "exit_pnl": trade.net_profit_bps,
                        "missed_signal": sig.signal_type.value,
                        "missed_direction": sig.direction.value,
                        "missed_rsi": sig.rsi,
                    })
                i += 1
                continue
            i += 1
            continue

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

    return pm.trades, missed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def calc_stats(trades):
    """Calculate stats dict from trade list."""
    if not trades:
        return None
    tdf = pd.DataFrame([asdict(t) for t in trades])
    winners = tdf[tdf["net_profit_bps"] > 0]
    losers = tdf[tdf["net_profit_bps"] <= 0]
    gw = winners["net_profit_bps"].sum() if len(winners) > 0 else 0
    gl = abs(losers["net_profit_bps"].sum()) if len(losers) > 0 else 1
    equity = tdf["net_profit_bps"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    lt = tdf[tdf["direction"] == "LONG"]
    st = tdf[tdf["direction"] == "SHORT"]
    return {
        "trades": len(tdf),
        "win_pct": len(winners) / len(tdf) * 100,
        "net_bps": tdf["net_profit_bps"].sum(),
        "pf": gw / gl,
        "dd": max_dd,
        "avg": tdf["net_profit_bps"].mean(),
        "long_bps": lt["net_profit_bps"].sum(),
        "short_bps": st["net_profit_bps"].sum(),
        "tdf": tdf,
    }


def print_comparison(results):
    print("\n" + "=" * 100)
    print("EXP-014 RESULTS COMPARISON")
    print("=" * 100)
    print(f"{'Config':<14} {'Trades':>6} {'Win%':>6} {'Net bps':>9} {'PF':>7} {'DD':>7} {'Avg/T':>7} {'LONG':>8} {'SHORT':>8}")
    print("-" * 100)

    for name, trades in results.items():
        s = calc_stats(trades)
        if not s:
            print(f"{name:<14}   NO TRADES")
            continue
        print(f"{name:<14} {s['trades']:>6} {s['win_pct']:>5.1f}% {s['net_bps']:>+8.0f} {s['pf']:>7.2f} {s['dd']:>+7.0f} {s['avg']:>+6.1f} {s['long_bps']:>+7.0f} {s['short_bps']:>+7.0f}")

    print("=" * 100)

    # Signal type breakdown
    for name, trades in results.items():
        s = calc_stats(trades)
        if not s:
            continue
        tdf = s["tdf"]
        print(f"\n{name} — By Signal Type:")
        for st_name in sorted(tdf["signal_type"].unique()):
            st_df = tdf[tdf["signal_type"] == st_name]
            st_w = st_df[st_df["net_profit_bps"] > 0]
            st_l = st_df[st_df["net_profit_bps"] <= 0]
            st_gw = st_w["net_profit_bps"].sum() if len(st_w) > 0 else 0
            st_gl = abs(st_l["net_profit_bps"].sum()) if len(st_l) > 0 else 1
            print(f"  {st_name:<12} {len(st_df):>4}t | Win: {len(st_w)/len(st_df)*100:>5.1f}% | "
                  f"Net: {st_df['net_profit_bps'].sum():>+7.0f} bps | PF: {st_gw/st_gl:.2f}")

    # Year breakdown
    print(f"\n{'=' * 80}")
    print("YEAR BREAKDOWN")
    print(f"{'=' * 80}")
    for name, trades in results.items():
        s = calc_stats(trades)
        if not s:
            continue
        tdf = s["tdf"]
        tdf["year"] = pd.to_datetime(tdf["entry_time"]).dt.year
        print(f"\n{name}:")
        for y in sorted(tdf["year"].unique()):
            yt = tdf[tdf["year"] == y]
            yw = yt[yt["net_profit_bps"] > 0]
            yl = yt[yt["net_profit_bps"] <= 0]
            ygw = yw["net_profit_bps"].sum() if len(yw) > 0 else 0
            ygl = abs(yl["net_profit_bps"].sum()) if len(yl) > 0 else 1
            print(f"  {y}: {len(yt)}t | Win: {len(yw)/len(yt)*100:.1f}% | "
                  f"Net: {yt['net_profit_bps'].sum():+.0f} bps | PF: {ygw/ygl:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("EXP-014: Missed Entry Investigation")
    print("=" * 60)

    config = load_config()
    print(f"Config hash: {config.config_hash()}")

    print("Loading data and computing indicators...")
    test = load_data(config)
    print(f"Test period: {test.index[0]} to {test.index[-1]} ({len(test)} bars)")

    # --- Generate signals ---
    print("\nGenerating signals...")
    baseline_sigs = generate_baseline_signals(config, test)
    option_a_sigs = generate_option_a_signals(config, test)
    option_b_sigs = generate_option_b_signals(config, test)

    # Signal counts by type
    def sig_summary(sigs, label):
        by_type = {}
        for s in sigs:
            by_type.setdefault(s.signal_type.value, 0)
            by_type[s.signal_type.value] += 1
        parts = [f"{k}={v}" for k, v in sorted(by_type.items())]
        print(f"  {label}: {len(sigs)} total — {', '.join(parts)}")

    sig_summary(baseline_sigs, "Baseline")
    sig_summary(option_a_sigs, "Option A (multi-level)")
    sig_summary(option_b_sigs, "Option B (level-based)")

    # --- Run backtests ---
    print("\nRunning backtests...")
    results = {}

    print("  Baseline...")
    results["Baseline"] = run_backtest_standard(config, test, baseline_sigs)

    print("  Option A (multi-level crosses)...")
    results["A:MultiLvl"] = run_backtest_standard(config, test, option_a_sigs)

    print("  Option B (level-based)...")
    results["B:LevelBsd"] = run_backtest_standard(config, test, option_b_sigs)

    print("  Option C (same-bar fix, baseline signals)...")
    results["C:SameBar"] = run_backtest_option_c(config, test, baseline_sigs)

    print("  A+C (multi-level + same-bar)...")
    results["A+C"] = run_backtest_option_c(config, test, option_a_sigs)

    print("  B+C (level-based + same-bar)...")
    results["B+C"] = run_backtest_option_c(config, test, option_b_sigs)

    # --- Print comparison ---
    print_comparison(results)

    # --- Option D: missed signals ---
    print(f"\n{'=' * 80}")
    print("OPTION D: MISSED TRADES ANALYSIS")
    print(f"{'=' * 80}")

    for label, sigs in [
        ("Baseline signals", baseline_sigs),
        ("Option A signals (multi-level)", option_a_sigs),
        ("Option B signals (level-based)", option_b_sigs),
    ]:
        _, missed = count_missed_signals(config, test, sigs)
        print(f"\n{label}: {len(missed)} missed signals on exit bars")
        if missed:
            mdf = pd.DataFrame(missed)
            print(f"  By exit reason: {dict(mdf['exit_reason'].value_counts())}")
            print(f"  By missed signal: {dict(mdf['missed_signal'].value_counts())}")
            print(f"\n  Details:")
            for _, row in mdf.iterrows():
                print(f"    Bar {row['exit_bar']:>5} | {row['exit_time']} | "
                      f"Exit: {row['exit_reason']:<13} ({row['exit_pnl']:>+6.1f} bps) | "
                      f"Missed: {row['missed_signal']:<12} {row['missed_direction']:<5} RSI={row['missed_rsi']:.1f}")

    # --- Delta analysis: new trades in each option vs baseline ---
    print(f"\n{'=' * 80}")
    print("DELTA ANALYSIS: NEW TRADES vs BASELINE")
    print(f"{'=' * 80}")

    baseline_trades = results["Baseline"]
    baseline_times = {(t.entry_time, t.direction) for t in baseline_trades}

    for name in ["A:MultiLvl", "B:LevelBsd", "C:SameBar", "A+C", "B+C"]:
        option_trades = results[name]
        new_trades = [t for t in option_trades if (t.entry_time, t.direction) not in baseline_times]
        if new_trades:
            print(f"\n{name}: {len(new_trades)} new trades (not in baseline)")
            for t in new_trades:
                print(f"    {t.entry_time} | {t.signal_type:<12} {t.direction:<5} | "
                      f"Net: {t.net_profit_bps:>+6.1f} bps | Exit: {t.exit_reason} bar {t.exit_bar}")
            total_new = sum(t.net_profit_bps for t in new_trades)
            wins_new = sum(1 for t in new_trades if t.net_profit_bps > 0)
            print(f"    >> Total new: {total_new:+.1f} bps, {wins_new}/{len(new_trades)} winners")
        else:
            print(f"\n{name}: 0 new trades (identical to baseline)")


if __name__ == "__main__":
    main()
