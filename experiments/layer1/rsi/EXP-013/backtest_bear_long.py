"""EXP-013: Bear Market Extreme Oversold LONG

Hypothesis: RSI < 10 in bear market (price < SMA200) = so extreme that
a bounce is likely, even against the trend.

Tests multiple RSI thresholds: 5, 8, 10, 15, 20
With and without LONG filters (ATR >= 25pctl, EMA sep >= 0.5%)

Exit: Same as V1.2 (20 bps trailing stop, bar 10 time exit)
Re-entry: Same as V1.2 (1 max, 2 bar cooldown)

OOS period: 2024-2025

Run: python experiments/rsi/EXP-013/backtest_bear_long.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
FEES_BPS = 8.0

# V1.2 parameters
RSI_PERIOD = 14
SMA_PERIOD = 200
EMA_SHORT = 50
EMA_LONG = 200
ATR_PERIOD = 14
ATR_ROLLING = 200

# Exit
TRAILING_STOP_BPS = 20.0
MAX_BARS = 10

# Re-entry
RE_ENABLED = True
RE_MAX = 1
RE_COOLDOWN = 2

# Test configs
RSI_THRESHOLDS = [5, 8, 10, 15, 20]
OOS_START = "2024-01-01"
OOS_END = "2025-12-31"
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # RSI
    delta = out["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    out["rsi"] = 100 - (100 / (1 + rs))

    # SMA200
    out["sma"] = out["close"].rolling(SMA_PERIOD).mean()

    # EMA separation
    out["ema_short"] = out["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    out["ema_long"] = out["close"].ewm(span=EMA_LONG, adjust=False).mean()
    out["ema_separation"] = abs(out["ema_short"] - out["ema_long"]) / out["close"] * 100

    # ATR percentile
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            abs(out["high"] - out["close"].shift(1)),
            abs(out["low"] - out["close"].shift(1)),
        ),
    )
    out["atr"] = tr.rolling(ATR_PERIOD).mean()
    atr_bps = out["atr"] / out["close"] * 10000
    out["atr_percentile"] = atr_bps.rolling(ATR_ROLLING).rank(pct=True) * 100

    # Regime
    out["bull_market"] = out["close"] > out["sma"]
    out["bear_market"] = out["close"] < out["sma"]

    # RSI cross detection (first bar entering zone)
    for thresh in RSI_THRESHOLDS:
        oversold = out["rsi"] < thresh
        out[f"rsi_cross_{thresh}"] = oversold & ~oversold.shift(1, fill_value=False)

    return out


# ---------------------------------------------------------------------------
# Backtest engine (simplified from V1.2)
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, rsi_threshold: int, use_filters: bool) -> list[dict]:
    """Run backtest for bear-market LONG with given RSI threshold."""
    cross_col = f"rsi_cross_{rsi_threshold}"

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    times = df.index
    bear = df["bear_market"].values
    atr_pctl = df["atr_percentile"].values
    ema_sep = df["ema_separation"].values
    crosses = df[cross_col].values

    n = len(df)
    trades = []

    # Position state
    in_pos = False
    entry_price = 0.0
    entry_time = None
    signal_time = None
    bars_held = 0
    highest_profit = 0.0
    mfe = 0.0
    mae = 0.0
    is_reentry = False

    # Re-entry state
    last_exit_reason = None
    last_exit_bar = -999
    last_exit_dir = None
    reentry_count = 0

    def close_position(gross_bps, exit_time, exit_bar, reason):
        nonlocal in_pos, last_exit_reason, last_exit_bar, last_exit_dir, reentry_count
        exit_px = entry_price * (1 + gross_bps / 10000)
        trades.append({
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": "LONG",
            "entry_price": entry_price,
            "exit_price": exit_px,
            "gross_profit_bps": gross_bps,
            "net_profit_bps": gross_bps - FEES_BPS,
            "mfe_bps": mfe,
            "mae_bps": mae,
            "exit_bar": bars_held,
            "exit_reason": reason,
            "is_reentry": is_reentry,
        })
        last_exit_reason = reason
        last_exit_bar = exit_bar
        last_exit_dir = "LONG"
        if is_reentry:
            reentry_count += 1
        elif reason == "TRAILING_STOP":
            reentry_count = 0
        in_pos = False

    i = 0
    while i < n:
        if in_pos:
            bars_held += 1
            bar_mfe = (highs[i] - entry_price) / entry_price * 10000
            bar_mae = (lows[i] - entry_price) / entry_price * 10000
            bar_pnl = (closes[i] - entry_price) / entry_price * 10000

            if bar_mfe > mfe:
                mfe = bar_mfe
            if bar_mae < mae:
                mae = bar_mae
            if bar_mfe > highest_profit:
                highest_profit = bar_mfe

            # Trailing stop
            drawdown = highest_profit - bar_pnl
            if drawdown >= TRAILING_STOP_BPS and highest_profit > 0:
                exit_profit = highest_profit - TRAILING_STOP_BPS
                close_position(exit_profit, times[i], i, "TRAILING_STOP")
                i += 1
                continue

            # Time exit
            if bars_held >= MAX_BARS:
                close_position(bar_pnl, times[i], i, "TIME_EXIT")
                i += 1
                continue

            i += 1
            continue

        # Not in position — check re-entry
        if RE_ENABLED and last_exit_reason == "TRAILING_STOP" and reentry_count < RE_MAX:
            if i == last_exit_bar + RE_COOLDOWN:
                # One-shot re-entry check
                if bear[i]:  # regime still bear (our setup)
                    if i + 1 < n:
                        entry_price = opens[i + 1]
                        entry_time = times[i + 1]
                        signal_time = times[i]
                        bars_held = 0
                        highest_profit = 0.0
                        mfe = 0.0
                        mae = 0.0
                        is_reentry = True
                        in_pos = True
                        i += 2
                        continue
                else:
                    last_exit_reason = None  # abandon re-entry

        # Check for new signal
        if crosses[i] and bear[i]:
            # Apply filters if enabled
            if use_filters:
                ap = atr_pctl[i]
                es = ema_sep[i]
                if np.isnan(ap) or np.isnan(es):
                    i += 1
                    continue
                if ap < 25:
                    i += 1
                    continue
                if es < 0.5:
                    i += 1
                    continue

            # Enter at next bar open
            if i + 1 < n:
                # Reset re-entry state
                reentry_count = 0
                last_exit_reason = None
                last_exit_dir = None
                last_exit_bar = -999

                entry_price = opens[i + 1]
                entry_time = times[i + 1]
                signal_time = times[i]
                bars_held = 0
                highest_profit = 0.0
                mfe = 0.0
                mae = 0.0
                is_reentry = False
                in_pos = True
                i += 2
                continue

        i += 1

    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(trades: list[dict], label: str) -> dict:
    if not trades:
        print(f"  {label}: NO TRADES")
        return {"label": label, "trades": 0}

    tdf = pd.DataFrame(trades)
    winners = tdf[tdf["net_profit_bps"] > 0]
    losers = tdf[tdf["net_profit_bps"] <= 0]
    gw = winners["net_profit_bps"].sum() if len(winners) > 0 else 0
    gl = abs(losers["net_profit_bps"].sum()) if len(losers) > 0 else 1
    pf = gw / gl

    total_bps = tdf["net_profit_bps"].sum()
    avg_bps = tdf["net_profit_bps"].mean()
    equity = tdf["net_profit_bps"].cumsum()
    max_dd = (equity - equity.cummax()).min()

    orig = tdf[~tdf["is_reentry"]]
    re = tdf[tdf["is_reentry"]]

    print(f"  {label}:")
    print(f"    Trades: {len(tdf)} ({len(orig)} orig + {len(re)} RE)")
    print(f"    Win Rate: {len(winners)/len(tdf)*100:.1f}%")
    print(f"    Net Profit: {total_bps:+.0f} bps")
    print(f"    Profit Factor: {pf:.2f}")
    print(f"    Avg/Trade: {avg_bps:+.1f} bps")
    print(f"    Max Drawdown: {max_dd:+.0f} bps")
    print(f"    Avg MFE: {tdf['mfe_bps'].mean():+.1f} bps | Avg MAE: {tdf['mae_bps'].mean():+.1f} bps")

    # Year split
    tdf["year"] = pd.to_datetime(tdf["entry_time"]).dt.year
    for y in sorted(tdf["year"].unique()):
        yt = tdf[tdf["year"] == y]
        yw = yt[yt["net_profit_bps"] > 0]
        yl = yt[yt["net_profit_bps"] <= 0]
        ygw = yw["net_profit_bps"].sum() if len(yw) > 0 else 0
        ygl = abs(yl["net_profit_bps"].sum()) if len(yl) > 0 else 1
        print(f"    {y}: {len(yt)}t | Win: {len(yw)/len(yt)*100:.1f}% | "
              f"Net: {yt['net_profit_bps'].sum():+.0f} bps | PF: {ygw/ygl:.2f}")

    return {
        "label": label,
        "trades": len(tdf),
        "win_rate": len(winners) / len(tdf) * 100,
        "net_bps": total_bps,
        "pf": pf,
        "avg_bps": avg_bps,
        "max_dd": max_dd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EXP-013: Bear Market Extreme Oversold LONG")
    print("Entry: RSI crosses below threshold + price < SMA200 -> LONG")
    print("Exit: 20bps trailing stop, bar 10 time exit, RE(1,2)")
    print("=" * 70)

    # Load data
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = compute_indicators(df)

    # Run on both train and OOS
    for period_name, start, end in [
        ("TRAIN (2020-2023)", TRAIN_START, TRAIN_END),
        ("OOS (2024-2025)", OOS_START, OOS_END),
    ]:
        print(f"\n{'='*70}")
        print(f"  {period_name}")
        print(f"{'='*70}")
        test = df[start:end]
        print(f"  Bars: {len(test)}")

        results = []
        for rsi_thresh in RSI_THRESHOLDS:
            for use_filters in [False, True]:
                filter_label = "+filters" if use_filters else "no_filter"
                label = f"RSI<{rsi_thresh} {filter_label}"
                trades = run_backtest(test, rsi_thresh, use_filters)
                r = report(trades, label)
                results.append(r)
                print()

        # Summary table
        print(f"\n  --- SUMMARY ({period_name}) ---")
        print(f"  {'Config':<25} {'Trades':>6} {'Win%':>6} {'Net bps':>9} {'PF':>6} {'Avg':>7} {'MaxDD':>7}")
        print(f"  {'-'*67}")
        for r in results:
            if r["trades"] > 0:
                print(f"  {r['label']:<25} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['net_bps']:>+8.0f} "
                      f"{r['pf']:>6.2f} {r['avg_bps']:>+6.1f} {r['max_dd']:>+6.0f}")
            else:
                print(f"  {r['label']:<25} {'NO TRADES':>6}")


if __name__ == "__main__":
    main()
