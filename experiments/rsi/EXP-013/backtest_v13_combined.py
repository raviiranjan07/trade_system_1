"""EXP-013d: V1.3 Combined Backtest — All 4 Signals

Signals:
  1. LONG (V1.2):      RSI crosses <20 + bull (price>SMA200) + ATR>=25 + EMA>=0.5%
  2. SHORT (V1.2):     RSI crosses >80 + bear (price<SMA200) [no filters]
  3. Bear LONG (new):  RSI crosses <10 + bear (price<SMA200) + EMA>=1.0%
  4. Bull SHORT (new): RSI crosses >90 + bull (price>SMA200) + ATR>=60 + EMA>=1.0%

Exit:
  LONG:  20bps trailing stop, bar 10 time exit
  SHORT: 30bps trailing stop, bar 10 time exit

Re-entry: 1 max, 2 bar cooldown (all signals)

Also runs V1.2 baseline for direct comparison.

Run: python experiments/rsi/EXP-013/backtest_v13_combined.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
FEES_BPS = 8.0

# Shared indicator params
RSI_PERIOD = 14
SMA_PERIOD = 200
EMA_SHORT = 50
EMA_LONG = 200
ATR_PERIOD = 14
ATR_ROLLING = 200

# Exit
LONG_TS_BPS = 20.0
SHORT_TS_BPS = 30.0
MAX_BARS = 10

# Re-entry
RE_MAX = 1
RE_COOLDOWN = 2

OOS_START = "2024-01-01"
OOS_END = "2025-12-31"
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"


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

    # RSI crosses
    for thresh in [10, 20]:
        oversold = out["rsi"] < thresh
        out[f"rsi_oversold_cross_{thresh}"] = oversold & ~oversold.shift(1, fill_value=False)
    for thresh in [80, 90]:
        overbought = out["rsi"] > thresh
        out[f"rsi_overbought_cross_{thresh}"] = overbought & ~overbought.shift(1, fill_value=False)

    return out


def check_signals(i, df_arrays, version="v13"):
    """Check all signal conditions at bar i. Returns (direction, signal_type, trailing_stop_bps) or None."""
    rsi = df_arrays["rsi"][i]
    bull = df_arrays["bull"][i]
    bear = df_arrays["bear"][i]
    atr_p = df_arrays["atr_pctl"][i]
    ema_s = df_arrays["ema_sep"][i]

    if np.isnan(atr_p) or np.isnan(ema_s) or np.isnan(rsi):
        return None

    # Signal 1: V1.2 LONG — RSI crosses <20 + bull + ATR>=25 + EMA>=0.5
    if df_arrays["oversold_cross_20"][i] and bull:
        if atr_p >= 25 and ema_s >= 0.5:
            return ("LONG", "V12_LONG", LONG_TS_BPS)

    # Signal 2: V1.2 SHORT — RSI crosses >80 + bear (no filters)
    if df_arrays["overbought_cross_80"][i] and bear:
        return ("SHORT", "V12_SHORT", SHORT_TS_BPS)

    if version == "v12":
        return None

    # Signal 3: Bear LONG — RSI crosses <10 + bear + EMA>=1.0
    if df_arrays["oversold_cross_10"][i] and bear:
        if ema_s >= 1.0:
            return ("LONG", "BEAR_LONG", LONG_TS_BPS)

    # Signal 4: Bull SHORT — RSI crosses >90 + bull + ATR>=60 + EMA>=1.0
    if df_arrays["overbought_cross_90"][i] and bull:
        if atr_p >= 60 and ema_s >= 1.0:
            return ("SHORT", "BULL_SHORT", SHORT_TS_BPS)

    return None


def run_backtest(df: pd.DataFrame, version: str = "v13") -> list[dict]:
    """Run combined backtest. version='v12' for baseline, 'v13' for all 4 signals."""
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    times = df.index

    df_arrays = {
        "rsi": df["rsi"].values,
        "bull": df["bull_market"].values,
        "bear": df["bear_market"].values,
        "atr_pctl": df["atr_percentile"].values,
        "ema_sep": df["ema_separation"].values,
        "oversold_cross_10": df["rsi_oversold_cross_10"].values,
        "oversold_cross_20": df["rsi_oversold_cross_20"].values,
        "overbought_cross_80": df["rsi_overbought_cross_80"].values,
        "overbought_cross_90": df["rsi_overbought_cross_90"].values,
    }

    n = len(df)
    trades = []

    # Position state
    in_pos = False
    direction = None
    signal_type = None
    trailing_stop = 0.0
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
    last_exit_direction = None
    last_exit_signal_type = None
    last_exit_ts = 0.0
    reentry_count = 0

    def close_position(gross_bps, exit_time_val, exit_bar, reason):
        nonlocal in_pos, last_exit_reason, last_exit_bar, last_exit_direction
        nonlocal last_exit_signal_type, last_exit_ts, reentry_count

        if direction == "LONG":
            exit_px = entry_price * (1 + gross_bps / 10000)
        else:
            exit_px = entry_price * (1 - gross_bps / 10000)

        trades.append({
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": exit_time_val,
            "direction": direction,
            "signal_type": signal_type,
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
        last_exit_direction = direction
        last_exit_signal_type = signal_type
        last_exit_ts = trailing_stop
        if is_reentry:
            reentry_count += 1
        elif reason == "TRAILING_STOP":
            reentry_count = 0
        in_pos = False

    i = 0
    while i < n:
        if in_pos:
            bars_held += 1

            if direction == "LONG":
                bar_mfe = (highs[i] - entry_price) / entry_price * 10000
                bar_mae = (lows[i] - entry_price) / entry_price * 10000
                bar_pnl = (closes[i] - entry_price) / entry_price * 10000
            else:
                bar_mfe = (entry_price - lows[i]) / entry_price * 10000
                bar_mae = (entry_price - highs[i]) / entry_price * 10000
                bar_pnl = (entry_price - closes[i]) / entry_price * 10000

            if bar_mfe > mfe:
                mfe = bar_mfe
            if bar_mae < mae:
                mae = bar_mae
            if bar_mfe > highest_profit:
                highest_profit = bar_mfe

            drawdown = highest_profit - bar_pnl
            if drawdown >= trailing_stop and highest_profit > 0:
                exit_profit = highest_profit - trailing_stop
                close_position(exit_profit, times[i], i, "TRAILING_STOP")
                i += 1
                continue

            if bars_held >= MAX_BARS:
                close_position(bar_pnl, times[i], i, "TIME_EXIT")
                i += 1
                continue

            i += 1
            continue

        # Re-entry check
        if last_exit_reason == "TRAILING_STOP" and reentry_count < RE_MAX:
            if i == last_exit_bar + RE_COOLDOWN:
                # Check regime still valid for re-entry direction
                regime_ok = False
                if last_exit_direction == "LONG":
                    if last_exit_signal_type == "V12_LONG":
                        regime_ok = df_arrays["bull"][i]
                    elif last_exit_signal_type == "BEAR_LONG":
                        regime_ok = df_arrays["bear"][i]
                elif last_exit_direction == "SHORT":
                    if last_exit_signal_type == "V12_SHORT":
                        regime_ok = df_arrays["bear"][i]
                    elif last_exit_signal_type == "BULL_SHORT":
                        regime_ok = df_arrays["bull"][i]

                if regime_ok and i + 1 < n:
                    direction = last_exit_direction
                    signal_type = last_exit_signal_type
                    trailing_stop = last_exit_ts
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
                    last_exit_reason = None

        # Check for new signal
        sig = check_signals(i, df_arrays, version)
        if sig is not None:
            reentry_count = 0
            last_exit_reason = None
            last_exit_bar = -999

            direction = sig[0]
            signal_type = sig[1]
            trailing_stop = sig[2]

            if i + 1 < n:
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


def print_results(trades: list[dict], label: str) -> dict:
    if not trades:
        print(f"  {label}: NO TRADES")
        return {}

    tdf = pd.DataFrame(trades)
    winners = tdf[tdf["net_profit_bps"] > 0]
    losers = tdf[tdf["net_profit_bps"] <= 0]
    gw = winners["net_profit_bps"].sum() if len(winners) > 0 else 0
    gl = abs(losers["net_profit_bps"].sum()) if len(losers) > 0 else 1
    pf = gw / gl

    total_bps = tdf["net_profit_bps"].sum()
    equity = tdf["net_profit_bps"].cumsum()
    max_dd = (equity - equity.cummax()).min()

    orig = tdf[~tdf["is_reentry"]]
    re = tdf[tdf["is_reentry"]]

    print(f"\n  {'='*65}")
    print(f"  {label}")
    print(f"  {'='*65}")
    print(f"  Trades: {len(tdf)} ({len(orig)} orig + {len(re)} RE)")
    print(f"  Win Rate: {len(winners)/len(tdf)*100:.1f}%")
    print(f"  Net Profit: {total_bps:+.0f} bps")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg/Trade: {tdf['net_profit_bps'].mean():+.1f} bps")
    print(f"  Max Drawdown: {max_dd:+.0f} bps")

    # By signal type
    print(f"\n  By Signal Type:")
    for st in sorted(tdf["signal_type"].unique()):
        st_df = tdf[tdf["signal_type"] == st]
        st_w = st_df[st_df["net_profit_bps"] > 0]
        st_l = st_df[st_df["net_profit_bps"] <= 0]
        st_gw = st_w["net_profit_bps"].sum() if len(st_w) > 0 else 0
        st_gl = abs(st_l["net_profit_bps"].sum()) if len(st_l) > 0 else 1
        print(f"    {st:<15} {len(st_df):>4}t | Win: {len(st_w)/len(st_df)*100:>5.1f}% | "
              f"Net: {st_df['net_profit_bps'].sum():>+7.0f} bps | PF: {st_gw/st_gl:.2f}")

    # By direction
    print(f"\n  By Direction:")
    for d in ["LONG", "SHORT"]:
        d_df = tdf[tdf["direction"] == d]
        if len(d_df) == 0:
            continue
        d_w = d_df[d_df["net_profit_bps"] > 0]
        d_l = d_df[d_df["net_profit_bps"] <= 0]
        d_gw = d_w["net_profit_bps"].sum() if len(d_w) > 0 else 0
        d_gl = abs(d_l["net_profit_bps"].sum()) if len(d_l) > 0 else 1
        print(f"    {d:<10} {len(d_df):>4}t | Win: {len(d_w)/len(d_df)*100:>5.1f}% | "
              f"Net: {d_df['net_profit_bps'].sum():>+7.0f} bps | PF: {d_gw/d_gl:.2f}")

    # Year split
    tdf["year"] = pd.to_datetime(tdf["entry_time"]).dt.year
    print(f"\n  By Year:")
    for y in sorted(tdf["year"].unique()):
        yt = tdf[tdf["year"] == y]
        yw = yt[yt["net_profit_bps"] > 0]
        yl = yt[yt["net_profit_bps"] <= 0]
        ygw = yw["net_profit_bps"].sum() if len(yw) > 0 else 0
        ygl = abs(yl["net_profit_bps"].sum()) if len(yl) > 0 else 1
        print(f"    {y}: {len(yt):>4}t | Win: {len(yw)/len(yt)*100:>5.1f}% | "
              f"Net: {yt['net_profit_bps'].sum():>+7.0f} bps | PF: {ygw/ygl:.2f}")

        # Year + signal type
        for st in sorted(yt["signal_type"].unique()):
            st_yt = yt[yt["signal_type"] == st]
            st_yw = st_yt[st_yt["net_profit_bps"] > 0]
            print(f"      {st:<15} {len(st_yt):>3}t | Win: {len(st_yw)/len(st_yt)*100:>5.1f}% | "
                  f"Net: {st_yt['net_profit_bps'].sum():>+6.0f} bps")

    # Quarter split
    tdf["quarter"] = pd.to_datetime(tdf["entry_time"]).dt.to_period("Q").astype(str)
    print(f"\n  By Quarter:")
    for q in sorted(tdf["quarter"].unique()):
        qt = tdf[tdf["quarter"] == q]
        qw = qt[qt["net_profit_bps"] > 0]
        ql = qt[qt["net_profit_bps"] <= 0]
        qgw = qw["net_profit_bps"].sum() if len(qw) > 0 else 0
        qgl = abs(ql["net_profit_bps"].sum()) if len(ql) > 0 else 1
        print(f"    {q}: {len(qt):>3}t | Win: {len(qw)/len(qt)*100:>5.1f}% | "
              f"Net: {qt['net_profit_bps'].sum():>+6.0f} bps | PF: {qgw/qgl:.2f}")

    return {
        "trades": len(tdf),
        "win_rate": len(winners) / len(tdf) * 100,
        "net_bps": total_bps,
        "pf": pf,
        "max_dd": max_dd,
    }


def main():
    print("=" * 70)
    print("EXP-013d: V1.3 Combined Backtest")
    print("=" * 70)
    print("V1.2 signals: RSI<20+bull LONG, RSI>80+bear SHORT")
    print("V1.3 adds:    RSI<10+bear LONG (EMA>=1.0)")
    print("              RSI>90+bull SHORT (ATR>=60, EMA>=1.0)")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = compute_indicators(df)

    for period_name, start, end in [
        ("TRAIN (2020-2023)", TRAIN_START, TRAIN_END),
        ("OOS (2024-2025)", OOS_START, OOS_END),
    ]:
        test = df[start:end]
        print(f"\n\n{'#'*70}")
        print(f"  {period_name} | {len(test)} bars")
        print(f"{'#'*70}")

        # V1.2 baseline
        v12_trades = run_backtest(test, version="v12")
        v12_r = print_results(v12_trades, f"V1.2 BASELINE ({period_name})")

        # V1.3 combined
        v13_trades = run_backtest(test, version="v13")
        v13_r = print_results(v13_trades, f"V1.3 COMBINED ({period_name})")

        # Comparison
        if v12_r and v13_r:
            print(f"\n  {'='*65}")
            print(f"  COMPARISON: V1.2 vs V1.3 ({period_name})")
            print(f"  {'='*65}")
            print(f"  {'Metric':<20} {'V1.2':>12} {'V1.3':>12} {'Delta':>12}")
            print(f"  {'-'*56}")
            print(f"  {'Trades':<20} {v12_r['trades']:>12} {v13_r['trades']:>12} {v13_r['trades']-v12_r['trades']:>+12}")
            print(f"  {'Win Rate':<20} {v12_r['win_rate']:>11.1f}% {v13_r['win_rate']:>11.1f}% {v13_r['win_rate']-v12_r['win_rate']:>+11.1f}%")
            print(f"  {'Net Profit (bps)':<20} {v12_r['net_bps']:>+12.0f} {v13_r['net_bps']:>+12.0f} {v13_r['net_bps']-v12_r['net_bps']:>+12.0f}")
            print(f"  {'Profit Factor':<20} {v12_r['pf']:>12.2f} {v13_r['pf']:>12.2f} {v13_r['pf']-v12_r['pf']:>+12.2f}")
            print(f"  {'Max Drawdown':<20} {v12_r['max_dd']:>+12.0f} {v13_r['max_dd']:>+12.0f} {v13_r['max_dd']-v12_r['max_dd']:>+12.0f}")


if __name__ == "__main__":
    main()
