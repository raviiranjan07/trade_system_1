"""EXP-013c: Bull Market Extreme Overbought SHORT

Mirror of EXP-013 bear LONG:
  Bear LONG: RSI<10 + price<SMA200 -> buy extreme oversold in downtrend
  Bull SHORT: RSI>90 + price>SMA200 -> sell extreme overbought in uptrend

Grid:
  RSI thresholds: [85, 88, 90, 92, 95]  (crosses ABOVE)
  ATR percentile min: [0, 25, 40, 60, 75]
  EMA separation min: [0, 0.5, 1.0]

Exit: 30bps trailing stop (SHORT), bar 10 time exit, RE(1,2)

Run: python experiments/rsi/EXP-013/backtest_bull_short.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
FEES_BPS = 8.0

RSI_PERIOD = 14
SMA_PERIOD = 200
EMA_SHORT = 50
EMA_LONG = 200
ATR_PERIOD = 14
ATR_ROLLING = 200

TRAILING_STOP_BPS = 30.0  # SHORT uses 30bps (V1.2 setting)
MAX_BARS = 10

RE_ENABLED = True
RE_MAX = 1
RE_COOLDOWN = 2

RSI_THRESHOLDS = [85, 88, 90, 92, 95]
ATR_MINS = [0, 25, 40, 60, 75]
EMA_MINS = [0, 0.5, 1.0]

OOS_START = "2024-01-01"
OOS_END = "2025-12-31"
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    delta = out["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    out["rsi"] = 100 - (100 / (1 + rs))

    out["sma"] = out["close"].rolling(SMA_PERIOD).mean()

    out["ema_short"] = out["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    out["ema_long"] = out["close"].ewm(span=EMA_LONG, adjust=False).mean()
    out["ema_separation"] = abs(out["ema_short"] - out["ema_long"]) / out["close"] * 100

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

    out["bull_market"] = out["close"] > out["sma"]
    out["bear_market"] = out["close"] < out["sma"]

    for thresh in RSI_THRESHOLDS:
        overbought = out["rsi"] > thresh
        out[f"rsi_cross_{thresh}"] = overbought & ~overbought.shift(1, fill_value=False)

    return out


def run_backtest(df: pd.DataFrame, rsi_threshold: int, atr_min: float, ema_min: float) -> list[dict]:
    cross_col = f"rsi_cross_{rsi_threshold}"

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    times = df.index
    bull = df["bull_market"].values
    atr_pctl = df["atr_percentile"].values
    ema_sep = df["ema_separation"].values
    crosses = df[cross_col].values

    n = len(df)
    trades = []

    in_pos = False
    entry_price = 0.0
    entry_time = None
    signal_time = None
    bars_held = 0
    highest_profit = 0.0
    mfe = 0.0
    mae = 0.0
    is_reentry = False

    last_exit_reason = None
    last_exit_bar = -999
    reentry_count = 0

    def close_position(gross_bps, exit_time, exit_bar, reason):
        nonlocal in_pos, last_exit_reason, last_exit_bar, reentry_count
        exit_px = entry_price * (1 - gross_bps / 10000)  # SHORT: exit lower = profit
        trades.append({
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": "SHORT",
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
        if is_reentry:
            reentry_count += 1
        elif reason == "TRAILING_STOP":
            reentry_count = 0
        in_pos = False

    i = 0
    while i < n:
        if in_pos:
            bars_held += 1
            # SHORT P&L: price down = profit
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
            if drawdown >= TRAILING_STOP_BPS and highest_profit > 0:
                exit_profit = highest_profit - TRAILING_STOP_BPS
                close_position(exit_profit, times[i], i, "TRAILING_STOP")
                i += 1
                continue

            if bars_held >= MAX_BARS:
                close_position(bar_pnl, times[i], i, "TIME_EXIT")
                i += 1
                continue

            i += 1
            continue

        # Re-entry
        if RE_ENABLED and last_exit_reason == "TRAILING_STOP" and reentry_count < RE_MAX:
            if i == last_exit_bar + RE_COOLDOWN:
                if bull[i]:  # regime still bull (our setup)
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
                    last_exit_reason = None

        # New signal: RSI crosses above threshold + bull market
        if crosses[i] and bull[i]:
            ap = atr_pctl[i]
            es = ema_sep[i]
            if np.isnan(ap) or np.isnan(es):
                i += 1
                continue
            if ap < atr_min:
                i += 1
                continue
            if es < ema_min:
                i += 1
                continue

            if i + 1 < n:
                reentry_count = 0
                last_exit_reason = None
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


def main():
    print("=" * 70)
    print("EXP-013c: Bull Market Extreme Overbought SHORT")
    print("Entry: RSI crosses above threshold + price > SMA200 -> SHORT")
    print("Exit: 30bps trailing stop, bar 10 time exit, RE(1,2)")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = compute_indicators(df)

    for period_name, start, end in [
        ("TRAIN (2020-2023)", TRAIN_START, TRAIN_END),
        ("OOS (2024-2025)", OOS_START, OOS_END),
    ]:
        test = df[start:end]
        print(f"\n{'='*70}")
        print(f"  {period_name} | {len(test)} bars")
        print(f"{'='*70}")

        results = []
        for rsi_thresh in RSI_THRESHOLDS:
            for atr_min in ATR_MINS:
                for ema_min in EMA_MINS:
                    trades = run_backtest(test, rsi_thresh, atr_min, ema_min)
                    label = f"RSI>{rsi_thresh} ATR>={atr_min} EMA>={ema_min}"

                    if not trades:
                        results.append({
                            "rsi": rsi_thresh, "atr": atr_min, "ema": ema_min,
                            "label": label, "trades": 0, "win_rate": 0,
                            "net_bps": 0, "pf": 0, "avg_bps": 0, "max_dd": 0,
                            "y2024_bps": 0, "y2025_bps": 0,
                        })
                        continue

                    tdf = pd.DataFrame(trades)
                    winners = tdf[tdf["net_profit_bps"] > 0]
                    losers = tdf[tdf["net_profit_bps"] <= 0]
                    gw = winners["net_profit_bps"].sum() if len(winners) > 0 else 0
                    gl = abs(losers["net_profit_bps"].sum()) if len(losers) > 0 else 1
                    pf = gw / gl
                    total_bps = tdf["net_profit_bps"].sum()
                    equity = tdf["net_profit_bps"].cumsum()
                    max_dd = (equity - equity.cummax()).min()

                    tdf["year"] = pd.to_datetime(tdf["entry_time"]).dt.year
                    y2024 = tdf[tdf["year"] == 2024]["net_profit_bps"].sum() if 2024 in tdf["year"].values else 0
                    y2025 = tdf[tdf["year"] == 2025]["net_profit_bps"].sum() if 2025 in tdf["year"].values else 0

                    results.append({
                        "rsi": rsi_thresh, "atr": atr_min, "ema": ema_min,
                        "label": label,
                        "trades": len(tdf),
                        "win_rate": len(winners) / len(tdf) * 100,
                        "net_bps": total_bps,
                        "pf": pf,
                        "avg_bps": tdf["net_profit_bps"].mean(),
                        "max_dd": max_dd,
                        "y2024_bps": y2024,
                        "y2025_bps": y2025,
                    })

        rdf = pd.DataFrame(results)
        rdf = rdf[rdf["trades"] >= 10]
        rdf = rdf.sort_values("pf", ascending=False)

        print(f"\n  TOP 30 BY PF (min 10 trades):")
        print(f"  {'Config':<30} {'Tr':>4} {'Win%':>6} {'Net':>8} {'PF':>6} {'Avg':>7} {'DD':>7} {'2024':>7} {'2025':>7}")
        print(f"  {'-'*85}")
        for _, r in rdf.head(30).iterrows():
            print(f"  {r['label']:<30} {r['trades']:>4} {r['win_rate']:>5.1f}% {r['net_bps']:>+7.0f} "
                  f"{r['pf']:>6.2f} {r['avg_bps']:>+6.1f} {r['max_dd']:>+6.0f} "
                  f"{r['y2024_bps']:>+6.0f} {r['y2025_bps']:>+6.0f}")

        # Also show configs with min 20 trades
        print(f"\n  TOP 10 BY PF (min 20 trades):")
        rdf20 = rdf[rdf["trades"] >= 20].head(10)
        for _, r in rdf20.iterrows():
            print(f"  {r['label']:<30} {r['trades']:>4} {r['win_rate']:>5.1f}% {r['net_bps']:>+7.0f} "
                  f"{r['pf']:>6.2f} {r['avg_bps']:>+6.1f} {r['max_dd']:>+6.0f} "
                  f"{r['y2024_bps']:>+6.0f} {r['y2025_bps']:>+6.0f}")

        # Show ALL configs with losses for awareness
        losers_df = pd.DataFrame(results)
        losers_df = losers_df[(losers_df["trades"] >= 10) & (losers_df["net_bps"] < 0)]
        if len(losers_df) > 0:
            print(f"\n  LOSING CONFIGS (net < 0):")
            for _, r in losers_df.iterrows():
                print(f"  {r['label']:<30} {r['trades']:>4} {r['win_rate']:>5.1f}% {r['net_bps']:>+7.0f} "
                      f"{r['pf']:>6.2f}")


if __name__ == "__main__":
    main()
