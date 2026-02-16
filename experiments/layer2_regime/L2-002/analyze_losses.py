"""L2-002: Loss analysis for top config (ACTIVE_TREND + ROC5_-0.3_0.3)

Answers: WHERE and WHEN does the strategy lose?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "L2-001"))

import numpy as np
import pandas as pd

from v12.config.constants import FEES_BPS
from L2_001_feature_revalidation import compute_all_features
from framework import (
    RegimeClassifier, GateConfig, SignalConfig, ExitConfig,
    apply_gate, detect_signals, precompute_all_trades, run_backtest
)

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OOS_START, OOS_END = "2024-01-01", "2025-12-31"
TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"


def main():
    print("=" * 80)
    print("L2-002 LOSS ANALYSIS: ACTIVE_TREND + ROC5_-0.3_0.3")
    print("=" * 80)

    # Load and compute
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = compute_all_features(df)

    train = df[TRAIN_START:TRAIN_END].copy()
    oos = df[OOS_START:OOS_END].copy()

    classifier = RegimeClassifier(train)
    exit_config = ExitConfig()
    precomputed_oos = precompute_all_trades(oos, exit_config)

    # Run the top config
    gate = GateConfig(name="ACTIVE_TREND", gates=[("regime", "==", "ACTIVE_TREND")])
    signal = SignalConfig(name="ROC5_-0.3_0.3", feature="roc5", long_threshold=-0.3, short_threshold=0.3)

    br = run_backtest(oos, gate, signal, precomputed_oos, classifier)
    trades = br.trades.copy()

    # Add features from OOS data
    trades["date"] = oos.index[trades["entry_bar"].values.astype(int)]
    trades["year"] = trades["date"].dt.year
    trades["month"] = trades["date"].dt.month
    trades["hour_utc"] = trades["date"].dt.hour
    trades["day_of_week"] = trades["date"].dt.dayofweek  # 0=Mon, 6=Sun

    # Add state vector features at entry
    for feat in ["atr_pct", "ema_separation", "range_bps", "rsi7", "roc5",
                 "volume_ratio", "atr_percentile", "range_position"]:
        trades[feat] = oos[feat].iloc[trades["entry_bar"].values.astype(int)].values

    trades["is_winner"] = trades["net_bps"] > 0
    winners = trades[trades["is_winner"]]
    losers = trades[~trades["is_winner"]]

    print(f"\nTotal trades: {len(trades)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"Losers:  {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print(f"\nWinner avg: {winners['net_bps'].mean():+.1f} bps")
    print(f"Loser avg:  {losers['net_bps'].mean():+.1f} bps")
    print(f"Winner median: {winners['net_bps'].median():+.1f} bps")
    print(f"Loser median:  {losers['net_bps'].median():+.1f} bps")

    # --- 1. Exit reason breakdown ---
    print(f"\n{'='*60}")
    print("1. EXIT REASON BREAKDOWN")
    print(f"{'='*60}")
    for reason in ["TS", "TIME"]:
        subset = trades[trades["exit_reason"] == reason]
        w = subset[subset["is_winner"]]
        l = subset[~subset["is_winner"]]
        print(f"\n{reason}: {len(subset)} trades, {len(w)} win ({len(w)/len(subset)*100:.1f}%), "
              f"total {subset['net_bps'].sum():+.0f} bps")
        if len(l) > 0:
            print(f"  Losers: {len(l)}, avg {l['net_bps'].mean():+.1f} bps, "
                  f"worst {l['net_bps'].min():+.1f} bps")

    # --- 2. LONG vs SHORT ---
    print(f"\n{'='*60}")
    print("2. LONG vs SHORT")
    print(f"{'='*60}")
    for d in ["LONG", "SHORT"]:
        subset = trades[trades["direction"] == d]
        wr = (subset["net_bps"] > 0).mean() * 100
        total = subset["net_bps"].sum()
        w_sum = subset[subset["net_bps"] > 0]["net_bps"].sum()
        l_sum = abs(subset[subset["net_bps"] <= 0]["net_bps"].sum())
        pf = w_sum / l_sum if l_sum > 0 else float("inf")
        print(f"{d}: {len(subset)} trades, {wr:.1f}% win, {total:+.0f} bps, PF {pf:.2f}")
        # Show losers
        losers_d = subset[subset["net_bps"] <= 0]
        if len(losers_d) > 0:
            print(f"  Losers: {len(losers_d)}, avg {losers_d['net_bps'].mean():+.1f}, "
                  f"median {losers_d['net_bps'].median():+.1f}, worst {losers_d['net_bps'].min():+.1f}")

    # --- 3. By hour UTC ---
    print(f"\n{'='*60}")
    print("3. BY HOUR (UTC)")
    print(f"{'='*60}")
    print(f"{'Hour':>5} {'Trades':>7} {'Win%':>6} {'Total':>8} {'Avg':>7} {'Losers':>7} {'Loss avg':>9}")
    print("-" * 55)
    for hour in sorted(trades["hour_utc"].unique()):
        subset = trades[trades["hour_utc"] == hour]
        wr = (subset["net_bps"] > 0).mean() * 100
        total = subset["net_bps"].sum()
        avg = subset["net_bps"].mean()
        n_loss = (~subset["is_winner"]).sum()
        loss_avg = subset[~subset["is_winner"]]["net_bps"].mean() if n_loss > 0 else 0
        print(f"{hour:>5} {len(subset):>7} {wr:>5.1f}% {total:>+7.0f} {avg:>+6.1f} {n_loss:>7} {loss_avg:>+8.1f}")

    # --- 4. By day of week ---
    print(f"\n{'='*60}")
    print("4. BY DAY OF WEEK")
    print(f"{'='*60}")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"{'Day':>5} {'Trades':>7} {'Win%':>6} {'Total':>8} {'Avg':>7}")
    print("-" * 40)
    for dow in range(7):
        subset = trades[trades["day_of_week"] == dow]
        if len(subset) == 0:
            continue
        wr = (subset["net_bps"] > 0).mean() * 100
        total = subset["net_bps"].sum()
        avg = subset["net_bps"].mean()
        print(f"{day_names[dow]:>5} {len(subset):>7} {wr:>5.1f}% {total:>+7.0f} {avg:>+6.1f}")

    # --- 5. By month ---
    print(f"\n{'='*60}")
    print("5. BY MONTH (OOS)")
    print(f"{'='*60}")
    print(f"{'Month':>8} {'Trades':>7} {'Win%':>6} {'Total':>8} {'Avg':>7}")
    print("-" * 40)
    trades["ym"] = trades["date"].dt.to_period("M")
    for ym in sorted(trades["ym"].unique()):
        subset = trades[trades["ym"] == ym]
        wr = (subset["net_bps"] > 0).mean() * 100
        total = subset["net_bps"].sum()
        avg = subset["net_bps"].mean()
        print(f"{str(ym):>8} {len(subset):>7} {wr:>5.1f}% {total:>+7.0f} {avg:>+6.1f}")

    # --- 6. Feature comparison: winners vs losers ---
    print(f"\n{'='*60}")
    print("6. FEATURE COMPARISON: WINNERS vs LOSERS")
    print(f"{'='*60}")
    feats = ["atr_pct", "ema_separation", "range_bps", "rsi7", "roc5",
             "volume_ratio", "atr_percentile", "range_position", "mfe_bps"]
    print(f"{'Feature':<20} {'Winner med':>11} {'Loser med':>11} {'Diff':>8}")
    print("-" * 55)
    for f in feats:
        wm = winners[f].median()
        lm = losers[f].median()
        diff = wm - lm
        print(f"{f:<20} {wm:>10.2f} {lm:>10.2f} {diff:>+7.2f}")

    # --- 7. MFE distribution of losers ---
    print(f"\n{'='*60}")
    print("7. LOSER MFE DISTRIBUTION (how far did they go right?)")
    print(f"{'='*60}")
    print(f"MFE = 0 (never positive): {(losers['mfe_bps'] == 0).sum()} ({(losers['mfe_bps'] == 0).mean()*100:.1f}%)")
    print(f"MFE 0-10 bps:  {((losers['mfe_bps'] > 0) & (losers['mfe_bps'] < 10)).sum()}")
    print(f"MFE 10-20 bps: {((losers['mfe_bps'] >= 10) & (losers['mfe_bps'] < 20)).sum()}")
    print(f"MFE 20-30 bps: {((losers['mfe_bps'] >= 20) & (losers['mfe_bps'] < 30)).sum()}")
    print(f"MFE 30+ bps:   {(losers['mfe_bps'] >= 30).sum()}")
    print(f"\nLoser MFE median: {losers['mfe_bps'].median():.1f} bps")
    print(f"Loser MFE mean:   {losers['mfe_bps'].mean():.1f} bps")

    # --- 8. Worst 20 trades ---
    print(f"\n{'='*60}")
    print("8. WORST 20 TRADES")
    print(f"{'='*60}")
    worst = trades.nsmallest(20, "net_bps")
    print(f"{'Date':<20} {'Dir':>5} {'Net':>7} {'MFE':>6} {'Exit':>5} {'Bars':>5} "
          f"{'ATR%':>6} {'EMA%':>6} {'RSI':>5} {'ROC5':>6}")
    print("-" * 80)
    for _, t in worst.iterrows():
        print(f"{str(t['date']):<20} {t['direction']:>5} {t['net_bps']:>+6.0f} {t['mfe_bps']:>5.0f} "
              f"{t['exit_reason']:>5} {t['bars_held']:>5} "
              f"{t['atr_pct']:>5.2f} {t['ema_separation']:>5.2f} {t['rsi7']:>4.0f} {t['roc5']:>+5.2f}")

    # --- 9. Losing streaks ---
    print(f"\n{'='*60}")
    print("9. LOSING STREAKS")
    print(f"{'='*60}")
    streak = 0
    max_streak = 0
    streak_bps = 0
    max_streak_bps = 0
    streaks = []
    for _, t in trades.iterrows():
        if t["net_bps"] <= 0:
            streak += 1
            streak_bps += t["net_bps"]
        else:
            if streak > 0:
                streaks.append((streak, streak_bps))
            streak = 0
            streak_bps = 0
    if streak > 0:
        streaks.append((streak, streak_bps))

    if streaks:
        streaks.sort(key=lambda x: x[0], reverse=True)
        print(f"Max losing streak: {streaks[0][0]} trades ({streaks[0][1]:+.0f} bps)")
        print(f"\nTop 10 streaks:")
        for s_len, s_bps in streaks[:10]:
            print(f"  {s_len} trades: {s_bps:+.0f} bps")

    # --- 10. Drawdown analysis ---
    print(f"\n{'='*60}")
    print("10. DRAWDOWN ANALYSIS")
    print(f"{'='*60}")
    cumsum = trades["net_bps"].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    print(f"Max drawdown: {drawdown.min():+.0f} bps")
    worst_dd_idx = drawdown.idxmin()
    print(f"At trade #{worst_dd_idx}, date: {trades.loc[worst_dd_idx, 'date']}")

    # Find the drawdown period
    peak_idx = cumsum[:worst_dd_idx].idxmax()
    print(f"Drawdown period: trade #{peak_idx} to #{worst_dd_idx}")
    print(f"  Peak: {cumsum[peak_idx]:+.0f} bps -> Trough: {cumsum[worst_dd_idx]:+.0f} bps")
    dd_trades = trades.loc[peak_idx:worst_dd_idx]
    dd_losers = dd_trades[dd_trades["net_bps"] <= 0]
    print(f"  Trades in period: {len(dd_trades)}, Losers: {len(dd_losers)}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
