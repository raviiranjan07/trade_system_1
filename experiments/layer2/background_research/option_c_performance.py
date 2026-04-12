"""Layer 2 — Option C: Performance-Based Regime Analysis

What conditions separate V1.3.2 WINNERS from LOSERS?

Approach:
  1. Run V1.3.2 backtest on TRAIN (2020-2023) and OOS (2024-2025)
  2. For each trade, capture market features at entry time
  3. Compare features for winners vs losers
  4. Find which features best predict trade outcome
  5. Build simple "good market" vs "bad market" classifier

Run: PYTHONPATH=src python experiments/layer2_regime/option_c_performance.py
"""

import sys
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from v12.config.loader import load_config
from v12.backtest import run_backtest
from v12.strategy import V12Strategy

logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).parent
DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")


def build_trade_features(trades, df_indicators):
    """For each trade, extract market features at signal time."""
    rows = []
    for t in trades:
        sig_time = pd.Timestamp(t.signal_time)
        if sig_time not in df_indicators.index:
            continue
        bar = df_indicators.loc[sig_time]

        # Entry bar (next bar after signal)
        entry_time = pd.Timestamp(t.entry_time)
        entry_idx = df_indicators.index.get_loc(entry_time) if entry_time in df_indicators.index else None

        # Rolling features around signal bar
        sig_idx = df_indicators.index.get_loc(sig_time)

        # Look-back features (last N bars before signal)
        lb = 10
        if sig_idx >= lb:
            recent = df_indicators.iloc[sig_idx - lb:sig_idx]
            bar_range_avg = ((recent["high"] - recent["low"]) / recent["close"] * 10000).mean()
            vol_ratio = recent["volume"].iloc[-1] / recent["volume"].mean() if recent["volume"].mean() > 0 else 1
            close_changes = recent["close"].pct_change().dropna()
            recent_volatility = close_changes.std() * 10000 if len(close_changes) > 1 else 0
            recent_trend = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0] * 10000
        else:
            bar_range_avg = np.nan
            vol_ratio = np.nan
            recent_volatility = np.nan
            recent_trend = np.nan

        # Hour and day of week
        hour = sig_time.hour
        dow = sig_time.dayofweek  # 0=Mon, 6=Sun

        rows.append({
            "signal_time": sig_time,
            "direction": t.direction,
            "signal_type": t.signal_type,
            "net_profit_bps": t.net_profit_bps,
            "mfe_bps": t.mfe_bps,
            "mae_bps": t.mae_bps,
            "exit_reason": t.exit_reason,
            "exit_bar": t.exit_bar,
            "is_winner": t.net_profit_bps > 0,
            # Market features at signal time
            "rsi": bar["rsi"],
            "atr_percentile": bar["atr_percentile"],
            "ema_separation": bar["ema_separation"],
            "bull_market": bar["bull_market"],
            "bear_market": bar["bear_market"],
            # Price relative to SMA
            "price_vs_sma_pct": (bar["close"] - bar["sma"]) / bar["sma"] * 100 if bar["sma"] > 0 else 0,
            # Derived features
            "bar_range_avg_10": bar_range_avg,
            "vol_ratio": vol_ratio,
            "recent_volatility": recent_volatility,
            "recent_trend_bps": recent_trend,
            "hour_utc": hour,
            "day_of_week": dow,
            # Signal bar range
            "signal_bar_range_bps": (bar["high"] - bar["low"]) / bar["close"] * 10000,
        })

    return pd.DataFrame(rows)


def analyze_feature(df, feature, n_bins=4, label=""):
    """Analyze how a feature separates winners from losers."""
    valid = df.dropna(subset=[feature])
    if len(valid) < 20:
        return None

    # Create bins
    try:
        valid["bin"] = pd.qcut(valid[feature], n_bins, duplicates="drop")
    except ValueError:
        return None

    results = []
    for bin_label, group in valid.groupby("bin", observed=True):
        n = len(group)
        wins = group["is_winner"].sum()
        total_bps = group["net_profit_bps"].sum()
        avg_bps = group["net_profit_bps"].mean()
        win_rate = wins / n * 100 if n > 0 else 0

        gw = group[group["net_profit_bps"] > 0]["net_profit_bps"].sum()
        gl = abs(group[group["net_profit_bps"] <= 0]["net_profit_bps"].sum())
        pf = gw / gl if gl > 0 else 99

        results.append({
            "bin": str(bin_label),
            "trades": n,
            "win%": round(win_rate, 1),
            "avg_bps": round(avg_bps, 1),
            "total_bps": round(total_bps, 0),
            "PF": round(pf, 2),
        })

    return pd.DataFrame(results)


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    config = load_config()

    # Load data and compute indicators
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    strategy = V12Strategy(config)
    df = strategy.compute_indicators(df)

    # =========================================================================
    # Run backtest on TRAIN and OOS separately
    # =========================================================================
    print_separator("RUNNING BACKTESTS")

    trades_train = run_backtest(config, start="2020-01-01", end="2023-12-31")
    trades_oos = run_backtest(config, start="2024-01-01", end="2025-12-31")
    print(f"TRAIN trades: {len(trades_train)}")
    print(f"OOS trades:   {len(trades_oos)}")

    # =========================================================================
    # Build feature DataFrames
    # =========================================================================
    print_separator("BUILDING FEATURE MATRICES")

    tf_train = build_trade_features(trades_train, df)
    tf_oos = build_trade_features(trades_oos, df)
    print(f"TRAIN features built: {len(tf_train)} trades")
    print(f"OOS features built:   {len(tf_oos)} trades")

    # Save trade features for later use
    tf_train.to_csv(OUT_DIR / "trade_features_train.csv", index=False)
    tf_oos.to_csv(OUT_DIR / "trade_features_oos.csv", index=False)

    # =========================================================================
    # Winner vs Loser comparison (TRAIN)
    # =========================================================================
    print_separator("WINNER vs LOSER COMPARISON (TRAIN)")

    winners = tf_train[tf_train["is_winner"]]
    losers = tf_train[~tf_train["is_winner"]]
    print(f"Winners: {len(winners)} ({len(winners)/len(tf_train)*100:.1f}%)")
    print(f"Losers:  {len(losers)} ({len(losers)/len(tf_train)*100:.1f}%)")

    features = [
        "rsi", "atr_percentile", "ema_separation", "price_vs_sma_pct",
        "bar_range_avg_10", "vol_ratio", "recent_volatility",
        "recent_trend_bps", "signal_bar_range_bps", "hour_utc", "day_of_week",
    ]

    print(f"\n{'Feature':<25} {'Win Mean':>10} {'Loss Mean':>10} {'Diff':>10} {'Cohen d':>10}")
    print("-" * 70)
    for feat in features:
        w_vals = winners[feat].dropna()
        l_vals = losers[feat].dropna()
        if len(w_vals) < 5 or len(l_vals) < 5:
            continue
        w_mean = w_vals.mean()
        l_mean = l_vals.mean()
        diff = w_mean - l_mean
        pooled_std = np.sqrt((w_vals.std()**2 + l_vals.std()**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"{feat:<25} {w_mean:>10.2f} {l_mean:>10.2f} {diff:>10.2f} {cohens_d:>10.3f}")

    # =========================================================================
    # Feature-by-feature quartile analysis (TRAIN)
    # =========================================================================
    print_separator("FEATURE QUARTILE ANALYSIS (TRAIN)")

    for feat in features:
        result = analyze_feature(tf_train, feat, n_bins=4, label=feat)
        if result is not None:
            print(f"\n--- {feat} ---")
            print(result.to_string(index=False))

    # =========================================================================
    # By Signal Type breakdown (TRAIN)
    # =========================================================================
    print_separator("BY SIGNAL TYPE (TRAIN)")

    for sig_type in sorted(tf_train["signal_type"].unique()):
        subset = tf_train[tf_train["signal_type"] == sig_type]
        w = subset[subset["is_winner"]]
        l = subset[~subset["is_winner"]]
        gw = w["net_profit_bps"].sum() if len(w) > 0 else 0
        gl = abs(l["net_profit_bps"].sum()) if len(l) > 0 else 1
        print(f"\n{sig_type}: {len(subset)}t | Win: {len(w)/len(subset)*100:.1f}% | "
              f"Total: {subset['net_profit_bps'].sum():+.0f} bps | PF: {gw/gl:.2f}")

        # Top 3 separating features for this signal type
        for feat in ["atr_percentile", "ema_separation", "signal_bar_range_bps", "hour_utc"]:
            result = analyze_feature(subset, feat, n_bins=3)
            if result is not None:
                print(f"  {feat}:")
                for _, row in result.iterrows():
                    print(f"    {row['bin']:<30} {row['trades']:>3}t  Win:{row['win%']:>5.1f}%  "
                          f"Avg:{row['avg_bps']:>+6.1f}  PF:{row['PF']:>5.2f}")

    # =========================================================================
    # Time analysis (TRAIN)
    # =========================================================================
    print_separator("TIME ANALYSIS (TRAIN)")

    # By hour
    print("\n--- By Hour (UTC) ---")
    print(f"{'Hour':<6} {'Trades':>6} {'Win%':>6} {'Avg bps':>8} {'Total':>8} {'PF':>6}")
    for hour in range(24):
        h_trades = tf_train[tf_train["hour_utc"] == hour]
        if len(h_trades) < 3:
            continue
        hw = h_trades[h_trades["is_winner"]]
        hl = h_trades[~h_trades["is_winner"]]
        gw = hw["net_profit_bps"].sum() if len(hw) > 0 else 0
        gl = abs(hl["net_profit_bps"].sum()) if len(hl) > 0 else 1
        print(f"{hour:>4}   {len(h_trades):>5}  {len(hw)/len(h_trades)*100:>5.1f}  "
              f"{h_trades['net_profit_bps'].mean():>+7.1f}  {h_trades['net_profit_bps'].sum():>+7.0f}  {gw/gl:>5.2f}")

    # By day of week
    print("\n--- By Day of Week ---")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"{'Day':<6} {'Trades':>6} {'Win%':>6} {'Avg bps':>8} {'Total':>8} {'PF':>6}")
    for dow in range(7):
        d_trades = tf_train[tf_train["day_of_week"] == dow]
        if len(d_trades) < 3:
            continue
        dw = d_trades[d_trades["is_winner"]]
        dl = d_trades[~d_trades["is_winner"]]
        gw = dw["net_profit_bps"].sum() if len(dw) > 0 else 0
        gl = abs(dl["net_profit_bps"].sum()) if len(dl) > 0 else 1
        print(f"{day_names[dow]:<6} {len(d_trades):>5}  {len(dw)/len(d_trades)*100:>5.1f}  "
              f"{d_trades['net_profit_bps'].mean():>+7.1f}  {d_trades['net_profit_bps'].sum():>+7.0f}  {gw/gl:>5.2f}")

    # =========================================================================
    # OOS VALIDATION of TRAIN findings
    # =========================================================================
    print_separator("OOS VALIDATION")

    # Repeat key analyses on OOS
    winners_oos = tf_oos[tf_oos["is_winner"]]
    losers_oos = tf_oos[~tf_oos["is_winner"]]
    print(f"OOS Winners: {len(winners_oos)} ({len(winners_oos)/len(tf_oos)*100:.1f}%)")
    print(f"OOS Losers:  {len(losers_oos)} ({len(losers_oos)/len(tf_oos)*100:.1f}%)")

    print(f"\n{'Feature':<25} {'Win Mean':>10} {'Loss Mean':>10} {'Diff':>10} {'Cohen d':>10}")
    print("-" * 70)
    for feat in features:
        w_vals = winners_oos[feat].dropna()
        l_vals = losers_oos[feat].dropna()
        if len(w_vals) < 5 or len(l_vals) < 5:
            continue
        w_mean = w_vals.mean()
        l_mean = l_vals.mean()
        diff = w_mean - l_mean
        pooled_std = np.sqrt((w_vals.std()**2 + l_vals.std()**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"{feat:<25} {w_mean:>10.2f} {l_mean:>10.2f} {diff:>10.2f} {cohens_d:>10.3f}")

    # OOS feature quartiles
    print("\n--- OOS Feature Quartiles ---")
    for feat in features:
        result = analyze_feature(tf_oos, feat, n_bins=4, label=feat)
        if result is not None:
            print(f"\n  {feat}:")
            for _, row in result.iterrows():
                print(f"    {row['bin']:<30} {row['trades']:>3}t  Win:{row['win%']:>5.1f}%  "
                      f"Avg:{row['avg_bps']:>+6.1f}  PF:{row['PF']:>5.2f}")

    # =========================================================================
    # REGIME SCORING: Combine top features into "regime score"
    # =========================================================================
    print_separator("REGIME SCORING (TRAIN)")

    # Based on TRAIN analysis, define good/bad conditions
    # These thresholds will be refined after seeing results
    conditions = {
        "atr_high": tf_train["atr_percentile"] >= 50,
        "ema_strong": tf_train["ema_separation"] >= 1.0,
        "big_signal_bar": tf_train["signal_bar_range_bps"] >= 20,
        "not_asia_night": ~tf_train["hour_utc"].between(0, 4),
    }

    print(f"\n{'Condition':<20} {'True':>6} {'Win%':>6} {'AvgBps':>8} {'PF':>6}  |  "
          f"{'False':>6} {'Win%':>6} {'AvgBps':>8} {'PF':>6}")
    print("-" * 90)

    for name, mask in conditions.items():
        for is_true in [True, False]:
            subset = tf_train[mask] if is_true else tf_train[~mask]
            if len(subset) == 0:
                continue
            w = subset[subset["is_winner"]]
            l = subset[~subset["is_winner"]]
            gw = w["net_profit_bps"].sum() if len(w) > 0 else 0
            gl = abs(l["net_profit_bps"].sum()) if len(l) > 0 else 1
            wr = len(w) / len(subset) * 100
            avg = subset["net_profit_bps"].mean()
            pf = gw / gl

            if is_true:
                print(f"{name:<20} {len(subset):>5}  {wr:>5.1f}  {avg:>+7.1f}  {pf:>5.2f}  |  ", end="")
            else:
                print(f"{len(subset):>5}  {wr:>5.1f}  {avg:>+7.1f}  {pf:>5.2f}")

    # Combined score
    print("\n--- Combined Regime Score (count of positive conditions) ---")
    score = sum([c.astype(int) for c in conditions.values()])
    tf_train["regime_score"] = score

    print(f"{'Score':<8} {'Trades':>6} {'Win%':>6} {'Avg bps':>8} {'Total':>8} {'PF':>6}")
    for s in sorted(tf_train["regime_score"].unique()):
        subset = tf_train[tf_train["regime_score"] == s]
        w = subset[subset["is_winner"]]
        l = subset[~subset["is_winner"]]
        gw = w["net_profit_bps"].sum() if len(w) > 0 else 0
        gl = abs(l["net_profit_bps"].sum()) if len(l) > 0 else 1
        print(f"{s:<8} {len(subset):>5}  {len(w)/len(subset)*100:>5.1f}  "
              f"{subset['net_profit_bps'].mean():>+7.1f}  {subset['net_profit_bps'].sum():>+7.0f}  {gw/gl:>5.2f}")

    # Validate regime score on OOS
    print("\n--- OOS Regime Score Validation ---")
    conditions_oos = {
        "atr_high": tf_oos["atr_percentile"] >= 50,
        "ema_strong": tf_oos["ema_separation"] >= 1.0,
        "big_signal_bar": tf_oos["signal_bar_range_bps"] >= 20,
        "not_asia_night": ~tf_oos["hour_utc"].between(0, 4),
    }
    score_oos = sum([c.astype(int) for c in conditions_oos.values()])
    tf_oos["regime_score"] = score_oos

    print(f"{'Score':<8} {'Trades':>6} {'Win%':>6} {'Avg bps':>8} {'Total':>8} {'PF':>6}")
    for s in sorted(tf_oos["regime_score"].unique()):
        subset = tf_oos[tf_oos["regime_score"] == s]
        w = subset[subset["is_winner"]]
        l = subset[~subset["is_winner"]]
        gw = w["net_profit_bps"].sum() if len(w) > 0 else 0
        gl = abs(l["net_profit_bps"].sum()) if len(l) > 0 else 1
        print(f"{s:<8} {len(subset):>5}  {len(w)/len(subset)*100:>5.1f}  "
              f"{subset['net_profit_bps'].mean():>+7.1f}  {subset['net_profit_bps'].sum():>+7.0f}  {gw/gl:>5.2f}")


if __name__ == "__main__":
    main()
