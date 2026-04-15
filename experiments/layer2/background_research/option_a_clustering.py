"""Layer 2 — Option A: Feature-Based Clustering

Find natural market regimes by clustering rolling feature windows.

Approach:
  1. Compute rolling features on ALL bars (not just trade bars)
  2. K-means / DBSCAN to find natural clusters
  3. Map V1.3.2 trades to their cluster
  4. Check if performance differs per cluster
  5. Validate on OOS

Run: PYTHONPATH=src python experiments/layer2_regime/option_a_clustering.py
"""

import sys
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from engine.config.loader import load_config
from engine.backtest import run_backtest
from engine.strategy import V12Strategy

logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).parent
DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")


def compute_regime_features(df):
    """Compute rolling regime features for every bar."""
    out = df.copy()

    # Volatility: ATR bps (already have atr, convert to bps)
    out["atr_bps"] = out["atr"] / out["close"] * 10000

    # Trend strength: EMA separation (already computed)
    # Already have: ema_separation

    # Momentum: RSI distance from 50 (how extreme)
    out["rsi_extremity"] = abs(out["rsi"] - 50)

    # Price position vs SMA200 (%)
    out["price_vs_sma"] = (out["close"] - out["sma"]) / out["sma"] * 100

    # Rolling bar range (avg last 20 bars)
    bar_range = (out["high"] - out["low"]) / out["close"] * 10000
    out["avg_bar_range_20"] = bar_range.rolling(20).mean()

    # Rolling volume ratio (current / 20-bar avg)
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["vol_ratio_20"] = out["volume"] / out["vol_sma20"]

    # Directional consistency: % of last 10 bars that moved in same direction as overall trend
    close_diff = out["close"].diff()
    trend_10 = out["close"] - out["close"].shift(10)
    same_dir = ((close_diff > 0) & (trend_10 > 0)) | ((close_diff < 0) & (trend_10 < 0))
    out["trend_consistency_10"] = same_dir.rolling(10).mean()

    # Return over last 20 bars (bps)
    out["return_20"] = out["close"].pct_change(20) * 10000

    return out


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
    df = compute_regime_features(df)

    # Features for clustering
    cluster_features = [
        "atr_bps",
        "ema_separation",
        "rsi_extremity",
        "price_vs_sma",
        "avg_bar_range_20",
        "trend_consistency_10",
    ]

    # =========================================================================
    # Train on TRAIN period, validate on OOS
    # =========================================================================
    train = df["2020-01-01":"2023-12-31"].dropna(subset=cluster_features).copy()
    oos = df["2024-01-01":"2025-12-31"].dropna(subset=cluster_features).copy()

    print_separator("DATA SUMMARY")
    print(f"TRAIN bars: {len(train)}")
    print(f"OOS bars:   {len(oos)}")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[cluster_features])
    X_oos = scaler.transform(oos[cluster_features])

    # =========================================================================
    # Find optimal K
    # =========================================================================
    print_separator("FINDING OPTIMAL K (TRAIN)")

    results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train)
        sil = silhouette_score(X_train, labels, sample_size=min(10000, len(X_train)))
        inertia = km.inertia_
        results.append({"k": k, "silhouette": round(sil, 4), "inertia": round(inertia, 0)})
        print(f"  K={k}: silhouette={sil:.4f}, inertia={inertia:.0f}")

    # Pick K with best silhouette
    best = max(results, key=lambda x: x["silhouette"])
    print(f"\n  Best K={best['k']} (silhouette={best['silhouette']})")

    # =========================================================================
    # Fit best K and analyze clusters
    # =========================================================================
    for k in [3, 4, 5, best["k"]]:
        if k == best["k"] and k in [3, 4, 5]:
            continue  # avoid duplicate

        print_separator(f"CLUSTER ANALYSIS K={k}")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        train_labels = km.fit_predict(X_train)
        oos_labels = km.predict(X_oos)

        train["regime"] = train_labels
        oos["regime"] = oos_labels

        # Cluster centers (unscaled)
        centers = scaler.inverse_transform(km.cluster_centers_)
        print("\nCluster Centers:")
        print(f"{'Regime':<8}", end="")
        for f in cluster_features:
            print(f"{f:<20}", end="")
        print(f"{'bars':>8} {'%':>6}")
        for c in range(k):
            print(f"R{c:<7}", end="")
            for j, f in enumerate(cluster_features):
                print(f"{centers[c][j]:<20.2f}", end="")
            n_bars = (train_labels == c).sum()
            print(f"{n_bars:>8} {n_bars/len(train)*100:>5.1f}%")

        # =========================================================================
        # Map V1.3.2 trades to clusters (TRAIN)
        # =========================================================================
        trades_train = run_backtest(config, start="2020-01-01", end="2023-12-31")
        trades_oos = run_backtest(config, start="2024-01-01", end="2025-12-31")

        def map_trades_to_regime(trades, regime_df):
            """Map each trade to its regime at signal time."""
            rows = []
            for t in trades:
                sig_time = pd.Timestamp(t.signal_time)
                if sig_time in regime_df.index:
                    regime = regime_df.loc[sig_time, "regime"]
                    rows.append({
                        "signal_time": sig_time,
                        "direction": t.direction,
                        "signal_type": t.signal_type,
                        "net_profit_bps": t.net_profit_bps,
                        "exit_reason": t.exit_reason,
                        "regime": regime,
                    })
            return pd.DataFrame(rows)

        tf_train = map_trades_to_regime(trades_train, train)
        tf_oos = map_trades_to_regime(trades_oos, oos)

        # Performance by regime (TRAIN)
        print(f"\nV1.3.2 Performance by Regime (TRAIN, K={k}):")
        print(f"{'Regime':<8} {'Trades':>6} {'Win%':>6} {'Avg bps':>8} {'Total':>8} {'PF':>6}")
        for r in range(k):
            subset = tf_train[tf_train["regime"] == r]
            if len(subset) == 0:
                print(f"R{r:<7}     0   ---      ---      ---    ---")
                continue
            w = subset[subset["net_profit_bps"] > 0]
            l = subset[subset["net_profit_bps"] <= 0]
            gw = w["net_profit_bps"].sum() if len(w) > 0 else 0
            gl = abs(l["net_profit_bps"].sum()) if len(l) > 0 else 1
            print(f"R{r:<7} {len(subset):>5}  {len(w)/len(subset)*100:>5.1f}  "
                  f"{subset['net_profit_bps'].mean():>+7.1f}  "
                  f"{subset['net_profit_bps'].sum():>+7.0f}  {gw/gl:>5.2f}")

        # Performance by regime (OOS)
        print(f"\nV1.3.2 Performance by Regime (OOS, K={k}):")
        print(f"{'Regime':<8} {'Trades':>6} {'Win%':>6} {'Avg bps':>8} {'Total':>8} {'PF':>6}")
        for r in range(k):
            subset = tf_oos[tf_oos["regime"] == r]
            if len(subset) == 0:
                print(f"R{r:<7}     0   ---      ---      ---    ---")
                continue
            w = subset[subset["net_profit_bps"] > 0]
            l = subset[subset["net_profit_bps"] <= 0]
            gw = w["net_profit_bps"].sum() if len(w) > 0 else 0
            gl = abs(l["net_profit_bps"].sum()) if len(l) > 0 else 1
            print(f"R{r:<7} {len(subset):>5}  {len(w)/len(subset)*100:>5.1f}  "
                  f"{subset['net_profit_bps'].mean():>+7.1f}  "
                  f"{subset['net_profit_bps'].sum():>+7.0f}  {gw/gl:>5.2f}")

        # Which regime should we SKIP?
        print(f"\nRegime Recommendation (K={k}):")
        for r in range(k):
            train_sub = tf_train[tf_train["regime"] == r]
            oos_sub = tf_oos[tf_oos["regime"] == r]
            if len(train_sub) > 0 and len(oos_sub) > 0:
                train_avg = train_sub["net_profit_bps"].mean()
                oos_avg = oos_sub["net_profit_bps"].mean()
                consistent = (train_avg > 0 and oos_avg > 0) or (train_avg <= 0 and oos_avg <= 0)
                status = "TRADE" if train_avg > 10 and oos_avg > 10 else ("SKIP" if train_avg < 0 and oos_avg < 0 else "NEUTRAL")
                print(f"  R{r}: TRAIN avg={train_avg:+.1f}, OOS avg={oos_avg:+.1f} -> "
                      f"{'CONSISTENT' if consistent else 'INCONSISTENT'} -> {status}")

    print_separator("DONE")


if __name__ == "__main__":
    main()
