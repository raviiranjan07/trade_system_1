"""Layer 2 -- Option D: Hybrid Comparison

Compare ALL regime detection methods and test regime-based filtering.

The key question: If we SKIP trades in "bad" regimes, do we improve?

Methods compared:
  1. Option C "Regime Score" (simple feature thresholds)
  2. Option A K-Means (K=2, K=3, K=4)
  3. Option B HMM (N=2, N=3, N=4)
  4. Simple EMA separation threshold (baseline)

For each method:
  - TRAIN: Identify "bad" regime
  - OOS: Skip trades in "bad" regime -> measure improvement

Run: PYTHONPATH=src python experiments/layer2_regime/option_d_hybrid.py
"""

import sys
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from engine.config.loader import load_config
from engine.backtest import run_backtest
from engine.strategy import V12Strategy

logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).parent
DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")


def compute_all_features(df):
    """Compute all features needed for all methods."""
    out = df.copy()

    # For K-means (Option A)
    out["atr_bps"] = out["atr"] / out["close"] * 10000
    out["rsi_extremity"] = abs(out["rsi"] - 50)
    out["price_vs_sma"] = (out["close"] - out["sma"]) / out["sma"] * 100
    bar_range = (out["high"] - out["low"]) / out["close"] * 10000
    out["avg_bar_range_20"] = bar_range.rolling(20).mean()
    close_diff = out["close"].diff()
    trend_10 = out["close"] - out["close"].shift(10)
    same_dir = ((close_diff > 0) & (trend_10 > 0)) | ((close_diff < 0) & (trend_10 < 0))
    out["trend_consistency_10"] = same_dir.rolling(10).mean()

    # For HMM (Option B)
    out["log_return"] = np.log(out["close"] / out["close"].shift(1)) * 10000
    out["rolling_vol"] = out["log_return"].rolling(20).std()
    out["bar_range_bps"] = bar_range

    # For Option C regime score
    out["signal_bar_range_bps"] = bar_range

    return out


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def evaluate_filter(trades_all, trades_kept, label):
    """Compare filtered vs unfiltered results."""
    n_all = len(trades_all)
    n_kept = len(trades_kept)
    n_skip = n_all - n_kept

    if n_all == 0 or n_kept == 0:
        return None

    # Unfiltered stats
    w_all = sum(1 for t in trades_all if t.net_profit_bps > 0)
    bps_all = sum(t.net_profit_bps for t in trades_all)
    gw_all = sum(t.net_profit_bps for t in trades_all if t.net_profit_bps > 0)
    gl_all = abs(sum(t.net_profit_bps for t in trades_all if t.net_profit_bps <= 0))
    pf_all = gw_all / gl_all if gl_all > 0 else 99

    # Filtered stats
    w_kept = sum(1 for t in trades_kept if t.net_profit_bps > 0)
    bps_kept = sum(t.net_profit_bps for t in trades_kept)
    gw_kept = sum(t.net_profit_bps for t in trades_kept if t.net_profit_bps > 0)
    gl_kept = abs(sum(t.net_profit_bps for t in trades_kept if t.net_profit_bps <= 0))
    pf_kept = gw_kept / gl_kept if gl_kept > 0 else 99

    # Skipped stats
    kept_ids = {id(t) for t in trades_kept}
    skipped = [t for t in trades_all if id(t) not in kept_ids]
    bps_skip = sum(t.net_profit_bps for t in skipped) if skipped else 0

    # Equity drawdown
    equity = np.cumsum([t.net_profit_bps for t in trades_kept])
    max_dd = (equity - np.maximum.accumulate(equity)).min() if len(equity) > 0 else 0

    return {
        "label": label,
        "trades": n_kept,
        "skipped": n_skip,
        "win%": round(w_kept / n_kept * 100, 1) if n_kept > 0 else 0,
        "total_bps": round(bps_kept, 0),
        "PF": round(pf_kept, 2),
        "avg_bps": round(bps_kept / n_kept, 1) if n_kept > 0 else 0,
        "max_dd": round(max_dd, 0),
        "skip_bps": round(bps_skip, 0),
    }


def main():
    config = load_config()

    # Load data
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    strategy = V12Strategy(config)
    df = strategy.compute_indicators(df)
    df = compute_all_features(df)

    # Run backtests
    trades_train = run_backtest(config, start="2020-01-01", end="2023-12-31")
    trades_oos = run_backtest(config, start="2024-01-01", end="2025-12-31")

    print_separator("BASELINE")
    for period, trades in [("TRAIN", trades_train), ("OOS", trades_oos)]:
        w = [t for t in trades if t.net_profit_bps > 0]
        bps = sum(t.net_profit_bps for t in trades)
        gw = sum(t.net_profit_bps for t in w)
        gl = abs(sum(t.net_profit_bps for t in trades if t.net_profit_bps <= 0))
        pf = gw / gl if gl > 0 else 99
        equity = np.cumsum([t.net_profit_bps for t in trades])
        max_dd = (equity - np.maximum.accumulate(equity)).min()
        print(f"{period}: {len(trades)}t | Win: {len(w)/len(trades)*100:.1f}% | "
              f"Total: {bps:+.0f} bps | PF: {pf:.2f} | DD: {max_dd:.0f}")

    # =========================================================================
    # Prepare all regime labels
    # =========================================================================
    # K-means features
    km_features = ["atr_bps", "ema_separation", "rsi_extremity", "price_vs_sma",
                    "avg_bar_range_20", "trend_consistency_10"]
    train_df = df["2020-01-01":"2023-12-31"].dropna(subset=km_features + ["log_return", "rolling_vol", "bar_range_bps"]).copy()
    oos_df = df["2024-01-01":"2025-12-31"].dropna(subset=km_features + ["log_return", "rolling_vol", "bar_range_bps"]).copy()

    # K-Means models
    scaler = StandardScaler()
    X_train_km = scaler.fit_transform(train_df[km_features])
    X_oos_km = scaler.transform(oos_df[km_features])

    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        train_df[f"km{k}"] = km.fit_predict(X_train_km)
        oos_df[f"km{k}"] = km.predict(X_oos_km)

    # HMM models
    hmm_features = ["log_return", "rolling_vol", "bar_range_bps"]
    X_train_hmm = train_df[hmm_features].values
    X_oos_hmm = oos_df[hmm_features].values

    for n in [2, 3, 4]:
        model = GaussianHMM(n_components=n, covariance_type="full", n_iter=200, random_state=42)
        model.fit(X_train_hmm)
        train_df[f"hmm{n}"] = model.predict(X_train_hmm)
        oos_df[f"hmm{n}"] = model.predict(X_oos_hmm)

    # =========================================================================
    # For each method: find worst regime on TRAIN, skip it on OOS
    # =========================================================================
    print_separator("REGIME FILTERING COMPARISON (OOS)")

    results = []

    # Baseline: no filtering
    results.append({
        "label": "BASELINE (no filter)",
        "trades": len(trades_oos),
        "skipped": 0,
        "win%": round(sum(1 for t in trades_oos if t.net_profit_bps > 0) / len(trades_oos) * 100, 1),
        "total_bps": round(sum(t.net_profit_bps for t in trades_oos), 0),
        "PF": round(
            sum(t.net_profit_bps for t in trades_oos if t.net_profit_bps > 0) /
            abs(sum(t.net_profit_bps for t in trades_oos if t.net_profit_bps <= 0)), 2),
        "avg_bps": round(sum(t.net_profit_bps for t in trades_oos) / len(trades_oos), 1),
        "max_dd": round((np.cumsum([t.net_profit_bps for t in trades_oos]) -
                         np.maximum.accumulate(np.cumsum([t.net_profit_bps for t in trades_oos]))).min(), 0),
        "skip_bps": 0,
    })

    # Helper: filter trades by regime
    def filter_trades(trades, regime_df, col, skip_values):
        kept = []
        for t in trades:
            sig_time = pd.Timestamp(t.signal_time)
            if sig_time in regime_df.index:
                regime = regime_df.loc[sig_time, col]
                if regime not in skip_values:
                    kept.append(t)
            else:
                kept.append(t)  # keep if can't determine regime
        return kept

    # For each clustering method
    for k in [2, 3, 4]:
        col = f"km{k}"
        # Find worst regime on TRAIN
        train_regime_perf = {}
        for t in trades_train:
            sig_time = pd.Timestamp(t.signal_time)
            if sig_time in train_df.index:
                r = train_df.loc[sig_time, col]
                if r not in train_regime_perf:
                    train_regime_perf[r] = []
                train_regime_perf[r].append(t.net_profit_bps)

        # Find regime with worst avg bps on TRAIN
        worst_regime = min(train_regime_perf, key=lambda r: np.mean(train_regime_perf[r]))
        worst_avg = np.mean(train_regime_perf[worst_regime])

        # Skip worst regime on OOS
        kept = filter_trades(trades_oos, oos_df, col, {worst_regime})
        r = evaluate_filter(trades_oos, kept, f"KMeans K={k} (skip R{worst_regime}, TRAIN avg={worst_avg:+.1f})")
        if r:
            results.append(r)

    # For each HMM model
    for n in [2, 3, 4]:
        col = f"hmm{n}"
        train_regime_perf = {}
        for t in trades_train:
            sig_time = pd.Timestamp(t.signal_time)
            if sig_time in train_df.index:
                r = train_df.loc[sig_time, col]
                if r not in train_regime_perf:
                    train_regime_perf[r] = []
                train_regime_perf[r].append(t.net_profit_bps)

        worst_regime = min(train_regime_perf, key=lambda r: np.mean(train_regime_perf[r]))
        worst_avg = np.mean(train_regime_perf[worst_regime])

        kept = filter_trades(trades_oos, oos_df, col, {worst_regime})
        r = evaluate_filter(trades_oos, kept, f"HMM N={n} (skip S{worst_regime}, TRAIN avg={worst_avg:+.1f})")
        if r:
            results.append(r)

    # Option C: Simple regime score
    def get_regime_score(sig_time, regime_df):
        if sig_time not in regime_df.index:
            return 4  # default to full score
        bar = regime_df.loc[sig_time]
        score = 0
        if bar["atr_percentile"] >= 50:
            score += 1
        if bar["ema_separation"] >= 1.0:
            score += 1
        if bar["signal_bar_range_bps"] >= 20:
            score += 1
        hour = sig_time.hour
        if not (0 <= hour <= 4):
            score += 1
        return score

    for min_score in [1, 2, 3]:
        kept = []
        for t in trades_oos:
            sig_time = pd.Timestamp(t.signal_time)
            score = get_regime_score(sig_time, oos_df)
            if score >= min_score:
                kept.append(t)
        r = evaluate_filter(trades_oos, kept, f"Score >= {min_score} (Option C)")
        if r:
            results.append(r)

    # Simple EMA sep threshold
    for thresh in [0.5, 1.0, 1.5]:
        kept = []
        for t in trades_oos:
            sig_time = pd.Timestamp(t.signal_time)
            if sig_time in oos_df.index:
                ema = oos_df.loc[sig_time, "ema_separation"]
                if ema >= thresh:
                    kept.append(t)
                else:
                    pass  # skip
            else:
                kept.append(t)
        r = evaluate_filter(trades_oos, kept, f"EMA sep >= {thresh}%")
        if r:
            results.append(r)

    # ATR percentile threshold
    for thresh in [25, 40, 50]:
        kept = []
        for t in trades_oos:
            sig_time = pd.Timestamp(t.signal_time)
            if sig_time in oos_df.index:
                atr = oos_df.loc[sig_time, "atr_percentile"]
                if atr >= thresh:
                    kept.append(t)
            else:
                kept.append(t)
        r = evaluate_filter(trades_oos, kept, f"ATR pctl >= {thresh}")
        if r:
            results.append(r)

    # Print comparison table
    print(f"\n{'Method':<55} {'Trades':>6} {'Skip':>5} {'Win%':>6} {'Total':>7} {'PF':>6} "
          f"{'Avg':>6} {'DD':>6} {'SkipBps':>8}")
    print("-" * 115)
    for r in results:
        print(f"{r['label']:<55} {r['trades']:>5}  {r['skipped']:>4}  {r['win%']:>5.1f}  "
              f"{r['total_bps']:>+6.0f}  {r['PF']:>5.2f}  {r['avg_bps']:>+5.1f}  "
              f"{r['max_dd']:>+5.0f}  {r['skip_bps']:>+7.0f}")

    # =========================================================================
    # SAME ON TRAIN (to check consistency)
    # =========================================================================
    print_separator("REGIME FILTERING COMPARISON (TRAIN)")

    results_train = []
    results_train.append({
        "label": "BASELINE",
        "trades": len(trades_train),
        "skipped": 0,
        "win%": round(sum(1 for t in trades_train if t.net_profit_bps > 0) / len(trades_train) * 100, 1),
        "total_bps": round(sum(t.net_profit_bps for t in trades_train), 0),
        "PF": round(
            sum(t.net_profit_bps for t in trades_train if t.net_profit_bps > 0) /
            abs(sum(t.net_profit_bps for t in trades_train if t.net_profit_bps <= 0)), 2),
        "avg_bps": round(sum(t.net_profit_bps for t in trades_train) / len(trades_train), 1),
        "max_dd": round((np.cumsum([t.net_profit_bps for t in trades_train]) -
                         np.maximum.accumulate(np.cumsum([t.net_profit_bps for t in trades_train]))).min(), 0),
        "skip_bps": 0,
    })

    for min_score in [1, 2, 3]:
        kept = []
        for t in trades_train:
            sig_time = pd.Timestamp(t.signal_time)
            score = get_regime_score(sig_time, train_df)
            if score >= min_score:
                kept.append(t)
        r = evaluate_filter(trades_train, kept, f"Score >= {min_score}")
        if r:
            results_train.append(r)

    for thresh in [0.5, 1.0, 1.5]:
        kept = []
        for t in trades_train:
            sig_time = pd.Timestamp(t.signal_time)
            if sig_time in train_df.index:
                ema = train_df.loc[sig_time, "ema_separation"]
                if ema >= thresh:
                    kept.append(t)
            else:
                kept.append(t)
        r = evaluate_filter(trades_train, kept, f"EMA sep >= {thresh}%")
        if r:
            results_train.append(r)

    print(f"\n{'Method':<55} {'Trades':>6} {'Skip':>5} {'Win%':>6} {'Total':>7} {'PF':>6} "
          f"{'Avg':>6} {'DD':>6} {'SkipBps':>8}")
    print("-" * 115)
    for r in results_train:
        print(f"{r['label']:<55} {r['trades']:>5}  {r['skipped']:>4}  {r['win%']:>5.1f}  "
              f"{r['total_bps']:>+6.0f}  {r['PF']:>5.2f}  {r['avg_bps']:>+5.1f}  "
              f"{r['max_dd']:>+5.0f}  {r['skip_bps']:>+7.0f}")

    print_separator("DONE")


if __name__ == "__main__":
    main()
