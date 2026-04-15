"""Layer 2 -- Option B: Hidden Markov Model Regime Detection

Let HMM discover hidden states from price data.

Approach:
  1. Fit HMM on returns + volatility (TRAIN)
  2. Decode states on TRAIN and OOS
  3. Map V1.3.2 trades to HMM states
  4. Check if performance differs per state
  5. Compare with Option A clustering

Run: PYTHONPATH=src python experiments/layer2_regime/option_b_hmm.py
"""

import sys
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from engine.config.loader import load_config
from engine.backtest import run_backtest
from engine.strategy import V12Strategy

logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).parent
DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")


def compute_hmm_features(df):
    """Compute features for HMM: returns, volatility, volume change."""
    out = df.copy()

    # Log returns (bps)
    out["log_return"] = np.log(out["close"] / out["close"].shift(1)) * 10000

    # Rolling volatility (std of returns, 20 bars)
    out["rolling_vol"] = out["log_return"].rolling(20).std()

    # Bar range (bps)
    out["bar_range"] = (out["high"] - out["low"]) / out["close"] * 10000

    # Volume change ratio
    out["vol_change"] = out["volume"] / out["volume"].rolling(20).mean()

    return out


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def map_trades_to_state(trades, state_df):
    """Map each trade to its HMM state at signal time."""
    rows = []
    for t in trades:
        sig_time = pd.Timestamp(t.signal_time)
        if sig_time in state_df.index:
            state = state_df.loc[sig_time, "hmm_state"]
            rows.append({
                "signal_time": sig_time,
                "direction": t.direction,
                "signal_type": t.signal_type,
                "net_profit_bps": t.net_profit_bps,
                "exit_reason": t.exit_reason,
                "hmm_state": state,
            })
    return pd.DataFrame(rows)


def analyze_states(tf, n_states, label):
    """Print performance by HMM state."""
    print(f"\nV1.3.2 Performance by HMM State ({label}, N={n_states}):")
    print(f"{'State':<8} {'Trades':>6} {'Win%':>6} {'Avg bps':>8} {'Total':>8} {'PF':>6}")
    for s in range(n_states):
        subset = tf[tf["hmm_state"] == s]
        if len(subset) == 0:
            print(f"S{s:<7}     0   ---      ---      ---    ---")
            continue
        w = subset[subset["net_profit_bps"] > 0]
        l = subset[~(subset["net_profit_bps"] > 0)]
        gw = w["net_profit_bps"].sum() if len(w) > 0 else 0
        gl = abs(l["net_profit_bps"].sum()) if len(l) > 0 else 1
        print(f"S{s:<7} {len(subset):>5}  {len(w)/len(subset)*100:>5.1f}  "
              f"{subset['net_profit_bps'].mean():>+7.1f}  "
              f"{subset['net_profit_bps'].sum():>+7.0f}  {gw/gl:>5.2f}")


def main():
    config = load_config()

    # Load data and compute indicators
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    strategy = V12Strategy(config)
    df = strategy.compute_indicators(df)
    df = compute_hmm_features(df)

    # HMM features
    hmm_features = ["log_return", "rolling_vol", "bar_range"]

    # Split
    train = df["2020-01-01":"2023-12-31"].dropna(subset=hmm_features).copy()
    oos = df["2024-01-01":"2025-12-31"].dropna(subset=hmm_features).copy()

    print_separator("DATA SUMMARY")
    print(f"TRAIN bars: {len(train)}")
    print(f"OOS bars:   {len(oos)}")

    X_train = train[hmm_features].values
    X_oos = oos[hmm_features].values

    # Run backtest
    trades_train = run_backtest(config, start="2020-01-01", end="2023-12-31")
    trades_oos = run_backtest(config, start="2024-01-01", end="2025-12-31")

    # =========================================================================
    # Fit HMM with different N states
    # =========================================================================
    for n_states in [2, 3, 4, 5]:
        print_separator(f"HMM N={n_states} STATES")

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X_train)

        # Decode states
        train_states = model.predict(X_train)
        oos_states = model.predict(X_oos)

        train["hmm_state"] = train_states
        oos["hmm_state"] = oos_states

        # State characteristics
        print("\nState Characteristics (TRAIN):")
        print(f"{'State':<8} {'Bars':>8} {'%':>6} {'Avg Return':>12} {'Avg Vol':>10} {'Avg Range':>10}")
        for s in range(n_states):
            mask = train_states == s
            n_bars = mask.sum()
            avg_ret = train.loc[train["hmm_state"] == s, "log_return"].mean()
            avg_vol = train.loc[train["hmm_state"] == s, "rolling_vol"].mean()
            avg_rng = train.loc[train["hmm_state"] == s, "bar_range"].mean()
            print(f"S{s:<7} {n_bars:>7}  {n_bars/len(train)*100:>5.1f}  {avg_ret:>+11.2f}  {avg_vol:>9.2f}  {avg_rng:>9.2f}")

        # Map trades to states
        tf_train = map_trades_to_state(trades_train, train)
        tf_oos = map_trades_to_state(trades_oos, oos)

        # Performance by state
        analyze_states(tf_train, n_states, "TRAIN")
        analyze_states(tf_oos, n_states, "OOS")

        # State transition matrix
        print(f"\nTransition Matrix (TRAIN):")
        trans = model.transmat_
        print(f"{'From/To':<8}", end="")
        for s in range(n_states):
            print(f"{'S'+str(s):>8}", end="")
        print()
        for i in range(n_states):
            print(f"S{i:<7}", end="")
            for j in range(n_states):
                print(f"{trans[i][j]:>8.3f}", end="")
            print()

        # Consistency check
        print(f"\nConsistency Check (TRAIN vs OOS):")
        for s in range(n_states):
            train_sub = tf_train[tf_train["hmm_state"] == s]
            oos_sub = tf_oos[tf_oos["hmm_state"] == s]
            if len(train_sub) > 0 and len(oos_sub) > 0:
                t_avg = train_sub["net_profit_bps"].mean()
                o_avg = oos_sub["net_profit_bps"].mean()
                consistent = (t_avg > 0 and o_avg > 0) or (t_avg <= 0 and o_avg <= 0)
                print(f"  S{s}: TRAIN avg={t_avg:+.1f}, OOS avg={o_avg:+.1f} -> "
                      f"{'CONSISTENT' if consistent else 'INCONSISTENT'}")
            elif len(train_sub) > 0:
                print(f"  S{s}: TRAIN avg={train_sub['net_profit_bps'].mean():+.1f}, OOS: no trades")
            elif len(oos_sub) > 0:
                print(f"  S{s}: TRAIN: no trades, OOS avg={oos_sub['net_profit_bps'].mean():+.1f}")

        # By signal type within best/worst states
        if len(tf_train) > 0:
            state_avgs = tf_train.groupby("hmm_state")["net_profit_bps"].mean()
            if len(state_avgs) >= 2:
                best_state = state_avgs.idxmax()
                worst_state = state_avgs.idxmin()
                print(f"\nBest State S{best_state} by Signal Type (TRAIN):")
                best_trades = tf_train[tf_train["hmm_state"] == best_state]
                for sig in sorted(best_trades["signal_type"].unique()):
                    sig_sub = best_trades[best_trades["signal_type"] == sig]
                    print(f"  {sig}: {len(sig_sub)}t, avg={sig_sub['net_profit_bps'].mean():+.1f}")

                print(f"\nWorst State S{worst_state} by Signal Type (TRAIN):")
                worst_trades = tf_train[tf_train["hmm_state"] == worst_state]
                for sig in sorted(worst_trades["signal_type"].unique()):
                    sig_sub = worst_trades[worst_trades["signal_type"] == sig]
                    print(f"  {sig}: {len(sig_sub)}t, avg={sig_sub['net_profit_bps'].mean():+.1f}")

    print_separator("DONE")


if __name__ == "__main__":
    main()
