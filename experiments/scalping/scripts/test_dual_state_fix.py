"""
Quick test script to verify the dual-state direction fix.
Uses minimal parameters and 1 worker to run alongside main grid search.

Tests the fix where BOTH long and short are evaluated independently.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trade_system.similarity import SimilarityEngine
from trade_system.decision import DecisionEngine
from trade_system.backtest import TradeSimulator

# Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTCOME_PATH = DATA_DIR / "outcomes" / "BTCUSDT_1m_outcomes.parquet"
REGIME_PATH = DATA_DIR / "regimes" / "BTCUSDT_1m_regimes.parquet"
OHLCV_PATH = DATA_DIR / "ohlcv" / "BTCUSDT_1m_ohlcv.parquet"

# Small parameter grid for quick test
PARAM_GRID = {
    "horizon": [3],
    "normalization_window": [180],
    "min_expectancy": [0.0],
    "max_distance": [3.0, 4.0],
    "k": [100, 150],
    "min_mfe": [0.001, 0.002],
    "min_bars_between": [0, 3],
    "signal_interval": [3],
}

# Test config
SAMPLE_SIZE = 100_000  # Smaller sample
TRAIN_RATIO = 0.7


def generate_param_combinations(grid):
    """Generate all combinations from parameter grid."""
    from itertools import product
    keys = list(grid.keys())
    values = list(grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def run_single_test(params, outcome_df, regime_df, ohlcv_df, train_ratio):
    """Run backtest for single parameter combination."""
    horizon = params["horizon"]
    normalization_window = params["normalization_window"]
    min_expectancy = params["min_expectancy"]
    max_distance = params["max_distance"]
    k = params["k"]
    min_mfe = params["min_mfe"]
    min_bars_between = params["min_bars_between"]
    signal_interval = params["signal_interval"]

    # Split data
    split_idx = int(len(outcome_df) * train_ratio)
    train_outcome = outcome_df.iloc[:split_idx]
    test_outcome = outcome_df.iloc[split_idx:]
    train_regime = regime_df.loc[train_outcome.index]

    # Initialize components
    similarity_engine = SimilarityEngine(
        outcome_df=train_outcome,
        regime_df=train_regime,
        k=k,
        backend="bruteforce",
        normalization_window=normalization_window,
    )

    decision_engine = DecisionEngine(
        capital=100.0,
        risk_per_trade=0.02,
        min_expectancy=min_expectancy,
        max_distance=max_distance,
        blocked_regimes=["choppy"],
        min_mfe=min_mfe,
        max_leverage=3.0,
        stop_floor=0.001,
        use_stop_loss=False,
    )

    simulator = TradeSimulator(
        commission_pct=0.0004,
        max_bars_in_trade=horizon,
    )

    # Run backtest
    trades = []
    active_trade = None
    bars_since_trade = min_bars_between
    long_count = 0
    short_count = 0

    test_timestamps = test_outcome.index[::signal_interval]

    for timestamp in test_timestamps:
        if timestamp not in ohlcv_df.index:
            continue
        bar = ohlcv_df.loc[timestamp]

        # Update active trade
        if active_trade is not None:
            active_trade = simulator.update_trade(active_trade, bar, timestamp)
            if active_trade.exit_time is not None:
                active_trade = None
                bars_since_trade = 0
            continue

        bars_since_trade += 1
        if bars_since_trade < min_bars_between:
            continue

        # Get state and regime
        if timestamp not in test_outcome.index or timestamp not in regime_df.index:
            continue

        state = test_outcome.loc[timestamp]
        regime = regime_df.loc[timestamp, "regime"]

        # Query similar states
        sim_result = similarity_engine.query(
            current_state=state,
            regime=regime,
            horizon=horizon,
            max_timestamp=timestamp,
        )

        # Make decision
        decision = decision_engine.decide(sim_result, regime)

        if decision["action"] == "TRADE":
            direction = decision["direction"]
            if direction == "LONG":
                long_count += 1
            else:
                short_count += 1

            active_trade = simulator.open_trade(
                decision=decision,
                signal_time=timestamp,
                entry_price=bar["close"],
                regime=regime,
            )
            trades.append(active_trade)

    # Calculate results
    total_pnl = sum(t.pnl for t in trades if t.pnl is not None)

    return {
        "params": params,
        "total_pnl": total_pnl,
        "total_trades": len(trades),
        "long_trades": long_count,
        "short_trades": short_count,
        "long_ratio": long_count / len(trades) if trades else 0,
    }


def main():
    print("=" * 60)
    print("  DUAL-STATE FIX TEST (1 worker, minimal params)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    outcome_df = pd.read_parquet(OUTCOME_PATH)
    outcome_df.index = pd.to_datetime(outcome_df.index)
    if len(outcome_df) > SAMPLE_SIZE:
        outcome_df = outcome_df.iloc[-SAMPLE_SIZE:]

    regime_df = pd.read_parquet(REGIME_PATH)
    regime_df.index = pd.to_datetime(regime_df.index)
    regime_df = regime_df.loc[outcome_df.index[0]:outcome_df.index[-1]]

    ohlcv_df = pd.read_parquet(OHLCV_PATH)
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)

    print(f"  Outcome rows: {len(outcome_df):,}")
    print(f"  Test period: {outcome_df.index[0]} to {outcome_df.index[-1]}")

    # Generate combinations
    combinations = list(generate_param_combinations(PARAM_GRID))
    print(f"\nTesting {len(combinations)} parameter combinations...")

    # Run tests sequentially (1 worker)
    results = []
    start_time = time.time()

    for i, params in enumerate(combinations, 1):
        result = run_single_test(params, outcome_df, regime_df, ohlcv_df, TRAIN_RATIO)
        results.append(result)

        # Print progress
        pnl = result["total_pnl"]
        trades = result["total_trades"]
        longs = result["long_trades"]
        shorts = result["short_trades"]

        status = "+" if pnl > 0 else "-"
        print(f"[{i:2}/{len(combinations)}] {status} PnL=${pnl:+.2f} | Trades={trades} (L={longs}, S={shorts}) | k={params['k']} mfe={params['min_mfe']}")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    profitable = [r for r in results if r["total_pnl"] > 0]
    print(f"Profitable: {len(profitable)}/{len(results)}")

    total_longs = sum(r["long_trades"] for r in results)
    total_shorts = sum(r["short_trades"] for r in results)
    print(f"Total trades: {total_longs + total_shorts} (LONG={total_longs}, SHORT={total_shorts})")

    if total_shorts > 0:
        print(f"\n*** SHORT TRADES DETECTED! Fix is working! ***")
    else:
        print(f"\nNo short trades taken (market may be strongly bullish)")

    print(f"\nElapsed: {elapsed:.1f}s")

    # Save results
    output_path = PROJECT_ROOT / "logs" / "dual_state_test_results.csv"
    df = pd.DataFrame([
        {**r["params"], "total_pnl": r["total_pnl"], "trades": r["total_trades"],
         "long_trades": r["long_trades"], "short_trades": r["short_trades"]}
        for r in results
    ])
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
