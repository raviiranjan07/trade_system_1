#!/usr/bin/env python3
"""
Scalping Grid Search - Local Version with FAISS-CPU

Run all 648 combinations locally with progress tracking.
Estimated time: ~5-10 hours with FAISS-CPU

Usage:
    python -m experiments.scalping.run_scalping_grid_search
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trade_system.config import get_config
from trade_system.similarity.similarity_engine import SimilarityEngine
from trade_system.decision.decision_engine import DecisionEngine
from trade_system.backtest.trade_simulator import TradeSimulator
from trade_system.backtest.metrics import calculate_metrics


# =============================================================================
# SCALPING PARAMETER GRID
# =============================================================================
PARAM_GRID = {
    "horizon": [1, 2],
    "normalization_window": [200, 300, 500],
    "min_expectancy": [0.001, 0.002, 0.003, 0.005],
    "max_distance": [1.5, 2.0, 3.0],
    "k": [100, 150, 200],
    "blocked_regimes": [
        [],
        ["HIGH_VOL"],
        ["HIGH_VOL", "RANGE_LOW_VOL"]
    ]
}


# Global data for worker processes
_worker_data = {}

def init_worker(outcome_path, regime_path, ohlcv_path, train_ratio):
    """Initialize worker process with data."""
    global _worker_data

    outcome_df = pd.read_parquet(outcome_path)
    outcome_df.index = pd.to_datetime(outcome_df.index)

    regime_df = pd.read_parquet(regime_path)
    regime_df.index = pd.to_datetime(regime_df.index)

    ohlcv_df = pd.read_parquet(ohlcv_path)
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)

    split_idx = int(len(outcome_df) * train_ratio)

    _worker_data = {
        'outcome_df': outcome_df,
        'regime_df': regime_df,
        'ohlcv_df': ohlcv_df,
        'train_outcomes': outcome_df.iloc[:split_idx],
        'test_outcomes': outcome_df.iloc[split_idx:],
        'train_ratio': train_ratio
    }


def run_single_test(params):
    """Worker function for parallel execution."""
    global _worker_data

    horizon, norm_window, min_exp, max_dist, k, blocked = params

    # Import here to avoid issues with multiprocessing
    from trade_system.similarity.similarity_engine import SimilarityEngine
    from trade_system.decision.decision_engine import DecisionEngine
    from trade_system.backtest.trade_simulator import TradeSimulator
    from trade_system.backtest.metrics import calculate_metrics

    outcome_df = _worker_data['outcome_df']
    regime_df = _worker_data['regime_df']
    ohlcv_df = _worker_data['ohlcv_df']
    train_outcomes = _worker_data['train_outcomes']
    test_outcomes = _worker_data['test_outcomes']

    # Build similarity engine for this k
    similarity_engine = SimilarityEngine(
        outcome_df=train_outcomes,
        regime_df=regime_df,
        k=k,
        backend="faiss",
        faiss_nlist=100,
        faiss_nprobe=10,
        use_gpu=True  # Use GPU for FAISS (T4 on Colab)
    )

    # Run backtest
    sample_interval = 1

    simulator = TradeSimulator(
        slippage_pct=0.0005,
        commission_pct=0.0004,
        max_bars_in_trade=0,
        trailing_stop_pct=0.0,
        trailing_stop_activation_pct=0.0
    )

    decision_engine = DecisionEngine(
        capital=100,
        risk_per_trade=0.005,
        min_expectancy=min_exp,
        max_distance=max_dist,
        blocked_regimes=blocked
    )

    trades = []
    active_trade = None
    bar_counter = 0

    for timestamp, state_row in test_outcomes.iterrows():
        bar_counter += 1

        if timestamp not in ohlcv_df.index:
            continue
        bar = ohlcv_df.loc[timestamp]

        if active_trade is not None:
            active_trade = simulator.update_trade(active_trade, bar, timestamp)
            if active_trade.exit_time is not None:
                trades.append(active_trade)
                active_trade = None

        if active_trade is None and bar_counter % sample_interval == 0:
            if timestamp not in regime_df.index:
                continue
            regime = regime_df.loc[timestamp, "regime"]

            sim_result = similarity_engine.query(
                current_state=state_row,
                regime=regime,
                horizon=horizon,
                max_timestamp=timestamp
            )

            decision = decision_engine.decide(sim_result, regime)

            if decision["action"] == "TRADE":
                future_bars = ohlcv_df.loc[timestamp:].iloc[1:]
                if len(future_bars) > 0:
                    next_bar = future_bars.iloc[0]
                    active_trade = simulator.open_trade(
                        decision=decision,
                        signal_time=timestamp,
                        entry_price=next_bar["open"],
                        regime=regime
                    )
                    active_trade.entry_time = future_bars.index[0]

    if active_trade is not None:
        last_bar = ohlcv_df.iloc[-1]
        active_trade = simulator.force_exit(active_trade, last_bar["close"], ohlcv_df.index[-1])
        trades.append(active_trade)

    # Calculate metrics
    split_idx = int(len(outcome_df) * _worker_data['train_ratio'])

    if trades:
        metrics = calculate_metrics(
            trades=trades,
            capital=100,
            train_start=outcome_df.index[0],
            train_end=outcome_df.index[split_idx - 1],
            test_start=test_outcomes.index[0],
            test_end=test_outcomes.index[-1],
            pair="BTCUSDT"
        )
        return {
            "horizon": horizon,
            "norm_window": norm_window,
            "min_expectancy": min_exp,
            "max_distance": max_dist,
            "k": k,
            "blocked_regimes": ",".join(blocked) if blocked else "NONE",
            "total_pnl": metrics.total_pnl,
            "total_pnl_pct": metrics.total_pnl_pct * 100,
            "win_rate": metrics.win_rate * 100,
            "total_trades": metrics.total_trades,
            "profit_factor": metrics.profit_factor if metrics.profit_factor != float('inf') else 99.99,
            "max_drawdown_pct": metrics.max_drawdown_pct * 100,
            "sharpe": metrics.sharpe_ratio or 0,
            "expectancy": metrics.expectancy,
        }

    return {
        "horizon": horizon,
        "norm_window": norm_window,
        "min_expectancy": min_exp,
        "max_distance": max_dist,
        "k": k,
        "blocked_regimes": ",".join(blocked) if blocked else "NONE",
        "total_pnl": 0,
        "total_pnl_pct": 0,
        "win_rate": 0,
        "total_trades": 0,
        "profit_factor": 0,
        "max_drawdown_pct": 0,
        "sharpe": 0,
        "expectancy": 0,
    }


def run_single_backtest(params, outcome_df, regime_df, ohlcv_df, similarity_engine, config):
    """Run a single backtest with given parameters."""
    horizon, norm_window, min_exp, max_dist, k, blocked = params

    # For scalping, check every bar (most accurate)
    sample_interval = 1

    # Train/test split
    train_ratio = config.get("backtest.train_ratio", 0.70)
    split_idx = int(len(outcome_df) * train_ratio)
    test_outcomes = outcome_df.iloc[split_idx:]

    # Initialize engines
    simulator = TradeSimulator(
        slippage_pct=config.get("backtest.slippage_pct", 0.0005),
        commission_pct=config.get("backtest.commission_pct", 0.0004),
        max_bars_in_trade=0,
        trailing_stop_pct=0.0,
        trailing_stop_activation_pct=0.0
    )

    decision_engine = DecisionEngine(
        capital=config.get("decision.capital", 100),
        risk_per_trade=config.get("decision.risk_per_trade", 0.005),
        min_expectancy=min_exp,
        max_distance=max_dist,
        blocked_regimes=blocked
    )

    trades = []
    active_trade = None
    bar_counter = 0
    total_bars = len(test_outcomes)
    last_progress = 0

    for timestamp, state_row in test_outcomes.iterrows():
        bar_counter += 1

        # Show progress every 10%
        progress_pct = int(bar_counter / total_bars * 100)
        if progress_pct >= last_progress + 10:
            last_progress = progress_pct
            print(f"{progress_pct}%", end=" ", flush=True)

        if timestamp not in ohlcv_df.index:
            continue
        bar = ohlcv_df.loc[timestamp]

        # Update active trade
        if active_trade is not None:
            active_trade = simulator.update_trade(active_trade, bar, timestamp)
            if active_trade.exit_time is not None:
                trades.append(active_trade)
                active_trade = None

        # Check for new signal
        if active_trade is None and bar_counter % sample_interval == 0:
            if timestamp not in regime_df.index:
                continue
            regime = regime_df.loc[timestamp, "regime"]

            sim_result = similarity_engine.query(
                current_state=state_row,
                regime=regime,
                horizon=horizon,
                max_timestamp=timestamp
            )

            decision = decision_engine.decide(sim_result, regime)

            if decision["action"] == "TRADE":
                future_bars = ohlcv_df.loc[timestamp:].iloc[1:]
                if len(future_bars) > 0:
                    next_bar = future_bars.iloc[0]
                    active_trade = simulator.open_trade(
                        decision=decision,
                        signal_time=timestamp,
                        entry_price=next_bar["open"],
                        regime=regime
                    )
                    active_trade.entry_time = future_bars.index[0]

    # Close any remaining trade
    if active_trade is not None:
        last_bar = ohlcv_df.iloc[-1]
        active_trade = simulator.force_exit(active_trade, last_bar["close"], ohlcv_df.index[-1])
        trades.append(active_trade)

    # Calculate metrics
    if trades:
        metrics = calculate_metrics(
            trades=trades,
            capital=config.get("decision.capital", 100),
            train_start=outcome_df.index[0],
            train_end=outcome_df.index[split_idx - 1],
            test_start=test_outcomes.index[0],
            test_end=test_outcomes.index[-1],
            pair=config.get("data.pair", "BTCUSDT")
        )
        return {
            "horizon": horizon,
            "norm_window": norm_window,
            "min_expectancy": min_exp,
            "max_distance": max_dist,
            "k": k,
            "blocked_regimes": ",".join(blocked) if blocked else "NONE",
            "total_pnl": metrics.total_pnl,
            "total_pnl_pct": metrics.total_pnl_pct * 100,
            "win_rate": metrics.win_rate * 100,
            "total_trades": metrics.total_trades,
            "profit_factor": metrics.profit_factor if metrics.profit_factor != float('inf') else 99.99,
            "max_drawdown_pct": metrics.max_drawdown_pct * 100,
            "sharpe": metrics.sharpe_ratio or 0,
            "expectancy": metrics.expectancy,
        }

    return {
        "horizon": horizon,
        "norm_window": norm_window,
        "min_expectancy": min_exp,
        "max_distance": max_dist,
        "k": k,
        "blocked_regimes": ",".join(blocked) if blocked else "NONE",
        "total_pnl": 0,
        "total_pnl_pct": 0,
        "win_rate": 0,
        "total_trades": 0,
        "profit_factor": 0,
        "max_drawdown_pct": 0,
        "sharpe": 0,
        "expectancy": 0,
    }


def main():
    print()
    print("=" * 70)
    print("   SCALPING GRID SEARCH - 648 COMBINATIONS (PARALLEL)")
    print("   Backend: FAISS-CPU | Mode: Multi-process")
    print("=" * 70)

    # Use 4 workers (half of 8 logical cores) - safe for 16GB RAM
    # Close Chrome/VS Code if RAM gets tight
    n_workers = 2
    print(f"\n✓ Using {n_workers} CPU cores for parallel processing")

    # Load config
    config = get_config()
    data_dir = Path(config.get("paths.data_dir", "data"))
    pair = config.get("data.pair", "BTCUSDT")
    train_ratio = config.get("backtest.train_ratio", 0.70)

    # Get file paths for workers
    outcome_path = sorted((data_dir / "outcomes").glob(f"{pair}_*.parquet"))[-1]
    regime_path = sorted((data_dir / "regimes").glob(f"{pair}_*.parquet"))[-1]
    ohlcv_path = sorted((data_dir / "ohlcv").glob(f"{pair}_*.parquet"))[-1]

    print(f"\nData files:")
    print(f"  Outcomes: {outcome_path}")
    print(f"  Regimes: {regime_path}")
    print(f"  OHLCV: {ohlcv_path}")

    # Generate all combinations
    all_combinations = list(product(
        PARAM_GRID["horizon"],
        PARAM_GRID["normalization_window"],
        PARAM_GRID["min_expectancy"],
        PARAM_GRID["max_distance"],
        PARAM_GRID["k"],
        PARAM_GRID["blocked_regimes"]
    ))

    print(f"\nTotal combinations: {len(all_combinations)}")
    print(f"With {n_workers} workers: ~{len(all_combinations) / n_workers:.0f} tests per worker")

    # Run grid search in parallel
    print("\n" + "=" * 70)
    print(f"RUNNING GRID SEARCH IN PARALLEL ({n_workers} workers)...")
    print("=" * 70 + "\n")

    results = []
    grid_start = time.time()
    completed = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(str(outcome_path), str(regime_path), str(ohlcv_path), train_ratio)
    ) as executor:
        # Submit all jobs
        future_to_params = {executor.submit(run_single_test, params): params for params in all_combinations}

        # Collect results as they complete
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                # Print progress
                elapsed = time.time() - grid_start
                avg_time = elapsed / completed
                eta = avg_time * (len(all_combinations) - completed)

                pnl = result['total_pnl']
                trades = result['total_trades']
                h = result['horizon']
                k = result['k']

                print(f"[{completed}/{len(all_combinations)}] H={h} k={k} => ${pnl:+.2f} ({trades} trades) | "
                      f"Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {timedelta(seconds=int(eta))}")

            except Exception as e:
                print(f"Error with {params}: {e}")
                completed += 1

    total_time = time.time() - grid_start
    print(f"\nCompleted in {timedelta(seconds=int(total_time))}")

    # Save results
    results_df = pd.DataFrame(results).sort_values("total_pnl", ascending=False)

    output_dir = PROJECT_ROOT / "experiments" / "scalping" / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"scalping_FULL_GRID_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

    # Show top results
    print("\n" + "=" * 70)
    print("TOP 20 RESULTS")
    print("=" * 70)

    for i, row in results_df.head(20).iterrows():
        blocked_str = row['blocked_regimes']
        print(f"${row['total_pnl']:+8.2f} | H={row['horizon']} w={row['norm_window']} "
              f"exp={row['min_expectancy']:.3f} dist={row['max_distance']} k={row['k']} "
              f"blk={blocked_str:20} | WR={row['win_rate']:.1f}% ({row['total_trades']} trades)")

    # Show best config
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    best = results_df.iloc[0]
    print(f"""
# Scalping Best Config
horizon: {best['horizon']}
normalization_window: {best['norm_window']}
min_expectancy: {best['min_expectancy']}
max_distance: {best['max_distance']}
k: {best['k']}
blocked_regimes: [{best['blocked_regimes']}]

# Results
total_pnl: ${best['total_pnl']:.2f} ({best['total_pnl_pct']:.2f}%)
win_rate: {best['win_rate']:.1f}%
total_trades: {best['total_trades']}
profit_factor: {best['profit_factor']:.2f}
max_drawdown: {best['max_drawdown_pct']:.2f}%
""")

    # Summary stats
    profitable = len(results_df[results_df['total_pnl'] > 0])
    print(f"Summary: {profitable}/{len(results_df)} combinations profitable ({profitable/len(results_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
