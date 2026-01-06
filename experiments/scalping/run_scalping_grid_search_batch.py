#!/usr/bin/env python3
"""
Scalping Grid Search - BATCH PARALLEL VERSION

Runs in batches of 20 combinations at a time.
Each batch runs in parallel, then memory is freed.
Results saved after EACH batch for analysis.

Usage:
    python -m experiments.scalping.run_scalping_grid_search_batch
"""

import sys
import time
import gc
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Progress log file path
PROGRESS_LOG = Path(__file__).parent.parent.parent / "logs" / "scalping_progress.log"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trade_system.config import get_config


# =============================================================================
# CONFIGURATION - 432 combinations
# =============================================================================
PARAM_GRID = {
    "horizon": [3,5],                          # 2
    "normalization_window": [300],             # 1 (fixed)
    "min_expectancy": [0.0],                   # 1 (fixed - min_mfe handles filtering)
    "max_distance": [2.0, 3.0],                # 2
    "k": [100, 150, 200],                      # 3
    "min_mfe": [0.002, 0.003, 0.004, 0.005],   # 4 (0.2%, 0.3%, 0.4%, 0.5%)
    "max_bars_in_trade": [0, 5, 10],           # 3 (0 = no timeout)
    "sample_interval": [1, 5, 10],             # 3
    "blocked_regimes": [[]],                   # 1 (no blocking - 5m showed no blocking is best)
}
# Total: 2 * 1 * 1 * 2 * 3 * 4 * 3 * 3 * 1 = 432 combinations

# Batch settings
BATCH_SIZE = 108  # 4 batches of 108 = 432 total
SAMPLE_SIZE = 500_000  # Use 500K rows (~1 year) for faster testing


# Global data for worker processes
_worker_data = {}


def init_worker(outcome_path, regime_path, ohlcv_path, train_ratio, sample_size):
    """Initialize worker process with sampled data."""
    global _worker_data

    outcome_df = pd.read_parquet(outcome_path)
    outcome_df.index = pd.to_datetime(outcome_df.index)

    # Sample last N rows
    if len(outcome_df) > sample_size:
        outcome_df = outcome_df.iloc[-sample_size:]

    regime_df = pd.read_parquet(regime_path)
    regime_df.index = pd.to_datetime(regime_df.index)
    regime_df = regime_df.loc[regime_df.index.isin(outcome_df.index)]

    ohlcv_df = pd.read_parquet(ohlcv_path)
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
    ohlcv_df = ohlcv_df.loc[ohlcv_df.index.isin(outcome_df.index)]

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

    horizon, norm_window, min_exp, max_dist, k, min_mfe, max_bars, sample_interval, blocked = params

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

    # Build similarity engine
    similarity_engine = SimilarityEngine(
        outcome_df=train_outcomes,
        regime_df=regime_df,
        k=k,
        backend="faiss",
        faiss_nlist=50,
        faiss_nprobe=5,
        use_gpu=False  # CPU for local, set True for Colab GPU
    )

    simulator = TradeSimulator(
        slippage_pct=0.0005,
        commission_pct=0.0004,
        max_bars_in_trade=max_bars,
        trailing_stop_pct=0.0,
        trailing_stop_activation_pct=0.0
    )

    decision_engine = DecisionEngine(
        capital=100,
        risk_per_trade=0.005,
        min_expectancy=min_exp,
        max_distance=max_dist,
        blocked_regimes=blocked,
        min_mfe=min_mfe
    )

    trades = []
    active_trade = None
    bar_counter = 0
    total_bars = len(test_outcomes)
    test_start_time = time.time()

    for timestamp, state_row in test_outcomes.iterrows():
        bar_counter += 1

        # Write progress every 5% to log file
        if total_bars > 0 and bar_counter % max(1, total_bars // 20) == 0:
            pct = int(bar_counter / total_bars * 100)
            elapsed = time.time() - test_start_time
            eta = (elapsed / bar_counter) * (total_bars - bar_counter)
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(PROGRESS_LOG, "a") as f:
                f.write(f"[{timestamp_str}] [H={horizon} k={k} mfe={min_mfe}] {pct}% | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s\n")

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
            "min_mfe": min_mfe,
            "max_bars_in_trade": max_bars,
            "sample_interval": sample_interval,
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
        "min_mfe": min_mfe,
        "max_bars_in_trade": max_bars,
        "sample_interval": sample_interval,
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


def run_batch(batch_combinations, outcome_path, regime_path, ohlcv_path, train_ratio, n_workers):
    """Run a single batch of combinations in parallel."""
    results = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(str(outcome_path), str(regime_path), str(ohlcv_path), train_ratio, SAMPLE_SIZE)
    ) as executor:
        future_to_params = {executor.submit(run_single_test, params): params for params in batch_combinations}

        for future in as_completed(future_to_params):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                params = future_to_params[future]
                print(f"\n    Error: {e}")
                results.append({
                    "horizon": params[0], "norm_window": params[1],
                    "min_expectancy": params[2], "max_distance": params[3],
                    "k": params[4], "min_mfe": params[5], "max_bars_in_trade": params[6],
                    "sample_interval": params[7],
                    "blocked_regimes": ",".join(params[8]) if params[8] else "NONE",
                    "total_pnl": 0, "total_pnl_pct": 0, "win_rate": 0,
                    "total_trades": 0, "profit_factor": 0, "max_drawdown_pct": 0,
                    "sharpe": 0, "expectancy": 0,
                })

    return results


def main():
    print()
    print("=" * 70)
    print("   SCALPING GRID SEARCH - BATCH PARALLEL VERSION")
    print("=" * 70)

    # Create logs directory and initialize progress log
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "w") as f:
        f.write(f"=== SCALPING GRID SEARCH STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    # Use 2 workers to prevent system hang
    n_workers = 2
    print(f"\n✓ Workers: {n_workers} (safe for 16GB RAM)")
    print(f"✓ Progress log: {PROGRESS_LOG}")
    print(f"✓ Batch size: {BATCH_SIZE} combinations")
    print(f"✓ Data: {SAMPLE_SIZE:,} rows (~2 years)")

    # Load config
    config = get_config()
    data_dir = Path(config.get("paths.data_dir", "data"))
    pair = config.get("data.pair", "BTCUSDT")
    train_ratio = config.get("backtest.train_ratio", 0.70)

    # Get file paths
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
        PARAM_GRID["min_mfe"],
        PARAM_GRID["max_bars_in_trade"],
        PARAM_GRID["sample_interval"],
        PARAM_GRID["blocked_regimes"]
    ))

    total_combinations = len(all_combinations)
    total_batches = (total_combinations + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nTotal combinations: {total_combinations}")
    print(f"Total batches: {total_batches}")

    # Output directory
    output_dir = PROJECT_ROOT / "experiments" / "scalping" / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run batches
    print("\n" + "=" * 70)
    print("RUNNING BATCHES (results saved after each)...")
    print("=" * 70 + "\n")

    all_results = []
    grid_start = time.time()

    for batch_num in range(total_batches):
        batch_start = time.time()
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_combinations)
        batch_combinations = all_combinations[start_idx:end_idx]

        print(f"Batch {batch_num + 1}/{total_batches} [{start_idx + 1}-{end_idx}]...", end=" ", flush=True)

        # Log batch start
        with open(PROGRESS_LOG, "a") as f:
            f.write(f"\n--- BATCH {batch_num + 1}/{total_batches} STARTED ---\n")

        # Run batch
        batch_results = run_batch(batch_combinations, outcome_path, regime_path, ohlcv_path, train_ratio, n_workers)
        all_results.extend(batch_results)

        # Stats
        batch_time = time.time() - batch_start
        total_elapsed = time.time() - grid_start
        eta = (total_elapsed / (batch_num + 1)) * (total_batches - batch_num - 1)

        profitable = sum(1 for r in batch_results if r['total_pnl'] > 0)
        total_profitable = sum(1 for r in all_results if r['total_pnl'] > 0)

        print(f"Done [{batch_time:.0f}s] | Batch: {profitable}/{len(batch_results)} | Total: {total_profitable}/{len(all_results)} | ETA: {timedelta(seconds=int(eta))}")

        # Log batch completion
        with open(PROGRESS_LOG, "a") as f:
            f.write(f"--- BATCH {batch_num + 1} COMPLETED in {batch_time:.0f}s | {profitable}/{len(batch_results)} profitable ---\n")

        # Save results after EACH batch
        results_df = pd.DataFrame(all_results).sort_values("total_pnl", ascending=False)
        batch_file = output_dir / f"scalping_BATCH_{batch_num + 1}_{timestamp}.csv"
        results_df.to_csv(batch_file, index=False)
        print(f"    Saved: {batch_file.name}")

        # Free memory between batches
        gc.collect()
        time.sleep(1)  # Brief pause to let system stabilize

    total_time = time.time() - grid_start
    print(f"\n{'=' * 70}")
    print(f"Completed in {timedelta(seconds=int(total_time))}")

    # Save final results
    results_df = pd.DataFrame(all_results).sort_values("total_pnl", ascending=False)
    final_file = output_dir / f"scalping_BATCH_FINAL_{timestamp}.csv"
    results_df.to_csv(final_file, index=False)
    print(f"\nFinal results saved to: {final_file}")

    # Show top results
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)

    for idx, row in results_df.head(10).iterrows():
        print(f"${row['total_pnl']:+8.2f} | H={row['horizon']} mfe={row['min_mfe']:.3f} "
              f"k={row['k']} bars={row['max_bars_in_trade']} si={row['sample_interval']} | "
              f"WR={row['win_rate']:.1f}% ({row['total_trades']} trades)")

    # Summary
    profitable = len(results_df[results_df['total_pnl'] > 0])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {profitable}/{len(results_df)} combinations profitable ({profitable/len(results_df)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
