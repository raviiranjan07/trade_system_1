#!/usr/bin/env python3
"""
Scalping Grid Search - BATCH PARALLEL VERSION

Runs in batches of 108 combinations at a time (4 batches for 360 total).
Each batch runs in parallel, then memory is freed.
Results saved after EACH batch for analysis.

Usage:
    python -m experiments.scalping.scripts.run_scalping_grid_search_batch

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

# Add project root to path before local imports
# Script is at experiments/scalping/scripts/run_*.py, so 4 parents to reach project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trade_system.config import get_config

# Log files
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_RESULTS = None   # Main log - completion lines only
LOG_FILE_PROGRESS = None  # Progress log - detailed progress + completion


def log_results(msg: str):
    """Write to results log (completion lines only)."""
    if LOG_FILE_RESULTS:
        with open(LOG_FILE_RESULTS, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def log_progress(msg: str):
    """Write to progress log (detailed progress + completion)."""
    if LOG_FILE_PROGRESS:
        with open(LOG_FILE_PROGRESS, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


def log_both(msg: str):
    """Write to both log files."""
    log_results(msg)
    log_progress(msg)


# =============================================================================
# CONFIGURATION - OPTION B v2 (higher min_mfe, quality trades)
# Lower thresholds (0.0003, 0.0005) showed -20% losses - too aggressive
# Focus on higher min_mfe to filter for quality trades only
# =============================================================================
PARAM_GRID = {
    "horizon": [3],                              # 1 - h=3 only
    "normalization_window": [180, 300],          # 2 normalization contexts
    "min_expectancy": [0.0],                     # 1 - no filter
    "max_distance": [2.0, 3.0, 4.0, 5.0],        # 4 distance thresholds
    "k": [25, 50, 100, 150, 200],                # 5 neighborhood sizes
    "min_mfe": [0.002, 0.0025, 0.003],           # 3 MFE filters (higher - quality trades)
    "max_bars_in_trade": [0],                    # 1 - no time stop (bars=3 was catastrophic)
    "sample_interval": [3],                      # 1 - every 3rd bar
    "blocked_regimes": [[], ["RANGE_LOW_VOL"], ["RANGE_LOW_VOL", "TREND_LOW_VOL"]],  # 3 regime filters
}

# 1 × 2 × 1 × 4 × 5 × 3 × 1 × 1 × 3 = 360 combinations (4 batches)

# Batch settings
BATCH_SIZE = 108   # 4 batches for 360 combinations
SAMPLE_SIZE = 500_000  # Use 500K rows (~1 year) for faster testing


# Global data for worker processes
_worker_data = {}


def init_worker(outcome_path, regime_path, ohlcv_path, train_ratio, sample_size, log_progress_path):
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
        'train_ratio': train_ratio,
        'log_progress_file': log_progress_path
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
        slippage_pct=0.0001,      # Reduced: limit orders have minimal slippage
        commission_pct=0.0002,    # Reduced: maker fee on Binance (0.02%)
        max_bars_in_trade=max_bars,
        trailing_stop_pct=0.0,
        trailing_stop_activation_pct=0.0
    )
    # Total round-trip cost now: ~0.03% (was 0.09%)

    decision_engine = DecisionEngine(
        capital=100,
        risk_per_trade=0.005,
        min_expectancy=min_exp,
        max_distance=max_dist,
        blocked_regimes=blocked,
        min_mfe=min_mfe,
        max_leverage=1.0,
        stop_floor=1e-4,
    )

    trades = []
    active_trade = None
    bar_counter = 0
    total_bars = len(test_outcomes)
    last_pct_logged = 0
    log_progress_file = _worker_data.get('log_progress_file')
    test_start_time = time.time()

    for timestamp, state_row in test_outcomes.iterrows():
        bar_counter += 1

        # Log progress every 5% to progress file only
        current_pct = int((bar_counter / total_bars) * 100)
        if current_pct >= last_pct_logged + 5 and log_progress_file:
            elapsed = int(time.time() - test_start_time)
            eta = int(elapsed / current_pct * (100 - current_pct)) if current_pct > 0 else 0
            ts = datetime.now().strftime("%H:%M:%S")
            with open(log_progress_file, "a", encoding="utf-8") as f:
                f.write(f"    [{ts}] H={horizon} k={k} mfe={min_mfe} | {current_pct}% | Elapsed: {elapsed}s | ETA: {eta}s\n")
                f.flush()
            last_pct_logged = current_pct

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

    # Separate trades by direction
    long_trades_list = [t for t in trades if t.direction == "LONG"]
    short_trades_list = [t for t in trades if t.direction == "SHORT"]
    long_pnl = sum(t.pnl for t in long_trades_list)
    short_pnl = sum(t.pnl for t in short_trades_list)

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
            "long_trades": len(long_trades_list),
            "short_trades": len(short_trades_list),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
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
        "long_trades": 0,
        "short_trades": 0,
        "long_pnl": 0,
        "short_pnl": 0,
    }


def run_batch(batch_combinations, outcome_path, regime_path, ohlcv_path, train_ratio, n_workers, batch_start_idx, total_combinations, batch_num, total_batches):
    """Run a single batch of combinations in parallel."""
    results = []
    batch_size = len(batch_combinations)

    # Log batch header to both files
    log_both(f"\n--- BATCH {batch_num}/{total_batches} ---")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(str(outcome_path), str(regime_path), str(ohlcv_path), train_ratio, SAMPLE_SIZE, str(LOG_FILE_PROGRESS))
    ) as executor:
        # Track start times and indices for each future
        future_to_info = {}
        for idx, params in enumerate(batch_combinations):
            future = executor.submit(run_single_test, params)
            future_to_info[future] = {
                "params": params,
                "combo_idx": batch_start_idx + idx + 1,
                "start_time": time.time()
            }

        completed = 0
        for future in as_completed(future_to_info):
            info = future_to_info[future]
            params = info["params"]
            combo_idx = info["combo_idx"]
            duration = time.time() - info["start_time"]
            completed += 1

            try:
                result = future.result()
                results.append(result)
                # Log completion line to both files (batch-wise progress)
                ts = datetime.now().strftime("%H:%M:%S")
                batch_pct = (completed / batch_size) * 100
                log_both(f"[{ts}] {completed:3d}/{batch_size} ({batch_pct:5.1f}%) | "
                    f"H={params[0]} nw={params[1]} exp={params[2]} dist={params[3]} "
                    f"k={params[4]} mfe={params[5]} bars={params[6]} si={params[7]} | "
                    f"${result['total_pnl']:+.2f} ({result['total_pnl_pct']:+.1f}%) | {duration:.0f}s")
            except Exception as e:
                ts = datetime.now().strftime("%H:%M:%S")
                batch_pct = (completed / batch_size) * 100
                log_both(f"[{ts}] {completed:3d}/{batch_size} ({batch_pct:5.1f}%) | ERROR: {e}")
                results.append({
                    "horizon": params[0], "norm_window": params[1],
                    "min_expectancy": params[2], "max_distance": params[3],
                    "k": params[4], "min_mfe": params[5], "max_bars_in_trade": params[6],
                    "sample_interval": params[7],
                    "blocked_regimes": ",".join(params[8]) if params[8] else "NONE",
                    "total_pnl": 0, "total_pnl_pct": 0, "win_rate": 0,
                    "total_trades": 0, "profit_factor": 0, "max_drawdown_pct": 0,
                    "sharpe": 0, "expectancy": 0,
                    "long_trades": 0, "short_trades": 0, "long_pnl": 0, "short_pnl": 0,
                })

    return results


def main():
    global LOG_FILE_RESULTS, LOG_FILE_PROGRESS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup log files (overwrite each run)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE_RESULTS = LOG_DIR / "grid_results.log"
    LOG_FILE_PROGRESS = LOG_DIR / "grid_progress.log"

    # Clear log files at start
    open(LOG_FILE_RESULTS, "w").close()
    open(LOG_FILE_PROGRESS, "w").close()

    # Verify log files work with initial write
    with open(LOG_FILE_RESULTS, "a", encoding="utf-8") as f:
        f.write(f"=== Grid Search Started: {timestamp} ===\n")
        f.flush()
    with open(LOG_FILE_PROGRESS, "a", encoding="utf-8") as f:
        f.write(f"=== Grid Search Started: {timestamp} ===\n")
        f.flush()

    print("")
    print("=" * 70)
    print("   SCALPING GRID SEARCH - BATCH PARALLEL VERSION")
    print("=" * 70)

    # Use 3 workers for faster processing
    n_workers = 3
    print(f"\n✓ Workers: {n_workers}")
    print(f"✓ Batch size: {BATCH_SIZE} combinations")
    print(f"✓ Data: {SAMPLE_SIZE:,} rows (~2 years)")
    print(f"✓ Results log: {LOG_FILE_RESULTS}")
    print(f"✓ Progress log: {LOG_FILE_PROGRESS}")

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

        print(f"\nBatch {batch_num + 1}/{total_batches} [{start_idx + 1}-{end_idx}]")

        # Run batch
        batch_results = run_batch(batch_combinations, outcome_path, regime_path, ohlcv_path, train_ratio, n_workers, start_idx, total_combinations, batch_num + 1, total_batches)
        all_results.extend(batch_results)

        # Stats
        batch_time = time.time() - batch_start
        total_elapsed = time.time() - grid_start
        eta = (total_elapsed / (batch_num + 1)) * (total_batches - batch_num - 1)

        profitable = sum(1 for r in batch_results if r['total_pnl'] > 0)
        total_profitable = sum(1 for r in all_results if r['total_pnl'] > 0)

        print(f"Batch {batch_num + 1}/{total_batches} Done [{batch_time:.0f}s] | {profitable}/{len(batch_results)} profitable | Total: {total_profitable}/{len(all_results)} | ETA: {timedelta(seconds=int(eta))}")

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
