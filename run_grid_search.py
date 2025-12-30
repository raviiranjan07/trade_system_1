#!/usr/bin/env python3
"""
Grid Search for Optimal Backtest Configuration.

Tests all combinations of parameters to find the best performing setup.

Usage:
    python run_grid_search.py
    python run_grid_search.py --quick    # Fewer combinations for faster testing
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from backtest.backtester import Backtester
from backtest.metrics import trades_to_dataframe
from similarity.similarity_engine import SimilarityEngine


import threading


class InlineProgress:
    """Inline progress display with independent background timer."""

    def __init__(self, prefix: str, total_samples: int = 0):
        self.prefix = prefix
        self.start_time = time.time()
        self.current_pct = 0
        self.last_samples = 0
        self.last_update_time = time.time()
        self.samples_per_sec = 0
        self.total_samples = total_samples
        self.running = True
        # Print prefix
        print(f"{self.prefix}", end="", flush=True)
        # Start background timer thread
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

    def _timer_loop(self):
        """Background thread that updates timer every second."""
        last_printed = -1
        while self.running:
            elapsed = time.time() - self.start_time
            elapsed_int = int(elapsed)
            if elapsed_int != last_printed:
                last_printed = elapsed_int
                # Estimate current samples based on rate (real-time interpolation)
                time_since_update = time.time() - self.last_update_time
                estimated_samples = min(
                    self.last_samples + int(self.samples_per_sec * time_since_update),
                    self.total_samples
                )
                estimated_pct = int(estimated_samples / self.total_samples * 100) if self.total_samples > 0 else 0
                # Print timer update with estimated sample counts and rate
                print(f"\r{self.prefix} [{elapsed_int}s] {estimated_pct}% ({estimated_samples:,}/{self.total_samples:,}) {int(self.samples_per_sec):,}/s", end="", flush=True)
            time.sleep(0.5)

    def update(self, current: int, total: int):
        """Update progress and calculate real-time samples/sec rate."""
        now = time.time()
        time_delta = now - self.last_update_time
        if time_delta > 0:
            samples_delta = current - self.last_samples
            self.samples_per_sec = samples_delta / time_delta
        self.last_samples = current
        self.last_update_time = now
        self.total_samples = total
        self.current_pct = int(current / total * 100) if total > 0 else 0

    def finish(self, result_str: str):
        """Stop timer and print final result."""
        self.running = False
        self.timer_thread.join(timeout=1)
        elapsed = time.time() - self.start_time
        # Clear line and print final result
        print(f"\r{self.prefix} => {result_str} [{elapsed:.1f}s]" + " " * 10)


def run_single_backtest_with_timing(
    outcome_df, regime_df, ohlcv_df, pair,
    horizon, blocked_regimes, trailing_stop_pct, trailing_activation_pct,
    max_bars_in_trade, config, similarity_engine=None, progress_callback=None
):
    """Run a single backtest - simplified version without verbose output."""
    from decision.decision_engine import DecisionEngine
    from backtest.trade_simulator import Trade, TradeSimulator

    # Split data
    train_ratio = config.get("backtest.train_ratio", 0.70)
    split_idx = int(len(outcome_df) * train_ratio)
    test_outcomes = outcome_df.iloc[split_idx:]

    # Init components
    simulator = TradeSimulator(
        slippage_pct=config.get("backtest.slippage_pct", 0.0005),
        commission_pct=config.get("backtest.commission_pct", 0.0004),
        max_bars_in_trade=max_bars_in_trade,
        trailing_stop_pct=trailing_stop_pct,
        trailing_stop_activation_pct=trailing_activation_pct
    )

    decision_engine = DecisionEngine(
        capital=config.get("decision.capital", 10000),
        risk_per_trade=config.get("decision.risk_per_trade", 0.005),
        min_expectancy=config.get("decision.min_expectancy", -0.002),
        max_distance=config.get("decision.max_distance", 3.0),
        blocked_regimes=blocked_regimes
    )

    sample_interval = config.get("backtest.sample_interval", 60)
    trades = []
    active_trade = None
    bar_counter = 0
    total_bars = len(test_outcomes)
    progress_interval = max(1, total_bars // 50)  # Update ~50 times (every 2%)

    for timestamp, state_row in test_outcomes.iterrows():
        bar_counter += 1

        # Progress callback
        if progress_callback and bar_counter % progress_interval == 0:
            progress_callback(bar_counter, total_bars)

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

    # Close remaining trade
    if active_trade is not None:
        last_bar = ohlcv_df.iloc[-1]
        active_trade = simulator.force_exit(active_trade, last_bar["close"], ohlcv_df.index[-1])
        trades.append(active_trade)

    # Calculate metrics
    if trades:
        from backtest.metrics import calculate_metrics
        train_start = outcome_df.index[0]
        train_end = outcome_df.index[split_idx - 1]
        test_start = test_outcomes.index[0]
        test_end = test_outcomes.index[-1]

        return calculate_metrics(
            trades=trades,
            capital=config.get("decision.capital", 10000),
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            pair=pair
        )
    return None


def load_data(pair: str, data_dir: Path):
    """Load all required data files from parquet (no database needed)."""
    # Load outcomes
    outcomes_dir = data_dir / "outcomes"
    outcome_files = list(outcomes_dir.glob(f"{pair}_*.parquet"))
    if not outcome_files:
        raise FileNotFoundError(f"No outcome files found for {pair}")
    outcome_df = pd.read_parquet(sorted(outcome_files)[-1])
    outcome_df.index = pd.to_datetime(outcome_df.index)

    # Load regimes
    regimes_dir = data_dir / "regimes"
    regime_files = list(regimes_dir.glob(f"{pair}_*.parquet"))
    if not regime_files:
        raise FileNotFoundError(f"No regime files found for {pair}")
    regime_df = pd.read_parquet(sorted(regime_files)[-1])
    regime_df.index = pd.to_datetime(regime_df.index)

    # Load OHLCV from separate parquet file
    ohlcv_dir = data_dir / "ohlcv"
    ohlcv_files = list(ohlcv_dir.glob(f"{pair}_*.parquet"))
    if not ohlcv_files:
        raise FileNotFoundError(f"No OHLCV files found for {pair}. Re-run pipeline to generate.")
    ohlcv_df = pd.read_parquet(sorted(ohlcv_files)[-1])
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)

    return outcome_df, regime_df, ohlcv_df


def run_single_backtest(
    outcome_df, regime_df, ohlcv_df, pair,
    horizon, blocked_regimes, trailing_stop_pct, trailing_activation_pct,
    max_bars_in_trade, config, similarity_engine=None, progress_callback=None
):
    """Run a single backtest with given parameters."""

    backtester = Backtester(
        train_ratio=config.get("backtest.train_ratio", 0.70),
        slippage_pct=config.get("backtest.slippage_pct", 0.0005),
        commission_pct=config.get("backtest.commission_pct", 0.0004),
        max_bars_in_trade=max_bars_in_trade,
        capital=config.get("decision.capital", 10000),
        risk_per_trade=config.get("decision.risk_per_trade", 0.005),
        k=config.get("similarity.k", 200),
        horizon=horizon,
        verbose=False,  # Quiet mode for grid search
        sample_interval=config.get("backtest.sample_interval", 60),
        similarity_backend="faiss",  # Use FAISS for speed
        faiss_nlist=config.get("similarity.faiss_nlist", 100),
        faiss_nprobe=config.get("similarity.faiss_nprobe", 10),
        min_expectancy=config.get("decision.min_expectancy", -0.002),
        max_distance=config.get("decision.max_distance", 3.0),
        blocked_regimes=blocked_regimes,
        trailing_stop_pct=trailing_stop_pct,
        trailing_stop_activation_pct=trailing_activation_pct
    )

    try:
        result = backtester.run(
            outcome_df=outcome_df,
            regime_df=regime_df,
            ohlcv_df=ohlcv_df,
            pair=pair,
            similarity_engine=similarity_engine,  # Pass pre-built engine
            progress_callback=progress_callback   # Pass progress callback
        )
        return result
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Grid search for optimal backtest configuration")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer combinations")
    parser.add_argument("--pair", type=str, default=None, help="Trading pair (default: from config)")
    args = parser.parse_args()

    config = Config()
    pair = args.pair or config.get("data.pair", "BTCUSDT")
    data_dir = Path(config.get("paths.data_dir", "data"))

    print()
    print("=" * 80)
    print("                    GRID SEARCH FOR OPTIMAL CONFIGURATION")
    print("=" * 80)
    print()

    # Load data once
    print("Loading data...")
    data_load_start = time.time()
    try:
        outcome_df, regime_df, ohlcv_df = load_data(pair, data_dir)
        data_load_time = time.time() - data_load_start
        print(f"  Outcomes: {len(outcome_df):,} rows")
        print(f"  Regimes: {len(regime_df):,} rows")
        print(f"  OHLCV: {len(ohlcv_df):,} candles")
        print(f"  Load time: {data_load_time:.1f}s")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nRun 'python run_pipeline.py' first to generate data.")
        sys.exit(1)

    # Check available horizons in data
    available_horizons = []
    for h in [10, 15, 30, 120]:
        if f"mfe_long_{h}m" in outcome_df.columns:
            available_horizons.append(h)
    print(f"  Available horizons: {available_horizons}")
    print()

    # Define parameter grid
    if args.quick:
        # Quick mode - fewer combinations
        horizons = [30]  # Just test 30m
        blocked_regime_options = [
            ["TREND_LOW_VOL"],
        ]
        exit_strategies = [
            {"name": "TP_ONLY", "trailing": 0.0, "activation": 0.0, "timeout": 0},
            {"name": "TP+TIMEOUT", "trailing": 0.0, "activation": 0.0, "timeout": 120},
        ]
    else:
        # Full grid search
        horizons = available_horizons
        blocked_regime_options = [
            [],  # Trade all regimes
            ["HIGH_VOL"],
            ["TREND_LOW_VOL"],
            ["HIGH_VOL", "TREND_LOW_VOL"],
            ["TREND_HIGH_VOL"],
        ]
        exit_strategies = [
            {"name": "TP_ONLY", "trailing": 0.0, "activation": 0.0, "timeout": 0},
            {"name": "TP+TRAIL_0.5%", "trailing": 0.005, "activation": 0.002, "timeout": 0},
            {"name": "TP+TRAIL_1%", "trailing": 0.01, "activation": 0.003, "timeout": 0},
            {"name": "TP+TRAIL_2%", "trailing": 0.02, "activation": 0.005, "timeout": 0},
        ]

    # Calculate total combinations
    total = len(horizons) * len(blocked_regime_options) * len(exit_strategies)
    print(f"Testing {total} combinations...")
    print("-" * 80)

    # Pre-build similarity engine ONCE (major optimization!)
    # This takes ~500s but is reused across all combinations
    print()
    print("Building FAISS similarity engine (one-time cost)...")
    build_start = time.time()

    # Split data for training (same ratio used in backtester)
    train_ratio = config.get("backtest.train_ratio", 0.70)
    split_idx = int(len(outcome_df) * train_ratio)
    train_outcomes = outcome_df.iloc[:split_idx]
    total_test_samples = len(outcome_df) - split_idx  # Test samples for progress display

    similarity_engine = SimilarityEngine(
        outcome_df=train_outcomes,
        regime_df=regime_df,
        k=config.get("similarity.k", 200),
        backend="faiss",
        faiss_nlist=config.get("similarity.faiss_nlist", 100),
        faiss_nprobe=config.get("similarity.faiss_nprobe", 10)
    )
    build_elapsed = time.time() - build_start
    print(f"Similarity engine built in {build_elapsed:.1f}s")
    print("-" * 80)
    print()

    # Results storage
    results = []
    best_result = None
    best_pnl = float('-inf')

    # Timing tracking
    grid_start_time = time.time()
    combo_times = []

    # Run grid search
    count = 0
    for horizon in horizons:
        for blocked in blocked_regime_options:
            for exit_strat in exit_strategies:
                count += 1
                blocked_str = ",".join(blocked) if blocked else "NONE"

                # Create inline progress display
                prefix = f"[{count}/{total}] H={horizon}m | Blocked={blocked_str:15} | Exit={exit_strat['name']:15}"
                progress = InlineProgress(prefix, total_test_samples)

                # Create progress callback
                def make_progress_callback(prog):
                    def callback(current, total_bars):
                        prog.update(current, total_bars)
                    return callback

                result = run_single_backtest_with_timing(
                    outcome_df, regime_df, ohlcv_df, pair,
                    horizon=horizon,
                    blocked_regimes=blocked,
                    trailing_stop_pct=exit_strat["trailing"],
                    trailing_activation_pct=exit_strat["activation"],
                    max_bars_in_trade=exit_strat["timeout"],
                    config=config,
                    similarity_engine=similarity_engine,  # Reuse pre-built engine!
                    progress_callback=make_progress_callback(progress)
                )

                # Calculate time for this combination
                combo_elapsed = time.time() - progress.start_time
                combo_times.append(combo_elapsed)

                if result and result.total_trades > 0:
                    pnl = result.total_pnl
                    win_rate = result.win_rate * 100
                    trades = result.total_trades
                    pf = result.profit_factor if result.profit_factor != float('inf') else 99.99

                    # Calculate ETA
                    avg_time = sum(combo_times) / len(combo_times)
                    remaining = total - count
                    eta_seconds = avg_time * remaining
                    eta_str = str(timedelta(seconds=int(eta_seconds)))

                    # Print final result (overwrites progress bar)
                    result_str = f"${pnl:+,.0f} ({win_rate:.1f}% WR, {trades} trades, PF={pf:.2f}) | ETA: {eta_str}"
                    progress.finish(result_str)

                    # Extract pipeline timing breakdown
                    timing = result.pipeline_timing if hasattr(result, 'pipeline_timing') else {}

                    results.append({
                        "horizon": horizon,
                        "blocked_regimes": blocked_str,
                        "exit_strategy": exit_strat["name"],
                        "trailing_pct": exit_strat["trailing"],
                        "timeout": exit_strat["timeout"],
                        "total_pnl": pnl,
                        "total_pnl_pct": result.total_pnl_pct * 100,
                        "win_rate": win_rate,
                        "total_trades": trades,
                        "profit_factor": pf,
                        "max_drawdown_pct": result.max_drawdown_pct * 100,
                        "sharpe": result.sharpe_ratio or 0,
                        "expectancy": result.expectancy,
                        "time_seconds": combo_elapsed,
                        # Pipeline timing breakdown
                        "time_data_split": timing.get("1. Data splitting", 0),
                        "time_build_similarity": timing.get("2. Build similarity engine", 0),
                        "time_init_decision": timing.get("3. Init decision engine", 0),
                        "time_walk_forward": timing.get("4. Walk-forward simulation", 0),
                        "time_calc_metrics": timing.get("5. Calculate metrics", 0),
                    })

                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_result = results[-1].copy()
                else:
                    # Calculate ETA even for failed tests
                    avg_time = sum(combo_times) / len(combo_times)
                    remaining = total - count
                    eta_seconds = avg_time * remaining
                    eta_str = str(timedelta(seconds=int(eta_seconds)))

                    progress.finish(f"NO TRADES or ERROR | ETA: {eta_str}")

    # Calculate total time
    total_elapsed = time.time() - grid_start_time
    total_time_str = str(timedelta(seconds=int(total_elapsed)))
    avg_combo_time = sum(combo_times) / len(combo_times) if combo_times else 0

    print()
    print("-" * 80)
    print(f"Grid search completed in {total_time_str}")
    print(f"Average time per combination: {avg_combo_time:.1f}s")
    print("-" * 80)

    # Print average pipeline timing breakdown
    if results:
        print()
        print("AVERAGE PIPELINE TIMING (per combination):")
        print("-" * 50)
        avg_data_split = sum(r.get("time_data_split", 0) for r in results) / len(results)
        avg_build_sim = sum(r.get("time_build_similarity", 0) for r in results) / len(results)
        avg_init_dec = sum(r.get("time_init_decision", 0) for r in results) / len(results)
        avg_walk_fwd = sum(r.get("time_walk_forward", 0) for r in results) / len(results)
        avg_calc_met = sum(r.get("time_calc_metrics", 0) for r in results) / len(results)

        total_pipeline = avg_data_split + avg_build_sim + avg_init_dec + avg_walk_fwd + avg_calc_met
        if total_pipeline > 0:
            print(f"  1. Data splitting:          {avg_data_split:6.2f}s  ({avg_data_split/total_pipeline*100:5.1f}%)")
            print(f"  2. Build similarity engine: {avg_build_sim:6.2f}s  ({avg_build_sim/total_pipeline*100:5.1f}%)")
            print(f"  3. Init decision engine:    {avg_init_dec:6.2f}s  ({avg_init_dec/total_pipeline*100:5.1f}%)")
            print(f"  4. Walk-forward simulation: {avg_walk_fwd:6.2f}s  ({avg_walk_fwd/total_pipeline*100:5.1f}%)")
            print(f"  5. Calculate metrics:       {avg_calc_met:6.2f}s  ({avg_calc_met/total_pipeline*100:5.1f}%)")
            print("-" * 50)
            print(f"  TOTAL:                      {total_pipeline:6.2f}s  (100.0%)")
        print("-" * 50)

    # Create results DataFrame
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("total_pnl", ascending=False)

        print()
        print("=" * 80)
        print("                              TOP 10 RESULTS")
        print("=" * 80)
        print()

        top10 = df.head(10)
        for i, row in top10.iterrows():
            rank = list(top10.index).index(i) + 1
            print(f"#{rank}: ${row['total_pnl']:+,.2f} ({row['total_pnl_pct']:+.2f}%)")
            print(f"    Horizon: {row['horizon']}m | Blocked: {row['blocked_regimes']} | Exit: {row['exit_strategy']}")
            print(f"    Trades: {row['total_trades']} | Win Rate: {row['win_rate']:.1f}% | PF: {row['profit_factor']:.2f}")
            print(f"    Max DD: {row['max_drawdown_pct']:.2f}% | Sharpe: {row['sharpe']:.2f}")
            print()

        # Save results
        output_dir = data_dir / "grid_search"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"grid_search_{pair}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        # Print best configuration
        print()
        print("=" * 80)
        print("                         BEST CONFIGURATION")
        print("=" * 80)
        print()
        if best_result:
            print(f"  Horizon:         {best_result['horizon']}m")
            print(f"  Blocked Regimes: {best_result['blocked_regimes']}")
            print(f"  Exit Strategy:   {best_result['exit_strategy']}")
            print(f"  Trailing Stop:   {best_result['trailing_pct']*100:.1f}%")
            print(f"  Timeout:         {best_result['timeout']} bars")
            print()
            print(f"  Total P&L:       ${best_result['total_pnl']:+,.2f} ({best_result['total_pnl_pct']:+.2f}%)")
            print(f"  Win Rate:        {best_result['win_rate']:.1f}%")
            print(f"  Profit Factor:   {best_result['profit_factor']:.2f}")
            print(f"  Total Trades:    {best_result['total_trades']}")
            print(f"  Max Drawdown:    {best_result['max_drawdown_pct']:.2f}%")
        print()
        print("=" * 80)

    else:
        print("\nNo valid results found. Check your data and configuration.")


if __name__ == "__main__":
    main()
