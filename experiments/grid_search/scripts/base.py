#!/usr/bin/env python3
"""
Base module for grid search experiments across all horizons.
Import this module in horizon-specific experiment scripts.

Usage:
    from tests.grid_search.base import run_grid_search

    run_grid_search(
        experiment_name="exp1_min_expectancy",
        horizon=5,
        grid_params={...}
    )
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
import threading
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from similarity.similarity_engine import SimilarityEngine
from decision.decision_engine import DecisionEngine
from backtest.trade_simulator import TradeSimulator
from backtest.metrics import calculate_metrics


class ProgressTracker:
    """Inline progress display with timer thread."""

    def __init__(self, prefix: str, total_samples: int = 0):
        self.prefix = prefix
        self.start_time = time.time()
        self.total_samples = total_samples
        self.last_samples = 0
        self.last_update_time = time.time()
        self.samples_per_sec = 0
        self.running = True
        print(f"{self.prefix}", end="", flush=True)
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

    def _timer_loop(self):
        last_printed = -1
        while self.running:
            elapsed = int(time.time() - self.start_time)
            if elapsed != last_printed:
                last_printed = elapsed
                time_since_update = time.time() - self.last_update_time
                estimated_samples = min(
                    self.last_samples + int(self.samples_per_sec * time_since_update),
                    self.total_samples
                )
                pct = int(estimated_samples / self.total_samples * 100) if self.total_samples > 0 else 0
                print(f"\r{self.prefix} [{elapsed}s] {pct}% {int(self.samples_per_sec):,}/s", end="", flush=True)
            time.sleep(0.5)

    def update(self, current: int, total: int):
        now = time.time()
        time_delta = now - self.last_update_time
        if time_delta > 0:
            self.samples_per_sec = (current - self.last_samples) / time_delta
        self.last_samples = current
        self.last_update_time = now
        self.total_samples = total

    def finish(self, result_str: str):
        self.running = False
        self.timer_thread.join(timeout=1)
        elapsed = time.time() - self.start_time
        print(f"\r{self.prefix} => {result_str} [{elapsed:.1f}s]" + " " * 20)


def run_single_backtest(
    outcome_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    config: Config,
    similarity_engine: SimilarityEngine,
    horizon: int,
    min_expectancy: float,
    max_distance: float,
    blocked_regimes: List[str],
    progress_callback
):
    """Run a single backtest with specified parameters."""

    train_ratio = config.get("backtest.train_ratio", 0.70)
    split_idx = int(len(outcome_df) * train_ratio)
    test_outcomes = outcome_df.iloc[split_idx:]

    simulator = TradeSimulator(
        slippage_pct=config.get("backtest.slippage_pct", 0.0005),
        commission_pct=config.get("backtest.commission_pct", 0.0004),
        max_bars_in_trade=0,
        trailing_stop_pct=0.0,
        trailing_stop_activation_pct=0.0
    )

    decision_engine = DecisionEngine(
        capital=config.get("decision.capital", 10000),
        risk_per_trade=config.get("decision.risk_per_trade", 0.005),
        min_expectancy=min_expectancy,
        max_distance=max_distance,
        blocked_regimes=blocked_regimes,
        min_mfe=config.get("decision.min_mfe", 0.0),
        max_leverage=config.get("decision.max_leverage", 1.0),
    )

    sample_interval = config.get("backtest.sample_interval", 60)
    trades = []
    active_trade = None
    bar_counter = 0
    total_bars = len(test_outcomes)
    progress_interval = max(1, total_bars // 50)

    for timestamp, state_row in test_outcomes.iterrows():
        bar_counter += 1
        if progress_callback and bar_counter % progress_interval == 0:
            progress_callback(bar_counter, total_bars)

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

    if trades:
        return calculate_metrics(
            trades=trades,
            capital=config.get("decision.capital", 10000),
            train_start=outcome_df.index[0],
            train_end=outcome_df.index[split_idx - 1],
            test_start=test_outcomes.index[0],
            test_end=test_outcomes.index[-1],
            pair="BTCUSDT"
        )
    return None


def run_grid_search(
    experiment_name: str,
    horizon: int,
    grid_params: Dict[str, List[Any]]
):
    """
    Run grid search for specified parameters.

    Args:
        experiment_name: Name for this experiment (used in output files)
        horizon: Outcome horizon in minutes (5, 10, 15, 30, 120)
        grid_params: Dict with lists of values to test:
            - min_expectancy: List[float]
            - max_distance: List[float]
            - blocked_regimes: List[List[str]]
    """
    config = Config()
    data_dir = Path(config.get("paths.data_dir", "data"))
    pair = config.get("data.pair", "BTCUSDT")

    print()
    print("=" * 80)
    print(f"   GRID SEARCH H={horizon}m - {experiment_name}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    outcome_df = pd.read_parquet(sorted((data_dir / "outcomes").glob(f"{pair}_*.parquet"))[-1])
    outcome_df.index = pd.to_datetime(outcome_df.index)
    regime_df = pd.read_parquet(sorted((data_dir / "regimes").glob(f"{pair}_*.parquet"))[-1])
    regime_df.index = pd.to_datetime(regime_df.index)
    ohlcv_df = pd.read_parquet(sorted((data_dir / "ohlcv").glob(f"{pair}_*.parquet"))[-1])
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)

    print(f"  Outcomes: {len(outcome_df):,} | Regimes: {len(regime_df):,} | OHLCV: {len(ohlcv_df):,}")

    # Build similarity engine
    backend = config.get("similarity.backend", "bruteforce")
    print(f"\nBuilding similarity engine (backend={backend})...")
    build_start = time.time()
    train_ratio = config.get("backtest.train_ratio", 0.70)
    split_idx = int(len(outcome_df) * train_ratio)
    train_outcomes = outcome_df.iloc[:split_idx]
    total_test_samples = len(outcome_df) - split_idx

    similarity_engine = SimilarityEngine(
        outcome_df=train_outcomes,
        regime_df=regime_df,
        k=config.get("similarity.k", 200),
        backend=backend,
        faiss_nlist=config.get("similarity.faiss_nlist", 100),
        faiss_nprobe=config.get("similarity.faiss_nprobe", 10)
    )
    print(f"Built in {time.time() - build_start:.1f}s")

    # Generate all combinations
    min_exp_list = grid_params.get("min_expectancy", [0.0])
    max_dist_list = grid_params.get("max_distance", [3.0])
    blocked_list = grid_params.get("blocked_regimes", [[]])

    combinations = list(product(min_exp_list, max_dist_list, blocked_list))
    total = len(combinations)

    print(f"\nTesting {total} combinations for H={horizon}m...")
    print(f"  min_expectancy: {min_exp_list}")
    print(f"  max_distance: {max_dist_list}")
    print(f"  blocked_regimes: {[','.join(b) if b else 'NONE' for b in blocked_list]}")
    print("-" * 80)

    results = []
    count = 0
    grid_start = time.time()

    for min_exp, max_dist, blocked in combinations:
        count += 1
        blocked_str = ",".join(blocked) if blocked else "NONE"
        prefix = f"[{count}/{total}] exp={min_exp:.3f} dist={max_dist:.1f} blk={blocked_str:15}"
        progress = ProgressTracker(prefix, total_test_samples)

        def make_callback(p):
            def cb(cur, tot): p.update(cur, tot)
            return cb

        result = run_single_backtest(
            outcome_df=outcome_df,
            regime_df=regime_df,
            ohlcv_df=ohlcv_df,
            config=config,
            similarity_engine=similarity_engine,
            horizon=horizon,
            min_expectancy=min_exp,
            max_distance=max_dist,
            blocked_regimes=blocked,
            progress_callback=make_callback(progress)
        )

        if result and result.total_trades > 0:
            pnl = result.total_pnl
            wr = result.win_rate * 100
            pf = result.profit_factor if result.profit_factor != float('inf') else 99.99
            progress.finish(f"${pnl:+,.0f} ({wr:.1f}% WR, {result.total_trades} trades, PF={pf:.2f})")
            results.append({
                "horizon": horizon,
                "experiment": experiment_name,
                "min_expectancy": min_exp,
                "max_distance": max_dist,
                "blocked_regimes": blocked_str,
                "total_pnl": pnl,
                "total_pnl_pct": result.total_pnl_pct * 100,
                "win_rate": wr,
                "total_trades": result.total_trades,
                "profit_factor": pf,
                "max_drawdown_pct": result.max_drawdown_pct * 100,
                "sharpe": result.sharpe_ratio or 0,
                "expectancy": result.expectancy,
            })
        else:
            progress.finish("NO TRADES")
            results.append({
                "horizon": horizon,
                "experiment": experiment_name,
                "min_expectancy": min_exp,
                "max_distance": max_dist,
                "blocked_regimes": blocked_str,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "win_rate": 0,
                "total_trades": 0,
                "profit_factor": 0,
                "max_drawdown_pct": 0,
                "sharpe": 0,
                "expectancy": 0,
            })

    elapsed = time.time() - grid_start
    print()
    print("-" * 80)
    print(f"Completed in {timedelta(seconds=int(elapsed))}")

    # Save results
    if results:
        df = pd.DataFrame(results).sort_values("total_pnl", ascending=False)
        output_dir = data_dir / "grid_search" / f"h{horizon}"
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{experiment_name}_{pair}_{ts}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        print("\n" + "=" * 80)
        print("TOP 5 RESULTS (sorted by P&L):")
        print("=" * 80)
        for i, row in df.head(5).iterrows():
            print(f"  ${row['total_pnl']:+,.0f} | exp={row['min_expectancy']:.3f} | "
                  f"dist={row['max_distance']:.1f} | blk={row['blocked_regimes']} | "
                  f"WR={row['win_rate']:.1f}% | {row['total_trades']} trades")

        print("\n" + "=" * 80)
        print("WORST 3 RESULTS:")
        print("=" * 80)
        for i, row in df.tail(3).iterrows():
            print(f"  ${row['total_pnl']:+,.0f} | exp={row['min_expectancy']:.3f} | "
                  f"dist={row['max_distance']:.1f} | blk={row['blocked_regimes']} | "
                  f"WR={row['win_rate']:.1f}% | {row['total_trades']} trades")

    return results
