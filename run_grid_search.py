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
from pathlib import Path
from datetime import datetime
from itertools import product
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from backtest.backtester import Backtester
from backtest.metrics import trades_to_dataframe
from data.raw.ohlcv_loader import OHLCVLoader


def load_data(pair: str, data_dir: Path):
    """Load all required data files."""
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

    # Load OHLCV
    start_date = outcome_df.index[0]
    end_date = outcome_df.index[-1] + pd.Timedelta(days=1)
    loader = OHLCVLoader()
    ohlcv_df = loader.fetch_ohlcv(pair=pair, start_time=start_date, end_time=end_date)

    return outcome_df, regime_df, ohlcv_df


def run_single_backtest(
    outcome_df, regime_df, ohlcv_df, pair,
    horizon, blocked_regimes, trailing_stop_pct, trailing_activation_pct,
    max_bars_in_trade, config
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
            pair=pair
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
    try:
        outcome_df, regime_df, ohlcv_df = load_data(pair, data_dir)
        print(f"  Outcomes: {len(outcome_df):,} rows")
        print(f"  OHLCV: {len(ohlcv_df):,} candles")
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
            {"name": "TP+TIMEOUT_60", "trailing": 0.0, "activation": 0.0, "timeout": 60},
            {"name": "TP+TIMEOUT_120", "trailing": 0.0, "activation": 0.0, "timeout": 120},
            {"name": "TP+TRAIL_0.5%", "trailing": 0.005, "activation": 0.002, "timeout": 0},
            {"name": "TP+TRAIL_1%", "trailing": 0.01, "activation": 0.003, "timeout": 0},
            {"name": "TP+TRAIL_2%", "trailing": 0.02, "activation": 0.005, "timeout": 0},
            {"name": "TIMEOUT_ONLY_60", "trailing": 0.0, "activation": 0.0, "timeout": 60},
            {"name": "TIMEOUT_ONLY_120", "trailing": 0.0, "activation": 0.0, "timeout": 120},
        ]

    # Calculate total combinations
    total = len(horizons) * len(blocked_regime_options) * len(exit_strategies)
    print(f"Testing {total} combinations...")
    print("-" * 80)

    # Results storage
    results = []
    best_result = None
    best_pnl = float('-inf')

    # Run grid search
    count = 0
    for horizon in horizons:
        for blocked in blocked_regime_options:
            for exit_strat in exit_strategies:
                count += 1
                blocked_str = ",".join(blocked) if blocked else "NONE"

                print(f"[{count}/{total}] H={horizon}m | Blocked={blocked_str} | Exit={exit_strat['name']}", end="")

                result = run_single_backtest(
                    outcome_df, regime_df, ohlcv_df, pair,
                    horizon=horizon,
                    blocked_regimes=blocked,
                    trailing_stop_pct=exit_strat["trailing"],
                    trailing_activation_pct=exit_strat["activation"],
                    max_bars_in_trade=exit_strat["timeout"],
                    config=config
                )

                if result and result.total_trades > 0:
                    pnl = result.total_pnl
                    win_rate = result.win_rate * 100
                    trades = result.total_trades
                    pf = result.profit_factor if result.profit_factor != float('inf') else 99.99

                    print(f" => ${pnl:+,.0f} ({win_rate:.1f}% WR, {trades} trades, PF={pf:.2f})")

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
                        "expectancy": result.expectancy
                    })

                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_result = results[-1].copy()
                else:
                    print(" => NO TRADES or ERROR")

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
