#!/usr/bin/env python3
"""
Backtest CLI for the Trading System.

Runs walk-forward backtesting with proper train/test split to validate
the trading strategy on unseen data.

Usage:
    python run_backtest.py
    python run_backtest.py --pair ETHUSDT
    python run_backtest.py --train-ratio 0.80
    python run_backtest.py -v
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from backtest.backtester import Backtester, print_backtest_report
from backtest.metrics import trades_to_dataframe
from data.raw.ohlcv_loader import OHLCVLoader
from exceptions import TradingSystemError


def load_outcomes(pair: str, data_dir: Path) -> pd.DataFrame:
    """Load outcome data from parquet file."""
    outcomes_dir = data_dir / "outcomes"
    outcome_files = list(outcomes_dir.glob(f"{pair}_*.parquet"))

    if not outcome_files:
        raise FileNotFoundError(
            f"No outcome files found for {pair} in {outcomes_dir}\n"
            f"Run 'python run_pipeline.py' first to generate outcome data."
        )

    # Load most recent file
    outcome_file = sorted(outcome_files)[-1]
    print(f"  Loading outcomes: {outcome_file.name}")

    df = pd.read_parquet(outcome_file)
    df.index = pd.to_datetime(df.index)
    return df


def load_regimes(pair: str, data_dir: Path) -> pd.DataFrame:
    """Load regime data from parquet file."""
    regimes_dir = data_dir / "regimes"
    regime_files = list(regimes_dir.glob(f"{pair}_*.parquet"))

    if not regime_files:
        raise FileNotFoundError(
            f"No regime files found for {pair} in {regimes_dir}\n"
            f"Run 'python run_pipeline.py' first to generate regime data."
        )

    # Load most recent file
    regime_file = sorted(regime_files)[-1]
    print(f"  Loading regimes: {regime_file.name}")

    df = pd.read_parquet(regime_file)
    df.index = pd.to_datetime(df.index)
    return df


def load_state_vectors(pair: str, data_dir: Path) -> pd.DataFrame:
    """Load state vector data from parquet file."""
    states_dir = data_dir / "state_vectors"
    state_files = list(states_dir.glob(f"{pair}_*.parquet"))

    if not state_files:
        raise FileNotFoundError(
            f"No state vector files found for {pair} in {states_dir}\n"
            f"Run 'python run_pipeline.py' first to generate state vectors."
        )

    # Load most recent file
    state_file = sorted(state_files)[-1]
    print(f"  Loading states: {state_file.name}")

    df = pd.read_parquet(state_file)
    df.index = pd.to_datetime(df.index)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run backtesting on the trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_backtest.py
    python run_backtest.py --pair ETHUSDT
    python run_backtest.py --train-ratio 0.80
    python run_backtest.py --save-trades
        """
    )

    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Trading pair (default: from config)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Train/test split ratio (default: 0.70)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Starting capital (default: from config)"
    )
    parser.add_argument(
        "--save-trades",
        action="store_true",
        help="Save trade log to parquet file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=None,
        help="Check for signals every N bars (default: 60 for hourly). Use 1 for every bar (slow)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["bruteforce", "faiss"],
        default=None,
        help="Similarity search backend: 'bruteforce' (exact, slow) or 'faiss' (approximate, fast)"
    )

    args = parser.parse_args()

    try:
        # Load config
        config = Config()

        # Get parameters
        pair = args.pair or config.data.get("pair", "BTCUSDT")
        backtest_config = config.get_section("backtest")
        train_ratio = args.train_ratio or backtest_config.get("train_ratio", 0.70)
        capital = args.capital or config.decision.get("capital", 10000)
        slippage_pct = backtest_config.get("slippage_pct", 0.0005)
        commission_pct = backtest_config.get("commission_pct", 0.0004)
        max_bars = backtest_config.get("max_bars_in_trade", 120)
        sample_interval = args.sample_interval or backtest_config.get("sample_interval", 60)

        # Similarity engine settings
        similarity_config = config.similarity
        similarity_backend = args.backend or similarity_config.get("backend", "bruteforce")
        faiss_nlist = similarity_config.get("faiss_nlist", 100)
        faiss_nprobe = similarity_config.get("faiss_nprobe", 10)

        # Get data directory
        data_dir = Path(config.paths.get("data_dir", "data"))

        print()
        print("=" * 70)
        print("                    TRADING SYSTEM BACKTEST")
        print("=" * 70)
        print()
        print(f"  Pair: {pair}")
        print(f"  Train Ratio: {train_ratio*100:.0f}%")
        print(f"  Capital: ${capital:,.2f}")
        print(f"  Slippage: {slippage_pct*100:.3f}%")
        print(f"  Commission: {commission_pct*100:.3f}%")
        print(f"  Sample Interval: Every {sample_interval} bars")
        print(f"  Similarity Backend: {similarity_backend}")
        print()

        # Load data
        print("Loading data...")
        outcome_df = load_outcomes(pair, data_dir)
        regime_df = load_regimes(pair, data_dir)

        # Load OHLCV data for trade simulation
        print("  Loading OHLCV data...")
        start_date = outcome_df.index[0]
        end_date = outcome_df.index[-1] + pd.Timedelta(days=1)

        loader = OHLCVLoader()
        ohlcv_df = loader.fetch_ohlcv(
            pair=pair,
            start_time=start_date,
            end_time=end_date
        )
        print(f"  OHLCV: {len(ohlcv_df):,} candles")
        print()

        # Create and run backtester
        backtester = Backtester(
            train_ratio=train_ratio,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            max_bars_in_trade=max_bars,
            capital=capital,
            risk_per_trade=config.decision.get("risk_per_trade", 0.005),
            k=config.similarity.get("k", 200),
            horizon=config.similarity.get("default_horizon", 30),
            verbose=args.verbose or True,
            sample_interval=sample_interval,
            similarity_backend=similarity_backend,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe
        )

        result = backtester.run(
            outcome_df=outcome_df,
            regime_df=regime_df,
            ohlcv_df=ohlcv_df,
            pair=pair
        )

        # Print report
        print_backtest_report(result)

        # Save trades if requested
        if args.save_trades or backtest_config.get("save_trades", True):
            output_dir = data_dir / backtest_config.get("output_dir", "backtest")
            output_dir.mkdir(parents=True, exist_ok=True)

            trade_df = trades_to_dataframe(result.trades)
            if not trade_df.empty:
                filename = f"{pair}_trades_{result.test_start.strftime('%Y%m%d')}_{result.test_end.strftime('%Y%m%d')}.parquet"
                filepath = output_dir / filename
                trade_df.to_parquet(filepath)
                print(f"Trade log saved to: {filepath}")
                print()

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure to run the pipeline first:")
        print("  python run_pipeline.py")
        sys.exit(1)

    except TradingSystemError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nBacktest cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
