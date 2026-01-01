#!/usr/bin/env python3
"""
Paper Trading Runner.

Starts the paper trading system with real-time Binance data
but simulated order execution.

Usage:
    python run_paper_trade.py
    python run_paper_trade.py --capital 500 --horizon 5
    python run_paper_trade.py --verbose
    python run_paper_trade.py --dry-run
    python run_paper_trade.py --web-port 8080
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path for src layout
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trade_system.live.live_orchestrator import LiveOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Paper Trading System - Trade with real data, simulated execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Start with settings from config.yaml
  %(prog)s --capital 500            Override capital to $500
  %(prog)s --horizon 5              Override horizon to 5 minutes
  %(prog)s --sample-interval 15     Override signal check interval
  %(prog)s --verbose                Enable debug logging
  %(prog)s --dry-run                Validate setup without trading

All parameters default to values in config/config.yaml if not specified.
        """,
    )

    # Capital and risk (defaults from config)
    parser.add_argument(
        "-c", "--capital",
        type=float,
        default=None,
        help="Starting capital in USD (default: from config.yaml)",
    )

    # Strategy parameters (defaults from config)
    parser.add_argument(
        "-H", "--horizon",
        type=int,
        default=None,
        help="Outcome horizon in minutes (default: from config.yaml)",
    )

    parser.add_argument(
        "-s", "--sample-interval",
        type=int,
        default=None,
        help="Check for signals every N bars (default: from config.yaml)",
    )

    parser.add_argument(
        "-e", "--min-expectancy",
        type=float,
        default=None,
        help="Minimum expectancy to take trade (default: from config.yaml)",
    )

    parser.add_argument(
        "-d", "--max-distance",
        type=float,
        default=None,
        help="Maximum distance for similarity matching (default: from config.yaml)",
    )

    parser.add_argument(
        "--blocked-regimes",
        nargs="*",
        default=None,
        help="Regimes to avoid (default: from config.yaml)",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without starting trading",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug output",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (uses default if not specified)",
    )

    parser.add_argument(
        "-w", "--web-port",
        type=int,
        default=8080,
        help="Port for web dashboard (default: 8080)",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "paper_trading.log")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Reduce noise from libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def validate_setup(args: argparse.Namespace) -> bool:
    """Validate that all required files exist."""
    from trade_system.config import get_config

    print("\n" + "=" * 60)
    print("VALIDATING PAPER TRADING SETUP")
    print("=" * 60)

    # Load config
    config = get_config(args.config)

    # Get values from config, with command-line overrides
    pair = config.get("data.pair", "BTCUSDT")
    timeframe = config.get("data.timeframe", "1m")
    data_dir = Path(config.get("paths.data_dir", "data"))

    capital = args.capital if args.capital is not None else config.get("decision.capital", 200.0)
    horizon = args.horizon if args.horizon is not None else config.get("similarity.default_horizon", 5)
    sample_interval = args.sample_interval if args.sample_interval is not None else config.get("backtest.sample_interval", 15)
    min_expectancy = args.min_expectancy if args.min_expectancy is not None else config.get("decision.min_expectancy", 0.001)
    max_distance = args.max_distance if args.max_distance is not None else config.get("decision.max_distance", 3.0)
    blocked_regimes = args.blocked_regimes if args.blocked_regimes is not None else config.get("decision.blocked_regimes", [])

    errors = []

    # Check required data files using config values
    print(f"\nChecking data files for {pair} {timeframe}...")
    required_files = [
        data_dir / "outcomes" / f"{pair}_{timeframe}_outcomes.parquet",
        data_dir / "regimes" / f"{pair}_{timeframe}_regimes.parquet",
    ]

    for f in required_files:
        if f.exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            errors.append(f"Missing: {f}")

    # Check for required packages
    print("\nChecking dependencies...")
    try:
        import websockets
        print("  [OK] websockets")
    except ImportError:
        print("  [MISSING] websockets - install with: pip install websockets")
        errors.append("Missing package: websockets")

    try:
        import aiohttp
        print("  [OK] aiohttp")
    except ImportError:
        print("  [MISSING] aiohttp - install with: pip install aiohttp")
        errors.append("Missing package: aiohttp")

    # Check web dashboard dependencies (required)
    try:
        import fastapi
        print("  [OK] fastapi")
    except ImportError:
        print("  [MISSING] fastapi - install with: pip install fastapi")
        errors.append("Missing package: fastapi")

    try:
        import uvicorn
        print("  [OK] uvicorn")
    except ImportError:
        print("  [MISSING] uvicorn - install with: pip install uvicorn[standard]")
        errors.append("Missing package: uvicorn")

    # Print configuration (resolved from config + overrides)
    print("\nConfiguration:")
    print(f"  Trading Pair:     {pair}")
    print(f"  Timeframe:        {timeframe}")
    print(f"  Capital:          ${capital:.2f}")
    print(f"  Horizon:          {horizon} minutes")
    print(f"  Sample Interval:  {sample_interval} bars")
    print(f"  Min Expectancy:   {min_expectancy}")
    print(f"  Max Distance:     {max_distance}")
    print(f"  Blocked Regimes:  {blocked_regimes or 'None'}")
    print(f"  Config File:      {args.config or 'config/config.yaml (default)'}")
    print(f"  Web Dashboard:    http://localhost:{args.web_port}")

    print("=" * 60)

    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("\nSetup validation passed!")
    return True


async def run_paper_trading(args: argparse.Namespace) -> None:
    """Run the paper trading system."""
    orchestrator = LiveOrchestrator(
        config_path=args.config,
        capital=args.capital,
        horizon=args.horizon,
        sample_interval=args.sample_interval,
        min_expectancy=args.min_expectancy,
        max_distance=args.max_distance,
        blocked_regimes=args.blocked_regimes,
        web_port=args.web_port,
    )

    await orchestrator.start()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate setup
    if not validate_setup(args):
        return 1

    # Dry run mode
    if args.dry_run:
        print("\nDry run complete. Run without --dry-run to start trading.")
        return 0

    # Confirm start
    print("\n" + "=" * 60)
    print("PAPER TRADING - NO REAL MONEY AT RISK")
    print("=" * 60)
    print("\nPress Ctrl+C at any time to stop trading gracefully.\n")

    response = input("Start paper trading? [Y/n]: ").strip().lower()
    if response and response != "y":
        print("Aborted.")
        return 0

    # Run paper trading
    try:
        asyncio.run(run_paper_trading(args))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
