#!/usr/bin/env python3
"""
Trading System Pipeline CLI

Unified command-line interface for running the complete trading pipeline
or individual stages.

Usage:
    # Run all stages with default config
    python run_pipeline.py

    # Run specific stages
    python run_pipeline.py --stages state_vectors regime_labeling

    # Override pair and date range
    python run_pipeline.py --pair ETHUSDT --start 2023-06-01 --end 2023-09-01

    # Use custom config file
    python run_pipeline.py --config config/production.yaml

    # Verbose output
    python run_pipeline.py -v

    # Dry run (show what would be executed)
    python run_pipeline.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from exceptions import (
    DatabaseConnectionError,
    ConfigurationError,
    DataValidationError,
    TradingSystemError,
)
from pipeline.orchestrator import PipelineOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Trading System Pipeline - Unified Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    Run all stages
  %(prog)s --stages state_vectors             Run only state vector generation
  %(prog)s --pair ETHUSDT --start 2023-01-01  Override trading pair and dates
  %(prog)s --config config/prod.yaml          Use custom config file
  %(prog)s --dry-run                          Show execution plan without running
        """,
    )

    # Config
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config/config.yaml)",
    )

    # Stages
    parser.add_argument(
        "-s", "--stages",
        nargs="+",
        choices=["state_vectors", "regime_labeling", "outcome_labeling", "similarity", "decision"],
        default=None,
        help="Specific stages to run (default: all enabled in config)",
    )

    # Data overrides
    parser.add_argument(
        "-p", "--pair",
        type=str,
        default=None,
        help="Trading pair (e.g., BTCUSDT, ETHUSDT)",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD format)",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD format)",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser.parse_args()


def show_config_summary(config, args) -> None:
    """Display configuration summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    pair = args.pair or config.get("data.pair")
    start = args.start or config.get("data.start_date")
    end = args.end or config.get("data.end_date")

    print(f"  Pair:       {pair}")
    print(f"  Timeframe:  {config.get('data.timeframe')}")
    print(f"  Date Range: {start} to {end}")
    print()
    print("  Stages enabled:")

    if args.stages:
        stages = args.stages
    else:
        pipeline_config = config.get_section("pipeline").get("stages", {})
        stages = [s for s, enabled in pipeline_config.items() if enabled]

    for i, stage in enumerate(stages, 1):
        print(f"    {i}. {stage}")

    print()
    print("  Key Parameters:")
    print(f"    Normalization window: {config.get('normalization.window')}")
    print(f"    Similarity K:         {config.get('similarity.k')}")
    print(f"    Capital:              ${config.get('decision.capital'):,}")
    print(f"    Risk per trade:       {config.get('decision.risk_per_trade') * 100:.1f}%")
    print("=" * 60 + "\n")


def validate_startup(config, args) -> bool:
    """
    Validate configuration and prerequisites before running pipeline.

    Returns:
        True if validation passes, False otherwise.
    """
    print("\nValidating configuration...")

    try:
        # Validate config values
        config.validate()
        print("  [OK] Configuration valid")
    except ConfigurationError as e:
        print(f"  [FAIL] {e}", file=sys.stderr)
        return False

    # Check if state_vectors stage is enabled (requires DB)
    stages = args.stages
    if stages is None:
        pipeline_config = config.get_section("pipeline").get("stages", {})
        stages = [s for s, enabled in pipeline_config.items() if enabled]

    if "state_vectors" in stages:
        print("  Checking database connection...")
        try:
            from data.raw.ohlcv_loader import OHLCVLoader
            loader = OHLCVLoader()
            loader.test_connection()
            print("  [OK] Database connection successful")
        except DatabaseConnectionError as e:
            print(f"\n  [FAIL] Database connection failed\n", file=sys.stderr)
            print(str(e), file=sys.stderr)
            return False

    print("  [OK] All checks passed\n")
    return True


def _print_trading_decision(decision: dict) -> None:
    """Print formatted trading decision."""
    action = decision.get("action", "N/A")

    if action == "TRADE":
        direction = decision.get("direction", "")
        position_size = decision.get("position_size", 0)
        stop_pct = decision.get("stop_loss_pct", 0) * 100
        tp_pct = decision.get("take_profit_pct", 0) * 100
        expectancy = decision.get("expectancy", 0) * 100
        regime = decision.get("regime", "")

        # Color coding for terminal (green for LONG, red for SHORT)
        if direction == "LONG":
            arrow = "▲"
            signal = "BUY / LONG"
        else:
            arrow = "▼"
            signal = "SELL / SHORT"

        print("  " + "=" * 56)
        print(f"  {arrow} TRADING SIGNAL: {signal} {arrow}")
        print("  " + "=" * 56)
        print()
        print(f"    Position Size:    ${position_size:,.2f}")
        print(f"    Stop Loss:        {stop_pct:.2f}%")
        print(f"    Take Profit:      {tp_pct:.2f}%")
        print(f"    Expectancy:       {expectancy:+.3f}%")
        print(f"    Market Regime:    {regime}")
        print()
        print("  " + "-" * 56)
        print("  Risk/Reward Analysis:")
        if stop_pct > 0:
            rr_ratio = tp_pct / stop_pct
            print(f"    R:R Ratio:        1:{rr_ratio:.2f}")
        print(f"    Max Loss:         ${position_size * stop_pct / 100:,.2f}")
        print(f"    Max Gain:         ${position_size * tp_pct / 100:,.2f}")
        print("  " + "-" * 56)

    elif action == "NO_TRADE":
        reason = decision.get("reason", "Unknown")
        reason_messages = {
            "NO_DATA": "Insufficient historical data for analysis",
            "NEGATIVE_EXPECTANCY": "Historical expectancy is negative - no edge",
            "HIGH_VOL_REGIME": "Market is in HIGH_VOL regime - too risky",
            "LOW_SIMILARITY": "No similar historical states found",
            "INVALID_RISK": "Cannot calculate valid risk parameters",
        }
        print("  " + "=" * 56)
        print("  ⊘ NO TRADE SIGNAL")
        print("  " + "=" * 56)
        print()
        print(f"    Reason: {reason}")
        print(f"    Detail: {reason_messages.get(reason, reason)}")
        print()
        print("  Recommendation: Wait for better market conditions")
        print("  " + "-" * 56)

    else:
        print("  Trading Decision:")
        print(f"    Action: {action}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        # Load config
        config = get_config(args.config)

        # Override logging level if verbose
        if args.verbose:
            config._config["logging"]["level"] = "DEBUG"

        # Show summary
        show_config_summary(config, args)

        # Dry run - just show plan
        if args.dry_run:
            print("DRY RUN - No actions taken")
            print("Run without --dry-run to execute the pipeline")
            return 0

        # Validate before running
        if not validate_startup(config, args):
            return 1

        # Confirm execution
        response = input("Proceed with pipeline execution? [Y/n]: ").strip().lower()
        if response and response != "y":
            print("Aborted.")
            return 0

        # Run pipeline
        orchestrator = PipelineOrchestrator(args.config)
        result = orchestrator.run(
            stages=args.stages,
            pair=args.pair,
            start_date=args.start,
            end_date=args.end,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"  Status:     {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Completed:  {result.stages_completed}")
        if result.stages_failed:
            print(f"  Failed:     {result.stages_failed}")
        print(f"  Duration:   {result.execution_time_seconds:.2f} seconds")

        if result.decision:
            print()
            _print_trading_decision(result.decision)

        print("=" * 60 + "\n")

        return 0 if result.success else 1

    except DatabaseConnectionError as e:
        print("\n" + "=" * 60, file=sys.stderr)
        print("DATABASE CONNECTION ERROR", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        return 1

    except ConfigurationError as e:
        print("\n" + "=" * 60, file=sys.stderr)
        print("CONFIGURATION ERROR", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        return 1

    except TradingSystemError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
