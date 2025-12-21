#!/usr/bin/env python3
"""
Visualization Runner

Generates all visualizations for the trading system.

Usage:
    python run_visualizations.py                    # Generate all plots
    python run_visualizations.py --type regimes     # Only regime plots
    python run_visualizations.py --type outcomes    # Only outcome plots
    python run_visualizations.py --type states      # Only state plots
    python run_visualizations.py --pair ETHUSDT     # For specific pair
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from visualizations.plot_regimes import (
    plot_regime_distribution,
    plot_regime_transitions,
)
from visualizations.plot_outcomes import (
    plot_mfe_mae_distribution,
    plot_expectancy_by_regime,
    plot_horizon_comparison,
    plot_outcome_over_time,
)
from visualizations.plot_states import (
    plot_state_heatmap,
    plot_state_correlation,
    plot_state_distributions,
    plot_state_by_regime,
    plot_pca_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trading system visualizations",
    )
    parser.add_argument(
        "--type",
        choices=["all", "regimes", "outcomes", "states"],
        default="all",
        help="Type of visualizations to generate",
    )
    parser.add_argument(
        "--pair",
        default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Timeframe (default: 1m)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/charts",
        help="Output directory for charts",
    )
    return parser.parse_args()


def load_data(pair: str, timeframe: str) -> tuple:
    """Load outcome and regime data."""
    outcomes_path = Path(f"data/outcomes/{pair}_{timeframe}_outcomes.parquet")
    regimes_path = Path(f"data/regimes/{pair}_{timeframe}_regimes.parquet")

    if not outcomes_path.exists():
        raise FileNotFoundError(f"Outcomes file not found: {outcomes_path}\nRun the pipeline first.")

    if not regimes_path.exists():
        raise FileNotFoundError(f"Regimes file not found: {regimes_path}\nRun the pipeline first.")

    outcome_df = pd.read_parquet(outcomes_path)
    regime_df = pd.read_parquet(regimes_path)

    return outcome_df, regime_df


def generate_regime_charts(regime_df: pd.DataFrame, output_dir: Path, pair: str) -> None:
    """Generate regime visualizations."""
    print("\n[REGIMES]")

    print("  - Regime distribution...")
    plot_regime_distribution(
        regime_df,
        title=f"Regime Distribution - {pair}",
        save_path=str(output_dir / f"{pair}_regime_distribution.png"),
    )

    print("  - Regime transitions...")
    plot_regime_transitions(
        regime_df,
        save_path=str(output_dir / f"{pair}_regime_transitions.png"),
    )


def generate_outcome_charts(
    outcome_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    output_dir: Path,
    pair: str,
) -> None:
    """Generate outcome visualizations."""
    print("\n[OUTCOMES]")

    for horizon in [10, 30, 120]:
        print(f"  - MFE/MAE distribution ({horizon}m)...")
        plot_mfe_mae_distribution(
            outcome_df,
            horizon=horizon,
            save_path=str(output_dir / f"{pair}_mfe_mae_{horizon}m.png"),
        )

    print("  - Expectancy by regime...")
    plot_expectancy_by_regime(
        outcome_df,
        regime_df,
        horizon=30,
        save_path=str(output_dir / f"{pair}_expectancy_by_regime.png"),
    )

    print("  - Horizon comparison...")
    plot_horizon_comparison(
        outcome_df,
        save_path=str(output_dir / f"{pair}_horizon_comparison.png"),
    )

    print("  - Outcome over time...")
    plot_outcome_over_time(
        outcome_df,
        horizon=30,
        save_path=str(output_dir / f"{pair}_outcome_over_time.png"),
    )


def generate_state_charts(
    outcome_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    output_dir: Path,
    pair: str,
) -> None:
    """Generate state vector visualizations."""
    print("\n[STATES]")

    print("  - State heatmap...")
    plot_state_heatmap(
        outcome_df,
        save_path=str(output_dir / f"{pair}_state_heatmap.png"),
    )

    print("  - State correlation...")
    plot_state_correlation(
        outcome_df,
        save_path=str(output_dir / f"{pair}_state_correlation.png"),
    )

    print("  - State distributions...")
    plot_state_distributions(
        outcome_df,
        save_path=str(output_dir / f"{pair}_state_distributions.png"),
    )

    print("  - State by regime...")
    plot_state_by_regime(
        outcome_df,
        regime_df,
        save_path=str(output_dir / f"{pair}_state_by_regime.png"),
    )

    print("  - PCA projection...")
    try:
        plot_pca_states(
            outcome_df,
            regime_df,
            save_path=str(output_dir / f"{pair}_state_pca.png"),
        )
    except ImportError:
        print("    (skipped - scikit-learn not installed)")


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("TRADING SYSTEM VISUALIZATIONS")
    print("=" * 60)
    print(f"  Pair:       {args.pair}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Type:       {args.type}")
    print(f"  Output:     {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        print("\nLoading data...")
        outcome_df, regime_df = load_data(args.pair, args.timeframe)
        print(f"  Loaded {len(outcome_df):,} outcome records")
        print(f"  Loaded {len(regime_df):,} regime records")

        # Generate charts
        if args.type in ["all", "regimes"]:
            generate_regime_charts(regime_df, output_dir, args.pair)

        if args.type in ["all", "outcomes"]:
            generate_outcome_charts(outcome_df, regime_df, output_dir, args.pair)

        if args.type in ["all", "states"]:
            generate_state_charts(outcome_df, regime_df, output_dir, args.pair)

        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Charts saved to: {output_dir.absolute()}")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
