"""
Comparative Backtest: Standard vs Risk-Based Entry Filtering

This script compares:
1. Standard approach: Entry based on MFE expectancy prediction
2. Risk-based approach: Entry filtered by P(Case1), MAE, recovery time

The hypothesis: Risk filtering should reduce losses by avoiding high P(Case1) entries.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade_system.config import Config
from trade_system.backtest import (
    Backtester,
    RiskBacktester,
    print_risk_backtest_report,
)
from trade_system.backtest.backtester import print_backtest_report


def load_data(config: Config):
    """Load all required data files."""

    data_dir = Path("data")
    pair = config.get("data.pair", "BTCUSDT")
    timeframe = config.get("data.timeframe", "1m")

    print("Loading data files...")

    def ensure_datetime_index(df, name):
        """Ensure DataFrame has datetime index."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    # OHLCV
    ohlcv_path = data_dir / "ohlcv" / f"{pair}_{timeframe}_ohlcv.parquet"
    ohlcv_df = pd.read_parquet(ohlcv_path)
    ohlcv_df = ensure_datetime_index(ohlcv_df, "ohlcv")
    print(f"  OHLCV: {len(ohlcv_df):,} candles")

    # State vectors + outcomes
    outcome_path = data_dir / "outcomes" / f"{pair}_{timeframe}_outcomes.parquet"
    outcome_df = pd.read_parquet(outcome_path)
    outcome_df = ensure_datetime_index(outcome_df, "outcomes")
    print(f"  Outcomes: {len(outcome_df):,} rows")

    # Regimes
    regime_path = data_dir / "regimes" / f"{pair}_{timeframe}_regimes.parquet"
    regime_df = pd.read_parquet(regime_path)
    regime_df = ensure_datetime_index(regime_df, "regimes")
    print(f"  Regimes: {len(regime_df):,} rows")

    # Case labels (for risk-based backtest)
    case_path = data_dir / "outcomes" / f"{pair}_{timeframe}_cases.parquet"
    if case_path.exists():
        case_df = pd.read_parquet(case_path)
        case_df = ensure_datetime_index(case_df, "cases")
        print(f"  Cases: {len(case_df):,} rows")
    else:
        print(f"  Cases: NOT FOUND - run debug_case_labeler.py first")
        case_df = None

    return ohlcv_df, outcome_df, regime_df, case_df, pair


def run_standard_backtest(ohlcv_df, outcome_df, regime_df, config, pair):
    """Run standard backtester (MFE-based entry)."""

    print()
    print("=" * 70)
    print("  STANDARD BACKTEST (MFE-Based Entry)")
    print("=" * 70)

    backtester = Backtester(
        train_ratio=config.get("backtest.train_ratio", 0.70),
        slippage_pct=0.0,  # 0 bps - using limit orders
        commission_pct=0.0004,  # 4 bps per side = 8 bps round-trip
        max_bars_in_trade=config.get("backtest.max_bars_in_trade", 0),
        capital=config.get("decision.capital", 10000),
        risk_per_trade=config.get("decision.risk_per_trade", 0.005),
        k=config.get("similarity.k", 200),
        horizon=config.get("similarity.default_horizon", 5),
        verbose=True,
        sample_interval=config.get("backtest.sample_interval", 15),
        similarity_backend=config.get("similarity.backend", "faiss"),
        min_expectancy=config.get("decision.min_expectancy", 0.001),
        max_distance=config.get("decision.max_distance", 3.0),
        blocked_regimes=config.get("decision.blocked_regimes", []),
        min_mfe=0.0,
    )

    result = backtester.run(
        outcome_df=outcome_df,
        regime_df=regime_df,
        ohlcv_df=ohlcv_df,
        pair=pair,
    )

    print_backtest_report(result)
    return result


def run_risk_backtest(ohlcv_df, outcome_df, regime_df, case_df, config, pair):
    """Run risk-based backtester (Case Probability Model)."""

    print()
    print("=" * 70)
    print("  RISK-BASED BACKTEST (Case Probability Model)")
    print("=" * 70)

    # Get risk model config
    risk_config = config.get("risk_model", {})

    backtester = RiskBacktester(
        train_ratio=config.get("backtest.train_ratio", 0.70),
        # Risk entry thresholds
        max_p_case1=risk_config.get("max_p_case1", 0.10),
        max_mae_median=risk_config.get("max_mae_median", 30.0),
        max_recovery_median=risk_config.get("max_recovery_median", 50.0),
        min_p_recovery=risk_config.get("min_p_recovery", 0.80),
        max_distance=config.get("decision.max_distance", 3.0),
        # Position management
        mae_cut_threshold=risk_config.get("mae_cut_threshold", 50.0),
        max_bars_in_trade=risk_config.get("max_bars_in_trade", 200),
        # Trade settings
        target_bps=15.0,  # 15 bps target
        horizon=10,  # H=10 for risk analysis
        capital=config.get("decision.capital", 10000),
        risk_per_trade=config.get("decision.risk_per_trade", 0.005),
        # Fees (limit orders - no slippage)
        slippage_pct=0.0,  # 0 bps - using limit orders
        commission_pct=0.0004,  # 4 bps per side = 8 bps round-trip
        # Signal interval
        sample_interval=config.get("backtest.sample_interval", 15),
        # Similarity settings
        k=config.get("similarity.k", 200),
        similarity_backend=config.get("similarity.backend", "faiss"),
        verbose=True,
    )

    result = backtester.run(
        outcome_df=outcome_df,
        regime_df=regime_df,
        ohlcv_df=ohlcv_df,
        case_df=case_df,
        pair=pair,
    )

    print_risk_backtest_report(result)
    return result


def compare_results(standard_result, risk_result):
    """Print side-by-side comparison."""

    print()
    print("=" * 70)
    print("                    COMPARISON: Standard vs Risk-Based")
    print("=" * 70)
    print()

    # Helper to format values
    def fmt_pct(v):
        return f"{v*100:.1f}%"

    def fmt_money(v):
        sign = "+" if v >= 0 else ""
        return f"{sign}${v:,.2f}"

    # Comparison table
    metrics = [
        ("Total Trades", standard_result.total_trades, risk_result.total_trades),
        ("Win Rate", fmt_pct(standard_result.win_rate), fmt_pct(risk_result.win_rate)),
        ("Total P&L", fmt_money(standard_result.total_pnl), fmt_money(risk_result.total_pnl)),
        ("Total P&L %", fmt_pct(standard_result.total_pnl_pct), fmt_pct(risk_result.total_pnl_pct)),
        ("Avg Win", fmt_money(standard_result.avg_win), fmt_money(risk_result.avg_win)),
        ("Avg Loss", fmt_money(standard_result.avg_loss), fmt_money(risk_result.avg_loss)),
        ("Profit Factor", f"{standard_result.profit_factor:.2f}", f"{risk_result.profit_factor:.2f}"),
        ("Max Drawdown", fmt_pct(standard_result.max_drawdown_pct), fmt_pct(risk_result.max_drawdown_pct)),
        ("Expectancy/Trade", fmt_money(standard_result.expectancy), fmt_money(risk_result.expectancy)),
    ]

    print(f"  {'Metric':<20} {'Standard':>15} {'Risk-Based':>15} {'Diff':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")

    for name, std, risk in metrics:
        # Calculate difference where possible
        if isinstance(std, (int, float)) and isinstance(risk, (int, float)):
            diff = risk - std
            diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f}"
        else:
            diff_str = "-"

        print(f"  {name:<20} {std:>15} {risk:>15} {diff_str:>15}")

    print()

    # Interpretation
    print("-" * 70)
    print("  INTERPRETATION")
    print("-" * 70)

    # Check if risk filtering improved results
    std_pnl = standard_result.total_pnl
    risk_pnl = risk_result.total_pnl

    if risk_pnl > std_pnl:
        improvement = (risk_pnl - std_pnl) / abs(std_pnl) * 100 if std_pnl != 0 else float('inf')
        print(f"  Risk filtering IMPROVED P&L by {improvement:.1f}%")
    else:
        decline = (std_pnl - risk_pnl) / abs(std_pnl) * 100 if std_pnl != 0 else float('inf')
        print(f"  Risk filtering REDUCED P&L by {decline:.1f}%")

    # Win rate comparison
    std_wr = standard_result.win_rate
    risk_wr = risk_result.win_rate

    if risk_wr > std_wr:
        print(f"  Win rate improved: {std_wr*100:.1f}% -> {risk_wr*100:.1f}% (+{(risk_wr-std_wr)*100:.1f}%)")
    else:
        print(f"  Win rate changed: {std_wr*100:.1f}% -> {risk_wr*100:.1f}% ({(risk_wr-std_wr)*100:.1f}%)")

    # Trade count
    std_trades = standard_result.total_trades
    risk_trades = risk_result.total_trades

    filter_rate = 1 - (risk_trades / std_trades) if std_trades > 0 else 0
    print(f"  Filtering rate: {filter_rate*100:.1f}% of signals filtered out")
    print(f"  Trades: {std_trades} -> {risk_trades} ({risk_trades - std_trades:+d})")

    print()
    print("=" * 70)


def main():
    """Run comparative backtest."""

    print()
    print("=" * 70)
    print("       COMPARATIVE BACKTEST: Standard vs Risk-Based")
    print("=" * 70)
    print()

    # Load config
    config = Config()

    # Load data
    ohlcv_df, outcome_df, regime_df, case_df, pair = load_data(config)

    if case_df is None:
        print()
        print("ERROR: Case labels not found!")
        print("Run: python debug_case_labeler.py")
        print("This will generate the required case labels for risk-based backtest.")
        return

    # Run standard backtest
    standard_result = run_standard_backtest(
        ohlcv_df, outcome_df, regime_df, config, pair
    )

    # Run risk-based backtest
    risk_result = run_risk_backtest(
        ohlcv_df, outcome_df, regime_df, case_df, config, pair
    )

    # Compare results
    compare_results(standard_result, risk_result)


if __name__ == "__main__":
    main()
