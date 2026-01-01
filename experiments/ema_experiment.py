#!/usr/bin/env python3
"""
EMA Experiment - Test different EMA configurations

Compares:
1. Current: EMA 50/200 (10D state vector)
2. Expanded: EMA 21/50/100/200 (14D state vector)

Run: python -m tests.ema_experiment
"""

import sys
import time
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from features.momentum import compute_momentum_features
from features.volatility import compute_volatility_features
from features.volume import compute_volume_features
from features.location import compute_location_features
from regime.regime_labeler import smooth_regime
from state.normalizer import RollingNormalizer
from similarity.similarity_engine import SimilarityEngine
from decision.decision_engine import DecisionEngine
from backtest.trade_simulator import TradeSimulator
from backtest.metrics import calculate_metrics


# =============================================================================
# EMA CONFIGURATIONS
# =============================================================================

# Known baseline results (from previous backtest - no need to re-run)
BASELINE_RESULTS = {
    "50/200 (current)": {
        "total_trades": 137,
        "win_rate": 100.0,
        "total_pnl_pct": 12.63,
        "max_drawdown_pct": 0.0,
        "sharpe": 3.08,
        "profit_factor": 999.99,
        "no_trade_reasons": {},
    }
}

# Only test new configurations
EMA_CONFIGS = {
    "21/50/100/200": {
        "emas": [21, 50, 100, 200],
        "short_ema": 21,
        "long_ema": 100,
        "state_columns": [
            "ema21_slope_z",
            "ema50_slope_z",
            "ema100_slope_z",
            "ema200_slope_z",
            "trend_alignment_short",
            "trend_alignment_long",
            "return_5m_z",
            "return_15m_z",
            "rsi_z",
            "atr_percentile",
            "volume_z",
            "vwap_distance_z",
            "range_position",
        ]
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def ema_slope(ema_series: pd.Series, window: int = 5) -> pd.Series:
    """Compute EMA slope (difference over window)."""
    return ema_series.diff(window)


def compute_custom_trend_features(df: pd.DataFrame, ema_periods: List[int]) -> pd.DataFrame:
    """Compute trend features with custom EMA periods."""
    df = df.copy()

    # Compute all EMAs
    for period in ema_periods:
        df[f"ema{period}"] = ema(df["close"], period)
        # Use different slope windows based on EMA period
        slope_window = max(5, period // 10)
        df[f"ema{period}_slope"] = ema_slope(df[f"ema{period}"], slope_window)

    return df


def compute_trend_alignments(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute trend alignment features based on config."""
    df = df.copy()

    if len(config["emas"]) == 2:
        # Current config: single trend alignment
        short, long = config["short_ema"], config["long_ema"]
        df["trend_alignment"] = np.sign(df[f"ema{short}"] - df[f"ema{long}"])
    else:
        # Expanded config: two trend alignments
        df["trend_alignment_short"] = np.sign(df["ema21"] - df["ema100"])
        df["trend_alignment_long"] = np.sign(df["ema50"] - df["ema200"])

    return df


def normalize_features(df: pd.DataFrame, config: Dict, window: int = 2000) -> pd.DataFrame:
    """Normalize features to z-scores and percentiles."""
    df = df.copy()
    norm = RollingNormalizer(window)

    # Normalize EMA slopes
    for period in config["emas"]:
        df[f"ema{period}_slope_z"] = norm.zscore(df[f"ema{period}_slope"])

    # Normalize other features (already computed by existing modules)
    df["return_5m_z"] = norm.zscore(df["return_5m"])
    df["return_15m_z"] = norm.zscore(df["return_15m"])
    df["rsi_z"] = norm.zscore(df["rsi_14"])
    df["volume_z"] = norm.zscore(df["volume_raw"])
    df["vwap_distance_z"] = norm.zscore(df["vwap_distance"])
    df["atr_percentile"] = norm.percentile(df["atr_14"])

    return df


def compute_mfe_mae(prices: pd.Series, horizon: int):
    """Compute MFE and MAE for a single horizon."""
    future_max = prices.iloc[::-1].rolling(horizon).max().iloc[::-1].shift(-1)
    future_min = prices.iloc[::-1].rolling(horizon).min().iloc[::-1].shift(-1)

    entry = prices
    mfe = (future_max - entry) / entry
    mae = (future_min - entry) / entry

    return mfe, mae


def label_outcomes_custom(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Label outcomes for the specified horizon."""
    df = df.copy()

    mfe_long, mae_long = compute_mfe_mae(df["close"], horizon)
    mfe_short = -mae_long
    mae_short = -mfe_long

    df[f"mfe_long_{horizon}m"] = mfe_long
    df[f"mae_long_{horizon}m"] = mae_long
    df[f"mfe_short_{horizon}m"] = mfe_short
    df[f"mae_short_{horizon}m"] = mae_short

    return df


def label_regime_row_custom(row) -> str:
    """
    Custom regime labeler that works with both EMA configs.

    For expanded config (21/50/100/200):
    - Uses trend_alignment_long (50 vs 200) for regime detection
    - Falls back to trend_alignment if available (current config)
    """
    TREND_SLOPE_THRESHOLD = 0.7
    HIGH_VOL_THRESHOLD = 0.85
    LOW_VOL_THRESHOLD = 0.35

    trend_strength = abs(row["ema200_slope_z"])
    vol = row["atr_percentile"]

    # Get alignment based on config
    if "trend_alignment" in row.index:
        alignment = row["trend_alignment"]
    elif "trend_alignment_long" in row.index:
        alignment = row["trend_alignment_long"]
    else:
        alignment = 0

    # Volatility shock (directionless)
    if vol >= HIGH_VOL_THRESHOLD:
        return "HIGH_VOL"

    # Trending regimes
    if trend_strength >= TREND_SLOPE_THRESHOLD and alignment != 0:
        if vol <= LOW_VOL_THRESHOLD:
            return "TREND_LOW_VOL"
        else:
            return "TREND_HIGH_VOL"

    # Otherwise range / chop
    return "RANGE_LOW_VOL"


def label_regimes_custom(df: pd.DataFrame, smoothing_window: int = 30) -> pd.DataFrame:
    """Label market regimes with custom labeler."""
    df = df.copy()

    # Use custom regime labeler that handles both configs
    df["regime_raw"] = df.apply(label_regime_row_custom, axis=1)
    df["regime"] = smooth_regime(df["regime_raw"], window=smoothing_window)

    return df


def run_backtest(
    outcome_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    state_columns: List[str],
    horizon: int = 5,
    sample_interval: int = 15,
    min_expectancy: float = 0.001,
    max_distance: float = 3.0,
    capital: float = 200.0,
    train_ratio: float = 0.70,
) -> Dict[str, Any]:
    """Run backtest with specified parameters."""

    # Split data
    split_idx = int(len(outcome_df) * train_ratio)
    train_outcomes = outcome_df.iloc[:split_idx]
    test_outcomes = outcome_df.iloc[split_idx:]

    # Build similarity engine with custom state columns
    class CustomSimilarityEngine(SimilarityEngine):
        """Similarity engine with custom state columns."""
        pass

    # Temporarily override STATE_COLUMNS
    import similarity.similarity_engine as sim_module
    original_columns = sim_module.STATE_COLUMNS
    sim_module.STATE_COLUMNS = state_columns

    try:
        similarity_engine = SimilarityEngine(
            outcome_df=train_outcomes,
            regime_df=regime_df,
            k=200,
            backend="bruteforce"
        )

        decision_engine = DecisionEngine(
            capital=capital,
            risk_per_trade=0.005,
            min_expectancy=min_expectancy,
            max_distance=max_distance,
            blocked_regimes=[]
        )

        simulator = TradeSimulator(
            slippage_pct=0.0005,
            commission_pct=0.0004,
            max_bars_in_trade=0,
            trailing_stop_pct=0.0,
            trailing_stop_activation_pct=0.0
        )

        trades = []
        active_trade = None
        bar_counter = 0

        no_trade_reasons = {}

        for timestamp, state_row in test_outcomes.iterrows():
            bar_counter += 1

            if timestamp not in ohlcv_df.index:
                continue
            bar = ohlcv_df.loc[timestamp]

            # Update active trade
            if active_trade is not None:
                active_trade = simulator.update_trade(active_trade, bar, timestamp)
                if active_trade.exit_time is not None:
                    trades.append(active_trade)
                    active_trade = None

            # Check for new signal at sample interval
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
                else:
                    reason = decision.get("reason", "UNKNOWN")
                    no_trade_reasons[reason] = no_trade_reasons.get(reason, 0) + 1

        # Force close any remaining trade
        if active_trade is not None:
            last_bar = ohlcv_df.iloc[-1]
            active_trade = simulator.force_exit(active_trade, last_bar["close"], ohlcv_df.index[-1])
            trades.append(active_trade)

        # Calculate metrics
        if trades:
            metrics = calculate_metrics(
                trades=trades,
                capital=capital,
                train_start=outcome_df.index[0],
                train_end=outcome_df.index[split_idx - 1],
                test_start=test_outcomes.index[0],
                test_end=test_outcomes.index[-1],
                pair="BTCUSDT"
            )
            return {
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate * 100,
                "total_pnl_pct": metrics.total_pnl_pct * 100,
                "max_drawdown_pct": metrics.max_drawdown_pct * 100,
                "sharpe": metrics.sharpe_ratio or 0,
                "profit_factor": metrics.profit_factor if metrics.profit_factor != float('inf') else 999.99,
                "no_trade_reasons": no_trade_reasons,
            }
        else:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe": 0,
                "profit_factor": 0,
                "no_trade_reasons": no_trade_reasons,
            }
    finally:
        # Restore original STATE_COLUMNS
        sim_module.STATE_COLUMNS = original_columns


def run_ema_experiment(config_name: str, config: Dict, ohlcv_df: pd.DataFrame) -> Dict[str, Any]:
    """Run experiment for a single EMA configuration."""

    print(f"\n{'='*70}")
    print(f"  Testing: {config_name}")
    print(f"  EMAs: {config['emas']}")
    print(f"  State dimensions: {len(config['state_columns'])}")
    print(f"{'='*70}")

    start_time = time.time()

    # Step 1: Compute trend features with custom EMAs
    print("  [1/6] Computing EMA features...")
    df = compute_custom_trend_features(ohlcv_df.copy(), config["emas"])

    # Step 2: Compute other features
    print("  [2/6] Computing other features...")
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_location_features(df)

    # Step 3: Compute trend alignments
    print("  [3/6] Computing trend alignments...")
    df = compute_trend_alignments(df, config)

    # Step 4: Normalize features
    print("  [4/6] Normalizing features...")
    df = normalize_features(df, config)

    # Step 5: Label regimes and outcomes
    print("  [5/6] Labeling regimes and outcomes...")
    df = label_regimes_custom(df)
    df = label_outcomes_custom(df, horizon=5)

    # Drop NaN rows
    df = df.dropna()

    # Prepare dataframes
    outcome_df = df[config["state_columns"] + ["mfe_long_5m", "mae_long_5m", "mfe_short_5m", "mae_short_5m"]].copy()
    regime_df = df[["regime"]].copy()

    # Step 6: Run backtest
    print("  [6/6] Running backtest (si=15, exp=0.001, H=5)...")
    result = run_backtest(
        outcome_df=outcome_df,
        regime_df=regime_df,
        ohlcv_df=ohlcv_df,
        state_columns=config["state_columns"],
        horizon=5,
        sample_interval=15,
        min_expectancy=0.001,
        max_distance=3.0,
        capital=200.0,
    )

    elapsed = time.time() - start_time
    print(f"  Completed in {timedelta(seconds=int(elapsed))}")

    return result


def main():
    """Main entry point."""

    print("\n" + "="*70)
    print("           EMA EXPERIMENT - Testing Different Configurations")
    print("="*70)

    # Load config
    config = Config()
    data_dir = Path(config.get("paths.data_dir", "data"))

    # Load OHLCV data
    print("\nLoading OHLCV data...")
    ohlcv_files = sorted((data_dir / "ohlcv").glob("BTCUSDT_*.parquet"))
    if not ohlcv_files:
        print("ERROR: No OHLCV data found. Run the pipeline first.")
        return

    ohlcv_df = pd.read_parquet(ohlcv_files[-1])
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
    print(f"Loaded {len(ohlcv_df):,} candles")

    # Run experiments for new configurations only
    results = {}
    for config_name, ema_config in EMA_CONFIGS.items():
        results[config_name] = run_ema_experiment(config_name, ema_config, ohlcv_df)

    # Combine with baseline results
    all_results = {**BASELINE_RESULTS, **results}

    # Print comparison table
    print("\n" + "="*70)
    print("                    EMA EXPERIMENT RESULTS")
    print("              (H=5, si=15, exp=0.001, capital=$200)")
    print("="*70)
    print()
    print(f"{'Config':<20} | {'Trades':>7} | {'Win Rate':>9} | {'P&L':>9} | {'Max DD':>8} | {'Sharpe':>7}")
    print("-"*70)

    for config_name, result in all_results.items():
        trades = result["total_trades"]
        wr = result["win_rate"]
        pnl = result["total_pnl_pct"]
        dd = result["max_drawdown_pct"]
        sharpe = result["sharpe"]

        print(f"{config_name:<20} | {trades:>7} | {wr:>8.1f}% | {pnl:>+8.2f}% | {dd:>7.2f}% | {sharpe:>7.2f}")

    print("="*70)

    # Print no-trade reasons
    print("\nNo-Trade Reasons:")
    for config_name, result in results.items():
        print(f"\n  {config_name}:")
        for reason, count in result.get("no_trade_reasons", {}).items():
            print(f"    {reason}: {count:,}")


if __name__ == "__main__":
    main()
