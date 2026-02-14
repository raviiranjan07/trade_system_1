"""
Grid Search: Test all Target x Horizon combinations for Risk-Based Backtest

Tests:
- Targets: 15bp, 25bp (removed 8bp - structurally impossible after 8bp fees)
- Horizons: 3, 5, 15, 30 minutes
- Total: 8 combinations
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade_system.config import Config
from trade_system.backtest import RiskBacktester


def load_data():
    """Load all required data files."""
    data_dir = Path("data")

    def ensure_datetime_index(df):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    ohlcv_df = pd.read_parquet(data_dir / "ohlcv" / "BTCUSDT_1m_ohlcv.parquet")
    ohlcv_df = ensure_datetime_index(ohlcv_df)

    outcome_df = pd.read_parquet(data_dir / "outcomes" / "BTCUSDT_1m_outcomes.parquet")
    outcome_df = ensure_datetime_index(outcome_df)

    regime_df = pd.read_parquet(data_dir / "regimes" / "BTCUSDT_1m_regimes.parquet")
    regime_df = ensure_datetime_index(regime_df)

    case_df = pd.read_parquet(data_dir / "outcomes" / "BTCUSDT_1m_cases.parquet")
    case_df = ensure_datetime_index(case_df)

    return ohlcv_df, outcome_df, regime_df, case_df


def run_single_backtest(ohlcv_df, outcome_df, regime_df, case_df, target_bps, horizon, mae_cut):
    """Run a single backtest with given parameters."""

    backtester = RiskBacktester(
        train_ratio=0.70,
        # Entry filters
        min_edge_ratio=1.5,  # CRITICAL: filter out 50/50 noise
        max_p_case1=0.10,
        max_mae_median=30.0,
        max_recovery_median=50.0,
        min_p_recovery=0.80,
        max_distance=3.0,
        # Position management
        mae_cut_threshold=mae_cut,
        max_bars_in_trade=200,
        # Trade settings
        target_bps=float(target_bps),
        horizon=horizon,
        capital=100,
        risk_per_trade=0.005,
        # Fees (limit orders)
        slippage_pct=0.0,
        commission_pct=0.0004,
        # Other
        sample_interval=15,
        k=200,
        similarity_backend="faiss",
        verbose=False,
    )

    result = backtester.run(
        outcome_df=outcome_df,
        regime_df=regime_df,
        ohlcv_df=ohlcv_df,
        case_df=case_df,
        pair="BTCUSDT",
    )

    return result


def main():
    print()
    print("=" * 80)
    print("        GRID SEARCH: Target x Horizon x MAE Cut")
    print("=" * 80)
    print()

    # Load data once
    print("Loading data...")
    ohlcv_df, outcome_df, regime_df, case_df = load_data()
    print(f"  OHLCV: {len(ohlcv_df):,} | Outcomes: {len(outcome_df):,} | Cases: {len(case_df):,}")
    print()

    # Grid parameters
    # NOTE: Removed 8bp - structurally impossible (8bp target - 8bp fees = 0 net profit)
    targets = [15, 25]
    horizons = [3, 5, 15, 30]
    mae_cuts = [50]  # Can add more: [30, 50, 80]

    results = []

    total = len(targets) * len(horizons) * len(mae_cuts)
    current = 0

    for target in targets:
        for horizon in horizons:
            for mae_cut in mae_cuts:
                current += 1
                print(f"[{current}/{total}] Target={target}bp, H={horizon}, MAE_cut={mae_cut}bp...", end=" ", flush=True)

                try:
                    result = run_single_backtest(
                        ohlcv_df, outcome_df, regime_df, case_df,
                        target_bps=target,
                        horizon=horizon,
                        mae_cut=mae_cut
                    )

                    # Calculate net per trade
                    fee_bps = 8  # 4bp * 2
                    net_win_bps = target - fee_bps
                    net_loss_bps = mae_cut + fee_bps
                    required_ratio = net_loss_bps / net_win_bps if net_win_bps > 0 else float('inf')

                    # Get exit breakdown
                    exits = result.trades_by_exit or {}
                    win_count = exits.get("WIN", {}).get("count", 0)
                    cut_count = exits.get("CUT", {}).get("count", 0)
                    actual_ratio = win_count / cut_count if cut_count > 0 else float('inf')

                    results.append({
                        "target_bps": target,
                        "horizon": horizon,
                        "mae_cut": mae_cut,
                        "trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "pnl": result.total_pnl,
                        "pnl_pct": result.total_pnl_pct,
                        "max_dd": result.max_drawdown_pct,
                        "profit_factor": result.profit_factor,
                        "win_count": win_count,
                        "cut_count": cut_count,
                        "net_win_bps": net_win_bps,
                        "net_loss_bps": net_loss_bps,
                        "required_ratio": required_ratio,
                        "actual_ratio": actual_ratio,
                        "profitable": actual_ratio >= required_ratio,
                    })

                    status = "PROFIT" if result.total_pnl > 0 else "LOSS"
                    print(f"P&L: ${result.total_pnl:+.2f} ({status})")

                except Exception as e:
                    print(f"ERROR: {e}")
                    results.append({
                        "target_bps": target,
                        "horizon": horizon,
                        "mae_cut": mae_cut,
                        "trades": 0,
                        "win_rate": 0,
                        "pnl": 0,
                        "pnl_pct": 0,
                        "error": str(e),
                    })

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Print results table
    print()
    print("=" * 80)
    print("                           RESULTS SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Target':<8} {'H':<6} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'W/C Ratio':<12} {'Required':<10} {'Status':<8}")
    print("-" * 80)

    for _, row in df.iterrows():
        status = "OK" if row.get("profitable", False) else "BAD"
        pnl_str = f"${row['pnl']:+.2f}"
        ratio_str = f"{row.get('actual_ratio', 0):.1f}:1" if row.get('actual_ratio', 0) < float('inf') else "N/A"
        req_str = f"{row.get('required_ratio', 0):.1f}:1" if row.get('required_ratio', 0) < float('inf') else "N/A"

        print(f"{row['target_bps']}bp{'':<4} H={row['horizon']:<4} {row['trades']:<8} {row['win_rate']*100:>5.1f}% {pnl_str:<12} {ratio_str:<12} {req_str:<10} {status:<8}")

    print("-" * 80)

    # Best result
    if len(df) > 0:
        best = df.loc[df['pnl'].idxmax()]
        print(f"\nBest: Target={best['target_bps']}bp, H={best['horizon']} -> P&L: ${best['pnl']:+.2f}")

        # Profitable combinations
        profitable = df[df['pnl'] > 0]
        if len(profitable) > 0:
            print(f"\nProfitable combinations: {len(profitable)}/{len(df)}")
            for _, row in profitable.iterrows():
                print(f"  Target={row['target_bps']}bp, H={row['horizon']}: ${row['pnl']:+.2f}")
        else:
            print("\nNo profitable combinations found.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"experiments/grid_search_{timestamp}.csv"
    Path("experiments").mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
