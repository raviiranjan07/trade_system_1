"""
Risk-Based Backtester: Uses case probability model for entry filtering and MAE-based exits.

Key difference from standard backtester:
- Entry: Filter based on risk profile (P(Case1), MAE, recovery time)
- Exit: MAE-based management (HOLD if MAE < 50bp, CUT if > 50bp)
- No direction prediction - just risk management
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm

from .trade_simulator import Trade
from .metrics import BacktestResult, calculate_metrics
from .mae_position_manager import MAEPositionManager, PositionState, PositionAction


@dataclass
class RiskTrade:
    """Extended trade with risk-based metrics."""
    # Entry info
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    direction: str
    position_size: float
    target_bps: float
    horizon: int
    regime: str

    # Risk profile at entry
    p_case1: float
    p_recovery: float
    mae_expected: float
    recovery_expected: float
    risk_score: float

    # Exit info
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # WIN, CUT, TIMEOUT

    # Live tracking
    live_mae_bps: float = 0.0
    live_mfe_bps: float = 0.0
    bars_held: int = 0

    # P&L
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_bps: Optional[float] = None

    def to_standard_trade(self) -> Trade:
        """Convert to standard Trade for metrics calculation."""
        return Trade(
            signal_time=self.signal_time,
            entry_time=self.entry_time,
            entry_price=self.entry_price,
            direction=self.direction,
            position_size=self.position_size,
            stop_loss_price=0,  # Not used
            take_profit_price=0,  # Not used
            stop_loss_pct=0,
            take_profit_pct=self.target_bps / 10000,
            regime=self.regime,
            expectancy=0,  # Not used
            exit_time=self.exit_time,
            exit_price=self.exit_price,
            exit_reason=self.exit_reason,
            pnl=self.pnl,
            pnl_pct=self.pnl_pct,
            bars_held=self.bars_held,
        )


class RiskBacktester:
    """
    Walk-forward backtester using the Case Probability Model.

    This backtester:
    1. Uses RiskProfiler to get risk profile for each potential entry
    2. Filters entries based on risk thresholds (not direction prediction)
    3. Manages positions using MAE-based HOLD/CUT logic
    """

    def __init__(
        self,
        # Data split
        train_ratio: float = 0.70,
        # Risk entry thresholds
        min_edge_ratio: float = 1.5,  # CRITICAL: min directional edge to trade
        max_p_case1: float = 0.10,
        max_mae_median: float = 30.0,
        max_recovery_median: float = 50.0,
        min_p_recovery: float = 0.80,
        max_distance: float = 3.0,
        # Position management
        mae_cut_threshold: float = 50.0,
        max_bars_in_trade: int = 200,
        # Trade settings
        target_bps: float = 15.0,
        horizon: int = 10,
        capital: float = 10000,
        risk_per_trade: float = 0.005,
        # Fees
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0004,
        # Signal interval
        sample_interval: int = 1,
        # Similarity settings
        k: int = 200,
        similarity_backend: str = "faiss",
        faiss_nlist: int = 100,
        faiss_nprobe: int = 10,
        # Misc
        verbose: bool = True,
    ):
        self.train_ratio = train_ratio
        self.min_edge_ratio = min_edge_ratio
        self.max_p_case1 = max_p_case1
        self.max_mae_median = max_mae_median
        self.max_recovery_median = max_recovery_median
        self.min_p_recovery = min_p_recovery
        self.max_distance = max_distance
        self.mae_cut_threshold = mae_cut_threshold
        self.max_bars_in_trade = max_bars_in_trade
        self.target_bps = target_bps
        self.horizon = horizon
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.sample_interval = sample_interval
        self.k = k
        self.similarity_backend = similarity_backend
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.verbose = verbose

        # Fee in bps (round-trip)
        self.fee_bps = (slippage_pct + commission_pct) * 2 * 10000

        # Position manager
        self.position_manager = MAEPositionManager({
            "mae_cut_threshold": mae_cut_threshold,
            "max_bars_in_trade": max_bars_in_trade,
        })

    def run(
        self,
        outcome_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        case_df: pd.DataFrame,  # Case labels (from case_labeler)
        pair: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        Run risk-based backtest.

        Args:
            outcome_df: DataFrame with state vectors and MFE/MAE
            regime_df: DataFrame with regime labels
            ohlcv_df: Raw OHLCV data
            case_df: DataFrame with case labels (from case_labeler)
            pair: Trading pair name

        Returns:
            BacktestResult with trades and metrics
        """
        # Import here to avoid circular imports
        from ..similarity.similarity_engine import SimilarityEngine
        from ..similarity.risk_profiler import RiskProfiler
        from ..decision.risk_based_decision import RiskBasedDecisionEngine

        # Timings
        timings = {}
        t0 = time.time()

        # 1. Split data
        split_idx = int(len(outcome_df) * self.train_ratio)
        train_outcomes = outcome_df.iloc[:split_idx]
        test_outcomes = outcome_df.iloc[split_idx:]
        train_cases = case_df.iloc[:split_idx]

        train_start = train_outcomes.index[0]
        train_end = train_outcomes.index[-1]
        test_start = test_outcomes.index[0]
        test_end = test_outcomes.index[-1]

        if self.verbose:
            print()
            print("=" * 70)
            print("              RISK-BASED BACKTESTING (Case Probability Model)")
            print("=" * 70)
            print(f"  Training Period: {train_start} to {train_end}")
            print(f"  Training Samples: {len(train_outcomes):,}")
            print(f"  Test Period: {test_start} to {test_end}")
            print(f"  Test Samples: {len(test_outcomes):,}")
            print(f"  Target: {self.target_bps} bps | Horizon: {self.horizon} bars")
            print(f"  MAE Cut Threshold: {self.mae_cut_threshold} bps")
            print(f"  EDGE FILTER: min_edge_ratio >= {self.min_edge_ratio}x (filters 50/50 noise)")
            print(f"  Risk Filters: P(Case1)<{self.max_p_case1:.0%}, MAE<{self.max_mae_median}bp")
            print("=" * 70)
            print()

        timings["1. Data splitting"] = time.time() - t0
        t0 = time.time()

        # 2. Build similarity engine on training data
        similarity = SimilarityEngine(
            outcome_df=train_outcomes,
            regime_df=regime_df,
            k=self.k,
            backend=self.similarity_backend,
            faiss_nlist=self.faiss_nlist,
            faiss_nprobe=self.faiss_nprobe,
        )
        timings["2. Build similarity engine"] = time.time() - t0
        t0 = time.time()

        # 3. Create risk profiler (uses training case labels)
        risk_profiler = RiskProfiler(
            similarity_engine=similarity,
            case_df=train_cases,
            default_target_bps=int(self.target_bps),
            default_horizon=self.horizon,
        )
        timings["3. Init risk profiler"] = time.time() - t0
        t0 = time.time()

        # 4. Create decision engine
        decision_engine = RiskBasedDecisionEngine({
            "min_edge_ratio": self.min_edge_ratio,  # CRITICAL: filter noise
            "max_p_case1": self.max_p_case1,
            "max_mae_median": self.max_mae_median,
            "max_recovery_median": self.max_recovery_median,
            "min_p_recovery": self.min_p_recovery,
            "max_distance": self.max_distance,
        })
        timings["4. Init decision engine"] = time.time() - t0
        t0 = time.time()

        # 5. Walk-forward simulation
        trades: List[RiskTrade] = []
        active_trade: Optional[RiskTrade] = None
        active_position: Optional[PositionState] = None
        signals_checked = 0
        entries_allowed = 0
        entries_filtered = {}
        bar_counter = 0

        # Progress bar
        iterator = test_outcomes.iterrows()
        if self.verbose:
            iterator = tqdm(list(iterator), desc="Risk Backtest", unit="bars")

        for timestamp, state_row in iterator:
            bar_counter += 1

            if timestamp not in ohlcv_df.index:
                continue

            bar = ohlcv_df.loc[timestamp]
            current_price = bar["close"]

            # Update active trade if exists
            if active_trade is not None and active_position is not None:
                # Update position with current price
                action, reason = self.position_manager.update(
                    active_position,
                    current_price,
                    risk_profile=None  # Could pass for adaptive timeout
                )

                active_trade.bars_held = active_position.bars_in_trade
                active_trade.live_mae_bps = active_position.current_mae_bps
                active_trade.live_mfe_bps = active_position.current_mfe_bps

                # Check if trade should close
                if action != PositionAction.HOLD:
                    # Get exit price
                    exit_price = self.position_manager.get_exit_price(
                        active_position, action, current_price
                    )

                    # Apply slippage
                    if active_trade.direction == "LONG":
                        fill_price = exit_price * (1 - self.slippage_pct)
                    else:
                        fill_price = exit_price * (1 + self.slippage_pct)

                    # Calculate P&L
                    pnl_bps = self.position_manager.calculate_pnl_bps(
                        active_position, fill_price, self.fee_bps
                    )
                    pnl_pct = pnl_bps / 10000
                    pnl = active_trade.position_size * pnl_pct

                    # Close trade
                    active_trade.exit_time = timestamp
                    active_trade.exit_price = fill_price
                    active_trade.exit_reason = action.value
                    active_trade.pnl = pnl
                    active_trade.pnl_pct = pnl_pct
                    active_trade.pnl_bps = pnl_bps

                    trades.append(active_trade)
                    active_trade = None
                    active_position = None

            # Check for new entry (only if no active trade)
            if active_trade is None and bar_counter % self.sample_interval == 0:
                signals_checked += 1

                # Get regime
                if timestamp not in regime_df.index:
                    continue
                regime = regime_df.loc[timestamp, "regime"]

                # Get risk profile
                risk_profile = risk_profiler.get_risk_profile(
                    current_state=state_row,
                    regime=regime,
                    max_timestamp=timestamp,  # Only use past data
                )

                # Check if entry allowed
                should_enter, reason = decision_engine.should_enter(risk_profile)

                if should_enter:
                    entries_allowed += 1

                    # Get entry price (next bar open)
                    future_bars = ohlcv_df.loc[timestamp:].iloc[1:]
                    if len(future_bars) == 0:
                        continue

                    next_bar = future_bars.iloc[0]
                    entry_price = next_bar["open"]

                    # Apply slippage
                    direction = risk_profile.get("direction", "LONG")
                    if direction == "LONG":
                        fill_price = entry_price * (1 + self.slippage_pct)
                    else:
                        fill_price = entry_price * (1 - self.slippage_pct)

                    # Open trade
                    active_trade = RiskTrade(
                        signal_time=timestamp,
                        entry_time=future_bars.index[0],
                        entry_price=fill_price,
                        direction=direction,
                        position_size=self.capital * self.risk_per_trade,
                        target_bps=self.target_bps,
                        horizon=self.horizon,
                        regime=regime,
                        p_case1=risk_profile.get("p_case1", 0),
                        p_recovery=risk_profile.get("p_recovery", 0),
                        mae_expected=risk_profile.get("mae_median", 0),
                        recovery_expected=risk_profile.get("recovery_median", 0),
                        risk_score=risk_profile.get("risk_score", 0),
                    )

                    # Open position for MAE tracking
                    active_position = self.position_manager.open_position(
                        entry_price=fill_price,
                        direction=direction,
                        target_bps=self.target_bps,
                    )
                else:
                    # Track filter reasons
                    entries_filtered[reason] = entries_filtered.get(reason, 0) + 1

        # Close remaining trade
        if active_trade is not None and active_position is not None:
            last_bar = ohlcv_df.iloc[-1]
            exit_price = last_bar["close"]

            if active_trade.direction == "LONG":
                fill_price = exit_price * (1 - self.slippage_pct)
            else:
                fill_price = exit_price * (1 + self.slippage_pct)

            pnl_bps = self.position_manager.calculate_pnl_bps(
                active_position, fill_price, self.fee_bps
            )
            pnl_pct = pnl_bps / 10000
            pnl = active_trade.position_size * pnl_pct

            active_trade.exit_time = ohlcv_df.index[-1]
            active_trade.exit_price = fill_price
            active_trade.exit_reason = "FORCED"
            active_trade.pnl = pnl
            active_trade.pnl_pct = pnl_pct
            active_trade.pnl_bps = pnl_bps
            active_trade.bars_held = active_position.bars_in_trade

            trades.append(active_trade)

        timings["5. Walk-forward simulation"] = time.time() - t0
        t0 = time.time()

        # Print summary
        if self.verbose:
            print()
            print(f"  Signals Checked: {signals_checked:,}")
            print(f"  Entries Allowed: {entries_allowed} ({entries_allowed/max(1,signals_checked)*100:.1f}%)")
            print(f"  Trades Executed: {len(trades)}")
            print()
            if entries_filtered:
                print("  Filter Reasons:")
                for reason, count in sorted(entries_filtered.items(), key=lambda x: -x[1]):
                    print(f"    {reason}: {count:,}")
                print()

        # Convert to standard trades for metrics
        standard_trades = [t.to_standard_trade() for t in trades]

        # Calculate metrics
        result = calculate_metrics(
            trades=standard_trades,
            capital=self.capital,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            pair=pair,
        )

        timings["6. Calculate metrics"] = time.time() - t0

        # Print timing
        if self.verbose:
            self._print_timing(timings)

        # Store additional info
        result.pipeline_timing = timings
        result.risk_trades = trades  # Keep detailed trades

        return result

    def _print_timing(self, timings: Dict[str, float]):
        """Print timing summary."""
        total = sum(timings.values())
        print()
        print("-" * 70)
        print("  PIPELINE TIMING")
        print("-" * 70)
        for step, elapsed in timings.items():
            pct = (elapsed / total * 100) if total > 0 else 0
            print(f"  {step:35} {elapsed:8.2f}s  ({pct:5.1f}%)")
        print("-" * 70)
        print(f"  {'TOTAL':35} {total:8.2f}s  (100.0%)")
        print("-" * 70)


def print_risk_backtest_report(result: BacktestResult) -> None:
    """Print formatted risk backtest report with case analysis."""

    print()
    print("=" * 70)
    print("               RISK-BASED BACKTEST REPORT")
    print("=" * 70)
    print()

    # Period info
    test_days = (result.test_end - result.test_start).days
    print(f"  Pair: {result.pair}")
    print(f"  Test Period: {result.test_start.date()} to {result.test_end.date()} ({test_days} days)")
    print(f"  Starting Capital: ${result.capital:,.2f}")
    print()

    # Trade summary
    print("-" * 70)
    print("  TRADE SUMMARY")
    print("-" * 70)
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Winning Trades:    {result.winning_trades} ({result.win_rate*100:.1f}%)")
    print(f"  Losing Trades:     {result.losing_trades} ({(1-result.win_rate)*100:.1f}%)")
    print()

    # Performance
    print("-" * 70)
    print("  PERFORMANCE")
    print("-" * 70)
    pnl_sign = "+" if result.total_pnl >= 0 else ""
    print(f"  Total P&L:         {pnl_sign}${result.total_pnl:,.2f} ({pnl_sign}{result.total_pnl_pct*100:.2f}%)")
    print(f"  Avg Win:           ${result.avg_win:,.2f}")
    print(f"  Avg Loss:          ${result.avg_loss:,.2f}")
    if result.profit_factor != float('inf'):
        print(f"  Profit Factor:     {result.profit_factor:.2f}")
    else:
        print(f"  Profit Factor:     Inf (no losses)")
    print(f"  Expectancy:        ${result.expectancy:,.2f} per trade")
    print()

    # Risk metrics
    print("-" * 70)
    print("  RISK METRICS")
    print("-" * 70)
    print(f"  Max Drawdown:      ${result.max_drawdown:,.2f} ({result.max_drawdown_pct*100:.2f}%)")
    if result.sharpe_ratio is not None:
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    if result.sortino_ratio is not None:
        print(f"  Sortino Ratio:     {result.sortino_ratio:.2f}")
    print()

    # Exit reasons (specific to risk model)
    if result.trades_by_exit:
        print("-" * 70)
        print("  EXIT REASONS (Risk Model)")
        print("-" * 70)
        for reason, data in sorted(result.trades_by_exit.items()):
            pnl_sign = "+" if data["pnl"] >= 0 else ""
            win_rate = data.get("wins", 0) / data["count"] * 100 if data["count"] > 0 else 0
            print(f"  {reason:12} {data['count']:4} trades  {win_rate:5.1f}% win  {pnl_sign}${data['pnl']:,.2f}")
        print()

    # Analyze risk trades if available
    if hasattr(result, 'risk_trades') and result.risk_trades:
        print("-" * 70)
        print("  RISK PROFILE ANALYSIS")
        print("-" * 70)

        risk_trades = result.risk_trades
        avg_p_case1 = np.mean([t.p_case1 for t in risk_trades])
        avg_p_recovery = np.mean([t.p_recovery for t in risk_trades])
        avg_mae_expected = np.mean([t.mae_expected for t in risk_trades])
        avg_mae_actual = np.mean([t.live_mae_bps for t in risk_trades])

        print(f"  Avg P(Case1) at entry:   {avg_p_case1:.1%}")
        print(f"  Avg P(Recovery) at entry: {avg_p_recovery:.1%}")
        print(f"  Avg Expected MAE:        {avg_mae_expected:.1f} bps")
        print(f"  Avg Actual MAE:          {avg_mae_actual:.1f} bps")
        print()

    print("=" * 70)


if __name__ == "__main__":
    # Quick test with sample data
    print("Risk Backtester loaded successfully")
    print("Run with: python -m trade_system.backtest.risk_backtester")
