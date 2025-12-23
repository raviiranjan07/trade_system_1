import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .trade_simulator import Trade, TradeSimulator
from .metrics import BacktestResult, calculate_metrics


class Backtester:
    """
    Walk-forward backtester for the trading system.

    Splits data into training and test periods, then walks forward through
    the test period making trading decisions based only on historical data.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0004,
        max_bars_in_trade: int = 120,
        capital: float = 10000,
        risk_per_trade: float = 0.005,
        k: int = 200,
        horizon: int = 30,
        verbose: bool = True,
        sample_interval: int = 1,  # Check for signals every N bars (1 = every bar)
        # Similarity engine backend settings
        similarity_backend: str = "bruteforce",  # "bruteforce" or "faiss"
        faiss_nlist: int = 100,
        faiss_nprobe: int = 10,
        # Decision engine settings
        min_expectancy: float = 0.0,
        max_distance: float = 1.5,
        blocked_regimes: list = None,
        # Trailing stop settings
        trailing_stop_pct: float = 0.0,
        trailing_stop_activation_pct: float = 0.0
    ):
        self.train_ratio = train_ratio
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.k = k
        self.horizon = horizon
        self.verbose = verbose
        self.sample_interval = sample_interval
        self.similarity_backend = similarity_backend
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.min_expectancy = min_expectancy
        self.max_distance = max_distance
        self.blocked_regimes = blocked_regimes or ["HIGH_VOL"]
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_stop_activation_pct = trailing_stop_activation_pct

        self.simulator = TradeSimulator(
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            max_bars_in_trade=max_bars_in_trade,
            trailing_stop_pct=trailing_stop_pct,
            trailing_stop_activation_pct=trailing_stop_activation_pct
        )

    def run(
        self,
        outcome_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        pair: str = "UNKNOWN"
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            outcome_df: DataFrame with state vectors and outcome labels (MFE/MAE)
            regime_df: DataFrame with regime labels
            ohlcv_df: Raw OHLCV data for trade simulation
            pair: Trading pair name

        Returns:
            BacktestResult with all trades and metrics
        """
        # Import here to avoid circular imports
        from similarity.similarity_engine import SimilarityEngine
        from decision.decision_engine import DecisionEngine

        # 1. Split data into training and test
        split_idx = int(len(outcome_df) * self.train_ratio)
        train_outcomes = outcome_df.iloc[:split_idx]
        test_outcomes = outcome_df.iloc[split_idx:]

        train_start = train_outcomes.index[0]
        train_end = train_outcomes.index[-1]
        test_start = test_outcomes.index[0]
        test_end = test_outcomes.index[-1]

        if self.verbose:
            print()
            print("=" * 70)
            print("                        BACKTESTING")
            print("=" * 70)
            print(f"  Training Period: {train_start} to {train_end}")
            print(f"  Training Samples: {len(train_outcomes):,}")
            print(f"  Test Period: {test_start} to {test_end}")
            print(f"  Test Samples: {len(test_outcomes):,}")
            if self.sample_interval > 1:
                print(f"  Signal Check Interval: Every {self.sample_interval} bars")
                print(f"  Signal Checks: ~{len(test_outcomes) // self.sample_interval:,}")
            print(f"  Similarity Backend: {self.similarity_backend}")
            print("=" * 70)
            print()

        # 2. Build similarity engine on TRAINING data only
        similarity = SimilarityEngine(
            outcome_df=train_outcomes,
            regime_df=regime_df,
            k=self.k,
            backend=self.similarity_backend,
            faiss_nlist=self.faiss_nlist,
            faiss_nprobe=self.faiss_nprobe
        )

        # 3. Initialize decision engine
        decision_engine = DecisionEngine(
            capital=self.capital,
            risk_per_trade=self.risk_per_trade,
            min_expectancy=self.min_expectancy,
            max_distance=self.max_distance,
            blocked_regimes=self.blocked_regimes
        )

        # 4. Walk forward through test period
        trades: List[Trade] = []
        active_trade: Optional[Trade] = None
        signals_generated = 0
        no_trade_reasons = {}
        bar_counter = 0

        # Progress bar for test period
        iterator = test_outcomes.iterrows()
        if self.verbose:
            iterator = tqdm(
                list(iterator),
                desc="Backtesting",
                unit="bars"
            )

        for timestamp, state_row in iterator:
            bar_counter += 1
            # Skip if we don't have OHLCV data for this bar
            if timestamp not in ohlcv_df.index:
                continue

            bar = ohlcv_df.loc[timestamp]

            # Update active trade if exists
            if active_trade is not None:
                active_trade = self.simulator.update_trade(
                    active_trade, bar, timestamp
                )

                # Check if trade closed
                if active_trade.exit_time is not None:
                    trades.append(active_trade)
                    active_trade = None

            # Check for new trade signal (only if no active trade)
            # Use sample_interval to reduce computation (skip bars between checks)
            if active_trade is None:
                # Only check for signals at sample intervals
                if bar_counter % self.sample_interval != 0:
                    continue

                # Get regime for this bar
                if timestamp not in regime_df.index:
                    continue
                regime = regime_df.loc[timestamp, "regime"]

                # Query similarity engine with time boundary
                sim_result = similarity.query(
                    current_state=state_row,
                    regime=regime,
                    horizon=self.horizon,
                    max_timestamp=timestamp  # Only use past data!
                )

                # Get trading decision
                decision = decision_engine.decide(sim_result, regime)

                if decision["action"] == "TRADE":
                    signals_generated += 1

                    # Get next bar for entry
                    future_bars = ohlcv_df.loc[timestamp:].iloc[1:]
                    if len(future_bars) == 0:
                        continue

                    next_bar = future_bars.iloc[0]
                    entry_price = next_bar["open"]

                    # Open the trade
                    active_trade = self.simulator.open_trade(
                        decision=decision,
                        signal_time=timestamp,
                        entry_price=entry_price,
                        regime=regime
                    )
                    active_trade.entry_time = future_bars.index[0]
                else:
                    # Track no-trade reasons
                    reason = decision.get("reason", "UNKNOWN")
                    no_trade_reasons[reason] = no_trade_reasons.get(reason, 0) + 1

        # Close any remaining trade at end
        if active_trade is not None:
            last_bar = ohlcv_df.iloc[-1]
            active_trade = self.simulator.force_exit(
                active_trade,
                exit_price=last_bar["close"],
                exit_time=ohlcv_df.index[-1]
            )
            trades.append(active_trade)

        if self.verbose:
            print()
            print(f"  Signals Generated: {signals_generated}")
            print(f"  Trades Executed: {len(trades)}")
            print()
            if no_trade_reasons:
                print("  No-Trade Reasons:")
                for reason, count in sorted(no_trade_reasons.items(), key=lambda x: -x[1]):
                    print(f"    {reason}: {count:,}")
                print()

        # 5. Calculate metrics
        result = calculate_metrics(
            trades=trades,
            capital=self.capital,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            pair=pair
        )

        return result


def print_backtest_report(result: BacktestResult) -> None:
    """Print formatted backtest report."""

    print()
    print("=" * 70)
    print("                         BACKTEST REPORT")
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
    else:
        print(f"  Sharpe Ratio:      N/A")
    if result.sortino_ratio is not None:
        print(f"  Sortino Ratio:     {result.sortino_ratio:.2f}")
    else:
        print(f"  Sortino Ratio:     N/A")
    print()

    # Trade duration
    print("-" * 70)
    print("  TRADE DURATION")
    print("-" * 70)
    print(f"  Avg Duration:      {result.avg_trade_duration:.1f} bars (minutes)")
    print(f"  Avg Bars to Win:   {result.avg_bars_to_win:.1f}")
    print(f"  Avg Bars to Loss:  {result.avg_bars_to_loss:.1f}")
    print()

    # Exit reasons
    if result.trades_by_exit:
        print("-" * 70)
        print("  EXIT REASONS")
        print("-" * 70)
        for reason, data in sorted(result.trades_by_exit.items()):
            pnl_sign = "+" if data["pnl"] >= 0 else ""
            print(f"  {reason:12} {data['count']:4} trades  {pnl_sign}${data['pnl']:,.2f}")
        print()

    # By regime
    if result.trades_by_regime:
        print("-" * 70)
        print("  BY REGIME")
        print("-" * 70)
        for regime, data in sorted(result.trades_by_regime.items()):
            win_rate = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
            pnl_sign = "+" if data["pnl"] >= 0 else ""
            print(f"  {regime:15} {data['count']:4} trades  {win_rate:5.1f}% win  {pnl_sign}${data['pnl']:,.2f}")
        print()

    print("=" * 70)
