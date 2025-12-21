import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from .trade_simulator import Trade


@dataclass
class BacktestResult:
    """Complete backtest results with all metrics."""

    # Trade list
    trades: List[Trade]

    # Period info
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    pair: str
    capital: float

    # Summary metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]

    # Trade analysis
    avg_trade_duration: float
    avg_bars_to_win: float
    avg_bars_to_loss: float

    # By exit reason
    trades_by_exit: dict

    # By regime
    trades_by_regime: dict


def calculate_metrics(
    trades: List[Trade],
    capital: float,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    pair: str
) -> BacktestResult:
    """Calculate all performance metrics from trade list."""

    if len(trades) == 0:
        return BacktestResult(
            trades=trades,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            pair=pair,
            capital=capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            avg_trade_duration=0.0,
            avg_bars_to_win=0.0,
            avg_bars_to_loss=0.0,
            trades_by_exit={},
            trades_by_regime={},
        )

    # Basic counts
    pnls = [t.pnl for t in trades if t.pnl is not None]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]

    total_trades = len(pnls)
    winning_trades = len(winning)
    losing_trades = len(losing)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # P&L metrics
    total_pnl = sum(pnls)
    total_pnl_pct = total_pnl / capital if capital > 0 else 0.0

    avg_win = np.mean(winning) if winning else 0.0
    avg_loss = np.mean(losing) if losing else 0.0

    gross_profit = sum(winning) if winning else 0.0
    gross_loss = abs(sum(losing)) if losing else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    expectancy = np.mean(pnls) if pnls else 0.0

    # Risk metrics
    equity_curve = _build_equity_curve(trades, capital)
    max_drawdown, max_drawdown_pct = _calculate_drawdown(equity_curve, capital)
    sharpe_ratio = _calculate_sharpe(pnls, capital)
    sortino_ratio = _calculate_sortino(pnls, capital)

    # Trade duration
    durations = [t.bars_held for t in trades if t.exit_time is not None]
    avg_trade_duration = np.mean(durations) if durations else 0.0

    win_durations = [t.bars_held for t in trades if t.pnl is not None and t.pnl > 0]
    loss_durations = [t.bars_held for t in trades if t.pnl is not None and t.pnl <= 0]
    avg_bars_to_win = np.mean(win_durations) if win_durations else 0.0
    avg_bars_to_loss = np.mean(loss_durations) if loss_durations else 0.0

    # Group by exit reason
    trades_by_exit = {}
    for t in trades:
        reason = t.exit_reason or "UNKNOWN"
        if reason not in trades_by_exit:
            trades_by_exit[reason] = {"count": 0, "pnl": 0.0}
        trades_by_exit[reason]["count"] += 1
        trades_by_exit[reason]["pnl"] += t.pnl or 0.0

    # Group by regime
    trades_by_regime = {}
    for t in trades:
        regime = t.regime
        if regime not in trades_by_regime:
            trades_by_regime[regime] = {"count": 0, "pnl": 0.0, "wins": 0}
        trades_by_regime[regime]["count"] += 1
        trades_by_regime[regime]["pnl"] += t.pnl or 0.0
        if t.pnl and t.pnl > 0:
            trades_by_regime[regime]["wins"] += 1

    return BacktestResult(
        trades=trades,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        pair=pair,
        capital=capital,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        avg_trade_duration=avg_trade_duration,
        avg_bars_to_win=avg_bars_to_win,
        avg_bars_to_loss=avg_bars_to_loss,
        trades_by_exit=trades_by_exit,
        trades_by_regime=trades_by_regime,
    )


def _build_equity_curve(trades: List[Trade], initial_capital: float) -> List[float]:
    """Build equity curve from trade list."""
    equity = [initial_capital]
    current = initial_capital

    for trade in trades:
        if trade.pnl is not None:
            current += trade.pnl
            equity.append(current)

    return equity


def _calculate_drawdown(equity_curve: List[float], capital: float) -> tuple:
    """Calculate maximum drawdown."""
    if len(equity_curve) < 2:
        return 0.0, 0.0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity

    max_dd = np.max(drawdown)
    max_dd_pct = max_dd / capital if capital > 0 else 0.0

    return float(max_dd), float(max_dd_pct)


def _calculate_sharpe(pnls: List[float], capital: float, risk_free_rate: float = 0.0) -> Optional[float]:
    """Calculate Sharpe ratio (annualized, assuming 1-min bars)."""
    if len(pnls) < 2:
        return None

    returns = [p / capital for p in pnls]
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return None

    # Annualize (assuming ~525,600 minutes per year, but we use per-trade basis)
    # Since trades are irregular, we report per-trade Sharpe
    sharpe = (mean_return - risk_free_rate) / std_return

    return float(sharpe)


def _calculate_sortino(pnls: List[float], capital: float, risk_free_rate: float = 0.0) -> Optional[float]:
    """Calculate Sortino ratio (only considers downside volatility)."""
    if len(pnls) < 2:
        return None

    returns = [p / capital for p in pnls]
    mean_return = np.mean(returns)

    # Downside deviation (only negative returns)
    negative_returns = [r for r in returns if r < 0]
    if not negative_returns:
        return None  # No losses, can't calculate

    downside_std = np.std(negative_returns, ddof=1)

    if downside_std == 0:
        return None

    sortino = (mean_return - risk_free_rate) / downside_std

    return float(sortino)


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convert trade list to DataFrame for analysis/saving."""
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        records.append({
            "signal_time": t.signal_time,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss_price": t.stop_loss_price,
            "take_profit_price": t.take_profit_price,
            "position_size": t.position_size,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
            "regime": t.regime,
            "expectancy": t.expectancy,
        })

    return pd.DataFrame(records)
