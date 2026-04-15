"""Pluggable Monte Carlo simulation engine.

Every experiment defines its own sizing_fn and passes it here.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .exchange_constants import LEVERAGE, MAINT_MARGIN_RATE, N_SIMS, DEFAULT_CAPITAL


@dataclass
class MCResult:
    """Standard MC simulation result."""
    median: float = 0.0
    geo_mean: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    avg_dd: float = 0.0
    p95_dd: float = 0.0
    ruin_pct: float = 0.0
    skip_pct: float = 0.0
    n_trades: int = 0
    n_sims: int = 0
    mode_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'median': self.median,
            'geo_mean': self.geo_mean,
            'p5': self.p5,
            'p25': self.p25,
            'p75': self.p75,
            'p95': self.p95,
            'avg_dd': self.avg_dd,
            'p95_dd': self.p95_dd,
            'ruin_pct': self.ruin_pct,
            'skip_pct': self.skip_pct,
        }


# Type for sizing functions:
#   (wallet, trade_dict, stats_dict) -> (qty, mode_label)
SizingFn = Callable[[float, dict, dict], tuple[float, str]]


def _calc_max_dd(equity: list) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def simulate_one_path(
    trades: list,
    sizing_fn: SizingFn,
    stats: dict,
    capital: float = DEFAULT_CAPITAL,
) -> tuple[list, int, int, dict]:
    """Simulate one equity path through shuffled trades.

    Args:
        trades: List of enriched trade dicts.
        sizing_fn: Function(wallet, trade, stats) -> (qty, mode_label)
        stats: Strategy statistics dict (passed to sizing_fn).
        capital: Starting capital.

    Returns:
        (equity_curve, n_skipped, n_liquidated, mode_counts)
    """
    equity = [capital]
    skipped = 0
    liquidated = 0
    mode_counts = {}

    for td in trades:
        eq = equity[-1]
        if eq <= 0.01:
            equity.append(0.01)
            continue

        # Get sizing decision from the experiment's function
        qty, mode = sizing_fn(eq, td, stats)

        # Track modes
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Skip if qty is 0 or negative
        if qty <= 0:
            equity.append(eq)
            skipped += 1
            continue

        position = qty * td['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        # Skip if can't afford margin
        if eq < margin:
            equity.append(eq)
            skipped += 1
            continue

        # Calculate P&L
        pnl = position * (td['bps'] / 10000)
        max_loss = eq - maint  # liquidation threshold

        if pnl < -max_loss:
            # Liquidated
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated, mode_counts


def run_mc(
    trades: list,
    sizing_fn: SizingFn,
    stats: dict,
    n_sims: int = N_SIMS,
    capital: float = DEFAULT_CAPITAL,
    seed: int = 42,
) -> MCResult:
    """Run Monte Carlo simulation with pluggable sizing function.

    Args:
        trades: List of enriched trade dicts.
        sizing_fn: Function(wallet, trade, stats) -> (qty, mode_label)
        stats: Strategy statistics dict (passed to sizing_fn).
        n_sims: Number of MC paths.
        capital: Starting capital.
        seed: Random seed.

    Returns:
        MCResult with standard metrics.
    """
    rng = np.random.default_rng(seed)
    finals = []
    max_dds = []
    ruin_count = 0
    total_skipped = 0
    all_mode_counts = {}

    for _ in range(n_sims):
        shuffled = list(trades)
        rng.shuffle(shuffled)
        eq, skipped, liq, modes = simulate_one_path(shuffled, sizing_fn, stats, capital)
        finals.append(eq[-1])
        max_dds.append(_calc_max_dd(eq))
        total_skipped += skipped
        if eq[-1] < 1.0:
            ruin_count += 1
        for m, c in modes.items():
            all_mode_counts[m] = all_mode_counts.get(m, 0) + c

    finals_arr = np.array(finals)
    dds_arr = np.array(max_dds)

    return MCResult(
        median=float(np.median(finals_arr)),
        geo_mean=float(np.exp(np.mean(np.log(np.maximum(finals_arr, 0.01))))),
        p5=float(np.percentile(finals_arr, 5)),
        p25=float(np.percentile(finals_arr, 25)),
        p75=float(np.percentile(finals_arr, 75)),
        p95=float(np.percentile(finals_arr, 95)),
        avg_dd=float(np.mean(dds_arr)),
        p95_dd=float(np.percentile(dds_arr, 95)),
        ruin_pct=ruin_count / n_sims * 100,
        skip_pct=total_skipped / (n_sims * len(trades)) * 100 if trades else 0,
        n_trades=len(trades),
        n_sims=n_sims,
        mode_counts=all_mode_counts,
    )


def run_mc_fixed_step(
    trades: list,
    dollars_per_step: float,
    n_sims: int = N_SIMS,
    capital: float = DEFAULT_CAPITAL,
    seed: int = 42,
) -> MCResult:
    """Convenience wrapper: MC with fixed $/step sizing.

    qty = floor(wallet / dollars_per_step) * 0.001
    """
    from .exchange_constants import STEP_SIZE

    def fixed_step_fn(wallet: float, trade: dict, stats: dict) -> tuple[float, str]:
        steps = max(1, int(wallet / dollars_per_step))
        qty = steps * STEP_SIZE
        qty = max(qty, trade['qty_min'])
        return qty, "FIXED"

    return run_mc(trades, fixed_step_fn, {}, n_sims, capital, seed)
