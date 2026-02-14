"""Standard metrics reporting for all Layer 1 experiments."""
import numpy as np

from .mc_engine import MCResult


def print_trade_stats(trades: list, label: str = ""):
    """Print comprehensive trade statistics."""
    if label:
        print(f"  {label}")
        print(f"  {'=' * len(label)}")

    bps = [t['bps'] for t in trades]
    wins = [b for b in bps if b > 0]
    losses = [b for b in bps if b <= 0]

    print(f"    Trades: {len(bps)}")
    print(f"    Win rate: {len(wins)/len(bps)*100:.1f}%")
    print(f"    Avg win: +{np.mean(wins):.1f} bps" if wins else "    Avg win: N/A")
    print(f"    Avg loss: {np.mean(losses):.1f} bps" if losses else "    Avg loss: N/A")
    print(f"    Total: {sum(bps):+.0f} bps")
    print(f"    Best: +{max(bps):.1f} bps")
    print(f"    Worst: {min(bps):.1f} bps")
    print(f"    P95 loss: {np.percentile(bps, 5):.1f} bps")
    print(f"    P90 loss: {np.percentile(bps, 10):.1f} bps")

    if wins and losses:
        pf = abs(sum(wins)) / abs(sum(losses))
        print(f"    Profit Factor: {pf:.2f}")
    print()


def print_trade_stats_by_signal(trades: list):
    """Print stats broken down by signal type."""
    signal_types = sorted(set(t['signal_type'] for t in trades))
    for st in signal_types:
        st_trades = [t for t in trades if t['signal_type'] == st]
        print_trade_stats(st_trades, label=f"{st} ({len(st_trades)} trades)")


def print_mc_result(result: MCResult, label: str = ""):
    """Print a single MC result."""
    prefix = f"  {label}: " if label else "  "
    print(f"{prefix}Median ${result.median:,.0f} | GeoMean ${result.geo_mean:,.0f} | "
          f"P5 ${result.p5:,.0f} | AvgDD {result.avg_dd*100:.1f}% | "
          f"Ruin {result.ruin_pct:.1f}%")


def print_mc_comparison(results: dict, title: str = ""):
    """Print comparison table of multiple MC results.

    Args:
        results: Dict of {label: MCResult}
        title: Optional title.
    """
    if title:
        print(f"  {title}")
        print()

    print(f"  {'Config':>30s} | {'Median':>14s} | {'GeoMean':>14s} | {'P5':>14s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
    print(f"  {'-'*105}")

    for label, r in results.items():
        print(f"  {label:>30s} | ${r.median:>12,.0f} | ${r.geo_mean:>12,.0f} | "
              f"${r.p5:>12,.0f} | {r.avg_dd*100:5.1f}% | {r.p95_dd*100:5.1f}% | "
              f"{r.ruin_pct:5.1f}%")
    print()


def print_mc_comparison_train_oos(train_results: dict, oos_results: dict, title: str = ""):
    """Print side-by-side TRAIN vs OOS MC comparison.

    Args:
        train_results: Dict of {label: MCResult} for TRAIN
        oos_results: Dict of {label: MCResult} for OOS (same labels)
    """
    if title:
        print(f"  {title}")
        print()

    print(f"  {'Config':>20s} | {'TRAIN Median':>14s} {'Ruin%':>6s} | {'OOS Median':>14s} {'Ruin%':>6s} | {'TRAIN GeoM':>14s} | {'OOS GeoM':>14s}")
    print(f"  {'-'*110}")

    for label in train_results:
        tr = train_results[label]
        oo = oos_results.get(label)
        if oo is None:
            continue
        print(f"  {label:>20s} | ${tr.median:>12,.0f} {tr.ruin_pct:5.1f}% | "
              f"${oo.median:>12,.0f} {oo.ruin_pct:5.1f}% | "
              f"${tr.geo_mean:>12,.0f} | ${oo.geo_mean:>12,.0f}")
    print()
