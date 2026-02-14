"""L1R-001: Ground Truth Baseline

QUESTION: What are the actual TRAIN vs OOS trade statistics for V1.3.2,
          and what is the fixed-1x-qty baseline?

OUTPUT: Comprehensive stats that feed all subsequent L1R experiments.
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
from pathlib import Path

from experiments.layer1.lib.trade_loader import load_enriched_trades
from experiments.layer1.lib.binance_math import (
    calc_min_qty, calc_margin, calc_liq_distance_bps, calc_risk_pct,
)
from experiments.layer1.lib.mc_engine import run_mc_fixed_step
from experiments.layer1.lib.metrics import (
    print_trade_stats, print_trade_stats_by_signal, print_mc_comparison,
)
from experiments.layer1.lib.constants import DEFAULT_CAPITAL

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 100)
print("L1R-001: GROUND TRUTH BASELINE")
print("=" * 100)
print()

print("Loading TRAIN trades (2020-2023)...")
train = load_enriched_trades("train")
print(f"  -> {len(train)} trades")

print("Loading OOS trades (2024-2025)...")
oos = load_enriched_trades("oos")
print(f"  -> {len(oos)} trades")
print()


# ============================================================
# PART 1: COMPREHENSIVE STATS — TRAIN vs OOS
# ============================================================
print("=" * 100)
print("PART 1: TRADE STATISTICS — TRAIN vs OOS")
print("=" * 100)
print()

for label, data in [("TRAIN (2020-2023)", train), ("OOS (2024-2025)", oos)]:
    print_trade_stats(data, label=label)

# By signal type
for label, data in [("TRAIN", train), ("OOS", oos)]:
    print(f"  --- {label} by signal type ---")
    print_trade_stats_by_signal(data)


# ============================================================
# PART 2: TRAIN vs OOS SIDE-BY-SIDE
# ============================================================
print("=" * 100)
print("PART 2: TRAIN vs OOS SIDE-BY-SIDE")
print("=" * 100)
print()

def calc_stats(data):
    bps = [t['bps'] for t in data]
    wins = [b for b in bps if b > 0]
    losses = [b for b in bps if b <= 0]
    return {
        'n': len(bps),
        'win_rate': len(wins) / len(bps) * 100,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'worst': min(bps),
        'best': max(bps),
        'p5': np.percentile(bps, 5),
        'p10': np.percentile(bps, 10),
        'total': sum(bps),
        'avg_btc': np.mean([t['btc_price'] for t in data]),
        'payoff': abs(np.mean(wins) / np.mean(losses)) if losses and wins else 0,
        'pf': abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf'),
    }

ts = calc_stats(train)
os_ = calc_stats(oos)

print(f"  {'Metric':>25s} | {'TRAIN':>15s} | {'OOS':>15s} | {'Ratio':>10s}")
print(f"  {'-'*75}")
print(f"  {'Trades':>25s} | {ts['n']:>15d} | {os_['n']:>15d} | {os_['n']/ts['n']:>10.2f}")
print(f"  {'Win rate':>25s} | {ts['win_rate']:>14.1f}% | {os_['win_rate']:>14.1f}% |")
print(f"  {'Avg win (bps)':>25s} | {ts['avg_win']:>+14.1f} | {os_['avg_win']:>+14.1f} |")
print(f"  {'Avg loss (bps)':>25s} | {ts['avg_loss']:>14.1f} | {os_['avg_loss']:>14.1f} |")
print(f"  {'Payoff ratio':>25s} | {ts['payoff']:>15.2f} | {os_['payoff']:>15.2f} |")
print(f"  {'Profit Factor':>25s} | {ts['pf']:>15.2f} | {os_['pf']:>15.2f} |")
print(f"  {'Total bps':>25s} | {ts['total']:>+14.0f} | {os_['total']:>+14.0f} |")
print(f"  {'Best (bps)':>25s} | {ts['best']:>+14.1f} | {os_['best']:>+14.1f} |")
print(f"  {'Worst (bps)':>25s} | {ts['worst']:>14.1f} | {os_['worst']:>14.1f} | {os_['worst']/ts['worst']:>10.2f}")
print(f"  {'P5 (bps)':>25s} | {ts['p5']:>14.1f} | {os_['p5']:>14.1f} |")
print(f"  {'P10 (bps)':>25s} | {ts['p10']:>14.1f} | {os_['p10']:>14.1f} |")
print(f"  {'Avg BTC price':>25s} | ${ts['avg_btc']:>13,.0f} | ${os_['avg_btc']:>13,.0f} |")
print()

# KEY FINDING: train worst loss
print(f"  KEY: Train worst loss ({ts['worst']:.1f} bps) is {abs(ts['worst']/os_['worst']):.1f}x OOS worst ({os_['worst']:.1f} bps)")
print()


# ============================================================
# PART 3: LIQUIDATION DISTANCES AT DIFFERENT BTC PRICES
# ============================================================
print("=" * 100)
print("PART 3: LIQUIDATION DISTANCES & MINIMUM POSITION AT DIFFERENT BTC PRICES")
print("=" * 100)
print()

btc_prices = [60_000, 80_000, 97_000, 100_000, 120_000]

print(f"  {'BTC Price':>12s} | {'Min Qty':>8s} | {'Min Pos':>10s} | {'Margin':>10s} | {'Liq Dist @$10':>15s} | {'Liq Dist @$50':>15s}")
print(f"  {'-'*85}")

for price in btc_prices:
    mq = calc_min_qty(price)
    pos = mq * price
    margin = calc_margin(mq, price)
    ld_10 = calc_liq_distance_bps(10.0, mq, price)
    ld_50 = calc_liq_distance_bps(50.0, mq, price)
    print(f"  ${price:>10,d} | {mq:>8.3f} | ${pos:>8,.0f} | ${margin:>8.2f} | {ld_10:>13.0f} bps | {ld_50:>13.0f} bps")

print()


# ============================================================
# PART 4: RISK PER TRADE AT DIFFERENT WALLET SIZES
# ============================================================
print("=" * 100)
print("PART 4: RISK PER TRADE (worst-case loss as % of wallet)")
print("=" * 100)
print()

wallets = [5, 10, 15, 20, 30, 50, 100, 500, 1000]
train_worst = abs(ts['worst'])
oos_worst = abs(os_['worst'])

print(f"  Using TRAIN worst loss: {train_worst:.1f} bps (the conservative number)")
print()
print(f"  {'Wallet':>10s} |", end="")
for price in btc_prices:
    print(f" BTC ${price/1000:.0f}K", end=" |")
print()
print(f"  {'-'*70}")

for w in wallets:
    print(f"  ${w:>8d} |", end="")
    for price in btc_prices:
        mq = calc_min_qty(price)
        risk = calc_risk_pct(w, mq, price, train_worst)
        cell = f" {risk*100:>6.1f}%"
        if risk > 0.5:
            cell += " !!"
        elif risk > 0.2:
            cell += " !"
        print(f"{cell:>11s} |", end="")
    print()

print()
print("  !! = >50% of wallet at risk (DANGEROUS)")
print("  !  = >20% of wallet at risk (HIGH)")
print()


# ============================================================
# PART 5: FIXED 1x QTY BASELINE (MC)
# ============================================================
print("=" * 100)
print("PART 5: FIXED 1x QTY BASELINE (0.001 BTC, safest possible)")
print("=" * 100)
print()

# Fixed 1x means $/step = very high (so floor(wallet/step) always = 1 -> 0.001 BTC)
# We use $/step = 999 to ensure fixed 1x
from experiments.layer1.lib.mc_engine import run_mc, MCResult
from experiments.layer1.lib.constants import STEP_SIZE

def fixed_1x_fn(wallet, trade, stats):
    """Always trade minimum qty (0.001 BTC)."""
    return trade['qty_min'], "FIXED_1X"

train_1x = run_mc(train, fixed_1x_fn, {})
oos_1x = run_mc(oos, fixed_1x_fn, {})

mc_results = {
    "TRAIN fixed 1x": train_1x,
    "OOS fixed 1x": oos_1x,
}
print_mc_comparison(mc_results, title="Fixed 1x (0.001 BTC) baseline")


# ============================================================
# PART 6: BUILD StrategyStats OBJECT
# ============================================================
print("=" * 100)
print("PART 6: StrategyStats (JSON) — feeds all subsequent experiments")
print("=" * 100)
print()

def build_strategy_stats(data, label):
    bps = [t['bps'] for t in data]
    wins = [b for b in bps if b > 0]
    losses = [b for b in bps if b <= 0]

    # Classic Kelly: f = p - q/b
    p = len(wins) / len(bps)
    q = 1 - p
    b = abs(np.mean(wins) / np.mean(losses)) if losses and wins else 1
    kelly = p - q / b if b > 0 else 0

    # Per signal type
    by_signal = {}
    for st in sorted(set(t['signal_type'] for t in data)):
        st_trades = [t for t in data if t['signal_type'] == st]
        st_bps = [t['bps'] for t in st_trades]
        st_wins = [b for b in st_bps if b > 0]
        st_losses = [b for b in st_bps if b <= 0]
        by_signal[st] = {
            'n': len(st_bps),
            'win_rate': len(st_wins) / len(st_bps) * 100 if st_bps else 0,
            'avg_win': float(np.mean(st_wins)) if st_wins else 0,
            'avg_loss': float(np.mean(st_losses)) if st_losses else 0,
            'worst_loss': float(min(st_bps)),
            'total': float(sum(st_bps)),
        }

    stats = {
        'label': label,
        'n_trades': len(bps),
        'win_rate': p * 100,
        'avg_win_bps': float(np.mean(wins)) if wins else 0,
        'avg_loss_bps': float(np.mean(losses)) if losses else 0,
        'worst_loss_bps': float(min(bps)),
        'best_win_bps': float(max(bps)),
        'p5_bps': float(np.percentile(bps, 5)),
        'p10_bps': float(np.percentile(bps, 10)),
        'p90_bps': float(np.percentile(bps, 90)),
        'p95_bps': float(np.percentile(bps, 95)),
        'total_bps': float(sum(bps)),
        'kelly_fraction': float(kelly),
        'payoff_ratio': float(b),
        'profit_factor': float(abs(sum(wins) / sum(losses))) if sum(losses) != 0 else float('inf'),
        'avg_btc_price': float(np.mean([t['btc_price'] for t in data])),
        'by_signal': by_signal,
    }
    return stats

train_stats = build_strategy_stats(train, "TRAIN (2020-2023)")
oos_stats = build_strategy_stats(oos, "OOS (2024-2025)")

# Print key stats
for label, stats in [("TRAIN", train_stats), ("OOS", oos_stats)]:
    print(f"  {label}:")
    print(f"    Trades: {stats['n_trades']}, Win: {stats['win_rate']:.1f}%, PF: {stats['profit_factor']:.2f}")
    print(f"    Kelly: {stats['kelly_fraction']:.4f}")
    print(f"    Worst: {stats['worst_loss_bps']:.1f}, P5: {stats['p5_bps']:.1f}, P10: {stats['p10_bps']:.1f}")
    print(f"    Avg BTC: ${stats['avg_btc_price']:,.0f}")
    print()

# Save to JSON
output_dir = Path("experiments/layer1/L1R-001")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "train_stats.json", "w") as f:
    json.dump(train_stats, f, indent=2)
with open(output_dir / "oos_stats.json", "w") as f:
    json.dump(oos_stats, f, indent=2)

print(f"  Saved: {output_dir}/train_stats.json")
print(f"  Saved: {output_dir}/oos_stats.json")
print()

# ============================================================
# PART 7: VERDICT
# ============================================================
print("=" * 100)
print("PART 7: KEY FINDINGS")
print("=" * 100)
print()
print(f"  1. TRAIN worst loss: {train_stats['worst_loss_bps']:.1f} bps (vs OOS worst: {oos_stats['worst_loss_bps']:.1f} bps)")
print(f"     -> Train is {abs(train_stats['worst_loss_bps']/oos_stats['worst_loss_bps']):.1f}x worse than OOS")
print()
print(f"  2. TRAIN Kelly: {train_stats['kelly_fraction']:.4f} (vs OOS: {oos_stats['kelly_fraction']:.4f})")
print()
print(f"  3. At $10 wallet + BTC $97K:")
mq = calc_min_qty(97000)
risk_train = calc_risk_pct(10, mq, 97000, abs(train_stats['worst_loss_bps']))
risk_oos = calc_risk_pct(10, mq, 97000, abs(oos_stats['worst_loss_bps']))
print(f"     Min position: {mq:.3f} BTC (${mq*97000:,.0f})")
print(f"     TRAIN worst loss risk: {risk_train*100:.1f}% of wallet")
print(f"     OOS worst loss risk: {risk_oos*100:.1f}% of wallet")
print()
print(f"  4. At $10 wallet + BTC $100K (sweet spot):")
mq100 = calc_min_qty(100000)
risk_train100 = calc_risk_pct(10, mq100, 100000, abs(train_stats['worst_loss_bps']))
risk_oos100 = calc_risk_pct(10, mq100, 100000, abs(oos_stats['worst_loss_bps']))
print(f"     Min position: {mq100:.3f} BTC (${mq100*100000:,.0f})")
print(f"     TRAIN worst loss risk: {risk_train100*100:.1f}% of wallet")
print(f"     OOS worst loss risk: {risk_oos100*100:.1f}% of wallet")
print()
print(f"  5. Fixed 1x baseline:")
print(f"     TRAIN: median ${train_1x.median:,.0f} | OOS: median ${oos_1x.median:,.0f}")
print()
