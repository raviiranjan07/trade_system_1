"""L1R-007: Final Validation + Bot Module

QUESTION: Does the complete risk management system work end-to-end with
          real dollar equity tracking (not just MC)?

Runs a SEQUENTIAL backtest (not shuffled) through actual chronological trades
using the RiskCalculator, tracking real dollar P&L with compounding.
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
from pathlib import Path

from experiments.layer1.lib.trade_loader import load_enriched_trades
from experiments.layer1.lib.binance_math import (
    calc_min_qty, calc_margin, calc_liq_distance_bps,
)
from experiments.layer1.lib.mc_engine import run_mc
from experiments.layer1.lib.constants import (
    DEFAULT_CAPITAL, STEP_SIZE, LEVERAGE, MAINT_MARGIN_RATE,
)

# Import the RiskCalculator from L1R-006
# Can't import directly (hyphen in dir name), use importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "integrated_calculator",
    "experiments/layer1/L1R-006/integrated_calculator.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
RiskCalculator = mod.RiskCalculator
StrategyStats = mod.StrategyStats
SizingDecision = mod.SizingDecision


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 100)
print("L1R-007: FINAL VALIDATION + BOT MODULE")
print("=" * 100)
print()

print("Loading trades...")
train = load_enriched_trades("train")
oos = load_enriched_trades("oos")
print(f"  TRAIN: {len(train)} trades | OOS: {len(oos)} trades")

stats_path = Path("experiments/layer1/L1R-001")
with open(stats_path / "train_stats.json") as f:
    ts = json.load(f)

strategy_stats = StrategyStats(
    win_rate=ts['win_rate'],
    avg_win_bps=ts['avg_win_bps'],
    avg_loss_bps=abs(ts['avg_loss_bps']),
    worst_loss_bps=abs(ts['worst_loss_bps']),
    p5_bps=abs(ts['p5_bps']),
    kelly_fraction=ts['kelly_fraction'],
    n_trades=ts['n_trades'],
)

calc = RiskCalculator(strategy_stats, base_step=6.00, safety_pct=0.60)
print(f"  Calculator: base=$6.00, safety=60%")
print()


# ============================================================
# PART 1: SEQUENTIAL BACKTEST (chronological order, real $ P&L)
# ============================================================
print("=" * 100)
print("PART 1: SEQUENTIAL BACKTEST - Real dollar equity")
print("=" * 100)
print()

def run_sequential_backtest(trades, calculator, starting_capital=10.0):
    """Run trades in order with real $ P&L tracking."""
    wallet = starting_capital
    equity_curve = [wallet]
    trade_log = []
    peak = wallet
    max_dd = 0

    for i, t in enumerate(trades):
        d = calculator.calculate(wallet, t['btc_price'], t)

        if d.qty <= 0:
            equity_curve.append(wallet)
            trade_log.append({
                'trade': i + 1,
                'action': 'SKIP',
                'wallet': wallet,
                'reason': 'insufficient margin',
            })
            continue

        # Calculate P&L
        position = d.qty * t['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        # Apply safety stop: cap loss at safety_stop_bps
        trade_bps = t['bps']
        safety_bps = d.safety_stop_bps
        safety_triggered = False
        if trade_bps < -safety_bps:
            trade_bps = -safety_bps
            safety_triggered = True

        pnl = position * (trade_bps / 10000)

        # Check liquidation
        liquidated = False
        if pnl < -(wallet - maint):
            pnl = -(wallet - 0.01)
            liquidated = True

        wallet = max(wallet + pnl, 0.01)
        equity_curve.append(wallet)

        # Track drawdown
        if wallet > peak:
            peak = wallet
        dd = (peak - wallet) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        trade_log.append({
            'trade': i + 1,
            'qty': d.qty,
            'position': position,
            'pnl': pnl,
            'bps': t['bps'],
            'capped_bps': trade_bps,
            'wallet': wallet,
            'risk_pct': d.risk_pct,
            'quality': d.signal_quality.tier,
            'safety_triggered': safety_triggered,
            'liquidated': liquidated,
            'reasoning': d.reasoning,
        })

    return equity_curve, trade_log, max_dd

# Run on TRAIN
train_eq, train_log, train_dd = run_sequential_backtest(train, calc, 10.0)
print(f"  TRAIN (chronological, $10 start):")
print(f"    Final wallet: ${train_eq[-1]:,.2f}")
print(f"    Max drawdown: {train_dd*100:.1f}%")
print(f"    Trades: {len(train)}")
safety_hits = sum(1 for t in train_log if t.get('safety_triggered'))
liquidations = sum(1 for t in train_log if t.get('liquidated'))
skips = sum(1 for t in train_log if t.get('action') == 'SKIP')
print(f"    Safety stops: {safety_hits}")
print(f"    Liquidations: {liquidations}")
print(f"    Skipped: {skips}")
print()

# Run on OOS
oos_eq, oos_log, oos_dd = run_sequential_backtest(oos, calc, 10.0)
print(f"  OOS (chronological, $10 start):")
print(f"    Final wallet: ${oos_eq[-1]:,.2f}")
print(f"    Max drawdown: {oos_dd*100:.1f}%")
print(f"    Trades: {len(oos)}")
safety_hits = sum(1 for t in oos_log if t.get('safety_triggered'))
liquidations = sum(1 for t in oos_log if t.get('liquidated'))
skips = sum(1 for t in oos_log if t.get('action') == 'SKIP')
print(f"    Safety stops: {safety_hits}")
print(f"    Liquidations: {liquidations}")
print(f"    Skipped: {skips}")
print()


# ============================================================
# PART 2: REASONING LOG SAMPLE
# ============================================================
print("=" * 100)
print("PART 2: REASONING LOG - First 5 OOS trades")
print("=" * 100)
print()

for t in oos_log[:5]:
    if t.get('action') == 'SKIP':
        print(f"  Trade {t['trade']}: SKIP ({t['reason']})")
        continue
    print(f"  Trade {t['trade']}: {t['quality']} | qty={t['qty']:.3f} | "
          f"pos=${t['position']:,.0f} | bps={t['bps']:+.1f} -> {t['capped_bps']:+.1f} | "
          f"pnl=${t['pnl']:+.2f} | wallet=${t['wallet']:,.2f} | risk={t['risk_pct']*100:.1f}%")
    for r in t['reasoning']:
        print(f"    {r}")
    print()


# ============================================================
# PART 3: MC CONFIRMATION (should match L1R-006)
# ============================================================
print("=" * 100)
print("PART 3: MC CONFIRMATION - Does MC match sequential?")
print("=" * 100)
print()

def make_integrated_fn(calculator):
    def fn(wallet, trade, stats):
        d = calculator.calculate(wallet, trade['btc_price'], trade)
        return d.qty, d.signal_quality.tier
    return fn

fn = make_integrated_fn(calc)
mc_train = run_mc(train, fn, {})
mc_oos = run_mc(oos, fn, {})

print(f"  {'':>20s} | {'Sequential':>14s} | {'MC Median':>14s} | {'MC P5':>14s} | {'MC Ruin%':>8s}")
print(f"  {'-'*80}")
print(f"  {'TRAIN':>20s} | ${train_eq[-1]:>12,.2f} | ${mc_train.median:>12,.0f} | ${mc_train.p5:>12,.0f} | {mc_train.ruin_pct:>6.1f}%")
print(f"  {'OOS':>20s} | ${oos_eq[-1]:>12,.2f} | ${mc_oos.median:>12,.0f} | ${mc_oos.p5:>12,.0f} | {mc_oos.ruin_pct:>6.1f}%")
print()
print("  NOTE: Sequential uses actual trade order; MC shuffles randomly.")
print("        MC median is the expected value across all orderings.")
print()


# ============================================================
# PART 4: EQUITY CURVE MILESTONES
# ============================================================
print("=" * 100)
print("PART 4: EQUITY MILESTONES")
print("=" * 100)
print()

for label, eq_curve in [("TRAIN", train_eq), ("OOS", oos_eq)]:
    print(f"  {label}:")
    milestones = [20, 50, 100, 500, 1000, 5000, 10000]
    for m in milestones:
        reached = next((i for i, e in enumerate(eq_curve) if e >= m), None)
        if reached:
            print(f"    ${m:>6,d}: reached at trade {reached}")
        else:
            print(f"    ${m:>6,d}: NOT reached (max was ${max(eq_curve):,.2f})")
    print()


# ============================================================
# PART 5: FULL SPECIFICATION SUMMARY
# ============================================================
print("=" * 100)
print("PART 5: COMPLETE RISK MANAGEMENT SPECIFICATION")
print("=" * 100)
print()
print("  VERSION-AGNOSTIC RISK CALCULATOR")
print("  =================================")
print()
print("  INPUT (per strategy version):")
print(f"    StrategyStats: win_rate, avg_win, avg_loss, worst_loss, p5, kelly, n_trades")
print(f"    Current V1.3.2 TRAIN stats: worst={strategy_stats.worst_loss_bps:.0f}bps, "
      f"Kelly={strategy_stats.kelly_fraction:.4f}")
print()
print("  PARAMETERS:")
print(f"    Base $/step: $6.00 (L1R-002: safest on TRAIN)")
print(f"    WEAK multiplier: 2.0x (L1R-003: validated bad conditions)")
print(f"    STRONG multiplier: 0.7x (L1R-003: validated strong conditions)")
print(f"    Safety stop: 60% of liq distance (L1R-005: cuts ruin from 6.8% to 0.9%)")
print()
print("  SIGNAL QUALITY (from EXP-004, validated train->OOS):")
print("    WEAK conditions: monday_long, low_atr(<20), low_ema(<0.3), v12_long_monday")
print("    STRONG conditions: high_atr(>70), high_ema(>1.0)")
print()
print("  DECISION FLOW:")
print("    1. Calculate exchange minimum qty")
print("    2. Score signal quality (STRONG/NORMAL/WEAK)")
print("    3. Adjust $/step by quality tier")
print("    4. Calculate Kelly-optimal qty from adjusted step")
print("    5. Apply floor (min qty) - honest about survival mode")
print("    6. Calculate safety stop from liq distance")
print("    7. Output: qty, position, risk%, safety stop, full reasoning")
print()
print("  BOT INTEGRATION POINT: src/engine/bot.py")
print("    Before open_position(): decision = calculator.calculate(wallet, btc_price, conditions)")
print("    Place safety stop on exchange: decision.safety_stop_bps")
print("    Log reasoning: decision.reasoning")
print()
print("  PERFORMANCE:")
print(f"    TRAIN sequential ($10 start): ${train_eq[-1]:,.2f}")
print(f"    OOS sequential ($10 start):   ${oos_eq[-1]:,.2f}")
print(f"    MC TRAIN (median):             ${mc_train.median:,.0f} (ruin {mc_train.ruin_pct:.1f}%)")
print(f"    MC OOS (median):               ${mc_oos.median:,.0f} (ruin {mc_oos.ruin_pct:.1f}%)")
print()
