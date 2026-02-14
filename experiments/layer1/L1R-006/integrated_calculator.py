"""L1R-006: Integrated Risk Calculator

QUESTION: Does a single calculator using signal quality + wallet + BTC price
          beat all baselines on BOTH train and OOS?

Combines findings from L1R-001 through L1R-005:
- L1R-001: Train stats (worst -865, Kelly 0.33)
- L1R-002: No safe $/step on TRAIN (6.8% ruin floor at $5.50+)
- L1R-003: WEAK signals lose money on OOS, tier sizing boosts OOS 340-1350%
- L1R-004: Survival zone at <$43-84 wallet, $100K BTC sweet spot
- L1R-005: Safety stop at 60% cuts TRAIN ruin from 6.8% to 0.9%
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import json
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from experiments.layer1.lib.trade_loader import load_enriched_trades
from experiments.layer1.lib.binance_math import (
    calc_min_qty, calc_margin, calc_liq_distance_bps, calc_risk_pct, calc_max_qty,
)
from experiments.layer1.lib.mc_engine import run_mc, run_mc_fixed_step, MCResult
from experiments.layer1.lib.metrics import print_mc_comparison
from experiments.layer1.lib.signal_quality import score_signal, SignalQuality
from experiments.layer1.lib.constants import (
    DEFAULT_CAPITAL, STEP_SIZE, LEVERAGE, MAINT_MARGIN_RATE,
    MIN_QTY, MIN_NOTIONAL,
)


# ============================================================
# THE RISK CALCULATOR (version-agnostic)
# ============================================================
@dataclass(frozen=True)
class StrategyStats:
    """Version-agnostic strategy statistics. Recalculate per version."""
    win_rate: float
    avg_win_bps: float
    avg_loss_bps: float
    worst_loss_bps: float       # positive number (e.g., 865)
    p5_bps: float               # positive number
    kelly_fraction: float
    n_trades: int


@dataclass
class SizingDecision:
    qty: float
    position_usd: float
    margin_usd: float
    risk_pct: float             # worst-case loss as % of wallet
    risk_dollar: float          # worst-case loss in dollars
    safety_stop_bps: float
    signal_quality: SignalQuality
    base_step: float
    adjusted_step: float
    reasoning: list = field(default_factory=list)


class RiskCalculator:
    """Intelligent, transparent, version-agnostic risk calculator.

    No dumb modes. Every decision has a reasoning chain.
    """

    def __init__(
        self,
        stats: StrategyStats,
        base_step: float = 6.00,          # From L1R-002: safest on TRAIN
        weak_multiplier: float = 2.0,     # From L1R-003: size down 2x
        strong_multiplier: float = 0.7,   # From L1R-003: size up slightly
        safety_pct: float = 0.60,         # From L1R-005: 60% cuts ruin most
    ):
        self.stats = stats
        self.base_step = base_step
        self.weak_mult = weak_multiplier
        self.strong_mult = strong_multiplier
        self.safety_pct = safety_pct

    def calculate(
        self,
        wallet: float,
        btc_price: float,
        trade_conditions: dict,
    ) -> SizingDecision:
        """Calculate position size with full transparency."""
        reasoning = []

        # Step 1: Exchange minimums
        min_qty = calc_min_qty(btc_price)
        min_pos = min_qty * btc_price
        reasoning.append(f"exchange: min_qty={min_qty:.3f} BTC, min_pos=${min_pos:.0f}")

        # Step 2: Signal quality
        quality = score_signal(trade_conditions)
        reasoning.append(f"quality: {quality.tier} ({', '.join(quality.reasons) or 'none'})")

        # Step 3: Adjusted $/step based on quality
        if quality.tier == "WEAK":
            adj_step = self.base_step * self.weak_mult
        elif quality.tier == "STRONG":
            adj_step = self.base_step * self.strong_mult
        else:
            adj_step = self.base_step
        reasoning.append(f"step: base=${self.base_step:.2f} -> adj=${adj_step:.2f}")

        # Step 4: Kelly-optimal qty
        steps = max(1, int(wallet / adj_step))
        kelly_qty = steps * STEP_SIZE
        reasoning.append(f"kelly: {steps} steps -> {kelly_qty:.3f} BTC")

        # Step 5: Apply minimum floor
        qty = max(kelly_qty, min_qty)
        if kelly_qty < min_qty:
            # In survival zone — be honest about it
            risk_at_min = calc_risk_pct(wallet, min_qty, btc_price, self.stats.worst_loss_bps)
            reasoning.append(
                f"SURVIVAL: kelly ({kelly_qty:.3f}) < min ({min_qty:.3f}), "
                f"using min. worst_risk={risk_at_min*100:.1f}%"
            )
        else:
            reasoning.append(f"GROWTH: kelly ({kelly_qty:.3f}) >= min ({min_qty:.3f})")

        # Step 6: Calculate final metrics
        position = qty * btc_price
        margin = calc_margin(qty, btc_price)
        risk_pct = calc_risk_pct(wallet, qty, btc_price, self.stats.worst_loss_bps)
        risk_dollar = position * self.stats.worst_loss_bps / 10000

        # Step 7: Safety stop
        liq_dist = calc_liq_distance_bps(wallet, qty, btc_price)
        safety_bps = liq_dist * self.safety_pct
        reasoning.append(f"safety: liq={liq_dist:.0f}bps, stop={safety_bps:.0f}bps ({self.safety_pct*100:.0f}%)")

        # Step 8: Can we even afford the margin?
        if wallet < margin:
            reasoning.append(f"SKIP: wallet (${wallet:.2f}) < margin (${margin:.2f})")
            qty = 0

        return SizingDecision(
            qty=qty,
            position_usd=position,
            margin_usd=margin,
            risk_pct=risk_pct,
            risk_dollar=risk_dollar,
            safety_stop_bps=safety_bps,
            signal_quality=quality,
            base_step=self.base_step,
            adjusted_step=adj_step,
            reasoning=reasoning,
        )


# ============================================================
# LOAD DATA + BUILD STRATEGY STATS
# ============================================================
print("=" * 100)
print("L1R-006: INTEGRATED RISK CALCULATOR")
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
print(f"  Stats: worst={strategy_stats.worst_loss_bps:.0f}bps, Kelly={strategy_stats.kelly_fraction:.4f}")
print()


# ============================================================
# PART 1: SAMPLE DECISIONS (show reasoning)
# ============================================================
print("=" * 100)
print("PART 1: SAMPLE DECISIONS — Reasoning transparency")
print("=" * 100)
print()

calc = RiskCalculator(strategy_stats)

sample_scenarios = [
    ("$10 wallet, $97K BTC, STRONG signal", 10, 97000,
     {'signal_type': 'V12_SHORT', 'direction': 'SHORT', 'atr_pctl': 80, 'ema_sep': 1.5, 'entry_dow': 3, 'entry_hour': 14}),
    ("$10 wallet, $97K BTC, WEAK signal (Mon LONG)", 10, 97000,
     {'signal_type': 'V12_LONG', 'direction': 'LONG', 'atr_pctl': 15, 'ema_sep': 0.2, 'entry_dow': 0, 'entry_hour': 10}),
    ("$10 wallet, $100K BTC, NORMAL signal", 10, 100000,
     {'signal_type': 'V12_LONG', 'direction': 'LONG', 'atr_pctl': 50, 'ema_sep': 0.8, 'entry_dow': 2, 'entry_hour': 14}),
    ("$50 wallet, $100K BTC, STRONG signal", 50, 100000,
     {'signal_type': 'BEAR_LONG', 'direction': 'LONG', 'atr_pctl': 85, 'ema_sep': 2.0, 'entry_dow': 3, 'entry_hour': 14}),
    ("$100 wallet, $100K BTC, WEAK signal", 100, 100000,
     {'signal_type': 'V12_LONG', 'direction': 'LONG', 'atr_pctl': 8, 'ema_sep': 0.1, 'entry_dow': 0, 'entry_hour': 2}),
    ("$500 wallet, $100K BTC, STRONG signal", 500, 100000,
     {'signal_type': 'V12_SHORT', 'direction': 'SHORT', 'atr_pctl': 92, 'ema_sep': 2.5, 'entry_dow': 4, 'entry_hour': 16}),
]

for label, wallet, btc, conditions in sample_scenarios:
    d = calc.calculate(wallet, btc, conditions)
    print(f"  {label}")
    print(f"    QTY: {d.qty:.3f} BTC (${d.position_usd:,.0f})")
    print(f"    Risk: {d.risk_pct*100:.1f}% (${d.risk_dollar:.2f})")
    print(f"    Quality: {d.signal_quality.tier} | Safety: {d.safety_stop_bps:.0f} bps")
    print(f"    Reasoning:")
    for r in d.reasoning:
        print(f"      {r}")
    print()


# ============================================================
# PART 2: MC COMPARISON — Integrated vs Baselines
# ============================================================
print("=" * 100)
print("PART 2: MC — INTEGRATED vs BASELINES")
print("=" * 100)
print()

# Create sizing functions for MC

def make_integrated_fn(calculator):
    """Sizing fn using the integrated RiskCalculator."""
    def fn(wallet, trade, stats):
        d = calculator.calculate(wallet, trade['btc_price'], trade)
        return d.qty, d.signal_quality.tier
    return fn

# Test different base_step values
configs = {}

for base in [4.00, 5.00, 6.00, 8.00]:
    # Integrated with tier-specific sizing
    calc_i = RiskCalculator(strategy_stats, base_step=base)
    fn_i = make_integrated_fn(calc_i)

    # Baseline: uniform fixed step (no quality adjustment)
    label_uniform = f"Uniform ${base:.0f}"
    label_integrated = f"Integrated ${base:.0f}"

    r_train_u = run_mc_fixed_step(train, base)
    r_oos_u = run_mc_fixed_step(oos, base)
    r_train_i = run_mc(train, fn_i, {})
    r_oos_i = run_mc(oos, fn_i, {})

    configs[label_uniform] = (r_train_u, r_oos_u)
    configs[label_integrated] = (r_train_i, r_oos_i)

print(f"  {'Config':>25s} | {'TRAIN Median':>14s} {'Ruin':>6s} {'AvgDD':>7s} | {'OOS Median':>14s} {'Ruin':>6s} {'AvgDD':>7s}")
print(f"  {'-'*100}")

for label, (tr, oo) in sorted(configs.items()):
    print(f"  {label:>25s} | ${tr.median:>12,.0f} {tr.ruin_pct:5.1f}% {tr.avg_dd*100:5.1f}% | "
          f"${oo.median:>12,.0f} {oo.ruin_pct:5.1f}% {oo.avg_dd*100:5.1f}%")
print()


# ============================================================
# PART 3: SAFETY STOP INTEGRATION
# ============================================================
print("=" * 100)
print("PART 3: INTEGRATED + SAFETY STOP")
print("=" * 100)
print()

# Best integrated config with safety stop capping losses
# From L1R-005: 60% safety cut at $97K cuts ruin from 6.8% to 0.9%

for base, safety in [(6.00, 0.60), (6.00, 0.70), (6.00, 0.80)]:
    calc_s = RiskCalculator(strategy_stats, base_step=base, safety_pct=safety)

    def make_safety_fn(calculator, safety_pct):
        def fn(wallet, trade, stats):
            d = calculator.calculate(wallet, trade['btc_price'], trade)
            qty = d.qty
            if qty <= 0:
                return 0, "SKIP"

            # Cap losses at safety stop level
            # This simulates what happens when the exchange safety stop fires
            liq_dist = calc_liq_distance_bps(wallet, qty, trade['btc_price'])
            safety_bps = liq_dist * safety_pct

            # Modify trade bps if it exceeds safety stop
            # (This is handled in MC by pre-capping the trade data)
            return qty, d.signal_quality.tier
        return fn

    # Cap trade losses at safety stop level
    # Use average BTC price for approximate safety distance
    def cap_at_safety(data, wallet, step, safety_pct):
        """Cap trade losses at safety stop level for MC."""
        capped = []
        for t in data:
            td = dict(t)
            mq = calc_min_qty(t['btc_price'])
            steps = max(1, int(wallet / step))
            qty = max(steps * STEP_SIZE, mq)
            ld = calc_liq_distance_bps(wallet, qty, t['btc_price'])
            safety = ld * safety_pct
            if td['bps'] < -safety:
                td['bps'] = -safety
            capped.append(td)
        return capped

    # Simple approach: cap at average safety distance
    avg_btc = np.mean([t['btc_price'] for t in train])
    avg_mq = calc_min_qty(avg_btc)
    avg_ld = calc_liq_distance_bps(10.0, avg_mq, avg_btc)
    avg_safety = avg_ld * safety

    train_capped = [{**t, 'bps': max(t['bps'], -avg_safety)} for t in train]
    oos_capped = [{**t, 'bps': max(t['bps'], -avg_safety)} for t in oos]

    fn = make_integrated_fn(calc_s)
    r_train = run_mc(train_capped, fn, {})
    r_oos = run_mc(oos_capped, fn, {})

    label = f"Integrated $6 + Safety {safety*100:.0f}%"
    print(f"  {label:>35s} | TRAIN: ${r_train.median:>10,.0f} ruin {r_train.ruin_pct:.1f}% | OOS: ${r_oos.median:>10,.0f} ruin {r_oos.ruin_pct:.1f}%")

print()


# ============================================================
# PART 4: MODE DISTRIBUTION
# ============================================================
print("=" * 100)
print("PART 4: MODE DISTRIBUTION — How often is each tier used?")
print("=" * 100)
print()

calc_final = RiskCalculator(strategy_stats, base_step=6.00)
fn_final = make_integrated_fn(calc_final)

for label, data in [("TRAIN", train), ("OOS", oos)]:
    tiers = {"STRONG": 0, "NORMAL": 0, "WEAK": 0}
    survival = 0
    growth = 0
    for t in data:
        d = calc_final.calculate(10.0, t['btc_price'], t)
        tiers[d.signal_quality.tier] += 1
        has_survival = any("SURVIVAL" in r for r in d.reasoning)
        if has_survival:
            survival += 1
        else:
            growth += 1

    total = len(data)
    print(f"  {label} ({total} trades):")
    for tier, count in tiers.items():
        print(f"    {tier}: {count} ({count/total*100:.0f}%)")
    print(f"    SURVIVAL mode: {survival} ({survival/total*100:.0f}%)")
    print(f"    GROWTH mode: {growth} ({growth/total*100:.0f}%)")
    print()


# ============================================================
# PART 5: DIFFERENT STARTING CAPITALS
# ============================================================
print("=" * 100)
print("PART 5: STARTING CAPITAL SWEEP")
print("=" * 100)
print()

for cap in [5, 10, 20, 50, 100]:
    calc_c = RiskCalculator(strategy_stats, base_step=6.00)
    fn_c = make_integrated_fn(calc_c)

    r_train = run_mc(train, fn_c, {}, capital=cap)
    r_oos = run_mc(oos, fn_c, {}, capital=cap)

    print(f"  ${cap:>5d} start | TRAIN: ${r_train.median:>10,.0f} ruin {r_train.ruin_pct:>5.1f}% DD {r_train.avg_dd*100:.0f}% | "
          f"OOS: ${r_oos.median:>10,.0f} ruin {r_oos.ruin_pct:>5.1f}% DD {r_oos.avg_dd*100:.0f}%")

print()


# ============================================================
# VERDICT
# ============================================================
print("=" * 100)
print("VERDICT")
print("=" * 100)
print()

# Compare integrated $6 vs uniform $6
calc_v = RiskCalculator(strategy_stats, base_step=6.00)
fn_v = make_integrated_fn(calc_v)
r_int_train = run_mc(train, fn_v, {})
r_int_oos = run_mc(oos, fn_v, {})
r_uni_train = run_mc_fixed_step(train, 6.00)
r_uni_oos = run_mc_fixed_step(oos, 6.00)

print(f"  Uniform $6/step:")
print(f"    TRAIN: median ${r_uni_train.median:,.0f}, ruin {r_uni_train.ruin_pct:.1f}%")
print(f"    OOS:   median ${r_uni_oos.median:,.0f}, ruin {r_uni_oos.ruin_pct:.1f}%")
print()
print(f"  Integrated $6/step (quality-aware):")
print(f"    TRAIN: median ${r_int_train.median:,.0f}, ruin {r_int_train.ruin_pct:.1f}%")
print(f"    OOS:   median ${r_int_oos.median:,.0f}, ruin {r_int_oos.ruin_pct:.1f}%")
print()

if r_int_oos.geo_mean > 0 and r_uni_oos.geo_mean > 0:
    oos_improvement = (r_int_oos.geo_mean / r_uni_oos.geo_mean - 1) * 100
    print(f"  OOS GeoMean improvement: {oos_improvement:+.1f}%")

if r_int_train.geo_mean > 0 and r_uni_train.geo_mean > 0:
    train_improvement = (r_int_train.geo_mean / r_uni_train.geo_mean - 1) * 100
    print(f"  TRAIN GeoMean improvement: {train_improvement:+.1f}%")
print()

print("  RISK CALCULATOR SPECIFICATION:")
print(f"    Base $/step: ${calc_v.base_step:.2f} (from L1R-002: TRAIN-safe)")
print(f"    WEAK multiplier: {calc_v.weak_mult}x (size down on bad conditions)")
print(f"    STRONG multiplier: {calc_v.strong_mult}x (size up on good conditions)")
print(f"    Safety stop: {calc_v.safety_pct*100:.0f}% of liq distance")
print(f"    Decisions: transparent reasoning chain for every trade")
print(f"    Version-agnostic: takes StrategyStats as input")
print()
