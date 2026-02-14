"""Comprehensive tests for the Layer 1 Risk Calculator.

Three test layers:
  Layer 1: Unit tests (math, signal quality, qty stepping)
  Layer 2: End-to-end (hand-verified decisions, L1R-007 comparison)
  Layer 3: Stress tests (losing streaks, worst trades, edge cases)

Run: PYTHONPATH="src;." pytest experiments/layer1/tests/test_risk_calculator.py -v
"""
import sys
sys.path.insert(0, "src")

import math
import json
import pytest
from pathlib import Path

from experiments.layer1.lib.constants import (
    LEVERAGE, MAINT_MARGIN_RATE, MIN_QTY, STEP_SIZE, MIN_NOTIONAL,
)
from experiments.layer1.lib.binance_math import (
    calc_min_qty, calc_margin, calc_liq_distance_bps,
    calc_risk_pct, calc_max_qty,
)
from experiments.layer1.lib.signal_quality import (
    score_signal, SignalQuality,
    is_monday_long, is_low_atr, is_low_ema, is_v12_long_monday,
    is_high_atr, is_high_ema,
)

# Import RiskCalculator via importlib (hyphen in dir name)
# We only need the classes (lines 39-156), not the experiment script below.
# Extract them by reading the source and exec'ing just the class definitions.
import importlib.util
import textwrap

_calc_source = Path("experiments/layer1/L1R-006/integrated_calculator.py").read_text()

# Find the class definitions section (before the experiment script)
_class_end = _calc_source.index("\n# ============================================================\n# LOAD DATA")
_class_source = _calc_source[:_class_end]

# Remove the module-level imports that load data, keep only what classes need
_ns = {
    '__builtins__': __builtins__,
    'math': math,
    'dataclass': __import__('dataclasses').dataclass,
    'field': __import__('dataclasses').field,
    'calc_min_qty': calc_min_qty,
    'calc_margin': calc_margin,
    'calc_liq_distance_bps': calc_liq_distance_bps,
    'calc_risk_pct': calc_risk_pct,
    'calc_max_qty': calc_max_qty,
    'score_signal': score_signal,
    'SignalQuality': SignalQuality,
    'STEP_SIZE': STEP_SIZE,
}
exec(_class_source, _ns)

RiskCalculator = _ns['RiskCalculator']
StrategyStats = _ns['StrategyStats']
SizingDecision = _ns['SizingDecision']


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def train_stats():
    """V1.3.2 TRAIN stats from L1R-001."""
    return StrategyStats(
        win_rate=65.893,
        avg_win_bps=62.035,
        avg_loss_bps=60.617,       # positive
        worst_loss_bps=864.651,    # positive
        p5_bps=136.062,            # positive
        kelly_fraction=0.3257,
        n_trades=431,
    )


@pytest.fixture
def calc(train_stats):
    """Default RiskCalculator with production settings."""
    return RiskCalculator(
        train_stats,
        base_step=6.00,
        weak_multiplier=2.0,
        strong_multiplier=0.7,
        safety_pct=0.60,
    )


def make_conditions(signal_type='V12_SHORT', direction='SHORT',
                    atr_pctl=50, ema_sep=0.8, entry_dow=3, entry_hour=14):
    """Build a trade conditions dict."""
    return {
        'signal_type': signal_type,
        'direction': direction,
        'atr_pctl': atr_pctl,
        'ema_sep': ema_sep,
        'entry_dow': entry_dow,
        'entry_hour': entry_hour,
    }


# ############################################################
# LAYER 1: UNIT TESTS
# ############################################################

class TestBinanceMath:
    """Test Binance position calculation functions."""

    # --- calc_min_qty ---

    def test_min_qty_at_97k(self):
        """At $97K, min notional $100 -> 0.002 BTC (ceil to step)."""
        # $100 / $97000 = 0.001031 -> ceil to 0.002
        qty = calc_min_qty(97000)
        assert qty == 0.002

    def test_min_qty_at_100k(self):
        """At $100K, min notional $100 -> 0.001 BTC (exact)."""
        # $100 / $100000 = 0.001 -> exactly 0.001
        qty = calc_min_qty(100000)
        assert qty == 0.001

    def test_min_qty_at_50k(self):
        """At $50K, min notional $100 -> 0.002 BTC."""
        # $100 / $50000 = 0.002 -> exactly 0.002
        qty = calc_min_qty(50000)
        assert qty == 0.002

    def test_min_qty_at_200k(self):
        """At $200K, min notional $100 -> 0.001 BTC (floor is MIN_QTY)."""
        # $100 / $200000 = 0.0005 -> ceil to 0.001 = MIN_QTY
        qty = calc_min_qty(200000)
        assert qty == 0.001

    def test_min_qty_at_30k(self):
        """At $30K (avg TRAIN price), min notional -> 0.004 BTC."""
        # $100 / $30000 = 0.00333 -> ceil to 0.004
        qty = calc_min_qty(30000)
        assert qty == 0.004

    def test_min_qty_always_on_step(self):
        """Min qty must always be a multiple of STEP_SIZE."""
        for price in [20000, 30000, 50000, 75000, 97000, 100000, 120000, 200000]:
            qty = calc_min_qty(price)
            steps = qty / STEP_SIZE
            assert steps == int(steps), f"qty {qty} at ${price} not on step"
            assert qty >= MIN_QTY, f"qty {qty} below MIN_QTY at ${price}"

    def test_min_qty_covers_min_notional(self):
        """Min qty * price must always >= MIN_NOTIONAL."""
        for price in [20000, 30000, 50000, 75000, 97000, 100000, 120000]:
            qty = calc_min_qty(price)
            notional = qty * price
            assert notional >= MIN_NOTIONAL, \
                f"notional ${notional:.2f} < ${MIN_NOTIONAL} at ${price}"

    # --- calc_margin ---

    def test_margin_basic(self):
        """Margin = position / leverage."""
        m = calc_margin(0.001, 100000)
        assert m == pytest.approx(100000 * 0.001 / 125)  # $0.80

    def test_margin_zero_qty(self):
        """Zero qty -> zero margin."""
        assert calc_margin(0, 100000) == 0

    # --- calc_liq_distance_bps ---

    def test_liq_distance_known_value(self):
        """Hand-calculated liquidation distance."""
        # wallet=$10, qty=0.002, btc=$97000
        # position = 0.002 * 97000 = $194
        # maint = 194 * 0.004 = $0.776
        # buffer = 10 - 0.776 = $9.224
        # liq_dist = 9.224 / 194 * 10000 = 475.5 bps
        dist = calc_liq_distance_bps(10.0, 0.002, 97000)
        assert dist == pytest.approx(475.5, abs=0.5)

    def test_liq_distance_100k(self):
        """At $100K with 0.001 BTC, wider liquidation distance."""
        # position = 0.001 * 100000 = $100
        # maint = 100 * 0.004 = $0.40
        # buffer = 10 - 0.40 = $9.60
        # liq_dist = 9.60 / 100 * 10000 = 960 bps
        dist = calc_liq_distance_bps(10.0, 0.001, 100000)
        assert dist == pytest.approx(960.0, abs=0.5)

    def test_liq_distance_larger_wallet(self):
        """Larger wallet -> wider liquidation distance."""
        dist_10 = calc_liq_distance_bps(10.0, 0.001, 100000)
        dist_50 = calc_liq_distance_bps(50.0, 0.001, 100000)
        assert dist_50 > dist_10

    def test_liq_distance_zero_position(self):
        """Zero position -> infinite distance."""
        assert calc_liq_distance_bps(10.0, 0, 100000) == float('inf')

    def test_liq_distance_always_positive(self):
        """Liq distance must be positive when wallet > maintenance margin."""
        for wallet in [5, 10, 20, 50, 100]:
            for price in [50000, 100000]:
                qty = calc_min_qty(price)
                maint = qty * price * MAINT_MARGIN_RATE
                if wallet > maint:
                    dist = calc_liq_distance_bps(wallet, qty, price)
                    assert dist > 0, \
                        f"dist={dist} for wallet=${wallet}, qty={qty}, price=${price}"

    # --- calc_risk_pct ---

    def test_risk_pct_known_value(self):
        """Hand-calculated risk percentage."""
        # wallet=$10, qty=0.002, btc=$97000, loss=865 bps
        # position = 0.002 * 97000 = $194
        # loss = 194 * 865 / 10000 = $16.78
        # risk = 16.78 / 10 = 167.8%
        risk = calc_risk_pct(10.0, 0.002, 97000, 865)
        assert risk == pytest.approx(1.678, abs=0.01)

    def test_risk_pct_zero_wallet(self):
        """Zero wallet -> infinite risk."""
        assert calc_risk_pct(0, 0.001, 100000, 100) == float('inf')

    # --- calc_max_qty ---

    def test_max_qty_on_step(self):
        """Max qty must be on STEP_SIZE increments."""
        qty = calc_max_qty(100, 100000, 0.50, 865)
        steps = qty / STEP_SIZE
        assert steps == int(steps), f"max_qty {qty} not on step"

    def test_max_qty_respects_limit(self):
        """Max qty risk should not exceed max_risk_frac."""
        wallet = 100
        price = 100000
        max_frac = 0.50
        qty = calc_max_qty(wallet, price, max_frac, 865)
        actual_risk = calc_risk_pct(wallet, qty, price, 865)
        assert actual_risk <= max_frac + 0.01  # tiny tolerance from rounding


class TestSignalQuality:
    """Test signal quality scoring."""

    def test_normal_signal(self):
        """Default conditions -> NORMAL tier."""
        cond = make_conditions(atr_pctl=50, ema_sep=0.8, entry_dow=3)
        sq = score_signal(cond)
        assert sq.tier == "NORMAL"

    def test_strong_high_atr_and_ema(self):
        """High ATR + High EMA -> STRONG."""
        cond = make_conditions(atr_pctl=80, ema_sep=1.5)
        sq = score_signal(cond)
        assert sq.tier == "STRONG"
        assert "+high_atr" in sq.reasons
        assert "+high_ema" in sq.reasons

    def test_weak_monday_long(self):
        """Monday LONG -> WEAK (penalty 1.0 + v12_long_monday 0.5)."""
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=50, ema_sep=0.8, entry_dow=0,
        )
        sq = score_signal(cond)
        assert sq.tier == "WEAK"
        assert "monday_long" in sq.reasons

    def test_weak_low_atr_low_ema(self):
        """Low ATR + Low EMA -> WEAK (0.5 + 0.5 penalty)."""
        cond = make_conditions(atr_pctl=10, ema_sep=0.1)
        sq = score_signal(cond)
        assert sq.tier == "WEAK"
        assert "low_atr" in sq.reasons
        assert "low_ema" in sq.reasons

    def test_mixed_high_atr_low_ema(self):
        """High ATR but Low EMA -> cancels out, NORMAL."""
        cond = make_conditions(atr_pctl=80, ema_sep=0.1)
        sq = score_signal(cond)
        # score = 0.5 - 0.5(low_ema) + 0.5(high_atr) = 0.5 -> NORMAL
        assert sq.tier == "NORMAL"

    def test_score_clamped_low(self):
        """Score never goes below 0.0."""
        # Monday LONG + low ATR + low EMA + v12_long_monday
        # penalties: 1.0 + 0.5 + 0.5 + 0.5 = 2.5
        # score = 0.5 - 2.5 = -2.0 -> clamped to 0.0
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=5, ema_sep=0.1, entry_dow=0,
        )
        sq = score_signal(cond)
        assert sq.score >= 0.0
        assert sq.tier == "WEAK"

    def test_score_clamped_high(self):
        """Score never goes above 1.0."""
        cond = make_conditions(atr_pctl=90, ema_sep=2.0)
        sq = score_signal(cond)
        assert sq.score <= 1.0
        assert sq.tier == "STRONG"

    def test_v12_long_monday_double_penalty(self):
        """V12_LONG on Monday triggers both monday_long AND v12_long_monday."""
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=50, ema_sep=0.8, entry_dow=0,
        )
        sq = score_signal(cond)
        assert "monday_long" in sq.reasons
        assert "v12_long_monday" in sq.reasons

    def test_short_on_monday_not_weak(self):
        """Monday SHORT does NOT trigger monday_long."""
        cond = make_conditions(
            signal_type='V12_SHORT', direction='SHORT',
            atr_pctl=50, ema_sep=0.8, entry_dow=0,
        )
        sq = score_signal(cond)
        assert "monday_long" not in sq.reasons
        assert sq.tier != "WEAK"

    def test_individual_condition_functions(self):
        """Test each condition function directly."""
        mon_long = {'entry_dow': 0, 'direction': 'LONG'}
        assert is_monday_long(mon_long) is True
        assert is_monday_long({'entry_dow': 1, 'direction': 'LONG'}) is False

        assert is_low_atr({'atr_pctl': 10}) is True
        assert is_low_atr({'atr_pctl': 50}) is False

        assert is_low_ema({'ema_sep': 0.1}) is True
        assert is_low_ema({'ema_sep': 0.5}) is False

        assert is_high_atr({'atr_pctl': 80}) is True
        assert is_high_atr({'atr_pctl': 50}) is False

        assert is_high_ema({'ema_sep': 1.5}) is True
        assert is_high_ema({'ema_sep': 0.5}) is False

    def test_missing_fields_use_defaults(self):
        """Missing fields should use safe defaults, not crash."""
        sq = score_signal({})  # empty dict
        assert sq.tier in ("STRONG", "NORMAL", "WEAK")
        # Empty dict: atr_pctl defaults to 50 (not low/high),
        # ema_sep defaults to 0.5 (not low/high), direction=None (not LONG)
        assert sq.tier == "NORMAL"


class TestRiskCalculatorUnit:
    """Unit tests for RiskCalculator internals."""

    def test_qty_always_on_step(self, calc):
        """Output qty must always be a multiple of 0.001 BTC."""
        scenarios = [
            (10, 97000), (10, 100000), (50, 100000),
            (100, 100000), (500, 100000), (1000, 50000),
        ]
        for wallet, price in scenarios:
            cond = make_conditions()
            d = calc.calculate(wallet, price, cond)
            if d.qty > 0:
                steps = d.qty / STEP_SIZE
                assert abs(steps - round(steps)) < 1e-9, \
                    f"qty={d.qty} not on step for wallet=${wallet}, price=${price}"

    def test_qty_always_gte_min_or_zero(self, calc):
        """qty is either >= min_qty or exactly 0 (skip)."""
        for wallet in [0.01, 0.50, 1, 5, 10, 50, 100]:
            for price in [50000, 97000, 100000]:
                cond = make_conditions()
                d = calc.calculate(wallet, price, cond)
                min_qty = calc_min_qty(price)
                assert d.qty >= min_qty or d.qty == 0, \
                    f"qty={d.qty} < min={min_qty} for wallet=${wallet}"

    def test_safety_stop_below_liquidation(self, calc):
        """Safety stop must always be below liquidation distance."""
        for wallet in [5, 10, 20, 50, 100]:
            for price in [50000, 97000, 100000]:
                cond = make_conditions()
                d = calc.calculate(wallet, price, cond)
                if d.qty > 0:
                    liq = calc_liq_distance_bps(wallet, d.qty, price)
                    assert d.safety_stop_bps < liq, \
                        f"safety={d.safety_stop_bps} >= liq={liq}"

    def test_safety_stop_is_60pct_of_liq(self, calc):
        """Safety stop = exactly 60% of liquidation distance."""
        d = calc.calculate(10, 100000, make_conditions())
        if d.qty > 0:
            liq = calc_liq_distance_bps(10, d.qty, 100000)
            assert d.safety_stop_bps == pytest.approx(liq * 0.60, abs=0.1)

    def test_weak_gets_higher_step(self, calc):
        """WEAK signal -> adjusted step = base * 2.0 = $12."""
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=5, ema_sep=0.1, entry_dow=0,
        )
        d = calc.calculate(100, 100000, cond)
        assert d.adjusted_step == pytest.approx(12.0)

    def test_strong_gets_lower_step(self, calc):
        """STRONG signal -> adjusted step = base * 0.7 = $4.20."""
        cond = make_conditions(atr_pctl=80, ema_sep=1.5)
        d = calc.calculate(100, 100000, cond)
        assert d.adjusted_step == pytest.approx(4.20)

    def test_normal_gets_base_step(self, calc):
        """NORMAL signal -> adjusted step = base = $6.00."""
        cond = make_conditions(atr_pctl=50, ema_sep=0.8)
        d = calc.calculate(100, 100000, cond)
        assert d.adjusted_step == pytest.approx(6.00)

    def test_reasoning_is_populated(self, calc):
        """Every decision must have reasoning list."""
        d = calc.calculate(10, 100000, make_conditions())
        assert len(d.reasoning) >= 5  # exchange, quality, step, kelly, safety
        assert any("exchange" in r for r in d.reasoning)
        assert any("quality" in r for r in d.reasoning)
        assert any("step" in r for r in d.reasoning)
        assert any("kelly" in r for r in d.reasoning)
        assert any("safety" in r for r in d.reasoning)

    def test_survival_mode_flagged(self, calc):
        """When kelly qty < min qty, reasoning says SURVIVAL."""
        # $10 wallet, $97K BTC -> min 0.002, kelly likely 0.001 or 0.002
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=5, ema_sep=0.1, entry_dow=0,
        )
        d = calc.calculate(10, 97000, cond)
        # WEAK: step=$12, kelly=max(1, 10/12)=1 step=0.001, min=0.002 -> SURVIVAL
        assert any("SURVIVAL" in r for r in d.reasoning)

    def test_growth_mode_flagged(self, calc):
        """When kelly qty >= min qty, reasoning says GROWTH."""
        cond = make_conditions(atr_pctl=80, ema_sep=1.5)
        d = calc.calculate(100, 100000, cond)
        # $100 wallet, STRONG step=$4.20, kelly=23 steps=0.023, min=0.001 -> GROWTH
        assert any("GROWTH" in r for r in d.reasoning)

    def test_position_usd_matches(self, calc):
        """position_usd = qty * btc_price."""
        d = calc.calculate(50, 100000, make_conditions())
        assert d.position_usd == pytest.approx(d.qty * 100000)

    def test_margin_matches(self, calc):
        """margin_usd = qty * btc_price / LEVERAGE."""
        d = calc.calculate(50, 100000, make_conditions())
        expected = d.qty * 100000 / LEVERAGE
        assert d.margin_usd == pytest.approx(expected)

    def test_risk_dollar_matches(self, calc):
        """risk_dollar = position * worst_loss / 10000."""
        d = calc.calculate(50, 100000, make_conditions())
        expected = d.position_usd * 864.651 / 10000
        assert d.risk_dollar == pytest.approx(expected, rel=0.01)


# ############################################################
# LAYER 2: END-TO-END TESTS (hand-verified decisions)
# ############################################################

class TestEndToEnd:
    """Hand-verified complete decisions."""

    def test_10_wallet_97k_strong(self, calc):
        """$10 wallet, $97K BTC, STRONG signal -> hand-verified."""
        cond = make_conditions(
            signal_type='V12_SHORT', direction='SHORT',
            atr_pctl=80, ema_sep=1.5, entry_dow=3, entry_hour=14,
        )
        d = calc.calculate(10, 97000, cond)

        # Step-by-step verification:
        # 1. min_qty = ceil(100/97000 / 0.001) * 0.001 = ceil(1.031) * 0.001 = 0.002
        assert calc_min_qty(97000) == 0.002

        # 2. Quality: high_atr(80>70) + high_ema(1.5>1.0) -> STRONG
        assert d.signal_quality.tier == "STRONG"

        # 3. adj_step = 6.00 * 0.7 = 4.20
        assert d.adjusted_step == pytest.approx(4.20)

        # 4. kelly: steps = max(1, int(10/4.20)) = max(1, 2) = 2
        #    kelly_qty = 2 * 0.001 = 0.002
        # 5. qty = max(0.002, 0.002) = 0.002 -> GROWTH (or exact match)
        assert d.qty == 0.002

        # 6. position = 0.002 * 97000 = $194
        assert d.position_usd == pytest.approx(194.0)

        # 7. liq = (10 - 194*0.004) / 194 * 10000 = 475.5 bps
        #    safety = 475.5 * 0.60 = 285.3
        assert d.safety_stop_bps == pytest.approx(285.3, abs=1.0)

    def test_10_wallet_97k_weak(self, calc):
        """$10 wallet, $97K BTC, WEAK signal -> hand-verified."""
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=15, ema_sep=0.2, entry_dow=0, entry_hour=10,
        )
        d = calc.calculate(10, 97000, cond)

        # Quality: monday_long + low_atr + low_ema + v12_long_monday -> WEAK
        assert d.signal_quality.tier == "WEAK"

        # adj_step = 6.00 * 2.0 = 12.00
        assert d.adjusted_step == pytest.approx(12.0)

        # kelly: steps = max(1, int(10/12)) = max(1, 0) = 1
        # kelly_qty = 1 * 0.001 = 0.001
        # min_qty = 0.002
        # qty = max(0.001, 0.002) = 0.002 -> SURVIVAL
        assert d.qty == 0.002
        assert any("SURVIVAL" in r for r in d.reasoning)

    def test_10_wallet_100k_normal(self, calc):
        """$10 wallet, $100K BTC, NORMAL signal -> hand-verified."""
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=50, ema_sep=0.8, entry_dow=2, entry_hour=14,
        )
        d = calc.calculate(10, 100000, cond)

        # min_qty at $100K = 0.001
        assert calc_min_qty(100000) == 0.001

        # Quality: no bad, no strong -> NORMAL
        assert d.signal_quality.tier == "NORMAL"

        # adj_step = 6.00
        # kelly: steps = max(1, int(10/6)) = max(1, 1) = 1
        # kelly_qty = 0.001 = min_qty -> GROWTH (exact match)
        assert d.qty == 0.001
        assert d.position_usd == pytest.approx(100.0)

        # liq: (10 - 100*0.004) / 100 * 10000 = 960 bps
        # safety: 960 * 0.60 = 576
        assert d.safety_stop_bps == pytest.approx(576.0, abs=1.0)

    def test_50_wallet_100k_strong(self, calc):
        """$50 wallet, $100K BTC, STRONG -> more aggressive sizing."""
        cond = make_conditions(atr_pctl=85, ema_sep=2.0)
        d = calc.calculate(50, 100000, cond)

        # adj_step = 6.00 * 0.7 = 4.20
        # steps = int(50 / 4.20) = 11
        # kelly_qty = 0.011
        assert d.qty == 0.011
        assert d.position_usd == pytest.approx(1100.0)
        assert any("GROWTH" in r for r in d.reasoning)

    def test_100_wallet_100k_weak(self, calc):
        """$100 wallet, $100K BTC, WEAK -> still GROWTH but smaller."""
        cond = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=8, ema_sep=0.1, entry_dow=0, entry_hour=2,
        )
        d = calc.calculate(100, 100000, cond)

        # WEAK: adj_step = 12.00
        # steps = int(100 / 12) = 8
        # kelly_qty = 0.008 > min_qty 0.001 -> GROWTH
        assert d.qty == 0.008
        assert any("GROWTH" in r for r in d.reasoning)

    def test_strong_vs_weak_sizing_ratio(self, calc):
        """STRONG gets ~2.86x more qty than WEAK at same wallet/price."""
        strong = make_conditions(atr_pctl=80, ema_sep=1.5)
        weak = make_conditions(
            signal_type='V12_LONG', direction='LONG',
            atr_pctl=5, ema_sep=0.1, entry_dow=0,
        )

        d_strong = calc.calculate(100, 100000, strong)
        d_weak = calc.calculate(100, 100000, weak)

        # STRONG: step $4.20, steps=23, qty=0.023
        # WEAK: step $12, steps=8, qty=0.008
        # ratio: 23/8 = 2.875
        assert d_strong.qty > d_weak.qty
        ratio = d_strong.qty / d_weak.qty
        assert ratio == pytest.approx(2.875, abs=0.2)

    def test_wallet_growth_increases_qty(self, calc):
        """As wallet grows, qty increases proportionally."""
        cond = make_conditions()
        d10 = calc.calculate(10, 100000, cond)
        d50 = calc.calculate(50, 100000, cond)
        d100 = calc.calculate(100, 100000, cond)

        assert d50.qty > d10.qty
        assert d100.qty > d50.qty

    def test_skip_when_wallet_below_margin(self, calc):
        """When wallet < margin, qty = 0 (SKIP)."""
        # At $97K, min_qty=0.002, margin = 0.002*97000/125 = $1.552
        # wallet $0.50 < $1.552 -> should SKIP
        cond = make_conditions()
        d = calc.calculate(0.50, 97000, cond)
        assert d.qty == 0
        assert any("SKIP" in r for r in d.reasoning)


# ############################################################
# LAYER 3: STRESS TESTS
# ############################################################

class TestStress:
    """Stress tests: losing streaks, worst trades, edge cases."""

    def test_worst_train_trade_at_10_wallet(self, calc):
        """Apply -865 bps worst trade to $10 wallet, verify survival."""
        cond = make_conditions()
        d = calc.calculate(10, 97000, cond)

        # Position $194, loss = 194 * 865/10000 = $16.78
        # Wallet after: 10 - 16.78 = -$6.78 -> but safety stop fires first
        loss_dollar = d.position_usd * 864.651 / 10000
        assert loss_dollar > 10, "Worst trade exceeds wallet at $10"

        # Safety stop caps at 285 bps, not 865
        safety_loss = d.position_usd * d.safety_stop_bps / 10000
        assert safety_loss < 10, \
            f"Safety-capped loss ${safety_loss:.2f} still exceeds $10 wallet"

    def test_worst_train_trade_at_50_wallet(self, calc):
        """At $50 wallet, safety stop should protect."""
        cond = make_conditions(atr_pctl=80, ema_sep=1.5)
        d = calc.calculate(50, 100000, cond)

        # Full worst-case loss
        full_loss = d.position_usd * 864.651 / 10000
        # Safety-capped loss
        safety_loss = d.position_usd * d.safety_stop_bps / 10000

        assert safety_loss < 50, \
            f"Safety loss ${safety_loss:.2f} exceeds $50 wallet"

    def test_consecutive_max_losses(self, calc):
        """Simulate 5 consecutive worst-case losses at safety stop level."""
        wallet = 10.0
        price = 100000

        for i in range(5):
            cond = make_conditions()
            d = calc.calculate(wallet, price, cond)

            if d.qty == 0:
                break  # Can't trade anymore

            # Loss at safety stop level
            loss = d.position_usd * d.safety_stop_bps / 10000
            wallet = max(wallet - loss, 0.01)

        # After 5 worst-case (safety-capped) losses, wallet should still be > 0
        assert wallet > 0, "Wallet went to zero after 5 safety-stopped losses"

    def test_consecutive_avg_losses(self, calc):
        """Simulate 10 average losses (~-61 bps each)."""
        wallet = 10.0
        price = 100000
        avg_loss_bps = 60.617  # from train stats

        for i in range(10):
            cond = make_conditions()
            d = calc.calculate(wallet, price, cond)

            if d.qty == 0:
                break

            loss = d.position_usd * avg_loss_bps / 10000
            wallet = max(wallet - loss, 0.01)

        # Should survive 10 average losses
        assert wallet > 0.01, f"Wallet collapsed to ${wallet:.2f} after 10 avg losses"

    def test_jul_aug_2025_streak(self, calc):
        """Simulate the Jul-Aug 2025 worst period: 7-loss streak, -133 bps total."""
        # Actual loss sequence (approximated from analysis)
        loss_bps_sequence = [-20, -15, -25, -18, -22, -17, -16]  # ~-133 total
        wallet = 50.0  # reasonable mid-growth wallet
        price = 100000

        for loss_bps in loss_bps_sequence:
            cond = make_conditions()
            d = calc.calculate(wallet, price, cond)
            if d.qty == 0:
                break
            loss = d.position_usd * abs(loss_bps) / 10000
            wallet -= loss

        # 7 moderate losses from $50 should not cause ruin
        assert wallet > 10, \
            f"Wallet dropped to ${wallet:.2f} during 7-loss streak"

    def test_tiny_wallet_does_not_crash(self, calc):
        """$0.01 wallet doesn't crash, returns skip."""
        cond = make_conditions()
        d = calc.calculate(0.01, 100000, cond)
        # Should either skip or return min qty
        assert isinstance(d, SizingDecision)

    def test_very_large_wallet(self, calc):
        """$100K wallet produces reasonable results."""
        cond = make_conditions(atr_pctl=80, ema_sep=1.5)
        d = calc.calculate(100000, 100000, cond)

        # step=$4.20, steps=23809, qty=23.809 BTC
        assert d.qty > 20
        assert d.qty == pytest.approx(23.809, abs=0.01)
        assert d.position_usd > 2_000_000

    def test_extreme_btc_price_20k(self, calc):
        """Low BTC price ($20K) -> larger min_qty."""
        cond = make_conditions()
        d = calc.calculate(10, 20000, cond)

        # min_qty at $20K: ceil(100/20000 / 0.001) * 0.001 = ceil(5) * 0.001 = 0.005
        assert calc_min_qty(20000) == 0.005
        assert d.qty >= 0.005

    def test_extreme_btc_price_200k(self, calc):
        """High BTC price ($200K) -> smaller min_qty."""
        cond = make_conditions()
        d = calc.calculate(10, 200000, cond)

        assert calc_min_qty(200000) == 0.001
        assert d.qty >= 0.001

    def test_wallet_below_any_margin(self, calc):
        """Wallet so small it can't cover any margin."""
        cond = make_conditions()
        d = calc.calculate(0.001, 100000, cond)
        # margin for 0.001 BTC at $100K = $0.80
        # wallet $0.001 < $0.80 -> SKIP
        assert d.qty == 0

    def test_recovery_after_drawdown(self, calc):
        """After 50% drawdown, calculator still works and sizes down."""
        cond = make_conditions()
        d_before = calc.calculate(100, 100000, cond)
        d_after = calc.calculate(50, 100000, cond)

        # After losing half, qty should be roughly half
        assert d_after.qty < d_before.qty
        assert d_after.qty > 0

    def test_all_tiers_produce_valid_output(self, calc):
        """Every tier produces valid SizingDecision."""
        scenarios = [
            ("STRONG", make_conditions(atr_pctl=80, ema_sep=1.5)),
            ("NORMAL", make_conditions(atr_pctl=50, ema_sep=0.8)),
            ("WEAK", make_conditions(
                signal_type='V12_LONG', direction='LONG',
                atr_pctl=5, ema_sep=0.1, entry_dow=0,
            )),
        ]
        for tier_name, cond in scenarios:
            d = calc.calculate(50, 100000, cond)
            assert d.signal_quality.tier == tier_name
            assert d.qty > 0
            assert d.safety_stop_bps > 0
            assert len(d.reasoning) >= 5

    def test_deterministic(self, calc):
        """Same inputs -> same outputs (no randomness)."""
        cond = make_conditions()
        d1 = calc.calculate(10, 100000, cond)
        d2 = calc.calculate(10, 100000, cond)

        assert d1.qty == d2.qty
        assert d1.safety_stop_bps == d2.safety_stop_bps
        assert d1.signal_quality.tier == d2.signal_quality.tier
        assert d1.adjusted_step == d2.adjusted_step

    def test_monotonic_wallet_qty(self, calc):
        """Qty should be monotonically non-decreasing with wallet (same price/conditions)."""
        cond = make_conditions()
        wallets = [5, 10, 20, 30, 50, 100, 200, 500, 1000]
        qtys = []
        for w in wallets:
            d = calc.calculate(w, 100000, cond)
            qtys.append(d.qty)

        for i in range(len(qtys) - 1):
            assert qtys[i + 1] >= qtys[i], \
                f"qty decreased: ${wallets[i]}->qty={qtys[i]}, ${wallets[i+1]}->qty={qtys[i+1]}"


# ############################################################
# LAYER 2 BONUS: SEQUENTIAL BACKTEST CONSISTENCY
# ############################################################

class TestSequentialConsistency:
    """Verify sequential backtest logic matches calculator outputs."""

    def test_pnl_calculation(self, calc):
        """P&L = position * bps / 10000, capped at safety stop."""
        wallet = 50.0
        price = 100000
        cond = make_conditions()
        d = calc.calculate(wallet, price, cond)

        # Winning trade: +30 bps
        pnl_win = d.position_usd * 30 / 10000
        assert pnl_win > 0

        # Losing trade: -50 bps (within safety stop)
        pnl_loss = d.position_usd * (-50) / 10000
        assert pnl_loss < 0

        # Losing trade: -1000 bps (beyond safety stop, capped)
        capped_bps = min(1000, d.safety_stop_bps)
        pnl_capped = d.position_usd * (-capped_bps) / 10000
        # Capped loss should be less severe than uncapped
        pnl_uncapped = d.position_usd * (-1000) / 10000
        assert pnl_capped > pnl_uncapped  # less negative

    def test_wallet_update_consistency(self, calc):
        """wallet_after = wallet_before + pnl, bounded by 0.01."""
        wallet = 10.0
        price = 100000
        cond = make_conditions()

        # Win
        d = calc.calculate(wallet, price, cond)
        pnl = d.position_usd * 30 / 10000
        wallet_after = wallet + pnl
        assert wallet_after > wallet

        # Loss
        d = calc.calculate(wallet, price, cond)
        pnl = d.position_usd * (-20) / 10000
        wallet_after = wallet + pnl
        assert wallet_after < wallet
        assert wallet_after > 0
