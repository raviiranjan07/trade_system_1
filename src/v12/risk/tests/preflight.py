"""Pre-flight check — runs at bot startup before any trading.

Verifies the risk calculator is functioning correctly with known inputs.
If any check fails, bot should refuse to start.

Usage:
    from v12.risk.tests.preflight import run_preflight
    ok = run_preflight(wallet=10.0, btc_price=100000)
    if not ok:
        raise RuntimeError("Risk calculator preflight failed")
"""
import sys
sys.path.insert(0, "src")

from v12.risk.risk_calculator import RiskCalculator, RiskConfig
from v12.risk.account_health import AccountHealthMonitor, HealthConfig
from v12.risk.exchange_math import calc_min_qty, calc_margin
from v12.risk.exchange_constants import MIN_QTY, STEP_SIZE, LEVERAGE


def run_preflight(wallet: float = 10.0, btc_price: float = 100000.0) -> bool:
    """Run all preflight checks. Returns True if all pass.

    Args:
        wallet: Current wallet balance (from Binance).
        btc_price: Current BTC price.
    """
    checks = [
        ("Exchange math", _check_exchange_math),
        ("Calculator creates", _check_calculator_creates),
        ("Calculator output", lambda: _check_calculator_output(wallet, btc_price)),
        ("Health monitor", _check_health_monitor),
        ("Persistence", _check_persistence),
    ]

    all_ok = True
    for name, fn in checks:
        try:
            fn()
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_ok = False

    return all_ok


def _check_exchange_math():
    """Verify exchange math produces sane values."""
    min_qty = calc_min_qty(100000)
    assert min_qty == 0.001, f"min_qty wrong: {min_qty}"

    margin = calc_margin(0.001, 100000)
    assert 0.5 < margin < 1.5, f"margin out of range: {margin}"


def _check_calculator_creates():
    """Verify calculator can be instantiated."""
    health = AccountHealthMonitor()
    calc = RiskCalculator(worst_loss_bps=865, health=health)
    assert calc.worst_loss_bps == 865


def _check_calculator_output(wallet, btc_price):
    """Verify calculator returns valid decision for current wallet/price."""
    health = AccountHealthMonitor()
    health.update(wallet)
    calc = RiskCalculator(worst_loss_bps=865, health=health)

    d = calc.calculate(wallet, btc_price)

    # Must be TRADE or SKIP
    assert d.action in ("TRADE", "SKIP"), f"Bad action: {d.action}"

    if d.action == "TRADE":
        # qty must be positive and at step size
        assert d.qty >= MIN_QTY, f"qty below minimum: {d.qty}"
        assert d.qty % STEP_SIZE < 0.0001, f"qty not at step size: {d.qty}"
        assert d.position_usd > 0, f"position_usd not positive: {d.position_usd}"
        assert d.margin_usd > 0, f"margin_usd not positive: {d.margin_usd}"
        assert d.margin_usd <= wallet, f"margin exceeds wallet: {d.margin_usd} > {wallet}"

    # Reasoning should not be empty
    assert len(d.reasoning) > 0, "No reasoning provided"


def _check_health_monitor():
    """Verify health monitor tracks state correctly."""
    m = AccountHealthMonitor()
    m.update(100.0)
    assert m.peak == 100.0

    m.update(90.0, -20.0)
    assert m.consecutive_losses == 1
    assert m.get_drawdown_pct(90.0) > 0

    m.update(110.0, 30.0)
    assert m.consecutive_losses == 0
    assert m.peak == 110.0


def _check_persistence():
    """Verify state can be saved and restored."""
    m = AccountHealthMonitor()
    m.update(100.0)
    m.update(90.0, -20.0)

    state = m.to_dict()
    m2 = AccountHealthMonitor.from_dict(state)
    assert m2.peak == m.peak
    assert m2.consecutive_losses == m.consecutive_losses


if __name__ == "__main__":
    print("Risk Calculator Pre-flight Check")
    print("================================")
    ok = run_preflight()
    print()
    if ok:
        print("  ALL CHECKS PASSED — safe to trade")
    else:
        print("  CHECKS FAILED — do NOT trade")
    sys.exit(0 if ok else 1)
