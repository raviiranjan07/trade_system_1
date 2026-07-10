"""Unit tests for risk calculator and account health monitor.

Tests edge cases, persistence, and expected behavior.
Run: python -m engine.risk.tests.test_unit
"""
import sys
sys.path.insert(0, "src")


def test_health_monitor_basic():
    """Test basic health monitor behavior."""
    from engine.risk.account_health import AccountHealthMonitor, HealthConfig

    m = AccountHealthMonitor()

    # Initial state
    m.update(100.0)
    assert m.peak == 100.0
    assert m.consecutive_losses == 0
    assert m.get_drawdown_pct(100.0) == 0.0
    assert m.get_risk_multiplier(100.0) == 1.0
    print("  [PASS] Initial state correct")

    # After a loss
    m.update(90.0, -20.0)
    assert m.consecutive_losses == 1
    assert m.peak == 100.0
    assert abs(m.get_drawdown_pct(90.0) - 0.10) < 0.001
    print("  [PASS] Single loss tracked")

    # After a win — streak resets, peak updates
    m.update(110.0, 30.0)
    assert m.consecutive_losses == 0
    assert m.peak == 110.0
    assert m.get_drawdown_pct(110.0) == 0.0
    print("  [PASS] Win resets streak and updates peak")


def test_health_drawdown_thresholds():
    """Test drawdown triggers correct multipliers."""
    from engine.risk.account_health import AccountHealthMonitor, HealthConfig

    cfg = HealthConfig(
        drawdown_threshold_1=0.15,
        drawdown_threshold_2=0.30,
        drawdown_reduction_1=0.70,
        drawdown_reduction_2=0.40,
        consec_loss_threshold=999,  # disable streak
        recent_lookback=999,        # disable recent WR
    )
    m = AccountHealthMonitor(config=cfg)
    m.update(1000.0)

    # 10% DD — no reduction
    mult = m.get_risk_multiplier(900.0)
    assert mult == 1.0, f"Expected 1.0, got {mult}"
    print("  [PASS] 10% DD = no reduction")

    # 20% DD — threshold 1
    mult = m.get_risk_multiplier(800.0)
    assert mult == 0.70, f"Expected 0.70, got {mult}"
    print("  [PASS] 20% DD = 0.70 multiplier")

    # 35% DD — threshold 2
    mult = m.get_risk_multiplier(650.0)
    assert mult == 0.40, f"Expected 0.40, got {mult}"
    print("  [PASS] 35% DD = 0.40 multiplier")


def test_health_streak():
    """Test consecutive loss streak triggers."""
    from engine.risk.account_health import AccountHealthMonitor, HealthConfig

    cfg = HealthConfig(
        drawdown_threshold_1=1.0,   # disable DD
        drawdown_threshold_2=1.0,
        consec_loss_threshold=3,
        consec_loss_reduction=0.50,
        recent_lookback=999,
        recent_winrate_floor=0.0,   # disable recent WR check
    )
    m = AccountHealthMonitor(config=cfg)
    m.update(1000.0)

    # 2 losses — no reduction
    m.update(990.0, -10.0)
    m.update(980.0, -10.0)
    assert m.consecutive_losses == 2
    assert m.get_risk_multiplier(980.0) == 1.0
    print("  [PASS] 2 losses = no reduction")

    # 3 losses — triggers
    m.update(970.0, -10.0)
    assert m.consecutive_losses == 3
    assert m.get_risk_multiplier(970.0) == 0.50
    print("  [PASS] 3 losses = 0.50 multiplier")

    # Win resets
    m.update(980.0, 10.0)
    assert m.consecutive_losses == 0
    assert m.get_risk_multiplier(980.0) == 1.0
    print("  [PASS] Win resets streak")


def test_health_persistence():
    """Test save and restore state."""
    from engine.risk.account_health import AccountHealthMonitor, HealthConfig

    m = AccountHealthMonitor()
    m.update(1000.0)
    m.update(950.0, -20.0)
    m.update(930.0, -15.0)
    m.update(960.0, 30.0)

    # Save
    state = m.to_dict()
    assert state['peak'] == 1000.0
    assert state['consecutive_losses'] == 0
    assert len(state['recent_trades']) == 3

    # Restore
    m2 = AccountHealthMonitor.from_dict(state)
    assert m2.peak == m.peak
    assert m2.consecutive_losses == m.consecutive_losses
    assert list(m2.recent_trades) == list(m.recent_trades)
    print("  [PASS] Persistence save/restore works")


def test_calculator_skip_insufficient():
    """Test SKIP when wallet can't afford minimum."""
    from engine.risk.risk_calculator import RiskCalculator
    from engine.risk.account_health import AccountHealthMonitor

    health = AccountHealthMonitor()
    health.update(0.50)  # $0.50 wallet
    calc = RiskCalculator(worst_loss_bps=865, health=health)

    d = calc.calculate(0.50, 100000)
    assert d.action == "SKIP"
    assert d.skip_reason == "insufficient_funds"
    print("  [PASS] SKIP on insufficient funds")


def test_calculator_minimum_qty():
    """Test that calculator never goes below exchange minimum."""
    from engine.risk.risk_calculator import RiskCalculator
    from engine.risk.account_health import AccountHealthMonitor
    from engine.risk.exchange_math import calc_min_qty

    health = AccountHealthMonitor()
    health.update(5.0)
    calc = RiskCalculator(worst_loss_bps=865, health=health)

    d = calc.calculate(5.0, 100000)
    assert d.action == "TRADE"
    assert d.qty >= calc_min_qty(100000)
    print("  [PASS] Never below exchange minimum")


def test_calculator_scales_with_wallet():
    """Test that bigger wallet = bigger position."""
    from engine.risk.risk_calculator import RiskCalculator
    from engine.risk.account_health import AccountHealthMonitor

    results = []
    for wallet in [500, 1000, 5000]:
        health = AccountHealthMonitor()
        health.update(wallet)
        calc = RiskCalculator(worst_loss_bps=865, health=health)
        d = calc.calculate(wallet, 100000)
        results.append(d.qty)

    assert results[0] < results[1] < results[2], f"Expected scaling: {results}"
    print(f"  [PASS] Scales with wallet: {results}")


def test_calculator_health_reduces_size():
    """Test that drawdown reduces position size."""
    from engine.risk.risk_calculator import RiskCalculator, RiskConfig
    from engine.risk.account_health import AccountHealthMonitor, HealthConfig

    cfg = HealthConfig(drawdown_threshold_1=0.10, drawdown_reduction_1=0.50)

    # Healthy — no drawdown
    h1 = AccountHealthMonitor(config=cfg)
    h1.update(1000.0)
    calc1 = RiskCalculator(worst_loss_bps=865, health=h1)
    d1 = calc1.calculate(1000.0, 100000)

    # In drawdown — wallet dropped 20%
    h2 = AccountHealthMonitor(config=cfg)
    h2.update(1000.0)  # peak
    h2.update(800.0, -50.0)  # dropped
    calc2 = RiskCalculator(worst_loss_bps=865, health=h2)
    d2 = calc2.calculate(800.0, 100000)

    assert d2.qty < d1.qty, f"Drawdown should reduce: healthy={d1.qty}, drawdown={d2.qty}"
    print(f"  [PASS] Drawdown reduces size: {d1.qty} -> {d2.qty}")


def test_calculator_zero_wallet():
    """Test edge case: $0 wallet."""
    from engine.risk.risk_calculator import RiskCalculator
    from engine.risk.account_health import AccountHealthMonitor

    health = AccountHealthMonitor()
    health.update(0.0)
    calc = RiskCalculator(worst_loss_bps=865, health=health)

    d = calc.calculate(0.0, 100000)
    assert d.action == "SKIP"
    print("  [PASS] $0 wallet = SKIP")


def test_calculator_huge_wallet():
    """Test edge case: $100K wallet."""
    from engine.risk.risk_calculator import RiskCalculator
    from engine.risk.account_health import AccountHealthMonitor

    health = AccountHealthMonitor()
    health.update(100000.0)
    calc = RiskCalculator(worst_loss_bps=865, health=health)

    d = calc.calculate(100000.0, 100000)
    assert d.action == "TRADE"
    assert d.qty > 0
    assert d.risk_pct < 1.0  # risk should be bounded
    print(f"  [PASS] $100K wallet: qty={d.qty}, risk={d.risk_pct:.1%}")


def test_health_floor():
    """Test that multiplier never goes below floor."""
    from engine.risk.account_health import AccountHealthMonitor, HealthConfig

    cfg = HealthConfig(risk_floor=0.20)
    m = AccountHealthMonitor(config=cfg)
    m.update(1000.0)

    # Trigger everything: big DD + long streak + bad recent WR
    for i in range(10):
        m.update(1000.0 - (i+1)*80, -50.0)

    mult = m.get_risk_multiplier(200.0)  # 80% drawdown
    assert mult >= 0.20, f"Floor violated: {mult}"
    print(f"  [PASS] Floor holds: multiplier={mult:.2f} >= 0.20")


if __name__ == "__main__":
    tests = [
        ("Health Monitor — Basic", test_health_monitor_basic),
        ("Health Monitor — Drawdown Thresholds", test_health_drawdown_thresholds),
        ("Health Monitor — Streak", test_health_streak),
        ("Health Monitor — Persistence", test_health_persistence),
        ("Health Monitor — Floor", test_health_floor),
        ("Calculator — Skip Insufficient", test_calculator_skip_insufficient),
        ("Calculator — Minimum Qty", test_calculator_minimum_qty),
        ("Calculator — Scales With Wallet", test_calculator_scales_with_wallet),
        ("Calculator — Health Reduces Size", test_calculator_health_reduces_size),
        ("Calculator — Zero Wallet", test_calculator_zero_wallet),
        ("Calculator — Huge Wallet", test_calculator_huge_wallet),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{name}")
        print(f"{'-'*len(name)}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed out of {len(tests)}")
