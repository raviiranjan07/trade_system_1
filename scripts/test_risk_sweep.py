"""Parameter sweep for risk calculator.

Phase 1: base_risk_pct (5%, 10%, 15%, 20%, 25%)
Phase 2: drawdown thresholds (tight, default, loose, off)
Phase 3: losing streak threshold (3, 5, 7, off)

Each phase tests one parameter independently, others at default.
Runs on both TRAIN and OOS at multiple starting wallets.
"""
import sys
sys.path.insert(0, "src")

from v12.risk.trade_loader import load_enriched_trades
from v12.risk.risk_calculator import RiskCalculator, RiskConfig
from v12.risk.account_health import AccountHealthMonitor, HealthConfig
from v12.risk.exchange_constants import LEVERAGE, MAINT_MARGIN_RATE


def simulate(trades, capital, worst_loss_bps, risk_config, health_config):
    """Run trades sequentially through risk calculator."""
    health = AccountHealthMonitor(config=health_config)
    health.update(capital)
    calc = RiskCalculator(worst_loss_bps=worst_loss_bps, config=risk_config, health=health)

    wallet = capital
    peak = capital
    max_dd = 0.0
    wins = 0
    losses = 0
    skips = 0

    for t in trades:
        decision = calc.calculate(wallet, t['btc_price'])

        if decision.action == "SKIP":
            skips += 1
            continue

        pnl = decision.qty * t['btc_price'] * (t['bps'] / 10000)
        position = decision.qty * t['btc_price']
        maint = position * MAINT_MARGIN_RATE
        max_loss = wallet - maint

        if pnl < -max_loss:
            wallet = 0.01
        else:
            wallet = max(wallet + pnl, 0.01)

        if t['bps'] > 0:
            wins += 1
        else:
            losses += 1

        if wallet > peak:
            peak = wallet
        dd = (peak - wallet) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        health.update(wallet, t['bps'])

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    return {
        'final': wallet,
        'peak': peak,
        'max_dd': max_dd,
        'wins': wins,
        'losses': losses,
        'win_rate': wr,
        'skips': skips,
    }


def print_table(title, rows):
    """Print comparison table."""
    print()
    print(f"  {title}")
    print(f"  {'='*len(title)}")
    print(f"  {'Config':>25s} | {'TRAIN $':>12s} | {'TRAIN DD':>8s} | {'OOS $':>12s} | {'OOS DD':>8s} | {'Skips':>5s}")
    print(f"  {'-'*80}")
    for label, tr, oo in rows:
        print(f"  {label:>25s} | ${tr['final']:>10.2f} | {tr['max_dd']:>7.1%} | ${oo['final']:>10.2f} | {oo['max_dd']:>7.1%} | {tr['skips']+oo['skips']:>5d}")
    print()


if __name__ == "__main__":
    print("Loading trades...")
    train_trades = load_enriched_trades("train")
    oos_trades = load_enriched_trades("oos")
    worst_loss = abs(min(t['bps'] for t in train_trades))
    print(f"  TRAIN: {len(train_trades)} trades | OOS: {len(oos_trades)} trades")
    print(f"  Worst loss: {worst_loss:.1f} bps")

    # Test at multiple starting capitals
    for capital in [5, 100, 500, 1000]:
        print()
        print(f"{'#'*60}")
        print(f"  STARTING CAPITAL: ${capital}")
        print(f"{'#'*60}")

        default_risk = RiskConfig(base_risk_pct=0.10, safety_pct=0.60)
        default_health = HealthConfig()

        # PHASE 1: base_risk_pct
        rows = []
        for pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
            rc = RiskConfig(base_risk_pct=pct, safety_pct=0.60)
            tr = simulate(train_trades, capital, worst_loss, rc, default_health)
            oo = simulate(oos_trades, capital, worst_loss, rc, default_health)
            rows.append((f"risk={pct:.0%}", tr, oo))

        print_table(f"PHASE 1: Base Risk % (capital=${capital})", rows)

        # PHASE 2: Drawdown thresholds
        rows = []
        dd_configs = [
            ("tight (10%/25%)", HealthConfig(drawdown_threshold_1=0.10, drawdown_threshold_2=0.25)),
            ("default (15%/30%)", HealthConfig(drawdown_threshold_1=0.15, drawdown_threshold_2=0.30)),
            ("loose (20%/40%)", HealthConfig(drawdown_threshold_1=0.20, drawdown_threshold_2=0.40)),
            ("off", HealthConfig(drawdown_threshold_1=1.0, drawdown_threshold_2=1.0)),
        ]
        for label, hc in dd_configs:
            tr = simulate(train_trades, capital, worst_loss, default_risk, hc)
            oo = simulate(oos_trades, capital, worst_loss, default_risk, hc)
            rows.append((label, tr, oo))

        print_table(f"PHASE 2: Drawdown Thresholds (capital=${capital})", rows)

        # PHASE 3: Losing streak threshold
        rows = []
        streak_configs = [
            ("streak=3", HealthConfig(consec_loss_threshold=3)),
            ("streak=5", HealthConfig(consec_loss_threshold=5)),
            ("streak=7", HealthConfig(consec_loss_threshold=7)),
            ("streak=off", HealthConfig(consec_loss_threshold=999)),
        ]
        for label, hc in streak_configs:
            tr = simulate(train_trades, capital, worst_loss, default_risk, hc)
            oo = simulate(oos_trades, capital, worst_loss, default_risk, hc)
            rows.append((label, tr, oo))

        print_table(f"PHASE 3: Losing Streak (capital=${capital})", rows)
