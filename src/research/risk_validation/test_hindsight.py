"""Hindsight analysis — was each sizing decision good or bad?

For each trade:
  - What size did the calculator pick?
  - Did the trade win or lose?
  - Tag: "good_sizing" (small on loser, big on winner) or "bad_sizing" (big on loser)

Run: python src/engine/risk/tests/test_hindsight.py
"""
import sys
sys.path.insert(0, "src")

from research.trade_loader import load_enriched_trades
from engine.risk.risk_calculator import RiskCalculator, RiskConfig
from engine.risk.account_health import AccountHealthMonitor, HealthConfig
from engine.risk.exchange_constants import DEFAULT_CAPITAL, MAINT_MARGIN_RATE
from engine.risk.exchange_math import calc_min_qty


def run_hindsight(trades, capital, worst_loss_bps):
    """Run trades through calculator and tag each decision."""
    health = AccountHealthMonitor()
    health.update(capital)
    calc = RiskCalculator(worst_loss_bps=worst_loss_bps, health=health)

    min_qty = calc_min_qty(trades[0]['btc_price'])
    results = []
    wallet = capital

    for t in trades:
        decision = calc.calculate(wallet, t['btc_price'])

        if decision.action == "SKIP":
            results.append({
                'trade': t,
                'decision': decision,
                'pnl_usd': 0,
                'tag': 'skipped',
                'sized_above_min': False,
            })
            continue

        pnl = decision.qty * t['btc_price'] * (t['bps'] / 10000)
        sized_above_min = decision.qty > min_qty
        won = t['bps'] > 0

        # Tag the decision
        if won and sized_above_min:
            tag = 'good_big_win'       # sized up on a winner
        elif won and not sized_above_min:
            tag = 'neutral_min_win'    # minimum size, won anyway
        elif not won and sized_above_min:
            tag = 'bad_big_loss'       # sized up on a loser
        else:
            tag = 'neutral_min_loss'   # minimum size, lost (can't go lower)

        # Update wallet
        position = decision.qty * t['btc_price']
        maint = position * MAINT_MARGIN_RATE
        max_loss = wallet - maint
        if pnl < -max_loss:
            wallet = 0.01
        else:
            wallet = max(wallet + pnl, 0.01)

        health.update(wallet, t['bps'])

        results.append({
            'trade': t,
            'decision': decision,
            'pnl_usd': pnl,
            'tag': tag,
            'sized_above_min': sized_above_min,
            'wallet_after': wallet,
        })

    return results


def print_summary(results, label):
    """Print hindsight summary."""
    tags = {}
    for r in results:
        tags[r['tag']] = tags.get(r['tag'], 0) + 1

    total_good_pnl = sum(r['pnl_usd'] for r in results if r['tag'] == 'good_big_win')
    total_bad_pnl = sum(r['pnl_usd'] for r in results if r['tag'] == 'bad_big_loss')

    print(f"\n  {label}")
    print(f"  {'='*len(label)}")
    print(f"    Total trades: {len(results)}")
    for tag in ['good_big_win', 'bad_big_loss', 'neutral_min_win', 'neutral_min_loss', 'skipped']:
        count = tags.get(tag, 0)
        pct = count / len(results) * 100 if results else 0
        print(f"    {tag:>20s}: {count:>4d} ({pct:>5.1f}%)")

    print(f"\n    P&L from sized-up wins:   ${total_good_pnl:>+10.2f}")
    print(f"    P&L from sized-up losses: ${total_bad_pnl:>+10.2f}")
    net = total_good_pnl + total_bad_pnl
    print(f"    Net from sizing decisions: ${net:>+10.2f}")

    # Show worst bad_big_loss trades
    bad = [r for r in results if r['tag'] == 'bad_big_loss']
    if bad:
        bad.sort(key=lambda r: r['pnl_usd'])
        print(f"\n    Top 5 worst sized-up losses:")
        for r in bad[:5]:
            t = r['trade']
            d = r['decision']
            print(f"      {t['signal_type']:>12s} | qty={d.qty:.3f} | "
                  f"pnl=${r['pnl_usd']:>+8.2f} | {t['bps']:>+6.1f}bps | "
                  f"health_mult={d.health_multiplier:.2f}")


if __name__ == "__main__":
    print("Loading trades...")
    train_trades = load_enriched_trades("train")
    oos_trades = load_enriched_trades("oos")
    worst_loss = abs(min(t['bps'] for t in train_trades))

    for capital in [5, 500, 1000]:
        print(f"\n{'#'*50}")
        print(f"  Starting capital: ${capital}")
        print(f"{'#'*50}")

        train_results = run_hindsight(train_trades, capital, worst_loss)
        oos_results = run_hindsight(oos_trades, capital, worst_loss)

        print_summary(train_results, f"TRAIN (capital=${capital})")
        print_summary(oos_results, f"OOS (capital=${capital})")
