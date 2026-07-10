"""Failure mode analysis — WHERE and WHEN does the risk calculator fail?

Groups sizing mistakes by:
  - Signal type (V12_LONG, V12_SHORT, BEAR_LONG, BULL_SHORT)
  - Time (hour, day of week)
  - Market conditions (ATR, EMA separation)

Run: python src/engine/risk/tests/test_failure_modes.py
"""
import sys
sys.path.insert(0, "src")

from research.trade_loader import load_enriched_trades
from engine.risk.risk_calculator import RiskCalculator, RiskConfig
from engine.risk.account_health import AccountHealthMonitor, HealthConfig
from engine.risk.exchange_constants import DEFAULT_CAPITAL, MAINT_MARGIN_RATE
from engine.risk.exchange_math import calc_min_qty


def run_analysis(trades, capital, worst_loss_bps):
    """Run trades and collect detailed decision data."""
    health = AccountHealthMonitor()
    health.update(capital)
    calc = RiskCalculator(worst_loss_bps=worst_loss_bps, health=health)

    records = []
    wallet = capital

    for t in trades:
        decision = calc.calculate(wallet, t['btc_price'])
        if decision.action == "SKIP":
            continue

        pnl = decision.qty * t['btc_price'] * (t['bps'] / 10000)
        position = decision.qty * t['btc_price']
        maint = position * MAINT_MARGIN_RATE
        max_loss = wallet - maint

        if pnl < -max_loss:
            wallet = 0.01
        else:
            wallet = max(wallet + pnl, 0.01)

        health.update(wallet, t['bps'])

        records.append({
            'signal_type': t['signal_type'],
            'direction': t['direction'],
            'entry_hour': t['entry_hour'],
            'entry_dow': t['entry_dow'],
            'atr_pctl': t['atr_pctl'],
            'ema_sep': t['ema_sep'],
            'bps': t['bps'],
            'qty': decision.qty,
            'pnl_usd': pnl,
            'health_mult': decision.health_multiplier,
            'risk_pct': decision.risk_pct,
            'wallet_after': wallet,
        })

    return records


def group_stats(records, key_fn, label):
    """Group records by a key and show stats."""
    groups = {}
    for r in records:
        k = key_fn(r)
        if k not in groups:
            groups[k] = []
        groups[k].append(r)

    print(f"\n  By {label}:")
    print(f"  {'Group':>20s} | {'Trades':>6s} | {'Losses':>6s} | {'Loss $':>10s} | {'Big Loss $':>10s} | {'Avg Qty':>8s}")
    print(f"  {'-'*72}")

    for k in sorted(groups.keys()):
        recs = groups[k]
        losses = [r for r in recs if r['bps'] <= 0]
        loss_usd = sum(r['pnl_usd'] for r in losses)
        big_losses = [r for r in losses if r['pnl_usd'] < -1.0]
        big_loss_usd = sum(r['pnl_usd'] for r in big_losses)
        avg_qty = sum(r['qty'] for r in recs) / len(recs)

        print(f"  {str(k):>20s} | {len(recs):>6d} | {len(losses):>6d} | "
              f"${loss_usd:>+9.2f} | ${big_loss_usd:>+9.2f} | {avg_qty:>7.4f}")


def print_worst_trades(records, n=10):
    """Show the N worst trades by USD loss."""
    losses = [r for r in records if r['pnl_usd'] < 0]
    losses.sort(key=lambda r: r['pnl_usd'])

    print(f"\n  Top {n} worst trades (by $ loss):")
    print(f"  {'Signal':>12s} | {'Qty':>7s} | {'PnL $':>10s} | {'bps':>7s} | {'ATR%':>5s} | {'EMA%':>5s} | {'Hour':>4s} | {'DoW':>3s} | {'HMult':>5s}")
    print(f"  {'-'*80}")

    for r in losses[:n]:
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow = dow_names[r['entry_dow']] if r['entry_dow'] < 7 else '?'
        print(f"  {r['signal_type']:>12s} | {r['qty']:>6.3f} | ${r['pnl_usd']:>+9.2f} | "
              f"{r['bps']:>+6.1f} | {r['atr_pctl']:>4.0f} | {r['ema_sep']:>4.1f} | "
              f"{r['entry_hour']:>4d} | {dow:>3s} | {r['health_mult']:>4.2f}")


if __name__ == "__main__":
    print("Loading trades...")
    train_trades = load_enriched_trades("train")
    oos_trades = load_enriched_trades("oos")
    worst_loss = abs(min(t['bps'] for t in train_trades))

    for period_name, trades in [("TRAIN", train_trades), ("OOS", oos_trades)]:
        for capital in [500, 1000]:
            print(f"\n{'#'*60}")
            print(f"  {period_name} — Starting capital: ${capital}")
            print(f"{'#'*60}")

            records = run_analysis(trades, capital, worst_loss)

            group_stats(records, lambda r: r['signal_type'], "Signal Type")
            group_stats(records, lambda r: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][r['entry_dow']], "Day of Week")
            group_stats(records, lambda r: 'low' if r['atr_pctl'] < 30 else ('high' if r['atr_pctl'] > 70 else 'mid'), "ATR Level")
            group_stats(records, lambda r: 'low' if r['ema_sep'] < 0.5 else ('high' if r['ema_sep'] > 1.5 else 'mid'), "EMA Separation")

            print_worst_trades(records)
