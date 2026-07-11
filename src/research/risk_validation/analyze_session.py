"""Analyze a logged trading session from decision_logger.py.

Reads a CSV log file and produces:
  - Overall stats (trades, wins, losses, skips)
  - Sizing quality (were big positions on winners or losers?)
  - Health monitor activity (when did it reduce size?)
  - Timeline (wallet progression)

Run: python src/research/risk_validation/analyze_session.py <path_to_csv>
"""
import sys
import csv
from pathlib import Path


def load_log(filepath):
    """Load decision log CSV."""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['wallet_usd'] = float(row['wallet_usd'])
            row['btc_price'] = float(row['btc_price'])
            row['qty'] = float(row['qty'])
            row['position_usd'] = float(row['position_usd'])
            row['risk_pct'] = float(row['risk_pct'])
            row['health_multiplier'] = float(row['health_multiplier'])
            row['pnl_bps'] = float(row['pnl_bps'])
            row['pnl_usd'] = float(row['pnl_usd'])
            rows.append(row)
    return rows


def analyze(rows):
    """Analyze the logged decisions."""
    trades = [r for r in rows if r['action'] == 'TRADE']
    skips = [r for r in rows if r['action'] == 'SKIP']
    wins = [r for r in trades if r['pnl_bps'] > 0]
    losses = [r for r in trades if r['pnl_bps'] <= 0]

    print(f"\n  OVERALL")
    print(f"  {'='*40}")
    print(f"    Total decisions: {len(rows)}")
    print(f"    Trades: {len(trades)} | Skips: {len(skips)}")
    if trades:
        wr = len(wins) / len(trades) * 100
        print(f"    Wins: {len(wins)} | Losses: {len(losses)} | Win rate: {wr:.1f}%")
        total_pnl = sum(r['pnl_usd'] for r in trades)
        print(f"    Total P&L: ${total_pnl:+.2f}")
        print(f"    Start wallet: ${rows[0]['wallet_usd']:.2f}")
        last_trade = trades[-1]
        print(f"    Final wallet: ${last_trade['wallet_usd'] + last_trade['pnl_usd']:.2f}")

    # Sizing quality
    if trades:
        print(f"\n  SIZING QUALITY")
        print(f"  {'='*40}")
        big_wins = [r for r in wins if r['qty'] > 0.001]
        big_losses = [r for r in losses if r['qty'] > 0.001]
        print(f"    Sized-up wins:   {len(big_wins):>4d} (${sum(r['pnl_usd'] for r in big_wins):>+10.2f})")
        print(f"    Sized-up losses: {len(big_losses):>4d} (${sum(r['pnl_usd'] for r in big_losses):>+10.2f})")
        min_wins = [r for r in wins if r['qty'] <= 0.001]
        min_losses = [r for r in losses if r['qty'] <= 0.001]
        print(f"    Min-size wins:   {len(min_wins):>4d} (${sum(r['pnl_usd'] for r in min_wins):>+10.2f})")
        print(f"    Min-size losses: {len(min_losses):>4d} (${sum(r['pnl_usd'] for r in min_losses):>+10.2f})")

    # Health monitor activity
    reduced = [r for r in trades if r['health_multiplier'] < 1.0]
    if reduced:
        print(f"\n  HEALTH MONITOR ACTIVITY")
        print(f"  {'='*40}")
        print(f"    Trades with reduced size: {len(reduced)} ({len(reduced)/len(trades)*100:.1f}%)")
        reduced_wins = [r for r in reduced if r['pnl_bps'] > 0]
        reduced_losses = [r for r in reduced if r['pnl_bps'] <= 0]
        print(f"    Of those: {len(reduced_wins)} wins, {len(reduced_losses)} losses")
        if reduced_wins:
            missed = sum(r['pnl_usd'] for r in reduced_wins)
            print(f"    Reduced wins P&L: ${missed:+.2f} (would be more at full size)")
        if reduced_losses:
            saved = sum(r['pnl_usd'] for r in reduced_losses)
            print(f"    Reduced losses P&L: ${saved:+.2f} (saved by reducing)")
    else:
        print(f"\n  HEALTH MONITOR: Never activated (all trades at full size)")

    # Skip analysis
    if skips:
        print(f"\n  SKIP ANALYSIS")
        print(f"  {'='*40}")
        for s in skips:
            print(f"    {s['timestamp']} | wallet=${s['wallet_usd']:.2f} | {s['skip_reason']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent log
        log_dir = Path("data/trades/risk_logs")
        if log_dir.exists():
            logs = sorted(log_dir.glob("decisions_*.csv"))
            if logs:
                filepath = logs[-1]
                print(f"  Using most recent log: {filepath}")
            else:
                print("  No log files found in data/trades/risk_logs/")
                print("  Usage: python src/research/risk_validation/analyze_session.py <path_to_csv>")
                sys.exit(1)
        else:
            print("  No log directory found.")
            print("  Usage: python src/research/risk_validation/analyze_session.py <path_to_csv>")
            sys.exit(1)
    else:
        filepath = sys.argv[1]

    rows = load_log(filepath)
    analyze(rows)
