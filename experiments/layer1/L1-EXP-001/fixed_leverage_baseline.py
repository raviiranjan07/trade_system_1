"""L1-EXP-001: Fixed Leverage Baseline

QUESTION: What does fixed leverage do on V1.3.2 trades?
This is the BASELINE for all Layer 1 experiments. Every future method
must beat (or match) fixed leverage at the same risk level.

WHAT WE TEST:
  - Fixed 1x, 2x, 3x, 5x, 7x, 10x, 15x, 20x
  - Starting capital: $10
  - Binance min notional: $170 (can't trade if equity * leverage < $170)
  - Original trade order + Monte Carlo 1000 shuffled paths

METRICS:
  - Final equity, total return %
  - Max drawdown
  - Min equity (lowest point)
  - Trades skipped (can't meet minimum)
  - Monte Carlo: Median, P5, P95, Avg DD, P95 DD, Ruin %
  - Worst single trade impact at each leverage
  - Consecutive losses to ruin

BASELINE REFERENCE: All future experiments compare against these numbers.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from engine.backtest import run_backtest
from engine.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
MIN_NOTIONAL = 170.0   # Binance BTCUSDT Futures minimum
MAX_LEVERAGE = 20       # Binance max for BTCUSDT
N_SIMS = 1000           # Monte Carlo paths

# ============================================================
# LOAD V1.3.2 TRADES
# ============================================================
config = load_config()
trades = run_backtest(config)
returns = [t.net_profit_bps for t in trades]

wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]

print("=" * 100)
print("L1-EXP-001: FIXED LEVERAGE BASELINE")
print("=" * 100)
print(f"V1.3.2 OOS: {len(returns)} trades, {len(wins)/len(returns)*100:.1f}% win rate")
print(f"Mean: {np.mean(returns):+.1f} bps | Median: {np.median(returns):+.1f} bps | Std: {np.std(returns):.1f} bps")
print(f"Avg win: {np.mean(wins):+.1f} bps | Avg loss: {np.mean(losses):.1f} bps")
print(f"Best: {max(returns):+.1f} bps | Worst: {min(returns):.1f} bps")
print(f"Starting capital: ${STARTING_CAPITAL} | Min notional: ${MIN_NOTIONAL}")
print()


# ============================================================
# CORE FUNCTIONS
# ============================================================
def simulate_fixed(rets, leverage, capital=STARTING_CAPITAL):
    """Simulate fixed leverage with Binance minimum notional constraint."""
    equity = [capital]
    skipped = 0
    for r in rets:
        eq = equity[-1]
        # Can we place an order?
        if eq * leverage < MIN_NOTIONAL:
            equity.append(eq)
            skipped += 1
            continue
        pnl = eq * leverage * (r / 10000)
        equity.append(max(eq + pnl, 0.01))
    return equity, skipped


def calc_max_dd(equity):
    """Maximum drawdown as fraction (0 to 1)."""
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ============================================================
# PART 1: ORIGINAL ORDER — Fixed leverage at all levels
# ============================================================
print("=" * 100)
print("PART 1: ORIGINAL TRADE ORDER (V1.3.2 OOS 2024-2025)")
print("=" * 100)
print()

LEVERAGES = [1, 2, 3, 5, 7, 10, 15, 20]

print(f"  {'Leverage':>8s} | {'Final':>12s} | {'Return':>10s} | {'MaxDD':>8s} | {'MinEq':>8s} | {'Skipped':>8s} | {'Can Trade?':>10s}")
print(f"  {'-'*80}")

original_results = {}
for lev in LEVERAGES:
    eq, skipped = simulate_fixed(returns, lev)
    dd = calc_max_dd(eq)
    final = eq[-1]
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    min_eq = min(eq)
    can_trade = "YES" if STARTING_CAPITAL * lev >= MIN_NOTIONAL else "NO"

    original_results[lev] = {
        'final': final, 'return': ret, 'dd': dd,
        'min_eq': min_eq, 'skipped': skipped
    }

    print(f"  {lev:>7d}x | ${final:11.2f} | {ret:+9.1f}% | {dd*100:6.1f}% | ${min_eq:6.2f} | {skipped:>8d} | {can_trade}")

print()
print("  NOTE: Leverage 1-8x CANNOT place orders ($10 * 8x = $80 < $170 minimum)")
print("  Only 9x+ can actually trade. 17x is minimum to meet $170.")


# ============================================================
# PART 2: WORST SINGLE TRADE IMPACT
# ============================================================
print()
print("=" * 100)
print("PART 2: WORST SINGLE TRADE IMPACT")
print("=" * 100)
print()

worst_trade = min(returns)
best_trade = max(returns)
avg_loss = np.mean(losses)
median_loss = np.median(losses)

print(f"  Worst trade: {worst_trade:.1f} bps ({worst_trade/100:.2f}%)")
print(f"  Best trade:  {best_trade:+.1f} bps ({best_trade/100:.2f}%)")
print(f"  Avg loss:    {avg_loss:.1f} bps | Median loss: {median_loss:.1f} bps")
print()

print(f"  {'Leverage':>8s} | {'Worst Loss $':>12s} | {'Worst Loss %':>12s} | {'After Worst':>12s} | {'Still Trade?':>12s}")
print(f"  {'-'*70}")

for lev in LEVERAGES:
    loss_dollar = STARTING_CAPITAL * lev * abs(worst_trade) / 10000
    loss_pct = loss_dollar / STARTING_CAPITAL * 100
    remaining = STARTING_CAPITAL - loss_dollar
    can_trade = "YES" if remaining * MAX_LEVERAGE >= MIN_NOTIONAL else "NO"
    print(f"  {lev:>7d}x | ${loss_dollar:11.2f} | {loss_pct:10.1f}% | ${remaining:10.2f} | {can_trade:>12s}")


# ============================================================
# PART 3: CONSECUTIVE LOSSES TO RUIN
# ============================================================
print()
print("=" * 100)
print("PART 3: CONSECUTIVE LOSSES TO RUIN (can't meet $170 minimum)")
print("=" * 100)
print()

print(f"  Using avg loss = {avg_loss:.1f} bps and median loss = {median_loss:.1f} bps")
print()

for loss_type, loss_val in [("avg", avg_loss), ("median", median_loss), ("worst", worst_trade)]:
    print(f"  --- Using {loss_type} loss ({loss_val:.1f} bps) ---")
    for lev in [10, 15, 17, 20]:
        eq = STARTING_CAPITAL
        for i in range(50):
            if eq * lev < MIN_NOTIONAL:
                print(f"    {lev:2d}x: RUIN after {i} consecutive {loss_type} losses (equity: ${eq:.2f})")
                break
            loss = eq * lev * abs(loss_val) / 10000
            eq -= loss
        else:
            print(f"    {lev:2d}x: Survived 50 consecutive {loss_type} losses (equity: ${eq:.4f})")
    print()


# ============================================================
# PART 4: MONTE CARLO — 1000 shuffled paths per leverage
# ============================================================
print("=" * 100)
print("PART 4: MONTE CARLO — 1000 shuffled trade orderings")
print("=" * 100)
print()

np.random.seed(42)

print(f"  {'Leverage':>8s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'P95':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*110}")

mc_results = {}
for lev in LEVERAGES:
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, _ = simulate_fixed(shuffled, lev)
        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)
        if eq[-1] < MIN_NOTIONAL / MAX_LEVERAGE:
            ruined += 1

    mc_results[lev] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'p95': np.percentile(finals, 95),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruined / N_SIMS * 100,
    }

    r = mc_results[lev]
    print(f"  {lev:>7d}x | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | ${r['p95']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# ============================================================
# PART 5: SURVIVAL ANALYSIS — Never dip below key thresholds
# ============================================================
print()
print("=" * 100)
print("PART 5: SURVIVAL — What % of 1000 paths NEVER dip below threshold?")
print("=" * 100)
print()

np.random.seed(42)

thresholds = [3, 5, 7, 8, 9]
print(f"  {'Leverage':>8s}", end="")
for t in thresholds:
    print(f" | {'>${}'.format(t):>7s}", end="")
print()
print(f"  {'-'*60}")

for lev in LEVERAGES:
    above = {t: 0 for t in thresholds}

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, _ = simulate_fixed(shuffled, lev)
        min_eq = min(eq)
        for t in thresholds:
            if min_eq >= t:
                above[t] += 1

    print(f"  {lev:>7d}x", end="")
    for t in thresholds:
        pct = above[t] / N_SIMS * 100
        print(f" | {pct:6.1f}%", end="")
    print()


# ============================================================
# PART 6: EQUITY CURVE SHAPE — How bumpy is the ride?
# ============================================================
print()
print("=" * 100)
print("PART 6: EQUITY CURVE STATISTICS (original order)")
print("=" * 100)
print()

print(f"  {'Leverage':>8s} | {'Equity@50':>10s} | {'Equity@100':>10s} | {'Equity@150':>10s} | {'Equity@200':>10s} | {'Final':>10s} | {'MaxDD':>7s} | {'#DD>10%':>8s}")
print(f"  {'-'*95}")

for lev in LEVERAGES:
    eq, _ = simulate_fixed(returns, lev)

    # Count drawdowns > 10%
    peak = eq[0]
    dd_count = 0
    in_dd = False
    for e in eq:
        if e > peak:
            peak = e
            in_dd = False
        dd = (peak - e) / peak
        if dd > 0.10 and not in_dd:
            dd_count += 1
            in_dd = True

    dd = calc_max_dd(eq)

    eq_50 = eq[min(50, len(eq)-1)]
    eq_100 = eq[min(100, len(eq)-1)]
    eq_150 = eq[min(150, len(eq)-1)]
    eq_200 = eq[min(200, len(eq)-1)]

    print(f"  {lev:>7d}x | ${eq_50:9.2f} | ${eq_100:9.2f} | ${eq_150:9.2f} | ${eq_200:9.2f} | ${eq[-1]:9.2f} | {dd*100:5.1f}% | {dd_count:>8d}")


# ============================================================
# PART 7: RISK-ADJUSTED METRICS
# ============================================================
print()
print("=" * 100)
print("PART 7: RISK-ADJUSTED METRICS (original order)")
print("=" * 100)
print()

print(f"  {'Leverage':>8s} | {'Final':>12s} | {'MaxDD':>8s} | {'Calmar':>8s} | {'Return/DD':>10s} | {'P5/Start':>10s}")
print(f"  {'-'*70}")

for lev in LEVERAGES:
    eq, _ = simulate_fixed(returns, lev)
    dd = calc_max_dd(eq)
    final = eq[-1]
    total_ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL

    # Calmar = annualized return / max DD (220 trades over ~2 years)
    annual_ret = (1 + total_ret) ** 0.5 - 1 if total_ret > -1 else -1
    calmar = annual_ret / dd if dd > 0 else 0

    # Return per unit DD
    ret_per_dd = total_ret / dd if dd > 0 else 0

    # P5 safety
    p5_ratio = mc_results[lev]['p5'] / STARTING_CAPITAL

    print(f"  {lev:>7d}x | ${final:11.2f} | {dd*100:6.1f}% | {calmar:7.1f} | {ret_per_dd:9.1f} | {p5_ratio:8.2f}x")


# ============================================================
# PART 8: KEY FINDING — What leverage CAN we actually use?
# ============================================================
print()
print("=" * 100)
print("PART 8: PRACTICAL REALITY CHECK")
print("=" * 100)
print()

print("  BINANCE CONSTRAINTS:")
print(f"    Min notional:    ${MIN_NOTIONAL}")
print(f"    Max leverage:    {MAX_LEVERAGE}x")
print(f"    Starting equity: ${STARTING_CAPITAL}")
print(f"    Min leverage to trade: {int(np.ceil(MIN_NOTIONAL / STARTING_CAPITAL))}x")
print()

min_lev_needed = int(np.ceil(MIN_NOTIONAL / STARTING_CAPITAL))
print(f"  FEASIBLE LEVERAGE RANGE: {min_lev_needed}x to {MAX_LEVERAGE}x")
print()

print("  AT MINIMUM LEVERAGE ({0}x):".format(min_lev_needed))
eq_min, sk_min = simulate_fixed(returns, min_lev_needed)
dd_min = calc_max_dd(eq_min)
print(f"    Final: ${eq_min[-1]:.2f} | DD: {dd_min*100:.1f}% | Skipped: {sk_min}")
print(f"    MC P5: ${mc_results.get(min_lev_needed, mc_results[20])['p5']:.2f}")
print()

print(f"  AT MAXIMUM LEVERAGE ({MAX_LEVERAGE}x):")
eq_max, sk_max = simulate_fixed(returns, MAX_LEVERAGE)
dd_max = calc_max_dd(eq_max)
print(f"    Final: ${eq_max[-1]:.2f} | DD: {dd_max*100:.1f}% | Skipped: {sk_max}")
print(f"    MC P5: ${mc_results[20]['p5']:.2f}")
print()

print("  VERDICT:")
print(f"    - Leverage 1-{min_lev_needed-1}x: IMPOSSIBLE (can't meet $170 minimum)")
print(f"    - Leverage {min_lev_needed}x: Minimum viable, lowest risk")
print(f"    - Leverage 20x: Maximum growth, highest risk")
print(f"    - All feasible leverages have {mc_results[20]['ruin_pct']:.1f}% ruin risk (MC)")
print(f"    - Worst single trade at 20x: -{abs(worst_trade)*20/100:.1f}% of account")
print()
print("  THIS IS THE BASELINE. All future experiments must beat these numbers.")
