"""L1-EXP-004: Kelly Criterion - Optimal Sizing

QUESTION: What is the mathematically optimal sizing for V1.3.2?
- Phase 2 (scaling): What leverage maximizes geometric growth?
- Phase 1 (fixed position): What margin % is optimal?
- Does Kelly confirm our 15% margin / 20x leverage choice?

KELLY CRITERION:
  Classic: f* = W - (1-W)/R  where W=win_rate, R=avg_win/avg_loss
  Generalized (variable returns): maximize E[log(1 + f*r)]

WHY THIS MATTERS:
  - Over-bet -> ruin (leverage too high, drawdowns too deep)
  - Under-bet -> leave money on table (too conservative)
  - Kelly maximizes long-run geometric growth rate
  - Half-Kelly is standard practice (less variance, ~75% of full Kelly growth)

REFERENCE: EXP-003 best config = Iso 15% margin hybrid -> 20x @$25
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
MIN_NOTIONAL = 170.0
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004  # 0.4% for BTCUSDT Tier 1

# ============================================================
# LOAD V1.3.2 TRADES
# ============================================================
config = load_config()
trades = run_backtest(config)
returns = [t.net_profit_bps for t in trades]

wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]
win_rate = len(wins) / len(returns)
avg_win = np.mean(wins)
avg_loss = abs(np.mean(losses))
payoff_ratio = avg_win / avg_loss

print("=" * 100)
print("L1-EXP-004: KELLY CRITERION - Optimal Sizing for V1.3.2")
print("=" * 100)
print()
print(f"  V1.3.2: {len(trades)} trades, {win_rate*100:.1f}% win rate")
print(f"  Avg win:  +{avg_win:.1f} bps")
print(f"  Avg loss: -{avg_loss:.1f} bps")
print(f"  Payoff ratio (R): {payoff_ratio:.2f}")
print(f"  Return range: {min(returns):.1f} to {max(returns):.1f} bps")
print()


def calc_max_dd(equity):
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
# PART 1: CLASSIC KELLY FORMULA
# ============================================================
print("=" * 100)
print("PART 1: CLASSIC KELLY FORMULA")
print("=" * 100)
print()

kelly_classic = win_rate - (1 - win_rate) / payoff_ratio
half_kelly = kelly_classic / 2
quarter_kelly = kelly_classic / 4

print(f"  Classic Kelly formula: f* = W - (1-W)/R")
print(f"  f* = {win_rate:.3f} - {1-win_rate:.3f}/{payoff_ratio:.2f}")
print(f"  f* = {kelly_classic:.4f} = {kelly_classic*100:.1f}%")
print()
print(f"  Full Kelly:    {kelly_classic*100:.1f}%  (aggressive, max growth, high volatility)")
print(f"  Half Kelly:    {half_kelly*100:.1f}%  (standard practice, ~75% growth, ~50% variance)")
print(f"  Quarter Kelly: {quarter_kelly*100:.1f}%  (conservative, ~56% growth, ~25% variance)")
print()
print("  Classic Kelly assumes equal-sized bets. Our trades have variable returns.")
print("  -> Need GENERALIZED Kelly (Part 2)")
print()


# ============================================================
# PART 2: GENERALIZED KELLY - Phase 2 (Scaling Leverage)
# ============================================================
print("=" * 100)
print("PART 2: GENERALIZED KELLY - Optimal Leverage for Phase 2 (Scaling)")
print("=" * 100)
print()
print("  In Phase 2: position = equity * leverage")
print("  Each trade: equity_new = equity * (1 + leverage * return_bps / 10000)")
print("  Geometric growth rate G(L) = (1/N) * SUM[log(1 + L * r_i / 10000)]")
print("  Kelly leverage = argmax G(L)")
print()

# Calculate geometric growth rate for different leverages
leverages = np.arange(1, 61, 0.5)  # 1x to 60x in 0.5 steps
growth_rates = []

for lev in leverages:
    # G(L) = mean of log(1 + L * r / 10000)
    log_returns = []
    for r in returns:
        factor = 1 + lev * r / 10000
        if factor <= 0:
            log_returns.append(-100)  # Ruin -> -infinity (use large negative)
        else:
            log_returns.append(np.log(factor))
    growth_rates.append(np.mean(log_returns))

growth_rates = np.array(growth_rates)

# Find optimal
best_idx = np.argmax(growth_rates)
kelly_leverage = leverages[best_idx]
kelly_growth = growth_rates[best_idx]

# Also find where growth rate goes to zero (max safe leverage)
positive_mask = growth_rates > 0
if not all(positive_mask):
    zero_crossing = leverages[~positive_mask][0] if any(~positive_mask) else leverages[-1]
else:
    zero_crossing = leverages[-1]

print(f"  {'Leverage':>10s} | {'G(L)':>12s} | {'Geometric Mean':>15s} | {'Notes':>20s}")
print(f"  {'-'*70}")

key_leverages = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50]
for lev in key_leverages:
    idx = int((lev - 1) / 0.5)
    if idx < len(growth_rates):
        gr = growth_rates[idx]
        # Geometric mean per trade = exp(G(L))
        geo_mean = np.exp(gr)
        note = ""
        if abs(lev - kelly_leverage) < 0.5:
            note = "<-- KELLY OPTIMAL"
        elif lev == 20:
            note = "<-- Our current"
        print(f"  {lev:>8.0f}x | {gr:>12.6f} | {geo_mean:>14.6f}x | {note}")

print()
print(f"  Kelly Optimal Leverage: {kelly_leverage:.1f}x")
print(f"  Growth rate at Kelly: {kelly_growth:.6f} per trade")
print(f"  Growth rate at 20x:   {growth_rates[int((20-1)/0.5)]:.6f} per trade")
print(f"  Max safe leverage (G>0): {zero_crossing:.1f}x")
print()

# What does this mean for final equity?
# After N trades: equity = start * exp(N * G(L))
n_trades = len(returns)
for lev in [kelly_leverage, kelly_leverage/2, 20, 10]:
    idx = min(int((lev - 1) / 0.5), len(growth_rates) - 1)
    gr = growth_rates[idx]
    expected_final = STARTING_CAPITAL * np.exp(n_trades * gr)
    print(f"  At {lev:>5.1f}x: Expected final = $10 * exp({n_trades} * {gr:.6f}) = ${expected_final:,.0f}")

print()
print("  WARNING: Kelly expected values assume infinite samples.")
print("  Real performance depends on sequence (Part 4 MC will test this)")
print()


# ============================================================
# PART 3: PHASE 1 ANALYSIS - Optimal Margin % for Fixed Position
# ============================================================
print("=" * 100)
print("PART 3: PHASE 1 ANALYSIS - Optimal Margin % (Fixed $170 Position)")
print("=" * 100)
print()
print("  Phase 1 is NOT standard Kelly because position is FIXED ($170)")
print("  P&L per trade = $170 * return_bps / 10000 (constant, independent of equity)")
print("  Margin % affects SURVIVAL, not growth rate:")
print("    - Lower margin -> more liquidation risk but less capital locked")
print("    - Higher margin -> safer but slower (more equity at risk if liquidated)")
print()

# For Phase 1: the question is "what margin % lets us survive to $25?"
# Simulate Phase 1 ONLY (from $10 to $25) with different margin %s

def simulate_phase1_only(rets, position_size, pct, maint_rate=MAINT_MARGIN_RATE,
                         capital=STARTING_CAPITAL):
    """Simulate Phase 1 only: isolated margin, fixed position, until equity >= $25 or end."""
    equity = [capital]
    skipped = 0
    liquidated = 0
    maint_margin = position_size * maint_rate
    reached_25 = False
    bars_to_25 = len(rets)

    for i, r in enumerate(rets):
        eq = equity[-1]

        if eq >= 25.0:
            if not reached_25:
                reached_25 = True
                bars_to_25 = i
            # Stop simulating Phase 1
            equity.append(eq)
            continue

        margin = eq * pct

        # Can't open if margin < maintenance margin
        if margin <= maint_margin:
            equity.append(eq)
            skipped += 1
            continue

        pnl = position_size * (r / 10000)

        # Binance liquidation: lose margin when loss >= margin - maint_margin
        max_loss = margin - maint_margin
        if pnl < -max_loss:
            equity.append(max(eq - margin, 0.01))
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated, reached_25, bars_to_25


# Test margin %s from 5% to 50%
print(f"  Margin % sweep - Phase 1 only (original order, $170 position):")
print()
print(f"  {'Margin%':>8s} | {'Final Eq':>10s} | {'Reach $25?':>10s} | {'Bars to $25':>11s} | {'Skipped':>8s} | {'Liq':>4s} | {'Min Eq':>8s}")
print(f"  {'-'*80}")

for pct in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    eq, sk, liq, reached, bars = simulate_phase1_only(returns, 170, pct)
    final = eq[-1] if not reached else 25.0
    min_eq = min(eq)
    print(f"  {pct*100:>7.0f}% | ${final:>8.2f} | {'YES' if reached else 'NO':>10s} | {bars if reached else '-':>11} | {sk:>8d} | {liq:>4d} | ${min_eq:>7.2f}")

print()

# MC for Phase 1 only - how often do we reach $25?
print("  Monte Carlo: Phase 1 survival rate (1000 shuffled paths)")
print()
print(f"  {'Margin%':>8s} | {'Reach $25':>10s} | {'Median bars':>11s} | {'Med final':>10s} | {'P5 final':>10s} | {'Avg skip':>8s} | {'Avg liq':>8s}")
print(f"  {'-'*80}")

np.random.seed(42)
phase1_results = {}

for pct in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]:
    reach_count = 0
    bars_list = []
    finals = []
    total_skip = 0
    total_liq = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk, liq, reached, bars = simulate_phase1_only(shuffled, 170, pct)
        final = eq[-1] if not reached else 25.0
        finals.append(final)
        total_skip += sk
        total_liq += liq
        if reached:
            reach_count += 1
            bars_list.append(bars)

    phase1_results[pct] = {
        'reach_pct': reach_count / N_SIMS * 100,
        'median_bars': np.median(bars_list) if bars_list else float('inf'),
        'median_final': np.median(finals),
        'p5_final': np.percentile(finals, 5),
        'avg_skip': total_skip / N_SIMS,
        'avg_liq': total_liq / N_SIMS,
    }

    r = phase1_results[pct]
    bars_str = f"{r['median_bars']:.0f}" if r['median_bars'] < 999 else "-"
    print(f"  {pct*100:>7.0f}% | {r['reach_pct']:>8.1f}% | {bars_str:>11s} | ${r['median_final']:>8.2f} | ${r['p5_final']:>8.2f} | {r['avg_skip']:>7.1f} | {r['avg_liq']:>7.1f}")

print()
print("  Phase 1 insight:")
print("  - Lower margin % -> faster growth (less locked up) BUT more liquidation risk")
print("  - Higher margin % -> safer BUT slower, more equity exposed to liquidation loss")
print("  - Margin % in Phase 1 is about SURVIVAL, not Kelly growth optimization")
print()


# ============================================================
# PART 4: FULL HYBRID MC - Sweep Phase 2 Leverage
# ============================================================
print("=" * 100)
print("PART 4: FULL HYBRID MC - Optimal Phase 2 Leverage")
print("=" * 100)
print()
print("  Fix Phase 1 at 15% margin (EXP-003 winner), sweep Phase 2 leverage")
print()


def simulate_binance_pct_hybrid(rets, position_size, pct, threshold, scale_lev,
                                 maint_rate=MAINT_MARGIN_RATE, capital=STARTING_CAPITAL):
    """Phase 1: Isolated with Binance liquidation. Phase 2: Cross scaling."""
    equity = [capital]
    skipped = 0
    liquidated = 0
    maint_margin = position_size * maint_rate

    for r in rets:
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: Isolated margin
            margin = eq * pct

            if margin <= maint_margin:
                equity.append(eq)
                skipped += 1
                continue

            pnl = position_size * (r / 10000)
            max_loss = margin - maint_margin
            if pnl < -max_loss:
                equity.append(max(eq - margin, 0.01))
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            # Phase 2: Cross margin scaling
            pos = eq * scale_lev
            pnl = pos * (r / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


np.random.seed(42)

print(f"  {'Leverage':>10s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'Ruin%':>6s} | {'Notes':>15s}")
print(f"  {'-'*100}")

lev_results = {}
for lev in [5, 7, 10, 12, 15, 17, 20, 22, 25, 28, 30, 35, 40, 50]:
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk, liq = simulate_binance_pct_hybrid(shuffled, 170, 0.15, 25, lev)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruined += 1

    lev_results[lev] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'ruin_pct': ruined / N_SIMS * 100,
    }

    r = lev_results[lev]
    note = ""
    if lev == 20:
        note = "<-- Current"
    elif abs(lev - kelly_leverage) < 1:
        note = "<-- Kelly"
    elif abs(lev - kelly_leverage / 2) < 1:
        note = "<-- Half Kelly"

    print(f"  {lev:>8d}x | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}% | {note}")

# Find best by P5 (worst-case optimization)
best_p5_lev = max(lev_results, key=lambda x: lev_results[x]['p5'])
best_median_lev = max(lev_results, key=lambda x: lev_results[x]['median'])

print()
print(f"  Best by MEDIAN: {best_median_lev}x (${lev_results[best_median_lev]['median']:,.0f})")
print(f"  Best by P5 (safe): {best_p5_lev}x (${lev_results[best_p5_lev]['p5']:,.0f})")
print(f"  Our current 20x: Median ${lev_results[20]['median']:,.0f}, P5 ${lev_results[20]['p5']:,.0f}")
print()


# ============================================================
# PART 5: FULL HYBRID MC - Sweep Phase 1 Margin %
# ============================================================
print("=" * 100)
print("PART 5: FULL HYBRID MC - Optimal Phase 1 Margin %")
print("=" * 100)
print()
print("  Fix Phase 2 at 20x leverage, sweep Phase 1 margin %")
print()

np.random.seed(42)

print(f"  {'Margin%':>10s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'Ruin%':>6s} | {'Avg Skip':>8s}")
print(f"  {'-'*95}")

margin_results = {}
for pct_int in [10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50]:
    pct = pct_int / 100
    finals = []
    max_dds = []
    ruined = 0
    total_skip = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk, liq = simulate_binance_pct_hybrid(shuffled, 170, pct, 25, 20)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        total_skip += sk
        if eq[-1] < 1.0:
            ruined += 1

    margin_results[pct_int] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'ruin_pct': ruined / N_SIMS * 100,
        'avg_skip': total_skip / N_SIMS,
    }

    r = margin_results[pct_int]
    note = " <-- Current" if pct_int == 15 else ""
    print(f"  {pct_int:>8d}%  | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}% | {r['avg_skip']:>7.1f}{note}")

best_p5_margin = max(margin_results, key=lambda x: margin_results[x]['p5'])
best_median_margin = max(margin_results, key=lambda x: margin_results[x]['median'])

print()
print(f"  Best by MEDIAN: {best_median_margin}% (${margin_results[best_median_margin]['median']:,.0f})")
print(f"  Best by P5 (safe): {best_p5_margin}% (${margin_results[best_p5_margin]['p5']:,.0f})")
print(f"  Our current 15%: Median ${margin_results[15]['median']:,.0f}, P5 ${margin_results[15]['p5']:,.0f}")
print()


# ============================================================
# PART 6: 2D GRID - Margin % x Leverage (find global optimum)
# ============================================================
print("=" * 100)
print("PART 6: 2D GRID SEARCH - Margin % x Leverage")
print("=" * 100)
print()
print("  Sweep both Phase 1 margin % AND Phase 2 leverage simultaneously")
print()

np.random.seed(42)

margin_range = [10, 15, 20, 25, 30]
lev_range = [10, 15, 20, 25, 30]

# Store all results
grid_results = {}

for pct_int in margin_range:
    for lev in lev_range:
        pct = pct_int / 100
        finals = []
        max_dds = []
        ruined = 0

        for _ in range(N_SIMS):
            shuffled = list(returns)
            np.random.shuffle(shuffled)
            eq, sk, lq = simulate_binance_pct_hybrid(shuffled, 170, pct, 25, lev)
            finals.append(eq[-1])
            max_dds.append(calc_max_dd(eq))
            if eq[-1] < 1.0:
                ruined += 1

        grid_results[(pct_int, lev)] = {
            'median': np.median(finals),
            'p5': np.percentile(finals, 5),
            'avg_dd': np.mean(max_dds),
            'ruin_pct': ruined / N_SIMS * 100,
        }

# Print MEDIAN grid
print("  MEDIAN Final Equity ($)")
print(f"  {'':>8s}", end="")
for lev in lev_range:
    print(f" | {lev:>8d}x", end="")
print()
print(f"  {'-'*65}")

for pct_int in margin_range:
    print(f"  {pct_int:>6d}%", end="")
    for lev in lev_range:
        r = grid_results[(pct_int, lev)]
        val = f"${r['median']:,.0f}"
        print(f" | {val:>9s}", end="")
    print()

print()

# Print P5 grid
print("  P5 Final Equity (worst 5% luck)")
print(f"  {'':>8s}", end="")
for lev in lev_range:
    print(f" | {lev:>8d}x", end="")
print()
print(f"  {'-'*65}")

for pct_int in margin_range:
    print(f"  {pct_int:>6d}%", end="")
    for lev in lev_range:
        r = grid_results[(pct_int, lev)]
        val = f"${r['p5']:,.0f}"
        print(f" | {val:>9s}", end="")
    print()

print()

# Print Ruin % grid
print("  Ruin % (equity < $1)")
print(f"  {'':>8s}", end="")
for lev in lev_range:
    print(f" | {lev:>8d}x", end="")
print()
print(f"  {'-'*65}")

for pct_int in margin_range:
    print(f"  {pct_int:>6d}%", end="")
    for lev in lev_range:
        r = grid_results[(pct_int, lev)]
        val = f"{r['ruin_pct']:.1f}%"
        print(f" | {val:>9s}", end="")
    print()

# Find best combo
best_p5_combo = max(grid_results, key=lambda x: grid_results[x]['p5'])
best_median_combo = max(grid_results, key=lambda x: grid_results[x]['median'])

print()
print(f"  Best MEDIAN:  {best_median_combo[0]}% margin / {best_median_combo[1]}x leverage -> ${grid_results[best_median_combo]['median']:,.0f}")
print(f"  Best P5:      {best_p5_combo[0]}% margin / {best_p5_combo[1]}x leverage -> P5 ${grid_results[best_p5_combo]['p5']:,.0f}")
print(f"  Our current:  15% margin / 20x leverage -> Median ${grid_results[(15, 20)]['median']:,.0f}, P5 ${grid_results[(15, 20)]['p5']:,.0f}")
print()


# ============================================================
# PART 7: KELLY vs OUR CHOICE - Side by side comparison
# ============================================================
print("=" * 100)
print("PART 7: KELLY vs OUR CHOICE - Side by Side")
print("=" * 100)
print()

# Calculate theoretical Kelly leverage for Phase 2
print(f"  Theoretical Kelly leverage (Phase 2): {kelly_leverage:.1f}x")
print(f"  Half Kelly leverage: {kelly_leverage/2:.1f}x")
print(f"  Our leverage: 20x")
print()

# Run MC for Kelly, Half Kelly, and our config
np.random.seed(42)

comparison_configs = {
    f'Kelly {kelly_leverage:.0f}x (15% margin)': (0.15, kelly_leverage),
    f'Half Kelly {kelly_leverage/2:.0f}x (15% margin)': (0.15, kelly_leverage / 2),
    'Our config: 15% / 20x': (0.15, 20),
    f'Best P5 combo: {best_p5_combo[0]}% / {best_p5_combo[1]}x': (best_p5_combo[0] / 100, best_p5_combo[1]),
}

print(f"  {'Config':>40s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'MaxDD P95':>9s} | {'Ruin%':>6s}")
print(f"  {'-'*100}")

for name, (pct, lev) in comparison_configs.items():
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk, lq = simulate_binance_pct_hybrid(shuffled, 170, pct, 25, lev)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruined += 1

    median = np.median(finals)
    p5 = np.percentile(finals, 5)
    avg_dd = np.mean(max_dds)
    p95_dd = np.percentile(max_dds, 95)
    ruin = ruined / N_SIMS * 100

    print(f"  {name:>40s} | ${median:>10,.0f} | ${p5:>10,.0f} | {avg_dd*100:5.1f}% | {p95_dd*100:7.1f}% | {ruin:5.1f}%")

print()


# ============================================================
# PART 8: PHASE 2 THRESHOLD SENSITIVITY
# ============================================================
print("=" * 100)
print("PART 8: TRANSITION THRESHOLD SENSITIVITY")
print("=" * 100)
print()
print("  Currently: switch from Phase 1 -> Phase 2 at $25")
print("  Does Kelly suggest a different threshold?")
print()

np.random.seed(42)

print(f"  {'Threshold':>10s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*65}")

threshold_results = {}
for thresh in [15, 18, 20, 22, 25, 30, 35, 40, 50]:
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk, lq = simulate_binance_pct_hybrid(shuffled, 170, 0.15, thresh, 20)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruined += 1

    threshold_results[thresh] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'avg_dd': np.mean(max_dds),
        'ruin_pct': ruined / N_SIMS * 100,
    }

    r = threshold_results[thresh]
    note = " <-- Current" if thresh == 25 else ""
    print(f"  ${thresh:>8d} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%{note}")

best_thresh_p5 = max(threshold_results, key=lambda x: threshold_results[x]['p5'])
best_thresh_med = max(threshold_results, key=lambda x: threshold_results[x]['median'])

print()
print(f"  Best by MEDIAN: ${best_thresh_med} threshold")
print(f"  Best by P5:     ${best_thresh_p5} threshold")
print(f"  Current:        $25 threshold")
print()


# ============================================================
# VERDICT
# ============================================================
print("=" * 100)
print("VERDICT: KELLY CRITERION FOR V1.3.2")
print("=" * 100)
print()

print(f"  CLASSIC KELLY:")
print(f"    Full Kelly fraction: {kelly_classic*100:.1f}%")
print(f"    Half Kelly fraction: {half_kelly*100:.1f}%")
print()

print(f"  GENERALIZED KELLY (Phase 2 leverage):")
print(f"    Kelly optimal leverage: {kelly_leverage:.1f}x")
print(f"    Half Kelly leverage: {kelly_leverage/2:.1f}x")
print(f"    Our current leverage: 20x")
print()

our_config = grid_results.get((15, 20), {})
print(f"  OUR CONFIG (15% margin / 20x leverage / $25 threshold):")
print(f"    MC Median: ${our_config.get('median', 0):,.0f}")
print(f"    MC P5:     ${our_config.get('p5', 0):,.0f}")
print(f"    Ruin:      {our_config.get('ruin_pct', 0):.1f}%")
print()

best_combo_r = grid_results.get(best_p5_combo, {})
print(f"  KELLY-OPTIMAL COMBO ({best_p5_combo[0]}% / {best_p5_combo[1]}x by P5):")
print(f"    MC Median: ${best_combo_r.get('median', 0):,.0f}")
print(f"    MC P5:     ${best_combo_r.get('p5', 0):,.0f}")
print(f"    Ruin:      {best_combo_r.get('ruin_pct', 0):.1f}%")
print()

print("  DOES KELLY FIX THE 5% MARGIN ISSUE?")
print("    NO. The 5% margin problem is a Binance HARDWARE constraint:")
print(f"    - Maintenance margin = $170 * 0.4% = $0.68")
print(f"    - 5% of $10 = $0.50 < $0.68 -> CANNOT OPEN TRADE")
print(f"    - Kelly doesn't change Binance's rules, it optimizes within them")
print(f"    - Kelly says: given Binance constraints, use ~{half_kelly*100:.0f}% margin (Half Kelly)")
print()

if our_config and best_combo_r:
    p5_diff = ((best_combo_r['p5'] - our_config['p5']) / our_config['p5'] * 100) if our_config['p5'] > 0 else 0
    med_diff = ((best_combo_r['median'] - our_config['median']) / our_config['median'] * 100) if our_config['median'] > 0 else 0
    print(f"  SHOULD WE CHANGE FROM 15%/20x?")
    print(f"    P5 difference:     {p5_diff:+.1f}%")
    print(f"    Median difference: {med_diff:+.1f}%")
    if abs(p5_diff) < 15 and abs(med_diff) < 15:
        print(f"    -> Small difference. 15%/20x is near-optimal. NO CHANGE NEEDED.")
    else:
        print(f"    -> Meaningful difference. Consider adjusting to {best_p5_combo[0]}%/{best_p5_combo[1]}x.")
