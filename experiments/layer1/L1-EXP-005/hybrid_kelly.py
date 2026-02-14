"""L1-EXP-005: Hybrid Kelly (Continuous + Bayesian + Drawdown Constraint)

QUESTION: Can adaptive Kelly sizing beat fixed leverage?

THREE COMPONENTS:
  1. Bayesian: Beta prior for win rate, updated after each trade
     - Prior from development/training assumptions
     - Posterior mean used for Kelly calculation
  2. Continuous: Recalculate Kelly leverage after EVERY trade
     - Uses running statistics (win rate, payoff ratio)
     - Kelly fraction f = p - (1-p)/R
     - Leverage = f * calibration_multiplier
  3. Drawdown Constraint: Scale leverage down when equity drops from peak
     - dd_scale = max(0, 1 - current_dd / max_dd_limit)
     - At max_dd: leverage -> 0 (stop trading until recovery)

BASELINE COMPARISON (from EXP-002):
  - Cross/20x/$15: Median $41K, P5 $30K, DD 45%
  - Cross/25x/$15: Median $175K, P5 $100K, DD 55%

PHASE 1: Same as EXP-002 (Cross margin, dynamic position, until equity >= $15)
PHASE 2: Hybrid Kelly determines leverage (instead of fixed 20x or 25x)
"""
import sys
sys.path.insert(0, "src")

import math
import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100
PHASE1_THRESHOLD = 15  # switch to Phase 2 at $15 (EXP-002 winner)

# ============================================================
# LOAD V1.3.2 TRADES
# ============================================================
config = load_config()
trades = run_backtest(config)

trade_data = []
for t in trades:
    btc_price = t.entry_price
    qty = max(BINANCE_MIN_QTY, math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)
    position = qty * btc_price
    maint = position * MAINT_MARGIN_RATE
    trade_data.append({
        'bps': t.net_profit_bps,
        'btc_price': btc_price,
        'qty': qty,
        'position': position,
        'maint_margin': maint,
    })

returns = [td['bps'] for td in trade_data]
wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]
win_rate = len(wins) / len(returns)
avg_win = np.mean(wins)
avg_loss = abs(np.mean(losses))
payoff_ratio = avg_win / avg_loss

print("=" * 110)
print("L1-EXP-005: HYBRID KELLY (Continuous + Bayesian + Drawdown Constraint)")
print("=" * 110)
print()
print(f"  V1.3.2: {len(trades)} trades, {win_rate*100:.1f}% win rate")
print(f"  Avg win: +{avg_win:.1f} bps, Avg loss: -{avg_loss:.1f} bps, Payoff: {payoff_ratio:.2f}")
print(f"  Kelly optimal (EXP-004): 49.5x, Half-Kelly: 24.75x")
print(f"  Phase 1 threshold: ${PHASE1_THRESHOLD}")
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
# HYBRID KELLY CLASS
# ============================================================
class HybridKelly:
    """Adaptive leverage using Bayesian Kelly + Drawdown Constraint.

    Components:
      1. Bayesian win rate: Beta(alpha, beta) posterior
      2. Running payoff ratio: avg_win / avg_loss
      3. Kelly fraction: f = p - (1-p)/R
      4. Calibrated leverage: f * multiplier (where multiplier maps f to actual leverage)
      5. Drawdown scaling: max(0, 1 - dd/max_dd)
      6. Half-Kelly option for variance reduction
    """

    def __init__(self, alpha_prior=1.0, beta_prior=1.0, payoff_prior=1.5,
                 max_dd=0.30, half_kelly=True, min_lev=5.0, max_lev=50.0,
                 leverage_multiplier=None):
        # Bayesian prior for win rate
        self.alpha = alpha_prior
        self.beta = beta_prior

        # Running payoff ratio
        self.payoff_prior = payoff_prior
        self.sum_wins = 0.0
        self.n_wins = 0
        self.sum_losses = 0.0
        self.n_losses = 0

        # Drawdown constraint
        self.max_dd = max_dd
        self.peak_equity = None

        # Kelly settings
        self.half_kelly = half_kelly
        self.min_lev = min_lev
        self.max_lev = max_lev

        # Calibration: maps Kelly fraction to leverage
        # From EXP-004: full Kelly fraction ~ 0.427, Kelly leverage ~ 49.5x
        # So multiplier ~ 49.5 / 0.427 ~ 116
        # But that's for the FULL dataset. We use a calibrated value.
        if leverage_multiplier is None:
            self.leverage_multiplier = 116.0  # calibrated from EXP-004
        else:
            self.leverage_multiplier = leverage_multiplier

    def reset(self, alpha_prior=None, beta_prior=None):
        """Reset for new MC path (keep priors, reset observations)."""
        if alpha_prior is not None:
            self.alpha = alpha_prior
        if beta_prior is not None:
            self.beta = beta_prior
        self.sum_wins = 0.0
        self.n_wins = 0
        self.sum_losses = 0.0
        self.n_losses = 0
        self.peak_equity = None

    def update(self, return_bps):
        """Update after observing a trade result."""
        if return_bps > 0:
            self.alpha += 1
            self.sum_wins += return_bps
            self.n_wins += 1
        else:
            self.beta += 1
            self.sum_losses += abs(return_bps)
            self.n_losses += 1

    def get_leverage(self, current_equity):
        """Calculate effective leverage for next trade."""
        # Update peak equity
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # --- Component 1: Bayesian win rate ---
        p = self.alpha / (self.alpha + self.beta)

        # --- Component 2: Running payoff ratio ---
        if self.n_wins > 0 and self.n_losses > 0:
            R = (self.sum_wins / self.n_wins) / (self.sum_losses / self.n_losses)
        else:
            R = self.payoff_prior

        # --- Component 3: Kelly fraction ---
        if R > 0:
            f = p - (1 - p) / R
        else:
            f = 0
        f = max(0, f)

        # Half Kelly for variance reduction
        if self.half_kelly:
            f /= 2

        # Convert fraction to leverage
        kelly_lev = f * self.leverage_multiplier

        # --- Component 4: Drawdown constraint ---
        dd = (self.peak_equity - current_equity) / self.peak_equity
        if self.max_dd > 0:
            dd_scale = max(0, 1 - dd / self.max_dd)
        else:
            dd_scale = 1.0

        # Effective leverage
        effective = kelly_lev * dd_scale
        return max(self.min_lev, min(self.max_lev, effective))

    def get_kelly_fraction(self):
        """Return current Kelly fraction (for diagnostics)."""
        p = self.alpha / (self.alpha + self.beta)
        if self.n_wins > 0 and self.n_losses > 0:
            R = (self.sum_wins / self.n_wins) / (self.sum_losses / self.n_losses)
        else:
            R = self.payoff_prior
        if R > 0:
            f = p - (1 - p) / R
        else:
            f = 0
        return max(0, f)


# ============================================================
# SIMULATION FUNCTIONS
# ============================================================
def simulate_phase1_cross(trade_list, capital=STARTING_CAPITAL):
    """Phase 1: Cross margin, fixed position. Returns (equity, index_where_phase2_starts)."""
    equity = capital
    for i, td in enumerate(trade_list):
        if equity >= PHASE1_THRESHOLD:
            return equity, i

        pos = td['position']
        maint = td['maint_margin']
        margin_req = pos / MAX_LEVERAGE_SETTING

        if equity < margin_req:
            continue

        pnl = pos * (td['bps'] / 10000)
        max_loss = equity - maint
        if pnl < -max_loss:
            equity = 0.01
        else:
            equity = max(equity + pnl, 0.01)

    return equity, len(trade_list)


def simulate_hybrid_kelly(trade_list, kelly, capital=STARTING_CAPITAL):
    """Full simulation: Phase 1 (cross) + Phase 2 (hybrid Kelly)."""
    equity_curve = [capital]
    leverage_curve = []
    equity = capital
    phase2_started = False

    for td in trade_list:
        eq = equity

        if eq < PHASE1_THRESHOLD:
            # Phase 1: Cross margin, fixed position
            pos = td['position']
            maint = td['maint_margin']
            margin_req = pos / MAX_LEVERAGE_SETTING

            if eq < margin_req:
                equity_curve.append(eq)
                leverage_curve.append(0)
                continue

            pnl = pos * (td['bps'] / 10000)
            max_loss = eq - maint
            if pnl < -max_loss:
                equity = 0.01
            else:
                equity = max(eq + pnl, 0.01)

            equity_curve.append(equity)
            leverage_curve.append(0)  # Phase 1, no Kelly
        else:
            # Phase 2: Hybrid Kelly
            if not phase2_started:
                kelly.peak_equity = eq  # reset peak at phase 2 start
                phase2_started = True

            lev = kelly.get_leverage(eq)
            leverage_curve.append(lev)

            pos = eq * lev
            pnl = pos * (td['bps'] / 10000)
            equity = max(eq + pnl, 0.01)
            equity_curve.append(equity)

            # Update Kelly with this trade's result
            kelly.update(td['bps'])

    return equity_curve, leverage_curve


def simulate_fixed_leverage(trade_list, fixed_lev, capital=STARTING_CAPITAL):
    """Baseline: Phase 1 (cross) + Phase 2 (fixed leverage)."""
    equity_curve = [capital]
    equity = capital

    for td in trade_list:
        eq = equity

        if eq < PHASE1_THRESHOLD:
            pos = td['position']
            maint = td['maint_margin']
            margin_req = pos / MAX_LEVERAGE_SETTING

            if eq < margin_req:
                equity_curve.append(eq)
                continue

            pnl = pos * (td['bps'] / 10000)
            max_loss = eq - maint
            if pnl < -max_loss:
                equity = 0.01
            else:
                equity = max(eq + pnl, 0.01)
            equity_curve.append(equity)
        else:
            pos = eq * fixed_lev
            pnl = pos * (td['bps'] / 10000)
            equity = max(eq + pnl, 0.01)
            equity_curve.append(equity)

    return equity_curve


# ============================================================
# PART 1: ORIGINAL ORDER - Diagnostic Run
# ============================================================
print("=" * 110)
print("PART 1: ORIGINAL ORDER - Hybrid Kelly Diagnostic")
print("=" * 110)
print()
print("  Running hybrid Kelly on original trade order to see leverage adaptation...")
print()

# Test with moderate Bayesian prior (60% win, 10 pseudo-observations)
kelly_diag = HybridKelly(
    alpha_prior=6, beta_prior=4,  # 60% win, 10 pseudo-obs
    payoff_prior=2.0,
    max_dd=0.30,
    half_kelly=True,
    min_lev=5, max_lev=50
)

eq_curve, lev_curve = simulate_hybrid_kelly(trade_data, kelly_diag)

# Show leverage at key points
phase2_levs = [(i, l) for i, l in enumerate(lev_curve) if l > 0]
if phase2_levs:
    p2_start = phase2_levs[0][0]
    levs_only = [l for _, l in phase2_levs]
    print(f"  Phase 2 starts at trade {p2_start}")
    print(f"  Leverage range: {min(levs_only):.1f}x - {max(levs_only):.1f}x")
    print(f"  Leverage mean: {np.mean(levs_only):.1f}x")
    print(f"  Final equity: ${eq_curve[-1]:,.2f}")
    print(f"  Max DD: {calc_max_dd(eq_curve)*100:.1f}%")
    print()

    # Show leverage evolution every 20 trades
    print(f"  {'Trade':>6s} | {'Leverage':>8s} | {'Equity':>12s} | {'Kelly f':>8s} | {'DD':>6s}")
    print(f"  {'-'*55}")

    kelly_diag2 = HybridKelly(alpha_prior=6, beta_prior=4, payoff_prior=2.0,
                               max_dd=0.30, half_kelly=True, min_lev=5, max_lev=50)
    eq = STARTING_CAPITAL
    phase2 = False
    trade_num = 0
    for td in trade_data:
        if eq >= PHASE1_THRESHOLD and not phase2:
            phase2 = True
            kelly_diag2.peak_equity = eq

        if phase2:
            lev = kelly_diag2.get_leverage(eq)
            f = kelly_diag2.get_kelly_fraction()
            dd_val = (kelly_diag2.peak_equity - eq) / kelly_diag2.peak_equity if kelly_diag2.peak_equity and kelly_diag2.peak_equity > eq else 0

            if trade_num % 20 == 0 or trade_num == len(trade_data) - 1:
                print(f"  {trade_num:>6d} | {lev:>7.1f}x | ${eq:>10,.2f} | {f:>7.3f} | {dd_val*100:5.1f}%")

            pos = eq * lev
            pnl = pos * (td['bps'] / 10000)
            eq = max(eq + pnl, 0.01)
            kelly_diag2.update(td['bps'])
        else:
            pos = td['position']
            maint = td['maint_margin']
            margin_req = pos / MAX_LEVERAGE_SETTING
            if eq >= margin_req:
                pnl = pos * (td['bps'] / 10000)
                max_loss = eq - maint
                if pnl < -max_loss:
                    eq = 0.01
                else:
                    eq = max(eq + pnl, 0.01)

        trade_num += 1

print()

# Compare to fixed leverage baselines
for fixed_lev in [20, 25]:
    eq_fixed = simulate_fixed_leverage(trade_data, fixed_lev)
    print(f"  Fixed {fixed_lev}x baseline: Final ${eq_fixed[-1]:,.2f}, MaxDD {calc_max_dd(eq_fixed)*100:.1f}%")

print()


# ============================================================
# PART 2: MC GRID - Prior Strength x Max DD x Half/Full Kelly
# ============================================================
print("=" * 110)
print("PART 2: MONTE CARLO GRID - Hybrid Kelly Parameter Sweep")
print("=" * 110)
print()

np.random.seed(42)

# Parameter grid
prior_configs = {
    'Uniform (1,1)':    (1, 1),      # no prior knowledge
    'Weak (6,4)':       (6, 4),      # 60% win, 10 pseudo-obs
    'Moderate (30,20)': (30, 20),    # 60% win, 50 pseudo-obs
    'Strong (60,40)':   (60, 40),    # 60% win, 100 pseudo-obs
}

max_dd_values = [0.20, 0.30, 0.40, 0.50]
kelly_modes = [True, False]  # half kelly vs full kelly

# Also run fixed baselines for comparison
print("  BASELINES (fixed leverage, from EXP-002):")
print(f"  {'Config':>30s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'Ruin':>5s}")
print(f"  {'-'*85}")

baseline_results = {}
for fixed_lev in [20, 25, 30]:
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq = simulate_fixed_leverage(shuffled, fixed_lev)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruined += 1

    baseline_results[fixed_lev] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'avg_dd': np.mean(max_dds),
        'ruin_pct': ruined / N_SIMS * 100,
    }

    r = baseline_results[fixed_lev]
    print(f"  {'Fixed ' + str(fixed_lev) + 'x':>30s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}%")

print()
print()

# Run hybrid Kelly grid
print("  HYBRID KELLY GRID (1000 MC paths each):")
print()
print(f"  {'Prior':>16s} | {'MaxDD':>5s} | {'Kelly':>5s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s} | {'Ruin':>5s}")
print(f"  {'-'*105}")

hybrid_results = {}
config_count = 0
total_configs = len(prior_configs) * len(max_dd_values) * len(kelly_modes)

for prior_name, (alpha_p, beta_p) in prior_configs.items():
    for max_dd in max_dd_values:
        for half_kelly in kelly_modes:
            config_count += 1
            kelly_label = "Half" if half_kelly else "Full"
            key = (prior_name, max_dd, kelly_label)

            finals = []
            max_dds = []
            avg_levs = []
            ruined = 0

            kelly_obj = HybridKelly(
                alpha_prior=alpha_p, beta_prior=beta_p,
                payoff_prior=2.0,
                max_dd=max_dd,
                half_kelly=half_kelly,
                min_lev=5, max_lev=50
            )

            for _ in range(N_SIMS):
                shuffled = list(trade_data)
                np.random.shuffle(shuffled)

                # Reset kelly for each MC path (keep prior, reset observations)
                kelly_obj.reset(alpha_prior=alpha_p, beta_prior=beta_p)

                eq_curve, lev_curve = simulate_hybrid_kelly(shuffled, kelly_obj)
                finals.append(eq_curve[-1])
                max_dds.append(calc_max_dd(eq_curve))

                # Average leverage in Phase 2
                p2_levs = [l for l in lev_curve if l > 0]
                if p2_levs:
                    avg_levs.append(np.mean(p2_levs))

                if eq_curve[-1] < 1.0:
                    ruined += 1

            hybrid_results[key] = {
                'median': np.median(finals),
                'p5': np.percentile(finals, 5),
                'p25': np.percentile(finals, 25),
                'p75': np.percentile(finals, 75),
                'avg_dd': np.mean(max_dds),
                'avg_lev': np.mean(avg_levs) if avg_levs else 0,
                'ruin_pct': ruined / N_SIMS * 100,
            }

            r = hybrid_results[key]
            print(f"  {prior_name:>16s} | {max_dd*100:4.0f}% | {kelly_label:>5s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

    print()  # separate prior groups

print(f"  Done: {total_configs} configs x {N_SIMS} paths = {total_configs * N_SIMS:,} simulations")
print()


# ============================================================
# PART 3: BEST HYBRID vs FIXED BASELINES
# ============================================================
print("=" * 110)
print("PART 3: BEST HYBRID CONFIGS vs FIXED BASELINES")
print("=" * 110)
print()

# Sort by P5
sorted_hybrid = sorted(hybrid_results.items(), key=lambda x: x[1]['p5'], reverse=True)

print("  TOP 10 Hybrid Kelly configs by P5:")
print(f"  {'#':>3s} | {'Prior':>16s} | {'MaxDD':>5s} | {'Kelly':>5s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s}")
print(f"  {'-'*85}")

for i, (key, r) in enumerate(sorted_hybrid[:10]):
    prior_name, max_dd, kelly_label = key
    print(f"  {i+1:>3d} | {prior_name:>16s} | {max_dd*100:4.0f}% | {kelly_label:>5s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x")

print()
print("  COMPARISON TABLE:")
print()

# Best hybrid by P5
best_key, best_r = sorted_hybrid[0]
best_prior, best_dd, best_mode = best_key

# Best hybrid by median
best_med_key = max(hybrid_results, key=lambda k: hybrid_results[k]['median'])
best_med_r = hybrid_results[best_med_key]

# Best hybrid with DD < 50%
safe_configs = {k: v for k, v in hybrid_results.items() if v['avg_dd'] < 0.50}
if safe_configs:
    best_safe_key = max(safe_configs, key=lambda k: safe_configs[k]['p5'])
    best_safe_r = safe_configs[best_safe_key]
else:
    best_safe_key = best_key
    best_safe_r = best_r

print(f"  {'Config':>40s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s}")
print(f"  {'-'*90}")

# Fixed baselines
for lev in [20, 25, 30]:
    r = baseline_results[lev]
    print(f"  {'Fixed ' + str(lev) + 'x (baseline)':>40s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {lev:5.1f}x")

print(f"  {'-'*90}")

# Best hybrid configs
label = f"Best P5: {best_prior}/{best_dd*100:.0f}%DD/{best_mode}"
print(f"  {label:>40s} | ${best_r['median']:>10,.0f} | ${best_r['p5']:>10,.0f} | {best_r['avg_dd']*100:5.1f}% | {best_r['avg_lev']:5.1f}x")

label = f"Best Med: {best_med_key[0]}/{best_med_key[1]*100:.0f}%DD/{best_med_key[2]}"
print(f"  {label:>40s} | ${best_med_r['median']:>10,.0f} | ${best_med_r['p5']:>10,.0f} | {best_med_r['avg_dd']*100:5.1f}% | {best_med_r['avg_lev']:5.1f}x")

label = f"Safe: {best_safe_key[0]}/{best_safe_key[1]*100:.0f}%DD/{best_safe_key[2]}"
print(f"  {label:>40s} | ${best_safe_r['median']:>10,.0f} | ${best_safe_r['p5']:>10,.0f} | {best_safe_r['avg_dd']*100:5.1f}% | {best_safe_r['avg_lev']:5.1f}x")

print()


# ============================================================
# PART 4: DRAWDOWN CONSTRAINT DEEP DIVE
# ============================================================
print("=" * 110)
print("PART 4: DRAWDOWN CONSTRAINT - How does it protect during bad sequences?")
print("=" * 110)
print()

# Use the best prior/kelly combo, sweep max_dd more finely
best_prior_key = sorted_hybrid[0][0][0]  # best prior name
best_alpha, best_beta = prior_configs[best_prior_key]
best_half = sorted_hybrid[0][0][2] == "Half"

print(f"  Using: Prior={best_prior_key}, Kelly={'Half' if best_half else 'Full'}")
print(f"  Sweeping max drawdown limit: 15% to 60%")
print()

np.random.seed(42)

print(f"  {'MaxDD':>6s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'AvgLev':>7s} | {'MinLev%':>8s}")
print(f"  {'-'*80}")

dd_sweep_results = {}
for max_dd_pct in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
    max_dd = max_dd_pct / 100

    kelly_obj = HybridKelly(
        alpha_prior=best_alpha, beta_prior=best_beta,
        payoff_prior=2.0, max_dd=max_dd, half_kelly=best_half,
        min_lev=5, max_lev=50
    )

    finals = []
    max_dds = []
    avg_levs = []
    min_lev_pcts = []  # % of time at minimum leverage

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        kelly_obj.reset(alpha_prior=best_alpha, beta_prior=best_beta)

        eq_curve, lev_curve = simulate_hybrid_kelly(shuffled, kelly_obj)
        finals.append(eq_curve[-1])
        max_dds.append(calc_max_dd(eq_curve))

        p2_levs = [l for l in lev_curve if l > 0]
        if p2_levs:
            avg_levs.append(np.mean(p2_levs))
            at_min = sum(1 for l in p2_levs if l <= 5.5) / len(p2_levs) * 100
            min_lev_pcts.append(at_min)

    dd_sweep_results[max_dd_pct] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'avg_lev': np.mean(avg_levs) if avg_levs else 0,
        'min_lev_pct': np.mean(min_lev_pcts) if min_lev_pcts else 0,
    }

    r = dd_sweep_results[max_dd_pct]
    print(f"  {max_dd_pct:>5d}% | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['min_lev_pct']:6.1f}%")

print()


# ============================================================
# PART 5: PRIOR STRENGTH IMPACT
# ============================================================
print("=" * 110)
print("PART 5: BAYESIAN PRIOR STRENGTH - How much does it matter?")
print("=" * 110)
print()

# Fix max_dd at best value, sweep prior more finely
best_max_dd = sorted_hybrid[0][0][1]

print(f"  Using: MaxDD={best_max_dd*100:.0f}%, Kelly={'Half' if best_half else 'Full'}")
print(f"  Sweeping prior strength (all at 60% win rate assumption)")
print()

np.random.seed(42)

prior_sweep = [
    ('None (1,1)', 1, 1),
    ('Tiny (3,2)', 3, 2),
    ('Weak (6,4)', 6, 4),
    ('Light (12,8)', 12, 8),
    ('Med (30,20)', 30, 20),
    ('Strong (60,40)', 60, 40),
    ('Heavy (120,80)', 120, 80),
]

print(f"  {'Prior':>18s} | {'PseudoN':>8s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s} | {'Lev@t10':>8s} | {'Lev@t100':>8s}")
print(f"  {'-'*105}")

for prior_label, ap, bp in prior_sweep:
    kelly_obj = HybridKelly(
        alpha_prior=ap, beta_prior=bp,
        payoff_prior=2.0, max_dd=best_max_dd, half_kelly=best_half,
        min_lev=5, max_lev=50
    )

    finals = []
    max_dds = []
    avg_levs = []
    lev_at_10 = []  # leverage after 10 Phase 2 trades
    lev_at_100 = []  # leverage after 100 Phase 2 trades

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        kelly_obj.reset(alpha_prior=ap, beta_prior=bp)

        eq_curve, lev_curve = simulate_hybrid_kelly(shuffled, kelly_obj)
        finals.append(eq_curve[-1])
        max_dds.append(calc_max_dd(eq_curve))

        p2_levs = [l for l in lev_curve if l > 0]
        if p2_levs:
            avg_levs.append(np.mean(p2_levs))
            if len(p2_levs) >= 10:
                lev_at_10.append(p2_levs[9])
            if len(p2_levs) >= 100:
                lev_at_100.append(p2_levs[99])

    pseudo_n = ap + bp
    median_val = np.median(finals)
    p5_val = np.percentile(finals, 5)
    avg_dd = np.mean(max_dds)
    avg_lev = np.mean(avg_levs) if avg_levs else 0
    l10 = np.mean(lev_at_10) if lev_at_10 else 0
    l100 = np.mean(lev_at_100) if lev_at_100 else 0

    print(f"  {prior_label:>18s} | {pseudo_n:>8d} | ${median_val:>10,.0f} | ${p5_val:>10,.0f} | {avg_dd*100:5.1f}% | {avg_lev:5.1f}x | {l10:6.1f}x | {l100:6.1f}x")

print()


# ============================================================
# PART 6: HYBRID KELLY vs FIXED - Worst Case Analysis
# ============================================================
print("=" * 110)
print("PART 6: WORST CASE ANALYSIS - Does drawdown constraint protect?")
print("=" * 110)
print()

np.random.seed(42)

# Compare best hybrid vs fixed 20x and fixed 25x on worst paths
print("  Simulating 1000 paths, tracking worst outcomes...")
print()

# Run all three: hybrid, fixed 20x, fixed 25x
best_hybrid_key = sorted_hybrid[0][0]
bp_name, bp_dd, bp_mode = best_hybrid_key
bp_alpha, bp_beta = prior_configs[bp_name]
bp_half = bp_mode == "Half"

kelly_best = HybridKelly(
    alpha_prior=bp_alpha, beta_prior=bp_beta,
    payoff_prior=2.0, max_dd=bp_dd, half_kelly=bp_half,
    min_lev=5, max_lev=50
)

hybrid_finals = []
fixed20_finals = []
fixed25_finals = []
hybrid_dds = []
fixed20_dds = []
fixed25_dds = []

for _ in range(N_SIMS):
    shuffled = list(trade_data)
    np.random.shuffle(shuffled)

    # Hybrid
    kelly_best.reset(alpha_prior=bp_alpha, beta_prior=bp_beta)
    eq_h, _ = simulate_hybrid_kelly(shuffled, kelly_best)
    hybrid_finals.append(eq_h[-1])
    hybrid_dds.append(calc_max_dd(eq_h))

    # Fixed 20x
    eq_20 = simulate_fixed_leverage(shuffled, 20)
    fixed20_finals.append(eq_20[-1])
    fixed20_dds.append(calc_max_dd(eq_20))

    # Fixed 25x
    eq_25 = simulate_fixed_leverage(shuffled, 25)
    fixed25_finals.append(eq_25[-1])
    fixed25_dds.append(calc_max_dd(eq_25))

# Percentile comparison
print(f"  {'Metric':>20s} | {'Hybrid Kelly':>15s} | {'Fixed 20x':>15s} | {'Fixed 25x':>15s}")
print(f"  {'-'*75}")

for label, pct in [('P1 (worst 1%)', 1), ('P5 (worst 5%)', 5), ('P10', 10), ('P25', 25), ('P50 (median)', 50), ('P75', 75), ('P90', 90)]:
    h = np.percentile(hybrid_finals, pct)
    f20 = np.percentile(fixed20_finals, pct)
    f25 = np.percentile(fixed25_finals, pct)
    print(f"  {label:>20s} | ${h:>13,.0f} | ${f20:>13,.0f} | ${f25:>13,.0f}")

print()

# DD comparison
print(f"  {'DD Metric':>20s} | {'Hybrid Kelly':>15s} | {'Fixed 20x':>15s} | {'Fixed 25x':>15s}")
print(f"  {'-'*75}")
for label, vals_h, vals_20, vals_25 in [
    ('Avg MaxDD', hybrid_dds, fixed20_dds, fixed25_dds),
]:
    print(f"  {label:>20s} | {np.mean(vals_h)*100:13.1f}% | {np.mean(vals_20)*100:13.1f}% | {np.mean(vals_25)*100:13.1f}%")
    print(f"  {'P95 MaxDD':>20s} | {np.percentile(vals_h, 95)*100:13.1f}% | {np.percentile(vals_20, 95)*100:13.1f}% | {np.percentile(vals_25, 95)*100:13.1f}%")
    print(f"  {'P99 MaxDD':>20s} | {np.percentile(vals_h, 99)*100:13.1f}% | {np.percentile(vals_20, 99)*100:13.1f}% | {np.percentile(vals_25, 99)*100:13.1f}%")

print()

# How often does hybrid beat fixed?
hybrid_beats_20 = sum(1 for h, f in zip(hybrid_finals, fixed20_finals) if h > f) / N_SIMS * 100
hybrid_beats_25 = sum(1 for h, f in zip(hybrid_finals, fixed25_finals) if h > f) / N_SIMS * 100

print(f"  Hybrid beats Fixed 20x: {hybrid_beats_20:.1f}% of paths")
print(f"  Hybrid beats Fixed 25x: {hybrid_beats_25:.1f}% of paths")
print()


# ============================================================
# PART 7: NO DRAWDOWN CONSTRAINT (ablation)
# ============================================================
print("=" * 110)
print("PART 7: ABLATION - Each component's contribution")
print("=" * 110)
print()
print("  Testing: remove one component at a time to measure its impact")
print()

np.random.seed(42)

ablation_configs = {
    'Full Hybrid (best)': {
        'alpha': bp_alpha, 'beta': bp_beta,
        'max_dd': bp_dd, 'half_kelly': bp_half,
    },
    'No DD constraint': {
        'alpha': bp_alpha, 'beta': bp_beta,
        'max_dd': 1.0,  # effectively disabled (100% DD needed)
        'half_kelly': bp_half,
    },
    'No Bayesian (flat prior)': {
        'alpha': 1, 'beta': 1,
        'max_dd': bp_dd, 'half_kelly': bp_half,
    },
    'Full Kelly (no half)': {
        'alpha': bp_alpha, 'beta': bp_beta,
        'max_dd': bp_dd, 'half_kelly': False,
    },
    'Only DD (fixed 25x + DD)': None,  # special case
}

print(f"  {'Config':>30s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s}")
print(f"  {'-'*80}")

for config_name, params in ablation_configs.items():
    finals = []
    max_dds = []
    avg_levs = []

    if params is None:
        # Special: fixed 25x with DD constraint
        for _ in range(N_SIMS):
            shuffled = list(trade_data)
            np.random.shuffle(shuffled)

            equity_curve = [STARTING_CAPITAL]
            equity = STARTING_CAPITAL
            peak = STARTING_CAPITAL

            for td in shuffled:
                eq = equity
                if eq < PHASE1_THRESHOLD:
                    pos = td['position']
                    maint = td['maint_margin']
                    margin_req = pos / MAX_LEVERAGE_SETTING
                    if eq < margin_req:
                        equity_curve.append(eq)
                        continue
                    pnl = pos * (td['bps'] / 10000)
                    max_loss = eq - maint
                    if pnl < -max_loss:
                        equity = 0.01
                    else:
                        equity = max(eq + pnl, 0.01)
                    equity_curve.append(equity)
                else:
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak
                    dd_scale = max(0, 1 - dd / bp_dd)
                    lev = max(5, 25 * dd_scale)
                    avg_levs.append(lev)

                    pos = eq * lev
                    pnl = pos * (td['bps'] / 10000)
                    equity = max(eq + pnl, 0.01)
                    equity_curve.append(equity)

            finals.append(equity_curve[-1])
            max_dds.append(calc_max_dd(equity_curve))
    else:
        kelly_obj = HybridKelly(
            alpha_prior=params['alpha'], beta_prior=params['beta'],
            payoff_prior=2.0, max_dd=params['max_dd'],
            half_kelly=params['half_kelly'], min_lev=5, max_lev=50
        )

        for _ in range(N_SIMS):
            shuffled = list(trade_data)
            np.random.shuffle(shuffled)
            kelly_obj.reset(alpha_prior=params['alpha'], beta_prior=params['beta'])

            eq_curve, lev_curve = simulate_hybrid_kelly(shuffled, kelly_obj)
            finals.append(eq_curve[-1])
            max_dds.append(calc_max_dd(eq_curve))
            p2_levs = [l for l in lev_curve if l > 0]
            if p2_levs:
                avg_levs.extend(p2_levs)

    median_val = np.median(finals)
    p5_val = np.percentile(finals, 5)
    avg_dd = np.mean(max_dds)
    avg_lev = np.mean(avg_levs) if avg_levs else 0

    print(f"  {config_name:>30s} | ${median_val:>10,.0f} | ${p5_val:>10,.0f} | {avg_dd*100:5.1f}% | {avg_lev:5.1f}x")

print()


# ============================================================
# VERDICT
# ============================================================
print("=" * 110)
print("VERDICT: HYBRID KELLY FOR V1.3.2")
print("=" * 110)
print()

best_key, best_r = sorted_hybrid[0]
best_prior_name, best_max_dd, best_kelly_mode = best_key

print(f"  BEST HYBRID CONFIG (by P5):")
print(f"    Prior: {best_prior_name}")
print(f"    Max DD limit: {best_max_dd*100:.0f}%")
print(f"    Kelly mode: {best_kelly_mode}")
print(f"    Avg leverage: {best_r['avg_lev']:.1f}x")
print(f"    MC Median: ${best_r['median']:,.0f}")
print(f"    MC P5: ${best_r['p5']:,.0f}")
print(f"    Avg DD: {best_r['avg_dd']*100:.1f}%")
print()

print("  COMPARISON TO FIXED BASELINES:")
print()
print(f"    {'':>25s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s}")
print(f"    {'-'*65}")
print(f"    {'Fixed 20x':>25s} | ${baseline_results[20]['median']:>10,.0f} | ${baseline_results[20]['p5']:>10,.0f} | {baseline_results[20]['avg_dd']*100:5.1f}%")
print(f"    {'Fixed 25x':>25s} | ${baseline_results[25]['median']:>10,.0f} | ${baseline_results[25]['p5']:>10,.0f} | {baseline_results[25]['avg_dd']*100:5.1f}%")
print(f"    {'Fixed 30x':>25s} | ${baseline_results[30]['median']:>10,.0f} | ${baseline_results[30]['p5']:>10,.0f} | {baseline_results[30]['avg_dd']*100:5.1f}%")
print(f"    {'Hybrid Kelly':>25s} | ${best_r['median']:>10,.0f} | ${best_r['p5']:>10,.0f} | {best_r['avg_dd']*100:5.1f}%")
print()

# Calculate improvement
p5_vs_20 = (best_r['p5'] - baseline_results[20]['p5']) / baseline_results[20]['p5'] * 100
p5_vs_25 = (best_r['p5'] - baseline_results[25]['p5']) / baseline_results[25]['p5'] * 100
dd_vs_20 = (best_r['avg_dd'] - baseline_results[20]['avg_dd']) * 100
dd_vs_25 = (best_r['avg_dd'] - baseline_results[25]['avg_dd']) * 100

print(f"  vs Fixed 20x: P5 {p5_vs_20:+.1f}%, DD {dd_vs_20:+.1f}pp")
print(f"  vs Fixed 25x: P5 {p5_vs_25:+.1f}%, DD {dd_vs_25:+.1f}pp")
print()

# Final recommendation
if best_r['p5'] > baseline_results[25]['p5'] and best_r['avg_dd'] < baseline_results[25]['avg_dd']:
    print("  VERDICT: Hybrid Kelly WINS - better returns AND lower drawdown")
elif best_r['p5'] > baseline_results[25]['p5']:
    print("  VERDICT: Hybrid Kelly has better P5 but higher drawdown - TRADE-OFF")
elif best_r['avg_dd'] < baseline_results[25]['avg_dd']:
    print("  VERDICT: Hybrid Kelly has lower drawdown but lower P5 - PROTECTION focused")
else:
    print("  VERDICT: Hybrid Kelly does NOT beat fixed leverage - KEEP FIXED")

print()
print("  Hybrid Kelly beats fixed 20x: {:.1f}% of MC paths".format(hybrid_beats_20))
print("  Hybrid Kelly beats fixed 25x: {:.1f}% of MC paths".format(hybrid_beats_25))
print()
