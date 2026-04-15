"""L1-EXP-008: Dynamic Risk Calculator (v2 - with edge cases)

QUESTION: Can a dynamic position sizer that adapts to wallet + BTC price + strategy stats
           beat fixed $/step on BOTH train and OOS data?

KEY FIXES from v1:
  - Test across multiple BTC prices (not just $97K)
  - NEVER SKIP: always trade minimum (accept risk at small wallet)
  - 3 modes: FORCED_MIN (high risk), SURVIVAL (P95 risk), GROWTH (worst-case protected)
  - Test across multiple starting capitals ($5, $10, $20, $50, $100)

DESIGN:
  Per-trade logic:
    1. Calculate Binance minimum qty (from $100 min notional + BTC price)
    2. Calculate Kelly-optimal qty
    3. Calculate risk-capped qty (worst-case loss < survival_limit % of wallet)
    4. Decision:
       - If even min_qty worst loss > wallet -> FORCED_MIN (trade minimum, accept risk)
       - If min_qty P95 loss > max_risk -> FORCED_MIN (trade minimum, EV is positive)
       - If min_qty worst loss > max_risk -> SURVIVAL (size by P95)
       - If min_qty worst loss <= max_risk -> GROWTH (size by worst, Kelly-optimal)

TESTS:
  Part 1: BTC price impact on minimum position & risk
  Part 2: Calculator behavior across wallet x BTC price matrix
  Part 3: MC simulation - Dynamic (no-skip) vs Fixed on TRAIN
  Part 4: MC simulation - Dynamic (no-skip) vs Fixed on OOS
  Part 5: Starting capital sweep
  Part 6: Verdict
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
from engine.backtest import run_backtest
from engine.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
LEVERAGE = 125
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100
N_SIMS = 1000

# ============================================================
# LOAD TRADES
# ============================================================
config = load_config()

print("Loading TRAIN trades (2020-2023)...")
train_raw = run_backtest(config, start="2020-01-01", end="2023-12-31")
print("Loading OOS trades (2024-2025)...")
oos_raw = run_backtest(config, start="2024-01-01", end="2025-12-31")


def calc_min_qty(btc_price):
    """Calculate Binance minimum qty for a given BTC price."""
    return max(BINANCE_MIN_QTY,
               math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)


def make_trade_data(trades_raw):
    data = []
    for t in trades_raw:
        btc_price = t.entry_price
        qty_min = calc_min_qty(btc_price)
        data.append({
            'bps': t.net_profit_bps,
            'btc_price': btc_price,
            'qty_min': qty_min,
            'pos_min': qty_min * btc_price,
        })
    return data


train_data = make_trade_data(train_raw)
oos_data = make_trade_data(oos_raw)


def calc_stats(data):
    returns = [td['bps'] for td in data]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    p = len(wins) / len(returns)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    b = avg_win / avg_loss
    kelly_f = p - (1 - p) / b
    quarter_kelly = kelly_f / 4
    abs_losses = [abs(r) for r in losses]
    return {
        'n': len(returns),
        'win_rate': p,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'worst_loss': abs(min(returns)),
        'p95_loss': np.percentile(abs_losses, 95) if abs_losses else 0,
        'p90_loss': np.percentile(abs_losses, 90) if abs_losses else 0,
        'best_win': max(returns),
        'payoff': b,
        'kelly_f': kelly_f,
        'quarter_kelly': quarter_kelly,
        'total_bps': sum(returns),
    }


train_stats = calc_stats(train_data)
oos_stats = calc_stats(oos_data)

print()
print("=" * 100)
print("L1-EXP-008 v2: DYNAMIC RISK CALCULATOR (WITH EDGE CASES)")
print("=" * 100)
print()
print(f"  TRAIN: {train_stats['n']}t, {train_stats['win_rate']*100:.1f}% win, "
      f"avg_loss -{train_stats['avg_loss']:.0f}, P95 -{train_stats['p95_loss']:.0f}, "
      f"worst -{train_stats['worst_loss']:.0f}, Kelly f={train_stats['kelly_f']:.3f}")
print(f"  OOS:   {oos_stats['n']}t, {oos_stats['win_rate']*100:.1f}% win, "
      f"avg_loss -{oos_stats['avg_loss']:.0f}, P95 -{oos_stats['p95_loss']:.0f}, "
      f"worst -{oos_stats['worst_loss']:.0f}, Kelly f={oos_stats['kelly_f']:.3f}")
print()


# ============================================================
# PART 1: BTC PRICE IMPACT ON MINIMUM POSITION
# ============================================================
print("=" * 100)
print("PART 1: BTC PRICE IMPACT ON MINIMUM POSITION & RISK")
print("=" * 100)
print()

btc_prices = [30000, 40000, 50000, 60000, 70000, 80000, 90000, 95000, 97000,
              100000, 105000, 110000, 120000, 150000, 200000]

print(f"  {'BTC Price':>12s} | {'Min Qty':>8s} | {'Min Pos':>10s} | {'Margin@125x':>12s} | "
      f"{'WC Loss(865bp)':>15s} | {'%of $10':>8s} | {'P95 Loss(306bp)':>16s} | {'%of $10':>8s} | "
      f"{'Liq Buffer':>10s}")
print(f"  {'-'*120}")

for btc in btc_prices:
    mq = calc_min_qty(btc)
    pos = mq * btc
    margin = pos / LEVERAGE
    wc_loss = pos * train_stats['worst_loss'] / 10000
    wc_pct = wc_loss / 10 * 100
    p95_loss = pos * train_stats['p95_loss'] / 10000
    p95_pct = p95_loss / 10 * 100
    # Liquidation buffer: how much can price move before liquidation at $10 wallet
    # max_loss_before_liq = wallet - maint_margin = $10 - pos*0.004
    maint = pos * MAINT_MARGIN_RATE
    liq_buffer_bps = (10 - maint) / pos * 10000 if pos > 0 else 0
    marker = " <-- SWEET SPOT" if mq == 0.001 and btc >= 100000 and btc <= 100000 else ""
    if btc == 100000:
        marker = " <-- SWEET SPOT (0.001 = exactly $100)"
    print(f"  ${btc:>10,} | {mq:.3f} | ${pos:>8,.0f} | ${margin:>10,.2f} | "
          f"${wc_loss:>13,.2f} | {wc_pct:>6.0f}% | ${p95_loss:>14,.2f} | {p95_pct:>6.0f}% | "
          f"{liq_buffer_bps:>8.0f}bp{marker}")

print()
print("  KEY: At BTC >= $100K, min qty drops to 0.001 BTC ($100 position)")
print("       At BTC < $100K, min qty = 0.002+ BTC ($194+ position) -- DOUBLE the risk!")
print()


# ============================================================
# THE DYNAMIC RISK CALCULATOR (v2 - NO SKIP)
# ============================================================
def calculate_qty(wallet, btc_price, stats, survive_n=5, survive_pct=0.10):
    """Dynamic position sizing - NEVER skips.

    3 modes:
      FORCED_MIN: wallet too small for proper sizing, trade minimum anyway (EV > 0)
      SURVIVAL: size using P95 loss (accept 5% tail risk)
      GROWTH: size using worst_loss (fully protected), Kelly-optimal

    Returns: (qty, mode, risk_pct)
    """
    # 1. Binance minimum
    min_qty = calc_min_qty(btc_price)
    min_position = min_qty * btc_price

    # 2. Survival limit
    max_risk = 1 - survive_pct ** (1 / survive_n)

    worst_loss_frac = stats['worst_loss'] / 10000
    p95_loss_frac = stats['p95_loss'] / 10000

    # 3. Risk checks at minimum position
    min_worst_risk = min_position * worst_loss_frac / wallet if wallet > 0 else 999
    min_p95_risk = min_position * p95_loss_frac / wallet if wallet > 0 else 999

    kelly_frac = stats['quarter_kelly']
    avg_loss_frac = stats['avg_loss'] / 10000

    # 4. Decision tree (NO SKIP - always trade at least minimum)
    if min_p95_risk > max_risk:
        # Even P95 loss exceeds survival limit at minimum position
        # FORCED_MIN: trade minimum anyway - EV is positive, only way to grow
        return min_qty, 'FORCED_MIN', min_worst_risk

    if min_worst_risk > max_risk:
        # SURVIVAL MODE: size using P95 loss
        if kelly_frac > 0:
            kelly_position = kelly_frac * wallet / avg_loss_frac
            kelly_qty = kelly_position / btc_price
            kelly_qty = math.floor(kelly_qty / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE
        else:
            kelly_qty = min_qty

        max_position = wallet * max_risk / p95_loss_frac
        max_qty = math.floor(max_position / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE

        qty = min(kelly_qty, max_qty)
        qty = max(qty, min_qty)

        final_risk = qty * btc_price * worst_loss_frac / wallet
        return qty, 'SURVIVAL', final_risk

    else:
        # GROWTH MODE: size using worst_loss (fully protected)
        if kelly_frac > 0:
            kelly_position = kelly_frac * wallet / avg_loss_frac
            kelly_qty = kelly_position / btc_price
            kelly_qty = math.floor(kelly_qty / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE
        else:
            kelly_qty = min_qty

        max_position = wallet * max_risk / worst_loss_frac
        max_qty = math.floor(max_position / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE

        qty = min(kelly_qty, max_qty)
        qty = max(qty, min_qty)

        final_risk = qty * btc_price * worst_loss_frac / wallet
        return qty, 'GROWTH', final_risk


# ============================================================
# PART 2: CALCULATOR BEHAVIOR — WALLET x BTC PRICE MATRIX
# ============================================================
print("=" * 100)
print("PART 2: CALCULATOR BEHAVIOR — WALLET x BTC PRICE MATRIX (S5_10%)")
print("=" * 100)
print()
print(f"  Survive 5 worst trades with 10% left -> max_risk = {(1 - 0.10**(1/5))*100:.1f}%")
print(f"  Stats: TRAIN (worst={train_stats['worst_loss']:.0f}bp, P95={train_stats['p95_loss']:.0f}bp)")
print()

test_wallets = [5, 10, 15, 20, 30, 50, 100, 500, 1000]
test_btc_prices = [30000, 50000, 75000, 97000, 100000, 120000, 150000]

# Header
header = f"  {'Wallet':>8s} |"
for btc in test_btc_prices:
    header += f" ${btc//1000}K".rjust(16) + " |"
print(header)
print(f"  {'-' * (10 + 17 * len(test_btc_prices))}")

for w in test_wallets:
    row = f"  ${w:>6,} |"
    for btc in test_btc_prices:
        qty, mode, risk = calculate_qty(w, btc, train_stats, survive_n=5, survive_pct=0.10)
        pos = qty * btc
        mode_short = mode[0]  # F, S, or G
        row += f" {mode_short} {qty:.3f} {risk*100:>4.0f}%".rjust(16) + " |"
    print(row)

print()
print("  Legend: F=FORCED_MIN (high risk), S=SURVIVAL (P95 risk), G=GROWTH (safe)")
print()

# Show transition wallets at each BTC price
print("  Transition wallets:")
for btc in test_btc_prices:
    survival_w, growth_w = None, None
    for w in range(1, 5000):
        qty, mode, risk = calculate_qty(w, btc, train_stats, survive_n=5, survive_pct=0.10)
        if mode != 'FORCED_MIN' and survival_w is None:
            survival_w = w
        if mode == 'GROWTH' and growth_w is None:
            growth_w = w
            break
    min_pos = calc_min_qty(btc) * btc
    print(f"    BTC ${btc:>7,}: min_pos=${min_pos:>6,.0f} | "
          f"FORCED->SURVIVAL at ${survival_w or 'N/A'} | "
          f"SURVIVAL->GROWTH at ${growth_w or '>5000'}")

print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
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


def simulate_dynamic(trade_list, stats, survive_n=5, survive_pct=0.10, capital=STARTING_CAPITAL):
    """Dynamic risk calculator simulation - NEVER skips."""
    equity = [capital]
    modes = {'FORCED_MIN': 0, 'SURVIVAL': 0, 'GROWTH': 0}

    for td in trade_list:
        eq = equity[-1]
        if eq <= 0.01:
            equity.append(0.01)
            continue

        qty, mode, risk = calculate_qty(eq, td['btc_price'], stats,
                                         survive_n=survive_n, survive_pct=survive_pct)
        modes[mode] += 1

        position = qty * td['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        if eq < margin:
            # Can't even cover margin - use absolute minimum
            qty = BINANCE_MIN_QTY
            position = qty * td['btc_price']
            margin = position / LEVERAGE
            maint = position * MAINT_MARGIN_RATE
            if eq < margin:
                equity.append(eq)  # truly can't trade
                continue

        pnl = position * (td['bps'] / 10000)
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)  # liquidated
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, modes


def simulate_fixed(trade_list, dollars_per_step, capital=STARTING_CAPITAL):
    """Fixed $/step simulation (baseline)."""
    equity = [capital]

    for td in trade_list:
        eq = equity[-1]
        if eq <= 0.01:
            equity.append(0.01)
            continue

        steps = max(1, int(eq / dollars_per_step))
        qty = steps * BINANCE_STEP_SIZE
        qty = max(qty, td['qty_min'])

        position = qty * td['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        if eq < margin:
            equity.append(eq)
            continue

        pnl = position * (td['bps'] / 10000)
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity


def run_mc_dynamic(trade_list, stats, survive_n=5, survive_pct=0.10,
                   n_sims=N_SIMS, seed=42, capital=STARTING_CAPITAL):
    np.random.seed(seed)
    finals, dds, ruin = [], [], 0
    total_modes = {'FORCED_MIN': 0, 'SURVIVAL': 0, 'GROWTH': 0}

    for _ in range(n_sims):
        shuffled = list(trade_list)
        np.random.shuffle(shuffled)
        eq, modes = simulate_dynamic(shuffled, stats,
                                     survive_n=survive_n, survive_pct=survive_pct,
                                     capital=capital)
        finals.append(eq[-1])
        dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin += 1
        for k, v in modes.items():
            total_modes[k] += v

    avg_modes = {k: v / n_sims for k, v in total_modes.items()}
    return {
        'median': np.median(finals),
        'geo_mean': np.exp(np.mean(np.log(np.maximum(finals, 0.01)))),
        'p5': np.percentile(finals, 5),
        'p95': np.percentile(finals, 95),
        'avg_dd': np.mean(dds),
        'ruin_pct': ruin / n_sims * 100,
        'avg_modes': avg_modes,
    }


def run_mc_fixed(trade_list, step, n_sims=N_SIMS, seed=42, capital=STARTING_CAPITAL):
    np.random.seed(seed)
    finals, dds, ruin = [], [], 0
    for _ in range(n_sims):
        shuffled = list(trade_list)
        np.random.shuffle(shuffled)
        eq = simulate_fixed(shuffled, step, capital=capital)
        finals.append(eq[-1])
        dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin += 1
    return {
        'median': np.median(finals),
        'geo_mean': np.exp(np.mean(np.log(np.maximum(finals, 0.01)))),
        'p5': np.percentile(finals, 5),
        'p95': np.percentile(finals, 95),
        'avg_dd': np.mean(dds),
        'ruin_pct': ruin / n_sims * 100,
    }


def print_mc_row(label, r, show_modes=False):
    base = (f"  {label:>20s} | ${r['median']:>12,.0f} | ${r['p5']:>12,.0f} | "
            f"${r['p95']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")
    if show_modes and 'avg_modes' in r:
        m = r['avg_modes']
        total = sum(m.values())
        if total > 0:
            fm_pct = m.get('FORCED_MIN', 0) / total * 100
            sv_pct = m.get('SURVIVAL', 0) / total * 100
            gr_pct = m.get('GROWTH', 0) / total * 100
            base += f" | {fm_pct:4.0f}%F {sv_pct:4.0f}%S {gr_pct:4.0f}%G"
    print(base)


# ============================================================
# PART 3: MC ON TRAIN — Dynamic (no-skip) vs Fixed
# ============================================================
print("=" * 100)
print("PART 3: MC ON TRAIN (431 trades) -- Dynamic (no-skip) vs Fixed")
print("=" * 100)
print()
print(f"  Starting capital: ${STARTING_CAPITAL:.0f}")
print(f"  Stats used: TRAIN (worst={train_stats['worst_loss']:.0f}bp, P95={train_stats['p95_loss']:.0f}bp)")
print()

survive_configs = [
    ('S3_10%', 3, 0.10),
    ('S5_10%', 5, 0.10),
    ('S5_20%', 5, 0.20),
    ('S7_10%', 7, 0.10),
]

print(f"  {'Config':>20s} | {'Median':>14s} | {'P5':>14s} | {'P95':>14s} | {'AvgDD':>6s} | {'Ruin%':>6s} | Modes")
print(f"  {'-'*110}")

train_dynamic_results = {}
for label, sn, sp in survive_configs:
    r = run_mc_dynamic(train_data, train_stats, survive_n=sn, survive_pct=sp)
    train_dynamic_results[label] = r
    print_mc_row('DYN ' + label, r, show_modes=True)

print(f"  {'-'*110}")

fixed_steps = [2.00, 3.00, 4.00, 5.00, 8.00, 10.00]
train_fixed_results = {}
for step in fixed_steps:
    r = run_mc_fixed(train_data, step)
    train_fixed_results[step] = r
    print_mc_row('FIXED $%.2f' % step, r)

print()


# ============================================================
# PART 4: MC ON OOS — Dynamic (no-skip) vs Fixed
# ============================================================
print("=" * 100)
print("PART 4: MC ON OOS (220 trades) -- Dynamic (no-skip) vs Fixed")
print("=" * 100)
print()
print(f"  Starting capital: ${STARTING_CAPITAL:.0f}")
print(f"  Calculator uses TRAIN stats (proper validation)")
print()

print(f"  {'Config':>20s} | {'Median':>14s} | {'P5':>14s} | {'P95':>14s} | {'AvgDD':>6s} | {'Ruin%':>6s} | Modes")
print(f"  {'-'*110}")

oos_dynamic_results = {}
for label, sn, sp in survive_configs:
    r = run_mc_dynamic(oos_data, train_stats, survive_n=sn, survive_pct=sp)
    oos_dynamic_results[label] = r
    print_mc_row('DYN ' + label, r, show_modes=True)

print(f"  {'-'*110}")

oos_fixed_results = {}
for step in fixed_steps:
    r = run_mc_fixed(oos_data, step)
    oos_fixed_results[step] = r
    print_mc_row('FIXED $%.2f' % step, r)

print()

# Overfitted comparison
print("  --- Overfitted comparison (calculator uses OOS stats): ---")
r_ovf = run_mc_dynamic(oos_data, oos_stats, survive_n=5, survive_pct=0.10)
print_mc_row('OVF S5_10%', r_ovf, show_modes=True)
print()


# ============================================================
# PART 5: STARTING CAPITAL SWEEP
# ============================================================
print("=" * 100)
print("PART 5: STARTING CAPITAL SWEEP -- How much do you need?")
print("=" * 100)
print()
print("  Dynamic S5_10% with TRAIN stats, tested on OOS trades")
print()

capitals = [5, 10, 15, 20, 30, 50, 75, 100, 200, 500]

print(f"  {'Start $':>8s} | {'DYN Median':>12s} | {'DYN P5':>12s} | {'DYN Ruin':>8s} | {'DYN Modes':>20s} | "
      f"{'FIX$4 Median':>12s} | {'FIX$4 Ruin':>8s} | {'FIX$8 Median':>12s} | {'FIX$8 Ruin':>8s}")
print(f"  {'-'*130}")

for cap in capitals:
    r_dyn = run_mc_dynamic(oos_data, train_stats, survive_n=5, survive_pct=0.10, capital=cap)
    r_f4 = run_mc_fixed(oos_data, 4.0, capital=cap)
    r_f8 = run_mc_fixed(oos_data, 8.0, capital=cap)

    m = r_dyn['avg_modes']
    total = sum(m.values())
    if total > 0:
        modes_str = f"{m.get('FORCED_MIN',0)/total*100:.0f}%F {m.get('SURVIVAL',0)/total*100:.0f}%S {m.get('GROWTH',0)/total*100:.0f}%G"
    else:
        modes_str = "N/A"

    print(f"  ${cap:>6,} | ${r_dyn['median']:>10,.0f} | ${r_dyn['p5']:>10,.0f} | {r_dyn['ruin_pct']:>6.1f}% | "
          f"{modes_str:>20s} | ${r_f4['median']:>10,.0f} | {r_f4['ruin_pct']:>6.1f}% | "
          f"${r_f8['median']:>10,.0f} | {r_f8['ruin_pct']:>6.1f}%")

print()


# ============================================================
# PART 6: SURVIVAL PHASE ANALYSIS
# ============================================================
print("=" * 100)
print("PART 6: SURVIVAL PHASE -- How quickly do we escape FORCED_MIN?")
print("=" * 100)
print()

for dataset_name, dataset, stats_to_use in [("TRAIN", train_data, train_stats),
                                              ("OOS", oos_data, train_stats)]:
    # Historical order
    eq, modes = simulate_dynamic(dataset, stats_to_use, survive_n=5, survive_pct=0.10)

    transition_forced = None
    transition_growth = None
    for i, td in enumerate(dataset):
        qty, mode, risk = calculate_qty(eq[i], td['btc_price'], stats_to_use,
                                         survive_n=5, survive_pct=0.10)
        if mode != 'FORCED_MIN' and transition_forced is None:
            transition_forced = i
        if mode == 'GROWTH' and transition_growth is None:
            transition_growth = i

    print(f"  {dataset_name} (historical order, {len(dataset)} trades):")
    print(f"    FORCED_MIN: {modes.get('FORCED_MIN',0)} | SURVIVAL: {modes.get('SURVIVAL',0)} | GROWTH: {modes.get('GROWTH',0)}")
    print(f"    Final equity: ${eq[-1]:,.2f}")
    if transition_forced is not None:
        print(f"    Exit FORCED_MIN at trade #{transition_forced} (wallet ${eq[transition_forced]:,.2f})")
    if transition_growth is not None:
        print(f"    Enter GROWTH at trade #{transition_growth} (wallet ${eq[transition_growth]:,.2f})")
    else:
        print(f"    NEVER reached GROWTH mode")

    milestones = [10, 25, 50, 100, 150, 200, len(dataset)]
    print(f"    Equity: ", end="")
    for ms in milestones:
        if ms <= len(eq) - 1:
            print(f"T{ms}=${eq[ms]:,.0f}  ", end="")
    print()
    print()

    # MC analysis
    np.random.seed(42)
    trans_forced_list = []
    trans_growth_list = []
    died_forced = 0
    never_growth = 0

    for _ in range(N_SIMS):
        shuffled = list(dataset)
        np.random.shuffle(shuffled)

        sim_eq = [STARTING_CAPITAL]
        left_forced = False
        reached_growth = False
        died = False

        for i, td in enumerate(shuffled):
            w = sim_eq[-1]
            if w <= 0.01:
                died = True
                break

            qty, mode, risk = calculate_qty(w, td['btc_price'], stats_to_use,
                                             survive_n=5, survive_pct=0.10)

            if mode != 'FORCED_MIN' and not left_forced:
                left_forced = True
                trans_forced_list.append(i)
            if mode == 'GROWTH' and not reached_growth:
                reached_growth = True
                trans_growth_list.append(i)

            position = qty * td['btc_price']
            maint = position * MAINT_MARGIN_RATE
            pnl = position * (td['bps'] / 10000)
            max_loss = w - maint
            if pnl < -max_loss:
                sim_eq.append(0.01)
            else:
                sim_eq.append(max(w + pnl, 0.01))

        if died and not left_forced:
            died_forced += 1
        if not reached_growth and not died:
            never_growth += 1

    print(f"  {dataset_name} MC (1000 paths, S5_10%):")
    if trans_forced_list:
        print(f"    Trades to exit FORCED_MIN: median {int(np.median(trans_forced_list))}, "
              f"P25-P75: {int(np.percentile(trans_forced_list, 25))}-{int(np.percentile(trans_forced_list, 75))}")
    if trans_growth_list:
        print(f"    Trades to GROWTH: median {int(np.median(trans_growth_list))}, "
              f"P25-P75: {int(np.percentile(trans_growth_list, 25))}-{int(np.percentile(trans_growth_list, 75))}")
    print(f"    Died in FORCED_MIN: {died_forced}/{N_SIMS} ({died_forced/10:.1f}%)")
    print(f"    Never reached GROWTH: {never_growth}/{N_SIMS} ({never_growth/10:.1f}%)")
    print()


# ============================================================
# PART 7: VERDICT
# ============================================================
print("=" * 100)
print("PART 7: VERDICT")
print("=" * 100)
print()

print("  SUMMARY TABLE (all from $10 start):")
print(f"  {'Config':>20s} | {'TRAIN Med':>12s} | {'TRAIN Ruin':>10s} | {'OOS Med':>12s} | {'OOS Ruin':>10s}")
print(f"  {'-'*75}")

for label in ['S3_10%', 'S5_10%', 'S5_20%', 'S7_10%']:
    t = train_dynamic_results.get(label, {})
    o = oos_dynamic_results.get(label, {})
    if t and o:
        print(f"  {'DYN '+label:>20s} | ${t.get('median',0):>10,.0f} | {t.get('ruin_pct',0):>8.1f}% | "
              f"${o.get('median',0):>10,.0f} | {o.get('ruin_pct',0):>8.1f}%")

for step in fixed_steps:
    t = train_fixed_results.get(step, {})
    o = oos_fixed_results.get(step, {})
    if t and o:
        print(f"  {'FIXED $%.2f' % step:>20s} | ${t.get('median',0):>10,.0f} | {t.get('ruin_pct',0):>8.1f}% | "
              f"${o.get('median',0):>10,.0f} | {o.get('ruin_pct',0):>8.1f}%")

print()
