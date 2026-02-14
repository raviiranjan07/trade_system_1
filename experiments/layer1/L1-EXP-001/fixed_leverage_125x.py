"""L1-EXP-001b: Fixed Leverage Baseline at 125x (Dynamic Notional)

QUESTION: Same as EXP-001 but with correct Binance setup:
  - Leverage setting: 125x (maximum)
  - Position: calculated dynamically from BTC price (NOT hardcoded $170)
  - Cross margin: full wallet backs trade
  - Margin = position / 125 (minimal)

WHAT WE TEST:
  - 1x minimum position (0.001 BTC * BTC_price)
  - 2x minimum position (0.002 BTC * BTC_price)
  - 3x minimum position (0.003 BTC * BTC_price)
  - 5x minimum position (0.005 BTC * BTC_price)
  - Each with MC 1000 shuffled paths

DYNAMIC NOTIONAL:
  - Binance min qty: 0.001 BTC
  - Binance step size: 0.001 BTC
  - Binance min notional: $100
  - Actual min qty = max(0.001, ceil(100 / btc_price / 0.001) * 0.001)
  - Position = qty * btc_price (varies with BTC price!)

CROSS MARGIN AT 125x:
  - Initial margin = position / 125 (tiny: ~$1 for $130 position)
  - Liquidation: when equity drops to maintenance margin
  - Maintenance margin = position * 0.004
  - Max loss before liq = equity - maintenance_margin
  - Full wallet always backs the trade
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

# ============================================================
# CONSTANTS — no hardcoded notional for eligibility
# ============================================================
STARTING_CAPITAL = 10.0
LEVERAGE_SETTING = 125
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100  # Binance rule: position must be >= $100
N_SIMS = 1000

# ============================================================
# LOAD V1.3.2 TRADES
# ============================================================
config = load_config()
trades = run_backtest(config)

# Build trade data with DYNAMIC position sizes per BTC price
trade_data = []
for t in trades:
    btc_price = t.entry_price
    # Dynamic minimum quantity calculation
    qty_min = max(BINANCE_MIN_QTY,
                  math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)
    pos_min = qty_min * btc_price
    maint_min = pos_min * MAINT_MARGIN_RATE
    margin_min = pos_min / LEVERAGE_SETTING

    trade_data.append({
        'bps': t.net_profit_bps,
        'btc_price': btc_price,
        'qty_min': qty_min,        # minimum tradeable qty
        'pos_min': pos_min,        # minimum position in $
        'maint_min': maint_min,    # maintenance margin for min position
        'margin_min': margin_min,  # initial margin for min position
        'direction': t.direction,
        'signal_type': t.signal_type if hasattr(t, 'signal_type') else 'UNKNOWN',
    })

returns = [td['bps'] for td in trade_data]
wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]

print("=" * 110)
print("L1-EXP-001b: FIXED LEVERAGE BASELINE AT 125x (DYNAMIC NOTIONAL)")
print("=" * 110)
print(f"  V1.3.2 OOS: {len(returns)} trades, {len(wins)/len(returns)*100:.1f}% win rate")
print(f"  Mean: {np.mean(returns):+.1f} bps | Median: {np.median(returns):+.1f} bps | Std: {np.std(returns):.1f} bps")
print(f"  Avg win: {np.mean(wins):+.1f} bps | Avg loss: {np.mean(losses):.1f} bps")
print(f"  Best: {max(returns):+.1f} bps | Worst: {min(returns):.1f} bps")
print(f"  Starting capital: ${STARTING_CAPITAL} | Leverage setting: {LEVERAGE_SETTING}x")
print()
print(f"  DYNAMIC POSITION SIZES (from actual BTC prices):")
print(f"    BTC price range: ${min(td['btc_price'] for td in trade_data):,.0f} - ${max(td['btc_price'] for td in trade_data):,.0f}")
print(f"    1x min position range: ${min(td['pos_min'] for td in trade_data):,.0f} - ${max(td['pos_min'] for td in trade_data):,.0f}")
print(f"    1x min margin range: ${min(td['margin_min'] for td in trade_data):.2f} - ${max(td['margin_min'] for td in trade_data):.2f}")
print(f"    1x min qty range: {min(td['qty_min'] for td in trade_data)} - {max(td['qty_min'] for td in trade_data)} BTC")
print()


# ============================================================
# CORE FUNCTIONS
# ============================================================
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


def simulate_125x(trade_list, pos_multiplier, capital=STARTING_CAPITAL):
    """Simulate at 125x leverage with dynamic position sizing.

    Cross margin: full wallet backs every trade.
    Position = pos_multiplier * minimum_qty * btc_price
    Margin = position / 125 (tiny)

    Can trade if:
      1. equity > margin required (position / 125)
      2. position >= $100 (Binance min notional)
    """
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        # Calculate position for this trade
        qty = td['qty_min'] * pos_multiplier
        position = qty * td['btc_price']
        margin_required = position / LEVERAGE_SETTING
        maint = position * MAINT_MARGIN_RATE

        # Can we trade?
        if eq < margin_required:
            equity.append(eq)
            skipped += 1
            continue

        # PnL
        pnl = position * (td['bps'] / 10000)

        # Cross margin liquidation: full equity backs the trade
        # Liquidated when equity drops to maintenance margin level
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


# ============================================================
# PART 1: ORIGINAL ORDER — Different position multipliers
# ============================================================
print("=" * 110)
print("PART 1: ORIGINAL TRADE ORDER (V1.3.2 OOS 2024-2025)")
print("=" * 110)
print()

MULTIPLIERS = [1, 2, 3, 5]

print(f"  {'Mult':>6s} | {'Avg Pos':>10s} | {'Margin':>8s} | {'Eff Lev':>8s} | {'Final':>14s} | {'Return':>12s} | {'MaxDD':>8s} | {'MinEq':>8s} | {'Skip':>6s} | {'Liq':>4s}")
print(f"  {'-'*110}")

original_results = {}
for mult in MULTIPLIERS:
    eq, skipped, liquidated = simulate_125x(trade_data, mult)
    dd = calc_max_dd(eq)
    final = eq[-1]
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    min_eq = min(eq)

    avg_pos = np.mean([td['pos_min'] * mult for td in trade_data])
    avg_margin = avg_pos / LEVERAGE_SETTING
    eff_lev = avg_pos / STARTING_CAPITAL

    original_results[mult] = {
        'final': final, 'return': ret, 'dd': dd,
        'min_eq': min_eq, 'skipped': skipped, 'liquidated': liquidated,
        'avg_pos': avg_pos, 'eff_lev': eff_lev,
    }

    print(f"  {mult:>5d}x | ${avg_pos:>8,.0f} | ${avg_margin:>6.2f} | {eff_lev:>6.1f}x | ${final:>12,.2f} | {ret:>+10.1f}% | {dd*100:6.1f}% | ${min_eq:6.2f} | {skipped:>6d} | {liquidated:>4d}")

print()
print(f"  NOTE: At 125x, margin for 1x min position = ~${np.mean([td['margin_min'] for td in trade_data]):.2f}")
print(f"        Even $10 wallet can easily cover this margin.")
print(f"        Liquidation depends on FULL WALLET vs maintenance margin, not leverage setting.")


# ============================================================
# PART 2: WORST SINGLE TRADE IMPACT
# ============================================================
print()
print("=" * 110)
print("PART 2: WORST SINGLE TRADE IMPACT (at $10 equity)")
print("=" * 110)
print()

worst_trade = min(returns)
best_trade = max(returns)
avg_loss = np.mean(losses)
avg_pos_1x = np.mean([td['pos_min'] for td in trade_data])

print(f"  Worst trade: {worst_trade:.1f} bps ({worst_trade/100:.2f}%)")
print(f"  Best trade:  {best_trade:+.1f} bps ({best_trade/100:.2f}%)")
print(f"  Avg loss:    {avg_loss:.1f} bps")
print()

print(f"  {'Mult':>6s} | {'Avg Pos':>10s} | {'Worst $ Loss':>12s} | {'Worst % Loss':>12s} | {'After Worst':>12s} | {'Can Trade?':>10s}")
print(f"  {'-'*80}")

for mult in MULTIPLIERS:
    avg_pos = avg_pos_1x * mult
    loss_dollar = avg_pos * abs(worst_trade) / 10000
    loss_pct = loss_dollar / STARTING_CAPITAL * 100
    remaining = STARTING_CAPITAL - loss_dollar
    margin_needed = avg_pos * mult / LEVERAGE_SETTING  # for next trade
    min_margin = np.mean([td['margin_min'] for td in trade_data]) * mult
    can_trade = "YES" if remaining > min_margin else "NO"
    print(f"  {mult:>5d}x | ${avg_pos:>8,.0f} | ${loss_dollar:>10.2f} | {loss_pct:>10.1f}% | ${remaining:>10.2f} | {can_trade:>10s}")


# ============================================================
# PART 3: CONSECUTIVE LOSSES TO RUIN
# ============================================================
print()
print("=" * 110)
print("PART 3: CONSECUTIVE LOSSES TO RUIN (can't meet margin requirement)")
print("=" * 110)
print()

for loss_type, loss_val in [("avg", avg_loss), ("worst", worst_trade)]:
    print(f"  --- Using {loss_type} loss ({loss_val:.1f} bps) ---")
    for mult in MULTIPLIERS:
        eq = STARTING_CAPITAL
        for i in range(100):
            avg_pos = avg_pos_1x * mult
            margin_req = avg_pos / LEVERAGE_SETTING
            if eq < margin_req:
                print(f"    {mult}x pos: RUIN after {i} consecutive {loss_type} losses (equity: ${eq:.4f}, margin needed: ${margin_req:.2f})")
                break
            loss = avg_pos * abs(loss_val) / 10000
            # Cross liq check
            maint = avg_pos * MAINT_MARGIN_RATE
            if loss > eq - maint:
                print(f"    {mult}x pos: LIQUIDATED after {i+1} consecutive {loss_type} losses (equity: ${eq:.4f})")
                break
            eq -= loss
        else:
            print(f"    {mult}x pos: Survived 100 consecutive {loss_type} losses (equity: ${eq:.4f})")
    print()


# ============================================================
# PART 4: MONTE CARLO — 1000 shuffled paths per multiplier
# ============================================================
print("=" * 110)
print("PART 4: MONTE CARLO — 1000 shuffled trade orderings")
print("=" * 110)
print()

np.random.seed(42)

print(f"  {'Mult':>6s} | {'Median':>14s} | {'P5':>12s} | {'P25':>12s} | {'P75':>14s} | {'P95':>14s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s} | {'Liq%':>6s}")
print(f"  {'-'*130}")

mc_results = {}
for mult in MULTIPLIERS:
    finals = []
    max_dds = []
    ruined = 0
    liq_total = 0

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq, skipped, liquidated = simulate_125x(shuffled, mult)
        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)
        if eq[-1] < 0.1:  # effectively ruined
            ruined += 1
        liq_total += liquidated

    mc_results[mult] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'p95': np.percentile(finals, 95),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruined / N_SIMS * 100,
        'liq_pct': liq_total / N_SIMS / len(trade_data) * 100,
    }

    r = mc_results[mult]
    print(f"  {mult:>5d}x | ${r['median']:>12,.2f} | ${r['p5']:>10,.2f} | ${r['p25']:>10,.2f} | ${r['p75']:>12,.2f} | ${r['p95']:>12,.2f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}% | {r['liq_pct']:4.2f}%")


# ============================================================
# PART 5: SURVIVAL — What % of paths never dip below threshold?
# ============================================================
print()
print("=" * 110)
print("PART 5: SURVIVAL — What % of 1000 paths NEVER dip below threshold?")
print("=" * 110)
print()

np.random.seed(42)

thresholds = [1, 3, 5, 7, 8, 9]
print(f"  {'Mult':>6s}", end="")
for t in thresholds:
    print(f" | {'>${}'.format(t):>7s}", end="")
print()
print(f"  {'-'*70}")

for mult in MULTIPLIERS:
    above = {t: 0 for t in thresholds}

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq, _, _ = simulate_125x(shuffled, mult)
        min_eq = min(eq)
        for t in thresholds:
            if min_eq >= t:
                above[t] += 1

    print(f"  {mult:>5d}x", end="")
    for t in thresholds:
        pct = above[t] / N_SIMS * 100
        print(f" | {pct:6.1f}%", end="")
    print()


# ============================================================
# PART 6: EQUITY CURVE SHAPE (original order)
# ============================================================
print()
print("=" * 110)
print("PART 6: EQUITY CURVE STATISTICS (original order)")
print("=" * 110)
print()

print(f"  {'Mult':>6s} | {'Eq@50':>10s} | {'Eq@100':>10s} | {'Eq@150':>10s} | {'Eq@200':>10s} | {'Final':>14s} | {'MaxDD':>7s}")
print(f"  {'-'*90}")

for mult in MULTIPLIERS:
    eq, _, _ = simulate_125x(trade_data, mult)
    dd = calc_max_dd(eq)

    eq_50 = eq[min(50, len(eq)-1)]
    eq_100 = eq[min(100, len(eq)-1)]
    eq_150 = eq[min(150, len(eq)-1)]
    eq_200 = eq[min(200, len(eq)-1)]

    print(f"  {mult:>5d}x | ${eq_50:>8,.2f} | ${eq_100:>8,.2f} | ${eq_150:>8,.2f} | ${eq_200:>8,.2f} | ${eq[-1]:>12,.2f} | {dd*100:5.1f}%")


# ============================================================
# PART 7: COMPARISON WITH ORIGINAL EXP-001 (20x hardcoded)
# ============================================================
print()
print("=" * 110)
print("PART 7: COMPARISON — 125x DYNAMIC vs OLD 20x HARDCODED")
print("=" * 110)
print()

# Old EXP-001: 20x fixed, MIN_NOTIONAL=170 hardcoded
# Only 20x actually works (17x minimum). Result: $56K, 18.6% ruin
print("  OLD SETUP (EXP-001 original):")
print("    Leverage: 20x (hardcoded)")
print("    Min notional: $170 (hardcoded)")
print("    Position: equity * 20 (scales with wallet)")
print("    Eligibility: equity * 20 >= $170 -> need $8.50+ equity")
print("    Result: $56K final, 18.6% MC ruin")
print()
print("  NEW SETUP (125x dynamic):")
print(f"    Leverage setting: {LEVERAGE_SETTING}x")
print(f"    Min position: dynamic from BTC price (${min(td['pos_min'] for td in trade_data):,.0f}-${max(td['pos_min'] for td in trade_data):,.0f})")
print(f"    Margin per trade: ${np.mean([td['margin_min'] for td in trade_data]):.2f} average (position / 125)")
print(f"    Eligibility: equity > margin (~${np.mean([td['margin_min'] for td in trade_data]):.2f})")
print(f"    Full wallet backs trade (cross margin)")
print()

# Key difference: old scaled position with equity (eq * 20x), new uses FIXED position
print("  KEY DIFFERENCE:")
print("    OLD: position = equity * leverage (grows with wallet -> compounds)")
print("    NEW: position = fixed qty * BTC_price (constant size -> linear growth)")
print()
print("    At 1x min (~$130 position):")
print(f"      Effective leverage at $10: {np.mean([td['pos_min'] for td in trade_data])/10:.1f}x")
print(f"      Effective leverage at $50: {np.mean([td['pos_min'] for td in trade_data])/50:.1f}x")
print(f"      Effective leverage at $100: {np.mean([td['pos_min'] for td in trade_data])/100:.1f}x")
print(f"      -> Position stays same, eff leverage DROPS as wallet grows")
print()
print("    At 2x min (~$260 position):")
print(f"      Effective leverage at $10: {np.mean([td['pos_min'] for td in trade_data])*2/10:.1f}x")
print(f"      Effective leverage at $50: {np.mean([td['pos_min'] for td in trade_data])*2/50:.1f}x")
print(f"      Effective leverage at $100: {np.mean([td['pos_min'] for td in trade_data])*2/100:.1f}x")


# ============================================================
# PART 8: SCALING POSITION WITH WALLET (like old 20x but at 125x)
# ============================================================
print()
print("=" * 110)
print("PART 8: WHAT IF POSITION SCALES WITH WALLET? (margin% * 125x)")
print("=" * 110)
print()
print("  This tests: position = margin_pct * wallet * 125")
print("  Same as old EXP-001 but at 125x and with margin_pct control")
print()

MARGIN_PCTS = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.30, 0.40]

print(f"  {'Margin%':>8s} | {'Eff Lev':>8s} | {'Orig Final':>14s} | {'MC Median':>14s} | {'MC P5':>12s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*100}")

np.random.seed(42)

for pct in MARGIN_PCTS:
    eff_lev = pct * LEVERAGE_SETTING

    # Original order
    equity = [STARTING_CAPITAL]
    for td in trade_data:
        eq = equity[-1]
        margin = eq * pct
        position = margin * LEVERAGE_SETTING

        # Enforce Binance minimum
        if position < td['pos_min']:
            position = td['pos_min']

        margin_req = position / LEVERAGE_SETTING
        if eq < margin_req:
            equity.append(eq)
            continue

        pnl = position * (td['bps'] / 10000)
        maint = position * MAINT_MARGIN_RATE
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
        else:
            equity.append(max(eq + pnl, 0.01))

    orig_final = equity[-1]

    # MC
    np.random.seed(42)
    finals = []
    max_dds = []
    ruined = 0
    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)

        equity = [STARTING_CAPITAL]
        for td in shuffled:
            eq = equity[-1]
            margin = eq * pct
            position = margin * LEVERAGE_SETTING

            if position < td['pos_min']:
                position = td['pos_min']

            margin_req = position / LEVERAGE_SETTING
            if eq < margin_req:
                equity.append(eq)
                continue

            pnl = position * (td['bps'] / 10000)
            maint = position * MAINT_MARGIN_RATE
            max_loss = eq - maint
            if pnl < -max_loss:
                equity.append(0.01)
            else:
                equity.append(max(eq + pnl, 0.01))

        finals.append(equity[-1])
        max_dds.append(calc_max_dd(equity))
        if equity[-1] < 0.1:
            ruined += 1

    median = np.median(finals)
    p5 = np.percentile(finals, 5)
    avg_dd = np.mean(max_dds)
    ruin_pct = ruined / N_SIMS * 100

    print(f"  {pct*100:6.0f}% | {eff_lev:6.1f}x | ${orig_final:>12,.2f} | ${median:>12,.2f} | ${p5:>10,.2f} | {avg_dd*100:5.1f}% | {ruin_pct:5.1f}%")


# ============================================================
# PART 9: VERDICT
# ============================================================
print()
print("=" * 110)
print("PART 9: VERDICT")
print("=" * 110)
print()
print("  SETUP: Binance 125x leverage, Cross margin, $10 start")
print()
print("  TWO APPROACHES TESTED:")
print("    A) Fixed position (1x-5x minimum): linear growth, low risk, no compounding")
print("    B) Scaling position (margin% * 125x): exponential growth, compounds")
print()
print("  Compare with OLD EXP-001 (20x, hardcoded $170):")
print(f"    OLD: $56,043 final, 18.6% ruin, position = equity * 20")
print()
print("  THIS IS THE BASELINE FOR 125x SETUP.")
print("  All future experiments at 125x must beat these numbers.")
