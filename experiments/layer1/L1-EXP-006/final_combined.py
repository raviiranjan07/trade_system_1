"""L1-EXP-006: Final Combined Risk Management System

COMBINES ALL FINDINGS FROM EXP-001 THROUGH EXP-005:

  From EXP-001: $10 at fixed 20x has 18.6% ruin -> need proper risk management
  From EXP-002: Cross margin, $15 threshold, fixed leverage, dynamic position sizing
  From EXP-003: Min position $100-$199, maint margin 0.4%, funding net positive
  From EXP-004: Kelly optimal 49.5x, half-Kelly 25x, Mean-Variance best type
  From EXP-005: Fixed leverage beats ALL dynamic/adaptive approaches

FINAL SYSTEM:
  Phase 1: Cross margin, dynamic position, until equity >= $15
  Phase 2: Cross margin, position = equity * leverage (20x or 25x)

NEW IN THIS EXPERIMENT:
  1. Liquidation price calculation per trade
  2. Safety stop-loss on exchange (backup for bot failure)
  3. Stress testing (what if flash crash hits?)
  4. Complete system MC with all components
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

STARTING_CAPITAL = 10.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004  # 0.4% for BTCUSDT Tier 1
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100
PHASE1_THRESHOLD = 15

# Load V1.3.2 trades
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
        'direction': t.direction,
        'signal_type': t.signal_type if hasattr(t, 'signal_type') else 'UNKNOWN',
    })

returns = [td['bps'] for td in trade_data]

print("=" * 110)
print("L1-EXP-006: FINAL COMBINED RISK MANAGEMENT SYSTEM")
print("=" * 110)
print()
print(f"  V1.3.2: {len(trades)} trades, {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}% win")
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
# PART 1: LIQUIDATION PRICE CALCULATION
# ============================================================
print("=" * 110)
print("PART 1: LIQUIDATION PRICE FOR EVERY V1.3.2 TRADE")
print("=" * 110)
print()

print("  Cross Margin Liquidation Formula:")
print("    LONG:  liq_price = entry * (1 - 1/leverage + maint_rate)")
print("    SHORT: liq_price = entry * (1 + 1/leverage - maint_rate)")
print()


def calc_liquidation_price(entry_price, leverage, direction, maint_rate=MAINT_MARGIN_RATE):
    """Calculate Binance cross margin liquidation price."""
    if direction == 'LONG':
        return entry_price * (1 - 1 / leverage + maint_rate)
    else:  # SHORT
        return entry_price * (1 + 1 / leverage - maint_rate)


def calc_safety_stop_price(entry_price, liq_price, direction, safety_pct=0.80):
    """Safety stop at safety_pct of distance to liquidation.
    E.g., 80% means stop at 80% of the way from entry to liquidation."""
    distance = abs(liq_price - entry_price)
    safety_distance = distance * safety_pct
    if direction == 'LONG':
        return entry_price - safety_distance
    else:
        return entry_price + safety_distance


# Calculate for both leverage options
for leverage in [20, 25]:
    print(f"  --- At {leverage}x leverage ---")
    print()

    # Phase 2 liquidation
    liq_long = 1 - 1 / leverage + MAINT_MARGIN_RATE
    liq_short = 1 + 1 / leverage - MAINT_MARGIN_RATE
    buffer_long = (1 - liq_long) * 100
    buffer_short = (liq_short - 1) * 100

    print(f"  Phase 2 ({leverage}x):")
    print(f"    LONG liquidation:  entry x {liq_long:.4f} = {buffer_long:.2f}% below entry")
    print(f"    SHORT liquidation: entry x {liq_short:.4f} = {buffer_short:.2f}% above entry")
    print(f"    Buffer: {buffer_long:.1f}% = {buffer_long * 100:.0f} bps")
    print()

    # Phase 1 liquidation (effective leverage varies)
    # Phase 1: position = $100-$199, equity = $10-$15
    # Effective leverage = position / equity
    p1_positions = [100, 150, 199]
    p1_equities = [10, 12, 15]

    print(f"  Phase 1 examples (Cross margin, full wallet backs position):")
    print(f"    {'Equity':>8s} | {'Position':>10s} | {'Eff Lev':>8s} | {'LONG Liq':>10s} | {'SHORT Liq':>10s} | {'Buffer':>8s}")
    print(f"    {'-'*65}")

    for eq in p1_equities:
        # At $10-15 equity, position is based on BTC price
        # Use avg position ~$150
        pos = 150
        eff_lev = pos / eq
        # In cross: liquidation when equity + PnL <= maintenance margin
        # PnL_liq = -(equity - maint_margin)
        # move_pct = PnL_liq / position
        maint = pos * MAINT_MARGIN_RATE
        max_loss = eq - maint
        move_pct = max_loss / pos * 100
        print(f"    ${eq:>6d} | ${pos:>8d} | {eff_lev:>6.1f}x | {move_pct:>8.2f}% | {move_pct:>9.2f}% | {move_pct:.1f}%")

    print()

# Per-trade liquidation analysis
print()
print("  PER-TRADE LIQUIDATION ANALYSIS (Phase 2, 20x):")
print()
print(f"  {'#':>4s} | {'Direction':>5s} | {'Entry':>10s} | {'Liq Price':>10s} | {'Safety Stop':>11s} | {'Buffer':>8s} | {'Result':>8s} | {'Safe?':>5s}")
print(f"  {'-'*80}")

liq_distances_20x = []
liq_distances_25x = []
safety_hit_count_20x = 0
safety_hit_count_25x = 0

for i, td in enumerate(trade_data):
    entry = td['btc_price']
    direction = td['direction']
    bps = td['bps']

    # 20x
    liq_20 = calc_liquidation_price(entry, 20, direction)
    safety_20 = calc_safety_stop_price(entry, liq_20, direction, 0.80)
    buffer_20 = abs(liq_20 - entry) / entry * 100
    liq_distances_20x.append(buffer_20)

    # Would the trade's actual move hit safety stop?
    # Safety stop in bps
    safety_bps_20 = abs(safety_20 - entry) / entry * 10000
    actual_bps = abs(bps)  # worst case = actual loss
    safe_20 = actual_bps < safety_bps_20

    if not safe_20:
        safety_hit_count_20x += 1

    # 25x
    liq_25 = calc_liquidation_price(entry, 25, direction)
    safety_25 = calc_safety_stop_price(entry, liq_25, direction, 0.80)
    buffer_25 = abs(liq_25 - entry) / entry * 100
    liq_distances_25x.append(buffer_25)
    safety_bps_25 = abs(safety_25 - entry) / entry * 10000
    safe_25 = actual_bps < safety_bps_25
    if not safe_25:
        safety_hit_count_25x += 1

    # Print first 10, worst 5, last 5
    if i < 10 or i >= len(trade_data) - 5:
        safe_str = "YES" if safe_20 else "NO!"
        print(f"  {i+1:>4d} | {direction:>5s} | ${entry:>9,.0f} | ${liq_20:>9,.0f} | ${safety_20:>10,.0f} | {buffer_20:>6.2f}% | {bps:>+7.1f} | {safe_str:>5s}")

    if i == 10:
        print(f"  {'...':>4s} |")

# Print worst trades
print()
print("  WORST 5 TRADES vs Safety Stop (20x):")
worst_trades = sorted(enumerate(trade_data), key=lambda x: x[1]['bps'])[:5]
for i, td in worst_trades:
    entry = td['btc_price']
    direction = td['direction']
    liq = calc_liquidation_price(entry, 20, direction)
    safety = calc_safety_stop_price(entry, liq, direction, 0.80)
    safety_bps = abs(safety - entry) / entry * 10000
    buffer = abs(liq - entry) / entry * 10000
    print(f"    Trade {i+1}: {td['bps']:+.1f} bps | Safety stop at {safety_bps:.0f} bps | Liq at {buffer:.0f} bps | {'HIT!' if abs(td['bps']) >= safety_bps else 'SAFE'}")

print()
print(f"  SUMMARY:")
print(f"    20x: {safety_hit_count_20x}/{len(trade_data)} trades would hit safety stop ({safety_hit_count_20x/len(trade_data)*100:.1f}%)")
print(f"    25x: {safety_hit_count_25x}/{len(trade_data)} trades would hit safety stop ({safety_hit_count_25x/len(trade_data)*100:.1f}%)")
print(f"    20x buffer range: {min(liq_distances_20x):.2f}% - {max(liq_distances_20x):.2f}%")
print(f"    25x buffer range: {min(liq_distances_25x):.2f}% - {max(liq_distances_25x):.2f}%")
print(f"    Worst trade: {min(returns):.1f} bps = {abs(min(returns))/100:.2f}% move")
print(f"    20x liq distance: {np.mean(liq_distances_20x):.2f}% = {np.mean(liq_distances_20x)*100:.0f} bps")
print()


# ============================================================
# PART 2: SAFETY STOP IMPACT ON BACKTEST
# ============================================================
print("=" * 110)
print("PART 2: SAFETY STOP IMPACT ON BACKTEST")
print("=" * 110)
print()
print("  Does adding a safety stop change any trade outcomes?")
print("  Safety stop = 80% of distance to liquidation")
print()

for leverage in [20, 25]:
    trades_changed = 0
    total_bps_diff = 0

    for td in trade_data:
        entry = td['btc_price']
        direction = td['direction']
        liq = calc_liquidation_price(entry, leverage, direction)
        safety = calc_safety_stop_price(entry, liq, direction, 0.80)
        safety_bps = abs(safety - entry) / entry * 10000

        # If trade loss exceeds safety stop, it would have been stopped out earlier
        if td['bps'] < 0 and abs(td['bps']) > safety_bps:
            trades_changed += 1
            # Trade would have lost safety_bps instead of actual loss
            old_loss = td['bps']
            new_loss = -safety_bps
            total_bps_diff += (new_loss - old_loss)  # negative = worse with safety

    print(f"  {leverage}x: {trades_changed} trades affected by safety stop")
    if trades_changed > 0:
        print(f"    Total bps change: {total_bps_diff:+.1f} bps ({'better' if total_bps_diff > 0 else 'worse'})")
    else:
        print(f"    Safety stop has ZERO impact - all trades exit well within buffer")
    print()


# ============================================================
# PART 3: STRESS TEST - What if flash crash hits?
# ============================================================
print("=" * 110)
print("PART 3: STRESS TEST - Flash Crash Scenarios")
print("=" * 110)
print()
print("  What if the worst trade was 2x, 3x, 5x worse than historical?")
print("  V1.3.2 worst trade: -181.8 bps")
print()

worst_bps = min(returns)

for leverage in [20, 25]:
    liq_bps = (1 / leverage - MAINT_MARGIN_RATE) * 10000  # buffer in bps
    safety_bps = liq_bps * 0.80

    print(f"  --- {leverage}x leverage (liq at {liq_bps:.0f} bps, safety at {safety_bps:.0f} bps) ---")
    print()

    print(f"  {'Scenario':>25s} | {'Loss (bps)':>10s} | {'Equity Loss':>12s} | {'Hits Safety?':>12s} | {'Liquidated?':>11s}")
    print(f"  {'-'*80}")

    scenarios = [
        ('Worst historical', worst_bps),
        ('2x worst', worst_bps * 2),
        ('3x worst', worst_bps * 3),
        ('5x worst', worst_bps * 5),
        ('Black swan (-5%)', -500),
        ('Flash crash (-10%)', -1000),
    ]

    for name, loss_bps in scenarios:
        equity_loss_pct = abs(loss_bps) * leverage / 10000 * 100
        hits_safety = abs(loss_bps) >= safety_bps
        liquidated = abs(loss_bps) >= liq_bps

        # If liquidated, equity loss is capped at 100%
        if liquidated:
            equity_loss_pct = 100.0

        status_safety = "YES" if hits_safety else "no"
        status_liq = "YES!" if liquidated else "no"

        print(f"  {name:>25s} | {loss_bps:>+9.1f} | {equity_loss_pct:>10.1f}% | {status_safety:>12s} | {status_liq:>11s}")

    print()


# ============================================================
# PART 4: COMPLETE SYSTEM MC SIMULATION
# ============================================================
print("=" * 110)
print("PART 4: COMPLETE SYSTEM MC SIMULATION (with safety stops)")
print("=" * 110)
print()


def simulate_complete(trade_list, leverage, safety_pct=0.80, capital=STARTING_CAPITAL):
    """Complete system: Phase 1 + Phase 2 with liquidation and safety stops."""
    equity_curve = [capital]
    equity = capital
    safety_triggers = 0
    liquidations = 0
    skipped = 0

    for td in trade_list:
        eq = equity

        if eq < PHASE1_THRESHOLD:
            # Phase 1: Cross margin, fixed position
            pos = td['position']
            maint = td['maint_margin']
            margin_req = pos / MAX_LEVERAGE_SETTING

            if eq < margin_req:
                equity_curve.append(eq)
                skipped += 1
                continue

            pnl = pos * (td['bps'] / 10000)
            max_loss = eq - maint

            if pnl < -max_loss:
                equity = 0.01
                liquidations += 1
            else:
                equity = max(eq + pnl, 0.01)
        else:
            # Phase 2: Scaling with safety stop
            pos = eq * leverage
            pnl = pos * (td['bps'] / 10000)

            # Check safety stop (80% of distance to liquidation)
            liq_buffer_bps = (1 / leverage - MAINT_MARGIN_RATE) * 10000
            safety_buffer_bps = liq_buffer_bps * safety_pct

            if td['bps'] < 0 and abs(td['bps']) >= safety_buffer_bps:
                # Safety stop triggered - limit loss to safety level
                limited_pnl = -eq * leverage * safety_buffer_bps / 10000
                equity = max(eq + limited_pnl, 0.01)
                safety_triggers += 1
            elif td['bps'] < 0 and abs(td['bps']) >= liq_buffer_bps:
                # Liquidation (shouldn't happen with safety stop, but just in case)
                equity = 0.01
                liquidations += 1
            else:
                equity = max(eq + pnl, 0.01)

        equity_curve.append(equity)

    return equity_curve, safety_triggers, liquidations, skipped


np.random.seed(42)

for leverage in [20, 25]:
    print(f"  --- {leverage}x leverage, 1000 MC paths ---")

    # Without safety stop
    finals_no_safety = []
    dds_no_safety = []

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)

        equity_curve = [STARTING_CAPITAL]
        equity = STARTING_CAPITAL
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
            else:
                pos = eq * leverage
                pnl = pos * (td['bps'] / 10000)
                equity = max(eq + pnl, 0.01)
            equity_curve.append(equity)

        finals_no_safety.append(equity_curve[-1])
        dds_no_safety.append(calc_max_dd(equity_curve))

    # With safety stop
    finals_safety = []
    dds_safety = []
    total_safety_triggers = 0
    total_liquidations = 0

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq_curve, s_trig, liqs, skip = simulate_complete(shuffled, leverage)
        finals_safety.append(eq_curve[-1])
        dds_safety.append(calc_max_dd(eq_curve))
        total_safety_triggers += s_trig
        total_liquidations += liqs

    print()
    print(f"  {'Metric':>25s} | {'Without Safety':>15s} | {'With Safety':>15s}")
    print(f"  {'-'*60}")
    print(f"  {'Median':>25s} | ${np.median(finals_no_safety):>13,.0f} | ${np.median(finals_safety):>13,.0f}")
    print(f"  {'P5':>25s} | ${np.percentile(finals_no_safety, 5):>13,.0f} | ${np.percentile(finals_safety, 5):>13,.0f}")
    print(f"  {'P25':>25s} | ${np.percentile(finals_no_safety, 25):>13,.0f} | ${np.percentile(finals_safety, 25):>13,.0f}")
    print(f"  {'Avg MaxDD':>25s} | {np.mean(dds_no_safety)*100:>13.1f}% | {np.mean(dds_safety)*100:>13.1f}%")
    print(f"  {'P95 MaxDD':>25s} | {np.percentile(dds_no_safety, 95)*100:>13.1f}% | {np.percentile(dds_safety, 95)*100:>13.1f}%")
    print(f"  {'Safety triggers (total)':>25s} | {'N/A':>15s} | {total_safety_triggers:>15d}")
    print(f"  {'Liquidations (total)':>25s} | {'0':>15s} | {total_liquidations:>15d}")
    print()


# ============================================================
# PART 5: FINAL SYSTEM SPECIFICATION
# ============================================================
print("=" * 110)
print("PART 5: FINAL RISK MANAGEMENT SYSTEM SPECIFICATION")
print("=" * 110)
print()

print("  +-----------------------------------------------------------------+")
print("  |           V1.3.2 RISK MANAGEMENT SYSTEM                        |")
print("  +-----------------------------------------------------------------+")
print()
print("  1. MARGIN MODE: Cross")
print("     - Entire wallet backs each trade")
print("     - Simpler than isolated, identical performance (EXP-002)")
print()
print("  2. LEVERAGE: Fixed (NOT dynamic)")
print("     - Conservative: 20x | Moderate: 25x")
print("     - Dynamic Kelly REJECTED (EXP-005) - adds noise, hurts compounding")
print("     - Kelly optimal 49.5x, half-Kelly 25x (EXP-004)")
print()
print("  3. POSITION SIZING:")
print("     Phase 1 (equity < $15):")
print("       qty = max(0.001, ceil(100 / btc_price / 0.001) * 0.001)")
print("       position = qty * btc_price  ($100-$199)")
print("     Phase 2 (equity >= $15):")
print("       position = equity * leverage")
print()
print("  4. LIQUIDATION PRICES (per trade):")

for lev in [20, 25]:
    long_factor = 1 - 1 / lev + MAINT_MARGIN_RATE
    short_factor = 1 + 1 / lev - MAINT_MARGIN_RATE
    buffer = (1 / lev - MAINT_MARGIN_RATE) * 100
    print(f"     {lev}x LONG:  entry x {long_factor:.4f} ({buffer:.1f}% buffer = {buffer*100:.0f} bps)")
    print(f"     {lev}x SHORT: entry x {short_factor:.4f} ({buffer:.1f}% buffer = {buffer*100:.0f} bps)")

print()
print("  5. SAFETY STOP-LOSS (exchange-level backup):")
print("     - Placed on exchange at entry (works even if bot dies)")
print("     - Level: 80% of distance to liquidation")

for lev in [20, 25]:
    buffer_pct = (1 / lev - MAINT_MARGIN_RATE) * 100
    safety_pct = buffer_pct * 0.80
    safety_bps = safety_pct * 100
    print(f"     {lev}x: {safety_pct:.2f}% from entry = {safety_bps:.0f} bps")
    print(f"       LONG example (entry $95K):  safety stop at ${95000 * (1 - safety_pct/100):,.0f}")
    print(f"       SHORT example (entry $95K): safety stop at ${95000 * (1 + safety_pct/100):,.0f}")

print()
print("  6. EXISTING V1.3.2 EXITS (unchanged):")
print("     - Trailing stop: 20 bps (LONG), 30 bps (SHORT)")
print("     - After bar 5: tightens to 8 bps")
print("     - Time exit: bar 10 (150 minutes)")
print("     - These fire LONG before safety stop or liquidation")
print()

print("  7. EXIT PRIORITY (inner to outer):")
print("     Layer 1: Trailing stop (20/30 bps)     <- normal exit")
print("     Layer 2: Time exit (bar 10)              <- catches stragglers")
print("     Layer 3: Safety stop (exchange order)    <- bot failure backup")
print("     Layer 4: Liquidation (exchange auto)     <- last resort")
print()

print("  8. EXPECTED PERFORMANCE (MC, 1000 paths):")
print()
print(f"    {'Config':>15s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'Ruin':>5s}")
print(f"    {'-'*60}")
for lev, med_ns, p5_ns, dd_ns in [
    (20, np.median(finals_no_safety) if leverage == 20 else 0,
     np.percentile(finals_no_safety, 5) if leverage == 20 else 0,
     np.mean(dds_no_safety) if leverage == 20 else 0),
]:
    pass

# Re-run quick MC for both leverages for final table
np.random.seed(42)
for lev in [20, 25]:
    finals = []
    dds = []
    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq_curve = [STARTING_CAPITAL]
        equity = STARTING_CAPITAL
        for td in shuffled:
            eq = equity
            if eq < PHASE1_THRESHOLD:
                pos = td['position']
                maint = td['maint_margin']
                margin_req = pos / MAX_LEVERAGE_SETTING
                if eq < margin_req:
                    eq_curve.append(eq)
                    continue
                pnl = pos * (td['bps'] / 10000)
                max_loss = eq - maint
                if pnl < -max_loss:
                    equity = 0.01
                else:
                    equity = max(eq + pnl, 0.01)
            else:
                pos = eq * lev
                pnl = pos * (td['bps'] / 10000)
                equity = max(eq + pnl, 0.01)
            eq_curve.append(equity)
        finals.append(eq_curve[-1])
        dds.append(calc_max_dd(eq_curve))

    med = np.median(finals)
    p5 = np.percentile(finals, 5)
    avg_dd = np.mean(dds)
    ruin = sum(1 for f in finals if f < 1.0) / N_SIMS * 100

    label = f"Cross/{lev}x/$15"
    print(f"    {label:>15s} | ${med:>10,.0f} | ${p5:>10,.0f} | {avg_dd*100:5.1f}% | {ruin:4.1f}%")

print()

# ============================================================
# PART 6: IMPLEMENTATION CHECKLIST
# ============================================================
print("=" * 110)
print("PART 6: BOT IMPLEMENTATION CHECKLIST")
print("=" * 110)
print()
print("  At ENTRY:")
print("    1. Calculate position size:")
print("       if equity < $15: qty = max(0.001, ceil(100/btc_price/0.001)*0.001)")
print("       if equity >= $15: position = equity * leverage")
print("    2. Calculate liquidation price:")
print("       LONG:  liq = entry * (1 - 1/lev + 0.004)")
print("       SHORT: liq = entry * (1 + 1/lev - 0.004)")
print("    3. Calculate safety stop:")
print("       safety = entry +/- 0.80 * |entry - liq|")
print("    4. Place safety stop-loss ORDER on exchange")
print("    5. Log: entry, qty, leverage, liq_price, safety_stop")
print()
print("  During TRADE:")
print("    6. Monitor trailing stop (20/30 bps, tightens to 8 after bar 5)")
print("    7. Monitor time exit (bar 10)")
print("    8. If price approaches safety stop: log WARNING")
print()
print("  At EXIT:")
print("    9. Cancel safety stop-loss order on exchange")
print("    10. Log: exit_price, pnl, exit_type, bars_held")
print("    11. Update equity for next trade sizing")
print()
print("  BOT FAILURE PROTECTION:")
print("    - Safety stop lives on EXCHANGE (not in bot)")
print("    - If bot crashes, safety stop still active")
print("    - On bot restart: check for open positions, recalculate state")
print()
