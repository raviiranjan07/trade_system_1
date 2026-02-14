"""L1-EXP-004: Hybrid Condition-Based Position Sizing

QUESTION: Does using different $/step for different conditions beat fixed $/step?

APPROACH:
  - Base: fixed $/step for ALL trades (from EXP-002/003)
  - Hybrid: different $/step depending on market conditions at entry

CONDITIONS TESTED (from loss analysis):
  - Signal type (V12_LONG, V12_SHORT, BEAR_LONG, BULL_SHORT)
  - ATR percentile (high = strong, low = weak)
  - EMA separation (high = trending, low = choppy)
  - Day of week (Monday LONG = 87% loss rate!)
  - Hour of day (04-08 UTC = biggest losses)

CONFIGS:
  A: Fixed $2.00/step (aggressive baseline)
  B: Fixed $2.50/step (conservative baseline)
  C: Signal-type based (different $/step per signal)
  D: ATR-based (high ATR = aggressive, low ATR = conservative)
  E: Time-based (avoid bad times)
  F: Combined conditions (signal + ATR + time)
  G: "Size up on strong" (base conservative, aggressive on strong signals)
  H: "Size down on weak" (base aggressive, conservative on weak signals)
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from v12.backtest import run_backtest
from v12.config.loader import load_config
from v12.strategy import V12Strategy

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
LEVERAGE = 125
MAINT_MARGIN_RATE = 0.004
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100
N_SIMS = 1000

# ============================================================
# LOAD TRADES + MARKET CONDITIONS
# ============================================================
config = load_config()
trades = run_backtest(config)

# Load OHLCV and compute indicators to get ATR, EMA sep at each bar
data_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
df = pd.read_parquet(data_path)
df.index = pd.to_datetime(df.index).tz_localize(None)
strategy = V12Strategy(config)
df = strategy.compute_indicators(df)

# Build enriched trade data with market conditions
trade_data = []
for t in trades:
    btc_price = t.entry_price
    qty_min = max(0.001,
                  math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)

    # Look up indicators at signal_time (the bar where signal fired)
    signal_time = pd.Timestamp(t.signal_time)
    entry_time = pd.Timestamp(t.entry_time)

    # Find closest bar in indicator df
    idx = df.index.get_indexer([signal_time], method='nearest')[0]
    atr_pctl = float(df.iloc[idx]['atr_percentile']) if idx >= 0 and not pd.isna(df.iloc[idx]['atr_percentile']) else 50.0
    ema_sep = float(df.iloc[idx]['ema_separation']) if idx >= 0 and not pd.isna(df.iloc[idx]['ema_separation']) else 0.5

    trade_data.append({
        'bps': t.net_profit_bps,
        'btc_price': btc_price,
        'qty_min': qty_min,
        'signal_type': t.signal_type,
        'direction': t.direction,
        'entry_hour': entry_time.hour,
        'entry_dow': entry_time.dayofweek,  # 0=Monday
        'atr_pctl': atr_pctl,
        'ema_sep': ema_sep,
        'exit_reason': t.exit_reason,
    })

returns_bps = [td['bps'] for td in trade_data]
wins = [r for r in returns_bps if r > 0]

print("=" * 100)
print("L1-EXP-004: HYBRID CONDITION-BASED POSITION SIZING")
print("=" * 100)
print(f"  V1.3.2: {len(trade_data)} trades, {len(wins)/len(trade_data)*100:.1f}% win")
print(f"  Mean: {np.mean(returns_bps):+.1f} bps | Std: {np.std(returns_bps):.1f} bps")
print()

# Show condition distribution
print("  CONDITION DISTRIBUTION:")
st_counts = {}
for td in trade_data:
    st = td['signal_type']
    st_counts[st] = st_counts.get(st, 0) + 1
for st, cnt in sorted(st_counts.items()):
    st_trades = [td['bps'] for td in trade_data if td['signal_type'] == st]
    st_wins = [r for r in st_trades if r > 0]
    print(f"    {st:<12}: {cnt:>3} trades, {len(st_wins)/len(st_trades)*100:.1f}% win, {sum(st_trades):+.0f} bps")

monday_long = [td for td in trade_data if td['entry_dow'] == 0 and td['direction'] == 'LONG']
low_atr = [td for td in trade_data if td['atr_pctl'] < 10]
night_trades = [td for td in trade_data if 0 <= td['entry_hour'] < 8]
print(f"    Monday LONG:  {len(monday_long)} trades, {sum(td['bps'] for td in monday_long):+.0f} bps")
print(f"    Low ATR(<10): {len(low_atr)} trades, {sum(td['bps'] for td in low_atr):+.0f} bps")
print(f"    Night 00-08:  {len(night_trades)} trades, {sum(td['bps'] for td in night_trades):+.0f} bps")
print()


# ============================================================
# SIMULATION ENGINE
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


def simulate_hybrid(trade_list, step_fn, capital=STARTING_CAPITAL):
    """Simulate with condition-based $/step.

    step_fn: function(trade_dict, wallet) -> $/step value
    """
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]
        if eq <= 0.01:
            equity.append(0.01)
            continue

        # Get $/step based on conditions
        step = step_fn(td, eq)

        steps = max(1, int(eq / step))
        qty = steps * BINANCE_STEP_SIZE
        qty = max(qty, td['qty_min'])

        position = qty * td['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        if eq < margin:
            equity.append(eq)
            skipped += 1
            continue

        pnl = position * (td['bps'] / 10000)
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def run_mc(step_fn, n_sims=N_SIMS, seed=42):
    np.random.seed(seed)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(n_sims):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq, _, _ = simulate_hybrid(shuffled, step_fn)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin_count += 1

    return {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'geo_mean': np.exp(np.mean(np.log(np.maximum(finals, 0.01)))),
        'avg_dd': np.mean(max_dds),
        'ruin_pct': ruin_count / n_sims * 100,
    }


# ============================================================
# DEFINE CONFIGS
# ============================================================

# A: Fixed $2.00/step
def config_a(td, eq):
    return 2.00

# B: Fixed $2.50/step
def config_b(td, eq):
    return 2.50

# C: Signal-type based
# BEAR_LONG/BULL_SHORT have higher win rates -> more aggressive
# V12_SHORT is strong -> aggressive
# V12_LONG is weakest -> conservative
def config_c(td, eq):
    if td['signal_type'] == 'BEAR_LONG':
        return 1.85  # 71.4% win, PF 3.42
    elif td['signal_type'] == 'BULL_SHORT':
        return 2.00  # PF 8.71
    elif td['signal_type'] == 'V12_SHORT':
        return 2.00  # strong
    else:  # V12_LONG
        return 3.00  # weakest

# D: ATR-based
# High ATR = trending = bigger position
# Low ATR = choppy = smaller position
def config_d(td, eq):
    if td['atr_pctl'] >= 60:
        return 2.00  # strong volatility
    elif td['atr_pctl'] >= 30:
        return 2.50  # normal
    else:
        return 4.00  # low vol = weak signals

# E: Time-based
# Avoid Monday LONG, avoid 04-08 UTC
def config_e(td, eq):
    if td['entry_dow'] == 0 and td['direction'] == 'LONG':
        return 5.00  # Monday LONG = 87% loss
    elif 4 <= td['entry_hour'] < 8:
        return 3.50  # 04-08 UTC = biggest losses
    else:
        return 2.00  # normal

# F: Combined (signal + ATR + time)
def config_f(td, eq):
    base = 2.00

    # Signal type adjustment
    if td['signal_type'] == 'V12_LONG':
        base = 2.50  # weakest signal
    elif td['signal_type'] in ('BEAR_LONG', 'BULL_SHORT'):
        base = 1.85  # strongest signals

    # ATR adjustment
    if td['atr_pctl'] < 15:
        base *= 1.5  # lower position in low vol (bigger $/step = less qty)

    # Time penalty
    if td['entry_dow'] == 0 and td['direction'] == 'LONG':
        base *= 2.0  # half position on Monday LONG
    elif 4 <= td['entry_hour'] < 8:
        base *= 1.3  # reduce in bad hours

    return base

# G: "Size up on strong" — conservative base, aggressive on strong
def config_g(td, eq):
    # Base: conservative
    base = 2.50

    # Size UP conditions
    if td['atr_pctl'] >= 60 and td['ema_sep'] >= 1.0:
        return 1.85  # strong trend + vol = max position
    if td['signal_type'] in ('BEAR_LONG', 'BULL_SHORT') and td['ema_sep'] >= 1.0:
        return 2.00  # counter-trend in strong trend = good

    return base

# H: "Size down on weak" — aggressive base, conservative on weak
def config_h(td, eq):
    # Base: aggressive
    base = 2.00

    # Size DOWN conditions
    if td['entry_dow'] == 0 and td['direction'] == 'LONG':
        return 5.00  # Monday LONG
    if td['atr_pctl'] < 10:
        return 4.00  # dead zone
    if 4 <= td['entry_hour'] < 8:
        return 3.00  # bad hours
    if td['signal_type'] == 'V12_LONG' and td['ema_sep'] < 0.5:
        return 4.00  # choppy LONG

    return base


# ============================================================
# PART 1: ORIGINAL ORDER — all configs
# ============================================================
print("=" * 100)
print("PART 1: ORIGINAL TRADE ORDER")
print("=" * 100)
print()

configs = [
    ("A: Fixed $2.00", config_a),
    ("B: Fixed $2.50", config_b),
    ("C: Signal-type", config_c),
    ("D: ATR-based", config_d),
    ("E: Time-based", config_e),
    ("F: Combined", config_f),
    ("G: Size-up strong", config_g),
    ("H: Size-down weak", config_h),
]

print(f"  {'Config':>20s} | {'Final':>14s} | {'MaxDD':>7s} | {'MinEq':>8s}")
print(f"  {'-'*60}")

for name, fn in configs:
    eq, skip, liq = simulate_hybrid(trade_data, fn)
    final = eq[-1]
    dd = calc_max_dd(eq)
    min_eq = min(eq)
    print(f"  {name:>20s} | ${final:>12,.0f} | {dd*100:5.1f}% | ${min_eq:>6.2f}")


# ============================================================
# PART 2: MONTE CARLO — 1000 paths per config
# ============================================================
print()
print("=" * 100)
print("PART 2: MONTE CARLO (1000 paths per config)")
print("=" * 100)
print()

print(f"  {'Config':>20s} | {'Median':>14s} | {'GeoMean':>14s} | {'P5':>14s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*95}")

mc_results = {}
for name, fn in configs:
    r = run_mc(fn)
    mc_results[name] = r
    print(f"  {name:>20s} | ${r['median']:>12,.0f} | ${r['geo_mean']:>12,.0f} | ${r['p5']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# ============================================================
# PART 3: COMPARISON — Hybrid vs Fixed
# ============================================================
print()
print("=" * 100)
print("PART 3: DOES HYBRID BEAT FIXED?")
print("=" * 100)
print()

baseline_a = mc_results["A: Fixed $2.00"]
baseline_b = mc_results["B: Fixed $2.50"]

print(f"  BASELINES:")
print(f"    A (Fixed $2.00): Median ${baseline_a['median']:>12,.0f} | GeoMean ${baseline_a['geo_mean']:>12,.0f} | Ruin {baseline_a['ruin_pct']:.1f}%")
print(f"    B (Fixed $2.50): Median ${baseline_b['median']:>12,.0f} | GeoMean ${baseline_b['geo_mean']:>12,.0f} | Ruin {baseline_b['ruin_pct']:.1f}%")
print()

print(f"  HYBRID RESULTS (vs Fixed $2.00):")
for name, fn in configs:
    if name.startswith(("A:", "B:")):
        continue
    r = mc_results[name]
    vs_a_median = (r['median'] / baseline_a['median'] - 1) * 100
    vs_a_geo = (r['geo_mean'] / baseline_a['geo_mean'] - 1) * 100
    better = "BETTER" if r['geo_mean'] > baseline_a['geo_mean'] else "WORSE"
    print(f"    {name:>20s}: Median {vs_a_median:>+7.1f}% | GeoMean {vs_a_geo:>+7.1f}% | Ruin {r['ruin_pct']:.1f}% | {better}")

print()
print(f"  HYBRID RESULTS (vs Fixed $2.50):")
for name, fn in configs:
    if name.startswith(("A:", "B:")):
        continue
    r = mc_results[name]
    vs_b_median = (r['median'] / baseline_b['median'] - 1) * 100
    vs_b_geo = (r['geo_mean'] / baseline_b['geo_mean'] - 1) * 100
    better = "BETTER" if r['geo_mean'] > baseline_b['geo_mean'] else "WORSE"
    print(f"    {name:>20s}: Median {vs_b_median:>+7.1f}% | GeoMean {vs_b_geo:>+7.1f}% | Ruin {r['ruin_pct']:.1f}% | {better}")


# ============================================================
# PART 4: VERDICT
# ============================================================
print()
print("=" * 100)
print("PART 4: VERDICT")
print("=" * 100)
print()

# Find best by geo mean with ruin <= 1%
safe_configs = {k: v for k, v in mc_results.items() if v['ruin_pct'] <= 1.0}
if safe_configs:
    best = max(safe_configs, key=lambda k: safe_configs[k]['geo_mean'])
    r = safe_configs[best]
    print(f"  BEST CONFIG (ruin <= 1%): {best}")
    print(f"    Median: ${r['median']:,.0f} | GeoMean: ${r['geo_mean']:,.0f} | P5: ${r['p5']:,.0f}")
    print(f"    AvgDD: {r['avg_dd']*100:.1f}% | Ruin: {r['ruin_pct']:.1f}%")
    print()

    is_hybrid = not best.startswith(("A:", "B:"))
    if is_hybrid:
        print(f"  HYBRID WINS over fixed sizing!")
        print(f"    vs Fixed $2.00: GeoMean {(r['geo_mean']/baseline_a['geo_mean']-1)*100:+.1f}%")
        print(f"    vs Fixed $2.50: GeoMean {(r['geo_mean']/baseline_b['geo_mean']-1)*100:+.1f}%")
    else:
        print(f"  FIXED WINS. Hybrid approaches don't beat simple fixed $/step.")
        print(f"  This confirms old L1-EXP-005 finding: variable sizing hurts geometric growth.")
