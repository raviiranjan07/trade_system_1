"""L1-EXP-004b: Hybrid Sizing — PROPER Train/Test Validation

PROBLEM: First test (hybrid_sizing.py) was overfitted.
  Bad conditions found in OOS -> tested on same OOS = circular.

FIX: Find bad conditions in TRAIN (2020-2023), test on OOS (2024-2025).

PROCESS:
  1. Run V1.3.2 backtest on TRAIN (2020-2023)
  2. Analyze: which conditions lose money in TRAIN?
  3. Define sizing rules from TRAIN data ONLY
  4. Test those rules on OOS (2024-2025)
  5. If it still works -> real signal. If not -> overfit.
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
from pathlib import Path

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
# LOAD DATA + INDICATORS
# ============================================================
config = load_config()
data_path = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
df = pd.read_parquet(data_path)
df.index = pd.to_datetime(df.index).tz_localize(None)
strategy = V12Strategy(config)
df = strategy.compute_indicators(df)


def enrich_trades(trades):
    """Add market conditions to each trade."""
    enriched = []
    for t in trades:
        btc_price = t.entry_price
        qty_min = max(0.001,
                      math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)

        signal_time = pd.Timestamp(t.signal_time)
        entry_time = pd.Timestamp(t.entry_time)
        idx = df.index.get_indexer([signal_time], method='nearest')[0]
        atr_pctl = float(df.iloc[idx]['atr_percentile']) if idx >= 0 and not pd.isna(df.iloc[idx]['atr_percentile']) else 50.0
        ema_sep = float(df.iloc[idx]['ema_separation']) if idx >= 0 and not pd.isna(df.iloc[idx]['ema_separation']) else 0.5

        enriched.append({
            'bps': t.net_profit_bps,
            'btc_price': btc_price,
            'qty_min': qty_min,
            'signal_type': t.signal_type,
            'direction': t.direction,
            'entry_hour': entry_time.hour,
            'entry_dow': entry_time.dayofweek,
            'atr_pctl': atr_pctl,
            'ema_sep': ema_sep,
        })
    return enriched


# ============================================================
# STEP 1: GET TRAIN + OOS TRADES
# ============================================================
print("=" * 100)
print("STEP 1: LOAD TRAIN AND OOS TRADES")
print("=" * 100)
print()

train_trades = run_backtest(config, start="2020-01-01", end="2023-12-31")
oos_trades = run_backtest(config, start="2024-01-01", end="2025-12-31")

train_data = enrich_trades(train_trades)
oos_data = enrich_trades(oos_trades)

print(f"  TRAIN (2020-2023): {len(train_data)} trades")
print(f"  OOS (2024-2025):   {len(oos_data)} trades")
print()


# ============================================================
# STEP 2: ANALYZE CONDITIONS IN TRAIN DATA
# ============================================================
print("=" * 100)
print("STEP 2: FIND BAD CONDITIONS IN TRAIN DATA")
print("=" * 100)
print()

def analyze_condition(data, label, filter_fn):
    """Analyze a condition's performance."""
    filtered = [td for td in data if filter_fn(td)]
    rest = [td for td in data if not filter_fn(td)]
    if not filtered:
        return None

    f_bps = [td['bps'] for td in filtered]
    r_bps = [td['bps'] for td in rest]
    f_wins = [b for b in f_bps if b > 0]
    r_wins = [b for b in r_bps if b > 0]

    return {
        'label': label,
        'n': len(filtered),
        'win_rate': len(f_wins) / len(f_bps) * 100 if f_bps else 0,
        'total_bps': sum(f_bps),
        'avg_bps': np.mean(f_bps),
        'rest_n': len(rest),
        'rest_win': len(r_wins) / len(r_bps) * 100 if r_bps else 0,
        'rest_total': sum(r_bps),
        'rest_avg': np.mean(r_bps),
    }


# Test all conditions on TRAIN data
conditions = [
    ("Monday LONG", lambda td: td['entry_dow'] == 0 and td['direction'] == 'LONG'),
    ("Monday SHORT", lambda td: td['entry_dow'] == 0 and td['direction'] == 'SHORT'),
    ("Tuesday LONG", lambda td: td['entry_dow'] == 1 and td['direction'] == 'LONG'),
    ("Weekend (Sat)", lambda td: td['entry_dow'] == 5),
    ("Weekend (Sun)", lambda td: td['entry_dow'] == 6),
    ("Night 00-04 UTC", lambda td: 0 <= td['entry_hour'] < 4),
    ("Night 04-08 UTC", lambda td: 4 <= td['entry_hour'] < 8),
    ("Asia 08-12 UTC", lambda td: 8 <= td['entry_hour'] < 12),
    ("Europe 12-16 UTC", lambda td: 12 <= td['entry_hour'] < 16),
    ("US 16-20 UTC", lambda td: 16 <= td['entry_hour'] < 20),
    ("Late 20-24 UTC", lambda td: 20 <= td['entry_hour'] < 24),
    ("Low ATR (<10)", lambda td: td['atr_pctl'] < 10),
    ("Low ATR (<20)", lambda td: td['atr_pctl'] < 20),
    ("High ATR (>70)", lambda td: td['atr_pctl'] > 70),
    ("High ATR (>90)", lambda td: td['atr_pctl'] > 90),
    ("Low EMA (<0.3)", lambda td: td['ema_sep'] < 0.3),
    ("Low EMA (<0.5)", lambda td: td['ema_sep'] < 0.5),
    ("High EMA (>1.0)", lambda td: td['ema_sep'] > 1.0),
    ("High EMA (>2.0)", lambda td: td['ema_sep'] > 2.0),
    ("V12_LONG", lambda td: td['signal_type'] == 'V12_LONG'),
    ("V12_SHORT", lambda td: td['signal_type'] == 'V12_SHORT'),
    ("BEAR_LONG", lambda td: td['signal_type'] == 'BEAR_LONG'),
    ("BULL_SHORT", lambda td: td['signal_type'] == 'BULL_SHORT'),
    ("V12_LONG + choppy", lambda td: td['signal_type'] == 'V12_LONG' and td['ema_sep'] < 0.5),
    ("V12_LONG + Monday", lambda td: td['signal_type'] == 'V12_LONG' and td['entry_dow'] == 0),
]

print(f"  {'Condition':>22s} | {'N':>4s} | {'Win%':>6s} | {'Total':>8s} | {'Avg':>8s} | {'Rest N':>6s} | {'Rest Win%':>9s} | {'Rest Avg':>8s} | {'Verdict':>10s}")
print(f"  {'-'*110}")

train_bad = []  # conditions that are bad in train
train_good = []

for label, fn in conditions:
    r = analyze_condition(train_data, label, fn)
    if r is None:
        continue

    # Is this condition significantly worse than the rest?
    if r['n'] >= 5:  # need minimum sample
        verdict = ""
        if r['avg_bps'] < 0:
            verdict = "LOSING"
            train_bad.append((label, fn, r))
        elif r['avg_bps'] < r['rest_avg'] * 0.5:
            verdict = "WEAK"
            train_bad.append((label, fn, r))
        elif r['avg_bps'] > r['rest_avg'] * 1.5:
            verdict = "STRONG"
            train_good.append((label, fn, r))
        else:
            verdict = "NORMAL"

        print(f"  {label:>22s} | {r['n']:>4d} | {r['win_rate']:>5.1f}% | {r['total_bps']:>+7.0f} | {r['avg_bps']:>+7.1f} | {r['rest_n']:>6d} | {r['rest_win']:>8.1f}% | {r['rest_avg']:>+7.1f} | {verdict:>10s}")

print()
print(f"  BAD conditions in TRAIN: {len(train_bad)}")
for label, _, r in train_bad:
    print(f"    {label}: {r['n']}t, {r['win_rate']:.1f}% win, {r['avg_bps']:+.1f} avg bps")
print()
print(f"  STRONG conditions in TRAIN: {len(train_good)}")
for label, _, r in train_good:
    print(f"    {label}: {r['n']}t, {r['win_rate']:.1f}% win, {r['avg_bps']:+.1f} avg bps")


# ============================================================
# STEP 3: VERIFY CONDITIONS ON OOS
# ============================================================
print()
print("=" * 100)
print("STEP 3: DO TRAIN CONDITIONS HOLD IN OOS?")
print("=" * 100)
print()

print(f"  {'Condition':>22s} | TRAIN: {'N':>3s} {'Win%':>6s} {'Avg':>7s} | OOS: {'N':>3s} {'Win%':>6s} {'Avg':>7s} | {'Holds?':>8s}")
print(f"  {'-'*95}")

valid_bad = []
valid_good = []

for label, fn, train_r in train_bad:
    oos_r = analyze_condition(oos_data, label, fn)
    if oos_r is None or oos_r['n'] < 3:
        holds = "NO DATA"
        print(f"  {label:>22s} | TRAIN: {train_r['n']:>3d} {train_r['win_rate']:>5.1f}% {train_r['avg_bps']:>+6.1f} | OOS: {'N/A':>3s} {'N/A':>6s} {'N/A':>7s} | {holds:>8s}")
        continue

    # Does it still lose or underperform in OOS?
    holds = "YES" if oos_r['avg_bps'] < oos_r['rest_avg'] * 0.7 else "NO"
    if holds == "YES":
        valid_bad.append((label, fn))

    print(f"  {label:>22s} | TRAIN: {train_r['n']:>3d} {train_r['win_rate']:>5.1f}% {train_r['avg_bps']:>+6.1f} | OOS: {oos_r['n']:>3d} {oos_r['win_rate']:>5.1f}% {oos_r['avg_bps']:>+6.1f} | {holds:>8s}")

print()
for label, fn, train_r in train_good:
    oos_r = analyze_condition(oos_data, label, fn)
    if oos_r is None or oos_r['n'] < 3:
        holds = "NO DATA"
        print(f"  {label:>22s} | TRAIN: {train_r['n']:>3d} {train_r['win_rate']:>5.1f}% {train_r['avg_bps']:>+6.1f} | OOS: {'N/A':>3s} {'N/A':>6s} {'N/A':>7s} | {holds:>8s}")
        continue

    holds = "YES" if oos_r['avg_bps'] > oos_r['rest_avg'] * 1.3 else "NO"
    if holds == "YES":
        valid_good.append((label, fn))

    print(f"  {label:>22s} | TRAIN: {train_r['n']:>3d} {train_r['win_rate']:>5.1f}% {train_r['avg_bps']:>+6.1f} | OOS: {oos_r['n']:>3d} {oos_r['win_rate']:>5.1f}% {oos_r['avg_bps']:>+6.1f} | {holds:>8s}")

print()
print(f"  VALIDATED BAD (holds in OOS): {len(valid_bad)}")
for label, _ in valid_bad:
    print(f"    - {label}")
print(f"  VALIDATED STRONG (holds in OOS): {len(valid_good)}")
for label, _ in valid_good:
    print(f"    - {label}")


# ============================================================
# STEP 4: BUILD SIZING RULES FROM VALIDATED CONDITIONS ONLY
# ============================================================
print()
print("=" * 100)
print("STEP 4: TEST VALIDATED HYBRID ON OOS")
print("=" * 100)
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


def simulate_hybrid(trade_list, step_fn, capital=STARTING_CAPITAL):
    equity = [capital]
    skipped = 0

    for td in trade_list:
        eq = equity[-1]
        if eq <= 0.01:
            equity.append(0.01)
            continue

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
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped


def run_mc_oos(step_fn, data, n_sims=N_SIMS, seed=42):
    np.random.seed(seed)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(n_sims):
        shuffled = list(data)
        np.random.shuffle(shuffled)
        eq, _ = simulate_hybrid(shuffled, step_fn)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin_count += 1

    return {
        'median': np.median(finals),
        'geo_mean': np.exp(np.mean(np.log(np.maximum(finals, 0.01)))),
        'p5': np.percentile(finals, 5),
        'avg_dd': np.mean(max_dds),
        'ruin_pct': ruin_count / n_sims * 100,
    }


# Build validated hybrid function
def config_validated(td, eq):
    """Only use conditions that are BAD in BOTH train AND oos."""
    base = 2.00

    for label, fn in valid_bad:
        if fn(td):
            return 4.00  # size down on validated bad condition

    return base


# Also test: only time-based (most likely to be structural)
def config_time_only(td, eq):
    """Size down on bad time conditions only."""
    base = 2.00

    # Time conditions are more likely structural (market structure)
    if td['entry_dow'] == 0 and td['direction'] == 'LONG':
        return 5.00
    if 4 <= td['entry_hour'] < 8:
        return 3.00

    return base


# Baselines
def config_fixed_200(td, eq):
    return 2.00

def config_fixed_250(td, eq):
    return 2.50


# Run all configs on OOS
configs = [
    ("Fixed $2.00", config_fixed_200),
    ("Fixed $2.50", config_fixed_250),
    ("Time-only", config_time_only),
    ("Validated hybrid", config_validated),
]

print(f"  Testing on OOS (2024-2025) with {len(oos_data)} trades:")
print()
print(f"  {'Config':>20s} | {'MC Median':>14s} | {'GeoMean':>14s} | {'P5':>14s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*90}")

oos_mc = {}
for name, fn in configs:
    r = run_mc_oos(fn, oos_data)
    oos_mc[name] = r
    print(f"  {name:>20s} | ${r['median']:>12,.0f} | ${r['geo_mean']:>12,.0f} | ${r['p5']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# ============================================================
# STEP 5: VERDICT
# ============================================================
print()
print("=" * 100)
print("STEP 5: VERDICT — Does validated hybrid beat fixed?")
print("=" * 100)
print()

baseline = oos_mc["Fixed $2.00"]
for name, r in oos_mc.items():
    if name == "Fixed $2.00":
        continue
    vs = (r['geo_mean'] / baseline['geo_mean'] - 1) * 100
    better = "BETTER" if r['geo_mean'] > baseline['geo_mean'] else "WORSE"
    safe = "SAFE" if r['ruin_pct'] <= 0.1 else "RISKY"
    print(f"  {name:>20s} vs Fixed $2.00: GeoMean {vs:>+8.1f}% | Ruin {r['ruin_pct']:.1f}% ({safe}) | {better}")

print()
print("  If VALIDATED HYBRID beats fixed -> the conditions are REAL (not overfit)")
print("  If it doesn't -> the conditions were noise, stick with fixed $/step")
