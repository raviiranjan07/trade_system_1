"""
EXP-004: Path Analysis - Early Exit Detection

Objective: Compare how failures vs successes develop bar-by-bar
Can we detect failures early and exit before big losses?

We analyze:
1. RSI Oversold (LONG) - 2 failures in OOS
2. RSI Overbought (SHORT) - 11 failures in OOS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-004")
PLOTS_PATH = RESULTS_PATH / "plots"
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("PATH ANALYSIS - EARLY EXIT DETECTION")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df, RSI_PERIOD)

# Detect signals
df['oversold'] = df['rsi'] < RSI_OVERSOLD
df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# Use OOS data
oos_data = df['2024-01-01':'2025-12-30'].copy()
print(f"OOS Data: {len(oos_data):,} bars")

# =============================================================================
# PATH EXTRACTION FUNCTION
# =============================================================================

def extract_paths(data, signal_mask, signal_type='oversold'):
    """
    Extract bar-by-bar PnL paths for each signal
    Returns paths for successes and failures separately
    """
    signals = data[signal_mask].copy()

    success_paths = []
    failure_paths = []

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)

        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']

        # Calculate PnL at each bar
        path = []
        for bar in range(1, HORIZON + 1):
            if idx_pos + bar >= len(data):
                break

            bar_data = data.iloc[idx_pos + bar]

            if signal_type == 'oversold':
                # LONG: profit if price goes up
                # Use close price for PnL path
                pnl = (bar_data['close'] - entry_price) / entry_price * 10000
                # Also track best/worst within the bar
                best = (bar_data['high'] - entry_price) / entry_price * 10000
                worst = (bar_data['low'] - entry_price) / entry_price * 10000
            else:
                # SHORT: profit if price goes down
                pnl = (entry_price - bar_data['close']) / entry_price * 10000
                best = (entry_price - bar_data['low']) / entry_price * 10000
                worst = (entry_price - bar_data['high']) / entry_price * 10000

            path.append({
                'bar': bar,
                'pnl': pnl,
                'best': best,
                'worst': worst
            })

        if len(path) < HORIZON:
            continue

        # Determine if success or failure
        # Success = price moved in expected direction at some point
        if signal_type == 'oversold':
            future_high = max([p['best'] for p in path])
            is_success = future_high > 0
        else:
            future_low = max([p['best'] for p in path])
            is_success = future_low > 0

        path_data = {
            'timestamp': idx,
            'entry_price': entry_price,
            'path': path,
            'final_pnl': path[-1]['pnl'],
            'max_profit': max([p['best'] for p in path]),
            'max_drawdown': min([p['worst'] for p in path])
        }

        if is_success:
            success_paths.append(path_data)
        else:
            failure_paths.append(path_data)

    return success_paths, failure_paths

# =============================================================================
# ANALYZE OVERSOLD (LONG)
# =============================================================================

print("\n" + "="*70)
print("OVERSOLD (LONG) PATH ANALYSIS")
print("="*70)

oversold_mask = oos_data['rsi_oversold_cross'] == True
os_success, os_failure = extract_paths(oos_data, oversold_mask, 'oversold')

print(f"Successes: {len(os_success)}")
print(f"Failures: {len(os_failure)}")

# Calculate average path
def avg_path(paths, metric='pnl'):
    if not paths:
        return []
    avg = []
    for bar in range(HORIZON):
        values = [p['path'][bar][metric] for p in paths if len(p['path']) > bar]
        avg.append(np.mean(values) if values else 0)
    return avg

os_success_avg = avg_path(os_success, 'pnl')
os_failure_avg = avg_path(os_failure, 'pnl')

print("\nOversold - Average PnL by Bar:")
print(f"{'Bar':<5} {'Success':<12} {'Failure':<12} {'Difference':<12}")
print("-"*40)
for bar in range(HORIZON):
    s = os_success_avg[bar] if bar < len(os_success_avg) else 0
    f = os_failure_avg[bar] if bar < len(os_failure_avg) else 0
    diff = s - f
    print(f"{bar+1:<5} {s:+.1f} bps{'':<3} {f:+.1f} bps{'':<3} {diff:+.1f}")

# =============================================================================
# ANALYZE OVERBOUGHT (SHORT)
# =============================================================================

print("\n" + "="*70)
print("OVERBOUGHT (SHORT) PATH ANALYSIS")
print("="*70)

overbought_mask = oos_data['rsi_overbought_cross'] == True
ob_success, ob_failure = extract_paths(oos_data, overbought_mask, 'overbought')

print(f"Successes: {len(ob_success)}")
print(f"Failures: {len(ob_failure)}")

ob_success_avg = avg_path(ob_success, 'pnl')
ob_failure_avg = avg_path(ob_failure, 'pnl')

print("\nOverbought - Average PnL by Bar:")
print(f"{'Bar':<5} {'Success':<12} {'Failure':<12} {'Difference':<12}")
print("-"*40)
for bar in range(HORIZON):
    s = ob_success_avg[bar] if bar < len(ob_success_avg) else 0
    f = ob_failure_avg[bar] if bar < len(ob_failure_avg) else 0
    diff = s - f
    print(f"{bar+1:<5} {s:+.1f} bps{'':<3} {f:+.1f} bps{'':<3} {diff:+.1f}")

# =============================================================================
# KEY QUESTION: Can we detect failures at bar 1, 2, or 3?
# =============================================================================

print("\n" + "="*70)
print("EARLY DETECTION ANALYSIS")
print("="*70)

def early_detection_stats(success_paths, failure_paths, signal_name):
    print(f"\n--- {signal_name} ---")

    for bar in [1, 2, 3]:
        # What % are in profit at this bar?
        success_positive = sum(1 for p in success_paths if p['path'][bar-1]['pnl'] > 0)
        failure_positive = sum(1 for p in failure_paths if p['path'][bar-1]['pnl'] > 0)

        success_rate = success_positive / len(success_paths) * 100 if success_paths else 0
        failure_rate = failure_positive / len(failure_paths) * 100 if failure_paths else 0

        print(f"\nBar {bar}: % in profit")
        print(f"  Successes: {success_rate:.1f}% ({success_positive}/{len(success_paths)})")
        print(f"  Failures:  {failure_rate:.1f}% ({failure_positive}/{len(failure_paths)})")

        if failure_rate < success_rate:
            print(f"  --> Failures are {success_rate - failure_rate:.1f}pp LESS likely to be profitable")

        # What's the average PnL at this bar?
        success_avg = np.mean([p['path'][bar-1]['pnl'] for p in success_paths])
        failure_avg = np.mean([p['path'][bar-1]['pnl'] for p in failure_paths])

        print(f"  Avg PnL - Successes: {success_avg:+.1f} bps")
        print(f"  Avg PnL - Failures:  {failure_avg:+.1f} bps")

early_detection_stats(os_success, os_failure, "OVERSOLD (LONG)")
early_detection_stats(ob_success, ob_failure, "OVERBOUGHT (SHORT)")

# =============================================================================
# EARLY EXIT RULE SIMULATION
# =============================================================================

print("\n" + "="*70)
print("EARLY EXIT RULE TEST")
print("="*70)

def test_early_exit(success_paths, failure_paths, signal_name, exit_bar, min_pnl_threshold):
    """
    Test rule: If PnL at exit_bar < min_pnl_threshold, exit early
    """
    # Successes that would be wrongly exited
    wrong_exits = sum(1 for p in success_paths
                      if p['path'][exit_bar-1]['pnl'] < min_pnl_threshold)

    # Failures that would be correctly exited
    correct_exits = sum(1 for p in failure_paths
                        if p['path'][exit_bar-1]['pnl'] < min_pnl_threshold)

    # Calculate savings
    total_successes = len(success_paths)
    total_failures = len(failure_paths)

    wrong_exit_pct = wrong_exits / total_successes * 100 if total_successes else 0
    correct_exit_pct = correct_exits / total_failures * 100 if total_failures else 0

    # Estimate PnL impact
    # Lost profit from wrong exits (assume avg +30 bps per success)
    lost_profit = wrong_exits * 30

    # Saved loss from correct exits (difference between early exit loss and full loss)
    if failure_paths:
        avg_failure_loss = np.mean([abs(p['max_drawdown']) for p in failure_paths])
        avg_early_loss = np.mean([abs(p['path'][exit_bar-1]['pnl'])
                                  for p in failure_paths
                                  if p['path'][exit_bar-1]['pnl'] < min_pnl_threshold])
        saved_per_failure = avg_failure_loss - avg_early_loss if correct_exits > 0 else 0
        total_saved = correct_exits * saved_per_failure
    else:
        total_saved = 0
        saved_per_failure = 0

    return {
        'signal': signal_name,
        'rule': f"Exit at bar {exit_bar} if PnL < {min_pnl_threshold} bps",
        'wrong_exits': wrong_exits,
        'wrong_exit_pct': wrong_exit_pct,
        'correct_exits': correct_exits,
        'correct_exit_pct': correct_exit_pct,
        'lost_profit': lost_profit,
        'saved_loss': total_saved,
        'net_impact': total_saved - lost_profit
    }

# Test various rules for OVERBOUGHT
print("\nTesting early exit rules for OVERBOUGHT (SHORT):")
print("-"*80)

rules_to_test = [
    (1, 0),    # Exit bar 1 if not profitable
    (1, -10),  # Exit bar 1 if loss > 10 bps
    (2, 0),    # Exit bar 2 if not profitable
    (2, -10),  # Exit bar 2 if loss > 10 bps
    (2, -20),  # Exit bar 2 if loss > 20 bps
    (3, 0),    # Exit bar 3 if not profitable
    (3, -15),  # Exit bar 3 if loss > 15 bps
]

results = []
for exit_bar, threshold in rules_to_test:
    result = test_early_exit(ob_success, ob_failure, "OVERBOUGHT", exit_bar, threshold)
    results.append(result)

    print(f"\n{result['rule']}:")
    print(f"  Wrong exits (kill winners): {result['wrong_exits']}/{len(ob_success)} ({result['wrong_exit_pct']:.1f}%)")
    print(f"  Correct exits (save from failures): {result['correct_exits']}/{len(ob_failure)} ({result['correct_exit_pct']:.1f}%)")
    print(f"  Net impact: {result['net_impact']:+.0f} bps")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Average paths comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Oversold
ax = axes[0]
bars = range(1, HORIZON + 1)
ax.plot(bars, os_success_avg, 'g-', linewidth=2, label=f'Successes ({len(os_success)})', marker='o')
if os_failure_avg:
    ax.plot(bars, os_failure_avg, 'r-', linewidth=2, label=f'Failures ({len(os_failure)})', marker='x')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axhline(y=12, color='orange', linestyle='--', alpha=0.5, label='12 bps target')
ax.set_xlabel('Bar')
ax.set_ylabel('Average PnL (bps)')
ax.set_title('OVERSOLD (LONG) - Path Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Overbought
ax = axes[1]
ax.plot(bars, ob_success_avg, 'g-', linewidth=2, label=f'Successes ({len(ob_success)})', marker='o')
if ob_failure_avg:
    ax.plot(bars, ob_failure_avg, 'r-', linewidth=2, label=f'Failures ({len(ob_failure)})', marker='x')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axhline(y=12, color='orange', linestyle='--', alpha=0.5, label='12 bps target')
ax.set_xlabel('Bar')
ax.set_ylabel('Average PnL (bps)')
ax.set_title('OVERBOUGHT (SHORT) - Path Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_PATH / 'path_comparison.png', dpi=150)
plt.close()
print(f"Saved: {PLOTS_PATH / 'path_comparison.png'}")

# Plot 2: Individual failure paths for overbought
if ob_failure:
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, p in enumerate(ob_failure):
        pnl_path = [x['pnl'] for x in p['path']]
        ax.plot(bars, pnl_path, 'r-', alpha=0.5, linewidth=1,
               label=f"Failure {i+1}" if i < 5 else None)

    # Add average success path for reference
    ax.plot(bars, ob_success_avg, 'g-', linewidth=3, label='Avg Success', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bar')
    ax.set_ylabel('PnL (bps)')
    ax.set_title(f'OVERBOUGHT (SHORT) - All {len(ob_failure)} Failure Paths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'failure_paths_overbought.png', dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_PATH / 'failure_paths_overbought.png'}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n1. OVERSOLD (LONG):")
print(f"   - Only {len(os_failure)} failures out of {len(os_success) + len(os_failure)}")
print(f"   - Failures are rare, path analysis has limited sample")

print("\n2. OVERBOUGHT (SHORT):")
print(f"   - {len(ob_failure)} failures out of {len(ob_success) + len(ob_failure)}")
if ob_failure:
    bar1_diff = ob_success_avg[0] - ob_failure_avg[0]
    print(f"   - At bar 1: Successes avg {ob_success_avg[0]:+.1f} bps, Failures avg {ob_failure_avg[0]:+.1f} bps")
    print(f"   - Difference at bar 1: {bar1_diff:.1f} bps")

print("\n3. KEY FINDING:")
if ob_failure and len(ob_failure_avg) > 0:
    if ob_failure_avg[0] < 0 and ob_success_avg[0] > 0:
        print("   - Failures tend to go negative early while successes go positive")
        print("   - Early exit rule MAY be viable")
    else:
        print("   - Both failures and successes start similarly")
        print("   - Early detection is DIFFICULT")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
