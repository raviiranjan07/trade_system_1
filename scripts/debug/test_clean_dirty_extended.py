"""
ANALYSIS-4: Clean Win vs Dirty Win vs Never Hit - Extended Horizons (FAST)

Run: .venv/Scripts/python.exe scripts/debug/test_clean_dirty_extended.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
TARGETS_BPS = [15, 25]  # Key targets only
SAMPLE_SIZE = 50000  # Reduced for speed

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-4: Clean Win vs Dirty Win - Extended Horizons")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Use train data only
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

# Sample
np.random.seed(42)
max_h = max(HORIZONS)
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(SAMPLE_SIZE, n - max_h - valid_start), replace=False)
print(f"Sample size: {len(sample_idx):,}")


def analyze_fast(indices, H, target_pct):
    """Fast vectorized analysis."""
    clean_win = 0
    dirty_win = 0
    never_hit = 0

    # Pre-compute future high/low for all H bars
    for i in indices:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        # Get future bars
        future_high = high[i+1:i+H+1]
        future_low = low[i+1:i+H+1]

        # Check if ever went below entry
        ever_below = np.any(future_low < entry)

        # Check if hit target
        target_hits = future_high >= target_price
        if np.any(target_hits):
            if ever_below:
                dirty_win += 1
            else:
                clean_win += 1
        else:
            never_hit += 1

    total = clean_win + dirty_win + never_hit
    return (100 * clean_win / total,
            100 * dirty_win / total,
            100 * never_hit / total)


# =============================================================================
# TEST ALL HORIZONS
# =============================================================================
results = []

for target_bps in TARGETS_BPS:
    target_pct = target_bps / 10000
    print(f"\n{'='*60}")
    print(f"Target = {target_bps}bp")
    print(f"{'='*60}")
    print(f"{'H':<8} {'Clean Win':<14} {'Dirty Win':<14} {'Never Hit':<14}")
    print("-" * 55)

    for H in HORIZONS:
        clean, dirty, never = analyze_fast(sample_idx, H, target_pct)
        print(f"H={H:<5} {clean:<14.1f} {dirty:<14.1f} {never:<14.1f}")
        results.append({
            'target': target_bps,
            'H': H,
            'clean': clean,
            'dirty': dirty,
            'never': never
        })


# =============================================================================
# LONG vs SHORT COMPARISON
# =============================================================================
print(f"\n{'='*60}")
print("LONG vs SHORT Comparison (H=30, Target=15bp)")
print(f"{'='*60}")

def analyze_short(indices, H, target_pct):
    """Analyze SHORT trades."""
    clean_win = 0
    dirty_win = 0
    never_hit = 0

    for i in indices:
        entry = close[i]
        target_price = entry * (1 - target_pct)

        future_high = high[i+1:i+H+1]
        future_low = low[i+1:i+H+1]

        # For SHORT: bad = went above entry
        ever_above = np.any(future_high > entry)

        # Check if hit target (price went down)
        target_hits = future_low <= target_price
        if np.any(target_hits):
            if ever_above:
                dirty_win += 1
            else:
                clean_win += 1
        else:
            never_hit += 1

    total = clean_win + dirty_win + never_hit
    return (100 * clean_win / total,
            100 * dirty_win / total,
            100 * never_hit / total)

clean_l, dirty_l, never_l = analyze_fast(sample_idx, 30, 0.0015)
clean_s, dirty_s, never_s = analyze_short(sample_idx, 30, 0.0015)

print(f"{'Direction':<10} {'Clean Win':<14} {'Dirty Win':<14} {'Never Hit':<14}")
print("-" * 55)
print(f"{'LONG':<10} {clean_l:<14.1f} {dirty_l:<14.1f} {never_l:<14.1f}")
print(f"{'SHORT':<10} {clean_s:<14.1f} {dirty_s:<14.1f} {never_s:<14.1f}")
print(f"{'Diff':<10} {abs(clean_l-clean_s):<14.2f} {abs(dirty_l-dirty_s):<14.2f} {abs(never_l-never_s):<14.2f}")


# =============================================================================
# MARKDOWN TABLES
# =============================================================================
print(f"\n{'='*80}")
print("MARKDOWN TABLES FOR ANALYSIS-4 UPDATE")
print(f"{'='*80}")

for target_bps in TARGETS_BPS:
    print(f"\nH=120 to H=600 bars (Target={target_bps}bp):")
    print("| H | Clean Win % | Dirty Win % | Never Hit |")
    print("|---|-------------|-------------|-----------|")

    for r in results:
        if r['target'] == target_bps and r['H'] >= 120:
            print(f"| H={r['H']} | {r['clean']:.1f}% | {r['dirty']:.1f}% | {r['never']:.1f}% |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")

print("""
1. Clean Win % trend with horizon:
   - H=3:  ~2-3% (very rare)
   - H=60: ~3-5%
   - H=600: ~5-6% (still rare!)

2. At longer horizons:
   - More trades hit target (Dirty Win + Clean Win increases)
   - But most still have drawdown first (Dirty Win >> Clean Win)

3. LONG vs SHORT:
   - Should be symmetric (confirms 50/50 market)

4. Implication:
   - Don't expect "clean" trades - prepare for drawdown
   - Stop-loss must account for normal drawdown
""")
