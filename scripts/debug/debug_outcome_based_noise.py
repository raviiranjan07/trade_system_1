"""
Outcome-Based Noise Discovery

Run: .venv/Scripts/python.exe debug_outcome_based_noise.py

APPROACH:
Instead of guessing noise thresholds, we work BACKWARDS from outcomes:
1. Label each bar: WIN (hit target) vs LOSS (hit stop) vs TIMEOUT
2. Compare features between WIN and LOSS bars
3. Find features that strongly predict LOSS
4. Those features = data-driven noise boundaries

KEY INSIGHT:
- If features predict LOSS -> we can filter noise
- If features don't predict LOSS -> need directional edge, not noise filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZON = 60  # 1 hour
INVALIDATION_RATIO = 0.5  # 2:1 R:R

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("OUTCOME-BASED NOISE DISCOVERY")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# =============================================================================
# COMPUTE FEATURES
# =============================================================================
print("\nComputing features...")

df = ohlcv.copy()

# EMAs and slopes
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
df['ema50_slope'] = df['ema50'].pct_change(5) * 100
df['ema200_slope'] = df['ema200'].pct_change(5) * 100

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))

# Volume percentile (rolling 100 bars)
df['volume_pct'] = df['volume'].rolling(100).rank(pct=True) * 100

# ATR and ATR percentile
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr'] = df['tr'].rolling(14).mean()
df['atr_pct'] = df['atr'].rolling(100).rank(pct=True) * 100

# Candle metrics
df['body'] = abs(df['close'] - df['open'])
df['range'] = df['high'] - df['low']
df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)

# Range position (where is close in recent range)
df['range_high'] = df['high'].rolling(20).max()
df['range_low'] = df['low'].rolling(20).min()
df['range_position'] = (df['close'] - df['range_low']) / (df['range_high'] - df['range_low']).replace(0, np.nan)

# Direction changes (chop indicator)
df['direction'] = np.sign(df['close'].diff())
df['direction_change'] = (df['direction'] != df['direction'].shift(1)).astype(int)
df['direction_changes_10'] = df['direction_change'].rolling(10).sum()

# Hour and day
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Returns
df['return_5'] = df['close'].pct_change(5) * 100
df['return_15'] = df['close'].pct_change(15) * 100

# Trend alignment
df['trend_aligned'] = ((df['close'] > df['ema50']) & (df['ema50'] > df['ema200'])).astype(int) - \
                      ((df['close'] < df['ema50']) & (df['ema50'] < df['ema200'])).astype(int)

# Drop NaN
df = df.dropna()
print(f"After feature computation: {len(df):,} candles")

# Split
train_df = df[df.index <= TRAIN_END].copy()
test_df = df[df.index >= TEST_START].copy()
print(f"TRAIN: {len(train_df):,} | TEST: {len(test_df):,}")

# =============================================================================
# COMPUTE OUTCOMES
# =============================================================================
print(f"\nComputing outcomes (H={HORIZON})...")

def compute_outcomes(ohlcv_df, horizon, target_pct, stop_pct):
    """
    Label each bar with outcome:
    - WIN: Hit target before stop
    - LOSS: Hit stop before target
    - TIMEOUT: Neither within horizon
    """
    close = ohlcv_df['close'].values
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    n = len(ohlcv_df)

    # For LONG trades
    long_outcome = np.full(n, 'TIMEOUT', dtype=object)

    for i in range(n - horizon):
        entry = close[i]
        target = entry * (1 + target_pct)
        stop = entry * (1 - stop_pct)

        for j in range(i+1, i+1+horizon):
            if low[j] <= stop:
                long_outcome[i] = 'LOSS'
                break
            if high[j] >= target:
                long_outcome[i] = 'WIN'
                break

    # For SHORT trades
    short_outcome = np.full(n, 'TIMEOUT', dtype=object)

    for i in range(n - horizon):
        entry = close[i]
        target = entry * (1 - target_pct)
        stop = entry * (1 + stop_pct)

        for j in range(i+1, i+1+horizon):
            if high[j] >= stop:
                short_outcome[i] = 'LOSS'
                break
            if low[j] <= target:
                short_outcome[i] = 'WIN'
                break

    return long_outcome, short_outcome


# Compute thresholds from train (median move)
print("Computing thresholds from TRAIN...")
close = train_df['close'].values
high = train_df['high'].values
low = train_df['low'].values
n_train = len(train_df)

max_moves = []
for i in range(min(100000, n_train - HORIZON)):  # Sample for speed
    entry = close[i]
    future_high = np.max(high[i+1:i+1+HORIZON])
    future_low = np.min(low[i+1:i+1+HORIZON])
    max_up = (future_high - entry) / entry
    max_down = (entry - future_low) / entry
    max_moves.append(max(max_up, max_down))

median_move = np.percentile(max_moves, 50)
target_pct = median_move
stop_pct = target_pct * INVALIDATION_RATIO

print(f"  Target: {target_pct * 10000:.1f} bps")
print(f"  Stop: {stop_pct * 10000:.1f} bps")

# Compute outcomes
print("Computing outcomes on TRAIN...")
train_long_outcome, train_short_outcome = compute_outcomes(train_df, HORIZON, target_pct, stop_pct)

# Add to dataframe
train_df = train_df.iloc[:-HORIZON].copy()  # Remove last H bars
train_df['long_outcome'] = train_long_outcome[:-HORIZON]
train_df['short_outcome'] = train_short_outcome[:-HORIZON]

# =============================================================================
# ANALYZE FEATURES BY OUTCOME
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE ANALYSIS: LONG TRADES")
print("=" * 70)

features = [
    'volume_pct', 'atr_pct', 'rsi', 'body_ratio', 'range_position',
    'direction_changes_10', 'ema50_slope', 'ema200_slope',
    'return_5', 'return_15', 'trend_aligned', 'hour'
]

def analyze_features(df, outcome_col, features):
    """Compare feature distributions between WIN and LOSS outcomes."""
    wins = df[df[outcome_col] == 'WIN']
    losses = df[df[outcome_col] == 'LOSS']
    timeouts = df[df[outcome_col] == 'TIMEOUT']

    print(f"\nOutcome distribution:")
    print(f"  WIN:     {len(wins):>10,} ({len(wins)/len(df)*100:.1f}%)")
    print(f"  LOSS:    {len(losses):>10,} ({len(losses)/len(df)*100:.1f}%)")
    print(f"  TIMEOUT: {len(timeouts):>10,} ({len(timeouts)/len(df)*100:.1f}%)")

    print(f"\n{'Feature':<22} {'WIN Mean':>12} {'LOSS Mean':>12} {'Diff':>10} {'Effect':>10}")
    print("-" * 70)

    results = []
    for feat in features:
        win_mean = wins[feat].mean()
        loss_mean = losses[feat].mean()
        diff = win_mean - loss_mean

        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((wins[feat].std()**2 + losses[feat].std()**2) / 2)
        if pooled_std > 0:
            cohens_d = diff / pooled_std
        else:
            cohens_d = 0

        # Effect size interpretation
        if abs(cohens_d) >= 0.8:
            effect = "LARGE"
        elif abs(cohens_d) >= 0.5:
            effect = "MEDIUM"
        elif abs(cohens_d) >= 0.2:
            effect = "SMALL"
        else:
            effect = "TINY"

        results.append({
            'feature': feat,
            'win_mean': win_mean,
            'loss_mean': loss_mean,
            'diff': diff,
            'cohens_d': cohens_d,
            'effect': effect
        })

        print(f"{feat:<22} {win_mean:>12.2f} {loss_mean:>12.2f} {diff:>+10.2f} {effect:>10}")

    return results

print("\n--- LONG TRADES ---")
long_results = analyze_features(train_df, 'long_outcome', features)

print("\n--- SHORT TRADES ---")
short_results = analyze_features(train_df, 'short_outcome', features)

# =============================================================================
# FIND TOP DISCRIMINATING FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("TOP DISCRIMINATING FEATURES (by effect size)")
print("=" * 70)

# Sort by absolute Cohen's d
long_sorted = sorted(long_results, key=lambda x: abs(x['cohens_d']), reverse=True)
short_sorted = sorted(short_results, key=lambda x: abs(x['cohens_d']), reverse=True)

print("\n--- LONG (Top 5) ---")
print(f"{'Feature':<22} {'Cohen d':>10} {'Effect':>10} {'Interpretation'}")
print("-" * 60)
for r in long_sorted[:5]:
    interp = "WIN higher" if r['cohens_d'] > 0 else "LOSS higher"
    print(f"{r['feature']:<22} {r['cohens_d']:>+10.3f} {r['effect']:>10} {interp}")

print("\n--- SHORT (Top 5) ---")
print(f"{'Feature':<22} {'Cohen d':>10} {'Effect':>10} {'Interpretation'}")
print("-" * 60)
for r in short_sorted[:5]:
    interp = "WIN higher" if r['cohens_d'] > 0 else "LOSS higher"
    print(f"{r['feature']:<22} {r['cohens_d']:>+10.3f} {r['effect']:>10} {interp}")

# =============================================================================
# TEST: FILTER BY TOP FEATURE
# =============================================================================
print("\n" + "=" * 70)
print("TESTING DATA-DRIVEN FILTERS")
print("=" * 70)

# Find the feature with largest effect for LONG
top_long_feat = long_sorted[0]['feature']
top_long_d = long_sorted[0]['cohens_d']

wins = train_df[train_df['long_outcome'] == 'WIN']
losses = train_df[train_df['long_outcome'] == 'LOSS']

# Find threshold that separates WIN from LOSS
win_median = wins[top_long_feat].median()
loss_median = losses[top_long_feat].median()

print(f"\nTop feature for LONG: {top_long_feat}")
print(f"  WIN median:  {win_median:.2f}")
print(f"  LOSS median: {loss_median:.2f}")

# If WIN has higher value, filter for high values
if top_long_d > 0:
    threshold = (win_median + loss_median) / 2
    filter_mask = train_df[top_long_feat] >= threshold
    print(f"  Rule: {top_long_feat} >= {threshold:.2f}")
else:
    threshold = (win_median + loss_median) / 2
    filter_mask = train_df[top_long_feat] <= threshold
    print(f"  Rule: {top_long_feat} <= {threshold:.2f}")

# Apply filter and check win rate
filtered_df = train_df[filter_mask]
base_wr = (train_df['long_outcome'] == 'WIN').sum() / (train_df['long_outcome'].isin(['WIN', 'LOSS'])).sum() * 100
filtered_wr = (filtered_df['long_outcome'] == 'WIN').sum() / (filtered_df['long_outcome'].isin(['WIN', 'LOSS'])).sum() * 100

print(f"\n  Base win rate (all bars): {base_wr:.1f}%")
print(f"  Filtered win rate:        {filtered_wr:.1f}%")
print(f"  Improvement:              {filtered_wr - base_wr:+.1f}pp")
print(f"  Bars remaining:           {len(filtered_df):,} ({len(filtered_df)/len(train_df)*100:.1f}%)")

# =============================================================================
# COMBINED FILTER TEST
# =============================================================================
print("\n" + "-" * 70)
print("COMBINED FILTER TEST (Top 3 features)")
print("-" * 70)

# Get top 3 features for LONG
top_features = [r['feature'] for r in long_sorted[:3]]

# Build combined filter
combined_mask = pd.Series([True] * len(train_df), index=train_df.index)

for feat_name in top_features:
    feat_result = next(r for r in long_sorted if r['feature'] == feat_name)
    win_med = wins[feat_name].median()
    loss_med = losses[feat_name].median()
    thresh = (win_med + loss_med) / 2

    if feat_result['cohens_d'] > 0:
        mask = train_df[feat_name] >= thresh
    else:
        mask = train_df[feat_name] <= thresh

    combined_mask = combined_mask & mask

combined_filtered = train_df[combined_mask]
combined_wr = (combined_filtered['long_outcome'] == 'WIN').sum() / (combined_filtered['long_outcome'].isin(['WIN', 'LOSS'])).sum() * 100

print(f"\nFeatures used: {', '.join(top_features)}")
print(f"Base win rate:     {base_wr:.1f}%")
print(f"Combined filter:   {combined_wr:.1f}%")
print(f"Improvement:       {combined_wr - base_wr:+.1f}pp")
print(f"Bars remaining:    {len(combined_filtered):,} ({len(combined_filtered)/len(train_df)*100:.1f}%)")

# Break-even check
breakeven = stop_pct / (target_pct + stop_pct) * 100
print(f"\nBreak-even WR:     {breakeven:.1f}%")
print(f"Gap to break-even: {combined_wr - breakeven:+.1f}pp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Check if any effect size is meaningful
max_effect = max(abs(r['cohens_d']) for r in long_results)

print(f"""
FINDINGS:
- Largest effect size (Cohen's d): {max_effect:.3f}
- Effect size interpretation:
    - < 0.2 = TINY (negligible practical significance)
    - 0.2-0.5 = SMALL
    - 0.5-0.8 = MEDIUM
    - > 0.8 = LARGE

CONCLUSION:
""")

if max_effect < 0.2:
    print("  NO features meaningfully predict WIN vs LOSS.")
    print("  Noise filtering is NOT the solution.")
    print("  The problem is lack of DIRECTIONAL EDGE.")
elif max_effect < 0.5:
    print("  Features have SMALL predictive power.")
    print("  Noise filtering may help marginally.")
    print("  Directional prediction is still needed.")
else:
    print("  Features have MEDIUM-LARGE predictive power!")
    print("  Data-driven noise filtering could work.")
    print("  Test the filters on held-out data.")
