"""
Debug script to test ALL 4 noise types and their impact on expansion rates.

Run: .venv/Scripts/python.exe debug_all_noise_types.py

4 NOISE TYPES:
1. MAGNITUDE - Move too small to profit (max_move < MWNM)
2. STRUCTURAL - Direction unstable/choppy (oscillating EMA, RSI near 50)
3. LIQUIDITY - Low volume/fake moves (volume below threshold)
4. CHAOS - Uncontrollable spikes (ATR explosion)

KEY QUESTION: Which noise type(s) explain why expansion rate is only 20%
when 50% of bars theoretically could reach target?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZON = 60  # 1 hour
MWNM_BPS = 15  # Minimum Worthwhile Net Move
MWNM_PCT = MWNM_BPS / 10000
INVALIDATION_RATIO = 0.5

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

# Noise thresholds
STRUCTURAL_EMA_SLOPE_THRESHOLD = 0.5  # |z-score| < 0.5 = flat
STRUCTURAL_RSI_LOW = 45
STRUCTURAL_RSI_HIGH = 55
LIQUIDITY_VOLUME_PERCENTILE = 25  # Bottom 25% volume = noise
CHAOS_ATR_PERCENTILE = 95  # Top 5% ATR = chaos

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("COMPREHENSIVE NOISE ANALYSIS - ALL 4 TYPES")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    exit(1)

ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# We need features for structural/liquidity noise detection
# Let's compute them fresh to ensure consistency

print("\nComputing features for noise detection...")

# =============================================================================
# COMPUTE FEATURES FOR NOISE DETECTION
# =============================================================================

df = ohlcv.copy()

# EMA slopes
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
df['ema50_slope'] = df['ema50'].diff(5) / df['ema50'].shift(5)
df['ema200_slope'] = df['ema200'].diff(5) / df['ema200'].shift(5)

# Z-score the slopes
df['ema50_slope_z'] = (df['ema50_slope'] - df['ema50_slope'].rolling(100).mean()) / df['ema50_slope'].rolling(100).std()
df['ema200_slope_z'] = (df['ema200_slope'] - df['ema200_slope'].rolling(100).mean()) / df['ema200_slope'].rolling(100).std()

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))

# Volume percentile (rolling)
df['volume_pct'] = df['volume'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr'] = df['tr'].rolling(14).mean()
df['atr_pct'] = df['atr'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

# Direction changes (count of sign changes in last N bars)
df['return'] = df['close'].pct_change()
df['direction'] = np.sign(df['return'])
df['direction_change'] = (df['direction'] != df['direction'].shift(1)).astype(int)
df['direction_changes_10'] = df['direction_change'].rolling(10).sum()

# Drop NaN rows from feature computation
df = df.dropna()

print(f"After feature computation: {len(df):,} candles")

# Split train/test
train_df = df[df.index <= TRAIN_END].copy()
test_df = df[df.index >= TEST_START].copy()
print(f"TRAIN: {len(train_df):,} | TEST: {len(test_df):,}")

# =============================================================================
# NOISE LABELING FUNCTIONS
# =============================================================================

def compute_max_moves(ohlcv_df, horizon):
    """Compute max move in either direction for each bar."""
    close = ohlcv_df['close'].values
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    n = len(ohlcv_df)

    max_move = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = close[i]
        future_high = np.max(high[i+1:i+1+horizon])
        future_low = np.min(low[i+1:i+1+horizon])
        max_up = (future_high - entry) / entry
        max_down = (entry - future_low) / entry
        max_move[i] = max(max_up, max_down)

    return max_move


def compute_expansion_labels(ohlcv_df, horizon, target_pct, stop_pct):
    """Compute path-dependent expansion labels."""
    close = ohlcv_df['close'].values
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    n = len(ohlcv_df)

    long_expansion = np.zeros(n)
    short_expansion = np.zeros(n)

    for i in range(n - horizon):
        entry = close[i]
        target_up = entry * (1 + target_pct)
        stop_up = entry * (1 - stop_pct)
        target_down = entry * (1 - target_pct)
        stop_down = entry * (1 + stop_pct)

        # Long
        for j in range(i+1, i+1+horizon):
            if low[j] <= stop_up:
                break
            if high[j] >= target_up:
                long_expansion[i] = 1
                break

        # Short
        for j in range(i+1, i+1+horizon):
            if high[j] >= stop_down:
                break
            if low[j] <= target_down:
                short_expansion[i] = 1
                break

    return long_expansion, short_expansion


def label_all_noise_types(df, max_move_arr):
    """Label each bar with all 4 noise types."""
    n = len(df)

    # Type 1: MAGNITUDE NOISE (move too small)
    noise_magnitude = max_move_arr < MWNM_PCT

    # Type 2: STRUCTURAL NOISE (choppy/oscillating)
    # - EMA slopes near zero
    # - RSI in neutral zone (45-55)
    # - Many direction changes
    ema50_flat = np.abs(df['ema50_slope_z'].values) < STRUCTURAL_EMA_SLOPE_THRESHOLD
    ema200_flat = np.abs(df['ema200_slope_z'].values) < STRUCTURAL_EMA_SLOPE_THRESHOLD
    rsi_neutral = (df['rsi'].values >= STRUCTURAL_RSI_LOW) & (df['rsi'].values <= STRUCTURAL_RSI_HIGH)
    high_chop = df['direction_changes_10'].values >= 6  # 6+ direction changes in 10 bars

    # Structural noise if EMA flat AND (RSI neutral OR high chop)
    noise_structural = ema50_flat & ema200_flat & (rsi_neutral | high_chop)

    # Type 3: LIQUIDITY NOISE (low volume)
    noise_liquidity = df['volume_pct'].values < LIQUIDITY_VOLUME_PERCENTILE

    # Type 4: CHAOS NOISE (ATR explosion)
    noise_chaos = df['atr_pct'].values > CHAOS_ATR_PERCENTILE

    return {
        'magnitude': noise_magnitude,
        'structural': noise_structural,
        'liquidity': noise_liquidity,
        'chaos': noise_chaos,
    }


# =============================================================================
# ANALYZE TRAIN DATA
# =============================================================================
print("\n" + "=" * 70)
print(f"ANALYSIS ON TRAIN DATA (H={HORIZON})")
print("=" * 70)

# Compute max moves
print("\nComputing max moves...")
train_max_move = compute_max_moves(train_df, HORIZON)

# Valid indices (exclude last H bars)
valid_idx = ~np.isnan(train_max_move)
n_valid = np.sum(valid_idx)
print(f"Valid bars for analysis: {n_valid:,}")

# Compute thresholds from train
train_max_move_valid = train_max_move[valid_idx]
median_move = np.percentile(train_max_move_valid, 50)
target_pct = median_move
stop_pct = target_pct * INVALIDATION_RATIO

print(f"\nThresholds (from TRAIN):")
print(f"  Target: {target_pct * 10000:.1f} bps")
print(f"  Stop: {stop_pct * 10000:.1f} bps")
print(f"  R:R: 2:1")

# Compute expansion labels
print("\nComputing expansion labels...")
train_long_exp, train_short_exp = compute_expansion_labels(train_df, HORIZON, target_pct, stop_pct)

# Label all noise types
print("Labeling noise types...")
noise_labels = label_all_noise_types(train_df, train_max_move)

# =============================================================================
# NOISE STATISTICS
# =============================================================================
print("\n" + "-" * 70)
print("NOISE TYPE STATISTICS")
print("-" * 70)

# Get valid portion only
noise_magnitude_valid = noise_labels['magnitude'][valid_idx]
noise_structural_valid = noise_labels['structural'][valid_idx]
noise_liquidity_valid = noise_labels['liquidity'][valid_idx]
noise_chaos_valid = noise_labels['chaos'][valid_idx]

print(f"\n{'Noise Type':<20} {'Count':>12} {'Percentage':>12}")
print("-" * 46)
print(f"{'1. MAGNITUDE':<20} {np.sum(noise_magnitude_valid):>12,} {np.mean(noise_magnitude_valid)*100:>11.1f}%")
print(f"{'2. STRUCTURAL':<20} {np.sum(noise_structural_valid):>12,} {np.mean(noise_structural_valid)*100:>11.1f}%")
print(f"{'3. LIQUIDITY':<20} {np.sum(noise_liquidity_valid):>12,} {np.mean(noise_liquidity_valid)*100:>11.1f}%")
print(f"{'4. CHAOS':<20} {np.sum(noise_chaos_valid):>12,} {np.mean(noise_chaos_valid)*100:>11.1f}%")

# Any noise
any_noise = noise_magnitude_valid | noise_structural_valid | noise_liquidity_valid | noise_chaos_valid
print(f"\n{'ANY NOISE':<20} {np.sum(any_noise):>12,} {np.mean(any_noise)*100:>11.1f}%")
print(f"{'TRADEABLE (no noise)':<20} {np.sum(~any_noise):>12,} {np.mean(~any_noise)*100:>11.1f}%")

# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================
print("\n" + "-" * 70)
print("NOISE OVERLAP ANALYSIS")
print("-" * 70)

# Count bars with exactly 1, 2, 3, 4 noise types
noise_count = (noise_magnitude_valid.astype(int) +
               noise_structural_valid.astype(int) +
               noise_liquidity_valid.astype(int) +
               noise_chaos_valid.astype(int))

print(f"\n{'Noise Count':<20} {'Bars':>12} {'Percentage':>12}")
print("-" * 46)
for i in range(5):
    count = np.sum(noise_count == i)
    pct = count / n_valid * 100
    label = "Clean (tradeable)" if i == 0 else f"{i} noise type(s)"
    print(f"{label:<20} {count:>12,} {pct:>11.1f}%")

# =============================================================================
# EXPANSION RATES BY NOISE TYPE
# =============================================================================
print("\n" + "-" * 70)
print("EXPANSION RATES BY NOISE FILTER")
print("-" * 70)

train_long_valid = train_long_exp[valid_idx]
train_short_valid = train_short_exp[valid_idx]

base_long = np.mean(train_long_valid) * 100
base_short = np.mean(train_short_valid) * 100

print(f"\n{'Filter':<35} {'LONG':>10} {'SHORT':>10} {'Samples':>12}")
print("-" * 70)
print(f"{'ALL bars (no filter)':<35} {base_long:>9.1f}% {base_short:>9.1f}% {n_valid:>12,}")

# Filter by each noise type
filters = [
    ('Remove MAGNITUDE noise', ~noise_magnitude_valid),
    ('Remove STRUCTURAL noise', ~noise_structural_valid),
    ('Remove LIQUIDITY noise', ~noise_liquidity_valid),
    ('Remove CHAOS noise', ~noise_chaos_valid),
    ('Remove ALL noise', ~any_noise),
]

for name, mask in filters:
    if np.sum(mask) > 0:
        long_rate = np.mean(train_long_valid[mask]) * 100
        short_rate = np.mean(train_short_valid[mask]) * 100
        n_samples = np.sum(mask)
        print(f"{name:<35} {long_rate:>9.1f}% {short_rate:>9.1f}% {n_samples:>12,}")

# Combined filters (additive)
print("\n" + "-" * 70)
print("PROGRESSIVE NOISE REMOVAL")
print("-" * 70)

progressive_mask = np.ones(n_valid, dtype=bool)
print(f"\n{'Step':<45} {'LONG':>10} {'SHORT':>10} {'Remaining':>12}")
print("-" * 80)

long_rate = np.mean(train_long_valid) * 100
short_rate = np.mean(train_short_valid) * 100
print(f"{'0. Start (all bars)':<45} {long_rate:>9.1f}% {short_rate:>9.1f}% {np.sum(progressive_mask):>12,}")

progressive_mask = progressive_mask & ~noise_magnitude_valid
long_rate = np.mean(train_long_valid[progressive_mask]) * 100
short_rate = np.mean(train_short_valid[progressive_mask]) * 100
print(f"{'1. Remove MAGNITUDE':<45} {long_rate:>9.1f}% {short_rate:>9.1f}% {np.sum(progressive_mask):>12,}")

progressive_mask = progressive_mask & ~noise_structural_valid
long_rate = np.mean(train_long_valid[progressive_mask]) * 100
short_rate = np.mean(train_short_valid[progressive_mask]) * 100
print(f"{'2. Remove MAGNITUDE + STRUCTURAL':<45} {long_rate:>9.1f}% {short_rate:>9.1f}% {np.sum(progressive_mask):>12,}")

progressive_mask = progressive_mask & ~noise_liquidity_valid
long_rate = np.mean(train_long_valid[progressive_mask]) * 100
short_rate = np.mean(train_short_valid[progressive_mask]) * 100
print(f"{'3. Remove MAGNITUDE + STRUCTURAL + LIQUIDITY':<45} {long_rate:>9.1f}% {short_rate:>9.1f}% {np.sum(progressive_mask):>12,}")

progressive_mask = progressive_mask & ~noise_chaos_valid
long_rate = np.mean(train_long_valid[progressive_mask]) * 100
short_rate = np.mean(train_short_valid[progressive_mask]) * 100
print(f"{'4. Remove ALL NOISE':<45} {long_rate:>9.1f}% {short_rate:>9.1f}% {np.sum(progressive_mask):>12,}")

# =============================================================================
# PROFITABILITY CHECK
# =============================================================================
print("\n" + "-" * 70)
print("PROFITABILITY ANALYSIS")
print("-" * 70)

breakeven = stop_pct / (target_pct + stop_pct) * 100
print(f"\nBreak-even WR: {breakeven:.1f}%")

final_long = np.mean(train_long_valid[~any_noise]) * 100
final_short = np.mean(train_short_valid[~any_noise]) * 100

print(f"\nAfter removing ALL noise:")
print(f"  LONG expansion:  {final_long:.1f}% (gap: {final_long - breakeven:+.1f}pp)")
print(f"  SHORT expansion: {final_short:.1f}% (gap: {final_short - breakeven:+.1f}pp)")

if final_long >= breakeven:
    print(f"\n  >>> LONG CROSSES BREAK-EVEN!")
if final_short >= breakeven:
    print(f"\n  >>> SHORT CROSSES BREAK-EVEN!")

# =============================================================================
# TEST DATA VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("TEST DATA VALIDATION")
print("=" * 70)

# Compute on test
test_max_move = compute_max_moves(test_df, HORIZON)
test_valid_idx = ~np.isnan(test_max_move)
test_noise = label_all_noise_types(test_df, test_max_move)
test_long_exp, test_short_exp = compute_expansion_labels(test_df, HORIZON, target_pct, stop_pct)

# Valid portion
test_magnitude = test_noise['magnitude'][test_valid_idx]
test_structural = test_noise['structural'][test_valid_idx]
test_liquidity = test_noise['liquidity'][test_valid_idx]
test_chaos = test_noise['chaos'][test_valid_idx]
test_any_noise = test_magnitude | test_structural | test_liquidity | test_chaos

test_long_valid = test_long_exp[test_valid_idx]
test_short_valid = test_short_exp[test_valid_idx]

print(f"\nTEST noise breakdown:")
print(f"  MAGNITUDE: {np.mean(test_magnitude)*100:.1f}%")
print(f"  STRUCTURAL: {np.mean(test_structural)*100:.1f}%")
print(f"  LIQUIDITY: {np.mean(test_liquidity)*100:.1f}%")
print(f"  CHAOS: {np.mean(test_chaos)*100:.1f}%")
print(f"  ANY NOISE: {np.mean(test_any_noise)*100:.1f}%")

test_base_long = np.mean(test_long_valid) * 100
test_base_short = np.mean(test_short_valid) * 100
test_clean_long = np.mean(test_long_valid[~test_any_noise]) * 100
test_clean_short = np.mean(test_short_valid[~test_any_noise]) * 100

print(f"\nTEST expansion rates:")
print(f"  ALL bars:     LONG={test_base_long:.1f}%, SHORT={test_base_short:.1f}%")
print(f"  TRADEABLE:    LONG={test_clean_long:.1f}%, SHORT={test_clean_short:.1f}%")
print(f"  Improvement:  LONG={test_clean_long - test_base_long:+.1f}pp, SHORT={test_clean_short - test_base_short:+.1f}pp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
NOISE BREAKDOWN (TRAIN):
  - MAGNITUDE (move too small):    {np.mean(noise_magnitude_valid)*100:.1f}%
  - STRUCTURAL (choppy/flat):      {np.mean(noise_structural_valid)*100:.1f}%
  - LIQUIDITY (low volume):        {np.mean(noise_liquidity_valid)*100:.1f}%
  - CHAOS (volatility spike):      {np.mean(noise_chaos_valid)*100:.1f}%
  - ANY NOISE:                     {np.mean(any_noise)*100:.1f}%

EXPANSION RATE:
  - ALL bars:       {base_long:.1f}% / {base_short:.1f}% (LONG/SHORT)
  - TRADEABLE only: {final_long:.1f}% / {final_short:.1f}% (LONG/SHORT)
  - Break-even:     {breakeven:.1f}%
  - Gap remaining:  {final_long - breakeven:+.1f}pp / {final_short - breakeven:+.1f}pp

INSIGHT:
  If gap is still large after noise removal, the problem is NOT noise.
  The problem is lack of DIRECTIONAL EDGE - we're entering randomly.
""")
