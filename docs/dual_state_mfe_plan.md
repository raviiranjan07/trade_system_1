# Dual-State MFE Architecture

## Problem Statement

### Current Approach (Flawed)
The current outcome labeling stores only the "winning" direction for each state:

```
| timestamp | direction | mfe | mae |
|-----------|-----------|-----|-----|
| T1        | 1 (long)  | 0.3%| 0.1%|
| T2        | 1 (long)  | 0.2%| 0.05%|
| T3        | -1 (short)| 0.15%| 0.08%|
```

**Issues:**
1. Information loss - we don't know what the OTHER direction would have done
2. Market bias - in bull markets, 90%+ rows have direction=1
3. When querying similar states, shorts are statistically drowned out
4. System can't fairly evaluate both directions for a given state

### Example of the Problem
```
Query: Find K=100 similar states for current market state
Result: 95 neighbors have direction=long, 5 have direction=short

avg_long_mfe = calculated from 95 samples (biased high)
avg_short_mfe = calculated from 5 samples (biased low, statistically weak)

Decision: Always picks long (not because long is better, but because data is skewed)
```

---

## Solution: Dual-State MFE

### Core Concept
Store MFE/MAE for BOTH directions at every timestamp, regardless of which was "better":

```
| timestamp | mfe_long | mae_long | mfe_short | mae_short |
|-----------|----------|----------|-----------|-----------|
| T1        | 0.30%    | 0.10%    | 0.08%     | 0.30%     |
| T2        | 0.20%    | 0.05%    | 0.12%     | 0.20%     |
| T3        | 0.05%    | 0.15%    | 0.25%     | 0.05%     |
```

### Benefits
1. **No information loss** - complete picture for both directions
2. **Fair comparison** - both directions evaluated from same K samples
3. **Market agnostic** - system evaluates state, not market bias
4. **Better decision quality** - can properly compare long vs short potential

---

## Data Structure Changes

### Outcome File Schema

**Old Schema:**
```python
columns = [
    'direction',      # 1 (long) or -1 (short) - winner only
    'mfe',            # MFE of winning direction
    'mae',            # MAE of winning direction
    'pnl',            # PnL of winning direction
    'exit_reason',    # How trade closed
]
```

**New Schema:**
```python
columns = [
    # Long metrics
    'mfe_long',       # Max favorable excursion if went long (max price UP)
    'mae_long',       # Max adverse excursion if went long (max price DOWN before UP)
    'pnl_long',       # Final PnL if held long for full horizon

    # Short metrics
    'mfe_short',      # Max favorable excursion if went short (max price DOWN)
    'mae_short',      # Max adverse excursion if went short (max price UP before DOWN)
    'pnl_short',      # Final PnL if held short for full horizon

    # Optional: derived fields
    'best_direction', # 1 if mfe_long > mfe_short, -1 otherwise (for reference)
    'mfe_ratio',      # mfe_long / mfe_short (for analysis)
]
```

### File Naming
```
outcomes_h3_dual.parquet    # H=3 horizon with dual MFE
outcomes_h5_dual.parquet    # H=5 horizon with dual MFE
outcomes_h10_dual.parquet   # H=10 horizon with dual MFE
```

---

## MFE/MAE Calculation Logic

### For Long Position
```python
def compute_long_metrics(ohlcv_df, entry_idx, horizon):
    """
    MFE_long = max price INCREASE from entry within horizon
    MAE_long = max price DECREASE from entry within horizon
    """
    entry_price = ohlcv_df.iloc[entry_idx]['close']
    future_bars = ohlcv_df.iloc[entry_idx + 1 : entry_idx + 1 + horizon]

    # MFE: How much did price go UP? (favorable for long)
    max_high = future_bars['high'].max()
    mfe_long = (max_high - entry_price) / entry_price

    # MAE: How much did price go DOWN? (adverse for long)
    min_low = future_bars['low'].min()
    mae_long = (entry_price - min_low) / entry_price

    # PnL at horizon end
    exit_price = future_bars.iloc[-1]['close']
    pnl_long = (exit_price - entry_price) / entry_price

    return mfe_long, mae_long, pnl_long
```

### For Short Position
```python
def compute_short_metrics(ohlcv_df, entry_idx, horizon):
    """
    MFE_short = max price DECREASE from entry within horizon
    MAE_short = max price INCREASE from entry within horizon
    """
    entry_price = ohlcv_df.iloc[entry_idx]['close']
    future_bars = ohlcv_df.iloc[entry_idx + 1 : entry_idx + 1 + horizon]

    # MFE: How much did price go DOWN? (favorable for short)
    min_low = future_bars['low'].min()
    mfe_short = (entry_price - min_low) / entry_price

    # MAE: How much did price go UP? (adverse for short)
    max_high = future_bars['high'].max()
    mae_short = (max_high - entry_price) / entry_price

    # PnL at horizon end
    exit_price = future_bars.iloc[-1]['close']
    pnl_short = (entry_price - exit_price) / entry_price

    return mfe_short, mae_short, pnl_short
```

---

## Decision Engine Changes

### Current Logic (Flawed)
```python
def make_decision(neighbors):
    # Biased by direction distribution in data
    long_neighbors = neighbors[neighbors['direction'] == 1]
    short_neighbors = neighbors[neighbors['direction'] == -1]

    avg_long_mfe = long_neighbors['mfe'].mean()  # From N samples
    avg_short_mfe = short_neighbors['mfe'].mean()  # From M samples (M << N in bull market)

    return 'long' if avg_long_mfe > avg_short_mfe else 'short'
```

### New Logic (Fair)
```python
def make_decision(neighbors, min_mfe=0.002, min_gap_ratio=1.2):
    """
    Evaluate both directions from ALL K neighbors.

    Args:
        neighbors: K similar historical states
        min_mfe: Minimum MFE threshold to consider a trade
        min_gap_ratio: Winner must be X times better than loser
    """
    # All K neighbors have BOTH metrics
    avg_mfe_long = neighbors['mfe_long'].mean()
    avg_mfe_short = neighbors['mfe_short'].mean()
    avg_mae_long = neighbors['mae_long'].mean()
    avg_mae_short = neighbors['mae_short'].mean()

    # Risk-adjusted scores (MFE/MAE ratio)
    score_long = avg_mfe_long / avg_mae_long if avg_mae_long > 0 else 0
    score_short = avg_mfe_short / avg_mae_short if avg_mae_short > 0 else 0

    # Decision logic
    if avg_mfe_long < min_mfe and avg_mfe_short < min_mfe:
        return None  # No trade - both directions weak

    if avg_mfe_long >= min_mfe and avg_mfe_short < min_mfe:
        return 'long'  # Only long meets threshold

    if avg_mfe_short >= min_mfe and avg_mfe_long < min_mfe:
        return 'short'  # Only short meets threshold

    # Both meet threshold - pick better one with minimum gap
    if score_long > score_short * min_gap_ratio:
        return 'long'
    elif score_short > score_long * min_gap_ratio:
        return 'short'
    else:
        return None  # Too close - ambiguous state, skip
```

---

## Implementation Plan

### Phase 1: Outcome Labeler Update
**File:** `src/trade_system/outcomes/outcome_labeler.py`

1. Add `compute_long_metrics()` function
2. Add `compute_short_metrics()` function
3. Modify `label_outcomes()` to compute and store both directions
4. Update output schema to include all 6 columns

### Phase 2: Generate New Outcome Files
```bash
# Generate dual-state outcomes for each horizon
python -m trade_system.outcomes.run_outcome_labeling --horizon 3 --dual
python -m trade_system.outcomes.run_outcome_labeling --horizon 5 --dual
python -m trade_system.outcomes.run_outcome_labeling --horizon 10 --dual
```

### Phase 3: Decision Engine Update
**File:** `src/trade_system/decision/decision_engine.py`

1. Update neighbor aggregation to use both `mfe_long` and `mfe_short`
2. Implement new decision logic with:
   - Separate MFE thresholds per direction (optional)
   - Minimum gap ratio for ambiguous states
   - Risk-adjusted scoring (MFE/MAE)

### Phase 4: Backtester Update
**File:** `src/trade_system/backtest/backtester.py`

1. Update to work with new outcome schema
2. Track which direction was chosen and why
3. Add metrics for direction selection accuracy

### Phase 5: Grid Search Update
**File:** `experiments/scalping/scripts/run_scalping_grid_search_batch.py`

1. Update to use dual-state outcomes
2. Add `min_gap_ratio` as searchable parameter
3. Track long/short selection statistics

---

## Configuration

### New Config Parameters
```yaml
decision:
  # Existing
  min_mfe: 0.002
  max_distance: 4.0
  k: 100

  # New for dual-state
  min_mfe_long: 0.002      # Can be different per direction
  min_mfe_short: 0.002
  min_gap_ratio: 1.2       # Winner must be 20% better
  use_risk_adjusted: true  # Use MFE/MAE ratio instead of raw MFE
  skip_ambiguous: true     # Skip trades when directions are too close
```

---

## Expected Benefits

### Before (Current System)
```
H=3 Batch 1 Results:
- 4 profitable (ALL long-only)
- Shorts fail because they're statistically drowned out
- Can't properly evaluate short potential
```

### After (Dual-State)
```
Expected:
- Fair evaluation of both directions at each state
- Short signals when SHORT is genuinely better
- Long signals when LONG is genuinely better
- No trade when both are weak OR too similar
- True market-regime agnostic trading
```

---

## Risk Considerations

### 1. Ambiguous States
**Risk:** What if mfe_long ≈ mfe_short frequently?
**Mitigation:** `min_gap_ratio` parameter + `skip_ambiguous` flag

### 2. Overfitting to One Direction
**Risk:** If one direction is genuinely better, we might miss edge
**Mitigation:** Track direction selection stats, allow direction-specific thresholds

### 3. Increased Complexity
**Risk:** More parameters to tune
**Mitigation:** Start with symmetric thresholds, tune later based on results

---

## Success Metrics

1. **Fair Direction Distribution**: Long/short ratio reflects actual opportunity, not data bias
2. **Improved Short Performance**: Shorts taken only when genuinely favorable
3. **Higher Win Rate**: Better filtering of ambiguous states
4. **Consistent Performance**: Works across bull/bear/ranging markets

---

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1 | Create documentation | ✅ Complete |
| 2 | Update outcome labeler | Pending |
| 3 | Generate dual-state outcomes | Pending |
| 4 | Update decision engine | Pending |
| 5 | Update backtester | Pending |
| 6 | Run validation tests | Pending |
| 7 | Grid search with new system | Pending |
