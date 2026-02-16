# L2-001: Feature Re-Validation on 15-Minute Data

**Date:** 2026-02-14
**Timeframe:** 15-min BTCUSDT
**Train:** 2020-2023 (140,105 bars) | **OOS:** 2024-2025 (69,942 bars)

## Objective
Re-validate ALL features from WHEN analysis on 15-min bars. Determine which features predict MFE (magnitude) and direction on this timeframe.

## Methodology

### Stage A: Raw Opportunity (MFE prediction)
- 26 magnitude features tested
- 3 horizons: 10, 15, 20 bars (150/225/300 min)
- Target: 25bp minimum profitable move
- Metric: Q4/Q1 MFE median ratio on TRAIN, validated on OOS
- **Combinations tested: 78** (26 features × 3 horizons)

### Stage B: V1.3.2 PnL per Feature Quartile
- 15 features tested with Q1 vs Q4 PnL comparison
- Run on both TRAIN and OOS
- **Combinations tested: 60** (15 features × 2 quartiles × 2 datasets)

### V1.3.2 PnL Results
- 48 features tested against V1.3.2 trade-level performance
- Per-quartile PnL analysis
- **Combinations tested: ~94 rows** (48 features × quartiles)

## Total Combinations Tested
- Stage A: 78
- Stage B: 60
- V132 PnL: 94
- **Total: ~232 test combinations**

## Key Results (Stage A — Top Magnitude Features, Horizon 10)

| Feature | TRAIN Q4/Q1 MFE | OOS Q4/Q1 MFE | Status |
|---------|-----------------|---------------|--------|
| atr_pct | 3.76x | 2.77x | OK |
| keltner_width | 3.76x | 2.77x | OK |
| bar_range_avg_10 | 3.77x | 2.78x | OK |
| atr7_pct | 3.76x | 2.80x | OK |
| range_bps | 3.37x | 2.58x | OK |
| ema_separation | 2.21x | 1.48x | OK |

## Key Findings

1. **All volatility features pass** — ATR, range, keltner width all predict MFE consistently TRAIN→OOS
2. **26 features tested, majority validate** — real signal exists on 15-min timeframe
3. **ATR family dominates** — atr_pct, atr7_pct, keltner_width, bar_range_avg_10 all top tier
4. **EMA separation validates** — 2.21x TRAIN, 1.48x OOS (weaker but consistent)
5. **Hour/volume/weekend weakest** — minimal MFE separation power

## Output → L2-001b
Stage A validated features fed directly into L2-001b for strength ranking and deduplication.

## Files
- `L2_001_feature_revalidation.py` — Main script (also computes all 51 features, reused by L2-001b and L2-002)
- `results_stage_a_mfe.csv` — 78 rows: 26 features × 3 horizons, TRAIN + OOS MFE
- `results_stage_b_v132_pnl.csv` — 60 rows: 15 features × Q1/Q4 × TRAIN+OOS (dataset column)
- `results_v132_quartile_pnl.csv` — 94 rows: 48 features × quartile PnL vs V1.3.2
