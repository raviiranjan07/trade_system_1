# L2-001b: Feature Strength Ranking

**Date:** 2026-02-14
**Timeframe:** 15-min BTCUSDT
**Train:** 2020-2023 (140,105 bars) | **OOS:** 2024-2025 (69,942 bars)
**Horizon:** 10 bars (150 min) | **Target:** 25bp

## Goal
Rank all L2-001 validated features by STRENGTH (not just pass/fail).
- Magnitude features: ranked by Q4/Q1 MFE ratio (separation power)
- Direction features: ranked by Q4-Q1 directional accuracy spread + deduplicated via correlation

## Magnitude Features — Ranked by Q4/Q1 Best MFE Ratio

| Rank | Feature | TRAIN Ratio | OOS Ratio | Tier |
|------|---------|-------------|-----------|------|
| 1 | atr_pct | 4.01x | 2.77x | TOP |
| 2 | range_bps | 3.37x | 2.58x | TOP |
| 3 | dist_from_high20_pct | 2.24x | 1.67x | MID |
| 4 | ema_separation | 2.21x | 1.48x | MID |
| 5 | body_bps | 2.16x | 1.74x | MID |
| 6 | atr_percentile | 1.70x | 2.23x | MID |
| 7 | hour_utc | 1.22x | 1.17x | WEAK |
| 8 | volume_ratio | 1.20x | 1.25x | WEAK |

### Magnitude Details (TRAIN)

| Feature | Q1 Best MFE | Q4 Best MFE | Q4-Q1 Diff |
|---------|-------------|-------------|------------|
| atr_pct | 42.9 bp | 171.7 bp | +128.8 bp |
| range_bps | 48.0 bp | 161.7 bp | +113.6 bp |
| dist_from_high20_pct | 64.0 bp | 143.6 bp | +79.6 bp |
| ema_separation | 63.3 bp | 139.9 bp | +76.6 bp |
| body_bps | 65.1 bp | 140.9 bp | +75.8 bp |
| atr_percentile | 71.0 bp | 120.9 bp | +49.9 bp |
| hour_utc | 77.0 bp | 93.9 bp | +16.8 bp |
| volume_ratio | 85.7 bp | 102.7 bp | +17.0 bp |

**ATR is the #1 gate** — Q4 bars have 4x the MFE of Q1 bars on TRAIN (2.77x on OOS).

## Direction Features — Ranked by |Q4-Q1| Accuracy Spread

19 features tested, all 19/19 consistent between TRAIN and OOS.

| Rank | Feature | TRAIN Spread | OOS Spread | Direction |
|------|---------|-------------|------------|-----------|
| 1 | ema9_dist_pct | -8.4pp | -4.4pp | BEARISH |
| 2 | rsi7 | -8.4pp | -5.4pp | BEARISH |
| 3 | roc5 | -7.8pp | -4.4pp | BEARISH |
| 4 | range_position | -7.6pp | -4.4pp | BEARISH |
| 5 | rsi | -7.5pp | -4.4pp | BEARISH |
| 6 | bb_position | -7.2pp | -4.3pp | BEARISH |
| 7 | ema20_dist_pct | -7.1pp | -3.4pp | BEARISH |
| 8 | momentum5 | -6.7pp | -4.7pp | BEARISH |
| 9 | rsi21 | -6.5pp | -4.0pp | BEARISH |
| 10 | roc10 | -6.4pp | -3.1pp | BEARISH |
| 11 | momentum10 | -6.1pp | -3.3pp | BEARISH |
| 12 | roc20 | -5.1pp | -2.5pp | BEARISH |
| 13 | ema50_dist_pct | -5.1pp | -3.4pp | BEARISH |
| 14 | ema20_slope | -4.7pp | -2.4pp | BEARISH |
| 15 | ema100_dist_pct | -3.7pp | -2.7pp | BEARISH |
| 16 | ema50_slope | -3.0pp | -1.9pp | BEARISH |
| 17 | ema200_dist_pct | -2.1pp | -2.2pp | BEARISH |
| 18 | sma200_dist_pct | -1.8pp | -2.4pp | BEARISH |
| 19 | ema200_slope | -1.1pp | -1.8pp | BEARISH |

## Deduplication — 19 Features to 8 Distinct Groups

Pairwise Pearson correlation, threshold >90% = same signal.

| Group | Representative | Spread | Group Members |
|-------|---------------|--------|---------------|
| 1 | ema9_dist_pct | -8.4pp | ema20/50/100/200_dist_pct, sma200_dist_pct, ema20/50/200_slope |
| 2 | rsi7 | -8.4pp | rsi, rsi21, bb_position |
| 3 | roc5 | -7.8pp | unique |
| 4 | range_position | -7.6pp | unique |
| 5 | momentum5 | -6.7pp | unique |
| 6 | roc10 | -6.4pp | unique |
| 7 | momentum10 | -6.1pp | unique |
| 8 | roc20 | -5.1pp | unique |

### Key Correlations
- RSI family: rsi <-> rsi21 r=0.977, rsi <-> bb_position r=0.935, rsi <-> rsi7 r=0.933
- MA distance chain: ema200_dist <-> ema200_slope r=0.987, ema200_dist <-> sma200_dist r=0.982
- ema50_dist <-> ema50_slope r=0.950, ema50_dist <-> ema100_dist r=0.942

## Key Findings

1. **ALL direction features are BEARISH**: Q4 (high values) predicts DOWN
   - For LONG: use Q1 (low RSI, low momentum, low MA distance)
   - For SHORT: use Q4 (high RSI, high momentum, high MA distance)

2. **100% directional consistency**: All 19 features same direction on TRAIN and OOS

3. **Big MA family**: 9 features collapsed to 1 group (ema9_dist_pct is representative)

4. **RSI family**: 4 features collapsed to 1 group (rsi7 is representative)

5. **Magnitude tiers are clear**: ATR+range (TOP), structure/trend/body (MID), time/volume (WEAK)

6. **atr_percentile interesting**: TRAIN ratio 1.70x but OOS 2.23x (stronger on OOS than TRAIN)

## For L2-002 Priority Order

**Top gates:** atr_pct + range_bps
**Top signals:** ema9_dist_pct + rsi7 + roc5 + range_position

## Files
- `L2_001b_feature_ranking.py` — Main script
- `L2_001b_magnitude_ranking.csv` — Magnitude rankings
- `L2_001b_direction_ranking.csv` — Direction rankings
- `L2_001b_direction_correlation.csv` — Full 19x19 correlation matrix
- `L2_001b_summary.json` — Machine-readable summary
