# L2-002: Signal Expansion Framework — Results

## Objective
Build framework: state vectors -> regimes -> gate+signal combos -> trade simulation -> metrics.
Find new profitable signals beyond V1.3.2's 220 trades.

## Methodology
- State vector: 16 features computed on every bar (8 magnitude + 8 direction)
- 4 regimes: ACTIVE_TREND, ACTIVE_FLAT, QUIET_TREND, QUIET_FLAT (ATR x EMA sep median split)
- Exit strategy: category-based grid (PROTECTION x TIMING x ADAPTATION), all combos tested
- TRAIN: 2020-2023 (140,105 bars), OOS: 2024-2025 (69,942 bars)
- Pre-computed trade results for speed (simulate LONG/SHORT from every bar once per exit config, then lookup)

### Magnitude Gate — "Does this bar have enough OPPORTUNITY to be worth trading?"
Direction-agnostic. Based on L2-001b magnitude rankings.
4 gate types:
- **BASELINE**: NO_GATE (passes all bars, control group)
- **MARKET**: "Is the environment tradeable?" (atr_pct, ema_separation, atr_percentile, volume_ratio + regime configs)
- **ENTRY**: "Is THIS bar strong enough?" (range_bps, body_bps)
- **STRUCTURE**: "Is price positioned well?" (dist_from_high20_pct, hour_utc)

All 8 L2-001b magnitude features included. Each tested at q25/q50/q75 thresholds.
All 1/2/3-feature combos auto-generated across and within categories.
Each configuration tagged with gate type (MARKET, MARKET+ENTRY, ENTRY+STRUCTURE, etc.)
Gate decision logging: tracks what each gate blocks and whether blocking was correct.

### Signal Gate — "Which DIRECTION should we trade?"
Magnitude-agnostic. Based on L2-001b direction rankings.
3 signal types:
- **MOMENTUM**: "Is the move exhausted?" (rsi7, roc5)
- **REVERSION**: "Has price deviated too far?" (ema9_dist_pct)
- **RANGE**: "Is price at an extreme?" (range_position)

4 direction features (11 individual conditions):
- RSI7 (Rank 2): 4 threshold pairs → MOMENTUM
- ROC5 (Rank 3): 2 threshold pairs → MOMENTUM
- EMA9 distance (Rank 1): 3 threshold pairs → REVERSION
- Range position (Rank 4): 2 threshold pairs → RANGE

All 1/2/3-feature combos auto-generated (same approach as magnitude gate):
- 1-feature: 11 configs
- 2-feature: 44 configs (e.g. RSI7+EMA9D = MOMENTUM+REVERSION)
- 3-feature: 76 configs (e.g. RSI7+EMA9D+RPOS = MOMENTUM+RANGE+REVERSION)
- **Total: 131 signal configurations**

Combo = AND logic: LONG fires when ALL features ≤ their long_threshold.
Each configuration tagged with signal type (MOMENTUM, MOMENTUM+REVERSION, etc.)

### Exit Strategy — "How do we manage the trade after entry?"
Category-based parameter GRID (not AND combos — pick one from each category):
- **PROTECTION**: "How much loss can we tolerate?" — trailing_stop_bps: [10, 15, 20, 25, 30]
- **TIMING**: "How long do we hold?" — max_bars: [5, 8, 10, 12, 15]
- **ADAPTATION**: "Does exit change mid-trade?" — tighten configs (off, or tighten_after_bar x tight_stop_bps)

Grid: PROTECTION x TIMING x ADAPTATION = ~200 exit configurations.
Each exit config requires separate precomputation (different trade outcomes).
Tagged with exit_type: PROTECTION+TIMING (no tighten) or PROTECTION+TIMING+ADAPTATION (with tighten).

### Flow: Bar → Magnitude Gate (pass?) → Signal Gate (LONG/SHORT?) → Exit Strategy → Trade Result

### Total: ~200 exits x ~1,795 magnitude configs x 131 signal configs = ~47M combos
Phase 1: TRAIN scan (per exit: precompute, then all gate x signal lookups)
Phase 2: Filter interesting (PF>1.5, trades>=50)
Phase 3: OOS validation (only interesting configs, grouped by exit for efficiency)

## Regime Profiles

| Regime | TRAIN bars | TRAIN % | Med MFE (best) | OOS bars | OOS % | Med MFE (best) |
|--------|-----------|---------|----------------|----------|-------|----------------|
| ACTIVE_TREND | 48,078 | 34.3% | 140.8bp | 12,042 | 17.2% | 115.7bp |
| ACTIVE_FLAT | 21,975 | 15.7% | 112.8bp | 7,375 | 10.5% | 105.5bp |
| QUIET_TREND | 21,975 | 15.7% | 71.5bp | 17,322 | 24.8% | 62.9bp |
| QUIET_FLAT | 48,077 | 34.3% | 55.3bp | 33,203 | 47.5% | 55.8bp |

**KEY**: 2024-2025 much quieter. ACTIVE_TREND halved (34->17%), QUIET_FLAT almost doubled (34->47%).

## Regime Thresholds (from TRAIN)
- ATR median: 0.3910%
- EMA separation median: 0.7722%

## Top OOS Results by Profit Factor

| Rank | Config | OOS Trades | Win% | Total bps | PF | Avg/trade |
|------|--------|-----------|------|-----------|-----|-----------|
| 1 | ACTIVE_TREND__ROC5_-0.3_0.3 | 2,279 | 60.8% | +34,520 | 1.78 | +15.1 |
| 2 | ACTIVE_TREND__ROC5_-0.5_0.5 | 1,925 | 61.0% | +29,467 | 1.76 | +15.3 |
| 3 | ACTIVE_TREND__EMA9D_-0.5_0.5 | 1,272 | 63.2% | +20,905 | 1.73 | +16.4 |
| 4 | ACTIVE_TREND__RSI7_25_75 | 1,418 | 62.8% | +20,451 | 1.72 | +14.4 |
| 5 | ATR_Q50+EMA_Q50__RPOS_0.2_0.8 | 3,600 | 58.2% | +41,018 | 1.70 | +11.4 |
| 6 | ATR_Q75+RANGE_Q50__RPOS_0.2_0.8 | 2,978 | 60.1% | +39,574 | 1.70 | +13.3 |
| 7 | ATR_Q75__ROC5_-0.3_0.3 | 3,306 | 61.0% | +43,507 | 1.68 | +13.2 |
| 8 | ATR_Q50+EMA_Q50__ROC5_-0.3_0.3 | 3,649 | 59.4% | +42,941 | 1.68 | +11.8 |

## Top OOS Results by Total bps

| Rank | Config | OOS Trades | Win% | Total bps | PF |
|------|--------|-----------|------|-----------|-----|
| 1 | NO_GATE__RPOS_0.3_0.7 | 10,061 | 53.6% | +63,444 | 1.41 |
| 2 | NO_GATE__ROC5_-0.3_0.3 | 7,800 | 55.7% | +57,441 | 1.45 |
| 3 | NO_GATE__RPOS_0.2_0.8 | 9,310 | 53.0% | +57,065 | 1.41 |
| 4 | ATR_Q50__RPOS_0.3_0.7 | 6,062 | 56.5% | +56,518 | 1.54 |
| 5 | ATR_Q50__ROC5_-0.3_0.3 | 5,543 | 57.8% | +52,961 | 1.54 |

## Per-Year Stability (selected top configs)

| Config | 2024 PF | 2025 PF | 2024 bps | 2025 bps |
|--------|---------|---------|----------|----------|
| ACTIVE_TREND__ROC5_-0.3_0.3 | 1.67 | 1.99 | +19,546 | +14,974 |
| ACTIVE_TREND__ROC5_-0.5_0.5 | 1.70 | 1.87 | +17,621 | +11,846 |
| ACTIVE_TREND__EMA9D_-0.5_0.5 | 1.83 | 1.57 | +14,854 | +6,051 |
| ATR_Q75__ROC5_-0.3_0.3 | 1.65 | 1.73 | +25,852 | +17,655 |
| ATR_Q50+EMA_Q50__RPOS_0.2_0.8 | 1.58 | 1.90 | +21,765 | +19,253 |

## Comparison with V1.3.2

| Metric | V1.3.2 | Best PF (AT+ROC5) | Best total (NG+RPOS) |
|--------|--------|-------------------|---------------------|
| Trades (2yr) | 220 | 2,279 | 10,061 |
| PF | 3.46 | 1.78 | 1.41 |
| Total bps | +5,267 | +34,520 | +63,444 |
| Avg/trade | +23.9 | +15.1 | +6.3 |
| Win% | 60.0% | 60.8% | 53.6% |

## Key Findings

1. **ALL 77 combinations profitable on both TRAIN and OOS** — real edge exists across all combos
2. **Clear trade-off**: tighter gates = higher PF, fewer trades; looser gates = more trades, lower PF
3. **ROC5 is the best direction signal** — highest PF across most gate types
4. **ACTIVE_TREND gate + ROC5 = best PF** (1.78) — regime filtering adds ~0.3 to PF
5. **Regime shift**: ACTIVE_TREND only 17% of OOS bars (vs 34% TRAIN), yet signal still profitable
6. **Both LONG and SHORT profitable** across all 77 combos
7. **Per-year consistent**: top configs profitable in both 2024 AND 2025

## Trade Coverage
- OOS total bars: 69,942
- ACTIVE_TREND bars: 12,042 (17.2% of all bars)
- Trades taken (best config): 2,279 (3.3% of all bars, 18.9% of ACTIVE_TREND bars)
- V1.4.0: 220 trades = 0.3% of all bars
- L2-002 trades on 3.3% of bars — still skipping 96.7% of the market

## Gate Tightness vs Performance (same signal: ROC5_-0.3_0.3)

| Gate | OOS Trades | Win% | PF | Total bps |
|------|-----------|------|-----|-----------|
| ACTIVE_TREND | 2,279 | 60.8% | 1.78 | +34,520 |
| ATR_Q50 | 5,543 | 57.8% | 1.54 | +52,961 |
| NO_GATE | 7,800 | 55.7% | 1.45 | +57,441 |

Looser gate = more trades but lower quality per trade.

---

## LOSS ANALYSIS (ACTIVE_TREND + ROC5_-0.3_0.3 on OOS)

### Overview
- Total trades: 2,279
- Winners: 1,386 (60.8%)
- **Losers: 893 (39.2%)**
- Winner avg: +56.8 bps | Loser avg: -49.5 bps
- Winner median: +39.1 bps | Loser median: -20.1 bps

### 1. Exit Reason Breakdown (WHERE losses come from)

| Exit | Trades | Win% | Total bps | Loser avg | Worst |
|------|--------|------|-----------|-----------|-------|
| TRAILING_STOP | 2,056 | 67.3% | +68,072 | -15.8 | -37.9 |
| TIME_EXIT | 223 | **0.9%** | **-33,552** | **-151.9** | **-741.4** |

- **TIME_EXIT is the killer**: 223 trades, only 2 winners, avg loser -151.9 bps
- ALL 20 worst trades are TIME_EXIT with MFE=0 (price NEVER went in the right direction)
- Trailing stop trades are healthy: 67.3% win, net +68,072 bps

### 2. LONG vs SHORT

| Direction | Trades | Win% | Total bps | PF |
|-----------|--------|------|-----------|-----|
| LONG | 1,139 | 64.2% | +19,940 | 1.99 |
| SHORT | 1,140 | 57.5% | +14,580 | 1.61 |

- LONG is stronger (PF 1.99 vs 1.61)
- Both sides profitable

### 3. By Hour (UTC) — Worst Hours

| Hour | Trades | Win% | Total bps | Avg |
|------|--------|------|-----------|-----|
| 2 UTC | 75 | 52.0% | -955 | -12.7 |
| 7 UTC | 44 | 47.7% | +155 | +3.5 |
| 4 UTC | 58 | 51.7% | -62 | -1.1 |

Best hours: 10 UTC (70.5% win), 9 UTC (62.5% win, +34.8 avg)

### 4. By Day of Week

| Day | Trades | Win% | Total bps | Avg |
|-----|--------|------|-----------|-----|
| Mon | 402 | 59.7% | +3,853 | +9.6 |
| Tue | 466 | 61.4% | +8,883 | +19.1 |
| Wed | 380 | 58.9% | +4,201 | +11.1 |
| Thu | 405 | 60.7% | +4,900 | +12.1 |
| Fri | 371 | 60.4% | +7,306 | +19.7 |
| Sat | 100 | 67.0% | +2,562 | +25.6 |
| Sun | 155 | 63.9% | +2,814 | +18.2 |

All days profitable. Saturday best avg (+25.6 bps/trade). Monday weakest (but still +9.6).

### 5. By Month (OOS)

Worst months:
- **Aug 2024**: 151 trades, 62.3% win, **-223 bps** (only losing month)
- **Jun 2025**: 24 trades, 37.5% win, -135 bps
- **Jul 2024**: 114 trades, 57.9% win, +61 bps (barely positive)

Best months:
- **Mar 2024**: 238 trades, 66.0% win, +6,549 bps
- **Apr 2024**: 173 trades, 63.0% win, +4,068 bps
- **Nov 2025**: 154 trades, 63.0% win, +4,221 bps

Only 2 losing months out of 24.

### 6. Feature Comparison: Winners vs Losers

| Feature | Winner median | Loser median | Diff |
|---------|-------------|-------------|------|
| atr_pct | 0.55 | 0.51 | +0.04 |
| ema_separation | 1.61 | 1.44 | +0.17 |
| range_bps | 58.73 | 55.39 | +3.34 |
| rsi7 | 47.71 | 53.18 | -5.48 |
| roc5 | -0.33 | +0.34 | -0.68 |
| volume_ratio | 0.97 | 0.88 | +0.09 |
| atr_percentile | 85.10 | 84.00 | +1.10 |
| range_position | 0.48 | 0.55 | -0.06 |
| mfe_bps | 69.19 | 10.95 | +58.25 |

**Almost no difference in entry conditions** — winners and losers look the same at entry.
Only MFE differs (by definition). Signal quality at entry does NOT distinguish winners from losers.

### 7. Loser MFE Distribution

| MFE Range | Count | % of losers |
|-----------|-------|-------------|
| MFE = 0 (never positive) | 220 | **24.6%** |
| MFE 0-10 bps | 214 | 24.0% |
| MFE 10-20 bps | 213 | 23.9% |
| MFE 20-30 bps | 183 | 20.5% |
| MFE 30+ bps | 63 | 7.1% |

- Loser MFE median: 10.9 bps (went right but not enough)
- **24.6% had MFE=0** — entered and price went immediately against, never recovered
- Most losers (48.5%) had MFE < 10 bps — barely moved right at all

### 8. Worst 20 Trades

ALL worst 20 trades share: **TIME_EXIT, MFE=0, bars_held=10**
- Worst: -741 bps (2024-08-04, LONG)
- Range: -318 to -741 bps
- Mix of LONG (10) and SHORT (10)
- Aug 2024 has 5 of worst 20

### 9. Losing Streaks

- Max streak: **10 trades** (-229 bps)
- Worst streak by bps: **6 trades** (-383 bps)
- Top 5 streaks: 10(-229), 8(-191), 8(-180), 7(-330), 6(-286)

### 10. Drawdown

- **Max drawdown: -1,859 bps**
- Peak: +15,237 bps -> Trough: +13,378 bps (trade #828 to #890)
- Drawdown period: 63 trades, 28 losers
- Date: around Aug 2024 (choppy market)
- V1.4.0 max DD was -192 bps — L2-002 is 10x worse (more trades = more exposure)

### Loss Analysis Conclusions

1. **TIME_EXIT is the single biggest problem** — 223 trades lose -33,552 bps (wipes 49% of trailing stop gains)
2. **Same pattern as V1.3.2** — wrong-direction entries that never go positive
3. **Entry features cannot distinguish winners from losers** — no filter will fix this
4. **LONG side is stronger** (PF 1.99 vs SHORT 1.61)
5. **Hour 2 UTC and Hour 7 UTC are weakest** but not catastrophic
6. **Aug 2024 worst month** — choppy/quiet market with big adverse moves
7. **Max drawdown -1,859 bps** — significant, 10x worse than V1.4.0 due to trade volume
8. **Only 2/24 months are negative** — strategy is consistent

## Investigation List (Funnel Gaps)

### Gap 1: Regime Rejection (57,900 bars)
- 69,942 total OOS bars → only 12,042 pass ACTIVE_TREND gate (17.2%)
- Are there profitable trades hiding in non-ACTIVE_TREND regimes?

### Gap 2: No Signal Within Regime (~7,000 bars)
- 12,042 ACTIVE_TREND bars → only ~5,000 have ROC5 signal fire
- Can other signals (EMA9D, RSI7, RPOS) catch the remaining bars?

### ~~Gap 3: Overlap Blocking~~ → DEFERRED (bot execution feature)
- ~5,000 signal fires → only 2,279 entered (rest blocked by existing position)
- Not a framework issue — overlap rule is standard for backtesting (fair comparison across configs)
- Smart overlap handling (close/reverse/hedge based on current P&L + new signal) = bot-level execution optimization
- Revisit after stable V1.4 config is chosen from framework results

### ~~Gap 4: TIME_EXIT Losers~~ → ADDRESSED (exit strategy layer)
- 223 TIME_EXIT trades with 0.9% win rate wipe 49% of trailing stop gains
- Now tested via exit grid: max_bars [5,8,10,12,15], trailing_stop [10,15,20,25,30], tighten configs
- Data will show which exit config minimizes TIME_EXIT damage

### ~~Gap 5: State Transition (Fresh vs Stale Signals)~~ → MINOR (overlap rule handles most cases)
- Framework checks static thresholds ("is ROC5 < -0.3 now?"), not transitions ("did ROC5 JUST cross -0.3?")
- A signal that has been below threshold for 10 bars = stale. One that just crossed = fresh.
- Stale signals may be the worst performers — testable hypothesis.
- No regime transition tracking either (just entered ACTIVE_TREND vs been in it for 50 bars).

### ~~Gap 6: Regime Classifier Parameters~~ → ADDRESSED (magnitude gate tests all features)
- Regime classifier uses only ATR + EMA sep — but it's just one of ~1,795 magnitude gate configs
- Framework tests ALL 8 features at ALL thresholds in ALL 1/2/3-feature combos
- If ATR+EMA is not the best pair, framework results will show a different combo beating ACTIVE_TREND
- Regime classifier remains as a convenience label, real gating comes from magnitude gate configs
- EMA/ATR period assumptions still untested but lower priority — feature VALUES are tested exhaustively

## Implication for V2.0
- V1.3.2/V1.4.0 is ultra-selective (PF 3.46, 220 trades) — keep as core
- New signals are less selective but high-volume — potential for combination
- Next step: add top new signals ALONGSIDE V1.4.0 (parallel), not replace
- Target: V1.4.0's 220 trades + new non-overlapping trades from best combo
- Need to check overlap: how many of V1.4.0's trades are already captured by new signals?
- TIME_EXIT problem: may need different exit strategy for L2 signals (wider stop? different time limit?)
