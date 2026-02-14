# Layer 2: Signal Expansion Using Regimes + State Vectors

## Goal
V1.3.2 trades 220 times in 2 years (0.3% of bars). Find MORE profitable entries using validated features + regimes.

**Target:** 350-500 trades, PF > 2.0, total bps > +5,267

---

## State Vector Design

State vector = snapshot of market features at each bar. Must be built from VALIDATED features only.

**Feature candidates (need re-validation on 15-min):**

**L2-001 tested 52 features (all 38 WHEN + SMA200 + extras) with 5 tests:**
1. Directional accuracy — Q4 predicts direction better than Q1? (WHAT re-validation)
2. P(Case1) structural failure — Q4 has less Case1 than Q1? (WHEN re-validation)
3. Raw MFE separation — Q4 has higher forward MFE than Q1? (Stage A: raw opportunity)
4. V1.3.2 LONG PnL — Q4 makes more profit than Q1 trading LONG? (Stage B)
5. V1.3.2 SHORT PnL — Q4 makes more profit than Q1 trading SHORT? (Stage B)

### 5/5 VALIDATED (17 features → 7 distinct):
| Feature | WHEN status | Dir | Case1 | MFE | LONG | SHORT |
|---------|-------------|-----|-------|-----|------|-------|
| atr_pct | ROBUST | YES | YES | YES | YES | YES |
| atr7_pct | ROBUST | YES | YES | YES | YES | YES |
| atr21_pct | ROBUST | YES | YES | YES | YES | YES |
| std20 | ROBUST | YES | YES | YES | YES | YES |
| range_bps | ROBUST | YES | YES | YES | YES | YES |
| body_bps | ROBUST | YES | YES | YES | YES | YES |
| ema_separation | ROBUST | YES | YES | YES | YES | YES |
| dist_from_high20_pct | ROBUST | YES | YES | YES | YES | YES |
| atr_percentile | STRONG | YES | YES | YES | YES | YES |
| hour_utc | PARTIAL | YES | YES | YES | YES | YES |
| volume_ratio | WEAK | YES | YES | YES | YES | YES |
| volume_trend | INVALID | YES | YES | YES | YES | YES |
| keltner_width | EXTRA | YES | YES | YES | YES | YES |
| donchian_width | EXTRA | YES | YES | YES | YES | YES |
| bb_width | EXTRA | YES | YES | YES | YES | YES |
| bar_range_avg_10 | EXTRA | YES | YES | YES | YES | YES |
| recent_volatility | EXTRA | YES | YES | YES | YES | YES |

**After removing correlated duplicates (95%+ correlation), 7 DISTINCT features:**

| # | Feature | What it measures | Represents |
|---|---------|-----------------|------------|
| 1 | atr_pct | Volatility (14-bar ATR) | 9 vol features (atr7, atr21, std20, keltner, donchian, bb_width, bar_range_avg_10, recent_volatility) |
| 2 | atr_percentile | Relative volatility (ATR rank vs last 100 bars) | Unique — current vol vs recent history |
| 3 | ema_separation | Trend strength (EMA9 vs EMA21 distance) | Unique — how strongly market is trending |
| 4 | range_bps / body_bps | Current bar size | Correlated pair — how big THIS bar is |
| 5 | dist_from_high20_pct | Price position in 20-bar range | Unique — where price sits in recent range |
| 6 | hour_utc | Time of day | Unique — session/liquidity |
| 7 | volume_ratio | Volume vs 20-bar average | Unique — activity level |

### 3/5 VALIDATED (30 features — fail Case1 + raw MFE):
ALL 19 WHEN "INVALID" features score 3/5 on 15-min (pass Dir + LONG + SHORT, fail Case1 + MFE):
- RSI: rsi, rsi7, rsi21
- Momentum: roc5, roc10, roc20, momentum5, momentum10
- MA distance: ema9/20/50/100/200_dist_pct
- MA slope: ema20/50/200_slope
- Others: bb_position, range_position

Plus WHEN ROBUST that dropped: dist_from_low20_pct, ll_count5, day_of_week
Plus WHEN STRONG: hh_count5, session
Plus WHEN WEAK: up_bars5, down_bars5
Plus SMA200: sma200_dist_pct
Plus EXTRA: range_position_50

### 2/5 or less:
| Feature | Score | Notes |
|---------|-------|-------|
| sma200_slope | 2/5 | Direction no, Case1 no, MFE no |
| rsi_oversold_zone | 2/5 | Dir+Case1 yes, PnL fails (binary) |
| rsi_extreme_oversold | 2/5 | Same |
| session_asia_night | 2/5 | Dir+Case1 yes, PnL fails (binary) |
| price_above_sma200 | 0/5 | Binary — quartile test doesn't work |
| is_weekend | 0/5 | REJECTED |
| session_europe | 0/5 | REJECTED |
| session_us | 0/5 | REJECTED |

### KEY FINDINGS:

**1. Features serve TWO distinct roles — don't mix them**
- **MAGNITUDE features** (5/5): predict move SIZE → use as GATE (is there enough movement?)
- **DIRECTION features** (3/5): predict which WAY → use as SIGNAL (LONG or SHORT?)
- A feature doesn't need to do both. Combine: magnitude gate + direction signal = entry
- MFE/Case1 tests measure magnitude. Direction/PnL tests measure signal quality.

**2. Magnitude features (GATE) — 7 distinct, all 5/5:**
- atr_pct, atr_percentile, ema_separation, range_bps/body_bps, dist_from_high20_pct, hour_utc, volume_ratio
- These separate big-move bars from small-move bars

**3. Direction features (SIGNAL) — 3/5, validated for direction + V1.3.2 PnL:**
- RSI: rsi, rsi7, rsi21
- Momentum: roc5, roc10, roc20, momentum5, momentum10
- MA distance: ema9/20/50/100/200_dist_pct
- MA slope: ema20/50/200_slope
- Others: bb_position, range_position, sma200_dist_pct
- These predict direction but not move size — that's OK, magnitude features handle size

**4. WHEN "INVALID" features work on 15-min**
- All 19 WHEN INVALID features (tested on 1-min) score 3/5 on 15-min
- Timeframe matters: 1-min conclusions don't transfer to 15-min

**5. V1.3.2 captures only 20-25% of raw MFE**
- Example: atr_pct Q4 has 94bp median raw MFE, V1.3.2 captures ~22bp avg
- 75% of opportunity left on table by current exit mechanics
- Implication: better exits could significantly increase per-trade profit

**6. Framework for L2-002/003:**
- Magnitude gate (enough movement?) + Direction signal (which way?) = Entry
- Test each feature in its ROLE, not force every feature to pass every test

---

## Experiments

### L2-001: Feature Re-Validation on 15-min [DONE]

**Why:** WHAT was on 1-min horizons, WHEN was on random entry. Need to verify on 15-min with our context.

**Method:**
1. Compute all candidate features on every 15-min bar (TRAIN)
2. Compute forward MFE/MAE at 10, 15, 20 bars (raw opportunity)
3. For each feature: does it separate high-MFE from low-MFE bars? (Stage A)
4. For features that pass: simulate with V1.3.2 exits (Stage B)
5. Compare: raw opportunity vs captured profit
6. Validate survivors on OOS

**Output:**
- Validated state vector (features that actually work on 15-min)
- Which opportunities V1.3.2 can capture vs needs new exits

### L2-002: Relaxed RSI + State Vector Gating [TODO]
- Widen RSI thresholds (25/30/35) gated by validated regime features
- Depends on L2-001 results

### L2-003: Volume Spike + State Vector Gating [TODO]
- Filter raw Volume Spike (1,642 trades, PF 1.43) using validated features
- Depends on L2-001 results

### L2-004: V2.0 Assembly [TODO]
- Combine V1.3.2 + new signals from L2-002/003
- Deduplicate, backtest combined, validate OOS

---

## Key Concerns to Track

1. WHAT was on 1-min horizons — thresholds don't transfer to 15-min
2. WHEN was on random entry — may not apply to selective entry
3. Cohen's d was on 220 trades — small sample
4. Previous state vector (10D) had 8/10 useless features
5. Number of regimes should come from data, not assumed

---

## Results Log

| Experiment | Date | Result | Decision |
|------------|------|--------|----------|
| Layer 2 v1 (filtering) | 2026-02 | All filters reduce profit | REJECTED |
| L2-001 v1 (26 features, 4 tests) | 2026-02-14 | 14 at 4/4, 6 at 3/4 — incomplete feature set | SUPERSEDED |
| L2-001 v2 (52 features, 4 tests) | 2026-02-14 | 17 at 4/4, 30 at 3/4 — missing raw MFE test | SUPERSEDED |
| L2-001 v3 (52 features, 5 tests) | 2026-02-14 | 17 at 5/5 → 7 distinct features. V1.3.2 captures 20-25% of raw MFE | FINAL |

---

## Status: L2-001 COMPLETE — Ready for L2-002/L2-003
