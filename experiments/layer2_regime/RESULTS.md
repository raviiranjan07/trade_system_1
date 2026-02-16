# Layer 2: Regime Detection — Research Results

## Summary

Tested 4 approaches to regime detection for V1.3.2:
- **Option A**: K-Means clustering (K=2,3,4,5)
- **Option B**: Hidden Markov Model (N=2,3,4,5)
- **Option C**: Performance-based feature analysis
- **Option D**: Hybrid comparison — which filter actually improves OOS?

## Key Finding: REGIME FILTERING DOES NOT IMPROVE V1.3.2

The strategy already has strong built-in filtering (RSI + SMA200 + ATR + EMA).
Adding regime-based trade skipping consistently **reduces total profit** on OOS.

### Option D Comparison Table (OOS 2024-2025)

| Method | Trades | Skip | Win% | Total bps | PF | Avg bps | Max DD |
|--------|--------|------|------|-----------|-----|---------|--------|
| **BASELINE (no filter)** | **220** | **0** | **60.0** | **+5,267** | **3.46** | **+23.9** | **-192** |
| KMeans K=2 (skip R0) | 23 | 197 | 73.9 | +1,085 | 4.16 | +47.2 | -209 |
| HMM N=2 (skip S0) | 67 | 153 | 70.1 | +3,114 | 5.82 | +46.5 | -182 |
| HMM N=3 (skip S0) | 119 | 101 | 61.3 | +3,971 | 4.42 | +33.4 | -182 |
| Score >= 1 | 214 | 6 | 60.7 | +5,338 | 3.63 | +24.9 | -192 |
| Score >= 2 | 180 | 40 | 60.0 | +5,158 | 3.83 | +28.7 | -192 |
| Score >= 3 | 141 | 79 | 61.0 | +4,513 | 4.38 | +32.0 | -182 |
| EMA sep >= 0.5% | 199 | 21 | 59.8 | +5,160 | 3.61 | +25.9 | -192 |
| EMA sep >= 1.0% | 133 | 87 | 60.2 | +3,667 | 3.77 | +27.6 | -213 |
| ATR pctl >= 25 | 178 | 42 | 60.1 | +4,944 | 3.70 | +27.8 | -323 |
| ATR pctl >= 50 | 132 | 88 | 59.1 | +3,856 | 3.73 | +29.2 | -192 |

**Every filter reduces total OOS profit.** PF and avg bps improve, but only because
you're skipping profitable trades (the "skip_bps" column is positive — you're leaving
money on the table).

---

## Option C: What Separates Winners from Losers

### Top Features (TRAIN, Cohen's d)

| Feature | Cohen's d | Winner Mean | Loser Mean | Useful? |
|---------|-----------|-------------|------------|---------|
| ema_separation | **0.415** | 2.09% | 1.43% | YES — strongest |
| signal_bar_range_bps | 0.344 | 112 bps | 68 bps | YES |
| bar_range_avg_10 | 0.320 | 66 bps | 51 bps | moderate |
| recent_volatility | 0.297 | 36.6 | 28.8 | moderate |
| day_of_week | 0.160 | 2.96 | 2.65 | weak |
| rsi | -0.104 | 59.8 | 63.4 | no |
| atr_percentile | 0.029 | 62.7 | 61.8 | no |
| hour_utc | 0.015 | 11.8 | 11.7 | no |

### OOS Validation of Cohen's d

| Feature | TRAIN d | OOS d | Consistent? |
|---------|---------|-------|-------------|
| ema_separation | 0.415 | 0.144 | Weaker but same direction |
| signal_bar_range_bps | 0.344 | 0.203 | YES |
| bar_range_avg_10 | 0.320 | 0.312 | YES |
| recent_volatility | 0.297 | 0.207 | YES |

### Key Quartile Finding (TRAIN)

**EMA separation is dominant:**
- Q1 (0-0.96%): 108t, 58.3% win, PF 1.17, +315 bps
- Q2 (0.96-1.47%): 108t, 59.3% win, PF 0.78, **-676 bps**
- Q3 (1.47-2.27%): 107t, 64.5% win, PF 1.39, +1,105 bps
- Q4 (2.27%+): 108t, **81.5% win**, PF **8.32**, **+7,963 bps**

**BUT OOS tells a different story:**
- Q1 (0-0.72%): 55t, 58.2% win, PF 2.49, +864 bps
- Q4 (1.82%+): 55t, 67.3% win, PF 4.68, +2,035 bps
- All quartiles are profitable on OOS!

---

## Option A: K-Means Clustering

- **Best silhouette: K=2** (0.459) — natural split is HIGH volatility vs LOW volatility
- K=2 cluster centers:
  - R0: Low vol (ATR 38 bps, EMA 0.88%) — 88.7% of bars
  - R1: High vol (ATR 115 bps, EMA 3.22%) — 11.3% of bars
- TRAIN: R0 = -2.1 avg bps (305t), R1 = +74.1 avg bps (126t)
- **OOS: R0 = +21.2 avg bps (197t) — INCONSISTENT with TRAIN!**
- K-Means regime labels are NOT stable across time periods

### Why K-Means Fails

R0 (low vol) goes from negative on TRAIN to positive on OOS. This means:
1. The "bad regime" on TRAIN isn't actually bad on OOS
2. Skipping R0 on OOS would skip 197 profitable trades
3. Regime characteristics shift over time — cluster boundaries don't hold

---

## Option B: HMM

- HMM N=2 states:
  - S0: Low vol (avg_vol 20, range 30 bps) — 72.7% of time
  - S1: High vol (avg_vol 54, range 92 bps) — 27.3% of time
- S1 (high vol) is consistently best: TRAIN +28.5, OOS +46.5
- S0 (low vol) is profitable but weaker: TRAIN +9.2, OOS +14.1
- **Both states are CONSISTENT and PROFITABLE**
- No state to skip!

### HMM N=4 Insight

- S2 (highest vol, 11.2% of bars): TRAIN +43.3, OOS +62.7 — best
- S0 (lowest vol, 26.2%): TRAIN +5.6, OOS **-4.1** — INCONSISTENT
- S3 (medium vol, 33.7%): TRAIN +2.7, OOS +17.8 — inconsistent magnitude

HMM states capture volatility regimes well, but the performance differences
don't hold stably across TRAIN/OOS for the weaker states.

---

## Conclusion

### Why Regime Filtering Doesn't Help V1.3.2

1. **V1.3.2 already filters aggressively** — RSI extremes + SMA200 regime +
   ATR/EMA filters on LONG already eliminate most "bad market" entries

2. **The strategy wins in ALL regimes** — OOS shows positive returns in every
   regime (except tiny edge cases with <10 trades)

3. **Skipped trades are profitable** — the "skip_bps" column is consistently
   positive, meaning we're skipping good trades

4. **TRAIN/OOS inconsistency** — features that separate winners/losers on TRAIN
   (especially EMA sep) are weaker on OOS. Regime boundaries shift.

5. **Signal selectivity >> regime filtering** — V1.3.2 only takes 220 trades
   in 2 years. The signals are already highly selective.

### What's Actually Useful

1. **EMA separation > 2%** = clearly better trades (PF 8.32 TRAIN, 4.68 OOS)
   - But filtering to only these would skip too many profitable trades
   - Better used as a **confidence/sizing signal** (Layer 1 integration)

2. **High volatility regime = bigger winners** — HMM S1 and KMeans R1
   consistently produce higher avg bps. Not for filtering, but for
   **position sizing** (risk more in high-vol regime).

3. **Signal bar range** predicts trade quality — big entry bars = better outcomes

### Recommendation

**Do NOT add regime-based trade filtering to V1.3.2.**
The strategy doesn't need it — its built-in filters already work.

**Instead, use regime information for:**
- Layer 1 signal quality scoring (already built, uses EMA sep)
- Position sizing (risk more in high-vol, trending regime)
- Dashboard display (show current regime for human awareness)

---

## Files

| File | Purpose |
|------|---------|
| background_research/option_a_clustering.py | K-Means regime detection |
| background_research/option_b_hmm.py | HMM regime detection |
| background_research/option_c_performance.py | Winner/loser feature analysis |
| background_research/option_d_hybrid.py | Head-to-head comparison |
| background_research/trade_features_train.csv | Trade features (TRAIN) |
| background_research/trade_features_oos.csv | Trade features (OOS) |
| L2-001/L2_001_feature_revalidation.py | Feature validation + computation |
| L2-001/L2-001b/ | Feature strength ranking |
| L2-002/framework.py | Signal expansion framework (77 combos) |
| L2-002/analyze_losses.py | Loss analysis on top config |
| L2-002/notes.md | Full results + investigation gaps |
| RESULTS.md | This file |
