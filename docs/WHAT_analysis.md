# What We Learned About BTC 1-Minute Trading

Date: 2026-01-11 (Updated: 2026-01-15)

**Data:** 3,150,249 1-minute candles (2019-12-31 to 2025-12-30) - Full dataset

---

## The Core Question

Can we profitably scalp BTC at 1-minute timeframe using state vectors and similarity search?

---

## What We Tested

We ran extensive analysis on 3.15 million 1-minute BTC candles (full 6-year dataset) to find:
1. What moves does the market actually provide?
2. Is random entry profitable at any threshold combination?
3. Does our state vector provide predictive edge?

---

## ANALYSIS-1: The Market Moves Enough

**Question:** Does BTC 1-minute data have enough price movement to cover fees (8bp) and profit (12bp+ per Rule #1)?

**Data:** Full dataset (2019-2025), 100K sample

**Using 12bp threshold (Rule #1):**

| Horizon | Noise (<12bp) | Real UP only | Real DOWN only | Real BOTH | Total Real |
|---------|---------------|--------------|----------------|-----------|------------|
| H=3     | 58.7%         | 18.4%        | 18.7%          | 4.2%      | 41.3%      |
| H=5     | 45.7%         | 23.2%        | 23.6%          | 7.6%      | 54.3%      |
| H=10    | 27.9%         | 28.2%        | 28.6%          | 15.2%     | 72.1%      |
| H=15    | 19.3%         | 29.3%        | 29.7%          | 21.7%     | 80.7%      |
| H=30    | 8.7%          | 28.1%        | 28.3%          | 34.9%     | 91.3%      |
| H=60    | 3.3%          | 23.6%        | 23.4%          | 49.7%     | 96.7%      |
| H=120   | 1.0%          | 18.1%        | 17.9%          | 63.0%     | 99.0%      |
| H=240   | 0.2%          | 13.0%        | 12.7%          | 74.2%     | 99.8%      |
| H=360   | 0.1%          | 10.5%        | 10.0%          | 79.4%     | 99.9%      |
| H=480   | 0.0%          | 9.0%         | 8.5%           | 82.5%     | 100.0%     |
| H=600   | 0.0%          | 7.9%         | 7.4%           | 84.7%     | 100.0%     |

**Key observations:**
- At H=3: Only 41.3% have tradeable moves (≥12bp), 58.7% are noise
- At H=60: 96.7% have tradeable moves, only 3.3% are noise
- At H=600: 100% have tradeable moves, 85% hit BOTH directions
- Longer horizon = more movement = less noise
- At H=240+: Almost all bars hit BOTH 12bp up AND 12bp down (no edge in direction)

**Conclusion:** Market provides sufficient movement. Problem is NOT lack of movement, problem is DIRECTION.

---

## ANALYSIS-2: Direction is 50/50 (All Bars, By Horizon)

**Question:** For ALL bars, which direction hits ±12bp first?

**Data:** Full dataset (2019-2025), 100K sample

**Threshold:** 12bp (Rule #1: minimum profitable move)

| Horizon | UP First | DOWN First | Neither (Noise) | Ratio |
|---------|----------|------------|-----------------|-------|
| H=3     | 20.4%    | 20.8%      | 58.7%           | 0.98  |
| H=5     | 27.0%    | 27.4%      | 45.7%           | 0.99  |
| H=10    | 35.8%    | 36.2%      | 27.9%           | 0.99  |
| H=15    | 40.1%    | 40.6%      | 19.3%           | 0.99  |
| H=30    | 45.3%    | 45.9%      | 8.7%            | 0.99  |
| H=60    | 48.1%    | 48.7%      | 3.3%            | 0.99  |
| H=120   | 49.2%    | 49.8%      | 1.0%            | 0.99  |
| H=240   | 49.6%    | 50.2%      | 0.2%            | 0.99  |
| H=360   | 49.7%    | 50.2%      | 0.1%            | 0.99  |
| H=480   | 49.7%    | 50.3%      | 0.0%            | 0.99  |
| H=600   | 49.7%    | 50.3%      | 0.0%            | 0.99  |

**Key insight:**
- Ratio = 0.99 at ALL horizons (UP First / DOWN First)
- Direction is ~50/50 regardless of horizon - even at H=600!
- "Neither" matches "Noise" from ANALYSIS-1 (cross-validated)
- Even at H=600 (10 hours), direction is still 49.7% vs 50.3% - no edge

**Compare to ANALYSIS-6:** ANALYSIS-6 shows direction among real moves only (excludes "Neither").

**Conclusion:** Cannot predict direction from random entry at any horizon.

---

## ANALYSIS-3: Random Entry Has No Edge

**Question:** Can any target/stop/horizon combination be profitable with random entry?

**Data:** Full dataset (2019-2025), verified 0/18 combinations profitable

**Why random entry fails (the math):**

When target = stop, with 8bp fee:
```
Net Win  = Target - 8bp
Net Loss = Stop + 8bp
Break-even Win Rate = Net Loss / (Net Win + Net Loss)
```

| Target | Net Win | Net Loss | Break-even | Random gives | Gap |
|--------|---------|----------|------------|--------------|-----|
| 8bp | 0bp | 16bp | Impossible | 50% | - |
| 12bp | 4bp | 20bp | 83.3% | 50% | -33% |
| 15bp | 7bp | 23bp | 76.7% | 50% | -27% |
| 20bp | 12bp | 28bp | 70.0% | 50% | -20% |
| 25bp | 17bp | 33bp | 66.0% | 50% | -16% |
| 50bp | 42bp | 58bp | 58.0% | 50% | -8% |

**Random entry gives ~50% win rate (from ANALYSIS-2), but even the easiest target (50bp) needs 58%.**

**Grid search verification:**

We tested combinations including extended horizons:
- Horizons: 3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600 bars
- Targets: 8 to 50 bps
- Stops: 5 to 40 bps

Result: **ZERO combinations had positive expected value after 8 bps fees.**

Best combination (H=3, Target=20bps, Stop=40bps) had EV = -0.03 bps per trade.

**Extended horizon conclusion:**
- At H=600 (10 hours), direction is still 49.6% UP vs 50.4% DOWN
- Longer horizons do NOT improve win rate
- Random entry fails at ALL horizons, including multi-hour holds

**Conclusion:** Fees require >50% win rate. Random entry gives 50%. Therefore, random entry always loses - regardless of holding period.

---

## ANALYSIS-4: Trade Outcome Analysis (Complete)

**Question:** What happens after entry? How often do we win? How much drawdown? How long does it take?

**Categories:**
- **Clean Win:** Hit target with ZERO drawdown (price never went below entry)
- **Dirty Win:** Hit target but had drawdown first (price went below entry before winning)
- **Never Hit:** Never hit target within H bars

---

### TABLE 1: Win Rate % (Clean Win + Dirty Win)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 25.6 | 34.3 | 46.6 | 54.0 | 65.3 | 74.8 | 82.3 | 87.9 | 90.4 | 91.8 | 93.0 |
| 15bp | 19.1 | 27.0 | 38.9 | 46.5 | 58.7 | 69.3 | 78.1 | 84.8 | 87.9 | 89.7 | 91.1 |
| 25bp | 8.4 | 13.1 | 22.1 | 28.6 | 40.8 | 53.4 | 65.5 | 75.3 | 80.2 | 83.1 | 85.1 |
| 50bp | 1.9 | 3.4 | 7.2 | 10.3 | 18.5 | 29.3 | 42.2 | 56.0 | 63.3 | 68.1 | 71.6 |
| 100bp | 0.3 | 0.6 | 1.6 | 2.6 | 5.6 | 10.8 | 19.3 | 31.0 | 39.1 | 44.9 | 49.3 |
| 150bp | 0.1 | 0.2 | 0.5 | 0.9 | 2.3 | 5.0 | 10.1 | 18.6 | 25.2 | 30.5 | 34.8 |
| 200bp | 0.1 | 0.1 | 0.2 | 0.4 | 1.1 | 2.6 | 5.8 | 11.6 | 17.0 | 21.4 | 25.2 |

---

### TABLE 2: Clean Win % (no drawdown - perfect trades)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 2.8 | 3.3 | 3.7 | 3.8 | 3.9 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| 15bp | 2.1 | 2.6 | 2.9 | 3.1 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 |
| 25bp | 0.9 | 1.2 | 1.6 | 1.8 | 2.0 | 2.1 | 2.2 | 2.2 | 2.2 | 2.2 | 2.2 |
| 50bp | 0.2 | 0.3 | 0.5 | 0.6 | 0.8 | 1.0 | 1.1 | 1.1 | 1.2 | 1.2 | 1.2 |
| 100bp | 0.0 | 0.0 | 0.1 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.5 | 0.5 | 0.5 |
| 150bp | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.2 | 0.3 | 0.3 | 0.3 | 0.4 |
| 200bp | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.2 | 0.2 | 0.2 | 0.2 |

---

### TABLE 2b: Dirty Win % (hit target but had drawdown first)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 22.8 | 31.0 | 42.9 | 50.2 | 61.4 | 70.8 | 78.3 | 83.9 | 86.4 | 87.8 | 89.0 |
| 15bp | 17.0 | 24.4 | 36.0 | 43.4 | 55.4 | 66.0 | 74.8 | 81.5 | 84.6 | 86.4 | 87.8 |
| 25bp | 7.5 | 11.9 | 20.5 | 26.8 | 38.8 | 51.3 | 63.3 | 73.1 | 78.0 | 80.9 | 82.9 |
| 50bp | 1.7 | 3.1 | 6.7 | 9.7 | 17.7 | 28.3 | 41.1 | 54.9 | 62.1 | 66.9 | 70.4 |
| 100bp | 0.3 | 0.6 | 1.5 | 2.5 | 5.4 | 10.5 | 18.9 | 30.5 | 38.6 | 44.4 | 48.8 |
| 150bp | 0.1 | 0.2 | 0.5 | 0.9 | 2.2 | 4.9 | 9.9 | 18.3 | 24.9 | 30.2 | 34.4 |
| 200bp | 0.1 | 0.1 | 0.2 | 0.4 | 1.1 | 2.5 | 5.7 | 11.4 | 16.8 | 21.2 | 25.0 |

---

### TABLE 3: MAE Median (bp drawdown before hitting target - WINNERS ONLY)

**🔗 Related:** See **ANALYSIS-5** for detailed per-horizon breakdown with Mean, Max, Count stats.

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 3 | 3 | 4 | 5 | 6 | 7 | 9 | 10 | 10 | 11 | 11 |
| 15bp | 3 | 3 | 5 | 5 | 7 | 8 | 10 | 11 | 12 | 13 | 13 |
| 25bp | 3 | 4 | 5 | 6 | 8 | 11 | 13 | 16 | 17 | 18 | 19 |
| 50bp | 4 | 5 | 7 | 8 | 10 | 13 | 17 | 22 | 25 | 28 | 30 |
| 100bp | 7 | 7 | 9 | 10 | 12 | 15 | 21 | 28 | 32 | 36 | 40 |
| 150bp | 9 | 10 | 10 | 11 | 13 | 17 | 22 | 29 | 35 | 40 | 44 |
| 200bp | 9 | 9 | 10 | 12 | 17 | 19 | 23 | 30 | 36 | 41 | 45 |

---

### TABLE 4: MAE 75th Percentile (bp - worse 25% of winners)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 7 | 8 | 10 | 12 | 15 | 18 | 21 | 25 | 27 | 28 | 29 |
| 15bp | 7 | 9 | 11 | 13 | 16 | 20 | 24 | 28 | 31 | 32 | 34 |
| 25bp | 9 | 10 | 13 | 15 | 19 | 24 | 30 | 37 | 42 | 45 | 47 |
| 50bp | 12 | 14 | 17 | 18 | 23 | 29 | 39 | 50 | 57 | 63 | 67 |
| 100bp | 17 | 18 | 23 | 25 | 29 | 36 | 46 | 61 | 71 | 80 | 86 |
| 150bp | 23 | 28 | 32 | 32 | 35 | 42 | 51 | 65 | 78 | 87 | 95 |
| 200bp | 22 | 26 | 41 | 38 | 44 | 47 | 54 | 68 | 81 | 91 | 98 |

---

### TABLE 4b: MAE 95th Percentile (bp - worst 5% of winners)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 21 | 24 | 28 | 33 | 41 | 51 | 65 | 81 | 91 | 100 | 106 |
| 15bp | 23 | 26 | 30 | 34 | 43 | 55 | 71 | 87 | 101 | 110 | 118 |
| 25bp | 27 | 30 | 36 | 40 | 50 | 65 | 82 | 105 | 122 | 135 | 144 |
| 50bp | 43 | 46 | 52 | 54 | 64 | 82 | 104 | 130 | 151 | 168 | 183 |
| 100bp | 60 | 66 | 75 | 75 | 92 | 105 | 129 | 158 | 181 | 200 | 215 |
| 150bp | 66 | 77 | 102 | 105 | 123 | 124 | 143 | 173 | 199 | 218 | 233 |
| 200bp | 57 | 96 | 101 | 125 | 144 | 144 | 154 | 186 | 219 | 239 | 254 |

---

### TABLE 5: Bars in Drawdown - Median (bars spent below entry before winning - WINNERS ONLY)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 1 | 1 | 2 | 2 | 3 | 4 | 5 | 5 | 6 | 6 | 6 |
| 15bp | 1 | 1 | 2 | 2 | 3 | 4 | 6 | 7 | 8 | 8 | 9 |
| 25bp | 1 | 1 | 2 | 2 | 4 | 6 | 9 | 12 | 14 | 16 | 17 |
| 50bp | 1 | 1 | 2 | 2 | 4 | 6 | 11 | 19 | 25 | 30 | 34 |
| 100bp | 1 | 1 | 1 | 2 | 3 | 6 | 11 | 21 | 31 | 40 | 48 |
| 150bp | 1 | 1 | 1 | 2 | 3 | 6 | 10 | 20 | 30 | 40 | 49 |
| 200bp | 1 | 1 | 1 | 2 | 3 | 5 | 9 | 18 | 28 | 37 | 47 |

---

### TABLE 6: Bars in Drawdown - 75th Percentile (worse 25% of winners)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 2 | 2 | 4 | 5 | 8 | 12 | 17 | 23 | 27 | 29 | 31 |
| 15bp | 1 | 2 | 4 | 5 | 8 | 13 | 20 | 29 | 35 | 39 | 42 |
| 25bp | 1 | 2 | 3 | 5 | 9 | 15 | 27 | 44 | 57 | 66 | 74 |
| 50bp | 1 | 2 | 3 | 4 | 8 | 16 | 31 | 59 | 82 | 103 | 122 |
| 100bp | 1 | 2 | 3 | 4 | 8 | 14 | 29 | 59 | 89 | 116 | 144 |
| 150bp | 1 | 2 | 3 | 4 | 7 | 14 | 26 | 54 | 86 | 114 | 143 |
| 200bp | 1 | 1 | 3 | 4 | 8 | 13 | 24 | 49 | 81 | 106 | 134 |

---

### TABLE 7: Time to Target - Median (bars from entry to hitting target - WINNERS ONLY)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp | 2 | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 10 | 10 |
| 15bp | 2 | 2 | 4 | 4 | 6 | 8 | 11 | 13 | 14 | 14 | 15 |
| 25bp | 2 | 3 | 5 | 6 | 9 | 14 | 20 | 26 | 29 | 32 | 34 |
| 50bp | 2 | 3 | 6 | 8 | 14 | 23 | 37 | 56 | 69 | 79 | 86 |
| 100bp | 2 | 3 | 7 | 9 | 16 | 30 | 53 | 91 | 122 | 148 | 169 |
| 150bp | 2 | 3 | 6 | 10 | 18 | 33 | 61 | 111 | 154 | 190 | 221 |
| 200bp | 2 | 3 | 6 | 10 | 19 | 34 | 67 | 121 | 177 | 223 | 262 |

---

### LONG vs SHORT Comparison (H=30, Target=15bp)

| Direction | Clean Win | Dirty Win | Never Hit |
|-----------|-----------|-----------|-----------|
| LONG | 1.4% | 57.3% | 41.3% |
| SHORT | 1.2% | 57.8% | 41.0% |

Confirms 50/50 market - LONG and SHORT have nearly identical outcomes.

---

### Key Findings

1. **Win Rate scales with horizon:**
   - 15bp: 19% (H=3) → 91% (H=600)
   - 100bp: 0.3% (H=3) → 49% (H=600)
   - 200bp: 0.1% (H=3) → 25% (H=600)

2. **Clean wins are RARE at ALL horizons:**
   - Never exceeds 4% for any target/horizon
   - Larger targets = rarer clean wins (200bp: 0.2% max)

3. **MAE (drawdown) scales with target AND horizon:**
   - 15bp/H=30: 7bp median drawdown
   - 100bp/H=240: 28bp median drawdown
   - 200bp/H=600: 45bp median drawdown

4. **Time in drawdown increases with target:**
   - 15bp: 3-9 bars in drawdown
   - 100bp: 3-48 bars in drawdown
   - 200bp: 3-47 bars in drawdown

5. **Time to target increases with target:**
   - 15bp at H=600: 15 bars (~15 min)
   - 100bp at H=600: 169 bars (~2.8 hours)
   - 200bp at H=600: 262 bars (~4.4 hours)

6. **LONG vs SHORT are symmetric** - confirms no directional bias

---

### Legacy Tables: Clean/Dirty/Never Hit by Individual Horizon

**H=3 bars:**
| Target | Clean Win % | Dirty Win % | Never Hit |
|--------|-------------|-------------|-----------|
| 8bp    | 4.2%        | 34.2%       | 61.6%     |
| 12bp   | 2.9%        | 22.8%       | 74.4%     |
| 15bp   | 2.2%        | 17.1%       | 80.7%     |
| 20bp   | 1.4%        | 11.0%       | 87.6%     |
| 25bp   | 0.9%        | 7.5%        | 91.6%     |
| 30bp   | 0.6%        | 5.3%        | 94.1%     |
| 40bp   | 0.3%        | 2.9%        | 96.8%     |
| 50bp   | 0.2%        | 1.7%        | 98.1%     |

**H=5 bars:**
| Target | Clean Win % | Dirty Win % | Never Hit |
|--------|-------------|-------------|-----------|
| 8bp    | 4.6%        | 43.0%       | 52.4%     |
| 12bp   | 3.3%        | 30.9%       | 65.8%     |
| 15bp   | 2.6%        | 24.3%       | 73.1%     |
| 20bp   | 1.7%        | 16.9%       | 81.4%     |
| 25bp   | 1.2%        | 12.0%       | 86.8%     |
| 30bp   | 0.9%        | 8.7%        | 90.4%     |
| 40bp   | 0.5%        | 5.1%        | 94.4%     |
| 50bp   | 0.3%        | 3.2%        | 96.5%     |

**H=10 bars:**
| Target | Clean Win % | Dirty Win % | Never Hit |
|--------|-------------|-------------|-----------|
| 8bp    | 5.0%        | 54.6%       | 40.4%     |
| 12bp   | 3.7%        | 43.0%       | 53.3%     |
| 15bp   | 3.0%        | 36.1%       | 60.9%     |
| 20bp   | 2.1%        | 27.1%       | 70.8%     |
| 25bp   | 1.6%        | 20.7%       | 77.7%     |
| 30bp   | 1.2%        | 16.0%       | 82.8%     |
| 40bp   | 0.8%        | 10.0%       | 89.2%     |
| 50bp   | 0.5%        | 6.7%        | 92.8%     |

**H=15 bars:**
| Target | Clean Win % | Dirty Win % | Never Hit |
|--------|-------------|-------------|-----------|
| 8bp    | 5.1%        | 60.9%       | 34.0%     |
| 12bp   | 3.8%        | 50.3%       | 45.9%     |
| 15bp   | 3.1%        | 43.5%       | 53.4%     |
| 20bp   | 2.3%        | 34.1%       | 63.6%     |
| 25bp   | 1.8%        | 27.0%       | 71.2%     |
| 30bp   | 1.4%        | 21.6%       | 77.0%     |
| 40bp   | 0.9%        | 14.4%       | 84.7%     |
| 50bp   | 0.6%        | 10.0%       | 89.4%     |

**H=30 bars:**
| Target | Clean Win % | Dirty Win % | Never Hit |
|--------|-------------|-------------|-----------|
| 8bp    | 5.1%        | 70.3%       | 24.6%     |
| 12bp   | 4.0%        | 61.5%       | 34.6%     |
| 15bp   | 3.3%        | 55.6%       | 41.1%     |
| 20bp   | 2.5%        | 46.7%       | 50.8%     |
| 25bp   | 2.0%        | 39.3%       | 58.7%     |
| 30bp   | 1.6%        | 33.2%       | 65.2%     |
| 40bp   | 1.1%        | 24.2%       | 74.7%     |
| 50bp   | 0.8%        | 18.0%       | 81.2%     |

**H=60 bars:**
| Target | Clean Win % | Dirty Win % | Never Hit |
|--------|-------------|-------------|-----------|
| 8bp    | 5.2%        | 77.5%       | 17.3%     |
| 12bp   | 4.0%        | 71.0%       | 25.0%     |
| 15bp   | 3.4%        | 66.2%       | 30.4%     |
| 20bp   | 2.5%        | 54.7%       | 42.8%     |
| 25bp   | 2.1%        | 51.9%       | 46.0%     |
| 30bp   | 1.7%        | 43.7%       | 54.6%     |
| 40bp   | 1.2%        | 33.2%       | 65.6%     |
| 50bp   | 1.0%        | 28.9%       | 70.1%     |

---

## ANALYSIS-5: Adverse Excursion (AE) - Drawdown Before Winning

**Question:** For trades that eventually WIN, how much drawdown did they experience first?

**Data:** Full dataset (2019-2025) - Verified: median MAE = 8bp for +15bp target at H=60

**🔗 Related:** See **ANALYSIS-4 TABLE 3 & 4** for consolidated MAE data across ALL horizons (H=3 to H=600) and ALL targets (12bp to 200bp).

**This analysis** provides detailed per-horizon breakdown with additional stats (Mean, Max, Count).

**AE = Maximum drawdown BEFORE hitting target (for LONG trades: how much price dropped below entry before winning)**

**Coverage:** H=3 to H=600 (11 horizons) | Targets: 12bp to 200bp (7 targets) | Sample: 200,000

### Detailed Per-Horizon Data

H=3 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 5.9bp   | 2.7bp     | 7.1bp   | 407.3bp  | 50,986  |
| Hit 15bp target      | 6.2bp   | 2.8bp     | 7.5bp   | 407.3bp  | 38,339  |
| Hit 25bp target      | 7.6bp   | 3.3bp     | 8.9bp   | 407.3bp  | 16,403  |
| Hit 50bp target      | 10.9bp  | 4.2bp     | 12.3bp  | 397.1bp  | 3,765   |
| Hit 100bp target     | 18.7bp  | 7.6bp     | 19.5bp  | 397.1bp  | 638     |
| Hit 150bp target     | 24.4bp  | 9.9bp     | 22.8bp  | 373.8bp  | 236     |
| Hit 200bp target     | 32.7bp  | 13.7bp    | 34.8bp  | 331.0bp  | 106     |

H=5 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 6.7bp   | 3.3bp     | 8.4bp   | 407.3bp  | 68,210  |
| Hit 15bp target      | 7.0bp   | 3.4bp     | 8.8bp   | 407.3bp  | 53,654  |
| Hit 25bp target      | 8.4bp   | 4.0bp     | 10.4bp  | 407.3bp  | 25,973  |
| Hit 50bp target      | 11.8bp  | 5.0bp     | 13.6bp  | 397.1bp  | 6,821   |
| Hit 100bp target     | 18.7bp  | 7.5bp     | 20.0bp  | 397.1bp  | 1,263   |
| Hit 150bp target     | 27.5bp  | 11.0bp    | 26.6bp  | 1266.6bp | 450     |
| Hit 200bp target     | 39.7bp  | 13.9bp    | 35.2bp  | 1266.6bp | 187     |

H=10 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 8.2bp   | 4.3bp     | 10.5bp  | 1082.1bp | 93,392  |
| Hit 15bp target      | 8.8bp   | 4.6bp     | 11.2bp  | 1082.1bp | 78,071  |
| Hit 25bp target      | 10.4bp  | 5.4bp     | 13.0bp  | 1082.1bp | 44,286  |
| Hit 50bp target      | 13.9bp  | 6.6bp     | 16.3bp  | 1082.1bp | 14,309  |
| Hit 100bp target     | 20.7bp  | 8.6bp     | 23.3bp  | 740.2bp  | 3,132   |
| Hit 150bp target     | 29.1bp  | 10.9bp    | 28.4bp  | 1266.6bp | 1,074   |
| Hit 200bp target     | 39.3bp  | 13.3bp    | 34.9bp  | 1266.6bp | 482     |

H=15 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 9.4bp   | 5.0bp     | 12.0bp  | 1082.1bp | 108,043 |
| Hit 15bp target      | 10.1bp  | 5.4bp     | 12.8bp  | 1082.1bp | 93,010  |
| Hit 25bp target      | 11.9bp  | 6.4bp     | 14.8bp  | 1082.1bp | 57,267  |
| Hit 50bp target      | 15.3bp  | 7.6bp     | 18.3bp  | 1082.1bp | 20,979  |
| Hit 100bp target     | 21.8bp  | 9.4bp     | 25.0bp  | 1082.1bp | 5,060   |
| Hit 150bp target     | 30.2bp  | 11.6bp    | 30.0bp  | 1266.6bp | 1,842   |
| Hit 200bp target     | 39.0bp  | 14.1bp    | 38.7bp  | 1266.6bp | 821     |

H=30 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 11.8bp  | 6.3bp     | 14.9bp  | 1082.1bp | 130,774 |
| Hit 15bp target      | 12.7bp  | 6.9bp     | 16.1bp  | 1082.1bp | 117,642 |
| Hit 25bp target      | 15.1bp  | 8.4bp     | 19.2bp  | 1082.1bp | 82,320  |
| Hit 50bp target      | 18.8bp  | 10.1bp    | 23.3bp  | 1082.1bp | 37,547  |
| Hit 100bp target     | 24.8bp  | 12.1bp    | 29.4bp  | 1082.1bp | 11,163  |
| Hit 150bp target     | 30.4bp  | 13.6bp    | 32.8bp  | 1266.6bp | 4,529   |
| Hit 200bp target     | 37.4bp  | 15.8bp    | 39.9bp  | 1266.6bp | 2,124   |

H=60 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 14.7bp  | 7.6bp     | 18.3bp  | 1595.5bp | 149,805 |
| Hit 15bp target      | 16.0bp  | 8.6bp     | 20.1bp  | 1595.5bp | 139,105 |
| Hit 25bp target      | 19.3bp  | 10.9bp    | 24.4bp  | 1595.5bp | 107,897 |
| Hit 50bp target      | 24.1bp  | 13.5bp    | 29.9bp  | 1595.5bp | 59,603  |
| Hit 100bp target     | 30.2bp  | 15.7bp    | 36.5bp  | 1601.3bp | 21,989  |
| Hit 150bp target     | 35.3bp  | 17.3bp    | 41.1bp  | 1601.3bp | 10,014  |
| Hit 200bp target     | 40.8bp  | 19.2bp    | 45.9bp  | 1601.3bp | 5,037   |

H=120 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 17.9bp  | 8.8bp     | 21.7bp  | 2063.5bp | 164,879 |
| Hit 15bp target      | 19.7bp  | 10.2bp    | 24.2bp  | 2063.5bp | 156,608 |
| Hit 25bp target      | 24.3bp  | 13.4bp    | 30.6bp  | 2063.5bp | 131,248 |
| Hit 50bp target      | 30.7bp  | 17.5bp    | 38.8bp  | 2063.5bp | 85,246  |
| Hit 100bp target     | 37.3bp  | 20.8bp    | 46.2bp  | 2062.0bp | 39,439  |
| Hit 150bp target     | 42.2bp  | 22.1bp    | 50.8bp  | 2062.0bp | 20,563  |
| Hit 200bp target     | 46.5bp  | 23.6bp    | 55.2bp  | 2062.0bp | 11,527  |

H=240 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 21.4bp  | 9.9bp     | 25.1bp  | 2063.5bp | 176,066 |
| Hit 15bp target      | 23.9bp  | 11.6bp    | 28.7bp  | 2063.5bp | 170,155 |
| Hit 25bp target      | 30.1bp  | 16.1bp    | 37.8bp  | 2063.5bp | 151,114 |
| Hit 50bp target      | 39.1bp  | 22.5bp    | 50.1bp  | 2330.5bp | 112,313 |
| Hit 100bp target     | 47.6bp  | 27.8bp    | 60.2bp  | 2330.5bp | 63,107  |
| Hit 150bp target     | 52.8bp  | 29.6bp    | 65.2bp  | 2413.0bp | 37,927  |
| Hit 200bp target     | 56.3bp  | 30.4bp    | 68.4bp  | 2413.0bp | 23,530  |

H=360 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 23.7bp  | 10.4bp    | 27.1bp  | 2259.6bp | 181,190 |
| Hit 15bp target      | 26.6bp  | 12.3bp    | 31.2bp  | 2259.6bp | 176,417 |
| Hit 25bp target      | 34.1bp  | 17.6bp    | 42.2bp  | 2354.1bp | 160,822 |
| Hit 50bp target      | 44.9bp  | 25.6bp    | 57.5bp  | 2354.1bp | 126,867 |
| Hit 100bp target     | 55.2bp  | 32.4bp    | 70.3bp  | 2354.1bp | 79,088  |
| Hit 150bp target     | 61.0bp  | 35.0bp    | 76.6bp  | 2413.0bp | 51,127  |
| Hit 200bp target     | 64.6bp  | 36.0bp    | 79.8bp  | 2413.0bp | 34,078  |

H=480 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 25.4bp  | 10.8bp    | 28.4bp  | 2565.9bp | 184,134 |
| Hit 15bp target      | 28.6bp  | 12.8bp    | 33.0bp  | 2565.9bp | 180,109 |
| Hit 25bp target      | 37.0bp  | 18.6bp    | 45.4bp  | 2544.4bp | 166,794 |
| Hit 50bp target      | 49.4bp  | 27.8bp    | 62.9bp  | 2544.4bp | 136,528 |
| Hit 100bp target     | 61.1bp  | 36.2bp    | 78.3bp  | 2440.8bp | 90,468  |
| Hit 150bp target     | 67.7bp  | 39.5bp    | 85.7bp  | 2440.8bp | 61,662  |
| Hit 200bp target     | 71.5bp  | 40.6bp    | 89.1bp  | 2429.1bp | 43,174  |

H=600 bars:
| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |
|----------------------|---------|-----------|---------|----------|---------|
| Hit 12bp target      | 26.7bp  | 11.0bp    | 29.4bp  | 2565.9bp | 186,266 |
| Hit 15bp target      | 30.1bp  | 13.1bp    | 34.4bp  | 2565.9bp | 182,736 |
| Hit 25bp target      | 39.3bp  | 19.4bp    | 47.9bp  | 2544.4bp | 170,925 |
| Hit 50bp target      | 53.0bp  | 29.7bp    | 67.2bp  | 2544.4bp | 143,661 |
| Hit 100bp target     | 66.1bp  | 39.4bp    | 85.2bp  | 2440.8bp | 99,553  |
| Hit 150bp target     | 73.3bp  | 43.7bp    | 93.9bp  | 2440.8bp | 70,356  |
| Hit 200bp target     | 76.8bp  | 45.0bp    | 97.1bp  | 2429.1bp | 50,914  |

### Summary Tables

**MAE MEDIAN (All Winners)**
| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp   | 3   | 3   | 4    | 5    | 6    | 8    | 9     | 10    | 10    | 11    | 11    |
| 15bp   | 3   | 3   | 5    | 5    | 7    | 9    | 10    | 12    | 12    | 13    | 13    |
| 25bp   | 3   | 4   | 5    | 6    | 8    | 11   | 13    | 16    | 18    | 19    | 19    |
| 50bp   | 4   | 5   | 7    | 8    | 10   | 13   | 18    | 22    | 26    | 28    | 30    |
| 100bp  | 8   | 7   | 9    | 9    | 12   | 16   | 21    | 28    | 32    | 36    | 39    |
| 150bp  | 10  | 11  | 11   | 12   | 14   | 17   | 22    | 30    | 35    | 39    | 44    |
| 200bp  | 14  | 14  | 13   | 14   | 16   | 19   | 24    | 30    | 36    | 41    | 45    |

**MAE 75TH PERCENTILE (All Winners)**
| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp   | 7   | 8   | 11   | 12   | 15   | 18   | 22    | 25    | 27    | 28    | 29    |
| 15bp   | 8   | 9   | 11   | 13   | 16   | 20   | 24    | 29    | 31    | 33    | 34    |
| 25bp   | 9   | 10  | 13   | 15   | 19   | 24   | 31    | 38    | 42    | 45    | 48    |
| 50bp   | 12  | 14  | 16   | 18   | 23   | 30   | 39    | 50    | 58    | 63    | 67    |
| 100bp  | 19  | 20  | 23   | 25   | 29   | 37   | 46    | 60    | 70    | 78    | 85    |
| 150bp  | 23  | 27  | 28   | 30   | 33   | 41   | 51    | 65    | 77    | 86    | 94    |
| 200bp  | 35  | 35  | 35   | 39   | 40   | 46   | 55    | 68    | 80    | 89    | 97    |

**MAE MAX (Worst Case - All Winners)**
| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 12bp   | 407 | 407 | 1082 | 1082 | 1082 | 1595 | 2063  | 2063  | 2260  | 2566  | 2566  |
| 15bp   | 407 | 407 | 1082 | 1082 | 1082 | 1595 | 2063  | 2063  | 2260  | 2566  | 2566  |
| 25bp   | 407 | 407 | 1082 | 1082 | 1082 | 1595 | 2063  | 2063  | 2354  | 2544  | 2544  |
| 50bp   | 397 | 397 | 1082 | 1082 | 1082 | 1595 | 2063  | 2331  | 2354  | 2544  | 2544  |
| 100bp  | 397 | 397 | 740  | 1082 | 1082 | 1601 | 2062  | 2331  | 2354  | 2441  | 2441  |
| 150bp  | 374 | 1267| 1267 | 1267 | 1267 | 1601 | 2062  | 2413  | 2413  | 2441  | 2441  |
| 200bp  | 331 | 1267| 1267 | 1267 | 1267 | 1601 | 2062  | 2413  | 2413  | 2429  | 2429  |

### Key Insights

1. **Even WINNING trades experience significant drawdowns**
   - Median MAE increases with horizon (more time = more drawdown risk)
   - Max MAE can be HUGE (1000-2500 bps) even for small targets

2. **Larger targets have larger MAE**
   - 15bp target: Median MAE 3-13 bp across horizons
   - 100bp target: Median MAE 7-40 bp across horizons
   - 200bp target: Median MAE 14-45 bp across horizons

3. **Longer horizons = Larger drawdowns**
   - At H=3: Most winners have small MAE (median 3-14 bp)
   - At H=600: Even winners have large MAE (median 11-45 bp)

4. **Max AE shows extreme risk**
   - Even winners can see 400-2566 bp drawdowns
   - High leverage would cause liquidation on trades that eventually win
   - Stop-loss placement is critical

5. **Sample size patterns**
   - Small targets (12-15bp): ~180,000 winners at H=600
   - Large targets (200bp): ~50,000 winners at H=600
   - Larger targets are harder to hit (fewer wins)

---

## ANALYSIS-6: Direction First (Real Moves Only)

**Question:** For moves ≥12bp (excluding noise), which direction hits the threshold first?

**Population:** Only bars with real moves (≥12bp in either direction)

**Compare to ANALYSIS-2:** ANALYSIS-2 includes all bars (with "Neither" category for noise).

ANALYSIS-6 excludes noise to see if real moves have directional bias.

**Coverage:** H=3 to H=600 (11 horizons) | Sample: 200,000 ✓

| Horizon | UP First | DOWN First | Ratio | Real Moves |
|---------|----------|------------|-------|------------|
| H=3     | 49.4%    | 50.6%      | 0.98  | 44.7%      |
| H=5     | 49.5%    | 50.5%      | 0.98  | 59.3%      |
| H=10    | 49.5%    | 50.5%      | 0.98  | 75.4%      |
| H=15    | 49.5%    | 50.5%      | 0.98  | 83.2%      |
| H=30    | 49.5%    | 50.5%      | 0.98  | 92.1%      |
| H=60    | 49.4%    | 50.6%      | 0.98  | 95.3%      |
| H=120   | 49.4%    | 50.6%      | 0.98  | 97.8%      |
| H=240   | 49.4%    | 50.6%      | 0.98  | 98.9%      |
| H=360   | 49.5%    | 50.5%      | 0.98  | 99.3%      |
| H=480   | 49.5%    | 50.5%      | 0.98  | 99.5%      |
| H=600   | 49.5%    | 50.5%      | 0.98  | 98.5%      |

**Key Insights:**

1. **Direction remains 50/50 even among real moves**
   - At H=3: 49.4% UP vs 50.6% DOWN (only 44.7% of bars have real moves)
   - At H=60: 49.4% UP vs 50.6% DOWN (95.3% of bars have real moves)
   - At H=600: 49.5% UP vs 50.5% DOWN (98.5% of bars have real moves)

2. **Filtering out noise does NOT provide directional edge**
   - ANALYSIS-2 (all bars including noise): ~50/50
   - ANALYSIS-6 (real moves only, excluding noise): Still ~50/50

3. **Ratio consistently 0.98 across all horizons**
   - No horizon provides inherent directional bias
   - Market is perfectly balanced even at 10 hours (H=600)

**Conclusion:** Market has no inherent directional bias. Need entry filters (state vector features like EMA, RSI, etc.) to find edge.

---

## ANALYSIS-7: Recovery Cases (Complete)

**Question:** When you enter a trade, what happens? How does it end?

**Data:** Full dataset (2019-2025) - Verified at H=60: 8bp=87% recovery, 15bp=77% recovery, 25bp=64% recovery

**Four cases:**
- Case 1: Wrong Direction (went below entry, never hit target even with 600 bars extended time)
- Case 2: Quick Recovery (went below entry, but hit target WITHIN H bars)
- Case 3: Slow Recovery (went below entry, didn't hit within H, but hit AFTER H with extended time)
- Case 4: Clean Win (hit target WITHOUT ever going below entry - perfect trade)

---

### Visual Example (LONG trade, Target = 15bp, H = 60 bars)

```
Case 4 (Clean Win):     Entry ────────────► Target hit (never went below entry)

Case 2 (Quick Rec):     Entry ──┐  ┌──────► Target hit within 60 bars
                                └──┘ (dip below entry, but recovered fast)

Case 3 (Slow Rec):      Entry ──┐     ┌───► Target hit at bar 150 (after H=60)
                                └─────┘ (dip below, didn't hit by bar 60, but hit later)

Case 1 (Wrong Dir):     Entry ──┐
                                └─────────► Never hits target (wrong direction)
```

---

### TABLE A1: Case 3 % (Slow Recovery - hit target AFTER H bars)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 8bp | 57.1% | 47.9% | 35.7% | 29.2% | 19.9% | 12.6% | 7.3% | 3.4% | 1.6% | 0.7% | 0.0% |
| 12bp | 67.6% | 59.0% | 46.3% | 38.7% | 27.2% | 17.9% | 10.6% | 5.0% | 2.4% | 1.0% | 0.0% |
| 15bp | 72.2% | 64.5% | 52.3% | 44.7% | 32.3% | 21.6% | 13.0% | 6.2% | 3.1% | 1.3% | 0.0% |
| 25bp | 77.1% | 72.3% | 63.1% | 56.4% | 43.9% | 31.4% | 19.7% | 9.8% | 5.0% | 2.0% | 0.0% |
| 50bp | 70.0% | 68.5% | 64.7% | 61.4% | 53.1% | 42.2% | 29.3% | 15.8% | 8.6% | 3.6% | 0.0% |
| 100bp | 49.4% | 49.1% | 48.1% | 47.1% | 44.0% | 38.6% | 29.9% | 18.3% | 10.2% | 4.6% | 0.0% |
| 150bp | 35.3% | 35.2% | 34.8% | 34.4% | 33.1% | 30.3% | 24.9% | 16.3% | 9.7% | 4.4% | 0.0% |
| 200bp | 25.6% | 25.6% | 25.4% | 25.2% | 24.6% | 23.0% | 19.8% | 13.8% | 8.5% | 4.0% | 0.0% |

---

### TABLE A2: Case 1 % (Wrong Direction - NEVER recovered even with 600 bars)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 8bp | 4.5% | 4.5% | 4.5% | 4.5% | 4.5% | 4.5% | 4.5% | 4.5% | 4.5% | 4.5% | 4.5% |
| 12bp | 6.8% | 6.8% | 6.8% | 6.8% | 6.8% | 6.8% | 6.8% | 6.8% | 6.8% | 6.8% | 6.8% |
| 15bp | 8.7% | 8.7% | 8.7% | 8.7% | 8.7% | 8.7% | 8.7% | 8.7% | 8.7% | 8.7% | 8.7% |
| 25bp | 14.5% | 14.5% | 14.5% | 14.5% | 14.5% | 14.5% | 14.5% | 14.5% | 14.5% | 14.5% | 14.5% |
| 50bp | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% |
| 100bp | 50.3% | 50.3% | 50.3% | 50.3% | 50.3% | 50.3% | 50.3% | 50.3% | 50.3% | 50.3% | 50.3% |
| 150bp | 64.6% | 64.6% | 64.6% | 64.6% | 64.6% | 64.6% | 64.6% | 64.6% | 64.6% | 64.6% | 64.6% |
| 200bp | 74.4% | 74.4% | 74.4% | 74.4% | 74.4% | 74.4% | 74.4% | 74.4% | 74.4% | 74.4% | 74.4% |

**Key Insight:** Case 1 % is CONSTANT across all horizons - it's fundamentally about direction, not timing.

---

### TABLE A3: Recovery Rate % (eventually hit target = Case 2 + 3 + 4)

| Target | Recovery Rate |
|--------|---------------|
| 8bp | 95.5% |
| 12bp | 93.2% |
| 15bp | 91.3% |
| 25bp | 85.5% |
| 50bp | 72.0% |
| 100bp | 49.7% |
| 150bp | 35.4% |
| 200bp | 25.6% |

**Key Insight:** Recovery rate is CONSTANT across all horizons (same as 100% - Case 1%).

---

### TABLE A4: Case 4 % (Clean Win - hit target without ANY drawdown)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|
| 8bp | 4.1% | 4.6% | 5.0% | 5.2% | 5.3% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% | 5.4% |
| 12bp | 2.8% | 3.3% | 3.7% | 3.8% | 3.9% | 4.0% | 4.0% | 4.0% | 4.0% | 4.0% | 4.0% |
| 15bp | 2.1% | 2.6% | 2.9% | 3.1% | 3.3% | 3.3% | 3.3% | 3.3% | 3.3% | 3.3% | 3.3% |
| 25bp | 0.9% | 1.2% | 1.6% | 1.8% | 2.0% | 2.1% | 2.2% | 2.2% | 2.2% | 2.2% | 2.2% |
| 50bp | 0.2% | 0.3% | 0.5% | 0.6% | 0.8% | 1.0% | 1.1% | 1.1% | 1.2% | 1.2% | 1.2% |
| 100bp | 0.0% | 0.0% | 0.1% | 0.1% | 0.2% | 0.3% | 0.4% | 0.5% | 0.5% | 0.5% | 0.5% |
| 150bp | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | 0.2% | 0.3% | 0.3% | 0.3% | 0.4% |
| 200bp | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | 0.2% | 0.2% | 0.2% | 0.2% |

**Key Insight:** Clean Wins are RARE (only 2-5% for small targets). Most winning trades experience some drawdown first.

---

### Summary: All 4 Cases at H=60

| Target | Case 1 (Wrong Dir) | Case 2 (Quick Rec) | Case 3 (Slow Rec) | Case 4 (Clean Win) | Total |
|--------|--------------------|--------------------|-------------------|--------------------| ------|
| 8bp | 4.5% | 77.5% | 12.6% | 5.4% | 100% |
| 12bp | 6.8% | 71.3% | 17.9% | 4.0% | 100% |
| 15bp | 8.7% | 66.4% | 21.6% | 3.3% | 100% |
| 25bp | 14.5% | 52.0% | 31.4% | 2.1% | 100% |
| 50bp | 28.0% | 28.8% | 42.2% | 1.0% | 100% |
| 100bp | 50.3% | 11.1% | 38.6% | 0.0% | 100% |
| 150bp | 64.6% | 5.0% | 30.3% | 0.1% | 100% |
| 200bp | 74.4% | 2.5% | 23.0% | 0.1% | 100% |

**Reading the table (15bp target at H=60):**
- 8.7% = Wrong direction, never recover
- 66.4% = Hit target within 60 bars (with some drawdown)
- 21.6% = Hit target AFTER 60 bars (slow recovery)
- 3.3% = Perfect trade, hit target with zero drawdown

**Key Findings:**
1. Most "Never Hit" trades are Case 3 (timing issue), not Case 1 (wrong direction)
2. At H=3, Target=15bp: 72% are Case 3 (would have won with more time)
3. Larger targets have more wrong direction cases (100bp+: ~50% wrong direction)
4. Small targets (8-25bp): ~85-95% eventually recover
5. Clean Wins (Case 4) are RARE: only 2-5% of trades

---

## ANALYSIS-8: Case 3 MAE (Drawdown for Slow Recovery Trades)

**Question:** For trades that hit target AFTER H bars, how much drawdown did they experience?

---

### TABLE B1: Case 3 MAE Median (bp drawdown before eventually hitting target)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|
| 8bp | 16 | 19 | 25 | 30 | 41 | 55 | 71 | 90 | 103 | 108 |
| 12bp | 17 | 20 | 25 | 30 | 40 | 53 | 68 | 86 | 97 | 101 |
| 15bp | 18 | 21 | 26 | 30 | 39 | 52 | 66 | 84 | 93 | 101 |
| 25bp | 22 | 24 | 28 | 32 | 39 | 49 | 63 | 80 | 90 | 98 |
| 50bp | 31 | 31 | 34 | 36 | 41 | 48 | 59 | 72 | 81 | 90 |
| 100bp | 39 | 39 | 40 | 41 | 44 | 49 | 57 | 68 | 78 | 86 |
| 150bp | 43 | 43 | 44 | 44 | 46 | 50 | 56 | 66 | 74 | 84 |
| 200bp | 45 | 45 | 45 | 45 | 46 | 49 | 53 | 62 | 69 | 78 |

---

### TABLE B2: Case 3 MAE 75th Percentile (worse 25% of slow recovery trades)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|
| 8bp | 37 | 43 | 54 | 62 | 78 | 98 | 124 | 152 | 172 | 172 |
| 12bp | 40 | 45 | 55 | 62 | 77 | 96 | 120 | 147 | 166 | 169 |
| 15bp | 43 | 48 | 56 | 63 | 76 | 94 | 118 | 146 | 162 | 169 |
| 25bp | 52 | 55 | 61 | 66 | 77 | 92 | 114 | 143 | 158 | 170 |
| 50bp | 68 | 69 | 72 | 75 | 82 | 93 | 111 | 135 | 147 | 157 |
| 100bp | 86 | 86 | 87 | 88 | 92 | 100 | 112 | 131 | 144 | 149 |
| 150bp | 94 | 94 | 95 | 95 | 98 | 103 | 113 | 129 | 141 | 149 |
| 200bp | 98 | 98 | 98 | 99 | 100 | 104 | 112 | 126 | 136 | 144 |

---

### TABLE B3: Case 3 MAE MAX (worst case drawdown in bp)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|
| 8bp | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 956 | 686 |
| 12bp | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 1038 | 686 |
| 15bp | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 1038 | 913 |
| 25bp | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 2315 | 1038 | 913 |
| 50bp | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 1266 |
| 100bp | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 1153 |
| 150bp | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 2443 | 1153 |
| 200bp | 2345 | 2345 | 2345 | 2345 | 2345 | 2345 | 2345 | 2345 | 1153 | 1153 |

**Key Insight:** Case 3 trades can experience MASSIVE drawdowns (up to 2400+ bp = 24%) before eventually recovering. This is why stop-losses are critical.

---

### MAE Summary by Case

| Case | Median MAE | 75th MAE | Meaning |
|------|------------|----------|---------|
| Case 2 (Quick Rec) | 3-12bp | 7-25bp | Small drawdown, quick recovery |
| Case 3 (Slow Rec) | 16-108bp | 37-170bp | Moderate-large drawdown, needs more time |
| Case 1 (Wrong Dir) | 150-180bp | - | Huge drawdown, wrong direction |

**Key:** If drawdown exceeds ~50bp, increasingly likely to be Case 1 (wrong direction).

---

## ANALYSIS-9: Recovery Time for Case 3 (Complete)

**Question:** For Case 3 trades, how long does it take from entry to finally hit target?

---

### TABLE C1: Case 3 Total Recovery Time - Median (bars from entry)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|
| 8bp | 16 | 22 | 36 | 49 | 84 | 140 | 228 | 356 | 458 | 534 |
| 12bp | 21 | 26 | 40 | 54 | 88 | 145 | 229 | 358 | 456 | 533 |
| 15bp | 25 | 31 | 45 | 57 | 91 | 147 | 232 | 360 | 456 | 533 |
| 25bp | 41 | 47 | 60 | 72 | 105 | 157 | 239 | 363 | 458 | 533 |
| 50bp | 88 | 92 | 102 | 112 | 139 | 185 | 257 | 373 | 462 | 537 |
| 100bp | 166 | 168 | 173 | 178 | 195 | 227 | 285 | 382 | 468 | 538 |
| 150bp | 219 | 220 | 222 | 226 | 237 | 260 | 307 | 393 | 468 | 541 |
| 200bp | 263 | 263 | 265 | 267 | 275 | 291 | 328 | 403 | 473 | 542 |

---

### TABLE C2: Case 3 Total Recovery Time - 90th Percentile (worst 10% of slow recovery trades)

| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 |
|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|
| 8bp | 155 | 183 | 231 | 268 | 329 | 398 | 471 | 533 | 567 | 586 |
| 12bp | 189 | 210 | 254 | 286 | 342 | 406 | 472 | 532 | 565 | 585 |
| 15bp | 213 | 232 | 272 | 301 | 354 | 412 | 477 | 533 | 566 | 585 |
| 25bp | 289 | 300 | 324 | 342 | 377 | 426 | 483 | 535 | 565 | 586 |
| 50bp | 393 | 396 | 405 | 412 | 433 | 464 | 501 | 544 | 569 | 587 |
| 100bp | 472 | 472 | 476 | 478 | 486 | 499 | 521 | 550 | 572 | 587 |
| 150bp | 503 | 504 | 505 | 506 | 510 | 517 | 533 | 555 | 573 | 588 |
| 200bp | 524 | 524 | 524 | 525 | 527 | 531 | 543 | 561 | 575 | 589 |

---

### Key Findings

1. **Case 3 takes MUCH longer than Case 2:**
   - At H=3, Target=15bp: Median Case 3 takes 25 bars total (vs 2 bars for winners within H)
   - At H=60, Target=15bp: Median Case 3 takes 147 bars total (~2.5 hours)

2. **Worst 10% of Case 3 trades take 3-10 hours:**
   - Target 15bp: 90th percentile = 213-585 bars (3.5-10 hours)
   - Target 100bp: 90th percentile = 472-587 bars (8-10 hours)

3. **Larger targets have longer recovery times:**
   - 15bp: Median 25-533 bars
   - 100bp: Median 166-538 bars
   - 200bp: Median 263-542 bars

4. **If you have patience, most trades eventually win:**
   - 91% of 15bp targets eventually hit (only 9% wrong direction)
   - But some take 10+ hours to recover

---

## ANALYSIS-10: Case 3 Time Patterns (Rules for Noise and Exits)

**Question:** When do Case 3 trades typically recover? Can we define rules for when to wait vs when to exit?

**📊 Complete Data:** See [case3_time_patterns_summary.md](case3_time_patterns_summary.md) for full analysis with ALL horizons (H=3 to H=600) and ALL targets (12bp to 200bp).

**🔗 Related:** See **ANALYSIS-7/8/9** for Case definitions and recovery data.

This analysis looks at WHEN and HOW Case 3 trades recover to help define:
1. Exit timeout rules (when to give up)
2. Drawdown exit rules (when MAE is too large)
3. Patience rules (when to wait longer)
4. Noise definition

---

### PATTERN 1: Recovery Time Windows (% of Case 3 trades that recover)

**At H=60, Target=15bp:**
| Time Window | % Recovered | Meaning |
|-------------|-------------|---------|
| Within 2*H (120 bars) | 40% | Fast recoveries |
| Within 3*H (180 bars) | 59% | Most recover here |
| Within 4*H (240 bars) | 71% | Good majority |
| Within 5*H (300 bars) | 79% | Almost all |
| After 5*H (>300 bars) | 21% | Very slow |

**At H=60, Target=25bp:**
| Time Window | % Recovered | Meaning |
|-------------|-------------|---------|
| Within 2*H (120 bars) | 37% | Fast recoveries |
| Within 3*H (180 bars) | 57% | Most recover here |
| Within 4*H (240 bars) | 69% | Good majority |
| Within 5*H (300 bars) | 77% | Almost all |
| After 5*H (>300 bars) | 23% | Very slow |

**At H=60, Target=50bp:**
| Time Window | % Recovered | Meaning |
|-------------|-------------|---------|
| Within 2*H (120 bars) | 30% | Fast recoveries |
| Within 3*H (180 bars) | 49% | About half |
| Within 4*H (240 bars) | 62% | Majority |
| Within 5*H (300 bars) | 72% | Most |
| After 5*H (>300 bars) | 28% | Significant minority |

---

### PATTERN 2: MAE Distribution for Case 3 Trades

**At H=60, Target=15bp:**
| MAE Range | % of Case 3 | Interpretation |
|-----------|-------------|----------------|
| < 30bp | 29% | Small drawdown, likely to recover |
| 30-50bp | 19% | Moderate drawdown |
| 50-100bp | 28% | Large drawdown, borderline |
| > 100bp | 23% | Huge drawdown, likely wrong direction |

**At H=60, Target=25bp:**
| MAE Range | % of Case 3 | Interpretation |
|-----------|-------------|----------------|
| < 30bp | 32% | Small drawdown |
| 30-50bp | 19% | Moderate drawdown |
| 50-100bp | 27% | Large drawdown |
| > 100bp | 22% | Huge drawdown |

**At H=60, Target=50bp:**
| MAE Range | % of Case 3 | Interpretation |
|-----------|-------------|----------------|
| < 30bp | 33% | Small drawdown |
| 30-50bp | 18% | Moderate drawdown |
| 50-100bp | 26% | Large drawdown |
| > 100bp | 23% | Huge drawdown |

---

### PATTERN 3: Complete Distribution (Percentiles)

**Recovery Time (bars from entry) - Target 15bp:**
| H | 10th | 25th | 50th | 75th | 90th |
|---|------|------|------|------|------|
| 3 | 5 | 10 | 25 | 81 | 216 |
| 5 | 8 | 13 | 31 | 93 | 235 |
| 10 | 14 | 21 | 45 | 120 | 273 |
| 15 | 20 | 29 | 58 | 143 | 302 |
| 30 | 37 | 51 | 92 | 195 | 355 |
| 60 | 71 | 91 | 147 | 267 | 413 |
| 120 | 135 | 164 | 232 | 354 | 477 |

**MAE (bp drawdown) - Target 15bp:**
| H | 10th | 25th | 50th | 75th | 90th |
|---|------|------|------|------|------|
| 3 | 3 | 7 | 19 | 44 | 88 |
| 5 | 3 | 9 | 21 | 48 | 94 |
| 10 | 5 | 12 | 26 | 57 | 106 |
| 15 | 6 | 14 | 30 | 63 | 116 |
| 30 | 9 | 19 | 40 | 77 | 136 |
| 60 | 13 | 26 | 52 | 95 | 160 |
| 120 | 18 | 35 | 66 | 118 | 191 |

---

### RULES DERIVED FROM PATTERNS

#### RULE 1: Exit Timeout Rule
**Exit if trade doesn't hit target within 5*H bars**
- At H=60: Exit if not hit by 300 bars (5 hours)
- Rationale: 75-80% of Case 3 trades recover within 5*H bars
- The remaining 20-25% take too long or never recover

**Example:**
- Target 15bp, H=60: Exit at bar 300 if not hit
- Target 25bp, H=60: Exit at bar 300 if not hit

---

#### RULE 2: Drawdown Exit Rule
**Exit if MAE exceeds drawdown threshold**

| Target | MAE Threshold | Rationale |
|--------|---------------|-----------|
| 15bp | 50bp | 3.3x target - if down this much, likely wrong |
| 25bp | 60bp | 2.4x target |
| 50bp | 100bp | 2.0x target |

At H=60, Target=15bp:
- If MAE < 50bp: 48% of Case 3 trades (likely to recover)
- If MAE > 50bp: 52% of Case 3 trades (borderline/wrong direction)

---

#### RULE 3: Combined Exit Rule (Time + Drawdown)
**Exit if BOTH conditions met:**
1. Time elapsed > 3*H bars AND
2. MAE > 50bp

**Rationale:** If it's taking too long (>3*H) AND drawdown is large (>50bp), likely wrong direction.

At H=60, Target=15bp:
- By 180 bars (3*H), 59% of Case 3 have recovered
- Those still in with MAE > 50bp are likely losers

---

#### RULE 4: Patience Rule (When to Wait)
**If at H bars and MAE < 30bp, WAIT - don't exit**

**Rationale:**
- At H=60, Target=15bp: 29% of Case 3 trades have MAE < 30bp
- These trades typically recover within 2-3*H bars (120-180 bars)
- Small drawdown suggests timing issue, not direction issue

**Action:** Wait up to 3*H total (180 bars) if MAE < 30bp

---

#### RULE 5: Noise Definition
**A move is "noise" if ANY of these:**
1. Doesn't hit target within 5*H bars, OR
2. MAE > 100bp (clearly wrong direction), OR
3. At 3*H bars with MAE > 50bp (taking too long with large drawdown)

**Use case:** This helps filter out non-tradeable conditions during entry

---

### Summary: Practical Exit Strategy

**For Target=15bp, H=60:**

```
At bar 60 (H):
├─ If MAE < 30bp → WAIT (likely Case 3, will recover)
│  └─ Check again at bar 120 (2*H)
│     ├─ Hit target → Win
│     └─ Not hit → Exit at bar 180 (3*H)
│
├─ If MAE 30-50bp → MONITOR
│  └─ Exit at bar 180 (3*H) if not hit
│
└─ If MAE > 50bp → EXIT NOW
   (Likely wrong direction)
```

**Key Numbers (Target=15bp, H=60):**
- Wait until: 180 bars (3*H) if MAE < 50bp
- Exit at: 180 bars if MAE > 50bp OR 300 bars (5*H) regardless
- Drawdown threshold: 50bp (exit immediately if exceeded)

---

## What This Means

Random entry into BTC 1-minute bars is not profitable, regardless of target/stop thresholds.

The current state vector (EMAs, RSI, ATR, volume, range position) provides zero predictive edge.

To be profitable, we need either:
1. Different features that actually predict direction
2. Different timeframe with more signal
3. Different entry logic (momentum, mean reversion at extremes)
4. Accept that 1-minute BTC scalping may not be viable

---

## The Trading Costs

Fees (limit orders): 8 bps round-trip
Minimum target must exceed 8 bps to be profitable.

**Important**: Target=8bp is structurally impossible after fees (8bp - 8bp = 0 net profit).

---

## ANALYSIS-11: EMA Bounce Magnitude (Multi-Timeframe, All EMAs)

**Question:** When price touches EMA, how much does it bounce? Which EMA is best for each timeframe?

**Data:** Full dataset - 3,150,249 1-minute candles (2019-12-31 to 2025-12-30)

**Methodology:**
- EMAs tested: 9, 20, 50, 100, 200
- Timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes
- EMA calculated on EACH timeframe's candles (correct method)
- Support test: Price from above touches EMA, measures bounce UP
- Resistance test: Price from below touches EMA, measures rejection DOWN
- Minimum 100 samples required for statistical reliability

### Best EMA by SUCCESS RATE (Highest bounce/rejection frequency)

**SUPPORT (Price touches EMA from above -> bounces UP):**

| Timeframe | Best EMA | Success Rate | Bounce (bp) | Correction (bp) | R:R | Sample |
|-----------|----------|--------------|-------------|-----------------|-----|--------|
| 3min | EMA9 | 72.9% | 32.1bp | 18.4bp | 1.74 | 408,689 |
| 5min | EMA9 | 78.6% | 38.1bp | 24.3bp | 1.57 | 221,063 |
| 10min | EMA9 | 85.0% | 50.2bp | 35.5bp | 1.41 | 92,195 |
| 15min | EMA9 | 87.8% | 59.3bp | 45.1bp | 1.32 | 54,272 |
| 30min | EMA9 | 91.7% | 81.8bp | 65.3bp | 1.25 | 21,184 |
| 60min | EMA50 | 95.1% | 116.5bp | 87.6bp | 1.33 | 2,522 |
| 120min | EMA9 | 97.1% | 187.9bp | 144.8bp | 1.30 | 2,645 |
| 240min | EMA9 | 98.0% | 282.0bp | 232.2bp | 1.21 | 904 |
| 480min | EMA20 | 99.3% | 219.4bp | 300.5bp | 0.73 | 142 |

**RESISTANCE (Price touches EMA from below -> rejects DOWN):**

| Timeframe | Best EMA | Success Rate | Drop (bp) | Correction (bp) | R:R | Sample |
|-----------|----------|--------------|-----------|-----------------|-----|--------|
| 3min | EMA9 | 73.2% | 32.8bp | 18.6bp | 1.77 | 395,806 |
| 5min | EMA9 | 78.5% | 38.6bp | 24.0bp | 1.61 | 211,119 |
| 10min | EMA9 | 84.7% | 50.0bp | 35.2bp | 1.42 | 87,160 |
| 15min | EMA9 | 87.6% | 58.8bp | 44.8bp | 1.31 | 50,679 |
| 30min | EMA9 | 91.4% | 82.4bp | 67.8bp | 1.22 | 19,529 |
| 60min | EMA50 | 94.2% | 103.2bp | 97.8bp | 1.05 | 2,564 |
| 120min | EMA9 | 96.6% | 167.0bp | 152.0bp | 1.10 | 2,550 |
| 240min | EMA20 | 98.0% | 237.9bp | 258.9bp | 0.92 | 494 |
| 480min | EMA9 | 98.9% | 346.8bp | 404.6bp | 0.86 | 275 |

### Best EMA by R:R RATIO (Largest bounce vs correction)

**SUPPORT:**

| Timeframe | Best EMA | R:R | Success Rate | Bounce (bp) | Correction (bp) |
|-----------|----------|-----|--------------|-------------|-----------------|
| 3min | EMA200 | 3.71 | 41.4% | 18.4bp | 5.0bp |
| 5min | EMA200 | 2.91 | 50.4% | 20.8bp | 7.1bp |
| 10min | EMA100 | 2.15 | 62.5% | 25.1bp | 11.6bp |
| 15min | EMA50 | 1.92 | 69.9% | 28.9bp | 15.1bp |
| 30min | EMA100 | 1.57 | 79.0% | 37.7bp | 24.0bp |
| 60min | EMA50 | 1.40 | 86.1% | 53.6bp | 38.4bp |
| 120min | EMA200 | 1.89 | 90.1% | 95.8bp | 50.7bp |
| 240min | EMA100 | 1.36 | 97.7% | 167.6bp | 122.9bp |
| 480min | EMA9 | 1.14 | 97.0% | 389.0bp | 340.2bp |

**RESISTANCE:**

| Timeframe | Best EMA | R:R | Success Rate | Drop (bp) | Correction (bp) |
|-----------|----------|-----|--------------|-----------|-----------------|
| 3min | EMA200 | 3.76 | 42.0% | 18.9bp | 5.0bp |
| 5min | EMA100 | 2.98 | 52.4% | 21.3bp | 7.2bp |
| 10min | EMA100 | 2.32 | 63.2% | 25.7bp | 11.1bp |
| 15min | EMA50 | 2.00 | 70.0% | 29.4bp | 14.7bp |
| 30min | EMA50 | 1.61 | 78.0% | 37.0bp | 23.0bp |
| 60min | EMA9 | 1.40 | 86.0% | 55.7bp | 39.7bp |
| 120min | EMA200 | 1.80 | 89.6% | 95.4bp | 53.1bp |
| 240min | EMA100 | 1.51 | 95.4% | 189.6bp | 125.7bp |
| 480min | EMA20 | 1.20 | 94.1% | 148.2bp | 123.7bp |

### Key Findings

1. **EMA9 dominates by success rate** across most timeframes (73-99%)
2. **EMA200 dominates by R:R ratio** on shorter timeframes (3.71-3.76 R:R)
3. **Critical trade-off:**
   - Short EMAs (9, 20): High success rate, lower R:R
   - Long EMAs (100, 200): Low success rate, higher R:R

4. **Bounce magnitude scales with timeframe:**
   - 3min: ~32bp bounce
   - 15min: ~59bp bounce
   - 60min: ~117bp bounce
   - 240min: ~282bp bounce

5. **Higher timeframes = Higher success but worse R:R:**
   - 3min: 73% success, 1.74 R:R
   - 60min: 95% success, 1.33 R:R
   - 240min: 98% success, 1.21 R:R

### Tradeable Assessment

| Timeframe | Net Move (Bounce - Corr) | After 8bp Fees | Tradeable? |
|-----------|--------------------------|----------------|------------|
| 3min | 32 - 18 = 14bp | 6bp | Marginal |
| 5min | 38 - 24 = 14bp | 6bp | Marginal |
| 10min | 50 - 36 = 14bp | 6bp | Marginal |
| 15min | 59 - 45 = 14bp | 6bp | Marginal |
| 30min | 82 - 65 = 17bp | 9bp | Marginal |
| 60min | 117 - 88 = 29bp | 21bp | Possible |

**Conclusion:** EMA bounce provides ~13bp net move on most timeframes, leaving only ~5bp after fees. 60-min timeframe shows best potential with ~19bp net after fees, but requires significant capital and patience.

---

## ANALYSIS-12: RSI Mean Reversion (Multi-Timeframe)

**Question:** Does RSI predict direction? (Testing mean reversion hypothesis)

**Data:** Full dataset - 3,150,249 1-minute candles (2019-12-31 to 2025-12-30)

**Methodology:**
- RSI calculated on EACH timeframe's candles (correct method)
- Timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes
- RSI periods: 7, 14, 21
- Oversold levels: <20, <30
- Overbought levels: >70, >80
- Horizons: 5, 10, 20 bars per timeframe

**Hypothesis:**
- RSI < 30 (oversold) -> Price should go UP (mean reversion)
- RSI > 70 (overbought) -> Price should go DOWN (mean reversion)

### Critical Finding: Oversold Works, Overbought Does NOT

| Type | Avg Edge | Best Edge | Works? |
|------|----------|-----------|--------|
| **Oversold** (RSI<30 -> UP) | **+4.2%** | **+12.6%** | YES |
| **Overbought** (RSI>70 -> DOWN) | **-0.5%** | +5.7% | NO |

**Key Discovery:** Overbought RSI does NOT cause reversal - momentum continues upward!

### Best Oversold Setups by Timeframe

| Timeframe | Best RSI | Level | H | Accuracy | Edge |
|-----------|----------|-------|---|----------|------|
| 3min | RSI14 | <20 | 10b | 56.6% | +6.6% |
| 5min | RSI14 | <20 | 5b | 56.2% | +6.2% |
| 10min | RSI14 | <30 | 5b | 56.0% | +6.0% |
| 15min | RSI7 | <30 | 5b | 55.7% | +5.7% |
| 30min | RSI7 | <20 | 5b | 56.3% | +6.3% |
| 60min | RSI7 | <30 | 5b | 56.1% | +6.1% |
| 120min | RSI14 | <30 | 10b | 55.0% | +5.0% |
| 240min | RSI21 | <20 | 5b | 57.0% | +7.0% |
| **480min** | **RSI21** | **<30** | **10b** | **62.6%** | **+12.6%** |

### Best Overbought Setups by Timeframe

| Timeframe | Best RSI | Level | H | Accuracy | Edge |
|-----------|----------|-------|---|----------|------|
| 3min | RSI14 | >80 | 5b | 55.7% | +5.7% |
| 5min | RSI7 | >80 | 5b | 55.0% | +5.0% |
| 10min | RSI21 | >80 | 5b | 54.0% | +4.0% |
| 15min | RSI14 | >70 | 5b | 53.8% | +3.8% |
| 30min | RSI7 | >80 | 5b | 53.2% | +3.2% |
| 60min | RSI7 | >80 | 5b | 52.3% | +2.3% |
| 120min | RSI7 | >70 | 5b | 50.6% | +0.6% |
| 240min | RSI7 | >70 | 5b | 51.2% | +1.2% |
| 480min | RSI7 | >70 | 20b | 47.9% | **-2.1%** |

**Note:** Overbought edge DECREASES at longer timeframes and becomes NEGATIVE!

### Top 5 Overall Setups

**Oversold (LONG signals):**
1. 480min RSI21 <30 H=10b: **62.6%** (+12.6% edge)
2. 480min RSI21 <30 H=20b: **62.6%** (+12.6% edge)
3. 480min RSI21 <20 H=5b: 59.4% (+9.4% edge)
4. 480min RSI21 <20 H=10b: 57.8% (+7.8% edge)
5. 480min RSI14 <30 H=20b: 57.1% (+7.1% edge)

### Tradeable Assessment

| Requirement | Value | Met? |
|-------------|-------|------|
| Need for 50bp target | 58%+ accuracy | |
| Best RSI oversold | 62.6% | YES |
| Best RSI overbought | 55.7% | NO |

**Conclusion:**
- **RSI Oversold on 480min (8-hour) timeframe is TRADEABLE** - 62.6% accuracy exceeds 58% requirement
- **RSI Overbought is NOT tradeable** - momentum continues, no reversal
- Only use RSI for LONG entries when oversold on higher timeframes

---

## ANALYSIS-13: ATR, Volume, Range Position (Data-Driven Validation)

**Question:** Do the remaining state vector features predict direction?

### Test 1: ATR (Volatility)

**Hypothesis:** High/Low volatility does NOT predict direction (it measures magnitude, not direction)

| Period | Avg Absolute Edge |
|--------|-------------------|
| Train (2020-2023) | 0.55% |
| Test 2024 | 0.85% |
| Test 2025 | 0.44% |

**Verdict:** NO directional edge. ATR measures volatility magnitude, not direction.

### Test 2: Volume

**Hypothesis:** High/Low volume does NOT predict direction (it measures activity, not direction)

| Period | Avg Absolute Edge |
|--------|-------------------|
| Train (2020-2023) | 0.99% |
| Test 2024 | 1.57% |
| Test 2025 | 0.46% |

**Verdict:** NO directional edge. Volume measures activity, not direction.

### Test 3: Range Position (Support/Resistance) - MULTI-TIMEFRAME RE-ANALYSIS

**Data:** Full dataset (2019-2025), 3.15M 1-minute candles
**Method:** Range Position calculated on EACH timeframe's candles (correct multi-timeframe approach)

**Hypothesis:** Near Low -> UP (support), Near High -> DOWN (resistance)

**Timeframes tested:** 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes
**Lookback periods:** 10, 20, 50 bars
**Support levels:** <10%, <20%, <30% (range position from low)
**Resistance levels:** >70%, >80%, >90% (range position from low)

#### SUPPORT ZONE (Range < threshold -> expect UP)

| Timeframe | Lookback | Level | H(bars) | Accuracy | Edge |
|-----------|----------|-------|---------|----------|------|
| 240min | 50b | <10% | 5b | **60.4%** | **+10.4%** |
| 120min | 50b | <10% | 10b | **60.2%** | **+10.2%** |
| 60min | 50b | <10% | 5b | **60.1%** | **+10.1%** |
| 480min | 50b | <10% | 5b | 59.6% | +9.6% |
| 60min | 50b | <10% | 10b | 59.3% | +9.3% |

**Average support edge: +5.8%**

#### RESISTANCE ZONE (Range > threshold -> expect DOWN)

| Timeframe | Lookback | Level | H(bars) | Accuracy | Edge |
|-----------|----------|-------|---------|----------|------|
| 5min | 10b | >90% | 5b | 56.6% | +6.6% |
| 5min | 10b | >90% | 10b | 55.9% | +5.9% |
| 3min | 10b | >90% | 5b | 55.6% | +5.6% |
| 10min | 10b | >90% | 5b | 55.4% | +5.4% |
| 3min | 10b | >90% | 10b | 55.4% | +5.4% |

**Average resistance edge: +1.6%**

#### KEY DISCOVERY: ASYMMETRIC BEHAVIOR

| Zone | Average Edge | Best Accuracy | Tradeable? |
|------|--------------|---------------|------------|
| **Support (near low)** | +5.8% | **60.4%** | **YES (>58%)** |
| Resistance (near high) | +1.6% | 56.6% | NO (<58%) |

**Pattern observed:** Similar to RSI findings:
- **Support works well** - price near lows tends to bounce UP
- **Resistance is weaker** - price near highs shows momentum continuation, not reversal
- At longer timeframes, resistance edge becomes NEGATIVE (momentum continues through)

**Verdict:** Support zone is TRADEABLE (60.4% > 58% threshold). Resistance is NOT reliable.

### Complete Feature Comparison (Updated with Multi-Timeframe Results)

| Feature | Best Setup | Accuracy | Edge | Tradeable? |
|---------|------------|----------|------|------------|
| **RSI Oversold** | 480min RSI21 <20, H=20b | 57.0% | +7.0% | Near threshold |
| **Range Support** | 240min LB50 <10%, H=5b | **60.4%** | **+10.4%** | **YES** |
| RSI Overbought | 5min RSI7 >80, H=5b | 54.9% | +4.9% | NO |
| Range Resistance | 5min LB10 >90%, H=5b | 56.6% | +6.6% | NO |
| EMA Proximity | Any | ~51% | ~1% | NO |
| ATR (Volatility) | - | ~50% | ~0.5% | NO |
| Volume | - | ~50% | ~0.5% | NO |

**Key insight from multi-timeframe analysis:**
- **Support/Oversold zones work** - price bounces from lows
- **Resistance/Overbought zones fail** - momentum continues through highs
- Only Range Support at higher timeframes exceeds 58% tradeability threshold

### Key Findings

1. **Range Position SUPPORT is the best directional predictor** (60.4% accuracy, +10.4% edge) - TRADEABLE
2. **RSI OVERSOLD is second best** (57.0% accuracy, +7.0% edge) - near threshold
3. **Overbought/Resistance zones DO NOT work** - momentum continues through highs
4. **EMA, ATR, and Volume have NO meaningful directional edge**

**Critical Pattern:** Mean reversion works for LOWS (support/oversold) but NOT for HIGHS (resistance/overbought)

### Implication for State Vector

The current 10D state vector uses features that mostly have NO predictive power:

| Current Feature | Predictive Power |
|-----------------|------------------|
| ema50_slope_z | None (EMA decayed) |
| ema200_slope_z | None (EMA decayed) |
| trend_alignment | None (derived from EMA) |
| return_5m_z | Untested |
| return_15m_z | Untested |
| **rsi_z** | **Partial** (oversold works, overbought does NOT) |
| atr_percentile | None |
| volume_z | None |
| vwap_distance_z | Untested |
| **range_position** | **Partial** (support works, resistance does NOT) |

**Only 2 of 10 features have demonstrated ANY predictive power, and only in ONE direction:**
- RSI: Only oversold (<30) predicts UP
- Range Position: Only support (<10-20%) predicts UP

**Trading implication:** These features only work for LONG entries from oversold/support conditions. They do NOT work for SHORT entries from overbought/resistance conditions.

---

## ANALYSIS-14: Extended Horizon Analysis (REMOVED)

**Status:** Superseded by **ANALYSIS-12** and **ANALYSIS-13** multi-timeframe analyses.

**Reason:** The original analysis tested horizons on 1-minute bars only (wrong approach). ANALYSIS-12/13 now cover both scalping (short timeframes) and day trading (60-120min timeframes) with proper multi-timeframe indicator calculation.

---

## Scripts Created for This Analysis

debug_horizon_grid_search.py - Tests all horizon/target/stop combinations
debug_raw_price_paths.py - Analyzes raw price movement without preset thresholds
debug_outcome_based_noise.py - Compares features between WIN and LOSS bars
debug_similarity_noise.py - Tests if similar states have consistent outcomes
debug_structural_noise.py - Tests magnitude-based noise filtering
debug_all_noise_types.py - Tests multiple noise types together
debug_path_analysis.py - Analyzes clean wins, adverse excursion, noise vs real moves
debug_mae_analysis.py - MAE analysis per horizon and overall
debug_recovery_analysis.py - 3-case recovery analysis (wrong direction vs timing)
debug_recovery_mae.py - MAE by case (how much drawdown for each case)
debug_recovery_time.py - Recovery time for Case 3 (how long to recover)
test_ema_bounce_comprehensive.py - EMA support/resistance pattern test (672 combinations)
test_ema_bounce_validation.py - EMA out-of-sample validation on 2024 data
test_ema_bounce_validation_2025.py - EMA validation on 2025 data
test_rsi_comprehensive.py - RSI mean reversion test (400 combinations, train data)
test_rsi_validation.py - RSI out-of-sample validation on 2024 data
test_rsi_validation_2025.py - RSI validation on 2025 data
test_remaining_features_fast.py - ATR, Volume, Range Position directional tests
test_atr_only.py - ATR extended horizon test (H=3 to H=600)
test_volume_only.py - Volume extended horizon test (H=3 to H=600)
test_range_position_only.py - Range Position extended horizon test (H=3 to H=600)
test_rsi_extended.py - RSI extended horizon test (H=3 to H=600)
test_ema_extended.py - EMA extended horizon test (H=3 to H=600)
test_extended_horizons_analysis123.py - Extended horizons for ANALYSIS-1, 2, 3
test_all_features_extended.py - All features with H=360, H=480 included
test_noise_filtering_extended.py - Noise filtering with extended horizons (removed)
analysis_ema_bounce_magnitude.py - EMA bounce/drop magnitude analysis V1 (ANALYSIS-11)
analysis_ema_bounce_v2.py - EMA bounce/rejection CORRECTED analysis (ANALYSIS-11 V2)
analysis_ema_bounce_v3_backtest.py - Path sequence + P&L backtest (FLAWED - look-ahead bias)
analysis_ema_bounce_v4_clean_backtest.py - Clean backtest without look-ahead bias
analysis_ema_bounce_v5_optimized.py - Consecutive support filter (numba optimized)
analysis_ema_bounce_v6_market_driven.py - Market-driven target/stop from measured data
analysis_ema_bounce_v7_proper.py - Proper EMA slope + touch + engulfing detection
analysis_ema_bounce_v8_diagnostic.py - Diagnostic analysis (raw edge, fee sensitivity)

---

## ANALYSIS-15: EMA Bounce BACKTEST Results (V3-V8)

The theoretical edge from V2 (1.5-2:1 R:R, 75-80% success) was tested with actual backtests.

### V4: Clean Backtest (No Look-Ahead Bias)

**Entry:** At every EMA touch when price came from above (support test)
**Result:** 0/825 combinations profitable

| EMA | H | Best Avg PnL |
|-----|---|-------------|
| EMA50 | 30 | -7.5bp |
| EMA100 | 30 | -7.8bp |

**Problem:** Entering at EVERY touch loses money due to fees.

### V5: Consecutive Support Filter

**Entry:** Only when same EMA had successful bounce within lookback period
**Logic:** If EMA worked as support recently, more likely to work again

**Result:** 0/1200 combinations profitable

| Strategy | Trades | Win% | Avg PnL |
|----------|--------|------|---------|
| Unfiltered | 10,789 | 49.9% | -7.68bp |
| Filtered (LB=60) | 7,221 | 50.0% | -7.58bp |

**Conclusion:** Filter reduces trades but doesn't improve profitability.

### V6: Market-Driven Parameters

**Entry:** Target = 90% of measured bounce, Stop = measured correction
**Logic:** Use actual market behavior as parameters

**Result:** 0/38 profitable combinations

All combinations lose approximately 7-8bp per trade (the fee amount).

### V7: Proper Detection (EMA Slope + Engulfing)

**Added filters:**
1. EMA must be sloping UP for LONG (uptrend)
2. Low touches or goes slightly below EMA (proper touch)
3. Bullish engulfing pattern at touch point
4. Close above EMA after touch (bounce confirmed)

**Result:** 0/27 profitable combinations

| Test | Profitable | Best Avg PnL |
|------|------------|--------------|
| Without Engulfing | 0/27 | -7.47bp |
| With Engulfing | 0/27 | -7.57bp |

### V8: DIAGNOSTIC - Root Cause Analysis

**Purpose:** Understand WHY proper detection still doesn't work

#### Test 1: Raw Edge (Zero Fees)

| EMA | H | Trades | Avg Max Gain | Avg Max Loss | Avg Final PnL | Edge? |
|-----|---|--------|--------------|--------------|---------------|-------|
| EMA50 | 10 | 78,724 | 15.1bp | 14.7bp | **+0.11bp** | YES |
| EMA50 | 30 | 41,416 | 28.9bp | 29.2bp | **-0.07bp** | NO |
| EMA100 | 10 | 59,647 | 14.3bp | 14.2bp | **+0.09bp** | YES |
| EMA100 | 30 | 32,923 | 27.8bp | 28.0bp | **+0.03bp** | YES |

**KEY FINDING:** Raw edge is minimal (<0.5bp per trade) - BEFORE fees!

#### Test 2: Outcome Distribution

EMA50, H=30 (no engulfing):
- **Winners: 49.3%** (direction essentially random)
- **Losers: 50.7%**
- Mean P&L: -0.07bp (before fees)

P&L Distribution:
- 25th percentile: -16.5bp
- 50th percentile: -0.4bp
- 75th percentile: +16.1bp

**Symmetric distribution** - no directional edge.

#### Test 3: Target/Stop Fee Sensitivity

| Target | Stop | Win% | 0bp Fee | 4bp Fee | 8bp Fee |
|--------|------|------|---------|---------|---------|
| 35bp | 10bp | 23.8% | **+4.12bp** | +0.12bp | -3.88bp |
| 30bp | 10bp | 27.7% | **+3.94bp** | -0.06bp | -4.06bp |
| 25bp | 10bp | 32.2% | **+3.56bp** | -0.44bp | -4.44bp |
| 20bp | 10bp | 37.3% | **+2.88bp** | -1.12bp | -5.12bp |

**CRITICAL FINDING:**
- **At 0bp fees:** Strategy profitable (+4bp avg with T=35/S=10)
- **At 4bp fees:** Break-even (+0.12bp)
- **At 8bp fees:** All combinations lose money

#### Test 4: Longer Holding Period

| EMA | H | Win% | Avg PnL | After 8bp Fees |
|-----|---|------|---------|----------------|
| EMA50 | 240 | 50.5% | +3.07bp | **-4.93bp** |
| EMA100 | 240 | 50.0% | +2.38bp | **-5.62bp** |

Even at H=240 (4 hours), edge is +2-3bp which is eaten by 8bp fees.

### Root Cause Analysis

| Observation | Finding |
|-------------|---------|
| Direction accuracy | 49-50% (random) |
| Raw edge (0 fee) | +0.1 to +4bp depending on T/S |
| Max gain vs max loss | Nearly symmetric |
| Fee impact | 8bp kills all edge |

### Why V2 Analysis Showed Edge But Backtest Doesn't

| V2 Analysis | Backtest Reality |
|-------------|------------------|
| Measured AFTER bounce confirmed | Must ENTER before knowing if bounce confirms |
| Bounce rate 75-80% | Applies to confirmed bounces only |
| R:R 1.5-2:1 | Only for bounces that worked |
| Selection bias | We only measured successful cases |

**V2 committed SURVIVORSHIP BIAS** - measured characteristics of winners only.

### CONCLUSION

1. **EMA bounce strategy has MARGINAL raw edge** (~+4bp at optimal T/S)
2. **8bp round-trip fees completely eliminate the edge**
3. **Strategy becomes viable at 4bp fees** (maker orders)
4. **Direction prediction is essentially random** (49-50% win rate)
5. **The edge comes from R:R asymmetry**, not direction prediction

### To Make Strategy Profitable

| Approach | Requirement |
|----------|-------------|
| Lower fees | Use 2bp maker + 2bp maker = 4bp total |
| Better filter | Find additional signals that improve >50% win rate |
| Larger targets | Use T=35bp+ with tight S=10bp stop |
| Discretion | Don't enter every signal - only "clean" setups |

---

## CORRECTED ANALYSIS: Multi-Timeframe Re-Analysis (2026-01-15)

**IMPORTANT:** Previous analyses (ANALYSIS-4, 10-15) used 1-minute data with "horizon" interpreted as bars ahead, but indicators (EMA, RSI, ATR) were always calculated on 1-minute candles. This was INCORRECT.

**Correct interpretation:** Horizon = Timeframe. EMA must be calculated on EACH timeframe's candles.

### CORRECTED ANALYSIS 10-14: Indicator Predictive Power

Tested indicators on timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes.

**Cohen's d by Timeframe (Predictive Power):**

| Timeframe | RSI | EMA20 Dist | ATR Pctl | Volume | Range Pos |
|-----------|-----|------------|----------|--------|-----------|
| 3min | -0.14 | -0.12 | 0.01 | 0.01 | -0.15 |
| 5min | -0.15 | -0.13 | 0.01 | 0.01 | -0.17 |
| 10min | -0.14 | -0.11 | 0.01 | 0.01 | -0.16 |
| 15min | -0.14 | -0.11 | 0.02 | 0.01 | -0.16 |
| 30min | -0.15 | -0.11 | 0.03 | 0.01 | -0.18 |
| 60min | -0.08 | -0.09 | 0.02 | 0.01 | -0.14 |
| 120min | -0.08 | -0.05 | 0.05 | 0.02 | -0.10 |
| 240min | -0.09 | -0.08 | 0.02 | 0.01 | -0.11 |
| 480min | +0.03 | +0.02 | 0.06 | -0.02 | -0.04 |

**Interpretation:** |d| < 0.2 = negligible predictive power

**Key Findings:**
1. **ALL indicators have negligible predictive power** on ALL timeframes
2. **RSI shows weak mean reversion** (RSI<20: 56-58% UP, RSI>80: 44-46% UP)
3. **Range Position has strongest signal** (-0.15 to -0.18) but still negligible
4. **Volume and ATR have zero predictive power** (d < 0.03)

### CORRECTED ANALYSIS-11: EMA Bounce Magnitude (SUPERSEDED)

**See ANALYSIS-11 for comprehensive multi-timeframe EMA bounce analysis with all 5 EMAs (9, 20, 50, 100, 200) tested.**

**Summary from ANALYSIS-11:**
- Higher timeframes = Higher success rate (75% at 3min → 98% at 240min)
- Higher timeframes = Lower R:R ratio (1.65 at 3min → 0.98 at 240min)
- EMA9 dominates by success rate, EMA200 dominates by R:R ratio
- Net move after fees: ~5bp on most timeframes (marginal)

### CORRECTED ANALYSIS-15: EMA Bounce Backtest (V10)

Tested EMA bounce backtest on timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes.

**Result: 0 / 2,880 combinations profitable across ALL timeframes**

| Timeframe | Best Avg PnL | Best EMA | Target/Stop |
|-----------|--------------|----------|-------------|
| 3min | -7.37bp | EMA200 | T50/S20 |
| 5min | **-6.84bp** | EMA200 | T50/S30 |
| 10min | -7.46bp | EMA100 | T50/S20 |
| 15min | -7.17bp | EMA200 | T50/S30 |
| 30min | -8.24bp | EMA100 | T50/S30 |
| 60min | -9.48bp | EMA50 | T50/S15 |
| 120min | -10.71bp | EMA20 | T50/S10 |
| 240min | -12.17bp | EMA100 | T50/S10 |
| 480min | -11.60bp | EMA200 | T50/S15 |

**Key Finding:** Higher timeframes perform WORSE, not better (5-min best at -6.84bp, 240-min worst at -12.17bp)

### Summary: What Changed After Correction

| Aspect | Previous (Wrong) | Corrected |
|--------|------------------|-----------|
| EMA calculation | Always on 1-min candles | On each timeframe's candles |
| Indicator values | Same across "horizons" | Different for each timeframe |
| Predictive power | d < 0.06 (1-min only) | d < 0.18 (all timeframes) |
| EMA bounce success | Varied by "horizon" | Varies by timeframe |
| R:R ratio | ~1.5-2.0 | 0.98-1.79 (timeframe dependent) |
| Backtest result | 0 profitable (1-min) | 0 profitable (all timeframes) |

**CONCLUSION:** The corrected multi-timeframe analysis confirms:
1. **No indicator has predictive power** on any timeframe
2. **EMA bounce has high success rate** (especially on higher TFs)
3. **BUT R:R ratio decreases** as timeframe increases
4. **8bp fees eliminate all edge** on every timeframe tested

---

## W-MFE: MFE Distribution Analysis (2026-01-16)

**Question:** How far does price go IN OUR FAVOR (MFE) before reversal, by case?

**Parameters:** Targets: 12-50bp, Horizons: 5-120 bars

### MFE Distribution by Case (T=25bp, H=30, Train)

| Case | Count | MFE Median | MFE P75 | MFE P90 | MFE P95 | MFE P99 |
|------|-------|------------|---------|---------|---------|---------|
| Case 0 | 41,571 | 29.2bp | 34.4bp | 43.8bp | 53.2bp | 85.6bp |
| Case 1 | 339,912 | 12.6bp | 18.8bp | 22.5bp | 23.7bp | 24.7bp |
| Case 2 | 825,049 | 29.5bp | 34.9bp | 44.3bp | 53.8bp | 84.9bp |
| Case 3 | 889,123 | 29.1bp | 34.4bp | 44.6bp | 54.8bp | 86.4bp |

### MFE vs MAE Comparison (T=25bp, H=30)

| Case | MFE P95 | MAE P95 | Ratio (MFE/MAE) | Interpretation |
|------|---------|---------|-----------------|----------------|
| Case 0 | 53bp | 0bp | - | Clean wins, no drawdown |
| Case 2 | 54bp | 53bp | 1.02x | Equal upside/downside |
| Case 3 | 55bp | 180bp | 0.30x | **3x more downside than upside** |

### Key Findings

1. **Case 1 never sees much upside** - MFE median = 12.6bp (target was 25bp)
2. **Case 2 & 3 had profit available** - MFE median ~29bp (above 25bp target)
3. **Case 3 is dangerous** - MFE/MAE ratio = 0.30x (much more downside risk)
4. **Train vs Test consistent** - ~10bp reduction in MFE P95 for 2024 data

**Implication:** Case 2 & 3 are timing issues, not direction issues - the profit opportunity was there.

---

## W-EXP1: Range Expansion Analysis (2026-01-16)

**Question:** Does entry bar range (volatility) predict case outcomes?

**Parameters:** Targets: 12-50bp, Horizons: 5-120 bars

### Range Expansion by Case (T=25bp, H=30, Train)

| Case | Count | Entry Range | Future Range | Expansion Ratio |
|------|-------|-------------|--------------|-----------------|
| Case 0 | 41,571 | 9.1bp | 10.8bp | 1.21x |
| Case 1 | 339,912 | 5.2bp | 6.0bp | 1.18x |
| Case 2 | 825,049 | 10.9bp | 11.9bp | 1.11x |
| Case 3 | 889,123 | 6.1bp | 6.5bp | 1.09x |

### **CRITICAL FINDING: Entry Range Predicts Case 1**

| Entry Range Quartile | Range | Case 0 % | Case 1 % | Case 2 % | Case 3 % |
|---------------------|-------|----------|----------|----------|----------|
| Q1 (Low) | 0-4bp | 1.6% | **26.0%** | 17.0% | 55.4% |
| Q2 | 4-8bp | 1.8% | 16.9% | 32.8% | 48.6% |
| Q3 | 8-13bp | 2.0% | 12.9% | 45.5% | 39.6% |
| Q4 (High) | 13+bp | 2.6% | **9.1%** | 62.2% | 26.2% |

### Key Findings

1. **Low entry range = HIGH Case 1 risk** - Q1 (0-4bp): 26% Case 1
2. **High entry range = LOW Case 1 risk** - Q4 (13+bp): 9.1% Case 1
3. **~3x difference** in structural failure rate based on entry volatility
4. **High volatility favors Case 2** (quick recovery) - 62.2% at Q4
5. **Low volatility favors Case 3** (slow recovery) - 55.4% at Q1

**Actionable Rule:** Avoid entries when bar range < 4bp (26% Case 1 risk)

### OOS Validation (2024-2025 data)

| Quartile | Train P(Case1) | Test P(Case1) | Diff | Status |
|----------|----------------|---------------|------|--------|
| Q1 (0-4bp) | 26.0% | 25.4% | -0.6pp | HOLDS |
| Q4 (13+bp) | 9.1% | 11.9% | +2.8pp | HOLDS |
| **Effect (Q1-Q4)** | **16.9pp** | **13.5pp** | - | **VALID** |

**Validation: PASSED** - Pattern holds on 2024-2025 data. Q1 still has highest Case 1 rate, Q4 still has lowest.

---

## W-EXP2: Move Continuation Analysis (2026-01-16)

**Question:** Does initial move direction (first bar after entry) predict case outcomes?

**Parameters:** Targets: 12-50bp, Horizons: 5-120 bars

### Case Distribution by Initial Move (T=25bp, H=30, Train)

| Initial Move | Case 0 % | Case 1 % | Case 2 % | Case 3 % | Count |
|--------------|----------|----------|----------|----------|-------|
| UP (>3bp) | 6.2% | **9.0%** | 56.2% | 28.6% | 569,592 |
| DOWN (<-3bp) | 0.0% | **16.3%** | 39.3% | 44.4% | 573,845 |
| FLAT (-3 to +3bp) | 0.7% | 20.5% | 29.3% | 49.5% | 952,218 |

### Strong Moves Analysis (T=25bp, H=30)

| Strong Move | Case 0 % | Case 1 % | Case 2 % | Case 3 % | Count |
|-------------|----------|----------|----------|----------|-------|
| Strong UP (>10bp) | 11.3% | **3.6%** | 73.6% | 11.6% | 167,862 |
| Strong DOWN (<-10bp) | 0.0% | 14.9% | 45.8% | 39.3% | 167,006 |

### Success Rate by Initial Move (H=30, Train)

| Target | UP Success% | DOWN Success% | FLAT Success% |
|--------|-------------|---------------|---------------|
| 12bp | 87.4% | 59.6% | 56.0% |
| 15bp | 81.4% | 54.0% | 48.3% |
| 20bp | 71.5% | 46.0% | 37.9% |
| 25bp | **62.4%** | 39.3% | 30.0% |
| 30bp | 54.4% | 33.6% | 24.0% |
| 50bp | 31.9% | 18.8% | 11.1% |

### Key Findings

1. **UP initial move = Lower Case 1** - 9.0% vs 16.3% for DOWN (+7.3pp difference)
2. **Strong UP (>10bp) = Very low Case 1** - Only 3.6% structural failure
3. **FLAT initial = Worst outcome** - 20.5% Case 1 (avoid!)
4. **Success rate (Case 0+2):** UP = 62.4%, DOWN = 39.3%, FLAT = 30.0%
5. **Train vs Test consistent** - Pattern holds in 2024 data

**Actionable Rules:**
- Wait for UP initial move before confirming entry
- Strong UP (>10bp) = highest confidence (only 3.6% Case 1)
- Avoid FLAT conditions (20.5% Case 1)

### OOS Validation (2024-2025 data)

| Initial Move | Train P(Case1) | Test P(Case1) | Diff | Status |
|--------------|----------------|---------------|------|--------|
| Strong UP (>10bp) | 3.6% | 5.6% | +2.0pp | HOLDS |
| UP (>3bp) | 9.0% | 12.4% | +3.4pp | HOLDS |
| DOWN (<-3bp) | 16.3% | 20.8% | +4.5pp | HOLDS |
| **Effect (DOWN-UP)** | **7.3pp** | **8.4pp** | - | **VALID** |

**Validation: PASSED** - Pattern holds on 2024-2025 data. UP still better than DOWN, Strong UP still best.

---

## WHAT Phase Summary (Complete)

### Ground Truth Established

| Finding | Value | Implication |
|---------|-------|-------------|
| Direction | 50/50 | No directional edge possible |
| Case 1 baseline | ~16% | Some trades structurally unrecoverable |
| Case 2 & 3 | ~80% | Most losses are timing, not direction |
| MFE vs MAE Case 3 | 0.30x | Case 3 has 3x more downside than upside |

### Actionable Filters Discovered (OOS Validated)

| Filter | Condition | Train Case1 | Test Case1 | OOS Valid? |
|--------|-----------|-------------|------------|------------|
| Entry Range Q4 | >13bp | 9.1% | 11.9% | **YES** |
| Entry Range Q1 | <4bp | 26.0% | 25.4% | **YES** |
| Initial Move UP | >3bp | 9.0% | 12.4% | **YES** |
| Initial Move Strong UP | >10bp | 3.6% | 5.6% | **YES** |
| Initial Move FLAT | -3 to +3bp | 20.5% | 23.0% | **YES** |

**All filters validated on 2024-2025 data.** Patterns hold with similar effect sizes.

### Next Phase: WHEN

WHEN phase will use these ground truth measurements to build probabilistic filters for Case 1 risk.
