# EXP-001: RSI Mean Reversion Behavior Analysis

## What we're testing
Does price reverse when RSI hits extreme levels? How reliably, how fast, and how much?

## Hypothesis
When RSI goes oversold (<20/30), price should bounce UP.
When RSI goes overbought (>70/80), price should drop DOWN.

## Parameters
See config.yaml
- RSI periods: 7, 14, 21
- Thresholds: 20/80, 30/70, 40/60
- Horizons: 3, 5, 10, 15, 20 bars
- In-sample: 2020-2023, Out-of-sample: 2024-2025

## Results

### Initial Finding (bounce rate)
- RSI(14) < 20 oversold: 99.7% bounce rate at H=10
- RSI(14) > 80 overbought: 99% drop rate at H=10
- All RSI periods/thresholds show 98-99% reversal rate
- BUT: "bounce" just means the HIGH went above entry at any point - misleading metric

### Deep Analysis: RSI Depth vs Outcome (OOS 2024-2025)

**OVERSOLD (RSI < 20, 596 signals):**

| RSI Depth | N | Bounced | Bounce 1 bar | Max Up (med) | Drawdown (med) |
|---|---|---|---|---|---|
| 15-20 (barely) | 354 (59%) | 98% | 70% | +115 bps | -82 bps |
| 10-15 (moderate) | 169 (28%) | 85% | 39% | +86 bps | -126 bps |
| 5-10 (deep) | 59 (10%) | 80% | 22% | +74 bps | -156 bps |
| 0-5 (extreme) | 14 (2%) | 64% | 7% | +77 bps | -323 bps |

**OVERBOUGHT (RSI > 80, 664 signals):**

| RSI Peak | N | Dropped | Drop 1 bar | Max Down (med) | Drawdown (med) |
|---|---|---|---|---|---|
| 80-85 (barely) | 375 (57%) | 97% | 70% | +116 bps | -87 bps |
| 85-90 (moderate) | 209 (31%) | 90% | 43% | +92 bps | -117 bps |
| 90-95 (deep) | 69 (10%) | 74% | 25% | +59 bps | -151 bps |
| 95-100 (extreme) | 10 (2%) | 60% | 20% | +20 bps | -184 bps |

### Overall bounce/drop stats (RSI(14) < 20, OOS):
- Median time to bounce: 1 bar (15 min)
- 54.7% bounce within 1 bar, 74.2% within 3 bars
- 8.2% never bounce within 50 bars (12.5 hours)
- Median first bounce magnitude: +12.8 bps
- Median max favorable move: +100.9 bps
- Median drawdown before bounce: -105.9 bps (8x worse than first bounce)

## What we learned

1. **RSI oversold/overbought IS a real signal** - not random. 92-98% of the time, price eventually reverses.

2. **Deeper RSI = WORSE outcomes, not better:**
   - "Barely" oversold (RSI 15-20) = BEST: 98% bounce, 70% in 1 bar, least drawdown
   - "Extreme" oversold (RSI 0-5) = WORST: 64% bounce, 7% in 1 bar, -323 bps drawdown
   - Reason: barely oversold = small dip in healthy trend. Extreme = freefall/panic, keeps going.

3. **The bounce is real but drawdown is massive:**
   - Median bounce = +12.8 bps, but median drawdown = -105.9 bps BEFORE the bounce
   - You suffer 8x more pain than your first reward
   - Raw RSI alone is NOT tradeable

4. **RSI < 20 threshold is the right choice:**
   - Most signals (59%) land in RSI 15-20 "barely" category = best outcomes
   - Using RSI < 10 would give worse results (more drawdown, fewer bounces)
   - This VALIDATES V1's threshold choice

5. **Need additional filters to trade RSI:**
   - SMA200 trend filter (to avoid freefall scenarios) -> tested in EXP-004
   - Trailing stop (to capture the bounce without holding through drawdown) -> tested in EXP-005
   - ATR + EMA filters (to avoid choppy markets) -> tested in EXP-006

## Next steps
- EXP-002: Test RSI + MA trend filter (does filtering by trend help?)
- EXP-003: Deep dive into the failures (what causes the 2-8% that don't bounce?)
- EXP-004: Path analysis + SMA200 filter
- EXP-005: Exit strategy (trailing stops)
