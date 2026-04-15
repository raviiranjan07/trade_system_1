# EXP-002: RSI + MA Trend Filter

## What we're testing
Testing if adding a Moving Average (MA) trend filter improves RSI signal quality by filtering out signals that go AGAINST the main trend.

## Hypothesis
RSI failures happen when trading AGAINST the trend. Adding MA filter (only trade WITH trend) should reduce failures and improve MFE/MAE ratio.

## Parameters
- RSI: Period 14, Oversold < 20, Overbought > 80
- MA Filter: SMA with periods 20, 50, 100
- Horizon: 10 bars
- See config.yaml for full details

## Results

### Out-of-Sample (2024-2025):
| Filter | Signals | Success% | Failures | Avg MFE |
|--------|---------|----------|----------|---------|
| RSI Alone | 596 | 99.7% | 2 | 72.8 |
| RSI + SMA(50) WITH trend | 21 | 100% | 0 | 53.7 |
| RSI + SMA(50) AGAINST trend | 575 | 99.7% | 2 | 73.5 |

## What we learned

1. **HYPOTHESIS REJECTED** - MA filter does NOT meaningfully improve RSI

2. **RSI is trend-agnostic** - Works equally well WITH and AGAINST trend (both ~99.7%)

3. **Over-filtering is dangerous** - SMA(50) removes 97% of signals for 0.3% improvement

4. **AGAINST trend = BIGGER bounces** - MFE is higher when price < SMA (deeper pullback = more room to bounce)

5. **The math is clear**:
   - RSI Alone: 596 signals × 99.7% = 594 good trades
   - RSI + SMA(50): 21 signals × 100% = 21 good trades
   - We "saved" 2 failures but LOST 573 opportunities

### Extended Test: RSI<20 + Below MAs = LONG (OOS 2024-2025)

Tested whether RSI oversold in bearish zone (price below MAs) works as LONG entry:

| Config | Trades | Win% | Net bps | PF | Avg/trade |
|---|---|---|---|---|---|
| ALL RSI<20 = LONG (baseline) | 596 | 57.2% | +6,793 | 1.71 | +11.4 |
| RSI<20 + below ALL 3 (SMA50+SMA100+EMA9) | 532 | 58.6% | +6,598 | 1.77 | +12.4 |
| RSI<20 + below SMA200 | 470 | 59.4% | +5,973 | 1.77 | +12.7 |
| RSI<20 + above SMA50 | 21 | 47.6% | -107 | 0.74 | -5.1 |
| RSI<20 + above SMA100 | 63 | 46.0% | +220 | 1.23 | +3.5 |
| RSI<20 + above SMA200 | 126 | 49.2% | +821 | 1.47 | +6.5 |

**Key finding:** RSI<20 works BETTER in bearish zone (below MAs), not worse.
- Below all 3 MAs: PF 1.77, 58.6% win (better than baseline)
- Above SMA50: PF 0.74, LOSES money
- Above SMA100/200: weak (PF 1.23-1.47)
- Deeper pullback = bigger bounce

**But V1.2 still superior:** PF 2.47 with 202 selective trades vs PF 1.77 with 532 trades.

### Design Flaw Identified

Original EXP-002 only asked "keep or reject LONG?" when price was below MAs. Never tested:
1. Flipping direction (SHORT when below MAs) — tested later, RSI<20 SHORT = PF 1.23-1.47 (weaker than LONG)
2. Using MAs as regime selector (not filter) — this became V1's core design in EXP-004

## Next steps
See explanation.txt for options:
- A) Investigate failures differently (volume, time, volatility)
- B) Test MA crossover as entry (not filter)
- C) Accept RSI entry as-is, focus on EXIT rules
