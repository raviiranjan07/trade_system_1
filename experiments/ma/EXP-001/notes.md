# EXP-001: Moving Average Behavior Analysis

## What we're testing
Comprehensive MA analysis: Do MAs work as entry signals, support/resistance, or trend filters on 15min BTCUSDT?

## Hypothesis
- Price should continue after crossing above/below MA (trend following)
- MAs should act as support/resistance (price bounces off MA)
- Deeper MA types (EMA/WMA) might react faster than SMA

## Parameters
- MA Types: SMA, EMA, WMA
- Periods: 7, 9, 20, 50, 100, 200
- Horizons: 3, 5, 10, 15, 20 bars
- In-sample: 2020-2023, Out-of-sample: 2024-2025

## Results

### Initial Finding (continuation rate)
- ALL MA types show 98-99% continuation rate at H=10
- No difference between SMA, EMA, WMA
- BUT: same flaw as RSI test - checks if HIGH/LOW ever went in direction (nearly always true)

### MA Price Action Study (OOS 2024-2025)

**A) MA as Support/Resistance - NO EDGE**

| MA | Touches | Bounce% | Random Baseline |
|---|---|---|---|
| SMA200 | 1,954 | 76.3% | 78.6% |
| EMA100 | 3,429 | 76.4% | 78.6% |
| WMA20 | 8,721 | 76.7% | 78.6% |

All MAs perform BELOW random baseline (78.6%). MA support/resistance is an illusion on 15min.

**B) Distance from MA - Marginal edge in extremes only**
- Price far above SMA200 (>300bp): SHORT wins 93.3% - stretched price reverses
- Price near MA (-50 to +50): ~72% - choppy zone, worst performance
- Edge exists only at extremes but driven by high volatility (both directions reach targets)

**C) Zone Analysis - No directional edge**
- Bull trend (Price>EMA50>EMA200): LONG 80.4% vs random 78.6% = only +1.8pp
- All zones within 2-3pp of random. Not tradeable.

**D) SMA vs EMA vs WMA - Nearly identical**
- Support bounce: EMA slightly better at longer periods, WMA at shorter
- Differences are 1-2pp. No meaningful advantage for any type.

**E) MA Cross Direction Prediction - Coin flip**

| Cross | Stay in direction? |
|---|---|
| Cross UP (all MAs) | 50-53% |
| Cross DOWN (all MAs) | 45-47% |

MA crossings predict direction at ~50/50. Zero edge.

### MA Crossover Combination Grid (72 combos, OOS 2024-2025)

Tested all combinations: 3 types (SMA/EMA/WMA) x 3 fast periods (9/20/50) x 3 types x 3 slow periods (50/100/200). Exit: 20bps trailing stop + bar 10 time exit.

**Top 5 by Total Return:**

| Combo | Trades | Win% | Net bps | PF |
|---|---|---|---|---|
| WMA9/WMA50 | 2,718 | 48.4% | +12,818 | 1.30 |
| EMA9/WMA50 | 2,558 | 48.4% | +12,393 | 1.30 |
| WMA9/SMA50 | 2,146 | 47.3% | +12,095 | 1.37 |
| WMA9/EMA50 | 2,152 | 48.0% | +11,926 | 1.37 |
| SMA20/WMA50 | 2,035 | 48.6% | +11,818 | 1.39 |

**Top 5 by Profit Factor (min 50 trades):**

| Combo | Trades | Win% | Net bps | PF |
|---|---|---|---|---|
| EMA20/EMA100 | 943 | 49.5% | +9,662 | 1.72 |
| EMA9/EMA200 | 905 | 52.2% | +8,112 | 1.58 |
| SMA50/SMA200 | 443 | 48.8% | +3,409 | 1.58 |
| EMA20/WMA200 | 807 | 52.3% | +7,171 | 1.56 |
| EMA50/WMA200 | 561 | 51.5% | +4,367 | 1.54 |

**By MA type (averaged):**
- EMA fast: avg net +6,236 bps, avg PF 1.36
- WMA fast: avg net +6,763 bps, avg PF 1.35
- SMA fast: avg net +5,277 bps, avg PF 1.28

**By period pair (averaged across types):**
- 9/50: avg +10,932 bps (highest return, most trades)
- 9/200: avg +6,694 bps, PF 1.41 (highest PF for fast 9)
- 50/200: avg +2,559 bps, PF 1.37 (fewest trades, thin)

**ALL 72 combinations profitable but with thin edge:**
- Win rates: 45-54% (near coin flip)
- Best PF: 1.72 (EMA20/EMA100)
- High returns come from VOLUME (2,000+ trades), not per-trade quality
- Average per-trade profit: +3-5 bps (very thin)

### Comparison with our strategies:

| Strategy | Trades | Net bps | PF | Avg/trade |
|---|---|---|---|---|
| Best MA combo (WMA9/WMA50) | 2,718 | +12,818 | 1.30 | +4.7 |
| Best PF MA (EMA20/EMA100) | 943 | +9,662 | 1.72 | +10.2 |
| **V1.2 RSI** | **202** | **+3,250** | **2.47** | **+16.1** |
| **V2 VolSpike** | **826** | **+14,330** | **1.97** | **+17.3** |
| **V1.2 + V2 Combined** | **1,028** | **+18,820** | **1.99** | **+18.3** |

## What we learned

1. **MA support/resistance is an illusion on 15min** - bounce rate is below random baseline
2. **MA crossovers have zero directional edge** - 50/50 coin flip at predicting direction
3. **MA crossover strategies are profitable but THIN** - best PF 1.72 vs V1.2's 2.47
4. **High total return comes from volume** (many trades), not signal quality
5. **SMA vs EMA vs WMA - no meaningful difference** (1-2pp)
6. **EMA slightly better for longer periods**, WMA slightly better for shorter
7. **MAs work best as REGIME FILTERS** (bull/bear via SMA200) not as entry signals
8. **Our V1.2+V2 combined portfolio beats ALL 72 MA combinations** in both PF and total return

## Impact on other experiments
- **None.** V1.2 uses SMA200 as filter (correct use), not as entry signal
- V2 doesn't use any MA
- This analysis CONFIRMS our approach: use MAs for regime identification, not for entries
