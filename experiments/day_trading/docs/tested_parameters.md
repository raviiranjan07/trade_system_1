# Day Trading - Tested Parameters

All experiments conducted for day trading strategy (H=5-10 minutes).

---

## Horizon Testing

| Horizon | Win Rate | Return | Trades | Status |
|---------|----------|--------|--------|--------|
| H=5 | 100% | +11%/year | 137 | BEST |
| H=10 | 100% | +1.1% | 6 | OK (few trades) |

**Conclusion:** H=5 is optimal. H=10 works but generates very few trades.

---

## Sample Interval Testing (H=5)

| si | Win Rate | Return | Trades | Status |
|----|----------|--------|--------|--------|
| si=1 | In progress | - | - | TESTING |
| si=5 | 99.2% | -0.42% | 133 | LOSS (1 bad trade) |
| si=10 | - | - | - | Tested |
| si=15 | 100% | +20.65% | 137 | BEST |

**Conclusion:** si=15 is optimal. Lower intervals catch bad signals.

---

## min_expectancy Testing (H=5, si=15)

| Value | Win Rate | Return | Trades | Status |
|-------|----------|--------|--------|--------|
| 0.0 | 21% | -42.5% | 2,057 | LOSS |
| 0.0005 | 68% | -1.56% | 267 | LOSS |
| 0.0006 | 85% | -0.76% | 175 | LOSS |
| 0.0007 | 96% | +6.92% | 119 | OK |
| 0.0008 | 99% | +6.32% | 93 | OK |
| 0.0009 | 100% | +4.80% | 60 | GOOD |
| 0.001 | 100% | +3.73% | 42 | BEST (balanced) |
| 0.0011 | 100% | +2.57% | 26 | OK |
| 0.0012 | 100% | +1.83% | 18 | OK |
| 0.002 | 100% | +0.14% | 1 | Too strict |
| 0.003+ | - | 0% | 0 | No trades |

**Conclusion:**
- 0.001 is optimal balance of trades and win rate
- 0.0009 gives more return but slightly lower safety margin
- Below 0.0007, losses start appearing

---

## max_distance Testing (H=5)

| Value | Result |
|-------|--------|
| 0.5 | Fewer trades, lower returns |
| 1.0 | Fewer trades |
| 1.5 | Fewer trades |
| 2.0 | Fewer trades |
| 2.5 | Fewer trades |
| 3.0 | OPTIMAL |

**Conclusion:** max_distance=3.0 allows more neighbors, better results.

---

## blocked_regimes Testing (H=5)

| Blocked Regime | Win Rate | Return | Status |
|----------------|----------|--------|--------|
| NONE | 100% | +3.73% | BEST |
| HIGH_VOL | Lower | Worse | AVOID |
| RANGE_LOW_VOL | 100% | +3.52% | Slight drop |
| TREND_LOW_VOL | 100% | +3.52% | Slight drop |
| RANGE_LOW_VOL + TREND_LOW_VOL | 100% | +3.52% | Slight drop |

**Conclusion:** Don't block any regimes for H=5. Volatility helps short-term trades.

---

## EMA Configuration Testing

| Config | Win Rate | Return | Status |
|--------|----------|--------|--------|
| EMA 50/200 | 100% | +12.63% | BEST |
| EMA 21/50/100/200 | - | -7.21% | WORSE |

**Conclusion:** Keep EMA 50/200. Adding more EMAs hurts performance.

---

## Key Insights

1. **min_expectancy is critical:** Transforms -42% loss into +11% profit
2. **sample_interval matters:** si=15 avoids bad signals that si=5 catches
3. **Don't block regimes:** All regimes useful for short-term trading
4. **max_distance=3.0:** Allows sufficient neighbor matching
5. **EMA 50/200:** Simpler is better

---

## Optimal Configuration Summary

```yaml
# Day Trading - Best Parameters
decision:
  horizon: 5
  min_expectancy: 0.001
  max_distance: 3.0
  blocked_regimes: []

backtest:
  sample_interval: 15

similarity:
  k: 200

normalization:
  window: 2000
```
