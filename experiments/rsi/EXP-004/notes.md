# EXP-004: Path Analysis - Early Exit Detection

## What we're testing
Can we detect RSI failures early (at bar 1-3) and exit before big losses?
Also: Do failures eventually recover if we wait longer?

## Hypothesis
Failures might show different patterns early (go negative immediately) vs winners.
If we can detect this, we can exit early and limit losses.

## Key Findings

### 1. Path Analysis
- 100% of failures are negative at Bar 1
- Successes average +3 bps at Bar 1
- Clear early differentiation exists

### 2. Early Exit Rules DON'T Work
| Rule | Failures Caught | Winners Killed | Net Impact |
|------|-----------------|----------------|------------|
| Exit if < 0 at Bar 1 | 100% | 43% | -7,378 bps |
| Exit if < -10 at Bar 1 | 91% | 27% | -4,340 bps |

Problem: Many winners also start negative, then recover.

### 3. Recovery Analysis
| Outcome | Count | % |
|---------|-------|---|
| Recovered < 12 hours | 9 | 82% |
| Recovered 4.6 days | 1 | 9% |
| NEVER recovered | 1 | 9% |

### 4. The Account Killer
Failure #1 (2024-02-26):
- **Max drawdown: -4,208 bps (42%!)**
- Never recovered within 20 days
- Caused by: BTC ETF rally, RSI blind to macro trend

## What we learned

1. **Early exit doesn't work** - kills too many winners
2. **Most failures recover** - 82% within 12 hours
3. **9% are catastrophic** - account killers that never recover
4. **RSI is blind to macro trends** - can't predict which signal is the killer
5. **Overbought (SHORT) is dangerous** - 1 bad trade can wipe 84 winners

## Recommendations

**Safest Option:** Trade oversold (LONG) only
- 99.7% success rate
- Only 2 failures in OOS
- Avoid the catastrophic SHORT risk

**If trading both:** Use -300 bps hard stop on overbought
- Caps loss but still painful
- Better than -4,208 bps

## Next steps
- Consider macro trend filter (only SHORT in bear markets)
- Or simply skip overbought signals entirely
- Focus on building complete strategy with oversold LONG
