# Experiment Registry

> **FROZEN (2026-06-11):** Historical log only — no longer updated. The canonical experiment
> registry is `experiments/mlops_registry.csv` (auto-written by `src/mlops/registry.py`).
> NOTE: EXP numbering here does NOT match `registry.csv` (e.g. EXP-001 is EMA7 Breakout here,
> RSI Behavior Analysis there).

Master log of all trading strategy experiments. Each experiment tests one specific idea to find what works.

---

## EXP-001: EMA7 Breakout (Basic)
**Date:** 2024-02-03
**Status:** ❌ FAILED

**What we tested:**
Enter LONG when price crosses above EMA7, enter SHORT when price crosses below EMA7. Exit at 12 bps profit target.

**Why we tested it:**
EMA acts as dynamic support/resistance. We wanted to see if trading breakouts from EMA7 would be profitable.

**Parameters:**
- Timeframe: 15-min
- Entry: Cross above/below EMA7
- Exit: 12 bps take profit
- Fees: 8 bps

**Results:**
- Total trades: 16,929
- Win rate: 79.39%
- Total return: -$1,553 LOSS
- Avg winner: +4 bps
- Avg loser: -59.90 bps

**What we learned:**
High win rate doesn't mean profitable! Losers were HUGE (-60 bps average) compared to tiny winners (+4 bps). The strategy enters too many choppy signals and exits winners too early.

**Next step:** Add entry filter to avoid choppy signals.

---

## EXP-002: EMA7 with Stop-and-Reverse
**Date:** 2024-02-03
**Status:** ❌ FAILED

**What we tested:**
Same as EXP-001, but if trade goes -12 bps against us, reverse the position (if LONG goes wrong, flip to SHORT).

**Why we tested it:**
Maybe we're entering at the right time but wrong direction. Reversing could catch the move.

**Parameters:**
- Timeframe: 15-min
- Entry: Cross above/below EMA7
- Stop loss: -12 bps (then reverse)
- Fees: 8 bps per position

**Results:**
- Total trades: 16,929
- Win rate: 56.59%
- Total return: -$1,500 LOSS
- Reversals: 42.8% of trades
- Problem: Whipsaws in choppy markets

**What we learned:**
Reversing doesn't help. Choppy markets cause lots of whipsaws, eating profits with fees. Need better entry filter instead.

**Next step:** Wait longer and filter choppy entries.

---

## EXP-003: Two-Stage Exit (Wait 20 bars, then MAE check)
**Date:** 2024-02-03
**Status:** ❌ FAILED

**What we tested:**
Wait 20 bars (5 hours) before checking if trade is failing. If MAE > 50 bps after 20 bars, exit. Otherwise hold up to 60 bars.

**Why we tested it:**
From WHAT analysis, we know 90% of trades eventually recover. Maybe we just need to wait longer for "dirty wins" to recover.

**Parameters:**
- Timeframe: 15-min
- Entry: Cross above/below EMA7
- Stage 1: Wait 20 bars (5 hours)
- Stage 2: If MAE > 50 bps, exit
- Max hold: 60 bars (15 hours)
- Fees: 8 bps

**Results:**
- Total trades: 16,929
- Win rate: 87.15% (much better!)
- Total return: -$1,441 LOSS
- Avg loser: -98.02 bps (HUGE!)

**What we learned:**
Waiting too long created catastrophic losses. When a trade is truly wrong direction, holding for 20+ bars makes it much worse. Need a way to cut bad trades early while letting good ones run.

**Next step:** Use trailing stop instead of waiting.

---

## EXP-004: Hold Longer (60 bars, no MAE exit)
**Date:** 2024-02-03
**Status:** ❌ FAILED

**What we tested:**
Just hold every trade for up to 60 bars (15 hours) with 12 bps profit target. No MAE exit at all.

**Why we tested it:**
Testing the extreme: what if we just wait for ALL trades to recover?

**Parameters:**
- Timeframe: 15-min
- Entry: Cross above/below EMA7
- Exit: 12 bps TP OR 60 bars max
- No MAE threshold
- Fees: 8 bps

**Results:**
- Total trades: 16,929
- Win rate: 92.58% (very high!)
- Total return: -$1,495 LOSS
- Avg loser: -177.05 bps (catastrophic!)
- Worst trade: -1,511 bps

**What we learned:**
92% of trades DO eventually hit target, BUT the 7.4% that never recover destroy the account with massive losses. We need BOTH: hold longer for recoveries AND filter/cut bad entries.

**Next step:** Add entry filter AND trailing stop.

---

## EXP-005: Filtered Entry + Trailing Stop ⭐
**Date:** 2024-02-03
**Status:** ✅ BEST SO FAR

**What we tested:**
Two improvements: (1) Only enter if cross is >= 5 bps away from EMA (avoid tiny choppy crosses), (2) Use 10 bps trailing stop from peak (let winners run, cut losers quickly).

**Why we tested it:**
Combining lessons: filter choppy entries (reduces bad trades) + trailing stop (protects profit while letting trends run).

**Parameters:**
- Timeframe: 15-min
- Entry: Cross above/below EMA7 with >= 5 bps distance
- Exit: 10 bps trailing stop from peak
- Max hold: 60 bars
- Fees: 8 bps

**Results:**
- Total trades: 9,853 (filtered out 41.8% of signals!)
- Win rate: 35.04%
- Total return: **+$102.79 PROFIT** (+102.79%)
- Avg winner: +22.75 bps
- Avg loser: -10.66 bps
- Profit factor: 1.15
- Risk/Reward: 2.13

**What we learned:**
**THIS WORKS!** Even with low 35% win rate, it's profitable because:
- Entry filter removes choppy signals (7,063 skipped)
- Trailing stop cuts losers quickly (-10.66 avg)
- Trailing stop lets winners run (+22.75 avg)
- Winners are 2x bigger than losers

**Issues:**
- Still only 35% win rate (can we improve?)
- Only tested EMA7 (other EMA periods might be better?)
- Only tested on 15-min timeframe

**Next step:** Test other EMA periods, validate on train data, test other techniques.

---

## EXP-006: Filtered + Trailing with Activation Threshold
**Date:** 2024-02-04
**Status:** ❌ FAILED

**What we tested:**
Same as EXP-005 but trailing stop only activates AFTER reaching 12 bps gross profit (4 bps net). Before that, just hold.

**Why we tested it:**
Noticed that some trades got stopped out at small profits that would be losses after fees. Wanted to avoid this.

**Parameters:**
- Timeframe: 15-min
- Entry: Cross above/below EMA7 with >= 5 bps distance
- Activation: Trailing stop only activates at 12 bps gross profit
- Exit: 10 bps trailing stop from peak (after activation)
- Max hold: 60 bars
- Fees: 8 bps

**Results:**
- Total trades: 9,853
- Win rate: 62.65% (higher!)
- Total return: **-$76.29 LOSS** (-76.29%)
- Avg winner: +20.29 bps
- Avg loser: -36.11 bps (much worse!)

**What we learned:**
**WORSE than EXP-005!** The activation threshold created bigger losses. Trades that never reached 12 bps held for full 60 bars and went deeply negative. The original trailing stop (no activation) was better because it cut small losses early.

**Conclusion:** Sometimes protecting tiny profits is better than holding for bigger ones.

**Decision:** Revert to EXP-005 logic (no activation threshold).

---

## EMA Period Analysis
**Date:** 2024-02-04
**Status:** 📊 ANALYSIS

**What we tested:**
Compared different EMA periods (5, 7, 10, 15, 20, 30) to see which is the best trend follower.

**Why we tested it:**
We've been using EMA7 all along, but never tested if it's actually the best period.

**Results:**
| EMA | Total Crosses | Avg Bars Between | Distance (bps) |
|-----|---------------|------------------|----------------|
| 5   | 19,798        | 3.5 bars         | 14.7 bps       |
| 7   | 16,929        | 4.1 bars         | 18.6 bps       |
| 10  | 14,345        | 4.9 bars         | 23.3 bps       |
| 15  | 11,559        | 6.1 bars         | 29.8 bps       |
| 20  | 10,161        | 6.9 bars         | 35.3 bps       |
| 30  | 8,139         | 8.6 bars         | 44.5 bps       |

**What we learned:**
- EMA5: Very choppy (19K crosses), too sensitive
- EMA7: Still choppy (16K crosses)
- EMA15-20: Better balance - fewer crosses, still relevant
- EMA30: Cleanest (8K crosses) but far from price

**Tradeoff:** Faster EMAs = closer to price but choppier. Slower EMAs = cleaner trends but lag more.

**Next step:** Test EMA15 or EMA20 with the filtered + trailing stop strategy to see if results improve.

---

## Summary of Current State

**Best Strategy So Far:** EXP-005 (Filtered Entry + Trailing Stop)
**Return:** +102.79% on test data (2024-2025)
**Status:** Profitable but needs validation

**Open Questions:**
1. Would EMA15 or EMA20 work better than EMA7?
2. How does this perform on train data (2020-2023)?
3. Are there better techniques than EMA at 15-min level?
4. Would multi-timeframe filters improve results?

**Next Steps:**
1. Test other EMA periods with same strategy
2. Test other techniques (RSI, Bollinger, Price Action, Similarity)
3. Find best atomic component at 15-min level
4. Then add next building blocks (multi-TF, risk mgmt)
