# L1-EXP-005: Hybrid Kelly (Continuous + Bayesian + Drawdown Constraint)

## Question
Can adaptive Kelly sizing beat fixed leverage for V1.3.2?

## ANSWER: NO. Hybrid Kelly is dramatically WORSE than fixed leverage. REJECTED.

## Approach: Three Components
1. **Bayesian**: Beta prior for win rate, updated after each trade (posterior mean)
2. **Continuous**: Recalculate Kelly leverage every trade using running statistics
3. **Drawdown Constraint**: Scale leverage down when equity drops from peak
   - dd_scale = max(0, 1 - current_dd / max_dd_limit)
   - At max_dd: leverage goes to minimum (5x)

## Grid Tested
- 4 priors: Uniform(1,1), Weak(6,4), Moderate(30,20), Strong(60,40)
- 4 max DD limits: 20%, 30%, 40%, 50%
- 2 Kelly modes: Half, Full
- Total: 32 configs x 1000 MC paths = 32,000 simulations
- Phase 1: Cross/$15 (same as EXP-002 winner)

## Results: Hybrid Kelly vs Fixed Baselines

| Config | Median | P5 | AvgDD |
|--------|--------|-----|-------|
| Fixed 20x | $41,101 | $29,507 | 45.5% |
| Fixed 25x | $178,108 | $103,254 | 55.1% |
| Fixed 30x | $649,560 | $303,142 | 63.4% |
| **Best Hybrid (Strong/50%DD/Half)** | **$15,898** | **$5,721** | **45.3%** |

- Hybrid beats Fixed 20x on only 11.8% of MC paths
- Hybrid beats Fixed 25x on only 0.5% of MC paths

## Ablation: Why Each Component Hurts

| Config | Median | P5 | Insight |
|--------|--------|-----|---------|
| Full Hybrid (best) | $15K | $5.7K | All three components combined |
| No DD constraint | $52K | $19K | DD constraint costs 3.4x median |
| No Bayesian (flat prior) | $12K | $4K | Bayesian helps P5 by 40% |
| Full Kelly (no half) | $13K | $164 | Full Kelly -> near ruin |
| Fixed 25x + DD only | $22K | $10K | Even simple DD on fixed hurts |

## Root Causes of Failure

1. **Bayesian starts too conservative**: With prior at 50-60% win, early leverage is ~20x.
   Takes ~50+ trades for posterior to converge. Lost compounding during warmup.

2. **DD constraint kills recovery**: V1.3.2 has 60% win rate and PF 3.46 - drawdowns
   recover fast. Cutting leverage during DD means missing the recovery bounce.
   At 50% max DD, leverage is at minimum 9.1% of the time.

3. **Variable leverage hurts geometric growth**: Average hybrid leverage is 18.6x
   vs fixed 20x, but VARIANCE in leverage timing is worse than fixed.
   Sometimes low leverage on winners, high on losers (bad luck sequences).

4. **Strategy edge is stable**: Kelly adapts well when edge is uncertain or degrading.
   V1.3.2 edge is consistent (works 2024 and 2025) - no adaptation needed.

## Prior Strength Impact (marginal)

| Prior | Pseudo-N | Median | P5 |
|-------|----------|--------|-----|
| None (1,1) | 2 | $12K | $3.9K |
| Weak (6,4) | 10 | $13K | $4.6K |
| Moderate (30,20) | 50 | $14K | $5.1K |
| Strong (60,40) | 100 | $16K | $5.7K |
| Heavy (120,80) | 200 | $16K | $6.2K |

Stronger prior helps but doesn't fix the fundamental problem.

## Drawdown Limit Sweep

| MaxDD | Median | P5 | AvgDD | MinLev% |
|-------|--------|-----|-------|---------|
| 15% | $2.8K | $658 | 39.2% | 40.4% |
| 30% | $5.6K | $1.5K | 42.0% | 27.7% |
| 50% | $15K | $5.8K | 45.4% | 9.1% |
| 60% | $23K | $8.9K | 46.4% | 3.8% |

Higher MaxDD = better returns. The best DD constraint is... no DD constraint.

## Key Insight
Hybrid Kelly is theoretically elegant but practically inferior for V1.3.2 because:
- 220 trades is too few for Bayesian to converge quickly
- 60% win rate + PF 3.46 = edge is strong and stable
- Drawdowns recover naturally - no need to cut leverage
- Fixed leverage compounding is OPTIMAL for stable-edge strategies

## Verdict
**REJECTED.** Keep fixed leverage from EXP-002 (Cross/20x/$15 or Cross/25x/$15).
No adaptive sizing needed.

## Files
- hybrid_kelly.py (main experiment)
- cross_vs_isolated.py (from earlier, now part of EXP-002 conclusions)
