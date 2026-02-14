# L1-EXP-003: Kelly Criterion -> $/step

## Question
Which Kelly algorithm gives the best $/step for position sizing?

## Kelly Types Tested
1. **Classic**: f = p - q/b = 0.4266 (43%)
2. **Mean-Variance**: f = mean/var = 52.02 (useless - says bet 52x wallet)
3. **Geometric**: f = argmax E[log(1+f*r)] = 2.99 (useless - says bet 3x wallet)
4. **Continuous**: f = mean/var = 52.02 (same as Mean-Variance)

## Key Findings

### FINDING 1: Only Classic Kelly works
- Mean-Variance, Geometric, Continuous all give nonsense (bet more than wallet)
- Classic Kelly = 43% fraction, which is reasonable

### FINDING 2: Classic Quarter-Kelly matches empirical optimal
- Full Kelly ($0.47/step) = 100% ruin
- Half Kelly ($0.95/step) = 100% ruin
- **Quarter Kelly ($1.89/step) = 0.1% ruin, $7.5M median**
- Empirical brute-force optimal = $1.85/step (almost identical)

### FINDING 3: Sharp cliff at $1.80-$1.85
- $1.80/step = 91% ruin
- $1.85/step = 0.1% ruin
- Very sensitive - small change in position = life or death

## Verdict
Classic Quarter-Kelly ($1.89) and empirical ($1.85) agree. Both are aggressive with 0.1% ruin. $2.00-$2.50/step is safer.

## Files
- kelly_to_step.py
