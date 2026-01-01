# Grid Search Experiments

Structured parameter optimization for each trading horizon.

## Directory Structure

```
tests/grid_search/
├── base.py              # Shared grid search logic
├── __init__.py
├── README.md
├── h5/                  # 5-minute horizon experiments
│   ├── exp1_min_expectancy.py
│   ├── exp2_max_distance.py
│   ├── exp3_blocked_regimes.py
│   ├── exp4_combined.py
│   └── exp5_finetune.py
├── h10/                 # 10-minute horizon experiments
│   ├── exp1_min_expectancy.py
│   ├── exp2_max_distance.py
│   └── exp3_blocked_regimes.py
├── h15/                 # 15-minute horizon experiments
│   ├── exp1_min_expectancy.py
│   ├── exp2_max_distance.py
│   └── exp3_blocked_regimes.py
└── h30/                 # 30-minute horizon experiments
    ├── exp1_min_expectancy.py
    ├── exp2_max_distance.py
    └── exp3_blocked_regimes.py
```

## How to Run

From the project root directory:

```bash
# Run single experiment
python -m tests.grid_search.h5.exp1_min_expectancy
python -m tests.grid_search.h10.exp1_min_expectancy
python -m tests.grid_search.h15.exp1_min_expectancy
python -m tests.grid_search.h30.exp1_min_expectancy

# Run experiments in parallel (open separate terminals)
# Terminal 1:
python -m tests.grid_search.h5.exp1_min_expectancy
# Terminal 2:
python -m tests.grid_search.h10.exp1_min_expectancy
# etc.
```

## Experiment Types

| Experiment | Description | Parameters Tested |
|------------|-------------|-------------------|
| exp1 | min_expectancy | 0.0, 0.001, 0.002, 0.003, 0.004, 0.005 |
| exp2 | max_distance | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |
| exp3 | blocked_regimes | Various regime combinations |
| exp4 | combined | Best from exp1-3 |
| exp5 | finetune | Fine-tune around best min_expectancy |

## Workflow

1. **Run exp1, exp2, exp3** for each horizon (can run in parallel)
2. **Analyze results** in `data/grid_search/`
3. **Create exp4** combining best parameters from exp1-3
4. **Create exp5** to fine-tune the most impactful parameter

## Results Location

All results are saved to: `data/grid_search/grid_h{horizon}_{experiment}_{pair}_{timestamp}.csv`

## H=5m Results (Completed)

Best configuration:
- min_expectancy = 0.001
- max_distance = 3.0
- blocked_regimes = [] (none)
- Result: +$373, 100% WR, 42 trades
