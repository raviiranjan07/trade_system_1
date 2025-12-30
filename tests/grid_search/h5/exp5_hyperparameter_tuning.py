#!/usr/bin/env python3
"""
Grid Search H=5m - Experiment 5: Hyperparameter tuning for min_expectancy
Run: python -m tests.grid_search.h5.exp5_hyperparameter_tuning

Based on exp1 results: min_expectancy=0.001 was best (+$373, 100% WR, 42 trades)
Now testing finer granularity around that value.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.grid_search.base import run_grid_search

GRID_PARAMS = {
    "min_expectancy": [
        0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
        0.001,
        0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
    ],
    "max_distance": [3.0],
    "blocked_regimes": [[]],
}

if __name__ == "__main__":
    run_grid_search(
        experiment_name="exp5_hyperparameter_tuning",
        horizon=5,
        grid_params=GRID_PARAMS
    )
