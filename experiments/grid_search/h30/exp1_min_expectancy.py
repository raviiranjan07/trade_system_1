#!/usr/bin/env python3
"""
Grid Search H=30m - Experiment 1: min_expectancy variations
Run: python -m tests.grid_search.h30.exp1_min_expectancy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.grid_search.base import run_grid_search

GRID_PARAMS = {
    "min_expectancy": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
    "max_distance": [3.0],
    "blocked_regimes": [[]],
}

if __name__ == "__main__":
    run_grid_search(
        experiment_name="exp1_min_expectancy",
        horizon=30,
        grid_params=GRID_PARAMS
    )
