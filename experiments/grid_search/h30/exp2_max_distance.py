#!/usr/bin/env python3
"""
Grid Search H=30m - Experiment 2: max_distance variations
Run: python -m tests.grid_search.h30.exp2_max_distance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.grid_search.base import run_grid_search

GRID_PARAMS = {
    "min_expectancy": [0.0],
    "max_distance": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "blocked_regimes": [[]],
}

if __name__ == "__main__":
    run_grid_search(
        experiment_name="exp2_max_distance",
        horizon=30,
        grid_params=GRID_PARAMS
    )
