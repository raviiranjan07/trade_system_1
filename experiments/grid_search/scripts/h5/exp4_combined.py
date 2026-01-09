#!/usr/bin/env python3
"""
Grid Search H=5m - Experiment 4: Combined best parameters
Run: python -m tests.grid_search.h5.exp4_combined

Based on results from exp1, exp2, exp3:
- exp1: min_expectancy=0.001 is best (+$373, 100% WR, 42 trades)
- exp2: max_distance=3.0 is best (higher = better)
- exp3: blocked_regimes=[] is best (volatility helps short-term)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.grid_search.base import run_grid_search

GRID_PARAMS = {
    "min_expectancy": [0.001],
    "max_distance": [2.5, 3.0],
    "blocked_regimes": [
        [],
        ["RANGE_LOW_VOL"],
        ["RANGE_LOW_VOL", "TREND_LOW_VOL"],
    ],
}

if __name__ == "__main__":
    run_grid_search(
        experiment_name="exp4_combined",
        horizon=5,
        grid_params=GRID_PARAMS
    )
