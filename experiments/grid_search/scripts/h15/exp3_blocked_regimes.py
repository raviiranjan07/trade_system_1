#!/usr/bin/env python3
"""
Grid Search H=15m - Experiment 3: blocked_regimes variations
Run: python -m tests.grid_search.h15.exp3_blocked_regimes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.grid_search.base import run_grid_search

GRID_PARAMS = {
    "min_expectancy": [0.0],
    "max_distance": [3.0],
    "blocked_regimes": [
        [],
        ["HIGH_VOL"],
        ["TREND_LOW_VOL"],
        ["RANGE_LOW_VOL"],
        ["TREND_HIGH_VOL"],
        ["HIGH_VOL", "TREND_LOW_VOL"],
        ["HIGH_VOL", "RANGE_LOW_VOL"],
    ],
}

if __name__ == "__main__":
    run_grid_search(
        experiment_name="exp3_blocked_regimes",
        horizon=15,
        grid_params=GRID_PARAMS
    )
