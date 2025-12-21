"""
Visualization module for the trading system.

Provides charts and plots for analyzing:
- Market regimes over time
- MFE/MAE outcome distributions
- State vector patterns
- Similarity search results
"""

from .plot_regimes import plot_regime_timeline, plot_regime_distribution
from .plot_outcomes import plot_mfe_mae_distribution, plot_expectancy_by_regime
from .plot_states import plot_state_heatmap, plot_state_correlation

__all__ = [
    "plot_regime_timeline",
    "plot_regime_distribution",
    "plot_mfe_mae_distribution",
    "plot_expectancy_by_regime",
    "plot_state_heatmap",
    "plot_state_correlation",
]
