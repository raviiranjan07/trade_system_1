"""
Outcome Labeling Module.

Computes MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
for forward-looking trade outcome analysis.
"""

from .outcome_labeler import compute_mfe_mae, label_outcomes

__all__ = [
    "compute_mfe_mae",
    "label_outcomes",
]
