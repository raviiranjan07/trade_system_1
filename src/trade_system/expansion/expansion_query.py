"""
Expansion Query Engine - Queries expansion rates from similar historical states.

This module provides the key metric for trading decisions: expansion_rate.

Instead of asking "what's the average MFE?", we ask:
"What percentage of similar historical states led to expansion?"

Usage:
    engine = ExpansionQueryEngine(
        expansion_df=expansion_labels,
        outcome_df=outcome_df,
        regime_df=regime_df,
    )

    result = engine.query(
        current_state=state_vector,
        regime="TREND_HIGH_VOL",
        horizon=5,
    )

    if result["expansion_rate"] > 0.55:
        # Trade in the direction with higher expansion probability
        direction = result["direction"]
        tp = result["mfe_given_expansion"]
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd

from ..similarity.similarity_engine import SimilarityEngine, STATE_COLUMNS


@dataclass
class ExpansionQueryResult:
    """Result of an expansion query."""

    status: str  # "OK", "INSUFFICIENT_NEIGHBORS", "NO_REGIME_DATA"
    neighbors: int
    distance_mean: float

    # Primary signals (expansion-based)
    long_expansion_rate: float
    short_expansion_rate: float
    expansion_rate: float  # max of long/short
    direction: str  # "LONG" or "SHORT"

    # Conditional metrics (only from expanding neighbors)
    mfe_given_expansion: float
    avg_time_to_expansion: Optional[float]

    # Legacy metrics (for compatibility)
    mean_mfe: float
    mfe_40pct: float

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "neighbors": self.neighbors,
            "distance_mean": round(self.distance_mean, 4),
            "long_expansion_rate": round(self.long_expansion_rate, 4),
            "short_expansion_rate": round(self.short_expansion_rate, 4),
            "expansion_rate": round(self.expansion_rate, 4),
            "direction": self.direction,
            "mfe_given_expansion": round(self.mfe_given_expansion, 6),
            "avg_time_to_expansion": round(self.avg_time_to_expansion, 1) if self.avg_time_to_expansion else None,
            "mean_mfe": round(self.mean_mfe, 6),
            "mfe_40pct": round(self.mfe_40pct, 6),
        }


class ExpansionQueryEngine:
    """
    Queries expansion rates from similar historical states.

    This engine wraps the SimilarityEngine but returns expansion_rate
    as the primary signal instead of mean_mfe.

    The key insight is:
    - expansion_rate = "% of similar states that expanded"
    - This is the PROBABILITY of expansion, which is what we want to predict

    Args:
        expansion_df: DataFrame with expansion labels (from ExpansionLabeler)
        outcome_df: DataFrame with MFE/MAE outcomes (for TP sizing)
        regime_df: DataFrame with regime labels
        k: Number of neighbors to find
        backend: "bruteforce" or "faiss"
        min_neighbors: Minimum neighbors required for valid query

    Example:
        engine = ExpansionQueryEngine(expansion_df, outcome_df, regime_df)
        result = engine.query(state, "TREND_HIGH_VOL", horizon=5)
        if result.expansion_rate > 0.55:
            trade(result.direction)
    """

    def __init__(
        self,
        expansion_df: pd.DataFrame,
        outcome_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        k: int = 150,
        backend: Literal["bruteforce", "faiss"] = "faiss",
        faiss_nlist: int = 100,
        faiss_nprobe: int = 10,
        min_neighbors: int = 50,
    ):
        self.expansion_df = expansion_df
        self.outcome_df = outcome_df
        self.regime_df = regime_df
        self.k = k
        self.min_neighbors = min_neighbors

        # Initialize underlying similarity engine
        self.similarity_engine = SimilarityEngine(
            outcome_df=outcome_df,
            regime_df=regime_df,
            k=k,
            backend=backend,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
        )

        # Validate expansion_df has same index as outcome_df
        self._validate_alignment()

    def _validate_alignment(self) -> None:
        """Validate that expansion_df and outcome_df are aligned."""
        if not self.expansion_df.index.equals(self.outcome_df.index):
            # Try to align
            common_idx = self.expansion_df.index.intersection(self.outcome_df.index)
            if len(common_idx) < len(self.outcome_df) * 0.9:
                raise ValueError(
                    f"expansion_df and outcome_df indices don't align. "
                    f"Common: {len(common_idx)}, outcome: {len(self.outcome_df)}"
                )

    def query(
        self,
        current_state: pd.Series,
        regime: str,
        horizon: int,
        max_timestamp: Optional[pd.Timestamp] = None,
    ) -> ExpansionQueryResult:
        """
        Query expansion rate for a given state.

        Args:
            current_state: Current state vector
            regime: Current market regime
            horizon: Horizon to query (e.g., 5 for 5-bar expansion)
            max_timestamp: Don't use data after this timestamp (for backtesting)

        Returns:
            ExpansionQueryResult with expansion rates and trade signals
        """
        # Step 1: Find neighbors using existing similarity engine
        # We use the internal method to get indices and distances
        neighbors_result = self.similarity_engine.query(
            current_state=current_state,
            regime=regime,
            horizon=horizon,
            max_timestamp=max_timestamp,
        )

        # Check for errors
        if neighbors_result.get("status") != "OK":
            return ExpansionQueryResult(
                status=neighbors_result.get("status", "ERROR"),
                neighbors=neighbors_result.get("neighbors", 0),
                distance_mean=0.0,
                long_expansion_rate=0.0,
                short_expansion_rate=0.0,
                expansion_rate=0.0,
                direction="NONE",
                mfe_given_expansion=0.0,
                avg_time_to_expansion=None,
                mean_mfe=0.0,
                mfe_40pct=0.0,
            )

        # Get neighbor indices from the result
        neighbor_indices = neighbors_result.get("_neighbor_indices", [])

        if len(neighbor_indices) < self.min_neighbors:
            return ExpansionQueryResult(
                status="INSUFFICIENT_NEIGHBORS",
                neighbors=len(neighbor_indices),
                distance_mean=neighbors_result.get("distance_mean", 0.0),
                long_expansion_rate=0.0,
                short_expansion_rate=0.0,
                expansion_rate=0.0,
                direction="NONE",
                mfe_given_expansion=0.0,
                avg_time_to_expansion=None,
                mean_mfe=0.0,
                mfe_40pct=0.0,
            )

        # Step 2: Get expansion labels for neighbors
        suffix = f"{horizon}m"
        long_exp_col = f"long_expansion_{suffix}"
        short_exp_col = f"short_expansion_{suffix}"
        long_time_col = f"long_time_{suffix}"
        short_time_col = f"short_time_{suffix}"

        # Check columns exist
        if long_exp_col not in self.expansion_df.columns:
            return ExpansionQueryResult(
                status=f"MISSING_COLUMN_{long_exp_col}",
                neighbors=len(neighbor_indices),
                distance_mean=neighbors_result.get("distance_mean", 0.0),
                long_expansion_rate=0.0,
                short_expansion_rate=0.0,
                expansion_rate=0.0,
                direction="NONE",
                mfe_given_expansion=0.0,
                avg_time_to_expansion=None,
                mean_mfe=0.0,
                mfe_40pct=0.0,
            )

        # Get expansion labels for neighbor indices (using .loc since indices are timestamps)
        neighbor_expansions = self.expansion_df.loc[neighbor_indices]

        # Step 3: Compute expansion rates
        long_expansion_rate = neighbor_expansions[long_exp_col].mean()
        short_expansion_rate = neighbor_expansions[short_exp_col].mean()

        # Direction is whichever has higher expansion rate
        if long_expansion_rate >= short_expansion_rate:
            direction = "LONG"
            expansion_rate = long_expansion_rate
            exp_col = long_exp_col
            time_col = long_time_col
            mfe_col = f"mfe_long_{suffix}"
        else:
            direction = "SHORT"
            expansion_rate = short_expansion_rate
            exp_col = short_exp_col
            time_col = short_time_col
            mfe_col = f"mfe_short_{suffix}"

        # Step 4: Conditional stats (only from expanding neighbors)
        expanding_mask = neighbor_expansions[exp_col] == 1
        expanding_indices = [neighbor_indices[i] for i, m in enumerate(expanding_mask) if m]

        if len(expanding_indices) > 0 and mfe_col in self.outcome_df.columns:
            expanding_outcomes = self.outcome_df.loc[expanding_indices]
            mfe_given_expansion = expanding_outcomes[mfe_col].mean()

            expanding_times = neighbor_expansions.loc[expanding_mask, time_col]
            avg_time = expanding_times.mean() if len(expanding_times) > 0 else None
        else:
            mfe_given_expansion = 0.0
            avg_time = None

        # Step 5: Legacy stats for TP sizing
        neighbor_outcomes = self.outcome_df.loc[neighbor_indices]
        if mfe_col in neighbor_outcomes.columns:
            mean_mfe = neighbor_outcomes[mfe_col].mean()
            mfe_40pct = neighbor_outcomes[mfe_col].quantile(0.40)
        else:
            mean_mfe = 0.0
            mfe_40pct = 0.0

        return ExpansionQueryResult(
            status="OK",
            neighbors=len(neighbor_indices),
            distance_mean=neighbors_result.get("distance_mean", 0.0),
            long_expansion_rate=long_expansion_rate,
            short_expansion_rate=short_expansion_rate,
            expansion_rate=expansion_rate,
            direction=direction,
            mfe_given_expansion=mfe_given_expansion,
            avg_time_to_expansion=avg_time,
            mean_mfe=mean_mfe,
            mfe_40pct=mfe_40pct,
        )

    def should_trade(
        self,
        result: ExpansionQueryResult,
        min_expansion_rate: float = 0.55,
        min_neighbors: int = 50,
    ) -> Optional[Dict]:
        """
        Determine if we should trade based on expansion query result.

        Args:
            result: ExpansionQueryResult from query()
            min_expansion_rate: Minimum expansion rate to trade (0.55 = 55%)
            min_neighbors: Minimum neighbors required

        Returns:
            Trade dict if should trade, None otherwise
        """
        # Check status
        if result.status != "OK":
            return None

        # Check neighbor count
        if result.neighbors < min_neighbors:
            return None

        # Primary gate: expansion rate
        if result.expansion_rate < min_expansion_rate:
            return None

        # We have a trade signal
        # TP from conditional MFE (only expanding neighbors)
        tp = result.mfe_given_expansion
        if tp <= 0:
            tp = result.mfe_40pct if result.mfe_40pct > 0 else 0.001

        # SL from ratio (e.g., 50% of TP)
        sl = tp * 0.5

        return {
            "action": "TRADE",
            "direction": result.direction,
            "expansion_rate": result.expansion_rate,
            "tp": tp,
            "sl": sl,
            "neighbors": result.neighbors,
            "avg_time_to_expansion": result.avg_time_to_expansion,
        }
