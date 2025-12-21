import numpy as np
import pandas as pd
from typing import Dict


STATE_COLUMNS = [
    "ema50_slope_z",
    "ema200_slope_z",
    "trend_alignment",
    "return_5m_z",
    "return_15m_z",
    "rsi_z",
    "atr_percentile",
    "volume_z",
    "vwap_distance_z",
    "range_position",
]


class SimilarityEngine:
    def __init__(
        self,
        outcome_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        k: int = 200
    ):
        """
        outcome_df : state vectors + outcome labels (MFE / MAE)
        regime_df  : regime labels
        """
        self.k = k

        # Outcome df already contains state vectors
        self.df = (
            outcome_df
            .join(regime_df, how="inner")
            .dropna()
        )

    def _euclidean_distance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized Euclidean distance (dtype-safe).
        """
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        diff = X - y
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))


    def query(
        self,
        current_state: pd.Series,
        regime: str,
        horizon: int = 30,
        max_timestamp: pd.Timestamp = None
    ) -> Dict:
        """
        Find similar historical states and aggregate outcomes.

        Args:
            current_state: The state vector to query
            regime: Market regime to filter by
            horizon: Outcome horizon in minutes
            max_timestamp: Only search states BEFORE this time (prevents look-ahead bias)
        """

        # 1. Filter by regime
        df_regime = self.df[self.df["regime"] == regime]

        # 2. Enforce time boundary (prevent look-ahead bias in backtesting)
        if max_timestamp is not None:
            df_regime = df_regime[df_regime.index < max_timestamp]

        if len(df_regime) < self.k:
            return {"status": "INSUFFICIENT_DATA", "available": len(df_regime), "required": self.k}

        # 3. Distance computation
        X = df_regime[STATE_COLUMNS].values
        y = current_state[STATE_COLUMNS].values

        distances = self._euclidean_distance(X, y)

        # 4. Select nearest neighbors
        idx = np.argsort(distances)[: self.k]
        neighbors = df_regime.iloc[idx]

        # 5. Aggregate outcomes
        mfe = neighbors[f"mfe_long_{horizon}m"]
        mae = neighbors[f"mae_long_{horizon}m"]

        expectancy = mfe.mean() + mae.mean()

        return {
            "status": "OK",
            "neighbors": len(neighbors),
            "mean_mfe": float(mfe.mean()),
            "mean_mae": float(mae.mean()),
            "expectancy": float(expectancy),
            "mae_5pct": float(mae.quantile(0.05)),
            "distance_mean": float(distances[idx].mean()),
            "distance_max": float(distances[idx].max()),
        }
