import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


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
    """
    KNN-based similarity search engine with two backend options:

    1. "bruteforce" (default): Exact KNN using numpy
       - 100% accurate
       - Slow on large datasets (~50ms per query on 700K samples)

    2. "faiss": Approximate KNN using Facebook's FAISS library
       - ~95-99% accurate (may miss a few true neighbors)
       - Very fast (~0.5ms per query)
       - Requires: pip install faiss-cpu (or faiss-gpu)
    """

    def __init__(
        self,
        outcome_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        k: int = 200,
        backend: Literal["bruteforce", "faiss"] = "bruteforce",
        faiss_nlist: int = 100,  # Number of clusters for IVF
        faiss_nprobe: int = 10,  # Number of clusters to search
    ):
        """
        Args:
            outcome_df: State vectors + outcome labels (MFE / MAE)
            regime_df: Regime labels
            k: Number of nearest neighbors to retrieve
            backend: "bruteforce" for exact KNN, "faiss" for approximate KNN
            faiss_nlist: Number of IVF clusters (higher = more accurate, slower build)
            faiss_nprobe: Clusters to search at query time (higher = more accurate, slower query)
        """
        self.k = k
        self.backend = backend
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe

        # Validate FAISS availability
        if backend == "faiss" and not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS backend requested but faiss is not installed. "
                "Install with: pip install faiss-cpu"
            )

        # Join outcome and regime data
        self.df = (
            outcome_df
            .join(regime_df, how="inner")
            .dropna()
        )

        # FAISS indices (built per regime for efficiency)
        self._faiss_indices: Dict[str, faiss.Index] = {}
        self._faiss_data: Dict[str, pd.DataFrame] = {}

        # Pre-build FAISS indices if using FAISS backend
        if self.backend == "faiss":
            self._build_faiss_indices()

    def _build_faiss_indices(self):
        """Build FAISS IVF indices for each regime."""
        print("Building FAISS indices...")

        regimes = self.df["regime"].unique()
        dim = len(STATE_COLUMNS)

        for regime in regimes:
            df_regime = self.df[self.df["regime"] == regime]

            if len(df_regime) < self.faiss_nlist:
                # Not enough data for IVF, use flat index
                print(f"  {regime}: {len(df_regime):,} samples (using FlatL2)")
                X = df_regime[STATE_COLUMNS].values.astype(np.float32)
                index = faiss.IndexFlatL2(dim)
                index.add(X)
            else:
                # Use IVF index for large datasets
                print(f"  {regime}: {len(df_regime):,} samples (using IVF)")
                X = df_regime[STATE_COLUMNS].values.astype(np.float32)

                # Create IVF index
                quantizer = faiss.IndexFlatL2(dim)
                nlist = min(self.faiss_nlist, len(df_regime) // 40)  # At least 40 samples per cluster
                index = faiss.IndexIVFFlat(quantizer, dim, nlist)

                # Train and add vectors
                index.train(X)
                index.add(X)
                index.nprobe = self.faiss_nprobe

            self._faiss_indices[regime] = index
            self._faiss_data[regime] = df_regime.reset_index()  # Store with original index as column

        print("FAISS indices built successfully.")

    def _euclidean_distance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Vectorized Euclidean distance (dtype-safe)."""
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        diff = X - y
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    def _query_bruteforce(
        self,
        current_state: pd.Series,
        regime: str,
        horizon: int,
        max_timestamp: Optional[pd.Timestamp]
    ) -> Dict:
        """Brute force KNN query (exact)."""

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

    def _query_faiss(
        self,
        current_state: pd.Series,
        regime: str,
        horizon: int,
        max_timestamp: Optional[pd.Timestamp]
    ) -> Dict:
        """FAISS approximate KNN query (fast)."""

        if regime not in self._faiss_indices:
            return {"status": "UNKNOWN_REGIME", "regime": regime}

        index = self._faiss_indices[regime]
        df_regime = self._faiss_data[regime]

        # Handle time boundary for backtesting
        if max_timestamp is not None:
            # Filter data by timestamp
            mask = df_regime["timestamp"] < max_timestamp if "timestamp" in df_regime.columns else df_regime.index < max_timestamp

            # For time-filtered queries, we need to search more and filter
            # This is a trade-off: FAISS doesn't support dynamic filtering
            # So we fetch more neighbors and filter afterward
            df_filtered = df_regime[mask] if "timestamp" in df_regime.columns else df_regime.loc[mask]

            if len(df_filtered) < self.k:
                return {"status": "INSUFFICIENT_DATA", "available": len(df_filtered), "required": self.k}

            # For backtesting with time boundary, fall back to brute force on filtered data
            # This is still faster because we pre-filtered by regime
            X = df_filtered[STATE_COLUMNS].values
            y = current_state[STATE_COLUMNS].values.astype(np.float64)

            distances = self._euclidean_distance(X, y)
            idx = np.argsort(distances)[:self.k]
            neighbors = df_filtered.iloc[idx]

        else:
            # No time boundary - use FAISS directly (production mode)
            if index.ntotal < self.k:
                return {"status": "INSUFFICIENT_DATA", "available": index.ntotal, "required": self.k}

            # Query FAISS
            query_vector = current_state[STATE_COLUMNS].values.astype(np.float32).reshape(1, -1)
            distances, indices = index.search(query_vector, self.k)

            # Get neighbor data
            neighbors = df_regime.iloc[indices[0]]
            distances = np.sqrt(distances[0])  # FAISS returns squared L2

        # Aggregate outcomes
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
            "distance_mean": float(distances.mean()) if isinstance(distances, np.ndarray) else 0.0,
            "distance_max": float(distances.max()) if isinstance(distances, np.ndarray) else 0.0,
        }

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

        Returns:
            Dict with status, neighbor stats, and aggregated MFE/MAE outcomes
        """
        if self.backend == "faiss":
            return self._query_faiss(current_state, regime, horizon, max_timestamp)
        else:
            return self._query_bruteforce(current_state, regime, horizon, max_timestamp)
