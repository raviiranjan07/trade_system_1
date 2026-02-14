"""
Risk Profiler: Compute risk profile from similar historical states.

Instead of predicting direction, we compute:
- P(Case 1) = probability of wrong direction
- P(Case 2) = probability of quick recovery
- P(Case 3) = probability of slow recovery
- Expected MAE distribution
- Expected recovery time

This enables risk-based entry filtering instead of direction prediction.
"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from .similarity_engine import SimilarityEngine, STATE_COLUMNS


class RiskProfiler:
    """
    Computes risk profile based on K similar historical states.

    The key insight: Instead of asking "Will price go up?",
    we ask "What is the risk profile of similar historical states?"
    """

    def __init__(
        self,
        similarity_engine: SimilarityEngine,
        case_df: pd.DataFrame,
        default_target_bps: int = 15,
        default_horizon: int = 10
    ):
        """
        Args:
            similarity_engine: Pre-built SimilarityEngine for finding neighbors
            case_df: DataFrame with case labels (from case_labeler.py)
            default_target_bps: Default target in basis points
            default_horizon: Default horizon in bars
        """
        self.engine = similarity_engine
        self.case_df = case_df
        self.default_target_bps = default_target_bps
        self.default_horizon = default_horizon

        # Verify case_df has required columns
        sample_col = f"t{default_target_bps}_H{default_horizon}_case"
        if sample_col not in case_df.columns:
            available_cols = [c for c in case_df.columns if c.endswith("_case")]
            raise ValueError(
                f"Case column '{sample_col}' not found. "
                f"Available case columns: {available_cols[:5]}..."
            )

    def get_risk_profile(
        self,
        current_state: pd.Series,
        regime: str,
        target_bps: Optional[int] = None,
        H: Optional[int] = None,
        max_timestamp: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Compute risk profile based on K similar historical states.

        Args:
            current_state: Current state vector (10D)
            regime: Market regime
            target_bps: Target in basis points (default: 15)
            H: Horizon in bars (default: 10)
            max_timestamp: Only use historical states before this time

        Returns:
            Dictionary with risk profile:
            {
                "status": "OK" or error,
                "neighbors": K,

                # Case probabilities
                "p_case0": float,  # P(clean win)
                "p_case1": float,  # P(wrong direction)
                "p_case2": float,  # P(quick recovery)
                "p_case3": float,  # P(slow recovery)
                "p_recovery": float,  # P(Case 2 or 3)

                # MAE statistics
                "mae_median": float,
                "mae_75pct": float,
                "mae_90pct": float,

                # Recovery time statistics (for Case 2/3)
                "recovery_median": float,
                "recovery_75pct": float,
                "recovery_90pct": float,

                # Composite risk score
                "risk_score": float,  # 0-1, lower is better
                "recommendation": "ENTER" or "SKIP"
            }
        """
        target_bps = target_bps or self.default_target_bps
        H = H or self.default_horizon

        # Step 1: Find K similar neighbors using existing similarity engine
        # Use a placeholder horizon for the MFE/MAE query (we'll override with case data)
        sim_result = self.engine.query(
            current_state=current_state,
            regime=regime,
            horizon=H,  # This is for MFE/MAE cols, but we'll use case labels
            max_timestamp=max_timestamp
        )

        if sim_result.get("status") != "OK":
            return {
                "status": sim_result.get("status", "ERROR"),
                "error": sim_result.get("error", "Similarity query failed"),
                "recommendation": "SKIP"
            }

        # Step 2: Get neighbor indices from similarity result
        neighbor_indices = sim_result.get("_neighbor_indices", [])

        if len(neighbor_indices) == 0:
            return {
                "status": "NO_NEIGHBORS",
                "recommendation": "SKIP"
            }

        # Step 3: Look up case labels for these neighbors
        col_prefix = f"t{target_bps}_H{H}"
        case_col = f"{col_prefix}_case"
        mae_col = f"{col_prefix}_mae"
        bars_col = f"{col_prefix}_bars"

        # Filter to neighbors that exist in case_df
        valid_indices = [idx for idx in neighbor_indices if idx in self.case_df.index]

        if len(valid_indices) == 0:
            return {
                "status": "NO_CASE_DATA",
                "recommendation": "SKIP"
            }

        neighbor_cases = self.case_df.loc[valid_indices, case_col]
        neighbor_maes = self.case_df.loc[valid_indices, mae_col]
        neighbor_bars = self.case_df.loc[valid_indices, bars_col]

        # Filter out invalid cases
        valid_mask = neighbor_cases >= 0
        neighbor_cases = neighbor_cases[valid_mask]
        neighbor_maes = neighbor_maes[valid_mask]
        neighbor_bars = neighbor_bars[valid_mask]

        if len(neighbor_cases) == 0:
            return {
                "status": "NO_VALID_CASES",
                "recommendation": "SKIP"
            }

        # Step 4: Compute case probabilities
        total = len(neighbor_cases)
        case_counts = neighbor_cases.value_counts()

        p_case0 = case_counts.get(0, 0) / total
        p_case1 = case_counts.get(1, 0) / total
        p_case2 = case_counts.get(2, 0) / total
        p_case3 = case_counts.get(3, 0) / total
        p_recovery = p_case2 + p_case3

        # Step 5: Compute MAE statistics
        mae_median = float(neighbor_maes.median())
        mae_75pct = float(neighbor_maes.quantile(0.75))
        mae_90pct = float(neighbor_maes.quantile(0.90))

        # Step 6: Compute recovery time statistics (for Case 2/3 only)
        recovery_mask = (neighbor_cases == 2) | (neighbor_cases == 3)
        recovery_bars = neighbor_bars[recovery_mask]
        recovery_bars = recovery_bars[recovery_bars > 0]  # Filter invalid

        if len(recovery_bars) > 0:
            recovery_median = float(recovery_bars.median())
            recovery_75pct = float(recovery_bars.quantile(0.75))
            recovery_90pct = float(recovery_bars.quantile(0.90))
        else:
            recovery_median = float('nan')
            recovery_75pct = float('nan')
            recovery_90pct = float('nan')

        # Step 7: Compute composite risk score (0-1, lower is better)
        # Risk factors: P(Case1), MAE, slow recovery
        risk_score = self._compute_risk_score(
            p_case1=p_case1,
            mae_median=mae_median,
            recovery_median=recovery_median
        )

        # Step 8: Generate recommendation
        recommendation = self._get_recommendation(
            p_case1=p_case1,
            mae_median=mae_median,
            recovery_median=recovery_median
        )

        # Step 9: Determine direction and EDGE from similarity engine's MFE analysis
        # Key insight: Only trade if there's a clear directional edge, not 50/50
        long_stats = sim_result.get("long", {})
        short_stats = sim_result.get("short", {})
        long_mfe = long_stats.get("mean_mfe", 0)
        short_mfe = short_stats.get("mean_mfe", 0)

        # Calculate directional edge ratio
        # If long_mfe=30, short_mfe=10 → edge_ratio=3.0 (strong LONG edge)
        # If long_mfe=20, short_mfe=19 → edge_ratio=1.05 (no edge, noise)
        max_mfe = max(long_mfe, short_mfe)
        min_mfe = min(long_mfe, short_mfe)

        if min_mfe > 0:
            edge_ratio = max_mfe / min_mfe
        else:
            edge_ratio = float('inf') if max_mfe > 0 else 1.0

        direction = "LONG" if long_mfe >= short_mfe else "SHORT"

        return {
            "status": "OK",
            "neighbors": total,
            "target_bps": target_bps,
            "horizon": H,

            # Direction and edge (CRITICAL: only trade if edge_ratio > threshold)
            "direction": direction,
            "edge_ratio": float(edge_ratio),  # >1.0 = edge, 1.0 = no edge (50/50)
            "long_mfe": float(long_mfe),
            "short_mfe": float(short_mfe),

            # Case probabilities
            "p_case0": float(p_case0),
            "p_case1": float(p_case1),
            "p_case2": float(p_case2),
            "p_case3": float(p_case3),
            "p_recovery": float(p_recovery),

            # MAE statistics (in bps)
            "mae_median": mae_median,
            "mae_75pct": mae_75pct,
            "mae_90pct": mae_90pct,

            # Recovery time statistics (in bars)
            "recovery_median": recovery_median,
            "recovery_75pct": recovery_75pct,
            "recovery_90pct": recovery_90pct,

            # Risk assessment
            "risk_score": risk_score,
            "recommendation": recommendation,

            # Include original similarity stats for reference
            "distance_mean": sim_result.get("distance_mean", 0),
            "distance_max": sim_result.get("distance_max", 0),
        }

    def _compute_risk_score(
        self,
        p_case1: float,
        mae_median: float,
        recovery_median: float,
        max_p_case1: float = 0.20,
        max_mae: float = 100.0,
        max_recovery: float = 200.0
    ) -> float:
        """
        Compute composite risk score (0-1, lower is better).

        Factors:
        - P(Case 1) weight: 50% (wrong direction is worst)
        - MAE weight: 30% (drawdown matters)
        - Recovery time weight: 20% (slow recovery is less bad)
        """
        # Normalize each factor to 0-1
        p_case1_norm = min(p_case1 / max_p_case1, 1.0)
        mae_norm = min(mae_median / max_mae, 1.0) if not np.isnan(mae_median) else 0.5
        recovery_norm = min(recovery_median / max_recovery, 1.0) if not np.isnan(recovery_median) else 0.5

        # Weighted average
        risk_score = (
            0.50 * p_case1_norm +
            0.30 * mae_norm +
            0.20 * recovery_norm
        )

        return float(risk_score)

    def _get_recommendation(
        self,
        p_case1: float,
        mae_median: float,
        recovery_median: float,
        max_p_case1: float = 0.10,
        max_mae_median: float = 30.0,
        max_recovery_median: float = 50.0
    ) -> str:
        """
        Generate entry recommendation based on risk profile.

        Entry is allowed only if:
        - P(Case 1) < max_p_case1 (default 10%)
        - Median MAE < max_mae_median (default 30bp)
        - Median recovery < max_recovery_median (default 50 bars)
        """
        if p_case1 > max_p_case1:
            return "SKIP"

        if not np.isnan(mae_median) and mae_median > max_mae_median:
            return "SKIP"

        if not np.isnan(recovery_median) and recovery_median > max_recovery_median:
            return "SKIP"

        return "ENTER"

    def get_risk_profile_batch(
        self,
        states_df: pd.DataFrame,
        regimes: pd.Series,
        target_bps: Optional[int] = None,
        H: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute risk profiles for multiple states (batch processing).

        Args:
            states_df: DataFrame of state vectors
            regimes: Series of regime labels (same index as states_df)
            target_bps: Target in basis points
            H: Horizon in bars

        Returns:
            DataFrame with risk profile columns for each state
        """
        results = []

        for idx in states_df.index:
            current_state = states_df.loc[idx]
            regime = regimes.loc[idx]

            profile = self.get_risk_profile(
                current_state=current_state,
                regime=regime,
                target_bps=target_bps,
                H=H,
                max_timestamp=idx  # Use timestamp for look-ahead protection
            )

            profile["timestamp"] = idx
            results.append(profile)

        return pd.DataFrame(results).set_index("timestamp")


if __name__ == "__main__":
    # Test the risk profiler
    print("="*60)
    print("RISK PROFILER TEST")
    print("="*60)
    print("\nThis module requires:")
    print("1. A built SimilarityEngine")
    print("2. Case labels from case_labeler.py")
    print("\nRun the full pipeline to test.")
