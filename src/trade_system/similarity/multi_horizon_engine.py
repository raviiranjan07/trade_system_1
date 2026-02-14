"""
Multi-Horizon Similarity Engine V2.

Queries the same state against multiple horizon datasets and selects
the best horizon using time-normalized comparison and consistency filtering.

V2 Improvements:
- Time-normalized comparison (expectancy per bar, not raw MFE)
- Consistency filtering (win_rate, coefficient of variation)
- Configurable TP percentile (40th default, more achievable)
- Full explainability for every trade decision

Usage:
    from trade_system.similarity import MultiHorizonEngine

    engine = MultiHorizonEngine(
        outcome_df=outcome_df,
        regime_df=regime_df,
        horizons=[3, 5, 10],
        k=150,
        comparison_metric="expectancy_per_bar",
        tp_percentile=0.40,
        min_win_rate=0.55,
        max_mfe_cv=2.0,
    )

    result = engine.query_best_horizon(
        current_state=state_row,
        regime="TREND_HIGH_VOL",
        max_timestamp=timestamp
    )

    if result["action"] == "TRADE":
        print(f"Trade: {result['direction']} H={result['horizon']}")
        print(f"Explanation: {result['explanation']}")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal, Tuple, Any

from .similarity_engine import SimilarityEngine, STATE_COLUMNS


class MultiHorizonEngine:
    """
    Adaptive horizon selection engine V2.

    Key improvements over V1:
    1. Time-normalized comparison: expectancy/horizon instead of raw MFE
    2. Consistency filtering: win_rate and CV filters
    3. Configurable TP percentile: 40th (achievable) vs 60th (aggressive)
    4. Full explainability: every trade decision is explainable
    """

    def __init__(
        self,
        outcome_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        horizons: List[int] = [3, 5, 10],
        k: int = 150,
        backend: Literal["bruteforce", "faiss"] = "faiss",
        faiss_nlist: int = 100,
        faiss_nprobe: int = 10,
        use_gpu: bool = False,
        # V2: New parameters
        comparison_metric: Literal[
            "expectancy_per_bar",
            "risk_adjusted_per_bar",
            "win_rate_weighted",
            "raw_expectancy"
        ] = "expectancy_per_bar",
        tp_percentile: float = 0.40,
        min_win_rate: float = 0.55,
        max_mfe_cv: float = 2.0,
        min_neighbors: int = 50,
        min_expectancy: float = 0.0,
    ):
        """
        Initialize multi-horizon engine V2.

        Args:
            outcome_df: DataFrame with state vectors and MFE/MAE columns for all horizons
            regime_df: DataFrame with regime labels
            horizons: List of horizons to compare (e.g., [3, 5, 10])
            k: Number of nearest neighbors for each query
            backend: "bruteforce" or "faiss"
            faiss_nlist: Number of IVF clusters
            faiss_nprobe: Clusters to search at query time
            use_gpu: Use GPU for FAISS
            comparison_metric: How to compare horizons:
                - "expectancy_per_bar": expectancy / horizon (default, fair comparison)
                - "risk_adjusted_per_bar": (expectancy / horizon) / |mae| (Sharpe-like)
                - "win_rate_weighted": (expectancy / horizon) * win_rate
                - "raw_expectancy": No normalization (V1 behavior, not recommended)
            tp_percentile: Percentile for take profit (0.40 = 60% of trades reach it)
            min_win_rate: Minimum win rate to accept a horizon (0.55 = 55%)
            max_mfe_cv: Maximum coefficient of variation for MFE (2.0 = std < 2*mean)
            min_neighbors: Minimum neighbors required for valid statistics
            min_expectancy: Minimum expectancy to consider a horizon
        """
        self.horizons = sorted(horizons)
        self.k = k
        self.comparison_metric = comparison_metric
        self.tp_percentile = tp_percentile
        self.min_win_rate = min_win_rate
        self.max_mfe_cv = max_mfe_cv
        self.min_neighbors = min_neighbors
        self.min_expectancy = min_expectancy

        # Validate TP percentile
        if tp_percentile not in [0.40, 0.50, 0.60]:
            print(f"Warning: tp_percentile={tp_percentile} not in [0.40, 0.50, 0.60], using 0.40")
            self.tp_percentile = 0.40

        # Validate that outcome_df has columns for all horizons
        self._validate_horizon_columns(outcome_df)

        # Create the underlying SimilarityEngine
        self.similarity_engine = SimilarityEngine(
            outcome_df=outcome_df,
            regime_df=regime_df,
            k=k,
            backend=backend,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
            use_gpu=use_gpu,
        )

        print(f"MultiHorizonEngine V2 initialized:")
        print(f"  Horizons: {self.horizons}")
        print(f"  Comparison: {self.comparison_metric}")
        print(f"  TP percentile: {int(self.tp_percentile * 100)}th")
        print(f"  Filters: win_rate >= {self.min_win_rate}, CV <= {self.max_mfe_cv}")

    def _validate_horizon_columns(self, outcome_df: pd.DataFrame) -> None:
        """Validate that outcome_df has MFE/MAE columns for all requested horizons."""
        missing = []
        for h in self.horizons:
            mfe_col = f"mfe_long_{h}m"
            mae_col = f"mae_long_{h}m"
            if mfe_col not in outcome_df.columns:
                missing.append(mfe_col)
            if mae_col not in outcome_df.columns:
                missing.append(mae_col)

        if missing:
            available = [c for c in outcome_df.columns if "mfe" in c or "mae" in c]
            raise ValueError(
                f"Missing outcome columns: {missing}\n"
                f"Available: {available}\n"
                f"Make sure outcome_df was generated with horizons: {self.horizons}"
            )

    def query_all_horizons(
        self,
        current_state: pd.Series,
        regime: str,
        max_timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict[int, Dict]:
        """
        Query the same state against all horizons.

        Returns:
            Dict mapping horizon -> result dict with stats for long/short
        """
        results = {}
        for h in self.horizons:
            result = self.similarity_engine.query(
                current_state=current_state,
                regime=regime,
                horizon=h,
                max_timestamp=max_timestamp,
            )
            results[h] = result
        return results

    def select_best_horizon(
        self,
        horizon_results: Dict[int, Dict],
        current_state: pd.Series,
        regime: str,
    ) -> Tuple[Optional[int], Dict, Dict]:
        """
        Select the best horizon using V2 logic.

        Returns:
            (best_horizon, stats, explanation) or (None, {}, explanation) if no valid horizon
        """
        candidates = []
        horizon_details = {}  # For explainability

        for h, result in horizon_results.items():
            h_detail = {"horizon": h, "passed_filters": False, "reject_reasons": []}

            # Check 1: Query status
            if result.get("status") != "OK":
                h_detail["reject_reasons"].append(f"Query failed: {result.get('status')}")
                horizon_details[h] = h_detail
                continue

            # Check 2: Minimum neighbors
            neighbors = result.get("neighbors", 0)
            if neighbors < self.min_neighbors:
                h_detail["reject_reasons"].append(
                    f"Neighbors {neighbors} < min {self.min_neighbors}"
                )
                horizon_details[h] = h_detail
                continue

            # Get direction stats
            long_stats = result.get("long", {})
            short_stats = result.get("short", {})

            long_mfe = long_stats.get("mean_mfe", 0)
            short_mfe = short_stats.get("mean_mfe", 0)

            # Pick better direction
            if long_mfe >= short_mfe:
                direction = "LONG"
                stats = long_stats
            else:
                direction = "SHORT"
                stats = short_stats

            # Extract stats
            mean_mfe = stats.get("mean_mfe", 0)
            mfe_std = stats.get("mfe_std", 0)
            win_rate = stats.get("win_rate", 0)
            expectancy = stats.get("expectancy", 0)
            mean_mae = stats.get("mean_mae", 0)
            mae_5pct = stats.get("mae_5pct", 0)

            # Get TP based on configured percentile
            tp_key = f"mfe_{int(self.tp_percentile * 100)}pct"
            mfe_for_tp = stats.get(tp_key, stats.get("mfe_40pct", mean_mfe))

            # Calculate coefficient of variation
            cv = (mfe_std / mean_mfe) if mean_mfe > 0 else float('inf')

            # Store details for explainability
            h_detail.update({
                "direction": direction,
                "mean_mfe": round(mean_mfe, 6),
                "expectancy": round(expectancy, 6),
                "win_rate": round(win_rate, 3),
                "cv": round(cv, 2) if cv != float('inf') else "inf",
                "neighbors": neighbors,
                "distance_mean": round(result.get("distance_mean", 0), 3),
            })

            # Check 3: Minimum expectancy
            if expectancy < self.min_expectancy:
                h_detail["reject_reasons"].append(
                    f"Expectancy {expectancy:.5f} < min {self.min_expectancy}"
                )
                horizon_details[h] = h_detail
                continue

            # Check 4: Win rate filter
            if win_rate < self.min_win_rate:
                h_detail["reject_reasons"].append(
                    f"Win rate {win_rate:.1%} < min {self.min_win_rate:.1%}"
                )
                horizon_details[h] = h_detail
                continue

            # Check 5: Consistency filter (CV)
            if cv > self.max_mfe_cv:
                h_detail["reject_reasons"].append(
                    f"CV {cv:.2f} > max {self.max_mfe_cv} (too variable)"
                )
                horizon_details[h] = h_detail
                continue

            # All filters passed!
            h_detail["passed_filters"] = True

            # Calculate score based on comparison metric
            if self.comparison_metric == "expectancy_per_bar":
                score = expectancy / h
            elif self.comparison_metric == "risk_adjusted_per_bar":
                mae_abs = abs(mean_mae) if mean_mae != 0 else 0.001
                score = (expectancy / h) / mae_abs
            elif self.comparison_metric == "win_rate_weighted":
                score = (expectancy / h) * win_rate
            else:  # raw_expectancy
                score = expectancy

            h_detail["score"] = round(score, 8)
            horizon_details[h] = h_detail

            candidates.append({
                "horizon": h,
                "direction": direction,
                "score": score,
                "mfe": mfe_for_tp,
                "mae": mean_mae,
                "expectancy": expectancy,
                "mae_5pct": mae_5pct,
                "win_rate": win_rate,
                "cv": cv,
                "neighbors": neighbors,
                "distance_mean": result.get("distance_mean", 0),
            })

        # Build explanation
        explanation = {
            "state_summary": self._summarize_state(current_state),
            "regime": regime,
            "horizon_comparison": horizon_details,
            "why_trade_passed": [],
            "why_horizon_chosen": [],
        }

        if not candidates:
            explanation["why_trade_passed"].append("NO TRADE: No horizon passed all filters")
            return None, {}, explanation

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        # Build explanation for why this trade passed
        explanation["why_trade_passed"] = [
            f"Regime '{regime}' is allowed",
            f"At least one horizon passed all filters",
            f"Best horizon H={best['horizon']} has positive score",
        ]

        # Build explanation for why this horizon was chosen
        explanation["why_horizon_chosen"] = [
            f"H={best['horizon']} has highest {self.comparison_metric} score: {best['score']:.6f}",
            f"Win rate {best['win_rate']:.1%} >= min {self.min_win_rate:.1%}",
            f"CV {best['cv']:.2f} <= max {self.max_mfe_cv}",
        ]

        # Add comparison to other candidates
        for c in candidates[1:]:
            explanation["why_horizon_chosen"].append(
                f"Beat H={c['horizon']} (score {c['score']:.6f})"
            )

        # Add rejected horizons
        for h, detail in horizon_details.items():
            if not detail.get("passed_filters", False) and detail.get("reject_reasons"):
                explanation["why_horizon_chosen"].append(
                    f"H={h} rejected: {detail['reject_reasons'][0]}"
                )

        return best["horizon"], best, explanation

    def query_best_horizon(
        self,
        current_state: pd.Series,
        regime: str,
        max_timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict:
        """
        Full adaptive query with explainability.

        Returns:
            Dict with trade decision and full explanation:
            {
                "action": "TRADE" or "NO_TRADE",
                "direction": "LONG" or "SHORT",
                "horizon": 5,
                "mfe": 0.0015,
                "mae": -0.0008,
                "expectancy": 0.0007,
                "score": 0.00014,
                "neighbors": 150,
                "explanation": {
                    "state_summary": {...},
                    "regime": "TREND_HIGH_VOL",
                    "horizon_comparison": {...},
                    "why_trade_passed": [...],
                    "why_horizon_chosen": [...],
                }
            }
        """
        # Step 1: Query all horizons
        horizon_results = self.query_all_horizons(
            current_state=current_state,
            regime=regime,
            max_timestamp=max_timestamp,
        )

        # Step 2: Select best horizon with explanation
        best_h, stats, explanation = self.select_best_horizon(
            horizon_results=horizon_results,
            current_state=current_state,
            regime=regime,
        )

        # Add timestamp to explanation
        explanation["timestamp"] = str(max_timestamp) if max_timestamp else None

        if best_h is None:
            return {
                "action": "NO_TRADE",
                "reason": "no_qualifying_horizons",
                "explanation": explanation,
            }

        # Step 3: Build trade decision
        return {
            "action": "TRADE",
            "direction": stats["direction"],
            "horizon": best_h,
            "mfe": stats["mfe"],
            "mae": stats["mae"],
            "expectancy": stats["expectancy"],
            "mae_5pct": stats["mae_5pct"],
            "score": stats["score"],
            "win_rate": stats["win_rate"],
            "cv": stats["cv"],
            "neighbors": stats["neighbors"],
            "distance_mean": stats["distance_mean"],
            "reason": f"Best score at H={best_h} ({self.comparison_metric})",
            "explanation": explanation,
        }

    def _summarize_state(self, state: pd.Series) -> Dict[str, str]:
        """Convert state vector to human-readable summary."""
        return {
            "trend": self._describe_trend(state),
            "momentum": self._describe_momentum(state),
            "volatility": self._describe_volatility(state),
            "volume": self._describe_volume(state),
            "location": self._describe_location(state),
        }

    def _describe_trend(self, state: pd.Series) -> str:
        """Describe trend in plain English."""
        alignment = state.get("trend_alignment", 0)
        ema50_slope = state.get("ema50_slope_z", 0)
        ema200_slope = state.get("ema200_slope_z", 0)

        if alignment > 0 and ema50_slope > 0.5:
            strength = "strong" if ema50_slope > 1.5 else "moderate"
            return f"bullish ({strength}, alignment={alignment:.0f}, slope_z={ema50_slope:.1f})"
        elif alignment < 0 and ema50_slope < -0.5:
            strength = "strong" if ema50_slope < -1.5 else "moderate"
            return f"bearish ({strength}, alignment={alignment:.0f}, slope_z={ema50_slope:.1f})"
        else:
            return f"neutral (alignment={alignment:.0f}, slope_z={ema50_slope:.1f})"

    def _describe_momentum(self, state: pd.Series) -> str:
        """Describe momentum in plain English."""
        ret_5m = state.get("return_5m_z", 0)
        ret_15m = state.get("return_15m_z", 0)
        rsi = state.get("rsi_z", 0)

        if ret_5m > 1.5 and ret_15m > 1.0:
            return f"strong bullish (ret5m_z={ret_5m:.1f}, ret15m_z={ret_15m:.1f})"
        elif ret_5m < -1.5 and ret_15m < -1.0:
            return f"strong bearish (ret5m_z={ret_5m:.1f}, ret15m_z={ret_15m:.1f})"
        elif abs(ret_5m) > 1.0:
            direction = "bullish" if ret_5m > 0 else "bearish"
            return f"moderate {direction} (ret5m_z={ret_5m:.1f}, rsi_z={rsi:.1f})"
        else:
            return f"weak (ret5m_z={ret_5m:.1f}, rsi_z={rsi:.1f})"

    def _describe_volatility(self, state: pd.Series) -> str:
        """Describe volatility in plain English."""
        atr_pct = state.get("atr_percentile", 0.5)

        if atr_pct >= 0.85:
            return f"very high (atr_pct={atr_pct:.2f})"
        elif atr_pct >= 0.65:
            return f"high (atr_pct={atr_pct:.2f})"
        elif atr_pct >= 0.35:
            return f"medium (atr_pct={atr_pct:.2f})"
        else:
            return f"low (atr_pct={atr_pct:.2f})"

    def _describe_volume(self, state: pd.Series) -> str:
        """Describe volume in plain English."""
        vol_z = state.get("volume_z", 0)

        if vol_z > 2.0:
            return f"very high (z={vol_z:.1f})"
        elif vol_z > 1.0:
            return f"above average (z={vol_z:.1f})"
        elif vol_z > -1.0:
            return f"normal (z={vol_z:.1f})"
        else:
            return f"low (z={vol_z:.1f})"

    def _describe_location(self, state: pd.Series) -> str:
        """Describe price location in plain English."""
        vwap_dist = state.get("vwap_distance_z", 0)
        range_pos = state.get("range_position", 0.5)

        if range_pos > 0.8:
            loc = "near range high"
        elif range_pos < 0.2:
            loc = "near range low"
        else:
            loc = "mid-range"

        vwap_desc = "above VWAP" if vwap_dist > 0.5 else "below VWAP" if vwap_dist < -0.5 else "near VWAP"

        return f"{loc}, {vwap_desc} (range_pos={range_pos:.2f}, vwap_z={vwap_dist:.1f})"

    # Legacy method for backward compatibility
    def query(
        self,
        current_state: pd.Series,
        regime: str,
        horizon: int = 5,
        max_timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict:
        """Legacy query method - delegates to similarity engine."""
        return self.similarity_engine.query(
            current_state=current_state,
            regime=regime,
            horizon=horizon,
            max_timestamp=max_timestamp,
        )
