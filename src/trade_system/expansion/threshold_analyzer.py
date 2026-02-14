"""
Threshold Analyzer - Determines optimal expansion thresholds from price data.

This module analyzes historical price movements to determine:
1. Expansion threshold: What constitutes a "real move" (not noise)
2. Invalidation threshold: When to give up on an expansion

The thresholds are derived from data distribution, not guesswork.

Usage:
    analyzer = ThresholdAnalyzer(ohlcv_df)

    # Analyze single horizon
    result = analyzer.analyze(horizon=5, percentile=0.75)
    print(f"Expansion: {result.expansion_bps} bps")

    # Analyze multiple horizons
    results = analyzer.analyze_multiple(horizons=[3, 5, 10])
    save_thresholds(results, "data/expansion/thresholds.json")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ThresholdResult:
    """Result of threshold analysis for one horizon."""

    horizon: int
    expansion_pct: float  # e.g., 0.0025 (25 bps)
    invalidation_pct: float  # e.g., 0.0010 (10 bps)
    sample_size: int
    percentile_used: float  # e.g., 0.75

    # Distribution stats for reference
    median_move: float = 0.0
    p75_move: float = 0.0
    p90_move: float = 0.0

    @property
    def expansion_bps(self) -> float:
        """Expansion threshold in basis points."""
        return round(self.expansion_pct * 10000, 1)

    @property
    def invalidation_bps(self) -> float:
        """Invalidation threshold in basis points."""
        return round(self.invalidation_pct * 10000, 1)

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "expansion_pct": self.expansion_pct,
            "expansion_bps": self.expansion_bps,
            "invalidation_pct": self.invalidation_pct,
            "invalidation_bps": self.invalidation_bps,
            "sample_size": self.sample_size,
            "percentile_used": self.percentile_used,
            "median_move": round(self.median_move, 6),
            "p75_move": round(self.p75_move, 6),
            "p90_move": round(self.p90_move, 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdResult":
        return cls(
            horizon=d["horizon"],
            expansion_pct=d["expansion_pct"],
            invalidation_pct=d["invalidation_pct"],
            sample_size=d["sample_size"],
            percentile_used=d["percentile_used"],
            median_move=d.get("median_move", 0.0),
            p75_move=d.get("p75_move", 0.0),
            p90_move=d.get("p90_move", 0.0),
        )


class ThresholdAnalyzer:
    """
    Analyzes price data to determine expansion thresholds.

    The key insight is that thresholds should be data-driven:
    - Expansion threshold = 75th percentile of max moves (top 25% of moves)
    - Invalidation threshold = 50% of median move (conservative)

    This ensures we only label "real" expansions, not noise.

    Args:
        ohlcv: DataFrame with 'open', 'high', 'low', 'close' columns

    Example:
        analyzer = ThresholdAnalyzer(ohlcv)
        result = analyzer.analyze(horizon=5)
        print(f"H=5: expansion={result.expansion_bps} bps")
    """

    def __init__(self, ohlcv: pd.DataFrame):
        self._validate_ohlcv(ohlcv)
        self.ohlcv = ohlcv
        self.close = ohlcv["close"].values
        self.high = ohlcv["high"].values
        self.low = ohlcv["low"].values
        self.n = len(ohlcv)

    def _validate_ohlcv(self, ohlcv: pd.DataFrame) -> None:
        """Validate OHLCV DataFrame has required columns."""
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        if len(ohlcv) < 100:
            raise ValueError(f"Need at least 100 rows, got {len(ohlcv)}")

    def compute_max_moves(self, horizon: int) -> pd.DataFrame:
        """
        Compute max up/down moves for each bar within next H bars.

        For each bar, we look forward H bars and find:
        - max_up_pct: (max_high - close) / close
        - max_down_pct: (close - min_low) / close

        Args:
            horizon: Number of bars to look forward

        Returns:
            DataFrame with 'max_up_pct', 'max_down_pct' columns
        """
        n = self.n
        max_up = np.full(n, np.nan)
        max_down = np.full(n, np.nan)

        for i in range(n - horizon):
            entry = self.close[i]

            # Look at next H bars (excluding current bar)
            future_high = np.max(self.high[i + 1 : i + 1 + horizon])
            future_low = np.min(self.low[i + 1 : i + 1 + horizon])

            max_up[i] = (future_high - entry) / entry
            max_down[i] = (entry - future_low) / entry

        return pd.DataFrame(
            {"max_up_pct": max_up, "max_down_pct": max_down}, index=self.ohlcv.index
        )

    def analyze(
        self,
        horizon: int,
        expansion_percentile: float = 0.75,
        invalidation_ratio: float = 0.5,
    ) -> ThresholdResult:
        """
        Analyze data and determine thresholds for a horizon.

        Logic:
        - Expansion threshold = percentile of max moves (e.g., 75th = top 25%)
        - Invalidation threshold = median × ratio (e.g., 50% of median)

        Args:
            horizon: Number of bars to analyze
            expansion_percentile: Percentile for expansion threshold (0.75 = top 25%)
            invalidation_ratio: Multiply median by this for invalidation (0.5 = half)

        Returns:
            ThresholdResult with computed thresholds
        """
        moves = self.compute_max_moves(horizon)

        # Combine up and down moves (market should be symmetric)
        up = moves["max_up_pct"].dropna()
        down = moves["max_down_pct"].dropna()
        all_moves = pd.concat([up, down])

        # Compute thresholds
        expansion_pct = all_moves.quantile(expansion_percentile)
        median_move = all_moves.quantile(0.50)
        invalidation_pct = median_move * invalidation_ratio

        # Get distribution stats
        p75_move = all_moves.quantile(0.75)
        p90_move = all_moves.quantile(0.90)

        return ThresholdResult(
            horizon=horizon,
            expansion_pct=round(expansion_pct, 6),
            invalidation_pct=round(invalidation_pct, 6),
            sample_size=len(all_moves),
            percentile_used=expansion_percentile,
            median_move=median_move,
            p75_move=p75_move,
            p90_move=p90_move,
        )

    def analyze_multiple(
        self,
        horizons: List[int],
        expansion_percentile: float = 0.75,
        invalidation_ratio: float = 0.5,
    ) -> Dict[int, ThresholdResult]:
        """
        Analyze multiple horizons.

        Args:
            horizons: List of horizons to analyze (e.g., [3, 5, 10])
            expansion_percentile: Percentile for expansion threshold
            invalidation_ratio: Ratio for invalidation threshold

        Returns:
            Dict mapping horizon -> ThresholdResult
        """
        results = {}
        for h in horizons:
            results[h] = self.analyze(h, expansion_percentile, invalidation_ratio)
        return results

    def get_distribution_stats(self, horizon: int) -> dict:
        """
        Get detailed distribution statistics for a horizon.

        Useful for understanding the data before picking thresholds.
        """
        moves = self.compute_max_moves(horizon)
        up = moves["max_up_pct"].dropna()
        down = moves["max_down_pct"].dropna()

        def get_stats(series: pd.Series) -> dict:
            return {
                "count": len(series),
                "mean": round(series.mean(), 6),
                "std": round(series.std(), 6),
                "min": round(series.min(), 6),
                "p25": round(series.quantile(0.25), 6),
                "p50": round(series.quantile(0.50), 6),
                "p75": round(series.quantile(0.75), 6),
                "p90": round(series.quantile(0.90), 6),
                "p95": round(series.quantile(0.95), 6),
                "max": round(series.max(), 6),
            }

        return {
            "horizon": horizon,
            "up_moves": get_stats(up),
            "down_moves": get_stats(down),
        }

    def print_analysis(self, horizons: List[int]) -> None:
        """Print formatted analysis for multiple horizons."""
        print("\n" + "=" * 60)
        print("THRESHOLD ANALYSIS")
        print("=" * 60)

        for h in horizons:
            result = self.analyze(h)
            stats = self.get_distribution_stats(h)

            print(f"\nHorizon = {h} bars")
            print("-" * 40)
            print(f"  Sample size: {result.sample_size:,}")
            print(f"  Median move: {result.median_move * 10000:.1f} bps")
            print(f"  75th pct:    {result.p75_move * 10000:.1f} bps")
            print(f"  90th pct:    {result.p90_move * 10000:.1f} bps")
            print()
            print(f"  RECOMMENDED THRESHOLDS:")
            print(f"    Expansion:    {result.expansion_bps} bps ({result.expansion_pct:.4%})")
            print(f"    Invalidation: {result.invalidation_bps} bps ({result.invalidation_pct:.4%})")


def save_thresholds(
    results: Dict[int, ThresholdResult],
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save threshold results to JSON file.

    Args:
        results: Dict mapping horizon -> ThresholdResult
        output_path: Path to save JSON file
        metadata: Optional metadata to include
    """
    data = {f"h{h}": r.to_dict() for h, r in results.items()}

    data["_meta"] = metadata or {}
    data["_meta"]["created"] = pd.Timestamp.now().isoformat()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_thresholds(path: str) -> Dict[int, ThresholdResult]:
    """
    Load thresholds from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dict mapping horizon -> ThresholdResult
    """
    with open(path) as f:
        data = json.load(f)

    results = {}
    for key, value in data.items():
        if key.startswith("h") and key[1:].isdigit():
            horizon = int(key[1:])
            results[horizon] = ThresholdResult.from_dict(value)

    return results
