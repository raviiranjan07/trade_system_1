"""
Expansion Labeler - Labels each bar with binary expansion events.

This module labels whether a "real" price expansion occurred after each bar.

Key Concept:
    LONG EXPANSION = 1 if price hits +target BEFORE hitting -invalidation
    SHORT EXPANSION = 1 if price hits -target BEFORE hitting +invalidation

    This is different from MFE/MAE which just looks at max/min in window.
    Expansion labeling is PATH-DEPENDENT - order matters.

Usage:
    config = ExpansionConfig(horizon=5, target_pct=0.003, invalidation_pct=0.0015)
    labeler = ExpansionLabeler(ohlcv_df)
    labels = labeler.label(config)

    # Check expansion rate
    print(f"LONG expansion rate: {labels['long_expansion_5m'].mean():.1%}")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .threshold_analyzer import ThresholdResult


@dataclass
class ExpansionConfig:
    """Configuration for expansion labeling."""

    horizon: int  # Max bars to look forward
    target_pct: float  # Expansion target (e.g., 0.003 = 30 bps)
    invalidation_pct: float  # Invalidation threshold (e.g., 0.0015 = 15 bps)

    @property
    def target_bps(self) -> float:
        return round(self.target_pct * 10000, 1)

    @property
    def invalidation_bps(self) -> float:
        return round(self.invalidation_pct * 10000, 1)

    @classmethod
    def from_threshold_result(cls, result: ThresholdResult) -> "ExpansionConfig":
        """Create config from ThresholdResult."""
        return cls(
            horizon=result.horizon,
            target_pct=result.expansion_pct,
            invalidation_pct=result.invalidation_pct,
        )

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "target_pct": self.target_pct,
            "target_bps": self.target_bps,
            "invalidation_pct": self.invalidation_pct,
            "invalidation_bps": self.invalidation_bps,
        }


@dataclass
class ExpansionLabel:
    """Result of labeling a single bar."""

    long_expansion: int  # 0 or 1
    short_expansion: int  # 0 or 1
    long_time: Optional[int]  # Bars until LONG expansion (None if no expansion)
    short_time: Optional[int]  # Bars until SHORT expansion


class ExpansionLabeler:
    """
    Labels expansion events for each bar in the dataset.

    For each bar, we check:
    - Did price hit +target% before hitting -invalidation%? (LONG expansion)
    - Did price hit -target% before hitting +invalidation%? (SHORT expansion)

    These are checked INDEPENDENTLY - both can be True if price moves enough.

    Args:
        ohlcv: DataFrame with 'high', 'low', 'close' columns

    Example:
        labeler = ExpansionLabeler(ohlcv)
        config = ExpansionConfig(horizon=5, target_pct=0.003, invalidation_pct=0.0015)
        labels = labeler.label(config)
    """

    def __init__(self, ohlcv: pd.DataFrame):
        self._validate_ohlcv(ohlcv)
        self.ohlcv = ohlcv
        self.high = ohlcv["high"].values
        self.low = ohlcv["low"].values
        self.close = ohlcv["close"].values
        self.n = len(ohlcv)

    def _validate_ohlcv(self, ohlcv: pd.DataFrame) -> None:
        """Validate OHLCV DataFrame."""
        required = ["high", "low", "close"]
        missing = [c for c in required if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")

    def label_single(self, entry_idx: int, config: ExpansionConfig) -> ExpansionLabel:
        """
        Label expansion for a single entry point.

        Args:
            entry_idx: Index of entry bar
            config: Expansion configuration

        Returns:
            ExpansionLabel with results
        """
        entry_price = self.close[entry_idx]

        # LONG barriers
        long_target = entry_price * (1 + config.target_pct)
        long_invalid = entry_price * (1 - config.invalidation_pct)

        # SHORT barriers
        short_target = entry_price * (1 - config.target_pct)
        short_invalid = entry_price * (1 + config.invalidation_pct)

        # Track results
        long_exp, short_exp = 0, 0
        long_time, short_time = None, None
        long_done, short_done = False, False

        # Check each future bar
        for bars in range(1, config.horizon + 1):
            idx = entry_idx + bars
            if idx >= self.n:
                break

            h = self.high[idx]
            l = self.low[idx]

            # Check LONG expansion
            if not long_done:
                if h >= long_target:
                    # Hit target first - expansion!
                    long_exp = 1
                    long_time = bars
                    long_done = True
                elif l <= long_invalid:
                    # Hit invalidation first - no expansion
                    long_done = True

            # Check SHORT expansion
            if not short_done:
                if l <= short_target:
                    # Hit target first - expansion!
                    short_exp = 1
                    short_time = bars
                    short_done = True
                elif h >= short_invalid:
                    # Hit invalidation first - no expansion
                    short_done = True

            # Early exit if both resolved
            if long_done and short_done:
                break

        return ExpansionLabel(
            long_expansion=long_exp,
            short_expansion=short_exp,
            long_time=long_time,
            short_time=short_time,
        )

    def label(
        self,
        config: ExpansionConfig,
        show_progress: bool = True,
        progress_interval: int = 100_000,
    ) -> pd.DataFrame:
        """
        Label all bars for a single horizon.

        Args:
            config: Expansion configuration
            show_progress: Print progress updates
            progress_interval: Print every N bars

        Returns:
            DataFrame with columns:
            - long_expansion_{h}m
            - short_expansion_{h}m
            - long_time_{h}m
            - short_time_{h}m
        """
        h = config.horizon
        suffix = f"{h}m"

        # Pre-allocate arrays
        long_exp = np.zeros(self.n, dtype=np.int8)
        short_exp = np.zeros(self.n, dtype=np.int8)
        long_time = np.full(self.n, np.nan, dtype=np.float32)
        short_time = np.full(self.n, np.nan, dtype=np.float32)

        # Label each bar
        valid_range = self.n - h
        for i in range(valid_range):
            result = self.label_single(i, config)

            long_exp[i] = result.long_expansion
            short_exp[i] = result.short_expansion
            if result.long_time is not None:
                long_time[i] = result.long_time
            if result.short_time is not None:
                short_time[i] = result.short_time

            # Progress
            if show_progress and i > 0 and i % progress_interval == 0:
                pct = i / valid_range * 100
                print(f"  H={h}: {i:,}/{valid_range:,} ({pct:.1f}%)")

        return pd.DataFrame(
            {
                f"long_expansion_{suffix}": long_exp,
                f"short_expansion_{suffix}": short_exp,
                f"long_time_{suffix}": long_time,
                f"short_time_{suffix}": short_time,
            },
            index=self.ohlcv.index,
        )

    def label_multiple(
        self,
        configs: Dict[int, ExpansionConfig],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Label all bars for multiple horizons.

        Args:
            configs: Dict mapping horizon -> ExpansionConfig
            show_progress: Print progress updates

        Returns:
            DataFrame with expansion labels for all horizons
        """
        all_labels = []

        for h, config in sorted(configs.items()):
            if show_progress:
                print(f"\nLabeling H={h}...")
                print(f"  Target: {config.target_bps} bps")
                print(f"  Invalidation: {config.invalidation_bps} bps")

            labels = self.label(config, show_progress)
            all_labels.append(labels)

        return pd.concat(all_labels, axis=1)

    def get_summary(self, labels: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary statistics for expansion labels.

        Returns:
            Dict with expansion rates and average times
        """
        summary = {}

        for col in labels.columns:
            if "expansion" in col and "time" not in col:
                # Expansion rate
                rate = labels[col].mean()
                summary[col] = round(rate, 4)
            elif "time" in col:
                # Average time to expansion
                avg_time = labels[col].dropna().mean()
                if not np.isnan(avg_time):
                    summary[f"{col}_avg"] = round(avg_time, 2)

        return summary

    def print_summary(self, labels: pd.DataFrame) -> None:
        """Print formatted summary of expansion labels."""
        summary = self.get_summary(labels)

        print("\n" + "=" * 50)
        print("EXPANSION LABELING SUMMARY")
        print("=" * 50)

        # Group by horizon
        horizons = set()
        for col in labels.columns:
            if "expansion" in col:
                # Extract horizon from column name like "long_expansion_5m"
                h = col.split("_")[-1].replace("m", "")
                if h.isdigit():
                    horizons.add(int(h))

        for h in sorted(horizons):
            long_rate = summary.get(f"long_expansion_{h}m", 0)
            short_rate = summary.get(f"short_expansion_{h}m", 0)
            long_time = summary.get(f"long_time_{h}m_avg")
            short_time = summary.get(f"short_time_{h}m_avg")

            print(f"\nH={h}:")
            print(f"  LONG expansion:  {long_rate:.1%}" + (f" (avg {long_time} bars)" if long_time else ""))
            print(f"  SHORT expansion: {short_rate:.1%}" + (f" (avg {short_time} bars)" if short_time else ""))


def save_expansion_labels(
    labels: pd.DataFrame,
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save expansion labels to parquet file.

    Args:
        labels: DataFrame with expansion labels
        output_path: Path to save parquet file
        metadata: Optional metadata to save alongside
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, engine="pyarrow")

    # Save metadata
    if metadata:
        meta_path = output_path.replace(".parquet", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def load_expansion_labels(path: str) -> pd.DataFrame:
    """
    Load expansion labels from parquet file.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame with expansion labels
    """
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df
