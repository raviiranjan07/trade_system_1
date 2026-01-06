"""
Real-Time State Builder.

Converts live candle data to state vectors matching backtest format.
Uses the same feature computation as the pipeline but optimized for streaming.

Flow:
1. Bootstrap with ~2500 historical candles (for EMA/normalization warmup)
2. On each new candle: update buffer, recompute features, get state vector
3. State vector can then be used to query similarity engine
"""

import logging
from typing import Optional, Dict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from trade_system.features.trend import compute_trend_features
from trade_system.features.momentum import compute_momentum_features
from trade_system.features.volatility import compute_volatility_features
from trade_system.features.volume import compute_volume_features
from trade_system.features.location import compute_location_features
from trade_system.normalization import RollingNormalizer
from trade_system.regime.regime_labeler import label_regime_row, smooth_regime


logger = logging.getLogger(__name__)


# State vector columns (must match similarity engine)
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


class RealtimeStateBuilder:
    """
    Builds state vectors from real-time candle data.

    Maintains a rolling buffer of candles and computes features
    on each update. Designed to match the exact output of the
    batch pipeline's state vector generation.

    Usage:
        builder = RealtimeStateBuilder(normalization_window=2000)
        builder.initialize(historical_df)  # Bootstrap with history

        # On each new candle:
        state = builder.update(candle_dict)
        regime = builder.get_current_regime()
    """

    def __init__(
        self,
        normalization_window: int = 2000,
        regime_smoothing_window: int = 30,
        min_warmup_bars: int = 250,
    ):
        """
        Initialize the state builder.

        Args:
            normalization_window: Window for z-score/percentile normalization
            regime_smoothing_window: Window for regime smoothing
            min_warmup_bars: Minimum bars needed before generating states
        """
        self.normalization_window = normalization_window
        self.regime_smoothing_window = regime_smoothing_window
        self.min_warmup_bars = min_warmup_bars

        # Internal state
        self._ohlcv_buffer: Optional[pd.DataFrame] = None
        self._features_df: Optional[pd.DataFrame] = None
        self._current_state: Optional[pd.Series] = None
        self._current_regime: Optional[str] = None
        self._is_initialized = False
        self._normalizer = RollingNormalizer(normalization_window)

    def initialize(self, historical_df: pd.DataFrame) -> None:
        """
        Initialize with historical candle data.

        Args:
            historical_df: DataFrame with OHLCV columns indexed by timestamp
                          Should have at least `min_warmup_bars` rows
        """
        if len(historical_df) < self.min_warmup_bars:
            raise ValueError(
                f"Need at least {self.min_warmup_bars} bars for initialization, "
                f"got {len(historical_df)}"
            )

        logger.info(f"Initializing state builder with {len(historical_df)} bars")

        # Store OHLCV buffer
        self._ohlcv_buffer = historical_df.copy()

        # Ensure proper column names
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in self._ohlcv_buffer.columns:
                raise ValueError(f"Missing required column: {col}")

        # Compute all features
        self._compute_all_features()

        self._is_initialized = True
        logger.info("State builder initialized")

    def _compute_all_features(self) -> None:
        """Compute all features on the current buffer."""
        df = self._ohlcv_buffer.copy()

        # Compute raw features
        df = compute_trend_features(df)
        df = compute_momentum_features(df)
        df = compute_volatility_features(df)
        df = compute_volume_features(df)
        df = compute_location_features(df)

        # Normalize features
        df = self._normalize_features(df)

        # Compute regimes
        df = self._compute_regimes(df)

        self._features_df = df

        # Get latest state and regime
        if len(df.dropna()) > 0:
            latest = df.dropna().iloc[-1]
            self._current_state = latest[STATE_COLUMNS]
            self._current_regime = latest["regime"]

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using rolling z-score and percentile."""
        df = df.copy()

        # Z-score normalization
        df["ema50_slope_z"] = self._normalizer.zscore(df["ema50_slope"])
        df["ema200_slope_z"] = self._normalizer.zscore(df["ema200_slope"])
        df["return_5m_z"] = self._normalizer.zscore(df["return_5m"])
        df["return_15m_z"] = self._normalizer.zscore(df["return_15m"])
        df["rsi_z"] = self._normalizer.zscore(df["rsi_14"])
        df["volume_z"] = self._normalizer.zscore(df["volume_raw"])
        df["vwap_distance_z"] = self._normalizer.zscore(df["vwap_distance"])

        # Percentile normalization
        df["atr_percentile"] = self._normalizer.percentile(df["atr_14"])

        return df

    def _compute_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market regimes."""
        df = df.copy()

        # Label raw regimes
        df["regime_raw"] = df.apply(label_regime_row, axis=1)

        # Smooth regimes
        df["regime"] = smooth_regime(df["regime_raw"], self.regime_smoothing_window)

        return df

    def update(self, candle: Dict) -> Optional[pd.Series]:
        """
        Update with a new closed candle and return the new state.

        Args:
            candle: Dict with keys: open_time, open, high, low, close, volume

        Returns:
            State vector as pandas Series, or None if warmup not complete
        """
        if not self._is_initialized:
            raise RuntimeError("State builder not initialized. Call initialize() first.")

        # Create new row
        new_row = pd.DataFrame([{
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
        }], index=[candle["open_time"]])

        # Append to buffer (keep last N rows for memory efficiency)
        max_buffer = self.normalization_window + 500
        self._ohlcv_buffer = pd.concat([self._ohlcv_buffer, new_row])
        if len(self._ohlcv_buffer) > max_buffer:
            self._ohlcv_buffer = self._ohlcv_buffer.iloc[-max_buffer:]

        # Recompute features
        self._compute_all_features()

        return self._current_state

    def get_current_state(self) -> Optional[pd.Series]:
        """Get the current state vector."""
        return self._current_state

    def get_current_regime(self) -> Optional[str]:
        """Get the current market regime."""
        return self._current_regime

    def get_state_as_dict(self) -> Optional[Dict]:
        """Get current state as dictionary."""
        if self._current_state is None:
            return None
        return self._current_state.to_dict()

    def get_features_df(self) -> Optional[pd.DataFrame]:
        """Get the full features DataFrame."""
        return self._features_df

    @property
    def is_ready(self) -> bool:
        """Check if the builder has enough data to produce states."""
        return (
            self._is_initialized and
            self._current_state is not None and
            not self._current_state.isna().any()
        )

    @property
    def buffer_length(self) -> int:
        """Number of candles in buffer."""
        if self._ohlcv_buffer is None:
            return 0
        return len(self._ohlcv_buffer)


def test_state_builder():
    """Test the state builder with sample data."""
    logging.basicConfig(level=logging.INFO)

    # Load sample data
    data_path = Path(__file__).parent.parent / "data" / "ohlcv" / "BTCUSDT_1m_ohlcv.parquet"
    if not data_path.exists():
        print(f"Test data not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} candles")

    # Initialize with first 2500 bars
    builder = RealtimeStateBuilder()
    builder.initialize(df.iloc[:2500])

    print(f"\nInitial state:")
    print(builder.get_current_state())
    print(f"Regime: {builder.get_current_regime()}")

    # Simulate updating with new candles
    print("\nSimulating updates...")
    for i in range(5):
        candle_idx = 2500 + i
        candle = {
            "open_time": df.index[candle_idx],
            "open": df.iloc[candle_idx]["open"],
            "high": df.iloc[candle_idx]["high"],
            "low": df.iloc[candle_idx]["low"],
            "close": df.iloc[candle_idx]["close"],
            "volume": df.iloc[candle_idx]["volume"],
        }
        state = builder.update(candle)
        print(f"  Update {i+1}: regime={builder.get_current_regime()}, ready={builder.is_ready}")


if __name__ == "__main__":
    test_state_builder()
