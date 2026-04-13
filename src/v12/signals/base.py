"""Base class for ML signal generators.

All signal generators follow this interface:
1. Load ONNX model + scaler
2. Compute features from OHLCV DataFrame
3. Predict probability for a bar
4. Generate Signal objects (ML_LONG / ML_SHORT)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import onnxruntime as ort

from ..strategy import Signal, Direction, SignalType


class BaseSignalGenerator(ABC):
    """Base class for all ML signal generators."""

    def __init__(self, model_path: Path, scaler_path: Path,
                 conf_long: float = 0.60, conf_short: float = 0.35,
                 signal_type_long: SignalType = SignalType.ML_LONG,
                 signal_type_short: SignalType = SignalType.ML_SHORT):
        self.conf_long = conf_long
        self.conf_short = conf_short
        self.signal_type_long = signal_type_long
        self.signal_type_short = signal_type_short
        self.session = None
        self.loaded = False

        if model_path and model_path.exists():
            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self.loaded = True

        if scaler_path and scaler_path.exists():
            params = np.load(scaler_path)
            self.scaler_mean = params["mean"]
            self.scaler_std = params["std"]
        else:
            self.scaler_mean = None
            self.scaler_std = None

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute model-specific features from OHLCV DataFrame.
        Returns the DataFrame with feature columns added."""
        pass

    @abstractmethod
    def get_features_for_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[np.ndarray]:
        """Extract and normalize features for a single bar.
        Returns feature array ready for model input, or None."""
        pass

    def predict(self, features: np.ndarray) -> float:
        """Run ONNX inference. Returns probability of LONG (0-1)."""
        if not self.loaded:
            return 0.5
        logit = self.session.run(["logit"], {"features": features})[0].item()
        return 1 / (1 + np.exp(-logit))

    def predict_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[Signal]:
        """Generate signal for a single bar."""
        features = self.get_features_for_bar(df, bar_idx)
        if features is None:
            return None

        prob = self.predict(features)
        price = df.iloc[bar_idx]["close"]
        timestamp = df.index[bar_idx]

        if prob > self.conf_long:
            return Signal(
                direction=Direction.LONG,
                signal_type=self.signal_type_long,
                bar_index=bar_idx,
                timestamp=timestamp,
                rsi=prob,
                price=price,
            )
        elif prob < (1 - self.conf_short):
            return Signal(
                direction=Direction.SHORT,
                signal_type=self.signal_type_short,
                bar_index=bar_idx,
                timestamp=timestamp,
                rsi=prob,
                price=price,
            )
        return None
