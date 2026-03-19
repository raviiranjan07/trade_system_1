"""ML Signal Generator — MLP binary direction model.

Computes roc1-8 + range_position_50 + rsi7 and runs through trained MLP
to generate ML_LONG and ML_SHORT signals.

Thresholds: LONG at prob > 0.60, SHORT at prob < 0.35 (conf > 0.65).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .strategy import Signal, Direction, SignalType


# =============================================================================
# MODEL DEFINITION (must match training script)
# =============================================================================
class MLPBinaryDir(nn.Module):
    def __init__(self, input_size=10, hidden=128, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.out(h).squeeze(-1)


# =============================================================================
# ML SIGNAL GENERATOR
# =============================================================================
class MLSignalGenerator:
    """Generates ML_LONG and ML_SHORT signals from trained MLP model."""

    # Thresholds from backtest optimization
    CONF_LONG = 0.60   # prob > this → ML_LONG
    CONF_SHORT = 0.65  # confidence > this (prob < 0.35) → ML_SHORT

    # Feature computation params
    RSI_PERIOD = 7
    RANGE_POSITION_PERIOD = 50
    ROC_PERIODS = list(range(1, 9))  # roc1 to roc8

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """Load trained model and scaler parameters.

        Args:
            model_path: Path to trained .pt model file
            scaler_path: Path to .npz scaler file (mean/std from training)
        """
        self.model = MLPBinaryDir(input_size=10, hidden=128, dropout=0.0)
        self.model.eval()
        self.loaded = False

        if model_path and model_path.exists():
            ckpt = torch.load(model_path, map_location="cpu")
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
            else:
                self.model.load_state_dict(ckpt)
            self.loaded = True

        if scaler_path and scaler_path.exists():
            params = np.load(scaler_path)
            self.scaler_mean = params["mean"]
            self.scaler_std = params["std"]
        else:
            # Default scaler (will be overwritten when proper scaler is loaded)
            self.scaler_mean = np.zeros(10, dtype=np.float32)
            self.scaler_std = np.ones(10, dtype=np.float32)

        # Buffer for computing features (need history)
        self._close_history = []
        self._high_history = []
        self._low_history = []

    def compute_features_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ML features from OHLCV DataFrame.

        Adds columns: roc1-roc8, rsi7, range_position_50
        """
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)

        # ROC1-ROC8
        for n in self.ROC_PERIODS:
            roc = np.zeros(len(close), dtype=np.float32)
            roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
            df[f"roc{n}"] = roc

        # RSI7
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(self.RSI_PERIOD).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(self.RSI_PERIOD).mean()
        rs = gain / loss_s
        df["rsi7"] = (100 - (100 / (1 + rs))).values

        # Range position 50
        rp = np.zeros(len(close), dtype=np.float32)
        for i in range(self.RANGE_POSITION_PERIOD, len(close)):
            hh = np.max(high[i - self.RANGE_POSITION_PERIOD:i + 1])
            ll = np.min(low[i - self.RANGE_POSITION_PERIOD:i + 1])
            rng = hh - ll
            rp[i] = (close[i] - ll) / rng if rng > 0 else 0.5
        df["range_position_50"] = rp

        return df

    def get_features_for_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[np.ndarray]:
        """Extract and normalize 10 features for a single bar.

        Args:
            df: DataFrame with ML features computed (output of compute_features_from_df)
            bar_idx: Index position in the DataFrame

        Returns:
            Normalized feature array (10,) or None if features unavailable
        """
        feat_cols = [f"roc{n}" for n in self.ROC_PERIODS] + ["range_position_50", "rsi7"]

        try:
            features = df.iloc[bar_idx][feat_cols].values.astype(np.float32)
        except (KeyError, IndexError):
            return None

        if np.any(np.isnan(features)):
            return None

        # Normalize
        features = (features - self.scaler_mean) / self.scaler_std
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def predict(self, features: np.ndarray) -> float:
        """Run model prediction.

        Args:
            features: Normalized feature array (10,)

        Returns:
            Probability of LONG (0-1)
        """
        if not self.loaded:
            return 0.5  # neutral if model not loaded

        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            logit = self.model(x).item()
            prob = 1 / (1 + np.exp(-logit))

        return prob

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """Generate ML_LONG and ML_SHORT signals for all bars.

        Args:
            df: DataFrame with ML features computed

        Returns:
            List of Signal objects
        """
        if not self.loaded:
            return []

        signals = []
        feat_cols = [f"roc{n}" for n in self.ROC_PERIODS] + ["range_position_50", "rsi7"]

        # Check all required columns exist
        if not all(c in df.columns for c in feat_cols):
            return []

        feat_arr = df[feat_cols].values.astype(np.float32)
        feat_arr = (feat_arr - self.scaler_mean) / self.scaler_std
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

        closes = df["close"].values
        times = df.index

        # Batch predict all bars
        with torch.no_grad():
            x = torch.from_numpy(feat_arr)
            logits = self.model(x).numpy()
            probs = 1 / (1 + np.exp(-logits))

        for i in range(len(df)):
            prob = probs[i]

            if prob > self.CONF_LONG:
                signals.append(Signal(
                    direction=Direction.LONG,
                    signal_type=SignalType.ML_LONG,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=prob,  # store probability in rsi field for logging
                    price=closes[i],
                ))
            elif prob < (1 - self.CONF_SHORT):
                signals.append(Signal(
                    direction=Direction.SHORT,
                    signal_type=SignalType.ML_SHORT,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=prob,
                    price=closes[i],
                ))

        return signals

    def predict_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[Signal]:
        """Generate signal for a single bar (for live bot use).

        Args:
            df: DataFrame with ML features computed
            bar_idx: Current bar index

        Returns:
            Signal or None
        """
        features = self.get_features_for_bar(df, bar_idx)
        if features is None:
            return None

        prob = self.predict(features)
        price = df.iloc[bar_idx]["close"]
        timestamp = df.index[bar_idx]

        if prob > self.CONF_LONG:
            return Signal(
                direction=Direction.LONG,
                signal_type=SignalType.ML_LONG,
                bar_index=bar_idx,
                timestamp=timestamp,
                rsi=prob,
                price=price,
            )
        elif prob < (1 - self.CONF_SHORT):
            return Signal(
                direction=Direction.SHORT,
                signal_type=SignalType.ML_SHORT,
                bar_index=bar_idx,
                timestamp=timestamp,
                rsi=prob,
                price=price,
            )

        return None
