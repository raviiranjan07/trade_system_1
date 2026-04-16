"""V1.5 MLP Direction Signal Generator.

Features: roc1-8 + range_position_50 + rsi7 (10 flat features)
Label: direction H96
Model: models/ML_V1/direction_model.onnx
Thresholds: configs/params.yaml ml_v1.inference.{conf_long, conf_short}
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from ..strategy import SignalType
from .base import BaseSignalGenerator


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_inference_thresholds() -> tuple[float, float]:
    """Read ml_v1 inference thresholds from configs/params.yaml (single source of truth).

    Semantics (matches BaseSignalGenerator):
      LONG  if P(LONG)  > conf_long
      SHORT if P(SHORT) > conf_short  (i.e. P(LONG) < 1 - conf_short)
    """
    with open(REPO_ROOT / "configs/params.yaml") as f:
        p = yaml.safe_load(f)
    infer = p["ml_v1"]["inference"]
    return float(infer["conf_long"]), float(infer["conf_short"])


class MLV1(BaseSignalGenerator):
    """V1.5 production MLP — 10 flat features, H96 label."""

    RSI_PERIOD = 7
    RANGE_POSITION_PERIOD = 50
    ROC_PERIODS = list(range(1, 9))

    def __init__(
        self,
        model_path: Path = Path("models/ML_V1/direction_model.onnx"),
        scaler_path: Path = Path("models/ML_V1/scaler.npz"),
    ):
        conf_long, conf_short = _load_inference_thresholds()
        super().__init__(
            model_path=model_path,
            scaler_path=scaler_path,
            conf_long=conf_long,
            conf_short=conf_short,
            signal_type_long=SignalType.ML_LONG,
            signal_type_short=SignalType.ML_SHORT,
        )

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)

        for n in self.ROC_PERIODS:
            roc = np.zeros(len(close), dtype=np.float32)
            roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
            df[f"roc{n}"] = roc

        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(self.RSI_PERIOD).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(self.RSI_PERIOD).mean()
        rs = gain / loss_s
        df["rsi7"] = (100 - (100 / (1 + rs))).values

        rp = np.zeros(len(close), dtype=np.float32)
        for i in range(self.RANGE_POSITION_PERIOD, len(close)):
            hh = np.max(high[i - self.RANGE_POSITION_PERIOD:i + 1])
            ll = np.min(low[i - self.RANGE_POSITION_PERIOD:i + 1])
            rng = hh - ll
            rp[i] = (close[i] - ll) / rng if rng > 0 else 0.5
        df["range_position_50"] = rp

        return df

    def get_features_for_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[np.ndarray]:
        feat_cols = [f"roc{n}" for n in self.ROC_PERIODS] + ["range_position_50", "rsi7"]
        try:
            features = df.iloc[bar_idx][feat_cols].values.astype(np.float32)
        except (KeyError, IndexError):
            return None

        if np.any(np.isnan(features)):
            return None

        if self.scaler_mean is not None:
            features = (features - self.scaler_mean) / self.scaler_std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features.reshape(1, -1)

    # Backward-compatible aliases (used by backtest.py, exp12_cooldown.py, bot.py)
    def compute_features_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute_features(df)

    def generate_signals(self, df: pd.DataFrame) -> list:
        """Generate signals for all bars (batch). Used by backtest."""
        from ..strategy import Signal, Direction, SignalType
        signals = []
        feat_cols = [f"roc{n}" for n in self.ROC_PERIODS] + ["range_position_50", "rsi7"]
        if not all(c in df.columns for c in feat_cols):
            return []
        feat_arr = df[feat_cols].values.astype(np.float32)
        if self.scaler_mean is not None:
            feat_arr = (feat_arr - self.scaler_mean) / self.scaler_std
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.loaded:
            return []
        logits = self.session.run(["logit"], {"features": feat_arr})[0]
        probs = 1 / (1 + np.exp(-logits))
        closes = df["close"].values
        times = df.index
        for i in range(len(df)):
            prob = probs[i]
            if prob > self.conf_long:
                signals.append(Signal(
                    direction=Direction.LONG,
                    signal_type=self.signal_type_long,
                    bar_index=i, timestamp=times[i],
                    rsi=prob, price=closes[i],
                ))
            elif prob < (1 - self.conf_short):
                signals.append(Signal(
                    direction=Direction.SHORT,
                    signal_type=self.signal_type_short,
                    bar_index=i, timestamp=times[i],
                    rsi=prob, price=closes[i],
                ))
        return signals
