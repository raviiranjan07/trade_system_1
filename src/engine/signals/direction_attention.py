"""Attention Direction Signal Generator.

Features: 4 diffs (roc, rsi, rp, sma200) × 8 lookbacks = [8, 4] sequence
Label: direction H8
Model: models/ML_V2_ATTENTION_staging/attention_model.onnx
Thresholds: read from configs/params.yaml ml_v2_attention.inference.{conf_long,conf_short}
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from ..strategy import SignalType
from . import feature_lib
from .base import BaseSignalGenerator, load_feature_spec


REPO_ROOT = Path(__file__).resolve().parents[3]

_feat = load_feature_spec("ml_v2_attention")


def _load_inference_thresholds() -> tuple[float, float]:
    """Read attention inference thresholds from configs/params.yaml (single source of truth).

    Same semantics as MLV1:
      LONG if P(LONG) > conf_long
      SHORT if P(SHORT) > conf_short  (i.e. P(LONG) < 1 - conf_short)
    """
    with open(REPO_ROOT / "configs/params.yaml") as f:
        p = yaml.safe_load(f)
    infer = p["ml_v2_attention"]["inference"]
    return float(infer["conf_long"]), float(infer["conf_short"])


class DirectionAttention(BaseSignalGenerator):
    """LSTM+Attention — 32 diff features as [8,4] sequence, H8 label."""

    LOOKBACKS = list(_feat.get("lookbacks", range(1, 9)))
    RSI_PERIOD = int(_feat.get("rsi_period", 7))
    RANGE_POSITION_WINDOW = int(_feat.get("range_position_window", 20))

    def __init__(
        self,
        model_path: Path = Path("models/direction_attention/attention_model.onnx"),
        scaler_path: Path = Path("models/direction_attention/scaler.npz"),
    ):
        conf_long, conf_short = _load_inference_thresholds()
        super().__init__(
            model_path=model_path,
            scaler_path=scaler_path,
            conf_long=conf_long,
            conf_short=conf_short,
            signal_type_long=SignalType.ML_ATTN_LONG,
            signal_type_short=SignalType.ML_ATTN_SHORT,
        )
        # Pre-computed diff features stored here after compute_features()
        self._diff_arr = None
        self._df_index = None

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(np.float64)

        # Need rsi7, range_position, sma200_dist_pct from the dataframe
        # If not present, compute them (canonical formulas: feature_lib)
        if "rsi7" not in df.columns:
            df["rsi7"] = feature_lib.rsi_rolling(pd.Series(close), self.RSI_PERIOD).values

        if "range_position" not in df.columns:
            high = df["high"].values.astype(np.float64)
            low = df["low"].values.astype(np.float64)
            df["range_position"] = feature_lib.range_position_inclusive(
                high, low, close, self.RANGE_POSITION_WINDOW)

        if "sma200_dist_pct" not in df.columns:
            df["sma200_dist_pct"] = feature_lib.sma_dist_pct(close, 200)

        rsi7 = df["rsi7"].values.astype(np.float64)
        rp = df["range_position"].values.astype(np.float64)
        sma200 = df["sma200_dist_pct"].values.astype(np.float64)

        diff_arr = feature_lib.diff_features(close, rsi7, rp, sma200, self.LOOKBACKS)
        diff_arr = np.nan_to_num(diff_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize
        if self.scaler_mean is not None:
            diff_arr = (diff_arr - self.scaler_mean) / self.scaler_std
            diff_arr = np.nan_to_num(diff_arr, nan=0.0, posinf=0.0, neginf=0.0)

        self._diff_arr = diff_arr
        self._df_index = df.index

        return df

    def get_features_for_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[np.ndarray]:
        if self._diff_arr is None:
            return None

        if bar_idx < 0 or bar_idx >= len(self._diff_arr):
            return None

        # Reshape 32 flat features into [1, 8, 4] sequence
        features = self._diff_arr[bar_idx].reshape(1, len(self.LOOKBACKS), 4)

        if np.any(np.isnan(features)):
            return None

        return features

    # Backward-compatible aliases (used by backtest.py)
    def compute_features_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute_features(df)

    def generate_signals(self, df: pd.DataFrame) -> list:
        """Generate signals for all bars. Uses per-bar predict_bar (onnx inference per bar).
        Slower than ML_V1's batched path but produces same output shape.
        """
        signals = []
        for i in range(len(df)):
            s = self.predict_bar(df, i)
            if s is not None:
                signals.append(s)
        return signals
