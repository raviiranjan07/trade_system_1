"""ML V3 Signal Generator — Exit-Aware Direction Prediction.

Same features as V2 Attention (32 diffs, [8,4] sequence).
Different output: 3-class (LONG/SHORT/SKIP) + PnL predictions.

Thresholds: read from configs/params.yaml ml_v3.inference.{conf_long, conf_short}
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from ..strategy import Direction, Signal, SignalType
from .base import BaseSignalGenerator

REPO_ROOT = Path(__file__).resolve().parents[3]

LONG, SHORT, SKIP = 0, 1, 2


def _load_inference_thresholds() -> tuple[float, float]:
    with open(REPO_ROOT / "configs/params.yaml") as f:
        p = yaml.safe_load(f)
    infer = p["ml_v3"]["inference"]
    return float(infer["conf_long"]), float(infer["conf_short"])


class MLV3(BaseSignalGenerator):
    """V3 — 3-class direction with PnL heads. Same features as V2 Attention."""

    LOOKBACKS = [1, 2, 3, 4, 5, 6, 7, 8]

    def __init__(
        self,
        model_path: Path = Path("models/ML_V3/v3_model.onnx"),
        scaler_path: Path = Path("models/ML_V3/scaler.npz"),
    ):
        conf_long, conf_short = _load_inference_thresholds()
        super().__init__(
            model_path=model_path,
            scaler_path=scaler_path,
            conf_long=conf_long,
            conf_short=conf_short,
            signal_type_long=SignalType.ML_V3_LONG,
            signal_type_short=SignalType.ML_V3_SHORT,
        )
        self._diff_arr = None
        self._snap_arr = None
        self._df_index = None
        self._is_v3 = True
        # Load snapshot scaler
        self._snap_mean = None
        self._snap_std = None
        if scaler_path.exists():
            s = np.load(scaler_path)
            if "snap_mean" in s:
                self._snap_mean = s["snap_mean"]
                self._snap_std = s["snap_std"]

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same diff features as V2 Attention."""
        close = df["close"].values.astype(np.float64)

        if "rsi7" not in df.columns:
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(7).mean()
            loss_s = (-delta.where(delta < 0, 0)).rolling(7).mean()
            rs = gain / loss_s
            df["rsi7"] = (100 - (100 / (1 + rs))).values

        if "range_position" not in df.columns:
            high = df["high"].values.astype(np.float64)
            low = df["low"].values.astype(np.float64)
            rp = np.zeros(len(close), dtype=np.float32)
            for i in range(20, len(close)):
                hh = np.max(high[i - 20:i + 1])
                ll = np.min(low[i - 20:i + 1])
                rng = hh - ll
                rp[i] = (close[i] - ll) / rng if rng > 0 else 0.5
            df["range_position"] = rp

        if "sma200_dist_pct" not in df.columns:
            sma200 = pd.Series(close).rolling(200).mean()
            df["sma200_dist_pct"] = ((close - sma200) / sma200 * 100).values

        rsi7 = df["rsi7"].values.astype(np.float64)
        rp = df["range_position"].values.astype(np.float64)
        sma200 = df["sma200_dist_pct"].values.astype(np.float64)

        diff_list = []
        for n in self.LOOKBACKS:
            roc_d = np.zeros(len(close), dtype=np.float32)
            roc_d[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
            rsi_d = np.zeros(len(close), dtype=np.float32)
            rsi_d[n:] = (rsi7[n:] - rsi7[:-n]).astype(np.float32)
            rp_d = np.zeros(len(close), dtype=np.float32)
            rp_d[n:] = (rp[n:] - rp[:-n]).astype(np.float32)
            sma_d = np.zeros(len(close), dtype=np.float32)
            sma_d[n:] = (sma200[n:] - sma200[:-n]).astype(np.float32)
            diff_list.extend([roc_d, rsi_d, rp_d, sma_d])

        diff_arr = np.column_stack(diff_list).astype(np.float32)
        diff_arr = np.nan_to_num(diff_arr, nan=0.0, posinf=0.0, neginf=0.0)

        if self.scaler_mean is not None:
            diff_arr = (diff_arr - self.scaler_mean) / self.scaler_std
            diff_arr = np.nan_to_num(diff_arr, nan=0.0, posinf=0.0, neginf=0.0)

        self._diff_arr = diff_arr

        # Snapshot features (position)
        atr_pctl = df["atr_percentile"].values.astype(np.float32) if "atr_percentile" in df.columns else np.full(len(close), 50.0, dtype=np.float32)
        snap_arr = np.column_stack([
            rsi7.astype(np.float32),
            rp.astype(np.float32),
            sma200.astype(np.float32),
            atr_pctl,
        ])
        snap_arr = np.nan_to_num(snap_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self._snap_mean is not None:
            snap_arr = (snap_arr - self._snap_mean) / self._snap_std
            snap_arr = np.nan_to_num(snap_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._snap_arr = snap_arr

        self._df_index = df.index
        return df

    def compute_features_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute_features(df)

    def get_features_for_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[np.ndarray]:
        if self._diff_arr is None or bar_idx < 0 or bar_idx >= len(self._diff_arr):
            return None
        features = self._diff_arr[bar_idx].reshape(1, 8, 4)
        if np.any(np.isnan(features)):
            return None
        return features

    def predict_bar(self, df: pd.DataFrame, bar_idx: int) -> Optional[Signal]:
        """Generate signal for a single bar using 3-class output."""
        features = self.get_features_for_bar(df, bar_idx)
        if features is None:
            return None
        if self._snap_arr is None or bar_idx >= len(self._snap_arr):
            return None

        if not self.loaded:
            return None

        snapshot = self._snap_arr[bar_idx].reshape(1, -1)

        # V3 ONNX has 2 inputs (features, snapshot), 3 outputs
        outputs = self.session.run(None, {"features": features, "snapshot": snapshot})

        if len(outputs) == 3:
            long_pnl, short_pnl, dir_logits = outputs
            dir_logits = dir_logits[0]  # [3]
        elif len(outputs) == 1:
            dir_logits = outputs[0][0]
        else:
            return None

        # Softmax
        exp_l = np.exp(dir_logits - dir_logits.max())
        probs = exp_l / exp_l.sum()
        p_long, p_short, p_skip = probs[LONG], probs[SHORT], probs[SKIP]

        price = df.iloc[bar_idx]["close"]
        timestamp = df.index[bar_idx]

        if p_long >= self.conf_long and p_long > p_short:
            return Signal(
                direction=Direction.LONG,
                signal_type=self.signal_type_long,
                bar_index=bar_idx,
                timestamp=timestamp,
                rsi=float(p_long),
                price=price,
            )
        elif p_short >= self.conf_short and p_short > p_long:
            return Signal(
                direction=Direction.SHORT,
                signal_type=self.signal_type_short,
                bar_index=bar_idx,
                timestamp=timestamp,
                rsi=float(p_short),
                price=price,
            )

        return None

    def generate_signals(self, df: pd.DataFrame) -> list:
        """Generate signals for all bars."""
        signals = []
        for i in range(len(df)):
            s = self.predict_bar(df, i)
            if s is not None:
                signals.append(s)
        return signals
