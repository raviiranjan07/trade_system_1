import pandas as pd
import numpy as np


# -----------------------------
# Regime thresholds (configurable later)
# -----------------------------
TREND_SLOPE_THRESHOLD = 0.7
HIGH_VOL_THRESHOLD = 0.85
LOW_VOL_THRESHOLD = 0.35

REGIME_ENCODING = {
    "RANGE_LOW_VOL": 0,
    "TREND_LOW_VOL": 1,
    "TREND_HIGH_VOL": 2,
    "HIGH_VOL": 3,
}

REGIME_DECODING = {v: k for k, v in REGIME_ENCODING.items()}


def label_regime_row(row) -> str:
    """
    Label regime using ONLY present & past information.
    """

    trend_strength = abs(row["ema200_slope_z"])
    vol = row["atr_percentile"]
    alignment = row["trend_alignment"]

    # Volatility shock (directionless)
    if vol >= HIGH_VOL_THRESHOLD:
        return "HIGH_VOL"

    # Trending regimes
    if trend_strength >= TREND_SLOPE_THRESHOLD and alignment != 0:
        if vol <= LOW_VOL_THRESHOLD:
            return "TREND_LOW_VOL"
        else:
            return "TREND_HIGH_VOL"

    # Otherwise range / chop
    return "RANGE_LOW_VOL"

def smooth_regime(
    regime_series: pd.Series,
    window: int = 30
) -> pd.Series:
    """
    Smooth regime labels using rolling majority vote.
    Works safely on categorical data.
    """

    # Encode to integers
    encoded = regime_series.map(REGIME_ENCODING)

    # Rolling majority vote
    smoothed = (
        encoded
        .rolling(window, min_periods=1)
        .apply(lambda x: x.value_counts().idxmax())
        .astype(int)
    )

    # Decode back to string labels
    return smoothed.map(REGIME_DECODING)
