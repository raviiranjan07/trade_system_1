import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def ema_slope(ema_series: pd.Series, window: int) -> pd.Series:
    return ema_series.diff(window)

def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    df["ema50_slope"] = ema_slope(df["ema50"], 5)
    df["ema200_slope"] = ema_slope(df["ema200"], 20)

    df["trend_alignment"] = np.sign(df["ema50"] - df["ema200"])

    return df
