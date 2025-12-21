import pandas as pd
import numpy as np

def returns(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_5m"] = returns(df["close"], 5)
    df["return_15m"] = returns(df["close"], 15)
    df["rsi_14"] = rsi(df["close"], 14)

    return df
