import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_14"] = atr(df, 14)
    return df
