import pandas as pd
import numpy as np

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

def compute_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["vwap"] = vwap(df)
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]

    rolling_high = df["high"].rolling(50).max()
    rolling_low = df["low"].rolling(50).min()
    df["range_position"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)

    return df
