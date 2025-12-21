import pandas as pd

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume_raw"] = df["volume"]
    return df
