import pandas as pd

class RollingNormalizer:
    def __init__(self, window: int):
        self.window = window

    def zscore(self, series: pd.Series) -> pd.Series:
        mean = series.rolling(self.window).mean()
        std = series.rolling(self.window).std()
        return (series - mean) / (std + 1e-9)

    def percentile(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )
