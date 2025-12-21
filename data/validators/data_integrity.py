import pandas as pd

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce OHLCV sanity
    """
    assert df.index.is_monotonic_increasing, "Timestamps not sorted"
    assert not df.isnull().any().any(), "NaNs found in OHLCV"

    # Ensure 1-minute frequency
    expected = pd.date_range(
        start=df.index[0],
        end=df.index[-1],
        freq="1min"
    )

    missing = expected.difference(df.index)
    if len(missing) > 0:
        raise ValueError(f"Missing {len(missing)} candles")

    return df
