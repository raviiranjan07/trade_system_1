import os
import pandas as pd


STATE_COLUMNS = [
    "ema50_slope_z",
    "ema200_slope_z",
    "trend_alignment",
    "return_5m_z",
    "return_15m_z",
    "rsi_z",
    "atr_percentile",
    "volume_z",
    "vwap_distance_z",
    "range_position",
]


def save_state_vectors_parquet(
    df: pd.DataFrame,
    pair: str,
    timeframe: str = "1m",
    base_dir: str = "data/state_vectors"
):
    """
    Persist market state vectors to Parquet.
    """

    os.makedirs(base_dir, exist_ok=True)

    # Keep only required columns
    state_df = df[STATE_COLUMNS].copy()
    state_df["pair"] = pair
    state_df.index.name = "time"

    file_path = os.path.join(
        base_dir,
        f"{pair}_{timeframe}_state.parquet"
    )

    state_df.to_parquet(file_path, engine="pyarrow")

    print(f"✅ State vectors saved to: {file_path}")
