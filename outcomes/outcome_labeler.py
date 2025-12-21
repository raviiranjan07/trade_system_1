import os
import pandas as pd
import numpy as np


HORIZONS = [10, 30, 120]


def compute_mfe_mae(
    prices: pd.Series,
    horizon: int
):
    """
    Compute MFE and MAE for a single horizon.
    Returns two Series: mfe, mae (as percentages)
    """
    future_max = prices.shift(-1).rolling(horizon).max()
    future_min = prices.shift(-1).rolling(horizon).min()

    entry = prices

    mfe = (future_max - entry) / entry
    mae = (future_min - entry) / entry

    return mfe, mae


def label_outcomes(
    state_df: pd.DataFrame,
    price_series: pd.Series,
    pair: str,
    timeframe: str = "1m",
    output_dir: str = "data/outcomes"
) -> pd.DataFrame:
    """
    Generate MFE / MAE outcome labels for each state vector.
    """

    os.makedirs(output_dir, exist_ok=True)

    outcome_df = state_df.copy()

    for h in HORIZONS:
        # LONG outcomes
        mfe_long, mae_long = compute_mfe_mae(price_series, h)

        # SHORT outcomes (invert logic)
        mfe_short = -mae_long
        mae_short = -mfe_long

        outcome_df[f"mfe_long_{h}m"] = mfe_long
        outcome_df[f"mae_long_{h}m"] = mae_long
        outcome_df[f"mfe_short_{h}m"] = mfe_short
        outcome_df[f"mae_short_{h}m"] = mae_short

    outcome_df.dropna(inplace=True)

    # Drop non-serializable columns (e.g., MarketState objects)
    # These are not needed for outcome analysis
    columns_to_drop = [col for col in outcome_df.columns if col == 'state']
    if columns_to_drop:
        outcome_df = outcome_df.drop(columns=columns_to_drop)

    file_path = os.path.join(
        output_dir,
        f"{pair}_{timeframe}_outcomes.parquet"
    )

    outcome_df.to_parquet(file_path, engine="pyarrow")

    print(f"✅ Outcomes saved to: {file_path}")

    return outcome_df
