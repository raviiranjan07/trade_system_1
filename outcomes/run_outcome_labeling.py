import pandas as pd

from data.raw.ohlcv_loader import OHLCVLoader
from outcomes.outcome_labeler import label_outcomes


PAIR = "BTCUSDT"
TIMEFRAME = "1m"


if __name__ == "__main__":

    # 1️⃣ Load state vectors
    state_path = f"data/state_vectors/{PAIR}_{TIMEFRAME}_state.parquet"
    state_df = pd.read_parquet(state_path)

    # 2️⃣ Load close prices
    loader = OHLCVLoader()
    ohlcv = loader.fetch_ohlcv(pair=PAIR)

    close_prices = ohlcv["close"].loc[state_df.index]

    # 3️⃣ Label outcomes
    outcome_df = label_outcomes(
        state_df=state_df,
        price_series=close_prices,
        pair=PAIR,
        timeframe=TIMEFRAME
    )

    print(outcome_df.head())
    print(outcome_df.describe())
