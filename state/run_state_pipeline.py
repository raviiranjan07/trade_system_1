import pandas as pd

from data.raw.ohlcv_loader import OHLCVLoader
from data.validators.data_integrity import validate_ohlcv

from features.trend import compute_trend_features
from features.momentum import compute_momentum_features
from features.volatility import compute_volatility_features
from features.volume import compute_volume_features
from features.location import compute_location_features
from state.state_store import save_state_vectors_parquet


from state.normalizer import RollingNormalizer
from state.state_builder import build_state

DB_URL = "postgresql://localhost/crypto_data"
NORMALIZATION_WINDOW = 2000

def build_state_vectors_from_db(
    pair: str,
    start_time: str = None,
    end_time: str = None
) -> pd.DataFrame:

    # 1️⃣ Fetch from DB
    loader = OHLCVLoader()
    df = loader.fetch_ohlcv(
        pair=pair,
        start_time=start_time,
        end_time=end_time
    )

    # 2️⃣ Validate raw data
    df = validate_ohlcv(df)

    # 3️⃣ Feature computation
    df = compute_trend_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_location_features(df)

    # 4️⃣ Normalization
    norm = RollingNormalizer(NORMALIZATION_WINDOW)

    df["ema50_slope_z"] = norm.zscore(df["ema50_slope"])
    df["ema200_slope_z"] = norm.zscore(df["ema200_slope"])
    df["return_5m_z"] = norm.zscore(df["return_5m"])
    df["return_15m_z"] = norm.zscore(df["return_15m"])
    df["rsi_z"] = norm.zscore(df["rsi_14"])
    df["volume_z"] = norm.zscore(df["volume_raw"])
    df["vwap_distance_z"] = norm.zscore(df["vwap_distance"])
    df["atr_percentile"] = norm.percentile(df["atr_14"])

    # 5️⃣ Build states
    state_df = df.dropna().copy()
    state_df["state"] = state_df.apply(build_state, axis=1)

    return state_df


# state/run_state_pipeline.py

if __name__ == "__main__":
    df_states = build_state_vectors_from_db(
        pair="BTCUSDT",
        start_time="2023-01-01",
        end_time="2023-03-01"
    )

    print(df_states.head())
    print(df_states.describe())

    save_state_vectors_parquet(
        df=df_states,
        pair="BTCUSDT",
        timeframe="1m"
    )

