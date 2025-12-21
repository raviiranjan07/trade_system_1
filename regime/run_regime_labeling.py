import pandas as pd

from regime.regime_labeler import label_regime_row, smooth_regime


PAIR = "BTCUSDT"
TIMEFRAME = "1m"


if __name__ == "__main__":

    # 1️⃣ Load state vectors
    state_path = f"data/state_vectors/{PAIR}_{TIMEFRAME}_state.parquet"
    df = pd.read_parquet(state_path)

    # 2️⃣ Label regimes (row-wise, online-safe)
    df["regime_raw"] = df.apply(label_regime_row, axis=1)

    # 3️⃣ Smooth regimes
    df["regime"] = smooth_regime(df["regime_raw"], window=30)

    # 4️⃣ Save
    output_path = f"data/regimes/{PAIR}_{TIMEFRAME}_regimes.parquet"
    df[["regime"]].to_parquet(output_path, engine="pyarrow")

    print("✅ Regime labeling completed")
    print(df["regime"].value_counts())
