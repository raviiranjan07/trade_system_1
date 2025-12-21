import pandas as pd

df = pd.read_parquet("data/outcomes/BTCUSDT_1m_outcomes.parquet")

print(df[["mfe_long_30m", "mae_long_30m"]].describe())


df = pd.read_parquet("data/regimes/BTCUSDT_1m_regimes.parquet")
print(df["regime"].value_counts(normalize=True))