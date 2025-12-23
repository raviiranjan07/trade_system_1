import pandas as pd

# Load outcomes
df = pd.read_parquet("data/outcomes/BTCUSDT_1m_outcomes.parquet")

print("=" * 60)
print("OUTCOMES FILE ANALYSIS")
print("=" * 60)

print(f"\nTotal rows: {len(df)}")
print(f"\nAll columns ({len(df.columns)}):")
print(df.columns.tolist())

# Show all outcome columns (MFE/MAE for all horizons)
outcome_cols = [c for c in df.columns if c.startswith(('mfe_', 'mae_'))]
print(f"\n\nOutcome columns ({len(outcome_cols)}):")
print(outcome_cols)

print("\n\nOutcome Statistics (all horizons):")
print(df[outcome_cols].describe())

# Load regimes
print("\n" + "=" * 60)
print("REGIMES FILE ANALYSIS")
print("=" * 60)

df_regimes = pd.read_parquet("data/regimes/BTCUSDT_1m_regimes.parquet")
print(f"\nTotal rows: {len(df_regimes)}")
print("\nRegime distribution:")
print(df_regimes["regime"].value_counts())
print("\nRegime distribution (%):")
print(df_regimes["regime"].value_counts(normalize=True).mul(100).round(2))

# Load state vectors
print("\n" + "=" * 60)
print("STATE VECTORS FILE ANALYSIS")
print("=" * 60)

df_states = pd.read_parquet("data/state_vectors/BTCUSDT_1m_state.parquet")
print(f"\nTotal rows: {len(df_states)}")
print(f"\nAll columns ({len(df_states.columns)}):")
print(df_states.columns.tolist())

print("\n\nState Vector Statistics:")
print(df_states.describe())