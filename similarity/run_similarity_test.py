import pandas as pd

from similarity.similarity_engine import SimilarityEngine


PAIR = "BTCUSDT"
TIMEFRAME = "1m"


if __name__ == "__main__":

    # Load data
    outcome_df = pd.read_parquet(f"data/outcomes/{PAIR}_{TIMEFRAME}_outcomes.parquet")
    regime_df = pd.read_parquet(f"data/regimes/{PAIR}_{TIMEFRAME}_regimes.parquet")

    # Init engine
    engine = SimilarityEngine(
        outcome_df=outcome_df,
        regime_df=regime_df,
        k=200
    )

    # Pick latest state
    current_state = outcome_df.iloc[-1]
    current_regime = regime_df.iloc[-1]["regime"]

    result = engine.query(
        current_state=current_state,
        regime=current_regime,
        horizon=30
    )

    print("Current Regime:", current_regime)
    print("Similarity Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
