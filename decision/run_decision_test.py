import pandas as pd

from similarity.similarity_engine import SimilarityEngine
from decision.decision_engine import DecisionEngine


PAIR = "BTCUSDT"
TIMEFRAME = "1m"


if __name__ == "__main__":

    # Load data
    outcome_df = pd.read_parquet(f"data/outcomes/{PAIR}_{TIMEFRAME}_outcomes.parquet")
    regime_df = pd.read_parquet(f"data/regimes/{PAIR}_{TIMEFRAME}_regimes.parquet")

    # Init engines
    similarity_engine = SimilarityEngine(
        outcome_df=outcome_df,
        regime_df=regime_df,
        k=200
    )

    decision_engine = DecisionEngine(
        capital=10_000,        # $10k account
        risk_per_trade=0.005   # 0.5%
    )

    # Current context
    current_state = outcome_df.iloc[-1]
    current_regime = regime_df.iloc[-1]["regime"]

    sim_result = similarity_engine.query(
        current_state=current_state,
        regime=current_regime,
        horizon=30
    )

    decision = decision_engine.decide(
        similarity_result=sim_result,
        regime=current_regime
    )

    print("Current Regime:", current_regime)
    print("Similarity Result:", sim_result)
    print("Decision:", decision)
