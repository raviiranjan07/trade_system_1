"""
Market State Vector Engine.

Builds normalized 10-dimensional state vectors from OHLCV data.
"""

from .state_schema import MarketState
from .state_builder import build_state
from .state_store import save_state_vectors_parquet

__all__ = [
    "MarketState",
    "build_state",
    "save_state_vectors_parquet",
]
