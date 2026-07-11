"""Signal data types shared across the system.

The V1.4 rule strategy that used to live here was retired 2026-07-11
(git history has it). What remains are the pure types every signal
producer and consumer speaks:
  Direction   — LONG / SHORT
  SignalType  — identity labels for ML model signals
  Signal      — one entry proposal for one bar

Shared indicator computation moved to engine/indicators.py.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalType(Enum):
    # Generic defaults (signals/base.py). A new model that needs its own
    # identity labels adds members here (e.g. MY_MODEL_LONG/_SHORT).
    ML_LONG = "ML_LONG"
    ML_SHORT = "ML_SHORT"


@dataclass
class Signal:
    direction: Direction
    signal_type: SignalType
    bar_index: int
    timestamp: pd.Timestamp
    rsi: float
    price: float
    atr_percentile: Optional[float] = None
    ema_separation: Optional[float] = None
