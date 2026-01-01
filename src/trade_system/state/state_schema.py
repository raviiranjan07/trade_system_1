# state/state_schema.py

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class MarketState:
    ema50_slope_z: float
    ema200_slope_z: float
    trend_alignment: int      # -1, 0, +1
    return_5m_z: float
    return_15m_z: float
    rsi_z: float
    atr_percentile: float
    volume_z: float
    vwap_distance_z: float
    range_position: float

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "ema50_slope_z",
            "ema200_slope_z",
            "trend_alignment",
            "return_5m_z",
            "return_15m_z",
            "rsi_z",
            "atr_percentile",
            "volume_z",
            "vwap_distance_z",
            "range_position",
        ]
