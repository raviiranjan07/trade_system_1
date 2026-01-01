"""
Feature Engineering Module.

Computes technical indicators and market features from OHLCV data.
"""

from .trend import ema, ema_slope, compute_trend_features
from .momentum import returns, rsi, compute_momentum_features
from .volatility import atr, compute_volatility_features
from .volume import compute_volume_features
from .location import vwap, compute_location_features

__all__ = [
    # Trend
    "ema",
    "ema_slope",
    "compute_trend_features",
    # Momentum
    "returns",
    "rsi",
    "compute_momentum_features",
    # Volatility
    "atr",
    "compute_volatility_features",
    # Volume
    "compute_volume_features",
    # Location
    "vwap",
    "compute_location_features",
]
