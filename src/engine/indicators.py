"""Shared market indicators — computed once per bar, consumed by all tracks.

This is the surviving half of the retired V1.4 rule strategy: the entry
rules are gone (git history: engine/strategy.py pre-2026-07-11), but the
indicator columns feed the whole system:
  rsi              dashboard status + chart pane
  sma              chart overlay + regime
  bull_market      regime display (BULL/BEAR)
  atr_percentile   dashboard + ML V3's snapshot feature (drift note in
                   signals/feature_lib.py applies)
  ema_separation   dashboard status

Formulas come from signals/feature_lib.py where shared with model
features (RSI); windows come from settings.yaml `indicators:`.
"""

import numpy as np
import pandas as pd

from .config.schema import AppConfig
from .signals.feature_lib import rsi_rolling


def compute_indicators(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Compute the shared indicator columns from OHLCV.

    Args:
        df: OHLCV DataFrame (open/high/low/close/volume, DatetimeIndex).
        cfg: AppConfig — windows from cfg.indicators.

    Returns:
        Copy of df with indicator columns added.
    """
    out = df.copy()
    ind = cfg.indicators

    # RSI (canonical formula: signals/feature_lib.py)
    out["rsi"] = rsi_rolling(out["close"], ind.rsi_period)

    # SMA for regime
    out["sma"] = out["close"].rolling(ind.sma_period).mean()

    # EMA separation
    out["ema_short"] = out["close"].ewm(span=ind.ema_short, adjust=False).mean()
    out["ema_long"] = out["close"].ewm(span=ind.ema_long, adjust=False).mean()
    out["ema_separation"] = (
        abs(out["ema_short"] - out["ema_long"]) / out["close"] * 100
    )

    # ATR + percentile
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            abs(out["high"] - out["close"].shift(1)),
            abs(out["low"] - out["close"].shift(1)),
        ),
    )
    out["atr"] = tr.rolling(ind.atr_period).mean()
    atr_bps = out["atr"] / out["close"] * 10000
    out["atr_percentile"] = atr_bps.rolling(ind.atr_rolling_window).rank(pct=True) * 100

    # Regime
    out["bull_market"] = out["close"] > out["sma"]

    return out
