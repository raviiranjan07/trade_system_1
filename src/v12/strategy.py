"""V1.2 Strategy — Pure signal generation logic.

Entry rules (from experiments):
  LONG:  RSI crosses below 20 + price > SMA200 + ATR >= 25th pctl + EMA sep >= 0.5%
  SHORT: RSI crosses above 80 + price < SMA200 (no filters — EXP-007)

This module is stateless: takes data, returns signals. No execution or side effects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .config.schema import AppConfig


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Signal:
    direction: Direction
    bar_index: int
    timestamp: pd.Timestamp
    rsi: float
    price: float
    atr_percentile: Optional[float] = None
    ema_separation: Optional[float] = None


class V12Strategy:
    """V1.2 signal generator. Pure logic — no side effects."""

    def __init__(self, config: AppConfig):
        self.cfg = config

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators needed for signal generation.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
                Index must be DatetimeIndex.

        Returns:
            DataFrame with indicator columns added.
        """
        out = df.copy()
        c = self.cfg

        # RSI
        delta = out["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=c.strategy.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=c.strategy.rsi_period).mean()
        rs = gain / loss
        out["rsi"] = 100 - (100 / (1 + rs))

        # SMA for regime
        out["sma"] = out["close"].rolling(c.strategy.sma_period).mean()

        # EMA separation (LONG filter)
        lf = c.long_filters
        out["ema_short"] = out["close"].ewm(span=lf.ema_short, adjust=False).mean()
        out["ema_long"] = out["close"].ewm(span=lf.ema_long, adjust=False).mean()
        out["ema_separation"] = (
            abs(out["ema_short"] - out["ema_long"]) / out["close"] * 100
        )

        # ATR + percentile (LONG filter)
        tr = np.maximum(
            out["high"] - out["low"],
            np.maximum(
                abs(out["high"] - out["close"].shift(1)),
                abs(out["low"] - out["close"].shift(1)),
            ),
        )
        out["atr"] = tr.rolling(lf.atr_period).mean()
        atr_bps = out["atr"] / out["close"] * 10000
        out["atr_percentile"] = atr_bps.rolling(lf.atr_rolling_window).rank(pct=True) * 100

        # Signal detection (cross = first bar entering zone)
        rsi_oversold = out["rsi"] < c.strategy.rsi_oversold
        rsi_overbought = out["rsi"] > c.strategy.rsi_overbought
        out["rsi_oversold_cross"] = rsi_oversold & ~rsi_oversold.shift(1, fill_value=False)
        out["rsi_overbought_cross"] = rsi_overbought & ~rsi_overbought.shift(1, fill_value=False)

        # Regime
        out["bull_market"] = out["close"] > out["sma"]
        out["bear_market"] = out["close"] < out["sma"]

        return out

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """Generate all entry signals from indicator DataFrame.

        Args:
            df: DataFrame with indicators (output of compute_indicators).

        Returns:
            List of Signal objects, sorted by time.
        """
        c = self.cfg
        signals = []

        rsi_vals = df["rsi"].values
        atr_pctl = df["atr_percentile"].values
        ema_sep = df["ema_separation"].values
        bull = df["bull_market"].values
        bear = df["bear_market"].values
        oversold_cross = df["rsi_oversold_cross"].values
        overbought_cross = df["rsi_overbought_cross"].values
        prices = df["close"].values
        times = df.index

        for i in range(len(df)):
            # LONG: RSI oversold cross + bull market + filters
            if oversold_cross[i] and bull[i]:
                ap = atr_pctl[i]
                es = ema_sep[i]

                if np.isnan(ap) or np.isnan(es):
                    continue
                if ap < c.long_filters.atr_percentile_min:
                    continue
                if es < c.long_filters.ema_separation_min:
                    continue

                signals.append(Signal(
                    direction=Direction.LONG,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=rsi_vals[i],
                    price=prices[i],
                    atr_percentile=ap,
                    ema_separation=es,
                ))

            # SHORT: RSI overbought cross + bear market (no filters)
            elif overbought_cross[i] and bear[i]:
                signals.append(Signal(
                    direction=Direction.SHORT,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=rsi_vals[i],
                    price=prices[i],
                ))

        return signals
