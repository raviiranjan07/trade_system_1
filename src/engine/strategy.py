"""V1.3.2 Strategy — Pure signal generation logic.

Entry rules (from experiments):
  V12_LONG:    RSI crosses below 20 + bull (price>SMA200) + ATR>=25 + EMA>=0.5%  [EXP-006]
  V12_SHORT:   RSI crosses above 80 + bear (price<SMA200) [no filters]            [EXP-007]
  BEAR_LONG:   RSI < 10 (level) + bear (price<SMA200) + EMA>=1.0%                 [EXP-014]
  BULL_SHORT:  RSI > 90 (level) + bull (price>SMA200) + ATR>=60 + EMA>=1.0%       [EXP-014]

V1.3.2 change: BEAR_LONG and BULL_SHORT switched from cross-based to level-based.
  Cross = fires only on the bar RSI first crosses the threshold (misses re-entries).
  Level = fires on ANY bar where RSI is in the extreme zone (catches re-entries after exit).
  V12_LONG and V12_SHORT remain cross-based (unchanged).

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


class SignalType(Enum):
    V12_LONG = "V12_LONG"
    V12_SHORT = "V12_SHORT"
    BEAR_LONG = "BEAR_LONG"
    BULL_SHORT = "BULL_SHORT"
    ML_LONG = "ML_LONG"
    ML_SHORT = "ML_SHORT"
    ML_ATTN_LONG = "ML_ATTN_LONG"
    ML_ATTN_SHORT = "ML_ATTN_SHORT"


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


class V12Strategy:
    """V1.3.2 signal generator. Pure logic — no side effects."""

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

        # EMA separation
        lf = c.long_filters
        out["ema_short"] = out["close"].ewm(span=lf.ema_short, adjust=False).mean()
        out["ema_long"] = out["close"].ewm(span=lf.ema_long, adjust=False).mean()
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
        out["atr"] = tr.rolling(lf.atr_period).mean()
        atr_bps = out["atr"] / out["close"] * 10000
        out["atr_percentile"] = atr_bps.rolling(lf.atr_rolling_window).rank(pct=True) * 100

        # V1.2 RSI crosses (20/80)
        rsi_oversold = out["rsi"] < c.strategy.rsi_oversold
        rsi_overbought = out["rsi"] > c.strategy.rsi_overbought
        out["rsi_oversold_cross"] = rsi_oversold & ~rsi_oversold.shift(1, fill_value=False)
        out["rsi_overbought_cross"] = rsi_overbought & ~rsi_overbought.shift(1, fill_value=False)

        # V1.3.2 extreme RSI levels (no cross required — level-based)
        out["rsi_extreme_oversold"] = out["rsi"] < c.bear_long_filters.rsi_threshold
        out["rsi_extreme_overbought"] = out["rsi"] > c.bull_short_filters.rsi_threshold

        # Regime
        out["bull_market"] = out["close"] > out["sma"]
        out["bear_market"] = out["close"] < out["sma"]

        return out

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """Generate all entry signals from indicator DataFrame.

        Signal priority (elif chain): V12 signals fire first if multiple conditions met.

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
        extreme_os_level = df["rsi_extreme_oversold"].values
        extreme_ob_level = df["rsi_extreme_overbought"].values
        prices = df["close"].values
        times = df.index

        for i in range(len(df)):
            # Signal 1: V12_LONG — RSI<20 cross + bull + ATR>=25 + EMA>=0.5
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
                    signal_type=SignalType.V12_LONG,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=rsi_vals[i],
                    price=prices[i],
                    atr_percentile=ap,
                    ema_separation=es,
                ))

            # Signal 2: V12_SHORT — RSI>80 cross + bear (no filters)
            elif overbought_cross[i] and bear[i]:
                signals.append(Signal(
                    direction=Direction.SHORT,
                    signal_type=SignalType.V12_SHORT,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=rsi_vals[i],
                    price=prices[i],
                ))

            # Signal 3: BEAR_LONG — RSI<10 level + bear + EMA>=1.0  [V1.3.2: level-based]
            elif extreme_os_level[i] and bear[i]:
                es = ema_sep[i]
                if np.isnan(es):
                    continue
                if es < c.bear_long_filters.ema_separation_min:
                    continue

                signals.append(Signal(
                    direction=Direction.LONG,
                    signal_type=SignalType.BEAR_LONG,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=rsi_vals[i],
                    price=prices[i],
                    ema_separation=es,
                ))

            # Signal 4: BULL_SHORT — RSI>90 level + bull + ATR>=60 + EMA>=1.0  [V1.3.2: level-based]
            elif extreme_ob_level[i] and bull[i]:
                ap = atr_pctl[i]
                es = ema_sep[i]
                if np.isnan(ap) or np.isnan(es):
                    continue
                if ap < c.bull_short_filters.atr_percentile_min:
                    continue
                if es < c.bull_short_filters.ema_separation_min:
                    continue

                signals.append(Signal(
                    direction=Direction.SHORT,
                    signal_type=SignalType.BULL_SHORT,
                    bar_index=i,
                    timestamp=times[i],
                    rsi=rsi_vals[i],
                    price=prices[i],
                    atr_percentile=ap,
                    ema_separation=es,
                ))

        return signals
