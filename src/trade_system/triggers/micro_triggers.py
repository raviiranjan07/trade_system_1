"""
Micro-trigger detection for trade entries.

These are short-term price action patterns that signal potential entry points.
They should be used WITH a filter (like expansion rate) for best results.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict
import numpy as np
import pandas as pd


@dataclass
class TriggerConfig:
    """Configuration for trigger detection."""

    # Engulfing pattern settings
    engulfing_min_body_ratio: float = 0.6  # Min body/range ratio for engulfing candle
    engulfing_prev_min_body: float = 0.3   # Min body ratio for previous candle

    # Pin bar settings
    pin_bar_wick_ratio: float = 2.0        # Wick must be 2x body
    pin_bar_body_position: float = 0.33    # Body in top/bottom 33% of range

    # Inside bar settings
    inside_bar_lookback: int = 1           # Bars to check for inside bar

    # Volume spike settings
    volume_spike_multiplier: float = 2.0   # Volume must be 2x average
    volume_lookback: int = 20              # Bars to compute average volume

    # Break and retest settings
    retest_lookback: int = 10              # Bars to find break level
    retest_tolerance: float = 0.001        # 0.1% tolerance for retest
    retest_confirmation_bars: int = 2      # Bars to confirm retest

    # Minimum candle size (filter tiny moves)
    min_candle_range_bps: float = 5.0      # Minimum 5 bps range


@dataclass
class TriggerSignal:
    """A detected trigger signal."""

    timestamp: pd.Timestamp
    trigger_type: str  # "ENGULFING", "PIN_BAR", "INSIDE_BAR", "VOLUME_SPIKE", "BREAK_RETEST"
    direction: Literal["LONG", "SHORT"]
    strength: float  # 0-1 signal strength
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "trigger_type": self.trigger_type,
            "direction": self.direction,
            "strength": self.strength,
            **self.details,
        }


class MicroTriggerDetector:
    """
    Detects micro-triggers from OHLCV data.

    Usage:
        detector = MicroTriggerDetector(ohlcv_df)
        signals = detector.detect_all()

        # Or detect specific patterns
        engulfing = detector.detect_engulfing()
        pin_bars = detector.detect_pin_bars()
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        config: Optional[TriggerConfig] = None,
    ):
        self.ohlcv = ohlcv.copy()
        self.config = config or TriggerConfig()

        # Pre-compute useful columns
        self._precompute()

    def _precompute(self) -> None:
        """Pre-compute common values for efficiency."""
        df = self.ohlcv

        # Body and range
        df['body'] = df['close'] - df['open']
        df['body_abs'] = df['body'].abs()
        df['range'] = df['high'] - df['low']
        df['range'] = df['range'].replace(0, np.nan)  # Avoid division by zero

        # Body ratio (body size relative to range)
        df['body_ratio'] = df['body_abs'] / df['range']

        # Wick sizes
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

        # Direction (1 = bullish, -1 = bearish)
        df['candle_dir'] = np.sign(df['body'])

        # Range in bps
        df['range_bps'] = (df['range'] / df['close'].shift(1)) * 10000

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(self.config.volume_lookback).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        self.ohlcv = df

    def detect_engulfing(self) -> pd.DataFrame:
        """
        Detect engulfing patterns.

        Bullish engulfing: Current candle completely engulfs previous bearish candle
        Bearish engulfing: Current candle completely engulfs previous bullish candle
        """
        df = self.ohlcv.copy()
        cfg = self.config

        # Previous candle values
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_body_ratio = df['body_ratio'].shift(1)
        prev_dir = df['candle_dir'].shift(1)

        # Current candle requirements
        has_body = df['body_ratio'] >= cfg.engulfing_min_body_ratio
        prev_has_body = prev_body_ratio >= cfg.engulfing_prev_min_body
        min_range = df['range_bps'] >= cfg.min_candle_range_bps

        # Bullish engulfing: current bullish, prev bearish, current engulfs prev
        bullish_engulf = (
            (df['candle_dir'] == 1) &  # Current is bullish
            (prev_dir == -1) &          # Previous was bearish
            (df['close'] > prev_open) & # Current close above prev open
            (df['open'] < prev_close) & # Current open below prev close
            has_body & prev_has_body & min_range
        )

        # Bearish engulfing: current bearish, prev bullish, current engulfs prev
        bearish_engulf = (
            (df['candle_dir'] == -1) &  # Current is bearish
            (prev_dir == 1) &            # Previous was bullish
            (df['close'] < prev_open) &  # Current close below prev open
            (df['open'] > prev_close) &  # Current open above prev close
            has_body & prev_has_body & min_range
        )

        # Compute strength (based on body ratio and volume)
        strength = (df['body_ratio'] * 0.5 + df['volume_ratio'].clip(0, 2) / 2 * 0.5).clip(0, 1)

        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        result['trigger_engulfing_long'] = bullish_engulf.astype(int)
        result['trigger_engulfing_short'] = bearish_engulf.astype(int)
        result['engulfing_strength'] = strength * (bullish_engulf | bearish_engulf)

        return result

    def detect_pin_bars(self) -> pd.DataFrame:
        """
        Detect pin bar (rejection wick) patterns.

        Bullish pin bar: Long lower wick, small body at top
        Bearish pin bar: Long upper wick, small body at bottom
        """
        df = self.ohlcv.copy()
        cfg = self.config

        # Wick ratios
        lower_wick_ratio = df['lower_wick'] / df['body_abs'].replace(0, np.nan)
        upper_wick_ratio = df['upper_wick'] / df['body_abs'].replace(0, np.nan)

        # Body position (0 = bottom, 1 = top)
        body_mid = (df['open'] + df['close']) / 2
        body_position = (body_mid - df['low']) / df['range']

        min_range = df['range_bps'] >= cfg.min_candle_range_bps

        # Bullish pin bar: long lower wick, body at top
        bullish_pin = (
            (lower_wick_ratio >= cfg.pin_bar_wick_ratio) &
            (body_position >= (1 - cfg.pin_bar_body_position)) &
            min_range
        )

        # Bearish pin bar: long upper wick, body at bottom
        bearish_pin = (
            (upper_wick_ratio >= cfg.pin_bar_wick_ratio) &
            (body_position <= cfg.pin_bar_body_position) &
            min_range
        )

        # Strength based on wick ratio
        wick_strength = np.maximum(lower_wick_ratio, upper_wick_ratio).clip(0, 5) / 5

        result = pd.DataFrame(index=df.index)
        result['trigger_pinbar_long'] = bullish_pin.astype(int)
        result['trigger_pinbar_short'] = bearish_pin.astype(int)
        result['pinbar_strength'] = wick_strength * (bullish_pin | bearish_pin)

        return result

    def detect_inside_bars(self) -> pd.DataFrame:
        """
        Detect inside bar breakout patterns.

        Inside bar: Current bar's range is within previous bar's range
        Breakout: Next bar breaks the inside bar's high (long) or low (short)
        """
        df = self.ohlcv.copy()
        cfg = self.config

        # Previous bar values
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)

        # Inside bar condition
        is_inside = (df['high'] <= prev_high) & (df['low'] >= prev_low)

        # Breakout detection (using next bar)
        next_high = df['high'].shift(-1)
        next_low = df['low'].shift(-1)
        next_close = df['close'].shift(-1)

        # Bullish breakout: next bar breaks above inside bar high and closes above
        bullish_breakout = is_inside & (next_high > df['high']) & (next_close > df['high'])

        # Bearish breakout: next bar breaks below inside bar low and closes below
        bearish_breakout = is_inside & (next_low < df['low']) & (next_close < df['low'])

        # Shift forward so signal is on breakout bar
        bullish_breakout = bullish_breakout.shift(1).fillna(False)
        bearish_breakout = bearish_breakout.shift(1).fillna(False)

        # Strength based on breakout momentum
        breakout_range = df['range_bps'] / 20  # Normalize
        strength = breakout_range.clip(0, 1)

        result = pd.DataFrame(index=df.index)
        result['trigger_insidebar_long'] = bullish_breakout.astype(int)
        result['trigger_insidebar_short'] = bearish_breakout.astype(int)
        result['insidebar_strength'] = strength * (bullish_breakout | bearish_breakout)

        return result

    def detect_volume_spikes(self) -> pd.DataFrame:
        """
        Detect volume spike patterns with directional bias.

        Volume spike + bullish close = long signal
        Volume spike + bearish close = short signal
        """
        df = self.ohlcv.copy()
        cfg = self.config

        # Volume spike condition
        is_spike = df['volume_ratio'] >= cfg.volume_spike_multiplier

        # Direction based on candle
        is_bullish = df['candle_dir'] == 1
        is_bearish = df['candle_dir'] == -1

        min_range = df['range_bps'] >= cfg.min_candle_range_bps

        # Signals
        bullish_spike = is_spike & is_bullish & min_range
        bearish_spike = is_spike & is_bearish & min_range

        # Strength based on volume ratio
        strength = (df['volume_ratio'] / cfg.volume_spike_multiplier).clip(0, 2) / 2

        result = pd.DataFrame(index=df.index)
        result['trigger_volumespike_long'] = bullish_spike.astype(int)
        result['trigger_volumespike_short'] = bearish_spike.astype(int)
        result['volumespike_strength'] = strength * (bullish_spike | bearish_spike)

        return result

    def detect_break_retest(self) -> pd.DataFrame:
        """
        Detect break and retest patterns.

        1. Price breaks a recent high/low
        2. Price retraces back to the level
        3. Price bounces off (confirms the break)

        This is more complex and uses rolling windows.
        """
        df = self.ohlcv.copy()
        cfg = self.config

        lookback = cfg.retest_lookback
        tolerance = cfg.retest_tolerance
        confirm = cfg.retest_confirmation_bars

        # Rolling highs and lows
        rolling_high = df['high'].rolling(lookback).max().shift(1)
        rolling_low = df['low'].rolling(lookback).min().shift(1)

        # Break detection
        broke_high = df['close'] > rolling_high
        broke_low = df['close'] < rolling_low

        # Retest detection (price comes back near the broken level)
        # For bullish: after breaking high, price dips back near it
        retest_high = (
            broke_high.shift(confirm) &  # Broke high N bars ago
            (df['low'] <= rolling_high.shift(confirm) * (1 + tolerance)) &  # Came back to level
            (df['close'] > rolling_high.shift(confirm))  # But closed above
        )

        # For bearish: after breaking low, price rises back near it
        retest_low = (
            broke_low.shift(confirm) &  # Broke low N bars ago
            (df['high'] >= rolling_low.shift(confirm) * (1 - tolerance)) &  # Came back to level
            (df['close'] < rolling_low.shift(confirm))  # But closed below
        )

        min_range = df['range_bps'] >= cfg.min_candle_range_bps

        bullish_retest = retest_high & min_range
        bearish_retest = retest_low & min_range

        # Strength based on how cleanly price bounced
        bounce_ratio = df['body_ratio']
        strength = bounce_ratio.clip(0, 1)

        result = pd.DataFrame(index=df.index)
        result['trigger_retest_long'] = bullish_retest.astype(int)
        result['trigger_retest_short'] = bearish_retest.astype(int)
        result['retest_strength'] = strength * (bullish_retest | bearish_retest)

        return result

    def detect_all(self, show_progress: bool = True) -> pd.DataFrame:
        """
        Detect all trigger types and combine into single DataFrame.

        Returns DataFrame with columns:
        - trigger_* columns for each pattern (1 = signal, 0 = no signal)
        - *_strength columns for signal strength
        - combined_long / combined_short for any long/short signal
        """
        if show_progress:
            print("Detecting triggers...")

        # Detect each pattern
        engulfing = self.detect_engulfing()
        pin_bars = self.detect_pin_bars()
        inside_bars = self.detect_inside_bars()
        volume_spikes = self.detect_volume_spikes()
        break_retest = self.detect_break_retest()

        # Combine all
        result = pd.concat([
            engulfing,
            pin_bars,
            inside_bars,
            volume_spikes,
            break_retest,
        ], axis=1)

        # Combined signals (any trigger)
        long_cols = [c for c in result.columns if c.startswith('trigger_') and c.endswith('_long')]
        short_cols = [c for c in result.columns if c.startswith('trigger_') and c.endswith('_short')]

        result['any_trigger_long'] = result[long_cols].any(axis=1).astype(int)
        result['any_trigger_short'] = result[short_cols].any(axis=1).astype(int)
        result['any_trigger'] = (result['any_trigger_long'] | result['any_trigger_short']).astype(int)

        # Count triggers
        result['trigger_count_long'] = result[long_cols].sum(axis=1)
        result['trigger_count_short'] = result[short_cols].sum(axis=1)

        if show_progress:
            total_long = result['any_trigger_long'].sum()
            total_short = result['any_trigger_short'].sum()
            print(f"  Total LONG triggers: {total_long:,}")
            print(f"  Total SHORT triggers: {total_short:,}")

        return result

    def get_signals_list(self, triggers_df: pd.DataFrame) -> List[TriggerSignal]:
        """Convert triggers DataFrame to list of TriggerSignal objects."""
        signals = []

        trigger_types = ['engulfing', 'pinbar', 'insidebar', 'volumespike', 'retest']

        for idx, row in triggers_df.iterrows():
            for ttype in trigger_types:
                long_col = f'trigger_{ttype}_long'
                short_col = f'trigger_{ttype}_short'
                strength_col = f'{ttype}_strength'

                if long_col in row and row[long_col] == 1:
                    signals.append(TriggerSignal(
                        timestamp=idx,
                        trigger_type=ttype.upper(),
                        direction="LONG",
                        strength=row.get(strength_col, 0.5),
                    ))

                if short_col in row and row[short_col] == 1:
                    signals.append(TriggerSignal(
                        timestamp=idx,
                        trigger_type=ttype.upper(),
                        direction="SHORT",
                        strength=row.get(strength_col, 0.5),
                    ))

        return signals
