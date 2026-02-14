"""
Micro-trigger detection system for trade entries.

This module detects price action patterns that can serve as entry triggers:
- Engulfing patterns (bullish/bearish)
- Pin bars (rejection wicks)
- Inside bar breakouts
- Volume spikes with direction
- Break and retest patterns
"""

from .micro_triggers import (
    MicroTriggerDetector,
    TriggerConfig,
    TriggerSignal,
)

__all__ = [
    "MicroTriggerDetector",
    "TriggerConfig",
    "TriggerSignal",
]
