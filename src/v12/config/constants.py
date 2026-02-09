"""Hardcoded truths — never change these without re-running all experiments."""

# Fees (round-trip, limit orders on Binance Futures)
FEES_BPS = 8

# Minimum profitable move (Rule #1 from CLAUDE.md)
# 12 bps target - 8 bps fees = 4 bps net profit (minimum worthwhile)
MIN_PROFITABLE_MOVE_BPS = 12

# Trading pair and timeframe
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
TIMEFRAME_MINUTES = 15
