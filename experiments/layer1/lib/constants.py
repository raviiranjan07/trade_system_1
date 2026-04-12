"""Single source of truth for all Layer 1 experiments."""

# Binance BTCUSDT perpetual constraints
LEVERAGE = 125
MAINT_MARGIN_RATE = 0.004       # 0.4% maintenance margin (Tier 1)
MIN_QTY = 0.001                 # Minimum BTC order quantity
STEP_SIZE = 0.001               # BTC quantity step size
MIN_NOTIONAL = 100              # Minimum order value in USD

# Trading costs
FEES_BPS = 8                    # Round-trip fees (limit orders)

# Data splits
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
OOS_START = "2024-01-01"
OOS_END = "2025-12-31"

# Monte Carlo defaults
N_SIMS = 1000
DEFAULT_CAPITAL = 10.0

# Data path
DATA_PATH = "data/raw/BTCUSDT_15m_ohlcv.parquet"
