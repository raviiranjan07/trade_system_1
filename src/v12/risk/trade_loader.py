"""Load and enrich V1.3.2 trades with market conditions."""
import math
from pathlib import Path

import pandas as pd

from .exchange_constants import (
    DATA_PATH, MIN_NOTIONAL, STEP_SIZE, MIN_QTY,
    TRAIN_START, TRAIN_END, OOS_START, OOS_END,
)


def _get_indicator_df():
    """Load OHLCV data and compute V1.3.2 indicators (cached)."""
    if not hasattr(_get_indicator_df, '_cache'):
        from ..config.loader import load_config
        from ..strategy import V12Strategy

        config = load_config()
        df = pd.read_parquet(Path(DATA_PATH))
        df.index = pd.to_datetime(df.index).tz_localize(None)

        strategy = V12Strategy(config)
        df = strategy.compute_indicators(df)
        _get_indicator_df._cache = df
    return _get_indicator_df._cache


def _get_config():
    """Load V1.3.2 config (cached)."""
    if not hasattr(_get_config, '_cache'):
        from ..config.loader import load_config
        _get_config._cache = load_config()
    return _get_config._cache


def load_raw_trades(period: str = "train"):
    """Run backtest and return raw TradeRecord list.

    Args:
        period: "train" (2020-2023) or "oos" (2024-2025)
    """
    from ..backtest import run_backtest

    config = _get_config()
    if period == "train":
        return run_backtest(config, start=TRAIN_START, end=TRAIN_END)
    elif period == "oos":
        return run_backtest(config, start=OOS_START, end=OOS_END)
    else:
        raise ValueError(f"period must be 'train' or 'oos', got '{period}'")


def enrich_trades(trades_raw) -> list:
    """Enrich raw TradeRecord list with market conditions.

    Returns list of dicts with:
        bps, btc_price, qty_min, pos_min, signal_type, direction,
        entry_hour, entry_dow, atr_pctl, ema_sep, rsi,
        mfe_bps, mae_bps, exit_reason
    """
    df = _get_indicator_df()
    enriched = []

    for t in trades_raw:
        btc_price = t.entry_price
        qty_min = max(MIN_QTY,
                      math.ceil(MIN_NOTIONAL / btc_price / STEP_SIZE) * STEP_SIZE)

        signal_time = pd.Timestamp(t.signal_time)
        entry_time = pd.Timestamp(t.entry_time)

        # Look up indicators at signal time
        idx = df.index.get_indexer([signal_time], method='nearest')[0]
        if idx >= 0:
            row = df.iloc[idx]
            atr_pctl = float(row['atr_percentile']) if not pd.isna(row.get('atr_percentile', float('nan'))) else 50.0
            ema_sep = float(row['ema_separation']) if not pd.isna(row.get('ema_separation', float('nan'))) else 0.5
            rsi = float(row['rsi']) if not pd.isna(row.get('rsi', float('nan'))) else 50.0
        else:
            atr_pctl, ema_sep, rsi = 50.0, 0.5, 50.0

        enriched.append({
            'bps': t.net_profit_bps,
            'btc_price': btc_price,
            'qty_min': qty_min,
            'pos_min': qty_min * btc_price,
            'signal_type': t.signal_type,
            'direction': t.direction,
            'entry_hour': entry_time.hour,
            'entry_dow': entry_time.dayofweek,
            'atr_pctl': atr_pctl,
            'ema_sep': ema_sep,
            'rsi': rsi,
            'mfe_bps': getattr(t, 'mfe_bps', 0),
            'mae_bps': getattr(t, 'mae_bps', 0),
            'exit_reason': getattr(t, 'exit_reason', ''),
        })

    return enriched


def load_enriched_trades(period: str = "train") -> list:
    """Load and enrich trades in one call.

    Args:
        period: "train" (2020-2023) or "oos" (2024-2025)

    Returns:
        List of enriched trade dicts.
    """
    raw = load_raw_trades(period)
    return enrich_trades(raw)
