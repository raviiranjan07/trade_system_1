"""Canonical feature formulas — THE single source of truth.

Every formula that turns candles into model inputs lives here, once.
Consumers: engine/indicators.py, model signal adapters (live inference),
training/* (dataset build + trainers), research verifiers. Training code
MUST import from here — never copy. A formula copied instead of imported
can silently diverge between training and live inference, which degrades
model predictions without any error being raised.

OPEN DECISION for the new architecture (day-one, before building the
feature cache): range_position has TWO variants that DO NOT agree
(28% of bars differ on real data; up to 3.8 sigma on diff features):
  - range_position_rolling(window):   pandas rolling of `window` bars,
    NaN warmup filled 0.5 — was the TRAINING-cache variant.
  - range_position_inclusive(window): loop over [i-window, i] =
    `window + 1` bars, zeros warmup — was the LIVE-inference variant.
Similarly atr_percentile had a 500-bar rank window in the training
cache vs 200-bar live (settings atr_rolling_window). The retired models
were trained/served across this mismatch (frozen 2026-07-11, consumers
now deleted). The new model must PICK ONE variant + ONE atr window,
delete the loser, and use the survivor in cache-build AND live
inference. Details: memory bot_refactor.md / audit 2026-07-10.
"""

import numpy as np
import pandas as pd


def rsi_rolling(close: pd.Series, period: int) -> pd.Series:
    """Standard RSI on a rolling simple mean of gains/losses.

    The one RSI in the system: the shared indicators (indicators.py), the
    rsi7 model feature (all three ML models, training and live), and the
    feature cache all use exactly this.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def roc_bps(close: np.ndarray, n: int) -> np.ndarray:
    """Rate of change over n bars, in basis points. float32, zeros warmup."""
    roc = np.zeros(len(close), dtype=np.float32)
    roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
    return roc


def range_position_rolling(high: pd.Series, low: pd.Series, close: pd.Series,
                           window: int) -> pd.Series:
    """Where close sits in the [rolling_low, rolling_high] band — TRAINING
    CACHE variant: `window` bars, NaN warmup/zero-range filled with 0.5.
    See module docstring: does NOT agree with range_position_inclusive."""
    rh = high.rolling(window).max()
    rl = low.rolling(window).min()
    rng = rh - rl
    return ((close - rl) / rng.where(rng > 0, np.nan)).fillna(0.5)


def range_position_inclusive(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                             window: int) -> np.ndarray:
    """Where close sits in the band over [i-window, i] — LIVE variant:
    `window + 1` bars inclusive, zeros warmup, 0.5 on zero range. float32.
    See module docstring: does NOT agree with range_position_rolling."""
    rp = np.zeros(len(close), dtype=np.float32)
    for i in range(window, len(close)):
        hh = np.max(high[i - window:i + 1])
        ll = np.min(low[i - window:i + 1])
        rng = hh - ll
        rp[i] = (close[i] - ll) / rng if rng > 0 else 0.5
    return rp


def sma_dist_pct(close: np.ndarray, window: int = 200) -> np.ndarray:
    """Distance of close from its rolling SMA, in percent. NaN warmup."""
    sma = pd.Series(close).rolling(window).mean()
    return ((close - sma) / sma * 100).to_numpy()


def diff_features(close: np.ndarray, rsi7: np.ndarray, rp: np.ndarray,
                  sma200: np.ndarray, lookbacks) -> np.ndarray:
    """The [N, 4*len(lookbacks)] velocity features for ML V2/V3.

    For each lookback n: (roc_bps_n, rsi7 diff, range_position diff,
    sma200_dist diff) — float32, zeros warmup, NO nan handling (callers
    apply their own nan_to_num / scaler steps).
    Column order per lookback: roc, rsi, rp, sma.
    """
    diff_list = []
    for n in lookbacks:
        roc_d = roc_bps(close, n)
        rsi_d = np.zeros(len(close), dtype=np.float32)
        rsi_d[n:] = (rsi7[n:] - rsi7[:-n]).astype(np.float32)
        rp_d = np.zeros(len(close), dtype=np.float32)
        rp_d[n:] = (rp[n:] - rp[:-n]).astype(np.float32)
        sma_d = np.zeros(len(close), dtype=np.float32)
        sma_d[n:] = (sma200[n:] - sma200[:-n]).astype(np.float32)
        diff_list.extend([roc_d, rsi_d, rp_d, sma_d])
    return np.column_stack(diff_list).astype(np.float32)


def snapshot_features(rsi7: np.ndarray, rp: np.ndarray, sma200: np.ndarray,
                      atr_pctl: np.ndarray) -> np.ndarray:
    """V3's [N, 4] position snapshot: rsi7, range_position, sma200_dist,
    atr_percentile — float32, NO nan handling (callers own that)."""
    return np.column_stack([
        rsi7.astype(np.float32),
        rp.astype(np.float32),
        sma200.astype(np.float32),
        atr_pctl.astype(np.float32),
    ])
