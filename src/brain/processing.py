"""Brain Pipeline -- Step 2: Data Processing

Compute all features from raw OHLCV data.
Features computed per bar, then assembled into snapshots in dataset.py.

Categories:
  1. Price Action: roc, rsi7, range_position
  2. Market Structure: 13 S/R features (V5 confirmed zones)
  3. Volatility: atr_pct, atr_percentile, ema_separation
  4. Bar Characteristics: range_bps, body_bps, dist_from_high20_pct
  5. Activity: volume_ratio
"""

import numpy as np
import pandas as pd


# =========================================================================
# Price Action Features
# =========================================================================

def compute_roc(close: np.ndarray) -> np.ndarray:
    """Rate of change: (close - prev_close) / prev_close * 10000 bps."""
    roc = np.zeros(len(close), dtype=np.float32)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 10000
    return roc


def compute_rsi(close: np.ndarray, period: int = 7) -> np.ndarray:
    """RSI using exponential moving average of gains/losses."""
    rsi = np.full(len(close), 50.0, dtype=np.float32)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    if len(delta) < period:
        return rsi

    # Initial average
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_range_position(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                            period: int = 50) -> np.ndarray:
    """Where price is in the N-bar range. 0 = at low, 1 = at high."""
    rp = np.full(len(close), 0.5, dtype=np.float32)
    for i in range(period, len(close)):
        h = np.max(high[i - period:i + 1])
        l = np.min(low[i - period:i + 1])
        if h > l:
            rp[i] = (close[i] - l) / (h - l)
    return rp


# =========================================================================
# Market Structure Features (S/R V5)
# =========================================================================

def find_sr_zones(highs: np.ndarray, lows: np.ndarray, min_touches: int = 2,
                   method: str = "biggest_gap", bandwidth: float = 0.03,
                   support_weight: str = "reaction", resistance_weight: str = "recency",
                   peak_threshold: float = 0.2):
    """Find support and resistance zones.

    Methods:
      biggest_gap: original V5 — cluster by biggest gap in sorted prices
      hybrid_kde: KDE with configurable weighting per side

    Returns: (sup_low, sup_high, res_low, res_high) or Nones
    """
    if method == "hybrid_kde":
        return _find_sr_kde(highs, lows, bandwidth, support_weight, resistance_weight, peak_threshold)
    else:
        return _find_sr_biggest_gap(highs, lows, min_touches)


def _kde_weighted(prices, weight_type, highs, lows, bandwidth, peak_threshold, side):
    """Run weighted KDE on prices and return peak range.

    side: 'support' (pick lowest peak) or 'resistance' (pick highest peak)
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    n = len(prices)
    if n < 3:
        return None, None

    # Compute weights
    if weight_type == "plain":
        weights = np.ones(n)
    elif weight_type == "recency":
        weights = np.linspace(0.1, 1.0, n)
    elif weight_type == "reaction":
        weights = np.ones(n)
        for i in range(n - 1):
            if side == "support":
                mx = np.max(highs[i + 1:min(i + 10, n)] - lows[i]) if i + 1 < n else 1
            else:
                mx = np.max(highs[i] - lows[i + 1:min(i + 10, n)]) if i + 1 < n else 1
            weights[i] = max(mx, 1)
    elif weight_type == "combined":
        recency = np.linspace(0.1, 1.0, n)
        reaction = np.ones(n)
        for i in range(n - 1):
            if side == "support":
                mx = np.max(highs[i + 1:min(i + 10, n)] - lows[i]) if i + 1 < n else 1
            else:
                mx = np.max(highs[i] - lows[i + 1:min(i + 10, n)]) if i + 1 < n else 1
            reaction[i] = max(mx, 1)
        weights = recency * reaction
    else:
        weights = np.ones(n)

    # Weighted KDE
    wn = weights / weights.sum()
    reps = np.maximum((wn * 1000).astype(int), 1)
    repeated = np.repeat(prices, reps)

    if len(repeated) < 2:
        return None, None

    try:
        kde = gaussian_kde(repeated, bw_method=bandwidth)
    except Exception:
        return None, None

    pr = np.linspace(prices.min() - 20, prices.max() + 20, 200)
    d = kde(pr)
    pks, _ = find_peaks(d, height=d.max() * peak_threshold)

    if len(pks) == 0:
        return None, None

    # Pick peak based on side
    if side == "support":
        peak_idx = pks[0]  # lowest peak
    else:
        peak_idx = pks[-1]  # highest peak

    # Get range at half maximum
    hm = d[peak_idx] / 2
    l, r = peak_idx, peak_idx
    while l > 0 and d[l] > hm:
        l -= 1
    while r < len(d) - 1 and d[r] > hm:
        r += 1

    return float(pr[l]), float(pr[r])


def _find_sr_kde(highs, lows, bandwidth, support_weight, resistance_weight, peak_threshold):
    """Find S/R using hybrid KDE."""
    sup_low, sup_high = _kde_weighted(lows, support_weight, highs, lows, bandwidth, peak_threshold, "support")
    res_low, res_high = _kde_weighted(highs, resistance_weight, highs, lows, bandwidth, peak_threshold, "resistance")

    # Zone valid only if resistance above support
    if sup_low is not None and res_low is not None and res_low <= sup_high:
        return None, None, None, None

    return sup_low, sup_high, res_low, res_high


def _find_sr_biggest_gap(highs, lows, min_touches):
    """Original V5 biggest gap method."""
    # --- SUPPORT ---
    sorted_lows = np.sort(lows)
    gaps = np.diff(sorted_lows)
    sup_low, sup_high = None, None

    if len(gaps) > 0:
        gap_order = np.argsort(gaps)[::-1]
        for gap_idx in gap_order:
            bottom = sorted_lows[:gap_idx + 1]
            if len(bottom) >= min_touches:
                sup_low = float(bottom[0])
                sup_high = float(bottom[-1])
                break

    if sup_low is None and len(sorted_lows) >= min_touches:
        total_range = sorted_lows[-1] - sorted_lows[0]
        if total_range / sorted_lows[0] * 10000 < 20:
            sup_low = float(sorted_lows[0])
            sup_high = float(sorted_lows[1])

    # --- RESISTANCE ---
    sorted_highs = np.sort(highs)
    gaps_h = np.diff(sorted_highs)
    res_low, res_high = None, None

    if len(gaps_h) > 0:
        gap_order_h = np.argsort(gaps_h)[::-1]
        for gap_idx in gap_order_h:
            top = sorted_highs[gap_idx + 1:]
            if len(top) >= min_touches:
                res_low = float(top[0])
                res_high = float(top[-1])
                break

    if res_low is None and len(sorted_highs) >= min_touches:
        total_range = sorted_highs[-1] - sorted_highs[0]
        if total_range / sorted_highs[0] * 10000 < 20:
            res_low = float(sorted_highs[-2])
            res_high = float(sorted_highs[-1])

    if sup_low is not None and res_low is not None and res_low <= sup_high:
        return None, None, None, None

    return sup_low, sup_high, res_low, res_high


def count_bar_touches(highs: np.ndarray, lows: np.ndarray,
                       range_low: float, range_high: float) -> int:
    """Count how many bars touch a price range."""
    count = 0
    for i in range(len(highs)):
        if lows[i] <= range_high and highs[i] >= range_low:
            count += 1
    return count


def compute_recovery_speed(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                            range_low: float, range_high: float, direction: str) -> float:
    """Compute recovery speed through a zone.

    direction='up': speed of upward movement from support
    direction='down': speed of downward movement from resistance
    """
    n = len(highs)
    best_speed = 0.0

    if direction == "up":
        for i in range(n - 1):
            if lows[i] <= range_high and lows[i] >= range_low:
                for j in range(i + 1, n):
                    if closes[j] > range_high:
                        bars = j - i
                        move = (closes[j] - range_high) / range_high * 10000
                        speed = move / bars
                        if speed > best_speed:
                            best_speed = speed
                        break
    else:  # down
        for i in range(n - 1):
            if highs[i] >= range_low and highs[i] <= range_high:
                for j in range(i + 1, n):
                    if closes[j] < range_low:
                        bars = j - i
                        move = (range_low - closes[j]) / range_low * 10000
                        speed = move / bars
                        if speed > best_speed:
                            best_speed = speed
                        break

    return best_speed


def compute_sr_features(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         config: dict) -> pd.DataFrame:
    """Compute all 13 S/R features for every bar.

    Uses sliding window: at bar t, S/R computed from bars t-lookback to t.
    Carry-forward when no confirmed S/R found.
    """
    sr_cfg = config["features"]["market_structure"]
    lookback = sr_cfg.get("sr_lookback", 8)
    min_touches = sr_cfg.get("sr_min_touches", 2)
    carry_forward = sr_cfg.get("sr_carry_forward", True)

    N = len(high)
    result = {
        "support_range_low": np.full(N, np.nan, dtype=np.float64),
        "support_range_high": np.full(N, np.nan, dtype=np.float64),
        "resistance_range_low": np.full(N, np.nan, dtype=np.float64),
        "resistance_range_high": np.full(N, np.nan, dtype=np.float64),
        "zone_width": np.full(N, np.nan, dtype=np.float32),
        "support_retest": np.zeros(N, dtype=np.float32),
        "resistance_retest": np.zeros(N, dtype=np.float32),
        "distance_to_support": np.full(N, np.nan, dtype=np.float32),
        "distance_to_resistance": np.full(N, np.nan, dtype=np.float32),
        "recovery_up": np.zeros(N, dtype=np.float32),
        "recovery_down": np.zeros(N, dtype=np.float32),
        "window_low": np.full(N, np.nan, dtype=np.float64),
        "window_high": np.full(N, np.nan, dtype=np.float64),
    }

    prev_sup_low = None
    prev_sup_high = None
    prev_res_low = None
    prev_res_high = None

    for t in range(lookback, N):
        w_high = high[t - lookback + 1:t + 1]
        w_low = low[t - lookback + 1:t + 1]
        w_close = close[t - lookback + 1:t + 1]

        # Window extremes (always available)
        result["window_low"][t] = float(np.min(w_low))
        result["window_high"][t] = float(np.max(w_high))

        # Find confirmed S/R
        sl, sh, rl, rh = find_sr_zones(w_high, w_low, min_touches)

        # Carry forward if needed
        if sl is None and carry_forward and prev_sup_low is not None:
            sl, sh = prev_sup_low, prev_sup_high
        if rl is None and carry_forward and prev_res_low is not None:
            rl, rh = prev_res_low, prev_res_high

        # Update carry-forward state
        if sl is not None:
            prev_sup_low, prev_sup_high = sl, sh
        if rl is not None:
            prev_res_low, prev_res_high = rl, rh

        # Check zone validity
        has_zone = sl is not None and rl is not None and rl > sh

        if has_zone:
            result["support_range_low"][t] = sl
            result["support_range_high"][t] = sh
            result["resistance_range_low"][t] = rl
            result["resistance_range_high"][t] = rh
            result["zone_width"][t] = (rl - sh) / sh * 10000
            result["support_retest"][t] = count_bar_touches(w_high, w_low, sl, sh)
            result["resistance_retest"][t] = count_bar_touches(w_high, w_low, rl, rh)
            result["distance_to_support"][t] = (close[t] - sh) / sh * 10000
            result["distance_to_resistance"][t] = (rl - close[t]) / rl * 10000
            result["recovery_up"][t] = compute_recovery_speed(w_high, w_low, w_close, sl, sh, "up")
            result["recovery_down"][t] = compute_recovery_speed(w_high, w_low, w_close, rl, rh, "down")
        elif sl is not None:
            result["support_range_low"][t] = sl
            result["support_range_high"][t] = sh
            result["distance_to_support"][t] = (close[t] - sh) / sh * 10000
            result["support_retest"][t] = count_bar_touches(w_high, w_low, sl, sh)
        elif rl is not None:
            result["resistance_range_low"][t] = rl
            result["resistance_range_high"][t] = rh
            result["distance_to_resistance"][t] = (rl - close[t]) / rl * 10000
            result["resistance_retest"][t] = count_bar_touches(w_high, w_low, rl, rh)

        if t % 50000 == 0:
            print(f"    S/R: {t}/{N} bars processed...")

    return pd.DataFrame(result)


def convert_sr_to_zone_relative(sr_df: pd.DataFrame, close: np.ndarray, lookback: int = 25) -> pd.DataFrame:
    """Convert 13 raw S/R features to zone-relative features.

    Removes absolute price dependency. All features become relative to zone structure.
    No z-score normalization needed on top.
    """
    N = len(sr_df)
    result = pd.DataFrame(index=sr_df.index)

    sl = sr_df["support_range_low"].values
    sh = sr_df["support_range_high"].values
    rl = sr_df["resistance_range_low"].values
    rh = sr_df["resistance_range_high"].values
    wl = sr_df["window_low"].values
    wh = sr_df["window_high"].values
    zw = sr_df["zone_width"].values
    dist_sup = sr_df["distance_to_support"].values
    dist_res = sr_df["distance_to_resistance"].values
    rec_up = sr_df["recovery_up"].values
    rec_down = sr_df["recovery_down"].values

    # Mid price and zone width in price terms
    mid_price = (wh + wl) / 2.0
    zone_width_price = rl - sh  # price gap between support top and resistance bottom

    # Avoid division by zero
    safe_zw_price = np.where(zone_width_price > 0, zone_width_price, 1.0)
    safe_zw_bps = np.where(zw > 0, zw, 1.0)
    safe_window_range = np.where((wh - wl) > 0, wh - wl, 1.0)

    # Zone-relative prices (6)
    result["support_low_rel"] = (sl - mid_price) / safe_zw_price
    result["support_high_rel"] = (sh - mid_price) / safe_zw_price
    result["res_low_rel"] = (rl - mid_price) / safe_zw_price
    result["res_high_rel"] = (rh - mid_price) / safe_zw_price
    result["window_low_rel"] = (wl - mid_price) / safe_zw_price
    result["window_high_rel"] = (wh - mid_price) / safe_zw_price

    # Zone-relative derived (7)
    result["dist_to_support_pct"] = dist_sup / safe_zw_bps
    result["dist_to_res_pct"] = dist_res / safe_zw_bps
    result["price_position"] = (close - wl) / safe_window_range
    result["support_width_pct"] = (sh - sl) / safe_zw_price
    result["res_width_pct"] = (rh - rl) / safe_zw_price
    result["recovery_up_pct"] = rec_up / safe_zw_bps
    result["recovery_down_pct"] = rec_down / safe_zw_bps

    # Keep as-is (3)
    result["support_retest"] = sr_df["support_retest"].values
    result["resistance_retest"] = sr_df["resistance_retest"].values
    result["zone_width"] = zw

    # Multi-horizon approach speed: how fast price moved toward support
    # Positive = price was further from support before (approached it)
    # Negative = price was closer to support before (moved away)
    speed_short = np.zeros(N, dtype=np.float32)
    speed_mid = np.zeros(N, dtype=np.float32)
    speed_long = np.zeros(N, dtype=np.float32)

    for t in range(3, N):
        speed_short[t] = dist_sup[t - 3] - dist_sup[t]
    for t in range(10, N):
        speed_mid[t] = dist_sup[t - 10] - dist_sup[t]
    for t in range(lookback, N):
        speed_long[t] = dist_sup[t - lookback] - dist_sup[t]

    result["speed_short"] = speed_short / safe_zw_bps
    result["speed_mid"] = speed_mid / safe_zw_bps
    result["speed_long"] = speed_long / safe_zw_bps
    result["acceleration"] = (speed_short - speed_mid) / safe_zw_bps

    # Replace NaN/inf
    result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return result


# =========================================================================
# Volatility Features
# =========================================================================

def compute_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int = 14) -> np.ndarray:
    """ATR as percentage of price (bps)."""
    atr = np.zeros(len(close), dtype=np.float32)
    tr = np.zeros(len(close), dtype=np.float32)

    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    # EMA of TR
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    # Convert to bps
    for i in range(len(close)):
        if close[i] > 0:
            atr[i] = atr[i] / close[i] * 10000

    return atr


def compute_atr_percentile(atr: np.ndarray, window: int = 100) -> np.ndarray:
    """ATR rank vs last N bars (0-100)."""
    result = np.full(len(atr), 50.0, dtype=np.float32)
    for i in range(window, len(atr)):
        window_vals = atr[i - window:i + 1]
        rank = np.sum(window_vals <= atr[i])
        result[i] = rank / len(window_vals) * 100
    return result


def compute_ema_separation(close: np.ndarray, fast: int = 9, slow: int = 21) -> np.ndarray:
    """EMA separation: (EMA_fast - EMA_slow) / EMA_slow * 10000 bps."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    sep = np.zeros(len(close), dtype=np.float32)
    for i in range(slow, len(close)):
        if ema_slow[i] > 0:
            sep[i] = (ema_fast[i] - ema_slow[i]) / ema_slow[i] * 10000
    return sep


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    result = np.zeros(len(data), dtype=np.float64)
    multiplier = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
    return result


# =========================================================================
# Bar Characteristics Features
# =========================================================================

def compute_range_bps(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Current bar range (high - low) in bps."""
    result = np.zeros(len(close), dtype=np.float32)
    for i in range(len(close)):
        if low[i] > 0:
            result[i] = (high[i] - low[i]) / low[i] * 10000
    return result


def compute_body_bps(open_arr: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Current bar body (abs(close - open)) in bps."""
    result = np.zeros(len(close), dtype=np.float32)
    for i in range(len(close)):
        if open_arr[i] > 0:
            result[i] = abs(close[i] - open_arr[i]) / open_arr[i] * 10000
    return result


def compute_dist_from_high20(close: np.ndarray, high: np.ndarray, period: int = 20) -> np.ndarray:
    """Price position relative to 20-bar high (percentage)."""
    result = np.zeros(len(close), dtype=np.float32)
    for i in range(period, len(close)):
        h20 = np.max(high[i - period:i + 1])
        if h20 > 0:
            result[i] = (h20 - close[i]) / h20 * 100
    return result


# =========================================================================
# Activity Features
# =========================================================================

def compute_volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume relative to 20-bar average."""
    result = np.ones(len(volume), dtype=np.float32)
    for i in range(period, len(volume)):
        avg = np.mean(volume[i - period:i])
        if avg > 0:
            result[i] = volume[i] / avg
    return result


# =========================================================================
# Main Processing Function
# =========================================================================

def compute_all_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute all enabled features for every bar.

    Input: OHLCV dataframe
    Output: Feature dataframe with one column per enabled feature
    """
    feat_cfg = config["features"]
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_arr = df["open"].values
    volume = df["volume"].values

    features = pd.DataFrame(index=df.index)

    # --- Price Action ---
    if feat_cfg.get("price_action", {}).get("enabled", False):
        pa = feat_cfg["price_action"]
        print("  Computing price action features...")
        if pa.get("roc"):
            features["roc"] = compute_roc(close)
        if pa.get("rsi7"):
            features["rsi7"] = compute_rsi(close, 7)
        if pa.get("range_position"):
            features["range_position"] = compute_range_position(close, high, low, 50)

    # --- Market Structure (S/R) ---
    if feat_cfg.get("market_structure", {}).get("enabled", False):
        print("  Computing S/R features...")
        sr_df = compute_sr_features(high, low, close, config)
        sr_df.index = df.index
        if feat_cfg["market_structure"].get("zone_relative", False):
            sr_lookback = feat_cfg["market_structure"].get("sr_lookback", 25)
            print("  Converting S/R to zone-relative features...")
            sr_df = convert_sr_to_zone_relative(sr_df, close, lookback=sr_lookback)
        features = pd.concat([features, sr_df], axis=1)

    # --- Volatility ---
    if feat_cfg.get("volatility", {}).get("enabled", False):
        vol = feat_cfg["volatility"]
        print("  Computing volatility features...")
        if vol.get("atr_pct"):
            features["atr_pct"] = compute_atr_pct(high, low, close)
        if vol.get("atr_percentile"):
            atr_raw = compute_atr_pct(high, low, close)
            features["atr_percentile"] = compute_atr_percentile(atr_raw)
        if vol.get("ema_separation"):
            features["ema_separation"] = compute_ema_separation(close)

    # --- Bar Characteristics ---
    if feat_cfg.get("bar_characteristics", {}).get("enabled", False):
        bc = feat_cfg["bar_characteristics"]
        print("  Computing bar characteristics...")
        if bc.get("range_bps"):
            features["range_bps"] = compute_range_bps(high, low, close)
        if bc.get("body_bps"):
            features["body_bps"] = compute_body_bps(open_arr, close)
        if bc.get("dist_from_high20_pct"):
            features["dist_from_high20_pct"] = compute_dist_from_high20(close, high)

    # --- Activity ---
    if feat_cfg.get("activity", {}).get("enabled", False):
        act = feat_cfg["activity"]
        print("  Computing activity features...")
        if act.get("volume_ratio"):
            features["volume_ratio"] = compute_volume_ratio(volume)

    print(f"  Features computed: {len(features.columns)} columns, {len(features)} bars")

    return features
