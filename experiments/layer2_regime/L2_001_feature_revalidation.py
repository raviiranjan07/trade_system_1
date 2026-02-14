"""L2-001: Feature Re-Validation on 15-Minute Data (COMPLETE SET)

Re-validates ALL 38 WHEN features + SMA200 features + extras on 15-min bars.

Following PLAN.md methodology:
  Stage A (Raw Opportunity):
    1. Directional accuracy: does this feature predict UP vs DOWN?
    2. P(Case1): does this feature predict structural failure?
    3. Raw MFE: does this feature separate high-MFE from low-MFE bars?
  Stage B (V1.3.2 Comparison):
    4. V1.3.2 PnL: LONG-only and SHORT-only per feature quartile
  Comparison:
    5. Raw MFE vs V1.3.2 captured: how much edge is left on the table?

Data: BTCUSDT 15-min OHLCV
Train: 2020-2023, OOS: 2024-2025

Run: PYTHONPATH=src python experiments/layer2_regime/L2_001_feature_revalidation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from v12.config.loader import load_config
from v12.config.constants import FEES_BPS

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")

TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2025-12-31"

# Horizons for directional/Case1 testing (in 15-min bars)
HORIZONS = [10, 15, 20]
# Target in bps for Case1 definition (minimum profitable move)
TARGET_BPS = 25


# =============================================================================
# Feature computation — ALL 38 WHEN features + SMA200 + extras
# =============================================================================

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ALL features: 38 from WHEN + SMA200 + extras = ~51 features."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    opn = out["open"]

    # ===== VOLATILITY (from WHEN) =====
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr14 = tr.ewm(span=14, adjust=False).mean()
    out["atr_pct"] = atr14 / close * 100
    out["atr_percentile"] = (atr14 / close * 100).rolling(window=500, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    for period in [7, 21]:
        atr_p = tr.ewm(span=period, adjust=False).mean()
        out[f"atr{period}_pct"] = atr_p / close * 100

    out["std20"] = close.rolling(20).std() / close * 100

    # BB position (from WHEN — different from bb_width)
    ema20 = close.ewm(span=20, adjust=False).mean()
    bb_std = close.rolling(20).std()
    bb_upper = ema20 + 2 * bb_std
    bb_lower = ema20 - 2 * bb_std
    out["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 0.0001)

    # ===== EXTRAS: bandwidth features (from L2-001 v1) =====
    out["keltner_width"] = (atr14 * 2) / ema20 * 100
    donchian_high = high.rolling(50).max()
    donchian_low = low.rolling(50).min()
    out["donchian_width"] = (donchian_high - donchian_low) / close * 100
    sma20 = close.rolling(20).mean()
    out["bb_width"] = (bb_std * 4) / sma20 * 100

    # ===== MOVING AVERAGES (from WHEN — ALL were INVALID) =====
    for period in [9, 20, 50, 100, 200]:
        ema = close.ewm(span=period, adjust=False).mean()
        out[f"ema{period}"] = ema
        out[f"ema{period}_dist_pct"] = (close - ema) / ema * 100

    for period in [20, 50, 200]:
        out[f"ema{period}_slope"] = (out[f"ema{period}"] - out[f"ema{period}"].shift(5)) / out[f"ema{period}"].shift(5) * 100

    # ===== TREND =====
    out["ema_separation"] = abs(out["ema50"] - out["ema200"]) / close * 100

    # SMA200 features (NEW — core of V1.3.2)
    sma200 = close.rolling(200).mean()
    out["sma200_dist_pct"] = (close - sma200) / sma200 * 100
    out["sma200_slope"] = (sma200 - sma200.shift(5)) / sma200.shift(5) * 100
    out["price_above_sma200"] = (close > sma200).astype(int)

    out["bull_market"] = (close > sma200).astype(int)
    out["bear_market"] = (close < sma200).astype(int)

    # ===== MOMENTUM (from WHEN — ALL were INVALID) =====
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss_val = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss_val + 0.0001)
    out["rsi"] = 100 - (100 / (1 + rs))

    for period in [7, 21]:
        gain_p = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
        loss_p = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs_p = gain_p / (loss_p + 0.0001)
        out[f"rsi{period}"] = 100 - (100 / (1 + rs_p))

    for period in [5, 10, 20]:
        out[f"roc{period}"] = (close - close.shift(period)) / close.shift(period) * 100

    out["momentum5"] = close - close.shift(5)
    out["momentum10"] = close - close.shift(10)

    # ===== PRICE STRUCTURE (from WHEN) =====
    out["range_bps"] = (high - low) / close * 10000
    out["body_bps"] = abs(close - opn) / close * 10000
    out["range_position"] = (close - low) / (high - low + 0.0001)

    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    out["dist_from_high20_pct"] = (high20 - close) / close * 100
    out["dist_from_low20_pct"] = (close - low20) / close * 100

    # Structure
    higher_high = (high > high.shift(1)).astype(int)
    lower_low = (low < low.shift(1)).astype(int)
    out["hh_count5"] = higher_high.rolling(5).sum()
    out["ll_count5"] = lower_low.rolling(5).sum()
    out["up_bars5"] = (close > opn).astype(int).rolling(5).sum()
    out["down_bars5"] = (close < opn).astype(int).rolling(5).sum()

    # ===== EXTRAS (from L2-001 v1) =====
    out["bar_range_avg_10"] = out["range_bps"].rolling(10).mean()
    out["recent_volatility"] = close.pct_change().rolling(10).std() * 10000

    # Range position (WHAT-specific)
    high50 = high.rolling(50).max()
    low50 = low.rolling(50).min()
    range50 = high50 - low50
    out["range_position_50"] = (close - low50) / range50.replace(0, np.nan) * 100

    # RSI zones (from L2-001 v1)
    out["rsi_oversold_zone"] = (out["rsi"] < 30).astype(int)
    out["rsi_extreme_oversold"] = (out["rsi"] < 20).astype(int)

    # ===== VOLUME (from WHEN — were INVALID) =====
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(20).mean()
    out["volume_trend"] = out["volume"].rolling(5).mean() / out["volume"].rolling(20).mean()

    # ===== TIME =====
    out["hour_utc"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["session"] = pd.cut(
        out.index.hour, bins=[-1, 4, 8, 12, 16, 20, 24],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)
    out["session_asia_night"] = ((out.index.hour >= 0) & (out.index.hour < 4)).astype(int)
    out["session_europe"] = ((out.index.hour >= 8) & (out.index.hour < 12)).astype(int)
    out["session_us"] = ((out.index.hour >= 16) & (out.index.hour < 20)).astype(int)

    return out


# =============================================================================
# Test 1: Directional Accuracy
# =============================================================================

def test_directional_accuracy(df: pd.DataFrame, feature: str, horizon: int,
                              target_bps: float = TARGET_BPS) -> dict:
    """Test: does this feature predict which direction hits target first?"""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    feat_vals = df[feature].values
    n = len(df)

    up_first = np.zeros(n, dtype=bool)
    down_first = np.zeros(n, dtype=bool)
    neither = np.zeros(n, dtype=bool)

    for i in range(n - horizon - 1):
        entry = opens[i + 1]
        if entry <= 0 or np.isnan(entry):
            neither[i] = True
            continue

        up_target = entry * (1 + target_bps / 10000)
        down_target = entry * (1 - target_bps / 10000)

        hit_up = False
        hit_down = False
        for j in range(1, horizon + 1):
            idx = i + 1 + j
            if idx >= n:
                break
            if not hit_up and highs[idx] >= up_target:
                hit_up = True
                if not hit_down:
                    up_first[i] = True
                    break
            if not hit_down and lows[idx] <= down_target:
                hit_down = True
                if not hit_up:
                    down_first[i] = True
                    break

        if not hit_up and not hit_down:
            neither[i] = True

    valid = ~np.isnan(feat_vals) & (np.arange(n) < n - horizon - 1)
    if valid.sum() < 100:
        return None

    valid_feat = feat_vals[valid]
    try:
        q25, q50, q75 = np.percentile(valid_feat, [25, 50, 75])
    except Exception:
        return None

    results = {}
    for label, mask_fn in [
        ("Q1", lambda v: v <= q25),
        ("Q4", lambda v: v > q75),
        ("ALL", lambda v: np.ones_like(v, dtype=bool)),
    ]:
        mask = valid & mask_fn(feat_vals)
        total = mask.sum()
        if total == 0:
            continue
        n_up = up_first[mask].sum()
        n_down = down_first[mask].sum()
        n_real = n_up + n_down

        results[label] = {
            "total": total,
            "up_first_pct": n_up / total * 100 if total > 0 else 0,
            "up_accuracy": n_up / n_real * 100 if n_real > 0 else 50,
        }

    return results


# =============================================================================
# Test 2: P(Case1) — structural failure rate
# =============================================================================

def test_case1_rate(df: pd.DataFrame, feature: str, horizon: int,
                    target_bps: float = TARGET_BPS) -> dict:
    """Test: does this feature predict Case 1 (price NEVER hits target)?"""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    feat_vals = df[feature].values
    n = len(df)

    long_hit = np.zeros(n, dtype=bool)
    short_hit = np.zeros(n, dtype=bool)

    for i in range(n - horizon - 1):
        entry = opens[i + 1]
        if entry <= 0 or np.isnan(entry):
            continue

        up_target = entry * (1 + target_bps / 10000)
        down_target = entry * (1 - target_bps / 10000)

        for j in range(1, horizon + 1):
            idx = i + 1 + j
            if idx >= n:
                break
            if highs[idx] >= up_target:
                long_hit[i] = True
            if lows[idx] <= down_target:
                short_hit[i] = True
            if long_hit[i] and short_hit[i]:
                break

    long_case1 = ~long_hit
    short_case1 = ~short_hit

    valid = ~np.isnan(feat_vals) & (np.arange(n) < n - horizon - 1)
    if valid.sum() < 100:
        return None

    valid_feat = feat_vals[valid]
    try:
        q25, q75 = np.percentile(valid_feat, [25, 75])
    except Exception:
        return None

    results = {}
    for label, mask_fn in [
        ("Q1", lambda v: v <= q25),
        ("Q4", lambda v: v > q75),
        ("ALL", lambda v: np.ones_like(v, dtype=bool)),
    ]:
        mask = valid & mask_fn(feat_vals)
        total = mask.sum()
        if total == 0:
            continue

        results[label] = {
            "total": total,
            "long_case1_pct": long_case1[mask].mean() * 100,
            "short_case1_pct": short_case1[mask].mean() * 100,
            "avg_case1_pct": (long_case1[mask].mean() + short_case1[mask].mean()) / 2 * 100,
        }

    return results


# =============================================================================
# Test 3: Raw MFE per quartile (Stage A — no exits, pure opportunity)
# =============================================================================

def test_raw_mfe(df: pd.DataFrame, feature: str, horizon: int,
                 max_per_quartile: int = 3000) -> dict:
    """Measure raw forward MFE per feature quartile — NO exit mechanics.

    For each bar: enter at next bar's open, measure max favorable excursion
    over `horizon` bars for both LONG and SHORT directions.
    """
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    feat_vals = df[feature].values
    n = len(df)

    valid = ~np.isnan(feat_vals) & (np.arange(n) < n - horizon - 1)
    if valid.sum() < 100:
        return None

    valid_feat = feat_vals[valid]
    try:
        q25, q75 = np.percentile(valid_feat, [25, 75])
    except Exception:
        return None

    if q25 == q75:
        return None

    results = {}
    for label, mask_fn in [("Q1", lambda v: v <= q25), ("Q4", lambda v: v > q75)]:
        indices = np.where(valid & mask_fn(feat_vals))[0]

        if len(indices) > max_per_quartile:
            indices = np.random.choice(indices, max_per_quartile, replace=False)

        long_mfes = []
        short_mfes = []

        for i in indices:
            entry = opens[i + 1]
            if entry <= 0 or np.isnan(entry):
                continue

            long_mfe = 0.0
            short_mfe = 0.0
            for j in range(1, horizon + 1):
                idx = i + 1 + j
                if idx >= n:
                    break
                lmfe = (highs[idx] - entry) / entry * 10000
                smfe = (entry - lows[idx]) / entry * 10000
                long_mfe = max(long_mfe, lmfe)
                short_mfe = max(short_mfe, smfe)

            long_mfes.append(long_mfe)
            short_mfes.append(short_mfe)

        long_arr = np.array(long_mfes) if long_mfes else np.array([0])
        short_arr = np.array(short_mfes) if short_mfes else np.array([0])

        results[label] = {
            "n_bars": len(indices),
            "long_median_mfe": np.median(long_arr),
            "long_mean_mfe": long_arr.mean(),
            "long_pct_above_25bp": (long_arr >= 25).mean() * 100,
            "short_median_mfe": np.median(short_arr),
            "short_mean_mfe": short_arr.mean(),
            "short_pct_above_25bp": (short_arr >= 25).mean() * 100,
        }

    return results


# =============================================================================
# Test 4: V1.3.2 actual trade PnL (LONG-only and SHORT-only separately)
# =============================================================================

def simulate_v132_trade(opens, highs, lows, closes, start_idx, direction, config):
    """Simulate one V1.3.2 trade. Returns (net_pnl_bps, exit_reason) or None."""
    n = len(opens)
    if start_idx + 1 >= n:
        return None

    entry_price = opens[start_idx + 1]
    if entry_price <= 0 or np.isnan(entry_price):
        return None

    ts = config.exit.long_trailing_stop_bps if direction == "LONG" else config.exit.short_trailing_stop_bps
    tighten_bar = config.exit.tighten_after_bar
    tight_ts = config.exit.tight_trailing_stop_bps
    max_bars = config.exit.max_bars

    highest_profit = 0.0

    for bar in range(1, max_bars + 1):
        idx = start_idx + 1 + bar
        if idx >= n:
            break

        h, l, c = highs[idx], lows[idx], closes[idx]

        if direction == "LONG":
            bar_mfe = (h - entry_price) / entry_price * 10000
            bar_pnl = (c - entry_price) / entry_price * 10000
        else:
            bar_mfe = (entry_price - l) / entry_price * 10000
            bar_pnl = (entry_price - c) / entry_price * 10000

        highest_profit = max(highest_profit, bar_mfe)

        active_ts = tight_ts if bar > tighten_bar else ts
        drawdown = highest_profit - bar_pnl
        if drawdown >= active_ts and highest_profit > 0:
            gross = highest_profit - active_ts
            return (gross - FEES_BPS, "TS")

        if bar >= max_bars:
            return (bar_pnl - FEES_BPS, "TIME")

    return None


def test_v132_pnl(df: pd.DataFrame, feature: str, config,
                  max_per_quartile: int = 3000) -> dict:
    """Test LONG-only and SHORT-only V1.3.2 PnL per feature quartile."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    feat_vals = df[feature].values
    n = len(df)

    valid = ~np.isnan(feat_vals) & (np.arange(n) < n - 12)
    if valid.sum() < 100:
        return None

    valid_feat = feat_vals[valid]
    try:
        q25, q75 = np.percentile(valid_feat, [25, 75])
    except Exception:
        return None

    if q25 == q75:
        return None

    results = {}
    for label, mask_fn in [("Q1", lambda v: v <= q25), ("Q4", lambda v: v > q75)]:
        indices = np.where(valid & mask_fn(feat_vals))[0]

        if len(indices) > max_per_quartile:
            indices = np.random.choice(indices, max_per_quartile, replace=False)

        long_pnls = []
        short_pnls = []

        for idx in indices:
            lr = simulate_v132_trade(opens, highs, lows, closes, idx, "LONG", config)
            sr = simulate_v132_trade(opens, highs, lows, closes, idx, "SHORT", config)
            if lr:
                long_pnls.append(lr[0])
            if sr:
                short_pnls.append(sr[0])

        long_arr = np.array(long_pnls) if long_pnls else np.array([0])
        short_arr = np.array(short_pnls) if short_pnls else np.array([0])

        results[label] = {
            "n_bars": len(indices),
            "long_n": len(long_pnls),
            "long_win_pct": (long_arr > 0).mean() * 100,
            "long_avg_pnl": long_arr.mean(),
            "long_total": long_arr.sum(),
            "short_n": len(short_pnls),
            "short_win_pct": (short_arr > 0).mean() * 100,
            "short_avg_pnl": short_arr.mean(),
            "short_total": short_arr.sum(),
        }

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("L2-001: Feature Re-Validation on 15-Minute Data (COMPLETE SET)")
    print("Re-validating ALL 38 WHEN + SMA200 + extras (~51 features)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"Total bars: {len(df)}")

    config = load_config()

    # Compute features
    print("Computing features...")
    df = compute_all_features(df)

    # COMPLETE feature list: 38 WHEN + SMA200 + extras
    feature_cols = [
        # --- WHEN ROBUST (11) ---
        "atr_pct", "atr7_pct", "atr21_pct", "std20",       # volatility
        "range_bps", "body_bps",                             # price
        "ema_separation",                                    # trend
        "dist_from_high20_pct", "dist_from_low20_pct",       # structure
        "ll_count5",                                         # structure
        "day_of_week",                                       # time

        # --- WHEN STRONG (3) ---
        "hh_count5", "atr_percentile", "session",

        # --- WHEN PARTIAL (1) ---
        "hour_utc",

        # --- WHEN WEAK (3) ---
        "up_bars5", "down_bars5", "volume_ratio",

        # --- WHEN INVALID (19) — RE-TESTING on 15-min ---
        "rsi", "rsi7", "rsi21",                              # momentum
        "roc5", "roc10", "roc20",                            # momentum
        "momentum5", "momentum10",                           # momentum
        "ema9_dist_pct", "ema20_dist_pct", "ema50_dist_pct", # MA distance
        "ema100_dist_pct", "ema200_dist_pct",                 # MA distance
        "ema20_slope", "ema50_slope", "ema200_slope",         # MA slope
        "bb_position",                                        # volatility
        "volume_trend",                                       # volume
        "range_position",                                     # price

        # --- SMA200 features (NEW) ---
        "sma200_dist_pct", "sma200_slope", "price_above_sma200",

        # --- L2-001 v1 extras ---
        "keltner_width", "donchian_width", "bb_width",
        "bar_range_avg_10", "recent_volatility",
        "range_position_50",
        "rsi_oversold_zone", "rsi_extreme_oversold",
        "is_weekend", "session_asia_night", "session_europe", "session_us",
    ]

    print(f"Testing {len(feature_cols)} features\n")

    train = df[TRAIN_START:TRAIN_END].copy()
    oos = df[OOS_START:OOS_END].copy()
    print(f"TRAIN: {len(train)} bars | OOS: {len(oos)} bars\n")

    # =========================================================================
    # TEST 1: Directional Accuracy (WHAT Analysis re-validation)
    # =========================================================================
    print("=" * 70)
    print(f"TEST 1: Directional Accuracy (target={TARGET_BPS}bp)")
    print("Does this feature predict UP vs DOWN?")
    print("=" * 70)

    dir_results = {}  # feature -> {train_diff, oos_diff, valid}

    for h in HORIZONS:
        print(f"\n--- Horizon {h} bars ({h*15} min) ---")
        print(f"{'Feature':<22} {'Q1 UP%':>7} {'Q4 UP%':>7} {'Diff':>7} | "
              f"{'Q1_oos':>7} {'Q4_oos':>7} {'Diff_o':>7} {'Valid':>6}")
        print("-" * 85)

        for feat in feature_cols:
            if feat not in train.columns:
                continue
            t = test_directional_accuracy(train, feat, h)
            o = test_directional_accuracy(oos, feat, h)

            if t is None or o is None:
                continue
            if "Q1" not in t or "Q4" not in t or "Q1" not in o or "Q4" not in o:
                continue

            t_q1 = t["Q1"]["up_accuracy"]
            t_q4 = t["Q4"]["up_accuracy"]
            o_q1 = o["Q1"]["up_accuracy"]
            o_q4 = o["Q4"]["up_accuracy"]
            t_diff = t_q4 - t_q1
            o_diff = o_q4 - o_q1

            same_dir = (t_diff > 0 and o_diff > 0) or (t_diff < 0 and o_diff < 0)
            big_enough = abs(t_diff) > 1.0
            valid = "YES" if (same_dir and big_enough) else "no"

            # Store H=10 result for summary
            if h == 10:
                dir_results[feat] = {"train": t_diff, "oos": o_diff, "valid": valid}

            print(f"{feat:<22} {t_q1:>6.1f}% {t_q4:>6.1f}% {t_diff:>+6.1f} | "
                  f"{o_q1:>6.1f}% {o_q4:>6.1f}% {o_diff:>+6.1f} {valid:>6}")

    # =========================================================================
    # TEST 2: P(Case1) — Structural Failure Rate (WHEN Analysis re-validation)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"TEST 2: P(Case 1) — Failure Rate (target={TARGET_BPS}bp)")
    print("Does this feature predict structural failure?")
    print("=" * 70)

    c1_results = {}

    for h in HORIZONS:
        print(f"\n--- Horizon {h} bars ({h*15} min) ---")
        print(f"{'Feature':<22} {'Q1 C1%':>7} {'Q4 C1%':>7} {'Diff':>7} | "
              f"{'Q1_oos':>7} {'Q4_oos':>7} {'Diff_o':>7} {'Valid':>6}")
        print("-" * 85)

        for feat in feature_cols:
            if feat not in train.columns:
                continue
            t = test_case1_rate(train, feat, h)
            o = test_case1_rate(oos, feat, h)

            if t is None or o is None:
                continue
            if "Q1" not in t or "Q4" not in t or "Q1" not in o or "Q4" not in o:
                continue

            t_q1 = t["Q1"]["avg_case1_pct"]
            t_q4 = t["Q4"]["avg_case1_pct"]
            o_q1 = o["Q1"]["avg_case1_pct"]
            o_q4 = o["Q4"]["avg_case1_pct"]
            t_diff = t_q4 - t_q1
            o_diff = o_q4 - o_q1

            same_dir = (t_diff < 0 and o_diff < 0) or (t_diff > 0 and o_diff > 0)
            big_enough = abs(t_diff) > 2.0
            valid = "YES" if (same_dir and big_enough) else "no"

            if h == 10:
                c1_results[feat] = {"train": t_diff, "oos": o_diff, "valid": valid}

            print(f"{feat:<22} {t_q1:>6.1f}% {t_q4:>6.1f}% {t_diff:>+6.1f} | "
                  f"{o_q1:>6.1f}% {o_q4:>6.1f}% {o_diff:>+6.1f} {valid:>6}")

    # =========================================================================
    # STAGE A — TEST 3: Raw MFE per quartile (no exits)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3 (Stage A): Raw Forward MFE — NO exit mechanics")
    print("How much raw opportunity exists per feature quartile?")
    print("=" * 70)

    mfe_results = {}  # feature -> {Q1_long_median, Q4_long_median, ...}

    for h in [10]:  # Primary horizon only for MFE
        print(f"\n--- Horizon {h} bars ({h*15} min) ---")
        print(f"{'Feature':<22} {'Q':<3} | {'L Med MFE':>9} {'L Mean':>7} {'L>25bp':>7} | "
              f"{'S Med MFE':>9} {'S Mean':>7} {'S>25bp':>7} | OOS same")
        print("-" * 105)

        for feat in feature_cols:
            if feat not in train.columns:
                continue
            t = test_raw_mfe(train, feat, h)
            o = test_raw_mfe(oos, feat, h)

            if t is None or o is None:
                continue

            mfe_results[feat] = {}
            for q in ["Q4", "Q1"]:
                if q not in t or q not in o:
                    continue
                tr = t[q]
                osr = o[q]
                mfe_results[feat][q] = {
                    "train_long_med": tr["long_median_mfe"],
                    "train_short_med": tr["short_median_mfe"],
                    "oos_long_med": osr["long_median_mfe"],
                    "oos_short_med": osr["short_median_mfe"],
                }

                print(f"{feat:<22} {q:<3} | {tr['long_median_mfe']:>8.1f} {tr['long_mean_mfe']:>+6.1f} "
                      f"{tr['long_pct_above_25bp']:>6.1f}% | "
                      f"{tr['short_median_mfe']:>8.1f} {tr['short_mean_mfe']:>+6.1f} "
                      f"{tr['short_pct_above_25bp']:>6.1f}% | "
                      f"L:{osr['long_median_mfe']:.0f} S:{osr['short_median_mfe']:.0f}")
            print()

    # =========================================================================
    # STAGE B — TEST 4: V1.3.2 Trade PnL (LONG-only and SHORT-only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4 (Stage B): V1.3.2 Trade PnL (LONG and SHORT separately)")
    print("=" * 70)

    print(f"\n{'Feature':<22} {'Q':<3} | {'LONG Win%':>9} {'L Avg':>7} | "
          f"{'SHORT Win%':>10} {'S Avg':>7} | {'L_oos%':>7} {'S_oos%':>7}")
    print("-" * 95)

    pnl_results = {}
    all_csv_rows = []

    for feat in feature_cols:
        if feat not in train.columns:
            continue
        t = test_v132_pnl(train, feat, config)
        o = test_v132_pnl(oos, feat, config)

        if t is None or o is None:
            continue

        long_valid = "no"
        short_valid = "no"
        if "Q4" in t and "Q4" in o:
            if t["Q4"]["long_avg_pnl"] > 0 and o["Q4"]["long_avg_pnl"] > 0:
                long_valid = "YES"
            if t["Q4"]["short_avg_pnl"] > 0 and o["Q4"]["short_avg_pnl"] > 0:
                short_valid = "YES"
        pnl_results[feat] = {
            "long_valid": long_valid, "short_valid": short_valid,
            "train_Q4": t.get("Q4", {}), "oos_Q4": o.get("Q4", {}),
        }

        for q in ["Q4", "Q1"]:
            if q not in t or q not in o:
                continue
            tr = t[q]
            osr = o[q]

            print(f"{feat:<22} {q:<3} | {tr['long_win_pct']:>8.1f}% {tr['long_avg_pnl']:>+6.1f} | "
                  f"{tr['short_win_pct']:>9.1f}% {tr['short_avg_pnl']:>+6.1f} | "
                  f"{osr['long_win_pct']:>6.1f}% {osr['short_win_pct']:>6.1f}%")

            all_csv_rows.append({
                "feature": feat, "quartile": q,
                "train_long_win": tr["long_win_pct"], "train_long_avg": tr["long_avg_pnl"],
                "train_short_win": tr["short_win_pct"], "train_short_avg": tr["short_avg_pnl"],
                "oos_long_win": osr["long_win_pct"], "oos_long_avg": osr["long_avg_pnl"],
                "oos_short_win": osr["short_win_pct"], "oos_short_avg": osr["short_avg_pnl"],
            })

        print()

    # =========================================================================
    # COMPARISON: Raw MFE vs V1.3.2 Captured (Step 5 from PLAN.md)
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Raw Opportunity vs V1.3.2 Captured (Q4 only)")
    print("Efficiency = V1.3.2 avg PnL / Raw median MFE")
    print("=" * 70)

    print(f"\n{'Feature':<22} | {'Raw L MFE':>9} {'V132 L PnL':>10} {'L Eff%':>7} | "
          f"{'Raw S MFE':>9} {'V132 S PnL':>10} {'S Eff%':>7}")
    print("-" * 85)

    for feat in feature_cols:
        if feat not in mfe_results or feat not in pnl_results:
            continue
        mfe = mfe_results[feat].get("Q4", {})
        pnl = pnl_results[feat]

        if not mfe or "train_Q4" not in pnl or not pnl["train_Q4"]:
            continue

        raw_l = mfe.get("train_long_med", 0)
        raw_s = mfe.get("train_short_med", 0)
        cap_l = pnl["train_Q4"].get("long_avg_pnl", 0)
        cap_s = pnl["train_Q4"].get("short_avg_pnl", 0)

        eff_l = (cap_l / raw_l * 100) if raw_l > 0 else 0
        eff_s = (cap_s / raw_s * 100) if raw_s > 0 else 0

        print(f"{feat:<22} | {raw_l:>8.1f} {cap_l:>+9.1f} {eff_l:>6.1f}% | "
              f"{raw_s:>8.1f} {cap_s:>+9.1f} {eff_s:>6.1f}%")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: ALL FEATURES (H=10)")
    print("5 checks: Direction, Case1, Raw MFE, V1.3.2 LONG, V1.3.2 SHORT")
    print("=" * 70)

    # Group by WHEN category
    when_robust = ["atr_pct", "atr7_pct", "atr21_pct", "std20", "range_bps",
                   "body_bps", "ema_separation", "dist_from_high20_pct",
                   "dist_from_low20_pct", "ll_count5", "day_of_week"]
    when_strong = ["hh_count5", "atr_percentile", "session"]
    when_partial = ["hour_utc"]
    when_weak = ["up_bars5", "down_bars5", "volume_ratio"]
    when_invalid = ["rsi", "rsi7", "rsi21", "roc5", "roc10", "roc20",
                    "momentum5", "momentum10", "ema9_dist_pct", "ema20_dist_pct",
                    "ema50_dist_pct", "ema100_dist_pct", "ema200_dist_pct",
                    "ema20_slope", "ema50_slope", "ema200_slope",
                    "bb_position", "volume_trend", "range_position"]
    sma200_new = ["sma200_dist_pct", "sma200_slope", "price_above_sma200"]
    extras = ["keltner_width", "donchian_width", "bb_width", "bar_range_avg_10",
              "recent_volatility", "range_position_50", "rsi_oversold_zone",
              "rsi_extreme_oversold", "is_weekend", "session_asia_night",
              "session_europe", "session_us"]

    groups = [
        ("WHEN ROBUST (11)", when_robust),
        ("WHEN STRONG (3)", when_strong),
        ("WHEN PARTIAL (1)", when_partial),
        ("WHEN WEAK (3)", when_weak),
        ("WHEN INVALID (19)", when_invalid),
        ("SMA200 NEW (3)", sma200_new),
        ("L2-001 EXTRAS (12)", extras),
    ]

    print(f"\n{'Feature':<22} {'WHEN':>8} {'Dir':>5} {'C1':>5} {'MFE':>5} {'LONG':>5} {'SHORT':>5} {'Score':>6}")
    print("-" * 70)

    for group_name, feats in groups:
        print(f"\n--- {group_name} ---")
        for feat in feats:
            when_label = "ROBUST" if feat in when_robust else \
                         "STRONG" if feat in when_strong else \
                         "PARTIAL" if feat in when_partial else \
                         "WEAK" if feat in when_weak else \
                         "INVALID" if feat in when_invalid else \
                         "NEW" if feat in sma200_new else "EXTRA"

            d = dir_results.get(feat, {}).get("valid", "?")
            c = c1_results.get(feat, {}).get("valid", "?")
            lv = pnl_results.get(feat, {}).get("long_valid", "?")
            sv = pnl_results.get(feat, {}).get("short_valid", "?")

            # MFE valid = Q4 median MFE > Q1 median MFE on both train and OOS
            mfe_valid = "?"
            mfe = mfe_results.get(feat, {})
            if "Q4" in mfe and "Q1" in mfe:
                q4_l = mfe["Q4"].get("train_long_med", 0)
                q1_l = mfe["Q1"].get("train_long_med", 0)
                q4_l_o = mfe["Q4"].get("oos_long_med", 0)
                q1_l_o = mfe["Q1"].get("oos_long_med", 0)
                if q4_l > q1_l and q4_l_o > q1_l_o and (q4_l - q1_l) > 5:
                    mfe_valid = "YES"
                else:
                    mfe_valid = "no"

            yes_count = sum(1 for x in [d, c, mfe_valid, lv, sv] if x == "YES")
            score = f"{yes_count}/5"

            print(f"{feat:<22} {when_label:>8} {d:>5} {c:>5} {mfe_valid:>5} {lv:>5} {sv:>5} {score:>6}")

    # Save CSV
    if all_csv_rows:
        out_dir = Path("experiments/layer2_regime")
        pd.DataFrame(all_csv_rows).to_csv(out_dir / "L2_001_v132_pnl_results.csv", index=False)
        print(f"\nDetailed results saved to {out_dir}/L2_001_v132_pnl_results.csv")


if __name__ == "__main__":
    np.random.seed(42)
    main()
