"""
Case Labeler: Classify each bar into Case 0/1/2/3 based on recovery behavior.

Case Definitions (for LONG trades):
- Case 0: Clean Win - hit target without going below entry
- Case 1: Wrong Direction - went below entry, never hit target (even with extended time)
- Case 2: Quick Recovery - went below entry, but hit target within H bars
- Case 3: Slow Recovery - went below entry, didn't hit within H, but hit target later

This is the foundation of the Risk-Based Entry Filter system.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from numba import njit, prange

from ..config import Config


# Default configuration
DEFAULT_TARGETS_BPS = [8, 15, 25]
DEFAULT_HORIZONS = [3, 5, 10, 15, 30, 60]
DEFAULT_EXTENDED_HORIZON = 500


@njit
def classify_single_bar(
    entry: float,
    highs: np.ndarray,
    lows: np.ndarray,
    target_pct: float,
    H: int,
    extended_H: int
) -> Tuple[int, float, int]:
    """
    Classify a single bar into Case 0/1/2/3.

    Returns:
        case: 0 (clean win), 1 (wrong dir), 2 (quick rec), 3 (slow rec), -1 (no data)
        mae_bps: Maximum Adverse Excursion in basis points
        bars_to_target: Bars until target hit (-1 if never)
    """
    n = len(highs)
    if n == 0:
        return -1, 0.0, -1

    target_price = entry * (1 + target_pct)

    went_below = False
    hit_within_H = False
    hit_extended = False
    max_adverse_bps = 0.0
    bars_to_hit = -1

    # Check within H bars
    for j in range(min(H, n)):
        # Track MAE
        adverse = (entry - lows[j]) / entry * 10000  # in bps
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse

        # Check if went below entry
        if lows[j] < entry:
            went_below = True

        # Check if hit target
        if highs[j] >= target_price:
            hit_within_H = True
            bars_to_hit = j + 1
            break

    # If went below and didn't hit within H, check extended time
    if went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            # Continue tracking MAE
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse

            # Check if hit target
            if highs[j] >= target_price:
                hit_extended = True
                bars_to_hit = j + 1
                break

    # Classify
    if not went_below and hit_within_H:
        return 0, max_adverse_bps, bars_to_hit  # Case 0: Clean Win
    elif went_below and hit_within_H:
        return 2, max_adverse_bps, bars_to_hit  # Case 2: Quick Recovery
    elif went_below and not hit_within_H and hit_extended:
        return 3, max_adverse_bps, bars_to_hit  # Case 3: Slow Recovery
    elif went_below and not hit_within_H and not hit_extended:
        return 1, max_adverse_bps, -1  # Case 1: Wrong Direction
    elif not went_below and not hit_within_H:
        # Never went below but also never hit target within H
        # Check extended
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                if went_below:
                    return 3, max_adverse_bps, j + 1  # Slow Recovery
                else:
                    return 0, max_adverse_bps, j + 1  # Clean Win (late)
        # Never hit target
        return 1, max_adverse_bps, -1  # Wrong Direction (never moved enough)

    return -1, max_adverse_bps, -1


@njit(parallel=True)
def classify_all_bars(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    target_pct: float,
    H: int,
    extended_H: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify all bars in parallel using numba.

    Returns:
        cases: array of case labels (0/1/2/3)
        maes: array of MAE values in bps
        bars_to_target: array of bars until target hit
    """
    n = len(closes)
    cases = np.zeros(n, dtype=np.int32)
    maes = np.zeros(n, dtype=np.float64)
    bars_to_target = np.zeros(n, dtype=np.int32)

    for i in prange(n - extended_H):
        entry = closes[i]
        future_highs = highs[i+1:i+1+extended_H]
        future_lows = lows[i+1:i+1+extended_H]

        case, mae, bars = classify_single_bar(
            entry, future_highs, future_lows, target_pct, H, extended_H
        )
        cases[i] = case
        maes[i] = mae
        bars_to_target[i] = bars

    # Mark last extended_H bars as invalid
    for i in range(n - extended_H, n):
        cases[i] = -1
        maes[i] = np.nan
        bars_to_target[i] = -1

    return cases, maes, bars_to_target


def label_cases(
    ohlcv_df: pd.DataFrame,
    targets_bps: Optional[List[int]] = None,
    horizons: Optional[List[int]] = None,
    extended_horizon: int = DEFAULT_EXTENDED_HORIZON,
    output_dir: str = "data/outcomes",
    pair: str = "BTCUSDT",
    timeframe: str = "1m"
) -> pd.DataFrame:
    """
    Label all bars with case classifications for each target/horizon combination.

    Args:
        ohlcv_df: DataFrame with columns [open, high, low, close, volume]
        targets_bps: List of target values in basis points (default: [8, 15, 25])
        horizons: List of horizon values in bars (default: [3, 5, 10, 15, 30, 60])
        extended_horizon: Extended horizon for recovery check (default: 500)
        output_dir: Directory to save output
        pair: Trading pair name
        timeframe: Timeframe string

    Returns:
        DataFrame with case labels for each target/horizon combination
    """
    # Get config or use defaults
    if targets_bps is None:
        try:
            config = Config()
            targets_bps = config.get("risk_model.targets_bps", DEFAULT_TARGETS_BPS)
        except Exception:
            targets_bps = DEFAULT_TARGETS_BPS

    if horizons is None:
        try:
            config = Config()
            horizons = config.get("risk_model.horizons", DEFAULT_HORIZONS)
        except Exception:
            horizons = DEFAULT_HORIZONS

    print(f"Labeling cases for {len(ohlcv_df):,} bars...")
    print(f"Targets (bps): {targets_bps}")
    print(f"Horizons: {horizons}")
    print(f"Extended horizon: {extended_horizon}")

    # Extract numpy arrays for speed
    closes = ohlcv_df['close'].values.astype(np.float64)
    highs = ohlcv_df['high'].values.astype(np.float64)
    lows = ohlcv_df['low'].values.astype(np.float64)

    # Create result DataFrame with original index
    result_df = pd.DataFrame(index=ohlcv_df.index)

    # Label for each target/horizon combination
    for target_bps in targets_bps:
        target_pct = target_bps / 10000

        for H in horizons:
            print(f"  Processing Target={target_bps}bp, H={H}...")

            cases, maes, bars = classify_all_bars(
                closes, highs, lows, target_pct, H, extended_horizon
            )

            col_prefix = f"t{target_bps}_H{H}"
            result_df[f"{col_prefix}_case"] = cases
            result_df[f"{col_prefix}_mae"] = maes
            result_df[f"{col_prefix}_bars"] = bars

            # Print distribution for validation
            valid_cases = cases[cases >= 0]
            if len(valid_cases) > 0:
                case_counts = np.bincount(valid_cases, minlength=4)
                total = len(valid_cases)
                print(f"    Case 0 (Clean): {case_counts[0]/total*100:.1f}%")
                print(f"    Case 1 (Wrong): {case_counts[1]/total*100:.1f}%")
                print(f"    Case 2 (Quick): {case_counts[2]/total*100:.1f}%")
                print(f"    Case 3 (Slow):  {case_counts[3]/total*100:.1f}%")

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{pair}_{timeframe}_cases.parquet")
    result_df.to_parquet(file_path, engine="pyarrow")
    print(f"\nSaved case labels to: {file_path}")

    return result_df


def get_case_stats(case_df: pd.DataFrame, target_bps: int, H: int) -> Dict:
    """
    Get statistics for a specific target/horizon combination.

    Returns:
        Dictionary with case probabilities and MAE statistics
    """
    col_prefix = f"t{target_bps}_H{H}"
    cases = case_df[f"{col_prefix}_case"]
    maes = case_df[f"{col_prefix}_mae"]
    bars = case_df[f"{col_prefix}_bars"]

    # Filter valid cases
    valid_mask = cases >= 0
    cases = cases[valid_mask]
    maes = maes[valid_mask]
    bars = bars[valid_mask]

    total = len(cases)
    if total == 0:
        return {}

    # Case probabilities
    case_counts = cases.value_counts()
    p_case0 = case_counts.get(0, 0) / total
    p_case1 = case_counts.get(1, 0) / total
    p_case2 = case_counts.get(2, 0) / total
    p_case3 = case_counts.get(3, 0) / total

    # MAE statistics by case
    mae_stats = {}
    for case_num in [0, 1, 2, 3]:
        case_maes = maes[cases == case_num]
        if len(case_maes) > 0:
            mae_stats[f"mae_case{case_num}_median"] = case_maes.median()
            mae_stats[f"mae_case{case_num}_75pct"] = case_maes.quantile(0.75)
            mae_stats[f"mae_case{case_num}_90pct"] = case_maes.quantile(0.90)

    # Recovery time statistics (for Case 2 and 3)
    recovery_cases = bars[(cases == 2) | (cases == 3)]
    recovery_cases = recovery_cases[recovery_cases > 0]

    recovery_stats = {}
    if len(recovery_cases) > 0:
        recovery_stats["recovery_median"] = recovery_cases.median()
        recovery_stats["recovery_75pct"] = recovery_cases.quantile(0.75)
        recovery_stats["recovery_90pct"] = recovery_cases.quantile(0.90)

    return {
        "total": total,
        "p_case0": p_case0,
        "p_case1": p_case1,
        "p_case2": p_case2,
        "p_case3": p_case3,
        "p_recovery": p_case2 + p_case3,  # Probability of eventual recovery
        **mae_stats,
        **recovery_stats
    }


if __name__ == "__main__":
    # Test the case labeler
    from pathlib import Path

    print("="*60)
    print("CASE LABELER TEST")
    print("="*60)

    # Load OHLCV data
    ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
    if ohlcv_path.exists():
        ohlcv = pd.read_parquet(ohlcv_path)
        print(f"Loaded {len(ohlcv):,} candles")

        # Use train data only
        train_end = "2023-12-31"
        train_ohlcv = ohlcv[ohlcv.index <= train_end]
        print(f"Train data: {len(train_ohlcv):,} candles")

        # Label cases
        case_df = label_cases(
            train_ohlcv,
            targets_bps=[8, 15, 25],
            horizons=[3, 10, 30],
            extended_horizon=500
        )

        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)

        for target_bps in [8, 15, 25]:
            print(f"\n--- Target = {target_bps}bp ---")
            for H in [3, 10, 30]:
                stats = get_case_stats(case_df, target_bps, H)
                if stats:
                    print(f"  H={H}: P(Case1)={stats['p_case1']*100:.1f}%, "
                          f"P(Recovery)={stats['p_recovery']*100:.1f}%")
    else:
        print(f"OHLCV file not found: {ohlcv_path}")
