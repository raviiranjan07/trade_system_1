import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exceptions import DataValidationError


def validate_ohlcv(
    df: pd.DataFrame,
    max_gap_tolerance: int = 0,
    fill_gaps: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Validate OHLCV data integrity.

    Args:
        df: DataFrame with OHLCV data indexed by time
        max_gap_tolerance: Maximum allowed missing candles (0 = strict)
        fill_gaps: If True, fill small gaps with forward-fill
        verbose: Print validation details

    Returns:
        Validated (and optionally gap-filled) DataFrame

    Raises:
        DataValidationError: If validation fails
    """
    if df.empty:
        raise DataValidationError("DataFrame is empty", data_source="OHLCV")

    # Check sorted timestamps
    if not df.index.is_monotonic_increasing:
        raise DataValidationError(
            "Timestamps are not sorted in ascending order",
            data_source="OHLCV"
        )

    # Check for NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        nan_cols = nan_counts[nan_counts > 0]
        raise DataValidationError(
            f"NaN values found in columns: {nan_cols.to_dict()}",
            data_source="OHLCV"
        )

    # Check for missing candles (gaps)
    missing_candles, gap_details = _detect_gaps(df)

    if len(missing_candles) > 0:
        if verbose:
            _print_gap_report(missing_candles, gap_details, df)

        if len(missing_candles) > max_gap_tolerance:
            if fill_gaps and len(missing_candles) <= 100:
                # Fill small gaps
                df = _fill_gaps(df, missing_candles)
                if verbose:
                    print(f"  [INFO] Filled {len(missing_candles)} missing candles")
            else:
                raise DataValidationError(
                    f"Found {len(missing_candles)} missing candles (tolerance: {max_gap_tolerance})\n"
                    f"Use fill_gaps=True for small gaps or check your data source.",
                    data_source="OHLCV"
                )

    # Validate OHLC relationships
    _validate_ohlc_relationships(df)

    if verbose and len(missing_candles) == 0:
        print(f"  [OK] Data integrity validated: {len(df):,} candles, no gaps")

    return df


def _detect_gaps(df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, List[dict]]:
    """
    Detect missing candles in the time series.

    Returns:
        Tuple of (missing timestamps, list of gap details)
    """
    expected = pd.date_range(
        start=df.index[0],
        end=df.index[-1],
        freq="1min"
    )

    missing = expected.difference(df.index)

    # Group consecutive missing candles into gaps
    gap_details = []
    if len(missing) > 0:
        missing_sorted = missing.sort_values()
        gap_start = missing_sorted[0]
        gap_count = 1

        for i in range(1, len(missing_sorted)):
            diff = (missing_sorted[i] - missing_sorted[i-1]).total_seconds()
            if diff == 60:  # Consecutive minute
                gap_count += 1
            else:
                # End of gap
                gap_details.append({
                    "start": gap_start,
                    "end": missing_sorted[i-1],
                    "count": gap_count,
                })
                gap_start = missing_sorted[i]
                gap_count = 1

        # Last gap
        gap_details.append({
            "start": gap_start,
            "end": missing_sorted[-1],
            "count": gap_count,
        })

    return missing, gap_details


def _print_gap_report(
    missing: pd.DatetimeIndex,
    gap_details: List[dict],
    df: pd.DataFrame
) -> None:
    """Print a report of detected gaps."""
    total_expected = len(pd.date_range(df.index[0], df.index[-1], freq="1min"))
    coverage = (len(df) / total_expected) * 100

    print()
    print("  " + "=" * 56)
    print("  DATA GAP REPORT")
    print("  " + "=" * 56)
    print(f"  Total missing candles: {len(missing):,}")
    print(f"  Data coverage: {coverage:.2f}%")
    print(f"  Number of gaps: {len(gap_details)}")
    print()

    if len(gap_details) <= 10:
        print("  Gap Details:")
        print("  " + "-" * 56)
        for i, gap in enumerate(gap_details, 1):
            duration = gap["count"]
            print(f"    {i}. {gap['start']} to {gap['end']} ({duration} min)")
        print("  " + "-" * 56)
    else:
        # Show summary for many gaps
        print("  Largest Gaps (top 5):")
        print("  " + "-" * 56)
        sorted_gaps = sorted(gap_details, key=lambda x: x["count"], reverse=True)[:5]
        for i, gap in enumerate(sorted_gaps, 1):
            duration = gap["count"]
            print(f"    {i}. {gap['start']} ({duration} min)")
        print("  " + "-" * 56)
    print()


def _fill_gaps(df: pd.DataFrame, missing: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fill missing candles using forward-fill.

    For missing candles:
    - OHLC: Use previous close
    - Volume: Set to 0
    - num_trades: Set to 0
    """
    # Create complete index
    full_index = pd.date_range(
        start=df.index[0],
        end=df.index[-1],
        freq="1min"
    )

    # Reindex and forward-fill
    df_filled = df.reindex(full_index)

    # For price columns, forward-fill
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].ffill()

    # For volume/trades, fill with 0
    if "volume" in df_filled.columns:
        df_filled["volume"] = df_filled["volume"].fillna(0)
    if "num_trades" in df_filled.columns:
        df_filled["num_trades"] = df_filled["num_trades"].fillna(0)

    return df_filled


def _validate_ohlc_relationships(df: pd.DataFrame) -> None:
    """
    Validate OHLC price relationships.

    - High should be >= Open, Close, Low
    - Low should be <= Open, Close, High
    """
    if not all(col in df.columns for col in ["open", "high", "low", "close"]):
        return

    # Check high >= all others
    invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
    if invalid_high.any():
        count = invalid_high.sum()
        first_bad = df[invalid_high].index[0]
        raise DataValidationError(
            f"Invalid OHLC: High < other prices in {count} rows. First at: {first_bad}",
            data_source="OHLCV"
        )

    # Check low <= all others
    invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
    if invalid_low.any():
        count = invalid_low.sum()
        first_bad = df[invalid_low].index[0]
        raise DataValidationError(
            f"Invalid OHLC: Low > other prices in {count} rows. First at: {first_bad}",
            data_source="OHLCV"
        )


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Run comprehensive data quality checks.

    Returns:
        Dictionary with quality metrics
    """
    total_expected = len(pd.date_range(df.index[0], df.index[-1], freq="1min"))
    missing_count = total_expected - len(df)

    # Detect zero-volume candles (possible bad data)
    zero_volume = (df["volume"] == 0).sum() if "volume" in df.columns else 0

    # Detect flat candles (O=H=L=C, possibly fake)
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        flat_candles = ((df["open"] == df["high"]) &
                        (df["high"] == df["low"]) &
                        (df["low"] == df["close"])).sum()
    else:
        flat_candles = 0

    # Price anomalies (extreme moves > 10% in 1 min)
    if "close" in df.columns:
        returns = df["close"].pct_change().abs()
        extreme_moves = (returns > 0.10).sum()
    else:
        extreme_moves = 0

    return {
        "total_candles": len(df),
        "expected_candles": total_expected,
        "missing_candles": missing_count,
        "coverage_pct": (len(df) / total_expected) * 100,
        "zero_volume_candles": zero_volume,
        "flat_candles": flat_candles,
        "extreme_moves": extreme_moves,
        "start_time": df.index[0],
        "end_time": df.index[-1],
    }
