"""
Test Case Labeler: Validate case distribution matches analysis findings.

Run: .venv/Scripts/python.exe debug_case_labeler.py

Expected distributions (from debug_recovery_analysis.py):
- Target 15bp: Case 1 ~10%, Case 2+3 ~90%
- Target 25bp: Case 1 ~16%, Case 2+3 ~84%

NOTE: Target 8bp removed - structurally impossible (8bp - 8bp fees = 0 net profit)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade_system.outcomes.case_labeler import label_cases, get_case_stats

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_END = "2023-12-31"
# NOTE: Removed 8bp - structurally impossible (8bp target - 8bp fees = 0bp net profit)
TARGETS = [15, 25]
HORIZONS = [3, 5, 15, 30]

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("CASE LABELER VALIDATION")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    sys.exit(1)

ohlcv = pd.read_parquet(ohlcv_path)
print(f"Loaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"Train data: {len(train_ohlcv):,} candles")

# =============================================================================
# RUN CASE LABELER
# =============================================================================
print("\n" + "=" * 70)
print("RUNNING CASE LABELER")
print("=" * 70)

case_df = label_cases(
    train_ohlcv,
    targets_bps=TARGETS,
    horizons=HORIZONS,
    extended_horizon=500,
    output_dir="data/outcomes",
    pair="BTCUSDT",
    timeframe="1m"
)

# =============================================================================
# VALIDATE DISTRIBUTIONS
# =============================================================================
print("\n" + "=" * 70)
print("VALIDATION: Comparing with Analysis Findings")
print("=" * 70)

# Expected distributions (from debug_recovery_analysis.py)
# NOTE: Removed 8bp - structurally impossible
expected = {
    (15, 3): {"p_case1": 0.096, "p_recovery": 0.904},
    (15, 5): {"p_case1": 0.097, "p_recovery": 0.903},
    (15, 15): {"p_case1": 0.098, "p_recovery": 0.902},
    (15, 30): {"p_case1": 0.098, "p_recovery": 0.902},
    (25, 3): {"p_case1": 0.162, "p_recovery": 0.838},
    (25, 5): {"p_case1": 0.163, "p_recovery": 0.837},
    (25, 15): {"p_case1": 0.163, "p_recovery": 0.837},
    (25, 30): {"p_case1": 0.164, "p_recovery": 0.836},
}

print(f"\n{'Target':<10} {'H':<6} {'P(Case1)':<12} {'Expected':<12} {'Match':<8} {'P(Rec)':<12} {'Expected':<12} {'Match':<8}")
print("-" * 80)

all_match = True
for target_bps in TARGETS:
    for H in HORIZONS:
        stats = get_case_stats(case_df, target_bps, H)

        if not stats:
            print(f"{target_bps}bp{'':<6} H={H:<4} {'N/A':<12}")
            continue

        exp = expected.get((target_bps, H), {"p_case1": 0.1, "p_recovery": 0.9})

        # Check if within 2 percentage points
        p_case1_match = abs(stats["p_case1"] - exp["p_case1"]) < 0.02
        p_rec_match = abs(stats["p_recovery"] - exp["p_recovery"]) < 0.02

        match1 = "OK" if p_case1_match else "DIFF"
        match2 = "OK" if p_rec_match else "DIFF"

        if not (p_case1_match and p_rec_match):
            all_match = False

        print(f"{target_bps}bp{'':<6} H={H:<4} {stats['p_case1']*100:>10.1f}% {exp['p_case1']*100:>10.1f}% {match1:<8} {stats['p_recovery']*100:>10.1f}% {exp['p_recovery']*100:>10.1f}% {match2:<8}")

print("-" * 80)

if all_match:
    print("\nVALIDATION PASSED: Case distributions match analysis findings!")
else:
    print("\nVALIDATION WARNING: Some distributions differ slightly (may be due to sampling)")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Case Distributions")
print("=" * 70)

for target_bps in TARGETS:
    print(f"\n--- Target = {target_bps}bp ---")
    print(f"{'H':<6} {'Case0':<10} {'Case1':<10} {'Case2':<10} {'Case3':<10} {'Recovery':<10}")
    print("-" * 60)

    for H in HORIZONS:
        stats = get_case_stats(case_df, target_bps, H)
        if stats:
            print(f"H={H:<4} {stats['p_case0']*100:>8.1f}% {stats['p_case1']*100:>8.1f}% {stats['p_case2']*100:>8.1f}% {stats['p_case3']*100:>8.1f}% {stats['p_recovery']*100:>8.1f}%")

print("\n" + "=" * 70)
print("CASE LABELER VALIDATION COMPLETE")
print("=" * 70)
print(f"\nOutput saved to: data/outcomes/BTCUSDT_1m_cases.parquet")
print(f"Columns: {list(case_df.columns)[:6]}...")
