"""L2-001b: Feature Strength Ranking

Ranks all validated features by STRENGTH (not just pass/fail).
- Magnitude features: ranked by Q4/Q1 MFE ratio (separation power)
- Direction features: ranked by Q4-Q1 directional accuracy spread
- Direction features: deduplicated via correlation matrix (>90% = same signal)

Uses TRAIN data for ranking, OOS for confirmation.
Reuses compute_all_features from L2-001.

Run: PYTHONPATH=src python experiments/layer2_regime/L2_001b_feature_ranking.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from v12.config.loader import load_config
from v12.config.constants import FEES_BPS

# Import feature computation from L2-001
sys.path.insert(0, str(Path(__file__).parent.parent))
from L2_001_feature_revalidation import compute_all_features

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OUT_DIR = Path("experiments/layer2_regime/L2-001/L2-001b")

TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2025-12-31"

TARGET_BPS = 25
HORIZON = 10  # Primary horizon (10 bars = 150 min)

# Feature lists from L2-001 PLAN.md
MAGNITUDE_FEATURES = [
    "atr_pct", "atr_percentile", "ema_separation",
    "range_bps", "body_bps", "dist_from_high20_pct",
    "hour_utc", "volume_ratio",
]

DIRECTION_FEATURES = [
    # RSI
    "rsi", "rsi7", "rsi21",
    # Momentum
    "roc5", "roc10", "roc20", "momentum5", "momentum10",
    # MA distance
    "ema9_dist_pct", "ema20_dist_pct", "ema50_dist_pct",
    "ema100_dist_pct", "ema200_dist_pct", "sma200_dist_pct",
    # MA slope
    "ema20_slope", "ema50_slope", "ema200_slope",
    # Other
    "bb_position", "range_position",
]


# =============================================================================
# Part 1: Magnitude Feature Ranking (by MFE separation)
# =============================================================================

def rank_magnitude_features(df_train, df_oos):
    """Rank magnitude features by Q4/Q1 MFE ratio on TRAIN, confirm on OOS."""
    print("=" * 80)
    print("PART 1: MAGNITUDE FEATURE RANKING")
    print("Metric: Q4/Q1 median MFE ratio (higher = stronger gate)")
    print("=" * 80)

    results = []

    for feat in MAGNITUDE_FEATURES:
        if feat not in df_train.columns:
            print(f"  SKIP {feat} — not in data")
            continue

        for label, df in [("TRAIN", df_train), ("OOS", df_oos)]:
            feat_vals = df[feat].values
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            n = len(df)

            valid = ~np.isnan(feat_vals) & (np.arange(n) < n - HORIZON - 1)
            if valid.sum() < 100:
                continue

            valid_feat = feat_vals[valid]
            try:
                q25, q75 = np.percentile(valid_feat, [25, 75])
            except Exception:
                continue

            if q25 == q75:
                continue

            for qlabel, mask_fn in [("Q1", lambda v: v <= q25), ("Q4", lambda v: v > q75)]:
                indices = np.where(valid & mask_fn(feat_vals))[0]
                if len(indices) > 3000:
                    rng = np.random.RandomState(42)
                    indices = rng.choice(indices, 3000, replace=False)

                long_mfes = []
                short_mfes = []
                best_mfes = []

                for i in indices:
                    entry = opens[i + 1]
                    if entry <= 0 or np.isnan(entry):
                        continue

                    long_mfe = 0.0
                    short_mfe = 0.0
                    for j in range(1, HORIZON + 1):
                        idx = i + 1 + j
                        if idx >= n:
                            break
                        lmfe = (highs[idx] - entry) / entry * 10000
                        smfe = (entry - lows[idx]) / entry * 10000
                        long_mfe = max(long_mfe, lmfe)
                        short_mfe = max(short_mfe, smfe)

                    long_mfes.append(long_mfe)
                    short_mfes.append(short_mfe)
                    best_mfes.append(max(long_mfe, short_mfe))

                results.append({
                    "feature": feat,
                    "dataset": label,
                    "quartile": qlabel,
                    "n_bars": len(indices),
                    "long_median_mfe": np.median(long_mfes) if long_mfes else 0,
                    "short_median_mfe": np.median(short_mfes) if short_mfes else 0,
                    "best_median_mfe": np.median(best_mfes) if best_mfes else 0,
                    "long_mean_mfe": np.mean(long_mfes) if long_mfes else 0,
                    "short_mean_mfe": np.mean(short_mfes) if short_mfes else 0,
                    "pct_above_25bp_long": np.mean(np.array(long_mfes) >= 25) * 100 if long_mfes else 0,
                    "pct_above_25bp_short": np.mean(np.array(short_mfes) >= 25) * 100 if short_mfes else 0,
                })

    rdf = pd.DataFrame(results)

    # Compute Q4/Q1 ratio for TRAIN and OOS
    ranking = []
    for feat in MAGNITUDE_FEATURES:
        fdata = rdf[rdf["feature"] == feat]
        for dataset in ["TRAIN", "OOS"]:
            ddata = fdata[fdata["dataset"] == dataset]
            q1 = ddata[ddata["quartile"] == "Q1"]
            q4 = ddata[ddata["quartile"] == "Q4"]
            if q1.empty or q4.empty:
                continue

            q1_best = q1.iloc[0]["best_median_mfe"]
            q4_best = q4.iloc[0]["best_median_mfe"]
            ratio = q4_best / q1_best if q1_best > 0 else 0

            q1_long = q1.iloc[0]["long_median_mfe"]
            q4_long = q4.iloc[0]["long_median_mfe"]
            q1_short = q1.iloc[0]["short_median_mfe"]
            q4_short = q4.iloc[0]["short_median_mfe"]

            ranking.append({
                "feature": feat,
                "dataset": dataset,
                "q1_best_mfe": q1_best,
                "q4_best_mfe": q4_best,
                "q4_q1_ratio": ratio,
                "q4_q1_diff": q4_best - q1_best,
                "q1_long_mfe": q1_long,
                "q4_long_mfe": q4_long,
                "q1_short_mfe": q1_short,
                "q4_short_mfe": q4_short,
            })

    rank_df = pd.DataFrame(ranking)

    # Print TRAIN ranking
    train_rank = rank_df[rank_df["dataset"] == "TRAIN"].sort_values("q4_q1_ratio", ascending=False)
    oos_rank = rank_df[rank_df["dataset"] == "OOS"].sort_values("q4_q1_ratio", ascending=False)

    print(f"\n{'='*80}")
    print("MAGNITUDE RANKING (TRAIN) — sorted by Q4/Q1 best MFE ratio")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Feature':<22} {'Q1 MFE':>8} {'Q4 MFE':>8} {'Ratio':>7} {'Diff':>8} | "
          f"{'Q1 Long':>8} {'Q4 Long':>8} {'Q1 Short':>9} {'Q4 Short':>9}")
    print("-" * 110)

    for rank, (_, row) in enumerate(train_rank.iterrows(), 1):
        print(f"{rank:<5} {row['feature']:<22} {row['q1_best_mfe']:>7.1f} {row['q4_best_mfe']:>7.1f} "
              f"{row['q4_q1_ratio']:>6.2f}x {row['q4_q1_diff']:>+7.1f} | "
              f"{row['q1_long_mfe']:>7.1f} {row['q4_long_mfe']:>7.1f} "
              f"{row['q1_short_mfe']:>8.1f} {row['q4_short_mfe']:>8.1f}")

    print(f"\n{'='*80}")
    print("MAGNITUDE RANKING (OOS) — confirmation")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Feature':<22} {'Q1 MFE':>8} {'Q4 MFE':>8} {'Ratio':>7} {'Diff':>8}")
    print("-" * 60)

    for rank, (_, row) in enumerate(oos_rank.iterrows(), 1):
        print(f"{rank:<5} {row['feature']:<22} {row['q1_best_mfe']:>7.1f} {row['q4_best_mfe']:>7.1f} "
              f"{row['q4_q1_ratio']:>6.2f}x {row['q4_q1_diff']:>+7.1f}")

    # Save
    rank_df.to_csv(OUT_DIR / "L2_001b_magnitude_ranking.csv", index=False)
    return train_rank, oos_rank


# =============================================================================
# Part 2: Direction Feature Ranking (by directional accuracy spread)
# =============================================================================

def rank_direction_features(df_train, df_oos):
    """Rank direction features by Q4-Q1 directional accuracy spread."""
    print(f"\n\n{'='*80}")
    print("PART 2: DIRECTION FEATURE RANKING")
    print(f"Metric: |Q4 - Q1| directional accuracy spread (target={TARGET_BPS}bp, H={HORIZON})")
    print("Higher spread = stronger directional signal")
    print(f"{'='*80}")

    results = []

    for feat in DIRECTION_FEATURES:
        if feat not in df_train.columns:
            continue

        for label, df in [("TRAIN", df_train), ("OOS", df_oos)]:
            opens = df["open"].values
            highs = df["high"].values
            lows = df["low"].values
            feat_vals = df[feat].values
            n = len(df)

            valid = ~np.isnan(feat_vals) & (np.arange(n) < n - HORIZON - 1)
            if valid.sum() < 100:
                continue

            valid_feat = feat_vals[valid]
            try:
                q25, q50, q75 = np.percentile(valid_feat, [25, 50, 75])
            except Exception:
                continue

            if q25 == q75:
                continue

            for qlabel, mask_fn in [("Q1", lambda v: v <= q25), ("Q4", lambda v: v > q75)]:
                mask = valid & mask_fn(feat_vals)
                indices = np.where(mask)[0]

                n_up = 0
                n_down = 0
                n_neither = 0

                for i in indices:
                    entry = opens[i + 1]
                    if entry <= 0 or np.isnan(entry):
                        n_neither += 1
                        continue

                    up_target = entry * (1 + TARGET_BPS / 10000)
                    down_target = entry * (1 - TARGET_BPS / 10000)
                    hit_up = False
                    hit_down = False

                    for j in range(1, HORIZON + 1):
                        idx = i + 1 + j
                        if idx >= n:
                            break
                        if not hit_up and highs[idx] >= up_target:
                            hit_up = True
                            if not hit_down:
                                n_up += 1
                                break
                        if not hit_down and lows[idx] <= down_target:
                            hit_down = True
                            if not hit_up:
                                n_down += 1
                                break

                    if not hit_up and not hit_down:
                        n_neither += 1

                total_directional = n_up + n_down
                up_pct = n_up / total_directional * 100 if total_directional > 0 else 50

                results.append({
                    "feature": feat,
                    "dataset": label,
                    "quartile": qlabel,
                    "n_bars": len(indices),
                    "n_up_first": n_up,
                    "n_down_first": n_down,
                    "n_neither": n_neither,
                    "up_accuracy": up_pct,
                })

    rdf = pd.DataFrame(results)

    # Compute Q4-Q1 spread
    ranking = []
    for feat in DIRECTION_FEATURES:
        fdata = rdf[rdf["feature"] == feat]
        for dataset in ["TRAIN", "OOS"]:
            ddata = fdata[fdata["dataset"] == dataset]
            q1 = ddata[ddata["quartile"] == "Q1"]
            q4 = ddata[ddata["quartile"] == "Q4"]
            if q1.empty or q4.empty:
                continue

            q1_up = q1.iloc[0]["up_accuracy"]
            q4_up = q4.iloc[0]["up_accuracy"]
            spread = q4_up - q1_up  # Positive = Q4 is more bullish

            ranking.append({
                "feature": feat,
                "dataset": dataset,
                "q1_up_pct": q1_up,
                "q4_up_pct": q4_up,
                "spread": spread,
                "abs_spread": abs(spread),
                "direction": "BULLISH" if spread > 0 else "BEARISH",
                "q1_n": q1.iloc[0]["n_bars"],
                "q4_n": q4.iloc[0]["n_bars"],
            })

    rank_df = pd.DataFrame(ranking)

    # Print TRAIN ranking
    train_rank = rank_df[rank_df["dataset"] == "TRAIN"].sort_values("abs_spread", ascending=False)
    oos_rank = rank_df[rank_df["dataset"] == "OOS"].sort_values("abs_spread", ascending=False)

    print(f"\n{'='*80}")
    print("DIRECTION RANKING (TRAIN) — sorted by |Q4-Q1| accuracy spread")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Feature':<22} {'Q1 UP%':>8} {'Q4 UP%':>8} {'Spread':>8} {'Direction':>10} | "
          f"{'Q1 n':>6} {'Q4 n':>6}")
    print("-" * 80)

    for rank, (_, row) in enumerate(train_rank.iterrows(), 1):
        print(f"{rank:<5} {row['feature']:<22} {row['q1_up_pct']:>7.1f}% {row['q4_up_pct']:>7.1f}% "
              f"{row['spread']:>+7.1f}pp {row['direction']:>10} | "
              f"{row['q1_n']:>6.0f} {row['q4_n']:>6.0f}")

    # OOS confirmation
    print(f"\n{'='*80}")
    print("DIRECTION RANKING (OOS) — confirmation")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Feature':<22} {'Q1 UP%':>8} {'Q4 UP%':>8} {'Spread':>8} {'Direction':>10}")
    print("-" * 70)

    for rank, (_, row) in enumerate(oos_rank.iterrows(), 1):
        print(f"{rank:<5} {row['feature']:<22} {row['q1_up_pct']:>7.1f}% {row['q4_up_pct']:>7.1f}% "
              f"{row['spread']:>+7.1f}pp {row['direction']:>10}")

    # Check train vs OOS consistency
    print(f"\n{'='*80}")
    print("CONSISTENCY CHECK: Same direction on TRAIN and OOS?")
    print(f"{'='*80}")

    for feat in DIRECTION_FEATURES:
        t = rank_df[(rank_df["feature"] == feat) & (rank_df["dataset"] == "TRAIN")]
        o = rank_df[(rank_df["feature"] == feat) & (rank_df["dataset"] == "OOS")]
        if t.empty or o.empty:
            continue
        t_dir = t.iloc[0]["direction"]
        o_dir = o.iloc[0]["direction"]
        t_spread = t.iloc[0]["spread"]
        o_spread = o.iloc[0]["spread"]
        consistent = "YES" if t_dir == o_dir else "NO"
        print(f"  {feat:<22} TRAIN: {t_spread:>+6.1f}pp ({t_dir})  OOS: {o_spread:>+6.1f}pp ({o_dir})  Consistent: {consistent}")

    rank_df.to_csv(OUT_DIR / "L2_001b_direction_ranking.csv", index=False)
    return train_rank, oos_rank


# =============================================================================
# Part 3: Direction Feature Deduplication (correlation matrix)
# =============================================================================

def deduplicate_direction_features(df_train):
    """Compute pairwise correlation between direction features to find duplicates."""
    print(f"\n\n{'='*80}")
    print("PART 3: DIRECTION FEATURE DEDUPLICATION")
    print("Pairwise Pearson correlation — >90% = effectively same signal")
    print(f"{'='*80}")

    # Get feature values (drop NaN rows)
    available = [f for f in DIRECTION_FEATURES if f in df_train.columns]
    feat_df = df_train[available].dropna()

    print(f"\nFeatures: {len(available)}")
    print(f"Bars with all features: {len(feat_df)}")

    # Correlation matrix
    corr = feat_df.corr()

    # Find highly correlated pairs (>90%)
    print(f"\n--- Highly Correlated Pairs (|r| > 0.90) ---")
    print(f"{'Feature A':<22} {'Feature B':<22} {'Correlation':>12}")
    print("-" * 60)

    pairs = []
    for i, f1 in enumerate(available):
        for j, f2 in enumerate(available):
            if j <= i:
                continue
            r = corr.loc[f1, f2]
            if abs(r) > 0.90:
                pairs.append((f1, f2, r))
                print(f"{f1:<22} {f2:<22} {r:>+11.3f}")

    # Group correlated features
    print(f"\n--- Correlation Groups (|r| > 0.90) ---")

    # Build adjacency for clustering
    from collections import defaultdict
    adj = defaultdict(set)
    for f1, f2, r in pairs:
        adj[f1].add(f2)
        adj[f2].add(f1)

    visited = set()
    groups = []

    for feat in available:
        if feat in visited:
            continue
        if feat not in adj:
            groups.append([feat])
            visited.add(feat)
            continue

        # BFS
        group = []
        queue = [feat]
        while queue:
            f = queue.pop(0)
            if f in visited:
                continue
            visited.add(f)
            group.append(f)
            for neighbor in adj[f]:
                if neighbor not in visited:
                    queue.append(neighbor)
        groups.append(group)

    print(f"\n{len(groups)} distinct groups from {len(available)} features:\n")
    for i, group in enumerate(groups, 1):
        if len(group) == 1:
            print(f"  Group {i}: {group[0]} (unique)")
        else:
            print(f"  Group {i}: {', '.join(group)}")
            # Show internal correlations
            for a_idx, a in enumerate(group):
                for b in group[a_idx+1:]:
                    r = corr.loc[a, b]
                    print(f"    {a} <-> {b}: r={r:+.3f}")

    # Full correlation matrix (abbreviated)
    print(f"\n--- Full Correlation Matrix (|r| rounded) ---")
    # Print header
    short_names = {f: f[:8] for f in available}
    header = f"{'':>14} " + " ".join(f"{short_names[f]:>8}" for f in available)
    print(header)
    for f1 in available:
        row = f"{short_names[f1]:>14} "
        for f2 in available:
            r = corr.loc[f1, f2]
            if f1 == f2:
                row += f"{'1.00':>8} "
            else:
                row += f"{r:>+7.2f} "
        print(row)

    # Save correlation matrix
    corr.to_csv(OUT_DIR / "L2_001b_direction_correlation.csv")

    return groups, corr


# =============================================================================
# Part 4: Combined Summary
# =============================================================================

def print_combined_summary(mag_train, mag_oos, dir_train, dir_oos, groups):
    """Print final combined ranking for L2-002."""
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY: Feature Rankings for L2-002")
    print(f"{'='*80}")

    print(f"\n--- MAGNITUDE FEATURES (use as GATE) ---")
    print(f"Ranked by Q4/Q1 MFE separation on TRAIN")
    print(f"{'Rank':<5} {'Feature':<22} {'TRAIN Ratio':>12} {'OOS Ratio':>10} {'OOS Confirmed':>14}")
    print("-" * 65)

    for rank, (_, row) in enumerate(mag_train.iterrows(), 1):
        feat = row["feature"]
        oos_row = mag_oos[mag_oos["feature"] == feat]
        oos_ratio = oos_row.iloc[0]["q4_q1_ratio"] if not oos_row.empty else 0
        confirmed = "YES" if oos_ratio > 1.2 else "WEAK" if oos_ratio > 1.0 else "NO"
        print(f"{rank:<5} {feat:<22} {row['q4_q1_ratio']:>10.2f}x {oos_ratio:>9.2f}x {confirmed:>14}")

    print(f"\n--- DIRECTION FEATURES (use as SIGNAL) ---")
    print(f"Ranked by |spread| on TRAIN. After deduplication: {len(groups)} distinct groups")
    print(f"{'Rank':<5} {'Feature':<22} {'TRAIN Spread':>13} {'OOS Spread':>11} {'Group':>6} {'Representative':>15}")
    print("-" * 80)

    # Pick representative from each group (highest spread)
    group_map = {}
    representatives = {}
    for i, group in enumerate(groups, 1):
        for feat in group:
            group_map[feat] = i
        # Best in group = highest abs spread on train
        best_feat = None
        best_spread = 0
        for feat in group:
            row = dir_train[dir_train["feature"] == feat]
            if not row.empty:
                s = row.iloc[0]["abs_spread"]
                if s > best_spread:
                    best_spread = s
                    best_feat = feat
        representatives[i] = best_feat

    for rank, (_, row) in enumerate(dir_train.iterrows(), 1):
        feat = row["feature"]
        oos_row = dir_oos[dir_oos["feature"] == feat]
        oos_spread = oos_row.iloc[0]["spread"] if not oos_row.empty else 0
        group_id = group_map.get(feat, "?")
        is_rep = "<<<" if representatives.get(group_id) == feat else ""
        print(f"{rank:<5} {feat:<22} {row['spread']:>+12.1f}pp {oos_spread:>+10.1f}pp {group_id:>6} {is_rep:>15}")

    # Print deduplicated list
    print(f"\n--- DEDUPLICATED DIRECTION FEATURES ({len(groups)} distinct) ---")
    print(f"One representative per correlation group:\n")

    dedup_feats = []
    for i, group in enumerate(groups, 1):
        rep = representatives[i]
        if rep is None:
            continue
        row = dir_train[dir_train["feature"] == rep]
        if row.empty:
            continue
        spread = row.iloc[0]["spread"]
        direction = row.iloc[0]["direction"]
        others = [f for f in group if f != rep]
        dedup_feats.append({
            "rank": len(dedup_feats) + 1,
            "feature": rep,
            "spread": spread,
            "direction": direction,
            "group_members": others,
        })

    dedup_feats.sort(key=lambda x: abs(x["spread"]), reverse=True)
    for i, d in enumerate(dedup_feats, 1):
        others_str = f" (also: {', '.join(d['group_members'])})" if d["group_members"] else " (unique)"
        print(f"  {i}. {d['feature']:<22} {d['spread']:>+6.1f}pp {d['direction']}{others_str}")

    return dedup_feats


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("L2-001b: Feature Strength Ranking")
    print("Magnitude: ranked by MFE separation | Direction: ranked by accuracy spread")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(DATA_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"Total bars: {len(df)}")

    # Compute features
    print("Computing features...")
    df = compute_all_features(df)

    train = df[TRAIN_START:TRAIN_END].copy()
    oos = df[OOS_START:OOS_END].copy()
    print(f"TRAIN: {len(train)} bars | OOS: {len(oos)} bars\n")

    # Part 1: Magnitude ranking
    mag_train, mag_oos = rank_magnitude_features(train, oos)

    # Part 2: Direction ranking
    dir_train, dir_oos = rank_direction_features(train, oos)

    # Part 3: Direction deduplication
    groups, corr = deduplicate_direction_features(train)

    # Part 4: Combined summary
    dedup_feats = print_combined_summary(mag_train, mag_oos, dir_train, dir_oos, groups)

    # Save final summary
    summary = {
        "magnitude": [],
        "direction_deduplicated": [],
    }
    for _, row in mag_train.iterrows():
        oos_row = mag_oos[mag_oos["feature"] == row["feature"]]
        summary["magnitude"].append({
            "feature": row["feature"],
            "train_ratio": round(row["q4_q1_ratio"], 2),
            "oos_ratio": round(oos_row.iloc[0]["q4_q1_ratio"], 2) if not oos_row.empty else 0,
            "train_q4_mfe": round(row["q4_best_mfe"], 1),
            "train_q1_mfe": round(row["q1_best_mfe"], 1),
        })
    for d in dedup_feats:
        summary["direction_deduplicated"].append({
            "feature": d["feature"],
            "spread_pp": round(d["spread"], 1),
            "direction": d["direction"],
            "group_members": d["group_members"],
        })

    import json
    with open(OUT_DIR / "L2_001b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nFiles saved:")
    print(f"  {OUT_DIR}/L2_001b_magnitude_ranking.csv")
    print(f"  {OUT_DIR}/L2_001b_direction_ranking.csv")
    print(f"  {OUT_DIR}/L2_001b_direction_correlation.csv")
    print(f"  {OUT_DIR}/L2_001b_summary.json")
    print(f"\nDone.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
