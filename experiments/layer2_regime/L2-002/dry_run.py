"""Dry run: verify framework compiles, count configs, time 1 precompute, estimate total runtime."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "L2-001"))

import time
import yaml
import numpy as np
import pandas as pd

from framework import (
    FRAMEWORK_CONFIG_PATH, DATA_PATH,
    generate_gates, generate_signals, generate_exits,
    RegimeClassifier, precompute_all_trades, run_backtest,
    apply_gate, detect_signals, ExitConfig, GateConfig, SignalConfig,
)
from L2_001_feature_revalidation import compute_all_features

print("=" * 60)
print("DRY RUN: L2-002 Framework Validation")
print("=" * 60)

# 1. Load config
print("\n[1] Loading config...")
with open(FRAMEWORK_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
print("  OK")

# 2. Load data + compute features
print("\n[2] Loading data + computing features...")
df = pd.read_parquet(cfg["data"]["path"])
df.index = pd.to_datetime(df.index).tz_localize(None)
df = compute_all_features(df, cfg=cfg)
train = df[cfg["data"]["train_start"]:cfg["data"]["train_end"]].copy()
print(f"  TRAIN: {len(train)} bars")

# 3. Build regime classifier
print("\n[3] Building regime classifier...")
classifier = RegimeClassifier(train, cfg=cfg["regime"])
print("  OK")

# 4. Generate all configs — COUNT them
print("\n[4] Generating configurations...")
exits = generate_exits(cfg["exit_strategy"])
gates = generate_gates(cfg["magnitude_gate"])
signals = generate_signals(cfg["signal_gate"])

total = len(exits) * len(gates) * len(signals)
print(f"\n  TOTAL: {len(exits)} exits x {len(gates)} gates x {len(signals)} signals = {total:,} combos")

# 5. Time ONE precompute (TRAIN)
print("\n[5] Timing 1 precompute (TRAIN)...")
ec = exits[0]
t0 = time.time()
precomputed = precompute_all_trades(train, ec)
t_precompute = time.time() - t0
print(f"  Precompute time: {t_precompute:.1f}s for exit: {ec.name}")

# 6. Time ONE backtest
print("\n[6] Timing 1 backtest...")
g = gates[0]
s = signals[0]
t0 = time.time()
br = run_backtest(train, g, s, precomputed, classifier, exit_name=ec.name)
t_backtest = time.time() - t0
print(f"  Backtest time: {t_backtest*1000:.2f}ms")
print(f"  Result: {br.config_name} | trades={br.n_trades} | bps={br.total_bps:+.0f} | PF={br.profit_factor:.2f}")

# 7. Time a BATCH of gate x signal (all combos for 1 exit)
print(f"\n[7] Timing full gate x signal batch ({len(gates)} x {len(signals)} = {len(gates)*len(signals):,})...")
t0 = time.time()
batch_results = []
for g in gates:
    for s in signals:
        br = run_backtest(train, g, s, precomputed, classifier, exit_name=ec.name)
        batch_results.append(br)
t_batch = time.time() - t0
print(f"  Batch time: {t_batch:.1f}s for {len(batch_results):,} combos")

# 8. Estimate total runtime
print("\n[8] RUNTIME ESTIMATE:")
t_per_exit = t_precompute + t_batch
t_total_train = t_per_exit * len(exits)
# OOS precompute is ~half the size
t_total_oos_est = (t_precompute * 0.5 + t_batch) * len(exits) * 0.3  # assume 30% pass filter
t_total = t_total_train + t_total_oos_est

print(f"  Per exit iteration: {t_per_exit:.1f}s (precompute={t_precompute:.1f}s + batch={t_batch:.1f}s)")
print(f"  TRAIN total ({len(exits)} exits): {t_total_train:.0f}s = {t_total_train/60:.1f}min = {t_total_train/3600:.1f}hr")
print(f"  OOS estimate (30% filter): {t_total_oos_est:.0f}s = {t_total_oos_est/60:.1f}min")
print(f"  TOTAL estimate: {t_total:.0f}s = {t_total/60:.1f}min = {t_total/3600:.1f}hr")

# 9. Memory estimate
import sys as _sys
result_size = _sys.getsizeof(batch_results) + sum(_sys.getsizeof(br) for br in batch_results[:10]) * len(batch_results) / 10
total_results_est = result_size * len(exits)
print(f"\n[9] MEMORY ESTIMATE:")
print(f"  Results per exit batch: ~{result_size/1024/1024:.1f} MB")
print(f"  Total results ({total:,} combos): ~{total_results_est/1024/1024:.0f} MB")

# 10. Sanity check: first result makes sense?
print(f"\n[10] SANITY CHECK:")
interesting = [br for br in batch_results if br.profit_factor > 1.5 and br.n_trades >= 50]
print(f"  Interesting configs (PF>1.5, trades>=50): {len(interesting):,} / {len(batch_results):,}")
if interesting:
    best = max(interesting, key=lambda x: x.total_bps)
    print(f"  Best: {best.config_name} | trades={best.n_trades} | bps={best.total_bps:+.0f} | PF={best.profit_factor:.2f}")

print("\n" + "=" * 60)
print("DRY RUN COMPLETE — no errors")
print("=" * 60)
