"""Minimal dry run — step by step with immediate output."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "L2-001"))

def p(msg):
    print(msg, flush=True)

p("STEP 1: imports...")
import yaml, numpy as np, pandas as pd
from framework import (
    FRAMEWORK_CONFIG_PATH, generate_gates, generate_signals, generate_exits,
    RegimeClassifier, precompute_all_trades, run_backtest_fast, apply_gate,
)
from L2_001_feature_revalidation import compute_all_features
p("  OK")

p("STEP 2: config...")
with open(FRAMEWORK_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
p("  OK")

p("STEP 3: data...")
df = pd.read_parquet(cfg["data"]["path"])
df.index = pd.to_datetime(df.index).tz_localize(None)
p(f"  Loaded {len(df)} bars")

p("STEP 4: features...")
t0 = time.time()
df = compute_all_features(df, cfg=cfg)
p(f"  Features computed in {time.time()-t0:.1f}s")

train = df[cfg["data"]["train_start"]:cfg["data"]["train_end"]].copy()
p(f"  TRAIN: {len(train)} bars")

p("STEP 5: classifier...")
classifier = RegimeClassifier(train, cfg=cfg["regime"])
p("  OK")

p("STEP 6: config counts...")
exits = generate_exits(cfg["exit_strategy"])
gates = generate_gates(cfg["magnitude_gate"])
signals = generate_signals(cfg["signal_gate"])
total = len(exits) * len(gates) * len(signals)
p(f"  {len(exits)} exits x {len(gates)} gates x {len(signals)} signals = {total:,}")

# Pre-compute masks ONCE
p("\nSTEP 7: pre-compute gate masks...")
t0 = time.time()
gate_masks = {}
for gate in gates:
    mask = apply_gate(train, gate, classifier)
    gate_masks[gate.name] = mask.values
p(f"  {len(gate_masks)} gate masks in {time.time()-t0:.1f}s")

p("STEP 8: pre-compute signal masks...")
t0 = time.time()
signal_long_masks = {}
signal_short_masks = {}
for signal in signals:
    long_m = pd.Series(True, index=train.index)
    short_m = pd.Series(True, index=train.index)
    for feature, long_t, short_t in signal.conditions:
        feat_vals = train[feature]
        long_m = long_m & (feat_vals <= long_t)
        short_m = short_m & (feat_vals >= short_t)
    signal_long_masks[signal.name] = long_m.values
    signal_short_masks[signal.name] = short_m.values
p(f"  {len(signal_long_masks)} signal masks in {time.time()-t0:.1f}s")

# Precompute 1 exit
p("\nSTEP 9: precompute 1 exit (TRAIN)...")
ec = exits[0]
p(f"  Exit: {ec.name}")
t0 = time.time()
precomputed = precompute_all_trades(train, ec)
t_precompute = time.time() - t0
p(f"  Precompute: {t_precompute:.1f}s")

# 1 backtest
p("STEP 10: 1 fast backtest...")
br = run_backtest_fast(
    gate_masks[gates[0].name], signal_long_masks[signals[0].name],
    signal_short_masks[signals[0].name], precomputed,
    gates[0].name, signals[0].name, exit_name=ec.name,
)
p(f"  {br.config_name} | trades={br.n_trades} | bps={br.total_bps:+.0f} | PF={br.profit_factor:.2f}")

# Full batch for 1 exit
p(f"\nSTEP 11: batch timing (all gate x signal for 1 exit = {len(gates)*len(signals):,})...")
t0 = time.time()
count = 0
for g in gates:
    g_mask = gate_masks[g.name]
    for s in signals:
        br = run_backtest_fast(
            g_mask, signal_long_masks[s.name], signal_short_masks[s.name],
            precomputed, g.name, s.name, exit_name=ec.name,
        )
        count += 1
t_batch = time.time() - t0
p(f"  {count:,} combos in {t_batch:.1f}s ({count/t_batch:.0f} combos/sec)")

p(f"\nESTIMATE:")
per_exit = t_precompute + t_batch
total_s = per_exit * len(exits)
p(f"  Per exit: {per_exit:.1f}s (precompute={t_precompute:.1f}s + batch={t_batch:.1f}s)")
p(f"  TRAIN total: {total_s:.0f}s = {total_s/60:.1f}min = {total_s/3600:.1f}hr")

# Memory
mem_per_result = 200  # bytes rough estimate
total_mem = mem_per_result * total / 1024 / 1024
p(f"  Memory (results): ~{total_mem:.0f} MB")

p(f"\nDONE")
